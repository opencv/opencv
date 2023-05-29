// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF
#include "../graph_simplifier.hpp"
#include "onnx_graph_simplifier.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <queue>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

extern bool DNN_DIAGNOSTICS_RUN;

// This wrapper can behave differently for fake input nodes and real graph nodes.
class ONNXNodeWrapper : public ImportNodeWrapper
{
public:
    ONNXNodeWrapper(opencv_onnx::NodeProto* _node = 0) : node(_node) {}

    virtual int getNumInputs() const CV_OVERRIDE
    {
        return node ? node->input_size() : 0;
    }

    virtual std::string getInputName(int idx) const CV_OVERRIDE
    {
        CV_Assert_N(node, idx < node->input_size());
        return node->input(idx);
    }

    virtual std::string getType() const CV_OVERRIDE
    {
        return node ? node->op_type() : "";
    }

    virtual void setType(const std::string& type) CV_OVERRIDE
    {
        CV_Assert(node);
        node->set_op_type(type);
    }

    virtual void setInputNames(const std::vector<std::string>& inputs) CV_OVERRIDE
    {
        CV_Assert(node);
        node->clear_input();
        for (int i = 0; i < inputs.size(); ++i)
            node->add_input(inputs[i]);
    }

    opencv_onnx::NodeProto* node;
};

// ONNX graph's inputs are separate from nodes so we index them before the rest of nodes.
class ONNXGraphWrapper : public ImportGraphWrapper
{
public:
    ONNXGraphWrapper(opencv_onnx::GraphProto& _net) : net(_net)
    {
        numInputs = net.input_size();
        numInitializers = net.initializer_size();
    }

    virtual Ptr<ImportNodeWrapper> getNode(int idx) const CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = 0;
        if (idx >= numInputs + numInitializers)
            node = net.mutable_node(idx - numInputs - numInitializers);
        return makePtr<ONNXNodeWrapper>(node);
    }

    int getInputInitializerId(int node_id, int node_input_id)
    {
        auto node = getNode(node_id);
        std::string node_input_name = node->getInputName(node_input_id);
        for (int i = 0; i < numInitializers; ++i)
            if (net.initializer(i).name() == node_input_name)
                return i;
        // CV_Error(Error::StsParseError, "Initializer with name " + node_input_name + " not found");
        return -1;
    }

    Mat getMatFromInitializer(int idx)
    {
        const opencv_onnx::TensorProto& tensor_proto = net.initializer(idx);
        return getMatFromTensor(tensor_proto);
    }

    std::string getNameOfInitializer(int idx) const
    {
        const opencv_onnx::TensorProto& tensor_proto = net.initializer(idx);
        return tensor_proto.name();
    }

    virtual int getNumNodes() const CV_OVERRIDE
    {
        return numInputs + numInitializers + net.node_size();
    }

    virtual int getNumOutputs(int nodeId) const CV_OVERRIDE
    {
        if (nodeId < numInputs + numInitializers)
            return 1;
        else
            return net.node(nodeId - numInputs - numInitializers).output_size();
    }

    virtual std::string getOutputName(int nodeId, int outId) const CV_OVERRIDE
    {
        CV_Assert(outId < getNumOutputs(nodeId));
        if (nodeId < numInputs)
            return net.input(nodeId).name();
        else if (nodeId < numInputs + numInitializers)
            return net.initializer(nodeId - numInputs).name();
        else
            return net.node(nodeId - numInputs - numInitializers).output(outId);
    }

    virtual void removeNode(int idx) CV_OVERRIDE
    {
        CV_Assert(idx >= numInputs + numInitializers);
        net.mutable_node()->DeleteSubrange(idx - numInputs - numInitializers, 1);
    }

private:
    int numInputs, numInitializers;
    opencv_onnx::GraphProto& net;
};

/*  Fusion for Gelu.

    Graph before fusion:
           +---------------------------------------------+
           |                                             |
        [Input] -> Div[B=sqrt(2)] -> Erf -> Add[B=1] -> Mul -> Mul[B=0.5] -> [Output]

    Graph after fusion:
        [Input] -> Gelu -> [Output]

*/
class GeluSubGraph : public Subgraph
{
public:
    GeluSubGraph()
    {
        int input = addNodeToMatch("");
        int div = addNodeToMatch("Div", input, addNodeToMatch("") /* B=sqrt(2) */ );
        int erf = addNodeToMatch("Erf", div);
        int add = addNodeToMatch("Add", erf, addNodeToMatch("") /* B=1 */ );
        int mul = addNodeToMatch("Mul", input, add);
        addNodeToMatch("Mul", mul, addNodeToMatch("") /* B=0.5 */) ;

        setFusedNode("Gelu", input);
    }

    static float extractConstant(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
    {
        auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
        int initializer_id = onnx_net->getInputInitializerId(node_id, input_id);
        if (initializer_id != -1)
        {
            Mat const_mat = onnx_net->getMatFromInitializer(initializer_id);
            return *const_mat.ptr<float>();
        }
        else
        {
            const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
            int constant_id = getInputNodeId(net, node, input_id);
            Ptr<ImportNodeWrapper> constant_ptr = net->getNode(constant_id);
            opencv_onnx::NodeProto* constant_node = constant_ptr.dynamicCast<ONNXNodeWrapper>()->node;
            opencv_onnx::TensorProto constant_proto = constant_node->attribute(0).t();
            Mat constant_mat = getMatFromTensor(constant_proto);
            return *constant_mat.ptr<float>();
        }
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            // Check Div[B=sqrt(2)]
            float divisor = extractConstant(net, matchedNodesIds[0], 1);
            if (std::fabs(divisor - M_SQRT2) >= std::numeric_limits<float>::epsilon())
                return false;

            // Check Add[B=1]
            float add_const = extractConstant(net, matchedNodesIds[2], 1);
            if (std::fabs(add_const - 1.f) >= std::numeric_limits<float>::epsilon())
                return false;

            // Check Mul[B=0.5]
            float mul_const = extractConstant(net, matchedNodesIds[4], 1);
            if (std::fabs(mul_const - 0.5f) >= std::numeric_limits<float>::epsilon())
                return false;

            return true;
        }
        return false;
    }
};

/*  Fusion for GeluApproximation.

    Graph before fusion:
           +--------+------+----------------+------------------------------------+
           |        |      |                |                                    |
        [Input] -> Mul -> Mul -> Mul[ ] -> Add -> Mul[ ] -> Tanh -> Add[A=1] -> Mul -> Mul(A=0.5) -> [Output]
                                    /                  \
                    A=0.044714998453855515          A=sqrt(2/pie)

    Graph after fusion:
        [Input] -> GeluApproximation -> [Output]

*/
class GeluApproximationSubGraph : public Subgraph
{
public:
    GeluApproximationSubGraph()
    {
        int input = addNodeToMatch("");
        int mul0 = addNodeToMatch("Mul", input, input);
        int mul1 = addNodeToMatch("Mul", input, mul0);
        int mul2 = addNodeToMatch("Mul", addNodeToMatch("") /* A=0.044714998453855515 */, mul1);
        int add0 = addNodeToMatch("Add", input, mul2);
        int mul3 = addNodeToMatch("Mul", addNodeToMatch("") /* A=sqrt(2/pie) */, add0);
        int tanh = addNodeToMatch("Tanh", mul3);
        int add1 = addNodeToMatch("Add", addNodeToMatch("") /* A=1 */, tanh);
        int mul4 = addNodeToMatch("Mul", input, add1);
        addNodeToMatch("Mul", addNodeToMatch("") /* A=0.5 */, mul4);

        setFusedNode("GeluApproximation", input);
    }

    static float extractConstant(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
    {
        auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
        int initializer_id = onnx_net->getInputInitializerId(node_id, input_id);
        if (initializer_id != -1)
        {
            Mat const_mat = onnx_net->getMatFromInitializer(initializer_id);
            return *const_mat.ptr<float>();
        }
        else
        {
            const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
            int constant_id = getInputNodeId(net, node, input_id);
            Ptr<ImportNodeWrapper> constant_ptr = net->getNode(constant_id);
            opencv_onnx::NodeProto* constant_node = constant_ptr.dynamicCast<ONNXNodeWrapper>()->node;
            opencv_onnx::TensorProto constant_proto = constant_node->attribute(0).t();
            Mat constant_mat = getMatFromTensor(constant_proto);
            return *constant_mat.ptr<float>();
        }
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            // Check Mul[A=0.044714998453855515]
            float coef = extractConstant(net, matchedNodesIds[2], 0);
            if (coef - 0.044714998453855515 >= 1e-6)
                return false;

            // Check Mul[A=sqrt(2/pie)]
            float sqrt_2_pie = extractConstant(net, matchedNodesIds[4], 0);
            if (sqrt_2_pie - 0.7978845834732056 >= 1e-6)
                return false;

            // Check Add[A=1]
            float add_const = extractConstant(net, matchedNodesIds[6], 0);
            if (add_const - 1.f >= 1e-6)
                return false;

            // Check Mul[A=0.5]
            float mul_const = extractConstant(net, matchedNodesIds[8], 0);
            if (mul_const - 0.5f >= 1e-6)
                return false;

            return true;
        }
        return false;
    }
};

/*  Fusion for LayerNormalization.

    Graph before fusion
           +-> ReduceMean ->+
           |                |
        [Input]  ------->  Sub  ----------------------------------------------->  Div -> Mul(B=weight) -> Add(B=bias) -> [Output]
                            |                                                      |
                            +-> Pow(Y=2) -> ReduceMean -> Add(B=epsilon) -> Sqrt ->+

    Graph after fusion
        [Input] -> LayerNorm -> [Output]
                        \
                    [weight], [bias]
*/
class LayerNormSubGraph : public Subgraph
{
public:
    LayerNormSubGraph() : axis(-1), epsilon(1e-5)
    {
        int input = addNodeToMatch("");
        int mean = addNodeToMatch("ReduceMean", input);

        int sub = addNodeToMatch("Sub", input, mean);

        int pow = addNodeToMatch("Pow", sub, addNodeToMatch(""));
        int mean1 = addNodeToMatch("ReduceMean", pow);
        int add = addNodeToMatch("Add", mean1, addNodeToMatch(""));
        int sqrt = addNodeToMatch("Sqrt", add);

        int div = addNodeToMatch("Div", sub, sqrt);
        int mul = addNodeToMatch("Mul", div, addNodeToMatch(""));
        addNodeToMatch("Add", mul, addNodeToMatch(""));

        setFusedNode("LayerNormalization", input);
    }

    static float extractConstant(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
    {
        auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
        int initializer_id = onnx_net->getInputInitializerId(node_id, input_id);
        if (initializer_id != -1) // initializer
        {
            Mat const_mat = onnx_net->getMatFromInitializer(initializer_id);
            return *const_mat.ptr<float>();
        }
        else
        {
            const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
            int constant_id = getInputNodeId(net, node, input_id);
            Ptr<ImportNodeWrapper> constant_ptr = net->getNode(constant_id);
            opencv_onnx::NodeProto* constant_node = constant_ptr.dynamicCast<ONNXNodeWrapper>()->node;
            opencv_onnx::TensorProto constant_proto = constant_node->attribute(0).t();
            Mat constant_mat = getMatFromTensor(constant_proto);
            return *constant_mat.ptr<float>();
        }
    }

    static float extractAxis(const Ptr<ImportGraphWrapper>& net, int node_id)
    {
        Ptr<ImportNodeWrapper> mean_ptr = net->getNode(node_id);
        opencv_onnx::NodeProto* mean_node = mean_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        int axis_ = -1;
        for (int i = 0; i < mean_node->attribute_size(); i++)
        {
            opencv_onnx::AttributeProto attr = mean_node->attribute(i);
            if (attr.name() != "axes")
                continue;
            axis_ = static_cast<int>(attr.ints(0));
        }
        return axis_;
    }

    static std::string getInputName(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
    {
        auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
        int initializer_id = onnx_net->getInputInitializerId(node_id, input_id);
        if (initializer_id != -1)
        {
            return onnx_net->getNameOfInitializer(initializer_id);
        }
        else
        {
            const auto node = net->getNode(node_id);
            return node->getInputName(input_id);
        }
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            float pow_exp = extractConstant(net, matchedNodesIds[2], 1);
            if (pow_exp - 2 > 1e-5) // not pow(2)
                return false;

            int axis_mean1 = extractAxis(net, matchedNodesIds[0]);
            int axis_mean2 = extractAxis(net, matchedNodesIds[3]);
            if (axis_mean1 != axis_mean2)
                return false;
            axis = axis_mean1;

            epsilon = extractConstant(net, matchedNodesIds[4], 1);

            weight_name = getInputName(net, matchedNodesIds[7], 1);
            bias_name = getInputName(net, matchedNodesIds[8], 1);

            return true;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        // axis
        opencv_onnx::AttributeProto* attr_axis = node->add_attribute();
        attr_axis->set_name("axis");
        attr_axis->set_i(axis);
        // epsilon
        opencv_onnx::AttributeProto* attr_epsilon = node->add_attribute();
        attr_epsilon->set_name("epsilon");
        attr_epsilon->set_f(epsilon);
        // add input
        node->add_input(weight_name);
        node->add_input(bias_name);
    }

protected:
    int axis;
    float epsilon;
    std::string weight_name;
    std::string bias_name;
};

class SoftMaxSubgraphBase : public Subgraph
{
public:
    SoftMaxSubgraphBase() : axis(1), id(-1) {}

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            CV_Assert(id >= 0 && id < matchedNodesIds.size());
            Ptr<ImportNodeWrapper> sum = net->getNode(matchedNodesIds[id]);
            opencv_onnx::NodeProto* node = sum.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if (attr.name() != "axes")
                    continue;
                if (attr.ints_size() != 1)
                    CV_Error(Error::StsNotImplemented, format("Unexpected number of axes: %d", attr.ints_size()));
                axis = attr.ints(0);
                return true;
            }
            CV_Error(Error::StsNotImplemented, "Missed axes attribute");
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attr = node->add_attribute();
        attr->set_name("axis");
        attr->set_i(axis);
    }

protected:
    int axis;
    int id;
};

class SoftMaxSubgraph : public SoftMaxSubgraphBase
{
public:
    SoftMaxSubgraph()
    {
        int input = addNodeToMatch("");
        int inpExp = addNodeToMatch("Exp", input);

        int sum = addNodeToMatch("ReduceSum", inpExp);
        id = 1;

        addNodeToMatch("Div", inpExp, sum);
        setFusedNode("Softmax", input);
    }
};

class SoftMaxSubgraph2 : public SoftMaxSubgraphBase {
public:
    SoftMaxSubgraph2() {
        int input = addNodeToMatch("");

        int reducemax = addNodeToMatch("ReduceMax", input);
        id = 0;

        int sub = addNodeToMatch("Sub", input, reducemax);
        int exp = addNodeToMatch("Exp", sub);
        int reducesum = addNodeToMatch("ReduceSum", exp, addNodeToMatch(""));
        addNodeToMatch("Div", exp, reducesum);
        setFusedNode("Softmax", input);
    }
};

class LogSoftMaxSubgraph : public SoftMaxSubgraphBase
{
public:
    LogSoftMaxSubgraph()
    {
        int input = addNodeToMatch("");

        int reducemax = addNodeToMatch("ReduceMax", input);
        id = 0;

        int sub_1 = addNodeToMatch("Sub", input, reducemax);
        int exp = addNodeToMatch("Exp", sub_1);
        int reducesum = addNodeToMatch("ReduceSum", exp, addNodeToMatch(""));
        int log = addNodeToMatch("Log", reducesum);
        addNodeToMatch("Sub", sub_1, log);
        setFusedNode("LogSoftmax", input);
    }
};

class HardSwishSubgraph : public Subgraph
{
public:
    HardSwishSubgraph()
    {
        int input = addNodeToMatch("");
        int hardSigmoid = addNodeToMatch("HardSigmoid", input);
        addNodeToMatch("Mul", input, hardSigmoid);
        setFusedNode("HardSwish", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> hardSigmoid = net->getNode(matchedNodesIds[0]);
            opencv_onnx::NodeProto* node = hardSigmoid.dynamicCast<ONNXNodeWrapper>()->node;

            uint8_t matched = 0;
            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if ((attr.name() == "alpha" && attr.f() == 1.f / 6.f) ||
                    (attr.name() == "beta" && attr.f() == 0.5f))
                {
                    ++matched;
                }
            }
            return matched == 2;
        }
        return false;
    }
};

class CeluSubgraph : public Subgraph
{
public:
    CeluSubgraph() : alpha(1.f)
    {
        int input = addNodeToMatch("");
        int div = addNodeToMatch("Div", input, addNodeToMatch(""));
        int elu = addNodeToMatch("Elu", div);
        addNodeToMatch("Mul", addNodeToMatch(""), elu);
        setFusedNode("Celu", input);
    }

    static float extractAlpha(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
    {
        const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
        int const_id = getInputNodeId(net, node, input_id);
        Ptr<ImportNodeWrapper> alpha_ptr = net->getNode(const_id);
        opencv_onnx::NodeProto* alpha_node = alpha_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::TensorProto alpha_proto = alpha_node->attribute(0).t();
        Mat alpha_mat = getMatFromTensor(alpha_proto);
        return *alpha_mat.ptr<float>();
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            float alpha_div = extractAlpha(net, matchedNodesIds[0], 1);
            float alpha_mul = extractAlpha(net, matchedNodesIds[2], 0);
            float alpha_elu = 1.f;

            Ptr<ImportNodeWrapper> elu_ptr = net->getNode(matchedNodesIds[1]);
            opencv_onnx::NodeProto* elu_node = elu_ptr.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < elu_node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = elu_node->attribute(i);
                if (attr.name() != "alpha")
                    continue;
                alpha_elu = attr.f();
            }

            alpha = alpha_div;
            return alpha_elu == 1.f && alpha_div == alpha_mul;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* alpha_attr = node->add_attribute();
        alpha_attr->set_name("alpha");
        alpha_attr->set_f(alpha);
    }

protected:
    float alpha;
};

class NormalizeSubgraphBase : public Subgraph
{
public:
    NormalizeSubgraphBase(int _normNodeOrder = 0) : axis(1), normNodeOrder(_normNodeOrder) {}

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> norm = net->getNode(matchedNodesIds[normNodeOrder]);
            opencv_onnx::NodeProto* node = norm.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if (attr.name() != "axes")
                    continue;
                if (attr.ints_size() != 1)
                    CV_Error(Error::StsNotImplemented, format("Unexpected number of axes: %d", attr.ints_size()));
                axis = attr.ints(0);
                return true;
            }
            CV_Error(Error::StsNotImplemented, "Missed axes attribute");
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* axis_attr = node->add_attribute();
        axis_attr->set_name("axis");
        axis_attr->set_i(axis);

        opencv_onnx::AttributeProto* end_axis_attr = node->add_attribute();
        end_axis_attr->set_name("end_axis");
        end_axis_attr->set_i(axis);
    }

protected:
    int axis, normNodeOrder;
};

class NormalizeSubgraph1 : public NormalizeSubgraphBase
{
public:
    NormalizeSubgraph1()
    {
        int input = addNodeToMatch("");
        int norm = addNodeToMatch("ReduceL2", input);
        addNodeToMatch("Div", input, norm);
        setFusedNode("Normalize", input);
    }
};

class NormalizeSubgraph2 : public NormalizeSubgraphBase
{
public:
    NormalizeSubgraph2()
    {
        int input = addNodeToMatch("");
        int norm = addNodeToMatch("ReduceL2", input);
        int clip = addNodeToMatch("Clip", norm);
        int shape = addNodeToMatch("Shape", input);
        int expand = addNodeToMatch("Expand", clip, shape);
        addNodeToMatch("Div", input, expand);
        setFusedNode("Normalize", input);
    }
};

class NormalizeSubgraph2_2 : public NormalizeSubgraphBase
{
public:
    NormalizeSubgraph2_2()
    {
        int input = addNodeToMatch("");
        int norm = addNodeToMatch("ReduceL2", input);

        int min = addNodeToMatch("");
        int max = addNodeToMatch("");
        int clip = addNodeToMatch("Clip", norm, min, max);

        int shape = addNodeToMatch("");
        int expand = addNodeToMatch("Expand", clip, shape);

        addNodeToMatch("Div", input, expand);

        setFusedNode("Normalize", input);
    }
};

class NormalizeSubgraph3 : public NormalizeSubgraphBase
{
public:
    NormalizeSubgraph3() : NormalizeSubgraphBase(1)
    {
        int input = addNodeToMatch("");
        int power = addNodeToMatch("Constant");
        int squared = addNodeToMatch("Pow", input, power);
        int sum = addNodeToMatch("ReduceSum", squared);
        int sqrtNode = addNodeToMatch("Sqrt", sum);
        int eps = addNodeToMatch("Constant");
        int add = addNodeToMatch("Add", sqrtNode, eps);

        addNodeToMatch("Div", input, add);
        setFusedNode("Normalize", input);
    }
};

class NormalizeSubgraph4 : public NormalizeSubgraphBase
{
public:
    NormalizeSubgraph4() : NormalizeSubgraphBase(1)
    {
        int input = addNodeToMatch("");
        int mul = addNodeToMatch("Mul", input, input);
        int sum = addNodeToMatch("ReduceSum", mul);
        int eps = addNodeToMatch("");
        int max = addNodeToMatch("Max", sum, eps);
        int sqrt = addNodeToMatch("Sqrt", max);
        int reciprocal = addNodeToMatch("Reciprocal", sqrt);
        addNodeToMatch("Mul", input, reciprocal);
        setFusedNode("Normalize", input);
    }
};

class NormalizeSubgraph5 : public NormalizeSubgraphBase
{
public:
    NormalizeSubgraph5() : NormalizeSubgraphBase(1)
    {
        int input = addNodeToMatch("");
        int mul = addNodeToMatch("Mul", input, input);
        int sum = addNodeToMatch("ReduceSum", mul);
        int clip = addNodeToMatch("Clip", sum);
        int sqrt = addNodeToMatch("Sqrt", clip);
        int one = addNodeToMatch("Constant");
        int div = addNodeToMatch("Div", one, sqrt);
        addNodeToMatch("Mul", input, div);
        setFusedNode("Normalize", input);
    }
};

class GatherCastSubgraph : public Subgraph
{
public:
    GatherCastSubgraph()
    {
        int input = addNodeToMatch("");
        int index = addNodeToMatch("Constant");
        int gather = addNodeToMatch("Gather", input, index);
        addNodeToMatch("Cast", gather);
        setFusedNode("Gather", input, index);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        bool retVal = Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds);
        size_t matchedNodesNum = matchedNodesIds.size();
        // Now we check if merging can be made for these Gather and Cast nodes
        if (!retVal || matchedNodesNum < 2)
            return retVal;
        else {
            int nodeToMatch = matchedNodesIds[matchedNodesNum - 1];
            const Ptr<ImportNodeWrapper> node = net->getNode(nodeToMatch);
            if (node->getType() == "Cast") {
                int inpNodeId = matchedNodesIds[matchedNodesNum - 2];
                const Ptr<ImportNodeWrapper> inpNode = net->getNode(inpNodeId);
                if (inpNode->getType() == "Gather") {
                    int numNodes = net->getNumNodes();
                    std::string inpNodeName = node->getInputName(0);
                    for (int i = 0; i < numNodes; ++i) {
                        const Ptr<ImportNodeWrapper> node_to_check = net->getNode(i);
                        int numInp = node_to_check->getNumInputs();
                        for (int inp = 0; inp < numInp; ++inp) {
                            if (i != nodeToMatch && inpNodeName == node_to_check->getInputName(0)) {
                                // Another node has the same input node, so it cannot be merged.
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return retVal;
    }
};

class ExpandSubgraph : public Subgraph
{
public:
    ExpandSubgraph()
    {
        int input = addNodeToMatch("");
        int values = addNodeToMatch("");
        int init = addNodeToMatch("ConstantOfShape", values);
        int coeff = addNodeToMatch("Constant");
        int mul = addNodeToMatch("Mul", init, coeff);
        int shape = addNodeToMatch("Constant");
        int condition = addNodeToMatch("Equal", shape, mul);
        int where = addNodeToMatch("Where", condition, init, addNodeToMatch("Constant"));
        addNodeToMatch("Expand", input, where);
        setFusedNode("Expand", input, shape);
    }
};

class MishSubgraph : public Subgraph
{
public:
    MishSubgraph()
    {
        int input = addNodeToMatch("");
        int softplus = addNodeToMatch("Softplus", input);
        int tanh = addNodeToMatch("Tanh", softplus);
        addNodeToMatch("Mul", input, tanh);
        setFusedNode("Mish", input);
    }
};

// softplus(x) = log(exp(x) + 1)
class SoftplusSubgraph: public Subgraph
{
public:
    SoftplusSubgraph()
    {
        int input = addNodeToMatch("");
        int exp = addNodeToMatch("Exp", input);
        int addVal = addNodeToMatch("");
        int add = addNodeToMatch("Add", addVal, exp);
        addNodeToMatch("Log", add);
        setFusedNode("Softplus", input);
    }
};

class SoftplusSubgraph2: public Subgraph
{
public:
    SoftplusSubgraph2()
    {
        int input = addNodeToMatch("");
        int exp = addNodeToMatch("Exp", input);
        int addVal = addNodeToMatch("");
        int add = addNodeToMatch("Add", exp, addVal);
        addNodeToMatch("Log", add);
        setFusedNode("Softplus", input);
    }
};

class MulCastSubgraph : public Subgraph
{
public:
    MulCastSubgraph()
    {
        int input = addNodeToMatch("");
        int scaleNode = addNodeToMatch("Constant");
        int mul = addNodeToMatch("Mul", input, scaleNode);
        addNodeToMatch("Cast", mul);
        setFusedNode("Mul", input, scaleNode);
    }
};

class ExtractScalesSubgraph : public Subgraph
{
public:
    ExtractScalesSubgraph()
    {
        input = addNodeToMatch("");

        int indexH = addNodeToMatch("Constant");
        int shape1 = addNodeToMatch("Shape", input);
        int gather1 = addNodeToMatch("Gather", shape1, indexH);
        scaleHNode = addNodeToMatch("Constant");
        int mul1 = addNodeToMatch("Mul", gather1, scaleHNode);
        int floor1 = addNodeToMatch("Floor", mul1);

        int indexW = addNodeToMatch("Constant");
        int shape2 = addNodeToMatch("Shape", input);
        int gather2 = addNodeToMatch("Gather", shape2, indexW);
        scaleWNode = addNodeToMatch("Constant");
        int mul2 = addNodeToMatch("Mul", gather2, scaleWNode);
        int floor2 = addNodeToMatch("Floor", mul2);

        int unsqueeze1 = addNodeToMatch("Unsqueeze", floor1);
        int unsqueeze2 = addNodeToMatch("Unsqueeze", floor2);
        concatId = addNodeToMatch("Concat", unsqueeze1, unsqueeze2);
    }

    void finalize(const Ptr<ImportGraphWrapper>& net,
                  const Ptr<ImportNodeWrapper>& fusedNode,
                  std::vector<Ptr<ImportNodeWrapper> >& inputs) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* constant_node = inputs[1].dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::TensorProto tensor_proto = constant_node->attribute(0).t();
        Mat scaleW = getMatFromTensor(tensor_proto);
        CV_Assert(scaleW.total() == 1);
        scaleW.convertTo(scaleW, CV_32F);

        constant_node = inputs[2].dynamicCast<ONNXNodeWrapper>()->node;
        tensor_proto = constant_node->attribute(0).t();
        Mat scaleH = getMatFromTensor(tensor_proto);
        CV_Assert(scaleH.total() == 1);
        scaleH.convertTo(scaleH, CV_32F);

        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attrH = node->add_attribute();
        attrH->set_name("height_scale");
        attrH->set_i(scaleH.at<float>(0));
        opencv_onnx::AttributeProto* attrW = node->add_attribute();
        attrW->set_name("width_scale");
        attrW->set_i(scaleW.at<float>(0));

        node->mutable_input()->DeleteSubrange(1, 2);  // Remove two last inputs
    }

protected:
    int input, concatId;
    int scaleHNode, scaleWNode;
};

class UpsampleSubgraph : public ExtractScalesSubgraph
{
public:
    UpsampleSubgraph() : ExtractScalesSubgraph()
    {
        int shape = addNodeToMatch("Shape", input);
        int slice = addNodeToMatch("Slice", shape);

        int castConcat = addNodeToMatch("Cast", concatId);
        int castSlice = addNodeToMatch("Cast", slice);
        int divide = addNodeToMatch("Div", castConcat, castSlice);

        int constant = addNodeToMatch("Constant");
        int concat = addNodeToMatch("Concat", constant, divide);

        addNodeToMatch("Upsample", input, concat);
        setFusedNode("Upsample", input, scaleWNode, scaleHNode);
    }
};

class ResizeSubgraph1 : public ExtractScalesSubgraph
{
public:
    ResizeSubgraph1() : ExtractScalesSubgraph()
    {
        int shape = addNodeToMatch("Shape", input);
        int slice = addNodeToMatch("Slice", shape, addNodeToMatch("Constant"), addNodeToMatch("Constant"), addNodeToMatch("Constant"));

        int castConcat = addNodeToMatch("Cast", concatId);
        int concat = addNodeToMatch("Concat", slice, castConcat);
        int constant = addNodeToMatch("Constant");

        addNodeToMatch("Resize", input, constant, constant, concat);
        setFusedNode("Upsample", input, scaleWNode, scaleHNode);
    }
};

class ResizeSubgraph2 : public ExtractScalesSubgraph
{
public:
    ResizeSubgraph2() : ExtractScalesSubgraph()
    {
        int constantConcat = addNodeToMatch("Constant");
        int castConcat = addNodeToMatch("Cast", concatId);
        int concat = addNodeToMatch("Concat", constantConcat, castConcat);
        int constant = addNodeToMatch("Constant");

        addNodeToMatch("Resize", input, constant, constant, concat);
        setFusedNode("Upsample", input, scaleWNode, scaleHNode);
    }
};

class BatchNormalizationSubgraphBase : public Subgraph
{
public:
    BatchNormalizationSubgraphBase()
    {
        input  = addNodeToMatch("");
        var    = addNodeToMatch("");
        mean   = addNodeToMatch("");
        weight = addNodeToMatch("");
        bias   = addNodeToMatch("");
        A      = addNodeToMatch("");
        shape1 = addNodeToMatch("");
        shape2 = addNodeToMatch("");
    }
protected:
    int input, var, mean, weight, bias, A, shape1, shape2;
};

class BatchNormalizationSubgraph1 : public BatchNormalizationSubgraphBase
{
public:
    BatchNormalizationSubgraph1()
    {
        int reshape1 = addNodeToMatch("Reshape", weight, shape1);
        int reshape2 = addNodeToMatch("Reshape", bias, shape2);
        int shape3 = addNodeToMatch("Constant");
        int reshape3 = addNodeToMatch("Reshape", var, shape3);
        int shape4 = addNodeToMatch("Constant");
        int reshape4 = addNodeToMatch("Reshape", mean, shape4);
        int sqrtNode = addNodeToMatch("Sqrt", reshape3);
        int divNode = addNodeToMatch("Div", A, sqrtNode);
        int mul1 = addNodeToMatch("Mul", reshape1, divNode);
        int mul2 = addNodeToMatch("Mul", reshape4, mul1);
        int sub = addNodeToMatch("Sub", reshape2, mul2);
        int mul3 = addNodeToMatch("Mul", input, mul1);
        addNodeToMatch("Add", mul3, sub);
        setFusedNode("BatchNormalization", input, weight, bias, mean, var);
    }
};

class BatchNormalizationSubgraph2 : public BatchNormalizationSubgraphBase
{
public:
    BatchNormalizationSubgraph2()
    {
        int sqrtNode = addNodeToMatch("Sqrt", var);
        int divNode = addNodeToMatch("Div", A, sqrtNode);
        int mul1 = addNodeToMatch("Mul", weight, divNode);
        int reshape2 = addNodeToMatch("Reshape", mul1, shape2);

        int mulMean = addNodeToMatch("Mul", mean, mul1);
        int sub = addNodeToMatch("Sub", bias, mulMean);
        int reshape1 = addNodeToMatch("Reshape", sub, shape1);

        int mulInput = addNodeToMatch("Mul", input, reshape2);
        addNodeToMatch("Add", mulInput, reshape1);
        setFusedNode("BatchNormalization", input, weight, bias, mean, var);
    }
};

void simplifySubgraphs(opencv_onnx::GraphProto& net)
{
    std::vector<Ptr<Subgraph> > subgraphs;
    subgraphs.push_back(makePtr<GeluSubGraph>());
    subgraphs.push_back(makePtr<GeluApproximationSubGraph>());
    subgraphs.push_back(makePtr<LayerNormSubGraph>());
    subgraphs.push_back(makePtr<GatherCastSubgraph>());
    subgraphs.push_back(makePtr<MulCastSubgraph>());
    subgraphs.push_back(makePtr<UpsampleSubgraph>());
    subgraphs.push_back(makePtr<ResizeSubgraph1>());
    subgraphs.push_back(makePtr<ResizeSubgraph2>());
    subgraphs.push_back(makePtr<SoftMaxSubgraph>());
    subgraphs.push_back(makePtr<SoftMaxSubgraph2>());
    subgraphs.push_back(makePtr<LogSoftMaxSubgraph>());
    subgraphs.push_back(makePtr<HardSwishSubgraph>());
    subgraphs.push_back(makePtr<CeluSubgraph>());
    subgraphs.push_back(makePtr<NormalizeSubgraph1>());
    subgraphs.push_back(makePtr<NormalizeSubgraph2>());
    subgraphs.push_back(makePtr<NormalizeSubgraph2_2>());
    subgraphs.push_back(makePtr<NormalizeSubgraph3>());
    subgraphs.push_back(makePtr<BatchNormalizationSubgraph1>());
    subgraphs.push_back(makePtr<BatchNormalizationSubgraph2>());
    subgraphs.push_back(makePtr<ExpandSubgraph>());
    subgraphs.push_back(makePtr<SoftplusSubgraph>());
    subgraphs.push_back(makePtr<SoftplusSubgraph2>());
    subgraphs.push_back(makePtr<MishSubgraph>());
    subgraphs.push_back(makePtr<NormalizeSubgraph4>());
    subgraphs.push_back(makePtr<NormalizeSubgraph5>());

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new ONNXGraphWrapper(net)), subgraphs);
}

Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto)
{
    if (tensor_proto.raw_data().empty() && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty() &&
        tensor_proto.int32_data().empty())
        return Mat();

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;
    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes.push_back(tensor_proto.dims(i));
    }
    if (sizes.empty())
        sizes.assign(1, 1);
    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT) {

        if (!tensor_proto.float_data().empty()) {
            const ::google::protobuf::RepeatedField<float> field = tensor_proto.float_data();
            Mat(sizes, CV_32FC1, (void*)field.data()).copyTo(blob);
        }
        else {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_FLOAT16)
    {
        // FIXME, for now, we only load FP16 Tensor as FP32 Mat, full support for FP16 is required in the future.
        CV_LOG_ONCE_WARNING(NULL, "DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.");

        // ONNX saves float 16 data in two format: int32 and raw_data.
        // Link: https://github.com/onnx/onnx/issues/4460#issuecomment-1224373746
        if (!tensor_proto.int32_data().empty())
        {
            int offset = 0;
#ifdef WORDS_BIGENDIAN
            offset = 1;
#endif
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();

            AutoBuffer<float16_t, 16> aligned_val;
            size_t sz = tensor_proto.int32_data().size();
            aligned_val.allocate(sz);
            float16_t* bufPtr = aligned_val.data();

            float16_t *fp16Ptr = (float16_t *)field.data();
            for (int i = 0; i < sz; i++)
            {
                bufPtr[i] = fp16Ptr[i*2 + offset];
            }
            Mat(sizes, CV_16FC1, bufPtr).convertTo(blob, CV_32FC1);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required.
            AutoBuffer<float16_t, 16> aligned_val;
            if (!isAligned<sizeof(float16_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(float16_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (char*)aligned_val.data();
            }
#endif
            Mat(sizes, CV_16FC1, val).convertTo(blob, CV_32FC1);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        char* val = nullptr;
        if (!field.empty())
            val = (char *)field.data();
        else
            val = const_cast<char*>(tensor_proto.raw_data().c_str()); // sometime, the double will be stored at raw_data.

#if CV_STRONG_ALIGNMENT
        // Aligned pointer is required.
        AutoBuffer<double, 16> aligned_val;
        if (!isAligned<sizeof(double)>(val))
        {
            size_t sz = tensor_proto.raw_data().size();
            aligned_val.allocate(divUp(sz, sizeof(double)));
            memcpy(aligned_val.data(), val, sz);
            val = (char*)aligned_val.data();
        }
#endif
        Mat(sizes, CV_64FC1, val).convertTo(blob, CV_32FC1);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT32)
    {
        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).copyTo(blob);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32SC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        if (!tensor_proto.int64_data().empty()) {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();
            convertInt64ToInt32(src, dst, blob.total());
        }
        else
        {
            const char* val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<int64_t, 16> aligned_val;
            if (!isAligned<sizeof(int64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(int64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char*)aligned_val.data();
            }
#endif
            const int64_t* src = reinterpret_cast<const int64_t*>(val);
            convertInt64ToInt32(src, dst, blob.total());
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT8 ||
             datatype == opencv_onnx::TensorProto_DataType_UINT8)
    {
        // TODO : Add support for uint8 weights and acitvations. For now, converting uint8 tensors to int8.
        int offset = datatype == opencv_onnx::TensorProto_DataType_INT8 ? 0 : -128;
        int depth = datatype == opencv_onnx::TensorProto_DataType_INT8 ? CV_8S : CV_8U;

        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, CV_8S, 1.0, offset);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, depth, val).convertTo(blob, CV_8S, 1.0, offset);
        }
    }
    else
    {
        std::string errorMsg = "Unsupported data type: " +
                            opencv_onnx::TensorProto_DataType_Name(datatype);

        if (!DNN_DIAGNOSTICS_RUN)
        {
            CV_Error(Error::StsUnsupportedFormat, errorMsg);
        }
        CV_LOG_ERROR(NULL, errorMsg);
        return blob;
    }
    if (tensor_proto.dims_size() == 0)
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // HAVE_PROTOBUF
