// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include "../graph_simplifier.hpp"
#include "onnx_graph_simplifier.hpp"

#include <queue>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

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

class SoftMaxSubgraph : public Subgraph
{
public:
    SoftMaxSubgraph() : axis(1)
    {
        int input = addNodeToMatch("");
        int inpExp = addNodeToMatch("Exp", input);
        int sum = addNodeToMatch("ReduceSum", inpExp);
        addNodeToMatch("Div", inpExp, sum);
        setFusedNode("Softmax", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> sum = net->getNode(matchedNodesIds[1]);
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

private:
    int axis;
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
    subgraphs.push_back(makePtr<GatherCastSubgraph>());
    subgraphs.push_back(makePtr<MulCastSubgraph>());
    subgraphs.push_back(makePtr<UpsampleSubgraph>());
    subgraphs.push_back(makePtr<ResizeSubgraph1>());
    subgraphs.push_back(makePtr<ResizeSubgraph2>());
    subgraphs.push_back(makePtr<SoftMaxSubgraph>());
    subgraphs.push_back(makePtr<NormalizeSubgraph1>());
    subgraphs.push_back(makePtr<NormalizeSubgraph2>());
    subgraphs.push_back(makePtr<NormalizeSubgraph3>());
    subgraphs.push_back(makePtr<BatchNormalizationSubgraph1>());
    subgraphs.push_back(makePtr<BatchNormalizationSubgraph2>());

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new ONNXGraphWrapper(net)), subgraphs);
}

Mat getMatFromTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (tensor_proto.raw_data().empty() && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty())
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
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        CV_Assert(!field.empty());
        Mat(sizes, CV_64FC1, (void*)field.data()).convertTo(blob, CV_32FC1);
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
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data type: " +
                        opencv_onnx::TensorProto_DataType_Name(datatype));
    if (tensor_proto.dims_size() == 0)
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
