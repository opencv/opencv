// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF

#include "../graph_simplifier.hpp"
#include "tf_graph_simplifier.hpp"
#include <queue>

namespace cv { namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

using ::google::protobuf::RepeatedField;
using ::google::protobuf::MapPair;

static Mat getTensorContentRef_(const tensorflow::TensorProto& tensor);
static inline
bool isAlignedMat(const Mat& m)
{
    int depth = m.depth();
    int alignment = CV_ELEM_SIZE1(depth);
    return (((size_t)m.data) & (alignment - 1)) == 0;
}


class TFNodeWrapper : public ImportNodeWrapper
{
public:
    TFNodeWrapper(tensorflow::NodeDef* _node) : node(_node) {}

    virtual int getNumInputs() const CV_OVERRIDE
    {
        return node->input_size();
    }

    virtual std::string getInputName(int idx) const CV_OVERRIDE
    {
        // If operation produces several tensors, they are specified by index
        // after ':' character. In example, "input:0".
        std::string name = node->input(idx);
        return name.substr(0, name.rfind(':'));
    }

    virtual std::string getType() const CV_OVERRIDE
    {
        return node->op();
    }

    virtual void setType(const std::string& type) CV_OVERRIDE
    {
        node->set_op(type);
    }

    virtual void setInputNames(const std::vector<std::string>& inputs) CV_OVERRIDE
    {
        node->clear_input();
        for (int i = 0; i < inputs.size(); ++i)
            node->add_input(inputs[i]);
    }

    tensorflow::NodeDef* node;
};

class TFGraphWrapper : public ImportGraphWrapper
{
public:
    TFGraphWrapper(tensorflow::GraphDef& _net) : net(_net) {}

    virtual Ptr<ImportNodeWrapper> getNode(int idx) const CV_OVERRIDE
    {
        return makePtr<TFNodeWrapper>(net.mutable_node(idx));
    }

    virtual int getNumNodes() const CV_OVERRIDE
    {
        return net.node_size();
    }

    virtual int getNumOutputs(int nodeId) const CV_OVERRIDE
    {
        return 1;
    }

    virtual std::string getOutputName(int nodeId, int outId) const CV_OVERRIDE
    {
        CV_Assert(outId == 0);
        return net.node(nodeId).name();
    }

    virtual void removeNode(int idx) CV_OVERRIDE
    {
        net.mutable_node()->DeleteSubrange(idx, 1);
    }

    tensorflow::GraphDef& net;
};

class TFSubgraph : public Subgraph
{
    virtual void finalize(const Ptr<ImportGraphWrapper>& netWrapper,
                          const Ptr<ImportNodeWrapper>& fusedNodeWrapper,
                          std::vector<Ptr<ImportNodeWrapper> >& inputs) CV_OVERRIDE
    {
        std::vector<tensorflow::NodeDef*> inputNodes(inputs.size());
        for (int i = 0; i < inputs.size(); ++i)
            inputNodes[i] = inputs[i].dynamicCast<TFNodeWrapper>()->node;
        finalize(netWrapper.dynamicCast<TFGraphWrapper>()->net,
                 fusedNodeWrapper.dynamicCast<TFNodeWrapper>()->node, inputNodes);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) {}
};

class BatchNormSubgraph : public TFSubgraph
{
public:
    BatchNormSubgraph()
    {
        int input = addNodeToMatch("");
        int epsilon = addNodeToMatch("Const");
        int moving_variance = addNodeToMatch("Const");
        int moving_mean = addNodeToMatch("Const");
        int beta = addNodeToMatch("Const");
        int gamma = addNodeToMatch("Const");
        int add = addNodeToMatch("Add", moving_variance, epsilon);
        int rsqrt = addNodeToMatch("Rsqrt", add);
        int mul = addNodeToMatch("Mul", rsqrt, gamma);
        int mul_1 = addNodeToMatch("Mul", input, mul);
        int mul_2 = addNodeToMatch("Mul", moving_mean, mul);
        int sub = addNodeToMatch("Sub", beta, mul_2);
        addNodeToMatch("Add", mul_1, sub);

        setFusedNode("FusedBatchNorm", input, gamma, beta, moving_mean, moving_variance, epsilon);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        Mat epsMat = getTensorContent(inputNodes.back()->attr().at("value").tensor());
        CV_CheckEQ(epsMat.total(), (size_t)1, ""); CV_CheckTypeEQ(epsMat.type(), CV_32FC1, "");

        fusedNode->mutable_input()->RemoveLast();
        fusedNode->clear_attr();
        tensorflow::AttrValue epsilon;
        epsilon.set_f(epsMat.at<float>(0));
        fusedNode->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("epsilon", epsilon));
    }
};

class BatchNormNoGammaSubgraph : public TFSubgraph
{
public:
    BatchNormNoGammaSubgraph()
    {
        int input = addNodeToMatch("");
        int epsilon = addNodeToMatch("Const");
        int moving_variance = addNodeToMatch("Const");
        int moving_mean = addNodeToMatch("Const");
        int beta = addNodeToMatch("Const");
        int add = addNodeToMatch("Add", moving_variance, epsilon);
        int rsqrt = addNodeToMatch("Rsqrt", add);
        int mul = addNodeToMatch("Mul", input, rsqrt);
        int mul_1 = addNodeToMatch("Mul", moving_mean, rsqrt);
        int sub = addNodeToMatch("Sub", beta, mul_1);
        addNodeToMatch("Add", mul, sub);

        // There is a fake reference to beta that will be replaced to a new gamma tensor.
        setFusedNode("FusedBatchNorm", input, beta, beta, moving_mean, moving_variance, epsilon);
    }

    virtual void finalize(tensorflow::GraphDef& net, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        Mat epsMat = getTensorContent(inputNodes.back()->attr().at("value").tensor());
        CV_CheckEQ(epsMat.total(), (size_t)1, ""); CV_CheckTypeEQ(epsMat.type(), CV_32FC1, "");

        fusedNode->mutable_input()->RemoveLast();
        fusedNode->clear_attr();
        tensorflow::AttrValue epsilon;
        epsilon.set_f(epsMat.at<float>(0));
        fusedNode->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("epsilon", epsilon));

        tensorflow::NodeDef* gamma = net.add_node();
        gamma->set_op("Const");
        gamma->set_name(fusedNode->name() + "/gamma");
        // Just put a single value to recognize this node as Const.
        gamma->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("value", epsilon));
        fusedNode->set_input(1, gamma->name());
    }
};

// tf.contrib.layers.flatten
class FlattenSubgraph : public Subgraph
{
public:
    FlattenSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Const");
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int shape_pack = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", strided_slice, shape_pack);
        addNodeToMatch("Reshape", input, pack);

        setFusedNode("Flatten", input);
    }
};

// tf.contrib.layers.flatten in case of unknown batch size
class FlattenShapeSubgraph : public Subgraph
{
public:
    FlattenShapeSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int shape_pack = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", strided_slice, shape_pack);
        addNodeToMatch("Reshape", input, pack);

        setFusedNode("Flatten", input);
    }
};

class FlattenProdSubgraph : public Subgraph
{
public:
    FlattenProdSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int prod = addNodeToMatch("Prod", strided_slice, addNodeToMatch("Const"));
        int shape_pack = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", shape_pack, prod);
        addNodeToMatch("Reshape", input, pack);

        setFusedNode("Flatten", input);
    }
};

// K.layers.Softmax
class SoftMaxKerasSubgraph : public Subgraph
{
public:
    SoftMaxKerasSubgraph()
    {
        int input = addNodeToMatch("");
        int maxReductionIndices = addNodeToMatch("Const");
        int smMax = addNodeToMatch("Max", input, maxReductionIndices);
        int smSub = addNodeToMatch("Sub", input, smMax);
        int smExp = addNodeToMatch("Exp", smSub);
        int sumReductionIndices = addNodeToMatch("Const");
        int smSum = addNodeToMatch("Sum", smExp, sumReductionIndices);
        addNodeToMatch("RealDiv", smExp, smSum);

        setFusedNode("Softmax", input);
    }
};

class ReLU6KerasSubgraph : public Subgraph
{
public:
    ReLU6KerasSubgraph()
    {
        int input = addNodeToMatch("");
        int relu = addNodeToMatch("Relu", input);
        int maxValue = addNodeToMatch("Const");
        int clipValue = addNodeToMatch("Const");
        int minimum = addNodeToMatch("Minimum", relu, maxValue);
        addNodeToMatch("Maximum", minimum, clipValue);

        setFusedNode("Relu6", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (!Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
            return false;
        tensorflow::NodeDef* node = net->getNode(matchedNodesIds.front() + 1).dynamicCast<TFNodeWrapper>()->node;
        Mat maxValue = getTensorContent(node->attr().at("value").tensor());
        return maxValue.type() == CV_32FC1 && maxValue.total() == 1 && maxValue.at<float>(0) == 6;
    }
};

// Keras' reshape stores output shape in separate Const nodes by one value.
// Need to merge them into a single Const node.
class ReshapeKerasSubgraph : public TFSubgraph
{
public:
    ReshapeKerasSubgraph(int _numOutDims) : numOutDims(_numOutDims)
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        std::vector<int> ids(1 + numOutDims);
        ids[0] = strided_slice;
        for (int i = 0; i < numOutDims; ++i)
            ids[1 + i] = addNodeToMatch("Const");
        int pack = addNodeToMatch("Pack", ids);
        addNodeToMatch("Reshape", input, pack);

        ids[0] = input;
        setFusedNode("Reshape", ids);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        Ptr<ImportNodeWrapper> node = net->getNode(nodeId);
        if (node->getNumInputs() == 0)
            return false;

        inpName = node->getInputName(0);
        return Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds);
    }


    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        std::vector<int> shape(numOutDims + 1);  // batch size in Keras is implicit.
        shape[0] = -1;
        for (int i = 0; i < numOutDims; ++i)
        {
            shape[1 + i] = inputNodes[1 + i]->attr().at("value").tensor().int_val(0);
        }
        tensorflow::TensorProto* shapeTensor = inputNodes[1]->mutable_attr()->at("value").mutable_tensor();
        fusedNode->mutable_input()->DeleteSubrange(2, numOutDims - 1);
        fusedNode->set_input(0, inpName);

        shapeTensor->clear_int_val();
        for (int i = 0; i < shape.size(); ++i)
        {
            shapeTensor->add_int_val(shape[i]);
        }
    }

private:
    int numOutDims;
    std::string inpName;
};

class L2NormalizeSubgraph : public Subgraph
{
public:
    L2NormalizeSubgraph()
    {
        int input = addNodeToMatch("");
        int square = addNodeToMatch("Square", input);
        int reductionIndices = addNodeToMatch("Const");
        int sum = addNodeToMatch("Sum", square, reductionIndices);
        int y = addNodeToMatch("Const");
        int maximum = addNodeToMatch("Maximum", sum, y);
        int rsqrt = addNodeToMatch("Rsqrt", maximum);
        addNodeToMatch("Mul", input, rsqrt);
        setFusedNode("L2Normalize", input, reductionIndices);
    }
};

class DeconvolutionValidKerasSubgraph : public TFSubgraph
{
public:
    DeconvolutionValidKerasSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int kernel = addNodeToMatch("Const");

        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        stack = addNodeToMatch("Const");
        stack_1 = addNodeToMatch("Const");
        stack_2 = addNodeToMatch("Const");
        int strided_slice_1 = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        stack = addNodeToMatch("Const");
        stack_1 = addNodeToMatch("Const");
        stack_2 = addNodeToMatch("Const");
        int strided_slice_2 = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        int mul = addNodeToMatch("Mul", strided_slice_1, addNodeToMatch("Const"));
        int add = addNodeToMatch("Add", mul, addNodeToMatch("Const"));

        int mul_1 = addNodeToMatch("Mul", strided_slice_2, addNodeToMatch("Const"));
        int add_1 = addNodeToMatch("Add", mul_1, addNodeToMatch("Const"));
        int pack = addNodeToMatch("Pack", strided_slice, add, add_1, addNodeToMatch("Const"));
        addNodeToMatch("Conv2DBackpropInput", pack, kernel, input);
        // Put any unused Const op to the first input.
        setFusedNode("Conv2DBackpropInput", stack, kernel, input);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        // Disable adjusted paddings (see Conv2DBackpropInput layer at tf_importer.cpp)
        // adj_w = (outW - (pad == "SAME") ? 1 : kernelW) % strideX;
        // adj_h = (outH - (pad == "SAME") ? 1 : kernelH) % strideY;
        // Where outH and outW are 1st and 2nd dimensions (NHWC) or 2nd and third (NCHW).
        std::string padMode = fusedNode->attr().at("padding").s();
        CV_Assert(padMode == "VALID");

        const tensorflow::TensorShapeProto& kernelShape =
            inputNodes[1]->mutable_attr()->at("value").tensor().tensor_shape();

        CV_Assert(kernelShape.dim_size() == 4);
        const int kernelHeight = kernelShape.dim(0).size();
        const int kernelWidth = kernelShape.dim(1).size();

        tensorflow::TensorProto* outShape = inputNodes[0]->mutable_attr()->at("value").mutable_tensor();
        outShape->clear_int_val();
        outShape->add_int_val(-1);
        outShape->add_int_val(kernelHeight);
        outShape->add_int_val(kernelWidth);
        outShape->add_int_val(-1);
    }
};

class DeconvolutionSameKerasSubgraph : public TFSubgraph
{
public:
    DeconvolutionSameKerasSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int kernel = addNodeToMatch("Const");

        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        stack = addNodeToMatch("Const");
        stack_1 = addNodeToMatch("Const");
        stack_2 = addNodeToMatch("Const");
        int strided_slice_1 = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        stack = addNodeToMatch("Const");
        stack_1 = addNodeToMatch("Const");
        stack_2 = addNodeToMatch("Const");
        int strided_slice_2 = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);

        int mul = addNodeToMatch("Mul", strided_slice_1, addNodeToMatch("Const"));

        int mul_1 = addNodeToMatch("Mul", strided_slice_2, addNodeToMatch("Const"));
        int pack = addNodeToMatch("Pack", strided_slice, mul, mul_1, addNodeToMatch("Const"));
        addNodeToMatch("Conv2DBackpropInput", pack, kernel, input);
        // Put any unused Const op to the first input.
        setFusedNode("Conv2DBackpropInput", stack, kernel, input);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        // Disable adjusted paddings (see Conv2DBackpropInput layer at tf_importer.cpp)
        // adj_w = (outW - (pad == "SAME") ? 1 : kernelW) % strideX;
        // adj_h = (outH - (pad == "SAME") ? 1 : kernelH) % strideY;
        // Where outH and outW are 1st and 2nd dimensions (NHWC) or 2nd and third (NCHW).
        std::string padMode = fusedNode->attr().at("padding").s();
        CV_Assert(padMode == "SAME");

        const tensorflow::AttrValue_ListValue& strides = fusedNode->attr().at("strides").list();
        CV_Assert(strides.i_size() == 4);

        const int strideY = strides.i(1);
        const int strideX = strides.i(2);

        tensorflow::TensorProto* outShape = inputNodes[0]->mutable_attr()->at("value").mutable_tensor();
        outShape->clear_int_val();
        outShape->add_int_val(-1);
        outShape->add_int_val(strideY);
        outShape->add_int_val(strideX);
        outShape->add_int_val(-1);
    }
};

// In case of resizing by factor.
class ResizeBilinearSubgraph : public Subgraph
{
public:
    ResizeBilinearSubgraph()
    {
        int input = addNodeToMatch("");
        int shapeSource = addNodeToMatch("");

        int shape = addNodeToMatch("Shape", shapeSource);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int factorY = addNodeToMatch("Const");
        int mul = addNodeToMatch("Mul", strided_slice, factorY);

        shape = addNodeToMatch("Shape", shapeSource);
        stack = addNodeToMatch("Const");
        stack_1 = addNodeToMatch("Const");
        stack_2 = addNodeToMatch("Const");
        strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int factorX = addNodeToMatch("Const");
        int mul_1 = addNodeToMatch("Mul", strided_slice, factorX);

        int pack = addNodeToMatch("Pack", mul, mul_1);

        addNodeToMatch("ResizeBilinear", input, pack);
        setFusedNode("ResizeBilinear", input, factorY, factorX);
    }
};

// In case of resizing by factor.
class ResizeBilinearSubgraphDown : public TFSubgraph
{
public:
    ResizeBilinearSubgraphDown()
    {
        int input = addNodeToMatch("");
        int shapeSource = addNodeToMatch("");

        int shape = addNodeToMatch("Shape", shapeSource);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int factorY = addNodeToMatch("Const");
        int div = addNodeToMatch("RealDiv", addNodeToMatch("Cast", strided_slice), factorY);
        int cast = addNodeToMatch("Cast", div);

        shape = addNodeToMatch("Shape", shapeSource);
        stack = addNodeToMatch("Const");
        stack_1 = addNodeToMatch("Const");
        stack_2 = addNodeToMatch("Const");
        strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int factorX = addNodeToMatch("Const");
        int div_1 = addNodeToMatch("RealDiv", addNodeToMatch("Cast", strided_slice), factorX);
        int cast_1 = addNodeToMatch("Cast", div_1);

        int pack = addNodeToMatch("Pack", cast, cast_1);

        addNodeToMatch("ResizeBilinear", input, pack);
        setFusedNode("ResizeBilinear", input, factorY, factorX);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {

        for (int i = 1; i < 3; ++i)
        {
            tensorflow::TensorProto* factor = inputNodes[i]->mutable_attr()->at("value").mutable_tensor();
            factor->set_double_val(0, 1.0 / factor->double_val(0));
        }
    }
};

// In case of resizing by factor.
class UpsamplingKerasSubgraph : public TFSubgraph
{
public:
    UpsamplingKerasSubgraph(const std::string& type)
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int stack = addNodeToMatch("Const");
        int stack_1 = addNodeToMatch("Const");
        int stack_2 = addNodeToMatch("Const");
        int strided_slice = addNodeToMatch("StridedSlice", shape, stack, stack_1, stack_2);
        int factors = addNodeToMatch("Const");
        int mul = addNodeToMatch("Mul", strided_slice, factors);
        addNodeToMatch(type, input, mul);
        setFusedNode(type, input, factors);
    }

    virtual void finalize(tensorflow::GraphDef& net, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        Mat factorsMat = getTensorContent(inputNodes[1]->attr().at("value").tensor());
        CV_CheckEQ(factorsMat.total(), (size_t)2, ""); CV_CheckTypeEQ(factorsMat.type(), CV_32SC1, "");

        // Height scale factor
        tensorflow::TensorProto* factorY = inputNodes[1]->mutable_attr()->at("value").mutable_tensor();
        factorY->clear_int_val();
        factorY->clear_tensor_content();
        factorY->add_int_val(factorsMat.at<int>(0, 0));

        // Width scale factor.
        tensorflow::NodeDef* factorXNode = net.add_node();
        factorXNode->set_op("Const");
        factorXNode->set_name(fusedNode->name() + "/factor_y");

        tensorflow::AttrValue factorX;
        factorX.mutable_tensor()->set_dtype(tensorflow::DT_INT32);
        factorX.mutable_tensor()->add_int_val(factorsMat.at<int>(0, 1));
        factorXNode->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("value", factorX));

        fusedNode->add_input(factorXNode->name());
    }
};

class ReshapeAsShapeSubgraph : public Subgraph
{
public:
    ReshapeAsShapeSubgraph()
    {
        int input = addNodeToMatch("");
        int shapeSrc = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", shapeSrc);
        addNodeToMatch("Reshape", input, shape);
        setFusedNode("Reshape", input, shapeSrc);
    }
};

class SoftMaxSlimSubgraph : public Subgraph
{
public:
    SoftMaxSlimSubgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Const");
        int shapeOp = addNodeToMatch("Shape", input);
        int reshape = addNodeToMatch("Reshape", input, shape);
        int softmax = addNodeToMatch("Softmax", reshape);
        addNodeToMatch("Reshape", softmax, shapeOp);
        setFusedNode("Softmax", input);
    }
};

class SoftMaxSlimV2Subgraph : public Subgraph
{
public:
    SoftMaxSlimV2Subgraph()
    {
        int input = addNodeToMatch("");
        int shape = addNodeToMatch("Shape", input);
        int shape_2 = addNodeToMatch("Shape", input);
        int rank = addNodeToMatch("Const");
        int y = addNodeToMatch("Const");
        int sub = addNodeToMatch("Sub", rank, y);
        int begin = addNodeToMatch("Pack", sub);
        int size = addNodeToMatch("Const");
        int slice = addNodeToMatch("Slice", shape, begin, size);
        int values = addNodeToMatch("Const");
        int axis = addNodeToMatch("Const");
        int concat = addNodeToMatch("ConcatV2", values, slice, axis);
        int reshape = addNodeToMatch("Reshape", input, concat);
        int softmax = addNodeToMatch("Softmax", reshape);
        addNodeToMatch("Reshape", softmax, shape_2);
        setFusedNode("Softmax", input);
    }
};

class KerasMVNSubgraph : public TFSubgraph
{
public:
    KerasMVNSubgraph()
    {
        int input = addNodeToMatch("");
        int mean = addNodeToMatch("Mean", input, addNodeToMatch("Const"));
        int grad = addNodeToMatch("StopGradient", mean);
        int diff = addNodeToMatch("SquaredDifference", input, grad);
        int var = addNodeToMatch("Mean", diff, addNodeToMatch("Const"));
        int sub = addNodeToMatch("Sub", input, mean);
        int add_y = addNodeToMatch("Const");
        int add = addNodeToMatch("Add", var, add_y);
        int pow_y = addNodeToMatch("Const");
        int powNode = addNodeToMatch("Pow", add, pow_y);
        addNodeToMatch("RealDiv", sub, powNode);
        setFusedNode("MVN", input, add_y);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        tensorflow::AttrValue eps;

        Mat epsMat = getTensorContent(inputNodes[1]->attr().at("value").tensor());
        CV_CheckEQ(epsMat.total(), (size_t)1, "");
        CV_CheckTypeEQ(epsMat.type(), CV_32FC1, "");
        eps.set_f(epsMat.at<float>(0));
        fusedNode->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("eps", eps));

        fusedNode->mutable_input()->RemoveLast();
    }
};

class PReLUSubgraph : public TFSubgraph
{
public:
    PReLUSubgraph(bool negativeScales_) : negativeScales(negativeScales_)
    {
        int input = addNodeToMatch("");
        int scales = addNodeToMatch("Const");
        int neg = addNodeToMatch("Neg", input);
        int relu_neg = addNodeToMatch("Relu", neg);
        int finalScales = negativeScales ? addNodeToMatch("Neg", scales) : scales;
        int mul = addNodeToMatch("Mul", finalScales, relu_neg);
        int relu_pos = addNodeToMatch("Relu", input);
        addNodeToMatch("Add", relu_pos, mul);
        setFusedNode("PReLU", input, scales);
    }

    virtual void finalize(tensorflow::GraphDef&, tensorflow::NodeDef* fusedNode,
                          std::vector<tensorflow::NodeDef*>& inputNodes) CV_OVERRIDE
    {
        if (!negativeScales)
        {
            Mat scalesRef = getTensorContentRef_(inputNodes[1]->attr().at("value").tensor());
            // FIXME: This breaks the const guarantees of tensor() by writing to scalesRef
            if (isAlignedMat(scalesRef))
            {
                scalesRef *= -1;
            }
            else
            {
                Mat scales = scalesRef.clone() * -1;
                CV_Assert(scalesRef.isContinuous());
                CV_Assert(scales.isContinuous());
                memcpy(scalesRef.data, scales.data, scales.total() * scales.elemSize());
            }
        }
    }

private:
    bool negativeScales;
};

class ClipByValueSubgraph : public TFSubgraph
{
public:
    ClipByValueSubgraph()
    {
        int input = addNodeToMatch("");
        int maxValue = addNodeToMatch("Const");
        int minimum = addNodeToMatch("Minimum", input, maxValue);
        int minValue = addNodeToMatch("Const");
        addNodeToMatch("Maximum", minimum, minValue);

        setFusedNode("ClipByValue", input, minValue, maxValue);
    }
};

void simplifySubgraphs(tensorflow::GraphDef& net)
{
    std::vector<Ptr<Subgraph> > subgraphs;
    subgraphs.push_back(Ptr<Subgraph>(new BatchNormSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new BatchNormNoGammaSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new FlattenSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new FlattenShapeSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new SoftMaxKerasSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ReLU6KerasSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ReshapeKerasSubgraph(3)));
    subgraphs.push_back(Ptr<Subgraph>(new L2NormalizeSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new DeconvolutionValidKerasSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new DeconvolutionSameKerasSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ResizeBilinearSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new UpsamplingKerasSubgraph("ResizeNearestNeighbor")));
    subgraphs.push_back(Ptr<Subgraph>(new UpsamplingKerasSubgraph("ResizeBilinear")));
    subgraphs.push_back(Ptr<Subgraph>(new SoftMaxSlimSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new SoftMaxSlimV2Subgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ReshapeAsShapeSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new KerasMVNSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new PReLUSubgraph(true)));
    subgraphs.push_back(Ptr<Subgraph>(new PReLUSubgraph(false)));
    subgraphs.push_back(Ptr<Subgraph>(new FlattenProdSubgraph()));
    subgraphs.push_back(Ptr<Subgraph>(new ResizeBilinearSubgraphDown()));
    subgraphs.push_back(Ptr<Subgraph>(new ClipByValueSubgraph()));

    for (int i = 0; i < net.node_size(); ++i)
    {
        tensorflow::NodeDef* layer = net.mutable_node(i);
        if (layer->op() == "AddV2")
            layer->set_op("Add");
    }

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new TFGraphWrapper(net)), subgraphs);
}

void RemoveIdentityOps(tensorflow::GraphDef& net)
{
    typedef std::map<String, String>  IdentityOpsMap;
    IdentityOpsMap identity_ops;

    std::vector<int> identity_ops_idx;

    int layersCount = net.node_size();
    for (int li = 0; li < layersCount; li++)
    {
        const tensorflow::NodeDef &layer = net.node(li);
        String type = layer.op();

        if (type == "Identity" || type == "Dropout" || type == "PlaceholderWithDefault") {
            identity_ops_idx.push_back(li);
            identity_ops[layer.name()] = layer.input(0);
        }
    }

    for (int li = 0; li < layersCount; li++)
    {
        tensorflow::NodeDef* layer = net.mutable_node(li);
        for (int input_id = 0; input_id < layer->input_size(); input_id++) {
            String input_op_name = layer->input(input_id);
            input_op_name = input_op_name.substr(input_op_name.find('^') + 1,
                                                 input_op_name.rfind(':'));
            IdentityOpsMap::iterator it = identity_ops.find(input_op_name);

            if (it != identity_ops.end()) {
                // In case of Identity after Identity
                while (true)
                {
                    IdentityOpsMap::iterator nextIt = identity_ops.find(it->second);
                    if (nextIt != identity_ops.end())
                        it = nextIt;
                    else
                        break;
                }
                layer->set_input(input_id, it->second);
            }
        }
    }

    std::sort(identity_ops_idx.begin(), identity_ops_idx.end());

    int removed_nodes = 0;
    for(size_t i = 0; i < identity_ops_idx.size(); i++) {
        int start_id = identity_ops_idx[i] - removed_nodes;
        net.mutable_node()->DeleteSubrange(start_id, 1);
        removed_nodes++;
    }
}

// NB: returned Mat::data pointer may be unaligned
Mat getTensorContentRef_(const tensorflow::TensorProto& tensor)
{
    const std::string& content = tensor.tensor_content();
    Mat m;
    switch (tensor.dtype())
    {
        case tensorflow::DT_FLOAT:
        {
            if (!content.empty())
                m = Mat(1, content.size() / sizeof(float), CV_32FC1, (void*)content.c_str());
            else
            {
                const RepeatedField<float>& field = tensor.float_val();
                CV_Assert(!field.empty());
                m = Mat(1, field.size(), CV_32FC1, (void*)field.data());
            }
            break;
        }
        case tensorflow::DT_DOUBLE:
        {
            if (!content.empty())
                m = Mat(1, content.size() / sizeof(double), CV_64FC1, (void*)content.c_str());
            else
            {
                const RepeatedField<double>& field = tensor.double_val();
                CV_Assert(!field.empty());
                m = Mat(1, field.size(), CV_64FC1, (void*)field.data());
            }
            break;
        }
        case tensorflow::DT_INT32:
        {
            if (!content.empty())
                m = Mat(1, content.size() / sizeof(int32_t), CV_32SC1, (void*)content.c_str());
            else
            {
                const RepeatedField<int32_t>& field = tensor.int_val();
                CV_Assert(!field.empty());
                m = Mat(1, field.size(), CV_32SC1, (void*)field.data());
            }
            break;
        }
        case tensorflow::DT_HALF:
        {
            Mat halfs;
            if (!content.empty())
            {
                static const int kHalfSize = 2;
                halfs = Mat(1, content.size() / kHalfSize, CV_16UC1, (void*)content.c_str());
            }
            else
            {
                const RepeatedField<int32_t>& field = tensor.half_val();
                CV_Assert(!field.empty());
                Mat ints(1, field.size(), CV_32SC1, (void*)field.data());
                ints.convertTo(halfs, CV_16UC1);
            }
            // Reinterpret as a signed shorts just for a convertFp16 call.
            Mat halfsSigned(halfs.size(), CV_16SC1, halfs.data);
            convertFp16(halfsSigned, m);
            break;
        }
        case tensorflow::DT_QUINT8:
        {
            CV_Assert(!content.empty());
            m = Mat(1, content.size(), CV_8UC1, (void*)content.c_str());
            break;
        }
        default:
            CV_Error(Error::StsError, "Tensor's data type is not supported");
            break;
    }

    return m;
}

Mat getTensorContent(const tensorflow::TensorProto& tensor, bool forceCopy)
{
    // If necessary clone m to have aligned data pointer
    Mat m = getTensorContentRef_(tensor);
    if (forceCopy || !isAlignedMat(m))
        return m.clone();
    else
        return m;
}

void releaseTensor(tensorflow::TensorProto* tensor)
{
    if (!tensor->mutable_tensor_content()->empty())
    {
        delete tensor->release_tensor_content();
    }
}

static void permute(google::protobuf::RepeatedPtrField<tensorflow::NodeDef>* data,
                    const std::vector<int>& indices)
{
    const int num = data->size();
    CV_Assert(num == indices.size());

    std::vector<int> elemIdToPos(num);
    std::vector<int> posToElemId(num);
    for (int i = 0; i < num; ++i)
    {
        elemIdToPos[i] = i;
        posToElemId[i] = i;
    }
    for (int i = 0; i < num; ++i)
    {
        int elemId = indices[i];
        int pos = elemIdToPos[elemId];
        if (pos != i)
        {
            data->SwapElements(i, pos);
            const int swappedElemId = posToElemId[i];
            elemIdToPos[elemId] = i;
            elemIdToPos[swappedElemId] = pos;

            posToElemId[i] = elemId;
            posToElemId[pos] = swappedElemId;
        }
    }
}

// Is based on tensorflow::graph_transforms::SortByExecutionOrder
void sortByExecutionOrder(tensorflow::GraphDef& net)
{
    // Maps node's name to index at net.node() list.
    std::map<std::string, int> nodesMap;
    std::map<std::string, int>::iterator nodesMapIt;
    for (int i = 0; i < net.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = net.node(i);
        nodesMap.insert(std::make_pair(node.name(), i));
    }

    // Indices of nodes which use specific node as input.
    std::vector<std::vector<int> > edges(nodesMap.size());
    std::vector<int> numRefsToAdd(nodesMap.size(), 0);
    std::vector<int> nodesToAdd;
    for (int i = 0; i < net.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = net.node(i);
        int numInputsInGraph = 0;
        for (int j = 0; j < node.input_size(); ++j)
        {
            std::string inpName = node.input(j);
            inpName = inpName.substr(0, inpName.rfind(':'));
            inpName = inpName.substr(inpName.find('^') + 1);

            nodesMapIt = nodesMap.find(inpName);
            if (nodesMapIt != nodesMap.end())
            {
                edges[nodesMapIt->second].push_back(i);
                numInputsInGraph += 1;
            }
        }
        if (numInputsInGraph == 0)
            nodesToAdd.push_back(i);
        else
        {
            if (node.op() == "Merge" || node.op() == "RefMerge" || node.op() == "NoOp")
            {
                int numControlEdges = 0;
                for (int j = 0; j < numInputsInGraph; ++j)
                    numControlEdges += node.input(j)[0] == '^';
                numRefsToAdd[i] = numControlEdges + 1;
            }
            else
                numRefsToAdd[i] = numInputsInGraph;
        }
    }

    std::vector<int> permIds;
    permIds.reserve(net.node_size());
    while (!nodesToAdd.empty())
    {
        int nodeToAdd = nodesToAdd.back();
        nodesToAdd.pop_back();

        permIds.push_back(nodeToAdd);

        for (int i = 0; i < edges[nodeToAdd].size(); ++i)
        {
            int consumerId = edges[nodeToAdd][i];
            if (numRefsToAdd[consumerId] > 0)
            {
                if (numRefsToAdd[consumerId] == 1)
                    nodesToAdd.push_back(consumerId);
                else
                    CV_Assert(numRefsToAdd[consumerId] >= 0);
                numRefsToAdd[consumerId] -= 1;
            }
        }
    }
    CV_Assert(permIds.size() == net.node_size());
    permute(net.mutable_node(), permIds);
}

// Remove training switches (Switch and Merge nodes and corresponding subgraphs).
void removePhaseSwitches(tensorflow::GraphDef& net)
{
    std::vector<int> nodesToRemove;
    std::map<std::string, int> nodesMap;
    std::map<std::string, int>::iterator nodesMapIt;
    std::queue<int> mergeOpSubgraphNodes;
    for (int i = 0; i < net.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = net.node(i);
        nodesMap.insert(std::make_pair(node.name(), i));
        if (node.op() == "Switch" || node.op() == "Merge" || node.op() == "NoOp")
        {
            CV_Assert(node.input_size() > 0);
            // Replace consumers' inputs.
            for (int j = 0; j < net.node_size(); ++j)
            {
                tensorflow::NodeDef* consumer = net.mutable_node(j);
                for (int k = 0; k < consumer->input_size(); ++k)
                {
                    std::string inpName = consumer->input(k);
                    inpName = inpName.substr(0, inpName.rfind(':'));
                    if (inpName == node.name())
                    {
                        consumer->set_input(k, node.input(0));
                    }
                }
            }
            nodesToRemove.push_back(i);
            if (node.op() == "Merge" || node.op() == "Switch" || node.op() == "NoOp")
                mergeOpSubgraphNodes.push(i);
        }
    }

    std::vector<int> numConsumers(net.node_size(), 0);
    for (int i = 0; i < net.node_size(); ++i)
    {
        const tensorflow::NodeDef& node = net.node(i);
        for (int j = 0; j < node.input_size(); ++j)
        {
            std::string inpName = node.input(j);
            inpName = inpName.substr(1 + (int)inpName.find('^'), inpName.rfind(':'));
            nodesMapIt = nodesMap.find(inpName);
            CV_Assert(nodesMapIt != nodesMap.end());
            numConsumers[nodesMapIt->second] += 1;
        }
    }

    // Remove subgraphs of unused nodes which are terminated by Merge nodes.
    while (!mergeOpSubgraphNodes.empty())
    {
        const tensorflow::NodeDef& node = net.node(mergeOpSubgraphNodes.front());
        mergeOpSubgraphNodes.pop();
        for (int i = 0; i < node.input_size(); ++i)
        {
            std::string inpName = node.input(i);
            inpName = inpName.substr(1 + (int)inpName.find('^'), inpName.rfind(':'));
            nodesMapIt = nodesMap.find(inpName);
            CV_Assert(nodesMapIt != nodesMap.end());

            int inpNodeId = nodesMapIt->second;
            if (numConsumers[inpNodeId] == 1)
            {
                mergeOpSubgraphNodes.push(inpNodeId);
                nodesToRemove.push_back(inpNodeId);
            }
            else if (numConsumers[inpNodeId] > 0)
                numConsumers[inpNodeId] -= 1;
        }
    }
    std::sort(nodesToRemove.begin(), nodesToRemove.end());
    for (int i = nodesToRemove.size() - 1; i >= 0; --i)
    {
        if (nodesToRemove[i] < net.node_size())  // Ids might be repeated.
            net.mutable_node()->DeleteSubrange(nodesToRemove[i], 1);
    }
}


CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace dnn, namespace cv

#endif  // HAVE_PROTOBUF
