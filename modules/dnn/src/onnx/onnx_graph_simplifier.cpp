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
#include <limits>

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
        // Add a fake initializer with empty name.
        // Some ONNX models skip their inputs. For example,
        // Resize which has 4 inputs but 2 of them have empty names.
        // So we add a fake empty node to which such ops may refer as input.
        net.add_initializer();

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

    int getTensorShapeSize(int node_id, int node_input_id) {
        const auto node = getNode(node_id);
        const auto &input_name = node->getInputName(node_input_id);
        // try to get from value_info
        for (int i = 0; i < net.value_info_size(); i++) {
            const auto value_info = net.value_info(i);
            if (value_info.name() == input_name) {
                if (value_info.has_type() && value_info.type().has_tensor_type() &&
                    value_info.type().tensor_type().has_shape()) {
                    return value_info.type().tensor_type().shape().dim_size();
                } else {
                    return -1;
                }
            }
        }
        // try to get from input
        for (int i = 0; i < net.input_size(); i++) {
            const auto input = net.input(i);
            if (input.name() == input_name) {
                if (input.has_type() && input.type().has_tensor_type() &&
                    input.type().tensor_type().has_shape()) {
                    return input.type().tensor_type().shape().dim_size();
                } else {
                    return -1;
                }
            }
        }
        return -1;
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
        if (idx >= numInputs + numInitializers)
            net.mutable_node()->DeleteSubrange(idx - numInputs - numInitializers, 1);
    }

    virtual inline bool isCommutativeOp(const std::string& type) const CV_OVERRIDE
    {
        return type == "Add" || type == "Mul" || type == "Equal" || type == "Max";
    }

private:
    int numInputs, numInitializers;
    opencv_onnx::GraphProto& net;
};

static Mat extractConstant(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id)
{
    auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
    int initializer_id = onnx_net->getInputInitializerId(node_id, input_id);
    if (initializer_id != -1)
    {
        return onnx_net->getMatFromInitializer(initializer_id);
    }
    else
    {
        const Ptr<ImportNodeWrapper> node = net->getNode(node_id);
        int constant_id = Subgraph::getInputNodeId(net, node, input_id);
        Ptr<ImportNodeWrapper> constant_ptr = net->getNode(constant_id);
        opencv_onnx::NodeProto* constant_node = constant_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::TensorProto constant_proto = constant_node->attribute(0).t();
        return getMatFromTensor(constant_proto);
    }
}

static std::string getInputName(const Ptr<ImportGraphWrapper>& net, int node_id, int input_id) {
    auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();
    int initializer_id = onnx_net->getInputInitializerId(node_id, input_id);
    if (initializer_id != -1) {
        return onnx_net->getNameOfInitializer(initializer_id);
    } else {
        const auto node = net->getNode(node_id);
        return node->getInputName(input_id);
    }
}

/*  Slice operator has two optional inputs "axes" and "steps". Some models may be set to have
    Slice with optional inputs of default values, some of them don't. This Subgraph adjusts
    all optional inputs of Slice up to 5.
*/
class AdjustSliceAllOptionalInputsSubgraph : public Subgraph {
 public:
    AdjustSliceAllOptionalInputsSubgraph(size_t num_inputs = 4) {
        num_inputs_ = num_inputs;

        int input = addNodeToMatch("");
        int starts = addNodeToMatch("");
        int ends = addNodeToMatch("");
        std::vector<int> inputs{input, starts, ends};
        for (size_t i = 3; i < num_inputs_; i++) { // axes and steps
            inputs.push_back(addNodeToMatch(""));
        }

        slice_id = addNodeToMatch("Slice", inputs);

        setFusedNode("Slice", inputs);
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        for (int i = num_inputs_; i < 5; ++i) {
            node->add_input("");
        }
    }
 private:
    int slice_id;
    size_t num_inputs_;
};

/* Fusion for biased MatMul.

   Graph before fusion: [Input] -> MatMul -> Add -> [Output]

   Graph after fusion:  [Input] -> MatMul -> [Output]
                                     \
                                     bias
*/

class BiasedMatmulSubgraph : public Subgraph {
 public:
    BiasedMatmulSubgraph() {
        int input = addNodeToMatch("");
        matmul_id = addNodeToMatch("MatMul", input, addNodeToMatch(""));
        add_id = addNodeToMatch("Add", addNodeToMatch(""), matmul_id);

        setFusedNode("MatMul", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE {
        if (Subgraph::match(net, nodeId, matchedNodesIds)) {
            auto onnx_net = net.dynamicCast<ONNXGraphWrapper>();

            // get input weight from MatMul
            {
                // make sure that input A is not Constant
                if (onnx_net->getInputInitializerId(matchedNodesIds[matmul_id], 0) >= 0) {
                    return false;
                } else {
                    const Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds[matmul_id]);

                    int constant_id = Subgraph::getInputNodeId(net, node, 0);
                    auto constant_node = net->getNode(constant_id);
                    if (constant_node->getType() == "Constant") {
                        return false;
                    }
                }

                bool is_weight_const = false;
                int initializer_id = onnx_net->getInputInitializerId(matchedNodesIds[matmul_id], 1);
                if (initializer_id != -1) { // Initializer
                    weight_name = onnx_net->getNameOfInitializer(initializer_id);
                    is_weight_const = true;
                } else { // Constant layer
                    const Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds[matmul_id]);

                    int constant_id = Subgraph::getInputNodeId(net, node, 1);
                    auto constant_node = net->getNode(constant_id);
                    if (constant_node->getType() == "Constant") {
                        weight_name = node->getInputName(1);
                        is_weight_const = true;
                    }
                }

                if (!is_weight_const) {
                    return false;
                }
            }

            // get input bias from Add
            {
                bool is_bias_const = false;
                int initializer_id = std::max(onnx_net->getInputInitializerId(matchedNodesIds[add_id], 0),
                                              onnx_net->getInputInitializerId(matchedNodesIds[add_id], 1));
                if (initializer_id != -1) {
                    bias_name = onnx_net->getNameOfInitializer(initializer_id);
                    is_bias_const = true;
                } else { // Constant layer
                    const Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds[add_id]);

                    int constant_id = Subgraph::getInputNodeId(net, node, 0);
                    auto constant_node = net->getNode(constant_id);
                    if (constant_node->getType() == "Constant") {
                        bias_name = node->getInputName(0);
                        is_bias_const = true;
                    } else {
                        constant_id = Subgraph::getInputNodeId(net, node, 1);
                        constant_node = net->getNode(constant_id);
                        if (constant_node->getType() == "Constant") {
                            bias_name = node->getInputName(1);
                            is_bias_const = true;
                        }
                    }
                }
                if (!is_bias_const) {
                    return false;
                }
            }

            return true;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>& net,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        // add inputs
        node->add_input(weight_name);
        node->add_input(bias_name);
    }

 private:
    int matmul_id, add_id;
    std::string weight_name, bias_name;
};

/*  The fusion for the multi-head attention from vision transformer.

    Abbreviations:
        B - batch_size, symbolic;
        S - sequence_length, symbolic;
        W - hidden_size, W = N * H;
        N - num_heads;
        H - head_size;

    Graph before fusion:
                    [Input](BxSxW)
                      |
                   LayerNorm
                      |
                   Transpose(perm=[1, 0, 2])
                      |
                      | (SxBxW)
                      |
                    Matmul[Weight(Wx3W)]
                      |
                     Add[Bias(3W)]
          /           |           \
      q_Slice      k_Slice      v_Slice   (output(SxBxW))
         |            |            |
     q_Reshape    k_Reshape    v_Reshape  (output(Sx(BxN)xH), could be optional if N=1)
         |            |            |
    q_Transpose  k_Transpose  v_Transpose
      (1,0,2)      (1,2,0)    (perm=1,0,2)
         |((BxN)xSxH) |((BxN)xHxS) |
       q_Div         /            /
         \          /            /
          qk_MatMul             /
              |                /
         qk_Softmax           /
              | ((BxN)xSxS)  / ((BxN)xSxH)
               \            /
                 qkv_MatMul  (output((BxN)xSxH))
                     |
                 Transpose(perm=1,2,0)
                     |
                  Reshape  (output(SxH))
                     |
                   MatMul
                     |
                    Add
                     |
                  [Output](BxSxW)


    Attributes:
        num_heads - number of attention heads
        qkv_hidden_sizes - hidden size of qkv respectively, [qk_hidden_size, qk_hidden_size, v_hidden_size],
                          assume qk_hidden_size = v_hidden_size for now. TODO: support qk_hidden_size != v_hidden_size
        scale - scale factor of q, defaults to sqrt(1/num_heads)
    Inputs:
        weight - merged Q, K, V weights of shape [input_hidden_size, qk_hidden_size + qk_hidden_size + v_hidden_size]
        bias - bias of shape [qk_hidden_size + qk_hidden_size + v_hidden_size]

    Graph after fusion:
            [Input](BxSxW)
               |
            LayerNorm
               |
           Transpose
               |
           Attention[weight, bias]
               |
             MatMul
               |
              Add
               |
            [Output](BxSxW)

    More details see See https://github.com/microsoft/onnxruntime/blob/v1.16.1/docs/ContribOperators.md#com.microsoft.Attention.
*/
class AttentionSubGraph : public Subgraph {
 public:
    AttentionSubGraph() {
        int input = addNodeToMatch("");
        int transpose = addNodeToMatch("Transpose", input); // tranpose does not make any differences to the accuracy here in this subgraph
        att_matmul = addNodeToMatch("MatMul", transpose, addNodeToMatch(""), addNodeToMatch("")); // add is fused into matmul via BiasedMatMulSubgraph

        // v_path
        slice_v = addNodeToMatch("Slice", std::vector<int>{att_matmul, addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch("")});
        int reshape_v = addNodeToMatch("Reshape", slice_v, addNodeToMatch(""));
        int transpose_v = addNodeToMatch("Transpose", reshape_v);

        // q_path
        slice_q = addNodeToMatch("Slice", std::vector<int>{att_matmul, addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch("")});
        reshape_q = addNodeToMatch("Reshape", slice_q, addNodeToMatch(""));
        int transpose_q = addNodeToMatch("Transpose", reshape_q);
        div_q = addNodeToMatch("Div", transpose_q, addNodeToMatch(""));

        // k_path
        slice_k = addNodeToMatch("Slice", std::vector<int>{att_matmul, addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch("")});
        int reshape_k = addNodeToMatch("Reshape", slice_k, addNodeToMatch(""));
        int transpose_k = addNodeToMatch("Transpose", reshape_k);

        // qk
        int matmul_qk = addNodeToMatch("MatMul", div_q, transpose_k);
        int softmax_qk = addNodeToMatch("Softmax", matmul_qk);

        // qkv
        int matmul_qkv = addNodeToMatch("MatMul", softmax_qk, transpose_v);
        int transpose_qkv = addNodeToMatch("Transpose", matmul_qkv);
        last_reshape = addNodeToMatch("Reshape", transpose_qkv, addNodeToMatch(""));

        setFusedNode("Attention", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE {
        if (Subgraph::match(net, nodeId, matchedNodesIds)) {
            // get attrs - qkv_hidden_sizes
            qkv_hidden_sizes.clear();
            auto fill_qkv_hidden_sizes = [&] (const int slice_node_id) {
                int slice_start = extractConstant(net, matchedNodesIds[slice_node_id], 1).at<int64_t>(0);
                int slice_end = extractConstant(net, matchedNodesIds[slice_node_id], 2).at<int64_t>(0);
                if (slice_end == std::numeric_limits<int>::max()) {
                    qkv_hidden_sizes.push_back(0); // workaround for Slice with end=INT_MAX
                } else {
                    int64_t hidden_size = static_cast<int64_t>(slice_end - slice_start);
                    qkv_hidden_sizes.push_back(hidden_size);
                }
            };
            fill_qkv_hidden_sizes(slice_q);
            fill_qkv_hidden_sizes(slice_k);
            fill_qkv_hidden_sizes(slice_v); // TODO: take care of INT64_MAX
            CV_CheckEQ(qkv_hidden_sizes.size(), static_cast<size_t>(3), "ONNXSimplifier/Attention: invalid qkv hidden sizes");
            CV_CheckEQ(int(qkv_hidden_sizes[0]), int(qkv_hidden_sizes[1]), "ONNXSimplifier/Attention: invalid qkv hidden sizes, q_hidden_size == v_hidden_size is required");
            // get attrs - num_heads, scale
            num_heads = extractConstant(net, matchedNodesIds[reshape_q], 1).at<int64_t>(1);
            scale = extractConstant(net, matchedNodesIds[div_q], 1).at<float>(0);
            output_ndims = extractConstant(net, matchedNodesIds[last_reshape], 1).size[0];

            // get names
            weight_name = getInputName(net, matchedNodesIds[att_matmul], 1);
            bias_name = getInputName(net, matchedNodesIds[att_matmul], 2);
            return true;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>& net,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE {
        // add attrs
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attr_num_heads = node->add_attribute();
        attr_num_heads->set_name("num_heads");
        attr_num_heads->set_i(num_heads);
        opencv_onnx::AttributeProto* attr_qkv_hidden_sizes = node->add_attribute();
        attr_qkv_hidden_sizes->set_name("qkv_hidden_sizes");
        attr_qkv_hidden_sizes->add_ints(qkv_hidden_sizes[0]);
        attr_qkv_hidden_sizes->add_ints(qkv_hidden_sizes[1]);
        attr_qkv_hidden_sizes->add_ints(qkv_hidden_sizes[2]);
        opencv_onnx::AttributeProto* attr_scale = node->add_attribute();
        attr_scale->set_name("scale");
        attr_scale->set_f(scale);

        // add customized attrs
        opencv_onnx::AttributeProto* attr_output_ndims = node->add_attribute();
        attr_output_ndims->set_name("output_ndims");
        attr_output_ndims->set_i(output_ndims);

        // add inputs
        node->add_input(weight_name);
        node->add_input(bias_name);
    }

 private:
    int att_matmul;
    int slice_q, slice_k, slice_v;
    int reshape_q, div_q, last_reshape;

    std::vector<int64_t> qkv_hidden_sizes; // order: [qk_hidden_size, qk_hidden_size, v_hidden_size]
    int64_t num_heads;
    float scale;

    int64_t output_ndims;

    std::string weight_name;
    std::string bias_name;
};

/*  Attention subgraph with single head.
    No Reshape operator is appended after each Slice operator.
*/
class AttentionSingleHeadSubGraph : public Subgraph {
 public:
    AttentionSingleHeadSubGraph() {
        int input = addNodeToMatch("");
        int transpose = addNodeToMatch("Transpose", input); // tranpose does not make any differences to the accuracy here in this subgraph
        att_matmul = addNodeToMatch("MatMul", transpose, addNodeToMatch(""), addNodeToMatch("")); // add is fused into matmul via BiasedMatMulSubgraph

        // v_path
        slice_v = addNodeToMatch("Slice", std::vector<int>{att_matmul, addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch("")});
        int transpose_v = addNodeToMatch("Transpose", slice_v);

        // q_path
        slice_q = addNodeToMatch("Slice", std::vector<int>{att_matmul, addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch("")});
        int transpose_q = addNodeToMatch("Transpose", slice_q);
        div_q = addNodeToMatch("Div", transpose_q, addNodeToMatch(""));

        // k_path
        slice_k = addNodeToMatch("Slice", std::vector<int>{att_matmul, addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch(""), addNodeToMatch("")});
        int transpose_k = addNodeToMatch("Transpose", slice_k);

        // qk
        int matmul_qk = addNodeToMatch("MatMul", div_q, transpose_k);
        int softmax_qk = addNodeToMatch("Softmax", matmul_qk);

        // qkv
        int matmul_qkv = addNodeToMatch("MatMul", softmax_qk, transpose_v);
        int transpose_qkv = addNodeToMatch("Transpose", matmul_qkv);
        last_reshape = addNodeToMatch("Reshape", transpose_qkv, addNodeToMatch(""));

        setFusedNode("Attention", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE {
        if (Subgraph::match(net, nodeId, matchedNodesIds)) {
            // get attrs - qkv_hidden_sizes
            qkv_hidden_sizes.clear();
            auto fill_qkv_hidden_sizes = [&] (const int slice_node_id) {
                int slice_start = extractConstant(net, matchedNodesIds[slice_node_id], 1).at<int64_t>(0);
                int slice_end = extractConstant(net, matchedNodesIds[slice_node_id], 2).at<int64_t>(0);
                if (slice_end == std::numeric_limits<int>::max()) {
                    qkv_hidden_sizes.push_back(0); // workaround for Slice with end=INT_MAX
                } else {
                    int64_t hidden_size = static_cast<int64_t>(slice_end - slice_start);
                    qkv_hidden_sizes.push_back(hidden_size);
                }
            };
            fill_qkv_hidden_sizes(slice_q);
            fill_qkv_hidden_sizes(slice_k);
            fill_qkv_hidden_sizes(slice_v);
            CV_CheckEQ(qkv_hidden_sizes.size(), static_cast<size_t>(3), "ONNXSimplifier/Attention: invalid qkv hidden sizes");
            CV_CheckEQ(int(qkv_hidden_sizes[0]), int(qkv_hidden_sizes[1]), "ONNXSimplifier/Attention: invalid qkv hidden sizes, q_hidden_size == v_hidden_size is required");
            // get attrs - num_heads, scale
            num_heads = 1;
            scale = extractConstant(net, matchedNodesIds[div_q], 1).at<float>(0);
            output_ndims = extractConstant(net, matchedNodesIds[last_reshape], 1).size[0];

            // get names
            weight_name = getInputName(net, matchedNodesIds[att_matmul], 1);
            bias_name = getInputName(net, matchedNodesIds[att_matmul], 2);
            return true;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>& net,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE {
        // add attrs
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attr_num_heads = node->add_attribute();
        attr_num_heads->set_name("num_heads");
        attr_num_heads->set_i(num_heads);
        opencv_onnx::AttributeProto* attr_qkv_hidden_sizes = node->add_attribute();
        attr_qkv_hidden_sizes->set_name("qkv_hidden_sizes");
        attr_qkv_hidden_sizes->add_ints(qkv_hidden_sizes[0]);
        attr_qkv_hidden_sizes->add_ints(qkv_hidden_sizes[1]);
        attr_qkv_hidden_sizes->add_ints(qkv_hidden_sizes[2]);
        opencv_onnx::AttributeProto* attr_scale = node->add_attribute();
        attr_scale->set_name("scale");
        attr_scale->set_f(scale);

        // add customized attrs
        opencv_onnx::AttributeProto* attr_output_ndims = node->add_attribute();
        attr_output_ndims->set_name("output_ndims");
        attr_output_ndims->set_i(output_ndims);

        // add inputs
        node->add_input(weight_name);
        node->add_input(bias_name);
    }

 protected:
    int att_matmul;
    int slice_q, slice_k, slice_v;
    int div_q, last_reshape;

    std::vector<int64_t> qkv_hidden_sizes; // order: [qk_hidden_size, qk_hidden_size, v_hidden_size]
    int64_t num_heads;
    float scale;

    int64_t output_ndims;

    std::string weight_name;
    std::string bias_name;
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
        div = addNodeToMatch("Div", input, addNodeToMatch("") /* B=sqrt(2) */ );
        int erf = addNodeToMatch("Erf", div);
        add = addNodeToMatch("Add", erf, addNodeToMatch("") /* B=1 */ );
        int mul = addNodeToMatch("Mul", input, add);
        mul2 = addNodeToMatch("Mul", mul, addNodeToMatch("") /* B=0.5 */) ;

        setFusedNode("Gelu", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            // Check Div[B=sqrt(2)]
            float divisor = extractConstant(net, matchedNodesIds[div], 1).at<float>(0);
            if (std::fabs(divisor - M_SQRT2) >= std::numeric_limits<float>::epsilon())
                return false;

            // Check Add[B=1]
            float add_const = extractConstant(net, matchedNodesIds[add], 1).at<float>(0);
            if (std::fabs(add_const - 1.f) >= std::numeric_limits<float>::epsilon())
                return false;

            // Check Mul[B=0.5]
            float mul_const = extractConstant(net, matchedNodesIds[mul2], 1).at<float>(0);
            if (std::fabs(mul_const - 0.5f) >= std::numeric_limits<float>::epsilon())
                return false;

            return true;
        }
        return false;
    }

private:
    int div, add, mul2;
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
        mul2 = addNodeToMatch("Mul", addNodeToMatch("") /* A=0.044714998453855515 */, mul1);
        int add0 = addNodeToMatch("Add", input, mul2);
        mul3 = addNodeToMatch("Mul", addNodeToMatch("") /* A=sqrt(2/pie) */, add0);
        int tanh = addNodeToMatch("Tanh", mul3);
        add1 = addNodeToMatch("Add", addNodeToMatch("") /* A=1 */, tanh);
        int mul4 = addNodeToMatch("Mul", input, add1);
        mul5 = addNodeToMatch("Mul", addNodeToMatch("") /* A=0.5 */, mul4);

        setFusedNode("GeluApproximation", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            // Check Mul[A=0.044714998453855515]
            float coef = extractConstant(net, matchedNodesIds[mul2], 0).at<float>(0);
            if (coef - 0.044714998453855515 >= 1e-6)
                return false;

            // Check Mul[A=sqrt(2/pie)]
            float sqrt_2_pie = extractConstant(net, matchedNodesIds[mul3], 0).at<float>(0);
            if (sqrt_2_pie - 0.7978845834732056 >= 1e-6)
                return false;

            // Check Add[A=1]
            float add_const = extractConstant(net, matchedNodesIds[add1], 0).at<float>(0);
            if (add_const - 1.f >= 1e-6)
                return false;

            // Check Mul[A=0.5]
            float mul_const = extractConstant(net, matchedNodesIds[mul5], 0).at<float>(0);
            if (mul_const - 0.5f >= 1e-6)
                return false;

            return true;
        }
        return false;
    }

private:
    int mul2, mul3, add1, mul5;
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

    Note: axes of ReduceMean must be:
          - last element is the axis of last dimension (-1 or (input_ndims - 1))
          - a list of adjacent axes, e.g. [1, 2, 3, ..., input_ndims - 1]
*/
class LayerNormSubGraph : public Subgraph
{
public:
    LayerNormSubGraph() : axis(-1), epsilon(1e-5)
    {
        int input = addNodeToMatch("");
        mean = addNodeToMatch("ReduceMean", input);

        int sub = addNodeToMatch("Sub", input, mean);

        pow = addNodeToMatch("Pow", sub, addNodeToMatch(""));
        mean1 = addNodeToMatch("ReduceMean", pow);
        add = addNodeToMatch("Add", mean1, addNodeToMatch(""));
        int sqrt = addNodeToMatch("Sqrt", add);

        int div = addNodeToMatch("Div", sub, sqrt);
        mul = addNodeToMatch("Mul", div, addNodeToMatch(""));
        bias = addNodeToMatch("Add", mul, addNodeToMatch(""));

        setFusedNode("LayerNormalization", input);
    }

    static std::vector<int64_t> extractAxis(const Ptr<ImportGraphWrapper>& net, int node_id)
    {
        // TODO: consider ReduceMean-18 which has axes as one of the inputs instead of attributes
        Ptr<ImportNodeWrapper> mean_ptr = net->getNode(node_id);
        opencv_onnx::NodeProto* mean_node = mean_ptr.dynamicCast<ONNXNodeWrapper>()->node;
        std::vector<int64_t> axes;
        for (int i = 0; i < mean_node->attribute_size(); i++)
        {
            opencv_onnx::AttributeProto attr = mean_node->attribute(i);
            if (attr.name() != "axes")
                continue;
            for (int j = 0; j < attr.ints_size(); j++) {
                axes.push_back(attr.ints(j));
            }
        }
        return axes;
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            float pow_exp = extractConstant(net, matchedNodesIds[pow], 1).at<float>(0);
            if (pow_exp - 2 > 1e-5) // not pow(2)
                return false;

            std::vector<int64_t> axes = extractAxis(net, matchedNodesIds[mean]);
            // check whether it is -1 or last_axis or [axis, ..., last_axis]
            int64_t input_ndims = static_cast<int64_t>(net.dynamicCast<ONNXGraphWrapper>()->getTensorShapeSize(matchedNodesIds[mean], 0));
            if (input_ndims == -1) {
                return false; // input shape unknown
            }
            // assume that axes are sorted in ascending order, e.g. [0, 1, 2, 3] or [-3, -2, -1]
            if (axes.back() != -1 && axes.back() != (input_ndims - 1)) {
                return false;
            }
            for (size_t i = 0; i < axes.size() - 1; i++) {
                if (axes[i] - axes[i + 1] != -1) {
                    return false;
                }
            }

            std::vector<int64_t> axes1 = extractAxis(net, matchedNodesIds[mean1]);
            if (axes.size() != axes1.size())
                return false;
            for (size_t i = 0; i < axes.size(); i++) {
                if (((axes[i] + input_ndims) % input_ndims) != ((axes1[i] + input_ndims) % input_ndims)) {
                    return false;
                }
            }
            axis = axes[0];

            epsilon = extractConstant(net, matchedNodesIds[add], 1).at<float>(0);

            weight_name = getInputName(net, matchedNodesIds[mul], 1);
            bias_name = getInputName(net, matchedNodesIds[bias], 1);

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
    int pow, mean, mean1, add, mul, bias;
};

class SoftMaxSubgraphBase : public Subgraph
{
public:
    SoftMaxSubgraphBase() : axis(1), id(-1) {}

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
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
        id = sum;

        addNodeToMatch("Div", inpExp, sum);
        setFusedNode("Softmax", input);
    }
};

class SoftMaxSubgraph2 : public SoftMaxSubgraphBase {
public:
    SoftMaxSubgraph2() {
        int input = addNodeToMatch("");

        int reducemax = addNodeToMatch("ReduceMax", input);
        id = reducemax;

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
        id = reducemax;

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
        hardSigmoidId = addNodeToMatch("HardSigmoid", input);
        addNodeToMatch("Mul", input, hardSigmoidId);
        setFusedNode("HardSwish", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            Ptr<ImportNodeWrapper> hardSigmoid = net->getNode(matchedNodesIds[hardSigmoidId]);
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

private:
    int hardSigmoidId;
};

class CeluSubgraph : public Subgraph
{
public:
    CeluSubgraph() : alpha(1.f)
    {
        int input = addNodeToMatch("");
        div = addNodeToMatch("Div", input, addNodeToMatch(""));
        elu = addNodeToMatch("Elu", div);
        mul = addNodeToMatch("Mul", addNodeToMatch(""), elu);
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
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
        {
            float alpha_div = extractAlpha(net, matchedNodesIds[div], 1);
            float alpha_mul = extractAlpha(net, matchedNodesIds[mul], 0);
            float alpha_elu = 1.f;

            Ptr<ImportNodeWrapper> elu_ptr = net->getNode(matchedNodesIds[elu]);
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
    int div, mul, elu;
};

class NormalizeSubgraphBase : public Subgraph
{
public:
    NormalizeSubgraphBase(int _normNodeOrder = 1) : axis(1), normNodeOrder(_normNodeOrder) {}

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds))
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
    NormalizeSubgraph3() : NormalizeSubgraphBase(3)
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
    NormalizeSubgraph4() : NormalizeSubgraphBase(2)
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
    NormalizeSubgraph5() : NormalizeSubgraphBase(2)
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
        gather = addNodeToMatch("Gather", input, index);
        cast = addNodeToMatch("Cast", gather);
        setFusedNode("Gather", input, index);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE
    {
        bool retVal = Subgraph::match(net, nodeId, matchedNodesIds);
        size_t matchedNodesNum = matchedNodesIds.size();
        // Now we check if merging can be made for these Gather and Cast nodes
        if (!retVal || matchedNodesNum < 2)
            return retVal;
        else {
            int nodeToMatch = matchedNodesIds[cast];
            const Ptr<ImportNodeWrapper> node = net->getNode(nodeToMatch);
            if (node->getType() == "Cast") {
                int inpNodeId = matchedNodesIds[gather];
                const Ptr<ImportNodeWrapper> inpNode = net->getNode(inpNodeId);
                if (inpNode->getType() == "Gather") {
                    int numNodes = net->getNumNodes();
                    std::string inpNodeName = node->getInputName(0);
                    for (int i = 0; i < numNodes; ++i) {
                        const Ptr<ImportNodeWrapper> node_to_check = net->getNode(i);
                        int numInp = node_to_check->getNumInputs();
                        for (int inp = 0; inp < numInp; ++inp) {
                            if (i != nodeToMatch && inpNodeName == node_to_check->getInputName(inp)) {
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

private:
    int cast, gather;
};

/*  Constant folding shape for Expand.

    Before fusion:
             +--------------------------------------------------------------+ (X)
             |                                                              |
    ConstantOfShape[input=[4]] -> Mul[B=-1] -> Equal[A=[2, -1, -1, -1]] -> Where[Y=[2, -1, -1, -1]] -> Expand
             \                                                           \
             value=[1]                                                   (condition)

*/
class ExpandSubgraph : public Subgraph
{
public:
    ExpandSubgraph()
    {
        int input = addNodeToMatch("");
        int values = addNodeToMatch("");
        init = addNodeToMatch("ConstantOfShape", values);
        int coeff = addNodeToMatch("Constant");
        mul = addNodeToMatch("Mul", init, coeff);
        int shape = addNodeToMatch("Constant");
        condition = addNodeToMatch("Equal", shape, mul);
        where = addNodeToMatch("Where", condition, init, addNodeToMatch("Constant"));
        addNodeToMatch("Expand", input, where);
        setFusedNode("Expand", input, shape);
    }

    static int extractValue(const Ptr<ImportGraphWrapper>& net, int node_id, int64_t &val) {
        Ptr<ImportNodeWrapper> node_wrapper = net->getNode(node_id);
        opencv_onnx::NodeProto* node = node_wrapper.dynamicCast<ONNXNodeWrapper>()->node;

        if (node->attribute_size() == 0) {
            val = 0;
            return 1;
        } else if (node->attribute_size() == 1) {
            opencv_onnx::AttributeProto attr = node->attribute(0);
            if (attr.name() != "value") {
                return 0;
            }
            Mat mat_value = getMatFromTensor(attr.t());
            switch (mat_value.type()) {
                case CV_32S:
                    val = static_cast<int64_t>(mat_value.at<int>());
                    break;
                case CV_64S:
                    val = mat_value.at<int64_t>();
                    break;
                default: return 0;
            }
            return 1;
        }
        return 0;
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds) CV_OVERRIDE {
        if (Subgraph::match(net, nodeId, matchedNodesIds)) {
            int64_t value_ConstantOfShape;
            if (!extractValue(net, matchedNodesIds[init], value_ConstantOfShape)) {
                return false;
            }
            std::vector<int> input_ConstantOfShape = extractConstant(net, matchedNodesIds[init], 0);
            if (input_ConstantOfShape.size() != static_cast<size_t>(1)) {
                return false;
            }
            std::vector<int> B_Mul = extractConstant(net, matchedNodesIds[mul], 1);
            if (B_Mul.size() != static_cast<size_t>(1)) {
                return false;
            }

            std::vector<int> A_Equal = extractConstant(net, matchedNodesIds[condition], 0);
            if (A_Equal.size() != static_cast<size_t>(input_ConstantOfShape[0])) {
                return false;
            }

            std::vector<int> Y_Where = extractConstant(net, matchedNodesIds[where], 2);
            if (Y_Where.size() != A_Equal.size()) {
                return false;
            }

            // run ConstantOfShape
            std::vector<int64_t> output_ConstantOfShape(std::accumulate(input_ConstantOfShape.begin(), input_ConstantOfShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()), value_ConstantOfShape);
            // run Mul
            std::vector<int64_t> output_Mul = output_ConstantOfShape;
            for (size_t i = 0; i < output_Mul.size(); i++) {
                int64_t b = B_Mul[0];
                output_Mul[i] *= b;
            }
            // run Equal
            std::vector<bool> output_Equal(output_Mul.size());
            for (int i = 0; i < output_Equal.size(); i++) {
                if (A_Equal[i] == output_Mul[i]) {
                    output_Equal[i] = true;
                } else {
                    output_Equal[i] = false;
                }
            }
            // run Where
            std::vector<int64_t> output_Where(output_Equal.size());
            for (int i = 0; i < output_Where.size(); i++) {
                if (output_Equal[i]) {
                    output_Where[i] = output_ConstantOfShape[i];
                } else {
                    output_Where[i] = Y_Where[i];
                }
            }
            shape = output_Where;

            return true;
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>& graph,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >& inputs) CV_OVERRIDE {
        // replace values
        opencv_onnx::NodeProto* node_shape = inputs[1].dynamicCast<ONNXNodeWrapper>()->node;
        auto attr = node_shape->mutable_attribute()->Mutable(0);
        auto tensor = attr->mutable_t();
        tensor->clear_raw_data();
        tensor->set_raw_data(std::string((const char*)(shape.data()), shape.size() * sizeof(int64_t)));
    }

protected:
    std::vector<int64_t> shape;

private:
    int init, mul, condition, where;
};

class MishSubgraph : public Subgraph
{
public:
    MishSubgraph()
    {
        int input = addNodeToMatch("");
        int softplus = addNodeToMatch("Softplus", input);
        int tanh = addNodeToMatch("Tanh", softplus);
        addNodeToMatch("Mul", tanh, input);
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
        int slice = addNodeToMatch("Slice", {shape,
                                             addNodeToMatch(""),
                                             addNodeToMatch(""),
                                             addNodeToMatch(""),
                                             addNodeToMatch("")});

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

class ResizeSubgraph3 : public Subgraph
{
public:
    ResizeSubgraph3() : Subgraph()
    {
        int shapeSrc = addNodeToMatch("");
        int input = addNodeToMatch("");

        int shape_h = addNodeToMatch("Shape", shapeSrc);
        int shape_w = addNodeToMatch("Shape", shapeSrc);
        int gather_h = addNodeToMatch("Gather", shape_h, addNodeToMatch("Constant"));
        int gather_w = addNodeToMatch("Gather", shape_w, addNodeToMatch("Constant"));
        int unsqueeze_h = addNodeToMatch("Unsqueeze", gather_h);
        int unsqueeze_w = addNodeToMatch("Unsqueeze", gather_w);
        int concat1 = addNodeToMatch("Concat", unsqueeze_h, unsqueeze_w);
        int cast = addNodeToMatch("Cast", concat1);

        int shape2 = addNodeToMatch("Shape", input);
        int slice = addNodeToMatch("Slice", {shape2,
                                             addNodeToMatch(""),
                                             addNodeToMatch(""),
                                             addNodeToMatch(""),
                                             addNodeToMatch("")});
        int concat2 = addNodeToMatch("Concat", slice, cast);
        addNodeToMatch("Resize", input, addNodeToMatch("Constant"), addNodeToMatch("Constant"), concat2);

        setFusedNode("Upsample", input, shapeSrc);
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
    subgraphs.push_back(makePtr<BiasedMatmulSubgraph>());
    subgraphs.push_back(makePtr<AdjustSliceAllOptionalInputsSubgraph>(3));
    subgraphs.push_back(makePtr<AdjustSliceAllOptionalInputsSubgraph>(4));
    subgraphs.push_back(makePtr<GeluSubGraph>());
    subgraphs.push_back(makePtr<GeluApproximationSubGraph>());
    subgraphs.push_back(makePtr<LayerNormSubGraph>());
    subgraphs.push_back(makePtr<GatherCastSubgraph>());
    subgraphs.push_back(makePtr<MulCastSubgraph>());
    subgraphs.push_back(makePtr<UpsampleSubgraph>());
    subgraphs.push_back(makePtr<ResizeSubgraph1>());
    subgraphs.push_back(makePtr<ResizeSubgraph2>());
    subgraphs.push_back(makePtr<ResizeSubgraph3>());
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
    subgraphs.push_back(makePtr<MishSubgraph>());
    subgraphs.push_back(makePtr<NormalizeSubgraph4>());
    subgraphs.push_back(makePtr<NormalizeSubgraph5>());
    if (getParam_DNN_BACKEND_DEFAULT() == DNN_BACKEND_OPENCV) {
        subgraphs.push_back(makePtr<AttentionSubGraph>());
        subgraphs.push_back(makePtr<AttentionSingleHeadSubGraph>());
    }

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new ONNXGraphWrapper(net)), subgraphs);
}





void getExternalTensorData(const opencv_onnx::TensorProto& tensor_proto, std::vector<char>& tensor_data, const std::string base_path = ""){
    CV_Assert(tensor_proto.has_data_location() && tensor_proto.data_location() == opencv_onnx::TensorProto::EXTERNAL);
    auto it_begin = tensor_proto.external_data().begin();
    auto it_end = tensor_proto.external_data().end();
    // file path
    auto it = std::find_if(it_begin, it_end,[](const auto& entry) { return entry.key() == "location"; });
    CV_Assert(it != it_end);

    std::string location_path = it->value();
    std::string full_path = base_path.empty() ? location_path : (base_path + "/" + location_path);

    std::ifstream file(full_path, std::ios::binary | std::ios::ate);
    CV_Assert(file.is_open());

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    tensor_data.resize(size);

    CV_Assert(file.read(tensor_data.data(), size));
}


Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto, bool uint8ToInt8, const std::string base_path)
{
    if (tensor_proto.raw_data().empty() && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty() &&
        tensor_proto.int32_data().empty() &&
        (!tensor_proto.has_data_location() || tensor_proto.data_location() != opencv_onnx::TensorProto::EXTERNAL)
    )
        return Mat();

    // read binary data, should be just empty in case it is set in <DTYPE>_data field
    std::vector<char> external_tensor_data;
    char* val;
    if (tensor_proto.has_data_location() && tensor_proto.data_location() == opencv_onnx::TensorProto::EXTERNAL){
        getExternalTensorData(tensor_proto, external_tensor_data, base_path);
        val = external_tensor_data.data();
    }
    else
        val = const_cast<char*>(tensor_proto.raw_data().c_str());

    int datatype = tensor_proto.data_type();
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
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_FLOAT16)
    {
        // FIXME, for now, we only load FP16 Tensor as FP32 Mat, full support for FP16 is required in the future.
        CV_LOG_ONCE_INFO(NULL, "DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.");

        // ONNX saves float 16 data in two format: int32 and raw_data.
        // Link: https://github.com/onnx/onnx/issues/4460#issuecomment-1224373746
        if (!tensor_proto.int32_data().empty())
        {
            int offset = 0;
#ifdef WORDS_BIGENDIAN
            offset = 1;
#endif
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();

            AutoBuffer<hfloat, 16> aligned_val;
            size_t sz = tensor_proto.int32_data().size();
            aligned_val.allocate(sz);
            hfloat* bufPtr = aligned_val.data();

            hfloat *fp16Ptr = (hfloat *)field.data();
            for (int i = 0; i < sz; i++)
            {
                bufPtr[i] = fp16Ptr[i*2 + offset];
            }
            Mat(sizes, CV_16FC1, bufPtr).convertTo(blob, CV_32FC1);
        }
        else
        {
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required.
            AutoBuffer<hfloat, 16> aligned_val;
            if (!isAligned<sizeof(hfloat)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(hfloat)));
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
        {
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
            Mat(sizes, CV_32SC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        if (!tensor_proto.int64_data().empty()) {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();
            Mat(sizes, CV_64SC1, (void*)src.data()).copyTo(blob);
        }
        else
        {
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
            Mat(sizes, CV_64SC1, (void*)src).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT8)
    {
        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, CV_8S);
        }
        else
        {
            Mat(sizes, CV_8S, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_UINT8)
    {
        // TODO : Add support for uint8 weights and acitvations. For now, converting uint8 tensors to int8.

        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            if (uint8ToInt8)
                Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, CV_8S, 1, -128); // handle as ONNX quantized weight
            else
                Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, CV_8U);
        }
        else
        {
            if (uint8ToInt8)
                Mat(sizes, CV_8U, val).convertTo(blob, CV_8S, 1, -128);  // handle as ONNX quantized weight
            else
                Mat(sizes, CV_8U, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_BOOL)
    {
        Mat(sizes, CV_Bool, val).copyTo(blob);
    }
    else
    {
        // @TODO: refactor the error handling
        std::string errorMsg = "Unsupported data type: "; /* +
                            opencv_onnx::TensorProto_DataType_Name(datatype);*/

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
