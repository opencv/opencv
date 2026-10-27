// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class Test_Graph_Simplifier : public ::testing::Test {
 public:
    bool required;

    Test_Graph_Simplifier() : required(true) {}

    void test_conformance(const std::string &basename, const std::string &expected_layer) {
        test(basename + std::string("/model"), std::vector<std::string>{expected_layer}, std::string("dnn/onnx/conformance/node/"));
    }

    void test(const std::string &basename, const std::string &expected_layer) {
        test(basename, std::vector<std::string>{expected_layer});
    }

    void test(const std::string &basename, const std::vector<std::string> &expected_layers, const std::string &model_path_prefix = std::string("dnn/onnx/models/")) {
        std::string model_path = findDataFile(model_path_prefix + basename + std::string(".onnx"), required);
        auto net = readNet(model_path);
        std::vector<std::string> layers;
        net.getLayerTypes(layers);

        // remove Const, Identity (output layer), __NetInputLayer__ (input layer)
        layers.erase(std::remove_if(layers.begin(), layers.end(), [] (const std::string l) { return l == "Const" || l == "Identity" || l == "__NetInputLayer__"; }), layers.end());

        EXPECT_EQ(layers, expected_layers);
    }
};

TEST_F(Test_Graph_Simplifier, GeluSubGraph) {
    test("gelu", "Gelu");
    test("bias_gelu", std::vector<std::string>{"Gelu", "NaryEltwise"});
}

TEST_F(Test_Graph_Simplifier, GeluApproximationSubGraph) {
    test("gelu_approximation", "GeluApproximation");
}

TEST_F(Test_Graph_Simplifier, LayerNormSubGraph) {
    test("layer_norm_expanded", "LayerNormalization");
    test("layer_norm_expanded_with_initializers", "LayerNormalization");
}

TEST_F(Test_Graph_Simplifier, LayerNormNoFusionSubGraph) {
    test("layer_norm_no_fusion", std::vector<std::string>{"NaryEltwise", "Reduce", "Sqrt"});
}

TEST_F(Test_Graph_Simplifier, ResizeSubgraph) {
    /* Test for 6 subgraphs:
        - GatherCastSubgraph
        - MulCastSubgraph
        - UpsampleSubgraph
        - ResizeSubgraph1
        - ResizeSubgraph2
        - ResizeSubgraph3
    */
    test("upsample_unfused_torch1.2", std::vector<std::string>{"BatchNorm", "Resize"});
    test("resize_nearest_unfused_opset11_torch1.3", std::vector<std::string>{"BatchNorm", "Convolution", "Resize"});
    test("resize_nearest_unfused_opset11_torch1.4", std::vector<std::string>{"BatchNorm", "Convolution", "Resize"});
    test("upsample_unfused_opset9_torch1.4", std::vector<std::string>{"BatchNorm", "Convolution", "Resize"});
    test("two_resizes_with_shared_subgraphs", std::vector<std::string>{"NaryEltwise", "Resize"});
}

TEST_F(Test_Graph_Simplifier, SoftmaxSubgraph) {
    /* Test for 3 subgraphs
        - SoftMaxSubgraph
        - SoftMaxSubgraph2 (conformance)
        - LogSoftMaxSubgraph (conformance)
    */
    test("softmax_unfused", "Softmax");
    test_conformance("test_softmax_example_expanded", "Softmax");
    test_conformance("test_softmax_axis_2_expanded", "Softmax");
    test_conformance("test_softmax_default_axis_expanded", "Softmax");
    test_conformance("test_softmax_axis_0_expanded", "Softmax");
    test_conformance("test_softmax_axis_1_expanded", "Softmax");
    test_conformance("test_softmax_large_number_expanded", "Softmax");
    test_conformance("test_softmax_negative_axis_expanded", "Softmax");
    test_conformance("test_logsoftmax_axis_2_expanded", "Softmax");
    test_conformance("test_logsoftmax_example_1_expanded", "Softmax");
    test_conformance("test_logsoftmax_negative_axis_expanded", "Softmax");
    test_conformance("test_logsoftmax_axis_0_expanded", "Softmax");
    test_conformance("test_logsoftmax_axis_1_expanded", "Softmax");
    test_conformance("test_logsoftmax_large_number_expanded", "Softmax");
    test_conformance("test_logsoftmax_default_axis_expanded", "Softmax");
}

TEST_F(Test_Graph_Simplifier, HardSwishSubgraph) {
    test_conformance("test_hardswish_expanded", "HardSwish");
}

TEST_F(Test_Graph_Simplifier, CeluSubgraph) {
    test_conformance("test_celu_expanded", "Celu");
}

TEST_F(Test_Graph_Simplifier, NormalizeSubgraph) {
    /* Test for 6 subgraphs
        - NormalizeSubgraph1
        - NormalizeSubgraph2
        - NormalizeSubgraph2_2
        - NormalizeSubgraph3
        - NormalizeSubgraph4
        - NormalizeSubgraph5
    */
    test("reduceL2_subgraph_2", "Normalize");
    test("reduceL2_subgraph", "Normalize");
    test("normalize_fusion", "Normalize");
}

TEST_F(Test_Graph_Simplifier, BatchNormalizationSubgraph) {
    /* Test for 2 subgraphs
        - BatchNormalizationSubgraph1
        - BatchNormalizationSubgraph2
    */
    test("frozenBatchNorm2d", "BatchNorm");
    test("batch_norm_subgraph", "BatchNorm");
}

TEST_F(Test_Graph_Simplifier, ExpandSubgraph) {
    test("expand_neg_batch", "Expand");
}

TEST_F(Test_Graph_Simplifier, MishSubgraph) {
    /* Test for 2 subgraphs
        - SoftplusSubgraph
        - MishSubgraph
    */
    test("mish_no_softplus", "Mish");
    test("mish", "Mish");
}

TEST_F(Test_Graph_Simplifier, AttentionSubgraph) {
    /* Test for 2 subgraphs
        - AttentionSubgraph
        - AttentionSingleHeadSubgraph
    */
    test("attention", "Attention");
    test("attention_single_head", "Attention");
}

TEST_F(Test_Graph_Simplifier, BiasedMatMulSubgraph) {
    /* Test for 1 subgraphs
        - BiasedMatMulSubgraph
    */
    test("biased_matmul", "MatMul");
}

static std::string pbLenDelimited(int fieldNum, const std::string& payload)
{
    CV_Assert(payload.size() < 128u);
    std::string out;
    out.push_back(static_cast<char>((fieldNum << 3) | 2));
    out.push_back(static_cast<char>(payload.size()));
    out.append(payload);
    return out;
}

static std::string pbVarint(int fieldNum, int value)
{
    std::string out;
    out.push_back(static_cast<char>((fieldNum << 3) | 0));
    unsigned int v = static_cast<unsigned int>(value);
    while (v >= 0x80u)
    {
        out.push_back(static_cast<char>((v & 0x7fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<char>(v));
    return out;
}

static std::string pbAttrType(const std::string& key, int dtype)
{
    std::string attrVal = pbVarint(6, dtype);
    std::string entry = pbLenDelimited(1, key) + pbLenDelimited(2, attrVal);
    return pbLenDelimited(5, entry);
}

static std::string pbAttrTensorContent(const std::string& key, int dtype,
                                       const std::vector<int>& dims,
                                       const std::string& rawContent)
{
    std::string shapeProto;
    for (size_t i = 0; i < dims.size(); ++i)
        shapeProto += pbLenDelimited(2, pbVarint(1, dims[i]));
    std::string tensorProto = pbVarint(1, dtype) + pbLenDelimited(2, shapeProto);
    if (!rawContent.empty())
        tensorProto += pbLenDelimited(4, rawContent);
    std::string attrVal = pbLenDelimited(8, tensorProto);
    std::string entry = pbLenDelimited(1, key) + pbLenDelimited(2, attrVal);
    return pbLenDelimited(5, entry);
}

static std::string pbNodeDef(const std::string& name, const std::string& op,
                             const std::vector<std::string>& inputs,
                             const std::string& attrs)
{
    std::string body = pbLenDelimited(1, name) + pbLenDelimited(2, op);
    for (size_t i = 0; i < inputs.size(); ++i)
        body += pbLenDelimited(3, inputs[i]);
    body += attrs;
    return pbLenDelimited(1, body);
}

static std::string pbInt32Le(int value)
{
    std::string out(4, '\0');
    out[0] = static_cast<char>(value & 0xff);
    out[1] = static_cast<char>((value >> 8) & 0xff);
    out[2] = static_cast<char>((value >> 16) & 0xff);
    out[3] = static_cast<char>((value >> 24) & 0xff);
    return out;
}

TEST_F(Test_Graph_Simplifier, TensorFlowControlPortInputsAndKerasReshapeTensorContent)
{
    // DT_FLOAT = 1, DT_INT32 = 3
    std::string validGraph =
        pbNodeDef("x", "Placeholder", std::vector<std::string>(), pbAttrType("dtype", 1)) +
        pbNodeDef("id", "Identity", std::vector<std::string>{"x"}, pbAttrType("T", 1)) +
        pbNodeDef("y", "Relu", std::vector<std::string>{"id:0", "^id:0"}, pbAttrType("T", 1));

    Net net;
    ASSERT_NO_THROW(net = readNetFromTensorflow(validGraph.data(), validGraph.size()));
    Mat inp(std::vector<int>{1, 1, 2, 3}, CV_32F, Scalar(3.0f));
    net.setInput(inp);
    Mat out = net.forward();
    EXPECT_EQ(out.total(), (size_t)6);
    EXPECT_FLOAT_EQ(out.at<float>(0), 3.0f);

    const std::string zero4 = pbInt32Le(0);
    const std::string one4 = pbInt32Le(1);
    const std::string two4 = pbInt32Le(2);
    const std::string three4 = pbInt32Le(3);
    std::string kerasReshapeGraph =
        pbNodeDef("x", "Placeholder", std::vector<std::string>(), pbAttrType("dtype", 1)) +
        pbNodeDef("shape", "Shape", std::vector<std::string>{"x"}, pbAttrType("T", 1)) +
        pbNodeDef("stack", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{1}, zero4)) +
        pbNodeDef("stack_1", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{1}, one4)) +
        pbNodeDef("stack_2", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{1}, one4)) +
        pbNodeDef("strided_slice", "StridedSlice",
                  std::vector<std::string>{"shape", "stack", "stack_1", "stack_2"},
                  pbAttrType("T", 3) + pbAttrType("Index", 3)) +
        pbNodeDef("dim0", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{}, one4)) +
        pbNodeDef("dim1", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{}, two4)) +
        pbNodeDef("dim2", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{}, three4)) +
        pbNodeDef("pack", "Pack", std::vector<std::string>{"strided_slice", "dim0", "dim1", "dim2"},
                  pbAttrType("T", 3)) +
        pbNodeDef("reshape", "Reshape", std::vector<std::string>{"x", "pack"},
                  pbAttrType("T", 1));

    Net reshapeNet;
    ASSERT_NO_THROW(reshapeNet = readNetFromTensorflow(kerasReshapeGraph.data(), kerasReshapeGraph.size()));
    reshapeNet.setInput(inp);
    Mat reshaped = reshapeNet.forward();
    EXPECT_EQ(reshaped.dims, 4);
    EXPECT_EQ(reshaped.size[0], 1);
    EXPECT_EQ(reshaped.size[1], 3);
    EXPECT_EQ(reshaped.size[2], 1);
    EXPECT_EQ(reshaped.size[3], 2);
}

TEST_F(Test_Graph_Simplifier, TensorFlowRejectsTruncatedTensorContentAndMalformedKerasReshape)
{
    std::string truncatedBiasGraph =
        pbNodeDef("x", "Placeholder", std::vector<std::string>(), pbAttrType("dtype", 1)) +
        pbNodeDef("b", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 1, std::vector<int>{1}, std::string("\x01\x02\x03", 3))) +
        pbNodeDef("y", "BiasAdd", std::vector<std::string>{"x", "b"}, pbAttrType("T", 1));
    EXPECT_THROW(readNetFromTensorflow(truncatedBiasGraph.data(), truncatedBiasGraph.size()), cv::Exception);

    const std::string zero4 = pbInt32Le(0);
    const std::string one4 = pbInt32Le(1);
    std::string missingValueReshapeGraph =
        pbNodeDef("x", "Placeholder", std::vector<std::string>(), pbAttrType("dtype", 1)) +
        pbNodeDef("shape", "Shape", std::vector<std::string>{"x"}, pbAttrType("T", 1)) +
        pbNodeDef("stack", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{1}, zero4)) +
        pbNodeDef("stack_1", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{1}, one4)) +
        pbNodeDef("stack_2", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{1}, one4)) +
        pbNodeDef("strided_slice", "StridedSlice",
                  std::vector<std::string>{"shape", "stack", "stack_1", "stack_2"},
                  pbAttrType("T", 3) + pbAttrType("Index", 3)) +
        pbNodeDef("dim0", "Const", std::vector<std::string>(), pbAttrType("dtype", 3)) +
        pbNodeDef("dim1", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{}, one4)) +
        pbNodeDef("dim2", "Const", std::vector<std::string>(),
                  pbAttrTensorContent("value", 3, std::vector<int>{}, one4)) +
        pbNodeDef("pack", "Pack", std::vector<std::string>{"strided_slice", "dim0", "dim1", "dim2"},
                  pbAttrType("T", 3)) +
        pbNodeDef("reshape", "Reshape", std::vector<std::string>{"x", "pack"},
                  pbAttrType("T", 1));
    EXPECT_THROW(readNetFromTensorflow(missingValueReshapeGraph.data(), missingValueReshapeGraph.size()), cv::Exception);
}

}}
