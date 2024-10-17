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
        // Instead of 'Tile', 'Expand' etc. we may now have 'Tile2', 'Expand2' etc.
        // We should correctly match them with the respective patterns
        for (auto& l: layers) {
            if (!l.empty() && l[l.size()-1] == '2')
                l = l.substr(0, l.size()-1);
        }

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

}}
