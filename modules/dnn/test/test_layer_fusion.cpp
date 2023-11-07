// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class Test_Layer_Fusion : public DNNTestLayer {
 public:
    bool required;

    Test_Layer_Fusion() : required(true) {}

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

TEST_P(Test_Layer_Fusion, GeluSubGraph) {
    test("gelu", "Gelu");
}

TEST_P(Test_Layer_Fusion, GeluApproximationSubGraph) {
    test("gelu_approximation", "GeluApproximation");
}

TEST_P(Test_Layer_Fusion, LayerNormSubGraph) {
    test("layer_norm_expanded", "LayerNormalization");
}

TEST_P(Test_Layer_Fusion, ResizeSubgraph) {
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

TEST_P(Test_Layer_Fusion, SoftmaxSubgraph) {
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

TEST_P(Test_Layer_Fusion, HardSwishSubgraph) {
    test_conformance("test_hardswish_expanded", "HardSwish");
}

TEST_P(Test_Layer_Fusion, CeluSubgraph) {
    test_conformance("test_celu_expanded", "Celu");
}

TEST_P(Test_Layer_Fusion, NormalizeSubgraph) {
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

TEST_P(Test_Layer_Fusion, BatchNormalizationSubgraph) {
    /* Test for 2 subgraphs
        - BatchNormalizationSubgraph1
        - BatchNormalizationSubgraph2
    */
    test("frozenBatchNorm2d", "BatchNorm");
    test("batch_norm_subgraph", "BatchNorm");
}

TEST_P(Test_Layer_Fusion, ExpandSubgraph) {
    test("expand_neg_batch", "Expand");
}

TEST_P(Test_Layer_Fusion, MishSubgraph) {
    /* Test for 2 subgraphs
        - SoftplusSubgraph
        - MishSubgraph
    */
    test("mish_no_softplus", "Mish");
    test("mish", "Mish");
}

// Different backends are sharing the same subgraph fusion rule,
// so testing on tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU) is enough
using BackendTargetTuple = tuple<Backend, Target>;
INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_Layer_Fusion,
                        testing::ValuesIn(std::vector<BackendTargetTuple>{std::make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU)}));

}}
