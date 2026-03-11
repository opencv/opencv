// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

//build a single-layer network
static Net buildSingleLayerNet(LayerParams& lp, const MatShape& inputShape,
                               int inputType = CV_32F)
{
    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    Mat input(inputShape, inputType);
    randu(input, -1, 1);
    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    return net;
}

TEST(Test_GetFLOPS, Convolution)
{
    LayerParams lp;
    lp.type = "Convolution";
    lp.name = "conv";
    lp.set("kernel_size", 3);
    lp.set("num_output", 64);
    lp.set("pad", 1);
    lp.set("bias_term", true);

    int weightsShape[] = {64, 3, 3, 3};
    Mat weights(4, weightsShape, CV_32F);
    randu(weights, -1, 1);
    lp.blobs.push_back(weights);
    Mat bias(1, 64, CV_32F, Scalar(0));
    lp.blobs.push_back(bias);

    MatShape inputShape{1, 3, 224, 224};
    Net net = buildSingleLayerNet(lp, inputShape);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // Expected: output is [1, 64, 224, 224]
    // FLOPS per output element = 2 * 3*3*3 + 1 = 55
    // Total = 1 * 64 * 224 * 224 * 55 = 176,455,680
    // But the Data layer also contributes 0 flops, so total = conv flops
    int64 expectedFlops = (int64)1 * 64 * 224 * 224 * (2 * 3 * 3 * 3 + 1);
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, FullyConnected)
{
    LayerParams lp;
    lp.type = "InnerProduct";
    lp.name = "fc";
    lp.set("num_output", 1000);

    int weightsShape[] = {1000, 2048};
    Mat weights(2, weightsShape, CV_32F);
    randu(weights, -1, 1);
    lp.blobs.push_back(weights);
    Mat bias(1, 1000, CV_32F, Scalar(0));
    lp.blobs.push_back(bias);

    MatShape inputShape{1, 2048};
    Net net = buildSingleLayerNet(lp, inputShape);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // Expected: 3 * innerSize * output = 3 * 2048 * 1000 = 6,144,000
    int64 expectedFlops = (int64)3 * 2048 * 1000;
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, MaxPooling)
{
    LayerParams lp;
    lp.type = "Pooling";
    lp.name = "pool";
    lp.set("pool", "max");
    lp.set("kernel_size", 2);
    lp.set("stride", 2);

    MatShape inputShape{1, 64, 112, 112};
    Net net = buildSingleLayerNet(lp, inputShape);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // Output: [1, 64, 56, 56]
    // Max pool: karea comparisons per output element = 2*2 = 4
    // Total = 1 * 64 * 56 * 56 * 4 = 802,816
    int64 expectedFlops = (int64)1 * 64 * 56 * 56 * 4;
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, BatchNorm)
{
    LayerParams lp;
    lp.type = "BatchNorm";
    lp.name = "bn";
    lp.set("has_weight", true);
    lp.set("has_bias", true);
    lp.set("eps", 1e-5);

    int channels = 64;
    Mat mean(1, channels, CV_32F, Scalar(0));
    Mat var(1, channels, CV_32F, Scalar(1));
    Mat scale(1, channels, CV_32F, Scalar(1));
    Mat shift(1, channels, CV_32F, Scalar(0));
    lp.blobs.push_back(mean);
    lp.blobs.push_back(var);
    lp.blobs.push_back(scale);
    lp.blobs.push_back(shift);

    MatShape inputShape{1, 64, 56, 56};
    Net net = buildSingleLayerNet(lp, inputShape);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // BatchNorm: 3 flops per element
    int64 expectedFlops = (int64)3 * 1 * 64 * 56 * 56;
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, Softmax)
{
    LayerParams lp;
    lp.type = "Softmax";
    lp.name = "softmax";

    MatShape inputShape{1, 1000};
    Net net = buildSingleLayerNet(lp, inputShape);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // Softmax: 4 flops per element
    int64 expectedFlops = (int64)4 * 1000;
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, Scale)
{
    LayerParams lp;
    lp.type = "Scale";
    lp.name = "scale";
    lp.set("axis", 1);
    lp.set("has_bias", true);

    int channels = 64;
    Mat scaleData(1, channels, CV_32F, Scalar(1));
    Mat biasData(1, channels, CV_32F, Scalar(0));
    lp.blobs.push_back(scaleData);
    lp.blobs.push_back(biasData);

    MatShape inputShape{1, 64, 56, 56};
    Net net = buildSingleLayerNet(lp, inputShape);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // Scale: 2 flops per element (multiply + add)
    int64 expectedFlops = (int64)2 * 1 * 64 * 56 * 56;
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, MultiLayerNetwork)
{
    // Build a small network: Conv -> BatchNorm -> Pooling
    Net net;

    // Conv layer
    {
        LayerParams lp;
        lp.type = "Convolution";
        lp.name = "conv1";
        lp.set("kernel_size", 3);
        lp.set("num_output", 16);
        lp.set("pad", 1);
        lp.set("bias_term", true);

        int wShape[] = {16, 3, 3, 3};
        Mat w(4, wShape, CV_32F);
        randu(w, -1, 1);
        lp.blobs.push_back(w);
        Mat b(1, 16, CV_32F, Scalar(0));
        lp.blobs.push_back(b);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }

    // BatchNorm
    {
        LayerParams lp;
        lp.type = "BatchNorm";
        lp.name = "bn1";
        lp.set("has_weight", true);
        lp.set("has_bias", true);
        lp.set("eps", 1e-5);

        Mat mean(1, 16, CV_32F, Scalar(0));
        Mat var(1, 16, CV_32F, Scalar(1));
        Mat scale(1, 16, CV_32F, Scalar(1));
        Mat shift(1, 16, CV_32F, Scalar(0));
        lp.blobs.push_back(mean);
        lp.blobs.push_back(var);
        lp.blobs.push_back(scale);
        lp.blobs.push_back(shift);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }

    // MaxPool
    {
        LayerParams lp;
        lp.type = "Pooling";
        lp.name = "pool1";
        lp.set("pool", "max");
        lp.set("kernel_size", 2);
        lp.set("stride", 2);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }

    MatShape inputShape{1, 3, 32, 32};
    Mat input(inputShape, CV_32F);
    randu(input, -1, 1);
    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    int64 flops = net.getFLOPS(inputShape, CV_32F);

    // Conv: output [1,16,32,32], flops = 1*16*32*32*(2*3*3*3+1) = 903,168
    int64 convFlops = (int64)1 * 16 * 32 * 32 * (2 * 3 * 3 * 3 + 1);
    // BN: 3 * 1*16*32*32 = 49,152
    int64 bnFlops = (int64)3 * 1 * 16 * 32 * 32;
    // Pool: output [1,16,16,16], karea=4, flops = 1*16*16*16*4 = 16,384
    int64 poolFlops = (int64)1 * 16 * 16 * 16 * 4;

    int64 expectedFlops = convFlops + bnFlops + poolFlops;
    EXPECT_EQ(flops, expectedFlops);
}

TEST(Test_GetFLOPS, EmptyNet)
{
    Net net;
    MatShape inputShape{1, 3, 224, 224};
    // An empty net should not crash
    EXPECT_NO_THROW(net.getFLOPS(inputShape, CV_32F));
}

TEST(Test_GetFLOPS, PerLayerFLOPS)
{
    // Test getFLOPS with specific layerId
    Net net;

    // Conv layer
    {
        LayerParams lp;
        lp.type = "Convolution";
        lp.name = "conv1";
        lp.set("kernel_size", 3);
        lp.set("num_output", 8);
        lp.set("pad", 1);
        lp.set("bias_term", true);

        int wShape[] = {8, 3, 3, 3};
        Mat w(4, wShape, CV_32F);
        randu(w, -1, 1);
        lp.blobs.push_back(w);
        Mat b(1, 8, CV_32F, Scalar(0));
        lp.blobs.push_back(b);
        net.addLayerToPrev(lp.name, lp.type, lp);
    }

    // Softmax
    {
        LayerParams lp;
        lp.type = "Softmax";
        lp.name = "softmax";
        net.addLayerToPrev(lp.name, lp.type, lp);
    }

    MatShape inputShape{1, 3, 16, 16};
    Mat input(inputShape, CV_32F);
    randu(input, -1, 1);
    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    if (!net.getMainGraph()) {
        int convId = net.getLayerId("conv1");
        int64 convFlops = net.getFLOPS(convId, inputShape, CV_32F);
        int64 expectedConvFlops = (int64)1 * 8 * 16 * 16 * (2 * 3 * 3 * 3 + 1);
        EXPECT_EQ(convFlops, expectedConvFlops);

        int softmaxId = net.getLayerId("softmax");
        int64 softmaxFlops = net.getFLOPS(softmaxId, inputShape, CV_32F);
        // Softmax output: [1, 8, 16, 16] => 4 * 8 * 16 * 16
        int64 expectedSoftmaxFlops = (int64)4 * 1 * 8 * 16 * 16;
        EXPECT_EQ(softmaxFlops, expectedSoftmaxFlops);
    }
}


TEST(Test_GetFLOPS, MatMulLayer)
{
    // Test MatMul getFLOPS directly via the layer interface
    LayerParams lp;
    lp.type = "MatMul";
    lp.name = "matmul";
    lp.set("transA", false);
    lp.set("transB", false);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("MatMul", lp);
    ASSERT_TRUE(layer);

    // A=[2,4,8], B=[2,8,16] => output=[2,4,16], K=8
    std::vector<MatShape> inputs = {MatShape{2, 4, 8}, MatShape{2, 8, 16}};
    std::vector<MatShape> outputs = {MatShape{2, 4, 16}};

    int64 flops = layer->getFLOPS(inputs, outputs);

    // batch=2, M=4, N=16, K=8
    // flops = 2 * (2 * 4 * 16 * 8) = 2048
    int64 expected = (int64)2 * (2 * 4 * 16 * 8);
    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, MatMulLayerTranspose)
{
    LayerParams lp;
    lp.type = "MatMul";
    lp.name = "matmul_t";
    lp.set("transA", true);
    lp.set("transB", false);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("MatMul", lp);
    ASSERT_TRUE(layer);

    // transA: A=[2,8,4] => M=4,K=8; B=[2,8,16] => N=16
    std::vector<MatShape> inputs = {MatShape{2, 8, 4}, MatShape{2, 8, 16}};
    std::vector<MatShape> outputs = {MatShape{2, 4, 16}};

    int64 flops = layer->getFLOPS(inputs, outputs);
    int64 expected = (int64)2 * (2 * 4 * 16 * 8);
    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, GemmLayer)
{
    LayerParams lp;
    lp.type = "Gemm";
    lp.name = "gemm";
    lp.set("transA", false);
    lp.set("transB", false);
    lp.set("alpha", 1.0f);
    lp.set("beta", 1.0f);
    lp.set("have_bias", true);

    // B as blob: [128, 64]
    Mat B(128, 64, CV_32F);
    randu(B, -1, 1);
    lp.blobs.push_back(B);
    // C as blob: [1, 64]
    Mat C(1, 64, CV_32F, Scalar(0));
    lp.blobs.push_back(C);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("Gemm", lp);
    ASSERT_TRUE(layer);

    // A=[32, 128], B=[128, 64] => output=[32, 64]
    // M=32, K=128, N=64
    std::vector<MatShape> inputs = {MatShape{32, 128}};
    std::vector<MatShape> outputs = {MatShape{32, 64}};

    int64 flops = layer->getFLOPS(inputs, outputs);

    // 2*M*N*K + M*N (bias) = 2*32*64*128 + 32*64 = 524,288 + 2,048 = 526,336
    int64 expected = (int64)2 * 32 * 64 * 128 + (int64)32 * 64;
    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, AttentionLayer)
{
    LayerParams lp;
    lp.type = "Attention";
    lp.name = "attention";

    int num_heads = 4;
    int D = 32;  // input hidden size
    int hidden = 48;  // total projected size (q + k + v)
    // qkv_hidden_sizes: q=16, k=16, v=16
    int qkv_sizes[] = {16, 16, 16};
    lp.set("num_heads", num_heads);
    lp.set("qkv_hidden_sizes", DictValue::arrayInt(qkv_sizes, 3));

    // Weight blob: [D, hidden] = [32, 48]
    Mat weight(D, hidden, CV_32F);
    randu(weight, -1, 1);
    lp.blobs.push_back(weight);
    // Bias blob: [1, hidden]
    Mat bias(1, hidden, CV_32F, Scalar(0));
    lp.blobs.push_back(bias);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("Attention", lp);
    ASSERT_TRUE(layer);

    int64 B = 2, S = 8;
    int64 q_size = 16, k_size = 16;
    int64 v_size = hidden - q_size - k_size;  // 16
    int64 q_head = q_size / num_heads;  // 4
    int64 v_head = v_size / num_heads;  // 4

    std::vector<MatShape> inputs = {MatShape{(int)B, (int)S, D}};
    std::vector<MatShape> outputs = {MatShape{(int)B, (int)S, (int)(v_head * num_heads)}};

    int64 flops = layer->getFLOPS(inputs, outputs);

    // Input projection: B * S * 2 * D * hidden
    int64 expected = B * S * (CV_BIG_INT(2) * D * hidden);
    // QK^T: B * num_heads * 2 * S * S * q_head
    expected += B * num_heads * CV_BIG_INT(2) * S * S * q_head;
    // Softmax: B * num_heads * 4 * S * S
    expected += B * num_heads * 4 * S * S;
    // Attention * V: B * num_heads * 2 * S * v_head * S
    expected += B * num_heads * CV_BIG_INT(2) * S * v_head * S;

    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, AttentionOnnxAiLayer)
{
    // Test AttentionOnnxAi (multi-head attention with separate Q, K, V inputs)
    LayerParams lp;
    lp.type = "AttentionOnnxAi";
    lp.name = "attn_onnxai";

    int nhq = 4, nhkv = 4;
    lp.set("q_num_heads", nhq);
    lp.set("kv_num_heads", nhkv);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("AttentionOnnxAi", lp);
    ASSERT_TRUE(layer);

    int64 B = 2, Sq = 8, Skv = 8;
    int qk_head = 16, v_head = 16;

    // 4D inputs: [B, num_heads, seq_len, head_dim]
    std::vector<MatShape> inputs = {
        MatShape{(int)B, nhq, (int)Sq, qk_head},   // Q
        MatShape{(int)B, nhkv, (int)Skv, qk_head},  // K
        MatShape{(int)B, nhkv, (int)Skv, v_head}    // V
    };
    std::vector<MatShape> outputs = {MatShape{(int)B, nhq, (int)Sq, v_head}};

    int64 flops = layer->getFLOPS(inputs, outputs);

    // QK^T: B * nhq * 2 * Sq * Skv * qk_head
    int64 expected = B * nhq * CV_BIG_INT(2) * Sq * Skv * qk_head;
    // Softmax: B * nhq * 4 * Sq * Skv
    expected += B * nhq * 4 * Sq * Skv;
    // Attention * V: B * nhq * 2 * Sq * v_head * Skv
    expected += B * nhq * CV_BIG_INT(2) * Sq * v_head * Skv;

    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, EinsumLayer)
{
    // Test Einsum: batch matrix multiply "bij,bjk->bik"
    LayerParams lp;
    lp.type = "Einsum";
    lp.name = "einsum";
    lp.set("equation", "bij,bjk->bik");
    lp.set("inputSize", 2);
    lp.set("outputSize", 1);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("Einsum", lp);
    ASSERT_TRUE(layer);

    // A=[2,4,8], B=[2,8,6] => output=[2,4,6]
    // Indices: b=2, i=4, j=8, k=6
    std::vector<MatShape> inputs = {MatShape{2, 4, 8}, MatShape{2, 8, 6}};
    std::vector<MatShape> outputs = {MatShape{2, 4, 6}};

    int64 flops = layer->getFLOPS(inputs, outputs);

    // totalProduct = product of all subscript dims = 2 * 4 * 8 * 6 = 384
    // flops = 2 * totalProduct = 768
    int64 expected = CV_BIG_INT(2) * 2 * 4 * 8 * 6;
    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, EinsumLayerTranspose)
{
    // Test Einsum: transpose "ij->ji"
    LayerParams lp;
    lp.type = "Einsum";
    lp.name = "einsum_transpose";
    lp.set("equation", "ij->ji");
    lp.set("inputSize", 1);
    lp.set("outputSize", 1);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("Einsum", lp);
    ASSERT_TRUE(layer);

    std::vector<MatShape> inputs = {MatShape{3, 5}};
    std::vector<MatShape> outputs = {MatShape{5, 3}};

    int64 flops = layer->getFLOPS(inputs, outputs);

    // Indices: i=3, j=5, totalProduct = 15, flops = 2 * 15 = 30
    int64 expected = CV_BIG_INT(2) * 3 * 5;
    EXPECT_EQ(flops, expected);
}

TEST(Test_GetFLOPS, ZeroFlopsLayers)
{
    // Layers that should return 0 FLOPS (data movement only)
    std::vector<std::string> zeroFlopsTypes = {"Flatten", "Reshape"};

    for (const auto& typeName : zeroFlopsTypes) {
        LayerParams lp;
        lp.type = typeName;
        lp.name = typeName + "_test";
        if (typeName == "Reshape") {
            int newShape[] = {1, -1};
            lp.set("dim", DictValue::arrayInt(newShape, 2));
        }

        Ptr<Layer> layer = LayerFactory::createLayerInstance(typeName, lp);
        if (!layer) continue;

        std::vector<MatShape> inputs = {MatShape{1, 3, 4, 4}};
        std::vector<MatShape> outputs = {MatShape{1, 48}};

        int64 flops = layer->getFLOPS(inputs, outputs);
        EXPECT_EQ(flops, (int64)0) << "Layer type " << typeName << " should have 0 FLOPS";
    }
}

}} // namespace
