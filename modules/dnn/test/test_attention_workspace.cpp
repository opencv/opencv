// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#ifdef HAVE_CUDA
#include <opencv2/core/cuda.hpp>
#include <chrono>
#include <thread>
#endif
#include <cstring>

namespace opencv_test { namespace {

static LayerParams attentionWorkspaceParams(int valueHidden, bool rotary, Mat& weights, Mat& bias)
{
    const int channels = 32;
    int hidden[] = {channels, channels, valueHidden};
    LayerParams params;
    params.set("num_heads", 2);
    params.set("qkv_hidden_sizes", DictValue::arrayInt(hidden, 3));
    params.set("do_rotary", rotary);
    weights.create(channels, 2 * channels + valueHidden, CV_32F);
    bias.create(MatShape{2 * channels + valueHidden}, CV_32F);
    theRNG().fill(weights, RNG::UNIFORM, -.1f, .1f);
    theRNG().fill(bias, RNG::UNIFORM, -.1f, .1f);
    return params;
}

typedef testing::TestWithParam<tuple<bool, bool, bool> > Layer_AttentionWorkspace;

TEST_P(Layer_AttentionWorkspace, memoryPlan)
{
    const bool dynamicWeights = get<0>(GetParam()), hasMask = get<1>(GetParam()), rotary = get<2>(GetParam());
    Mat weights, bias;
    LayerParams params = attentionWorkspaceParams(48, rotary, weights, bias);
    if (!dynamicWeights)
        params.blobs = {weights, bias};
    Ptr<Layer> layer = AttentionLayer::create(params);
    size_t previous = 0;
    for (int seq : {1024, 2048})
    {
        std::vector<MatShape> inputs{MatShape{2, seq, 32}}, outputs, internals;
        if (dynamicWeights)
        {
            inputs.push_back(weights.shape());
            inputs.push_back(bias.shape());
        }
        if (hasMask)
            inputs.push_back(MatShape{seq, seq});
        layer->getMemoryShapes(inputs, 1, outputs, internals);
        ASSERT_EQ(size_t(rotary ? 4 : 3), internals.size());
        size_t elements = 0;
        for (const MatShape& shape : internals)
            elements += total(shape);
        // Linear workspace growth, including QKV and optional RoPE; the mask is applied tile by tile.
        EXPECT_LT(elements, size_t(2 * seq * 128));
        if (previous)
        {
            EXPECT_LE(elements, 2 * previous);
        }
        previous = elements;
    }
}

typedef testing::TestWithParam<bool> Layer_AttentionWorkspaceShapes;

// the same net runs inputs of different lengths one after another
TEST_P(Layer_AttentionWorkspaceShapes, changingInputShape)
{
    const bool rotary = GetParam();
    Mat weights, bias;
    LayerParams params = attentionWorkspaceParams(48, rotary, weights, bias);
    params.blobs = {weights, bias};
    Net net;
    net.addLayerToPrev("attention", "Attention", CV_32F, params);
    for (int seq : {17, 257, 1, 17})
    {
        SCOPED_TRACE(testing::Message() << "sequence=" << seq);
        Mat input(MatShape{2, seq, 32}, CV_32F);
        theRNG().fill(input, RNG::UNIFORM, -1.f, 1.f);
        std::vector<Mat> inputs{input};
        std::vector<Mat> expected;
        runLayer(AttentionLayer::create(params), inputs, expected);
        net.setInput(input);
        Mat actual = net.forward();
        ASSERT_EQ(expected[0].shape(), actual.shape());
        EXPECT_EQ(0, std::memcmp(expected[0].data, actual.data, expected[0].total() * expected[0].elemSize()));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionWorkspace, testing::Combine(
    testing::Bool(), testing::Bool(), testing::Bool()));
INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionWorkspaceShapes, testing::Bool());

typedef testing::TestWithParam<tuple<Backend, Target> > Layer_AttentionTiledCUDA;
TEST_P(Layer_AttentionTiledCUDA, MultiTileAccuracy)
{
    const Backend backendId = get<0>(GetParam());
    const Target targetId = get<1>(GetParam());
    if (backendId != DNN_BACKEND_CUDA)
        throw SkipTestException("Tests the sequence-tiled CUDA AttentionOp");

    const bool fp16 = targetId == DNN_TARGET_CUDA_FP16;
    const double l1 = fp16 ? 1.5e-2 : 1e-5;
    const double lInf = fp16 ? 6e-2 : 1e-4;

    // S = 520 gives query tiles of 256, 256 and 8 rows; the models fuse into one Attention layer
    const int B = 2, S = 520, H = 4, C = 64;
    Mat X({B, S, C}, CV_32F);
    theRNG().fill(X, RNG::UNIFORM, -0.5f, 0.5f);
    const std::pair<std::string, std::vector<int> > cases[] = {
        {"attention_tiled", {}},
        {"attention_tiled_mask_1x1x1xS", {1, 1, 1, S}},
        {"attention_tiled_mask_1x1xSxS", {1, 1, S, S}},
        {"attention_tiled_mask_Bx1xSxS", {B, 1, S, S}},
        {"attention_tiled_mask_BxHxSxS", {B, H, S, S}},
    };
    for (const auto& c : cases)
    {
        const std::string model = findDataFile("dnn/onnx/models/" + c.first + ".onnx");
        Mat mask;
        if (!c.second.empty())
        {
            mask.create(c.second, CV_32F);
            theRNG().fill(mask, RNG::UNIFORM, -2.f, 0.f);
        }

        Net cpuNet = readNetFromONNX(model);
        cpuNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        cpuNet.setPreferableTarget(DNN_TARGET_CPU);
        cpuNet.setInput(X, "X");
        if (!mask.empty())
            cpuNet.setInput(mask, "M");
        Mat ref = cpuNet.forward().clone();

        Net cudaNet = readNetFromONNX(model);
        cudaNet.setPreferableBackend(backendId);
        cudaNet.setPreferableTarget(targetId);
        cudaNet.setInput(X, "X");
        if (!mask.empty())
            cudaNet.setInput(mask, "M");
        normAssert(ref, cudaNet.forward(), c.first.c_str(), l1, lInf);
    }
}

#ifdef HAVE_CUDA
// The device memory one forward() keeps must stay well below the dense [B*H, S, S] score buffer. Free memory is
// read device-wide, so every trial uses a fresh Net and is dropped if free memory moved for another reason.
TEST_P(Layer_AttentionTiledCUDA, SubquadraticWorkspace)
{
    const Backend backendId = get<0>(GetParam());
    const Target targetId = get<1>(GetParam());
    if (backendId != DNN_BACKEND_CUDA || targetId != DNN_TARGET_CUDA)
        throw SkipTestException("Measures FP32 CUDA device memory");

    const int B = 2, S = 2048, H = 12, C = 192;
    const std::string model = findDataFile("dnn/onnx/models/attention_tiled_s2048.onnx");
    const Mat X({B, S, C}, CV_32F, Scalar(0.01f));
    cv::cuda::DeviceInfo devInfo;
    const int64 MiB = 1 << 20;

    // Free memory, or -1 if another process changed it by more than 1 MiB over 10 reads 0.5 ms apart
    auto stableFree = [&]() -> int64 {
        int64 lo = std::numeric_limits<int64>::max(), hi = 0;
        for (int i = 0; i < 10; i++)
        {
            const int64 f = (int64)devInfo.freeMemory();
            lo = std::min(lo, f);
            hi = std::max(hi, f);
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        return hi - lo <= MiB ? lo : -1;
    };
    // The bytes the Net holds after forward(), or -1 if a reading moved or free memory did not come back
    auto trial = [&]() -> int64 {
        const int64 before = stableFree();
        int64 during;
        {
            Net net = readNetFromONNX(model);
            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);
            net.setInput(X, "X");
            net.forward();
            during = stableFree();
        }
        const int64 after = stableFree();
        if (before < 0 || during < 0 || after < 0 || during > before || std::abs(after - before) > 16 * MiB)
            return -1;
        return before - during;
    };

    trial();  // warm-up: loads the libraries and kernels this model uses
    std::vector<int64> used;
    std::string trials;
    for (int i = 0; i < 5; i++)
    {
        const int64 u = trial();
        trials += u < 0 ? " dropped" : format(" %d", (int)(u / MiB));
        if (u >= 0)
            used.push_back(u);
    }
    std::sort(used.begin(), used.end());
    if (used.size() < 3 || used.back() - used.front() > 24 * MiB)
        throw SkipTestException("device memory moved during the trials (MiB:" + trials + ")");

    const int64 median = used[used.size() / 2];
    RecordProperty("median_mib", (int)(median / MiB));
    EXPECT_LT(median, (int64)B * H * S * S * (int64)sizeof(float) / 2) << "trials in MiB:" << trials;
}
#endif
INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionTiledCUDA, dnnBackendsAndTargets());

}} // namespace
