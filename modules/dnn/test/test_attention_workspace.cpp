// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/shape_utils.hpp>
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
    RNG rng(29126);
    rng.fill(weights, RNG::UNIFORM, -.1f, .1f);
    rng.fill(bias, RNG::UNIFORM, -.1f, .1f);
    return params;
}

static Mat runAttentionWithSmallWorkspace(const LayerParams& params, std::vector<Mat>& inputs)
{
    Ptr<Layer> layer = AttentionLayer::create(params);
    std::vector<MatShape> inputShapes, outputShapes, internalShapes;
    for (const Mat& input : inputs)
        inputShapes.push_back(input.shape());
    layer->getMemoryShapes(inputShapes, 1, outputShapes, internalShapes);
    internalShapes[1] = internalShapes[2] = MatShape{1};
    std::vector<Mat> outputs, internals;
    for (const MatShape& shape : outputShapes)
        outputs.emplace_back(shape, CV_32F);
    for (const MatShape& shape : internalShapes)
        internals.emplace_back(shape, CV_32F);
    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);
    return outputs[0];
}

TEST(Layer_AttentionWorkspace, memoryPlan)
{
    for (bool dynamicWeights : {false, true})
        for (bool hasMask : {false, true})
            for (bool rotary : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "dynamicWeights=" << dynamicWeights
                             << ", mask=" << hasMask << ", rotary=" << rotary);
                Mat weights, bias;
                LayerParams params = attentionWorkspaceParams(48, rotary, weights, bias);
                if (!dynamicWeights)
                    params.blobs = {weights, bias};
                Ptr<Layer> layer = AttentionLayer::create(params);
#ifdef HAVE_MLAS
                size_t previous = 0;
#endif
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
#ifdef HAVE_MLAS
                    if (!hasMask)
                    {
                        // Linear workspace growth, including QKV and optional RoPE.
                        EXPECT_LT(elements, size_t(2 * seq * 128));
                        if (previous)
                        {
                            EXPECT_LE(elements, 2 * previous);
                        }
                        previous = elements;
                    }
                    else
#endif
                    {
                        EXPECT_EQ(MatShape({4, seq, seq}), internals[1]);
                        EXPECT_EQ(MatShape({4, seq, 24}), internals[2]);
                    }
                }
            }
}

typedef testing::TestWithParam<tuple<bool, bool, int> > Layer_AttentionWorkspaceFallback;

TEST_P(Layer_AttentionWorkspaceFallback, maskedFallback)
{
    const bool dynamicWeights = get<0>(GetParam()), rotary = get<1>(GetParam());
    const int valueHidden = get<2>(GetParam());
    Mat weights, bias;
    LayerParams params = attentionWorkspaceParams(valueHidden, rotary, weights, bias);
    if (!dynamicWeights)
        params.blobs = {weights, bias};
    for (int seq : {1, 17, 257})
        for (int outputDims : {2, 3})
        {
            SCOPED_TRACE(testing::Message() << "sequence=" << seq << ", outputDims=" << outputDims);
            params.set("output_ndims", outputDims);
            Mat x(MatShape{2, seq, 32}, CV_32F);
            RNG(7544).fill(x, RNG::UNIFORM, -1.f, 1.f);
            std::vector<Mat> inputs{x};
            if (dynamicWeights)
            {
                inputs.push_back(weights);
                inputs.push_back(bias);
            }
            Mat mask(MatShape{seq, seq}, CV_32F, Scalar(0));
            if (seq > 1)
                mask.col(seq - 1).setTo(-10000.f);
            inputs.push_back(mask);

            std::vector<Mat> expected;
            runLayer(AttentionLayer::create(params), inputs, expected);
            // The mask forces the fallback to obtain dense scratch from small buffers.
            Mat actual = runAttentionWithSmallWorkspace(params, inputs);
            ASSERT_EQ(expected[0].shape(), actual.shape());
            EXPECT_EQ(0, std::memcmp(expected[0].data, actual.data, expected[0].total() * expected[0].elemSize()));
        }
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_AttentionWorkspaceFallback, testing::Combine(
    testing::Bool(), testing::Bool(), testing::Values(16, 32, 48)));

TEST(Layer_AttentionWorkspace, changingInputShape)
{
    for (bool rotary : {false, true})
    {
        Mat weights, bias;
        LayerParams params = attentionWorkspaceParams(48, rotary, weights, bias);
        params.blobs = {weights, bias};
        Net net;
        net.addLayerToPrev("attention", "Attention", CV_32F, params);
        for (int seq : {17, 257, 1, 17})
        {
            SCOPED_TRACE(testing::Message() << "sequence=" << seq << ", rotary=" << rotary);
            Mat input(MatShape{2, seq, 32}, CV_32F);
            RNG(7544).fill(input, RNG::UNIFORM, -1.f, 1.f);
            std::vector<Mat> inputs{input};
            std::vector<Mat> expected;
            runLayer(AttentionLayer::create(params), inputs, expected);
            net.setInput(input);
            Mat actual = net.forward();
            ASSERT_EQ(expected[0].shape(), actual.shape());
            EXPECT_EQ(0, std::memcmp(expected[0].data, actual.data, expected[0].total() * expected[0].elemSize()));
        }
    }
}

}} // namespace
