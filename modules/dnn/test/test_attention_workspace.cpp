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

}} // namespace
