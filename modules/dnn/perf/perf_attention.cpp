// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef TestBaseWithParam<tuple<int, int> > Layer_AttentionWorkspace;

PERF_TEST_P(Layer_AttentionWorkspace, forward, testing::Combine(
    testing::Values(257, 1024, 4096), testing::Values(2, 12)))
{
    const int sequence = get<0>(GetParam()), heads = get<1>(GetParam());
    const int channels = heads * 16;
    int hidden[] = {channels, channels, channels};
    LayerParams params;
    params.set("num_heads", heads);
    params.set("qkv_hidden_sizes", DictValue::arrayInt(hidden, 3));

    Mat input({1, sequence, channels}, CV_32F);
    Mat weights(channels, 3 * channels, CV_32F);
    Mat bias(std::vector<int>{3 * channels}, CV_32F);
    RNG rng(29126);
    rng.fill(input, RNG::UNIFORM, -1.f, 1.f);
    rng.fill(weights, RNG::UNIFORM, -.1f, .1f);
    rng.fill(bias, RNG::UNIFORM, -.1f, .1f);
    params.blobs = {weights, bias};

    Net net;
    net.addLayerToPrev("attention", "Attention", CV_32F, params);
    net.setInput(input);
    Mat output = net.forward();

    TEST_CYCLE()
    {
        output = net.forward();
    }
    SANITY_CHECK(output, 1e-5, ERROR_ABSOLUTE);
}

} // namespace opencv_test
