// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

#include <cmath>

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
    // Check a few query rows against a double-precision reference outside the
    // timed region, without requiring platform-specific regression data.
    Mat input64, weights64, bias64, qkv;
    input.reshape(1, sequence).convertTo(input64, CV_64F);
    weights.convertTo(weights64, CV_64F);
    bias.convertTo(bias64, CV_64F);
    gemm(input64, weights64, 1., noArray(), 0., qkv);
    for (int row = 0; row < sequence; ++row)
        for (int col = 0; col < 3 * channels; ++col)
            qkv.at<double>(row, col) += bias64.ptr<double>()[col];

    ASSERT_TRUE(checkRange(output));
    const int queries[] = {0, sequence / 2, sequence - 1};
    for (int query : queries)
        for (int head = 0; head < heads; ++head)
        {
            const int headSize = channels / heads;
            const int offset = head * headSize;
            const double* q = qkv.ptr<double>(query) + offset;
            std::vector<double> scores(sequence);
            double maximum = -DBL_MAX;
            for (int row = 0; row < sequence; ++row)
            {
                const double* k = qkv.ptr<double>(row) + channels + offset;
                double score = 0.;
                for (int col = 0; col < headSize; ++col)
                    score += q[col] * k[col];
                scores[row] = score / std::sqrt(double(headSize));
                maximum = std::max(maximum, scores[row]);
            }
            double denominator = 0.;
            for (double& score : scores)
            {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int col = 0; col < headSize; ++col)
            {
                double expected = 0.;
                for (int row = 0; row < sequence; ++row)
                    expected += scores[row] *
                        qkv.at<double>(row, 2 * channels + offset + col);
                ASSERT_NEAR(expected / denominator,
                            output.ptr<float>()[query * channels + offset + col], 1e-5);
            }
        }
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
