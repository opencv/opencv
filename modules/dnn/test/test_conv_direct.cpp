// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>

namespace opencv_test { namespace {

/** @brief Scalar NCHWc convolution reference, computed in parallel over (batch, output channel) planes. */
static void blockedConvReference(const Mat& input, const Mat& weights, const Mat& bias, int K,
                                 int groups, int stride, int dilation, int pad, Mat& expected)
{
    const MatShape si = input.shape(), so = expected.shape();
    const int N = si[0], C1 = si[1], H = si[2], W = si[3], C0 = si[4];
    const int K1 = so[1], Ho = so[2], Wo = so[3];
    const int Cg = weights.size[1], kernel = weights.size[2], Kg = K / groups;
    const float* inp = input.ptr<float>();
    const float* wptr = weights.ptr<float>();
    const float* bptr = bias.ptr<float>();
    float* out = expected.ptr<float>();
    parallel_for_(Range(0, N * K), [&](const Range& range) {
        for (int nk = range.start; nk < range.end; ++nk)
        {
            const int n = nk / K, k = nk % K;
            for (int y = 0; y < Ho; ++y)
                for (int x = 0; x < Wo; ++x)
                {
                    float sum = bptr[k];
                    for (int c = 0; c < Cg; ++c)
                        for (int ky = 0; ky < kernel; ++ky)
                            for (int kx = 0; kx < kernel; ++kx)
                            {
                                const int iy = y * stride + ky * dilation - pad;
                                const int ix = x * stride + kx * dilation - pad;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                    continue;
                                const int ic = (k / Kg) * Cg + c;
                                const float v = inp[((n * C1 + ic / C0) * H * W + iy * W + ix) * C0 + ic % C0];
                                sum += v * wptr[((k * Cg + c) * kernel + ky) * kernel + kx];
                            }
                    out[((n * K1 + k / C0) * Ho * Wo + y * Wo + x) * C0 + k % C0] = sum;
                }
        }
    });
}

/** @brief Runs Conv2 on NCHWc8 data against the scalar reference across shape, weight and bias changes. */
static void testBlockedConv2(int kernel, int stride, int dilation, int groups, int channels, bool winograd)
{
    const bool aligned = channels % 8 == 0;
    const int C = groups == 1 ? channels : 18;
    const int K = groups == 1 ? (aligned ? 64 : 67) : 22;
    const int Cg = C / groups;
    const int C0 = 8, C1 = (C + C0 - 1) / C0, K1 = (K + C0 - 1) / C0;
    const int pad = kernel == 1 || aligned ? 0 : 1;
    LayerParams lp;
    int strides[] = {stride, stride}, dilations[] = {dilation, dilation};
    int pads[] = {pad, pad, pad, pad};
    lp.set("stride", DictValue::arrayInt(strides, 2));
    lp.set("dilation", DictValue::arrayInt(dilations, 2));
    lp.set("pad", DictValue::arrayInt(pads, 4));
    lp.set("group", groups);
    Ptr<cv::dnn::Conv2Layer> layer = cv::dnn::Conv2Layer::create(lp);
    Net net;
    net.enableWinograd(winograd);
    layer->netimpl = net.getImpl();
    Mat weights({K, Cg, kernel, kernel}, CV_32F), bias({K}, CV_32F);
    for (size_t i = 0; i < weights.total(); ++i)
        weights.ptr<float>()[i] = (int(i * 7 % 11) - 5) * 0.125f;
    for (int k = 0; k < K; ++k)
        bias.ptr<float>()[k] = (k % 7 - 3) * 0.25f;
    layer->setWeights(weights, bias, C0, CV_32F);

    // Small and resized spatial dimensions cover scalar-only output, full SIMD
    // tiles, tails crossing rows, and the interior versus padded border paths.
    for (int width : {5, 17, 23, 23})
    {
        if (width == 23)
        {
            weights *= 0.5f;
            layer->setWeights(weights, bias, C0, CV_32F);
        }
        const int N = 2, H = winograd ? 13 : 7, W = width;
        const int Ho = (H + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        const int Wo = (W + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        Mat input(MatShape({N, C1, H, W, C0}, DATA_LAYOUT_BLOCK, C), CV_32F, Scalar::all(0));
        Mat expected(MatShape({N, K1, Ho, Wo, C0}, DATA_LAYOUT_BLOCK, K), CV_32F, Scalar::all(0));
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int p = 0; p < H * W; ++p)
                    input.ptr<float>()[((n * C1 + c / C0) * H * W + p) * C0 + c % C0] =
                        ((n * 3 + c * 13 + p * 7) % 17 - 8) * 0.125f;
        blockedConvReference(input, weights, bias, K, groups, stride, dilation, pad, expected);
        std::vector<Mat> inputs(1, input), outputs, internals;
        layer->forward(inputs, outputs, internals);
        ASSERT_EQ(1u, outputs.size());
        // Binary fractions keep the scalar reference exact in FP32.
        normAssert(expected, outputs[0], "Blocked convolution", winograd ? 1e-4 : 0., winograd ? 1e-3 : 0.);
        layer->forward(inputs, outputs, internals);
        normAssert(expected, outputs[0], "Repeated convolution", winograd ? 1e-4 : 0., winograd ? 1e-3 : 0.);
        if (winograd && width == 23)
        {
            // A late bias fusion must update the cached Winograd object too.
            Mat extraBias({K}, CV_32F, Scalar::all(0.125));
            ASSERT_TRUE(layer->fuseAddBias(extraBias));
            bias += 0.125;
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    for (int p = 0; p < Ho * Wo; ++p)
                        expected.ptr<float>()[((n * K1 + k / C0) * Ho * Wo + p) * C0 + k % C0] += 0.125f;
            layer->forward(inputs, outputs, internals);
            normAssert(expected, outputs[0], "Bias after cached forward", 1e-4, 1e-3);
        }
        if (winograd)
        {
            net.enableWinograd(false);
            layer->forward(inputs, outputs, internals);
            normAssert(expected, outputs[0], "Winograd disabled", 0., 0.);
            net.enableWinograd(true);
        }
    }
}

// Direct kernel, including groups that start inside a channel block.
typedef TestWithParam<tuple<int, int, int, int, int> > Conv2Direct;
TEST_P(Conv2Direct, Accuracy)
{
    testBlockedConv2(get<0>(GetParam()), get<1>(GetParam()), get<2>(GetParam()),
                     get<3>(GetParam()), get<4>(GetParam()), false);
}

INSTANTIATE_TEST_CASE_P(DNN, Conv2Direct, Combine(
    Values(1, 3), Values(1, 2), Values(1, 2), Values(1, 2), Values(3, 65, 512)));

// Blocked Winograd: partial tiles, several tile chunks and padded channel blocks.
typedef TestWithParam<int> Conv2Winograd;
TEST_P(Conv2Winograd, Accuracy)
{
    testBlockedConv2(3, 1, 1, 1, GetParam(), true);
}

INSTANTIATE_TEST_CASE_P(DNN, Conv2Winograd, Values(3, 65, 512));

}} // namespace opencv_test
