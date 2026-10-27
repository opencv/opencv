// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>

namespace opencv_test { namespace {

// Exercise the blocked FP32 kernel with a scalar reference independent of the
// SIMD tile size, including groups starting inside a channel block, and the
// blocked Winograd path with partial tiles and padded channel blocks.
typedef TestWithParam<tuple<int, int, int, int, int, bool> > Conv2Direct;
TEST_P(Conv2Direct, Accuracy)
{
    const int kernel = get<0>(GetParam()), stride = get<1>(GetParam());
    const int dilation = get<2>(GetParam()), groups = get<3>(GetParam());
    const int channels = get<4>(GetParam());
    const bool aligned = channels % 8 == 0;
    const bool winograd = get<5>(GetParam());
    const int C = groups == 1 ? channels : 18;
    const int K = groups == 1 ? (aligned ? 64 : 67) : 22;
    const int Cg = C / groups, Kg = K / groups;
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
    for (int width : {5, 17, 23, 23, 155})
    {
        // Cross the spatial chunk boundary without multiplying fallback cases.
        if (width == 155 && !(winograd && kernel == 3 && stride == 1 && dilation == 1 && groups == 1))
            continue;
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
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                for (int y = 0; y < Ho; ++y)
                    for (int x = 0; x < Wo; ++x)
                    {
                        float sum = bias.ptr<float>()[k];
                        for (int c = 0; c < Cg; ++c)
                            for (int ky = 0; ky < kernel; ++ky)
                                for (int kx = 0; kx < kernel; ++kx)
                                {
                                    const int iy = y * stride + ky * dilation - pad;
                                    const int ix = x * stride + kx * dilation - pad;
                                    if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                        continue;
                                    const int ic = (k / Kg) * Cg + c;
                                    const float v = input.ptr<float>()[((n * C1 + ic / C0) * H * W + iy * W + ix) * C0 + ic % C0];
                                    sum += v * weights.ptr<float>()[((k * Cg + c) * kernel + ky) * kernel + kx];
                                }
                        expected.ptr<float>()[((n * K1 + k / C0) * Ho * Wo + y * Wo + x) * C0 + k % C0] = sum;
                    }
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

INSTANTIATE_TEST_CASE_P(DNN, Conv2Direct, Combine(
    Values(1, 3), Values(1, 2), Values(1, 2), Values(1, 2), Values(3, 65, 512), testing::Bool()));

}} // namespace opencv_test
