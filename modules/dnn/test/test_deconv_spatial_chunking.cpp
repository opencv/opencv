// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#include "test_deconv_spatial_chunking.inc.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;

// Model: ConvTranspose2d(4->8), out_channels == C0 (so NK1 == N*K1 == 1) and a
// spatial output of 100 elements -- computeSpatChunks() always splits this
// regardless of thread count, so the new spatial decomposition in deconvBlock32f()
// is guaranteed to run, not just the pre-existing single-chunk path.
TEST(Layer_DeconvolutionSpatialChunking, NarrowOutputWideSpatialMatchesReference)
{
    Net net = readNetFromONNX((const char*)g_deconvNarrowOnnx, g_deconvNarrowOnnx_len);
    ASSERT_FALSE(net.empty());

    Mat input(std::vector<int>{1, 4, 5, 5}, CV_32F, (void*)g_deconvNarrowInput);
    Mat expected(std::vector<int>{1, 8, 10, 10}, CV_32F, (void*)g_deconvNarrowExpected);

    net.setInput(input);
    Mat out = net.forward();

    normAssert(expected, out, "deconv output with nSpatChunks > 1", 1e-4, 1e-4);
}

}} // namespace
