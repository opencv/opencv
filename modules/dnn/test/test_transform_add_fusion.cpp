// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#include "test_transform_add_fusion.inc.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;

// ConvTranspose2d(8->16, C0=8) -> Add(residual): the Add should fuse into
// TransformLayout, matching the unfused two-step result.
TEST(Layer_TransformLayoutAddFusion, FusedMatchesUnfusedComputation)
{
    Net net = readNetFromONNX((const char*)g_deconvAddOnnx, g_deconvAddOnnx_len);
    ASSERT_FALSE(net.empty());

    std::vector<std::string> preTypes;
    net.getLayerTypes(preTypes);
    ASSERT_NE(std::find(preTypes.begin(), preTypes.end(), std::string("NaryEltwise")), preTypes.end())
        << "test model must start with a separate NaryEltwise layer for this test to mean anything";

    RNG rng(12345);
    Mat input(std::vector<int>{1, 8, 6, 6}, CV_32F);
    rng.fill(input, RNG::UNIFORM, -1.0, 1.0);
    Mat residual(std::vector<int>{1, 16, 12, 12}, CV_32F);
    rng.fill(residual, RNG::UNIFORM, -1.0, 1.0);

    net.setInput(input, "input");
    net.setInput(residual, "residual");

    // net.forward(out, name) hands back a plain, already-converted tensor (not
    // the raw block-layout buffer), so it's directly comparable to the final result.
    Mat deconvOut;
    net.forward(deconvOut, "/deconv/ConvTranspose_output_0");

    Mat fusedOut = net.forward();

    // The fusion should have run: no NaryEltwise left in the graph.
    std::vector<std::string> postTypes;
    net.getLayerTypes(postTypes);
    EXPECT_EQ(std::find(postTypes.begin(), postTypes.end(), std::string("NaryEltwise")), postTypes.end())
        << "NaryEltwise should have been absorbed into TransformLayout by the fusion pass";

    // Unfused reference: the same two values added as a separate step, instead of
    // in one fused TransformLayout(+residual) pass.
    Mat unfusedOut;
    cv::add(deconvOut, residual, unfusedOut);

    normAssert(unfusedOut, fusedOut, "fused vs unfused TransformLayout+Add", 1e-5, 1e-5);
}

// ConvTranspose2d(8->12): C0=8 doesn't divide 12, so the last block is partial.
TEST(Layer_TransformLayoutAddFusion, PartialChannelBlockMatchesUnfused)
{
    Net net = readNetFromONNX((const char*)g_deconvAddPartialOnnx, g_deconvAddPartialOnnx_len);
    ASSERT_FALSE(net.empty());

    RNG rng(6789);
    Mat input(std::vector<int>{1, 8, 6, 6}, CV_32F);
    rng.fill(input, RNG::UNIFORM, -1.0, 1.0);
    Mat residual(std::vector<int>{1, 12, 12, 12}, CV_32F);
    rng.fill(residual, RNG::UNIFORM, -1.0, 1.0);

    net.setInput(input, "input");
    net.setInput(residual, "residual");

    Mat deconvOut;
    net.forward(deconvOut, "/deconv/ConvTranspose_output_0");
    Mat fusedOut = net.forward();

    std::vector<std::string> postTypes;
    net.getLayerTypes(postTypes);
    EXPECT_EQ(std::find(postTypes.begin(), postTypes.end(), std::string("NaryEltwise")), postTypes.end())
        << "NaryEltwise should have been absorbed into TransformLayout by the fusion pass";

    Mat unfusedOut;
    cv::add(deconvOut, residual, unfusedOut);

    normAssert(unfusedOut, fusedOut, "fused vs unfused, partial channel block", 1e-5, 1e-5);
}

}} // namespace
