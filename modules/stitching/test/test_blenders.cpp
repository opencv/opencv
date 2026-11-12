/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(MultiBandBlender, CanBlendTwoImages)
{
    Mat image1 = imread(string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/baboon.png");
    Mat image2 = imread(string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/lena.png");
    ASSERT_EQ(image1.rows, image2.rows); ASSERT_EQ(image1.cols, image2.cols);

    Mat image1s, image2s;
    image1.convertTo(image1s, CV_16S);
    image2.convertTo(image2s, CV_16S);

    Mat mask1(image1s.size(), CV_8U);
    mask1(Rect(0, 0, mask1.cols/2, mask1.rows)).setTo(255);
    mask1(Rect(mask1.cols/2, 0, mask1.cols - mask1.cols/2, mask1.rows)).setTo(0);

    Mat mask2(image2s.size(), CV_8U);
    mask2(Rect(0, 0, mask2.cols/2, mask2.rows)).setTo(0);
    mask2(Rect(mask2.cols/2, 0, mask2.cols - mask2.cols/2, mask2.rows)).setTo(255);

    detail::MultiBandBlender blender(false, 5);

    blender.prepare(Rect(0, 0, max(image1s.cols, image2s.cols), max(image1s.rows, image2s.rows)));
    blender.feed(image1s, mask1, Point(0,0));
    blender.feed(image2s, mask2, Point(0,0));

    Mat result_s, result_mask;
    blender.blend(result_s, result_mask);
    Mat result; result_s.convertTo(result, CV_8U);

    Mat expected = imread(string(cvtest::TS::ptr()->get_data_path()) + "stitching/baboon_lena.png");
    double psnr = cvtest::PSNR(expected, result);
    EXPECT_GE(psnr, 50);
}

TEST(StitchingMemoryHygiene, SeamFindersFeatherBlenderAndBlocksGain)
{
    RNG& rng = theRNG();
    UMat u1(96, 128, CV_32FC3), u2(96, 128, CV_32FC3);
    Mat m1(96, 128, CV_32FC3), m2(96, 128, CV_32FC3);
    rng.fill(m1, RNG::UNIFORM, 10.f, 220.f);
    rng.fill(m2, RNG::UNIFORM, 10.f, 220.f);
    m1.copyTo(u1);
    m2.copyTo(u2);

    std::vector<Point> corners = {Point(0, 0), Point(64, 0)};
    for (int costType : {(int)detail::GraphCutSeamFinder::COST_COLOR, (int)detail::GraphCutSeamFinder::COST_COLOR_GRAD})
    {
        std::vector<UMat> masks = {UMat(96, 128, CV_8U, Scalar(255)), UMat(96, 128, CV_8U, Scalar(255))};
        detail::GraphCutSeamFinder finder(costType);
        finder.find({u1, u2}, corners, masks);
        EXPECT_GT(countNonZero(masks[0]), 0);
        EXPECT_GT(countNonZero(masks[1]), 0);
    }

    {
        std::vector<UMat> masks = {UMat(96, 128, CV_8U, Scalar(255)), UMat(96, 128, CV_8U, Scalar(255))};
        detail::DpSeamFinder dp(detail::DpSeamFinder::COLOR);
        dp.find({u1, u2}, corners, masks);
        EXPECT_GT(countNonZero(masks[0]), 0);
        EXPECT_GT(countNonZero(masks[1]), 0);
    }

    // FeatherBlender repeated reuse + BlocksGainCompensator Mat vs UMat apply exactness
    detail::FeatherBlender fb;
    fb.prepare(Rect(0, 0, 192, 96));
    Mat s1, s2;
    m1.convertTo(s1, CV_16SC3);
    m2.convertTo(s2, CV_16SC3);
    fb.feed(s1, Mat(96, 128, CV_8U, Scalar(255)), Point(0, 0));
    fb.feed(s2, Mat(96, 128, CV_8U, Scalar(255)), Point(64, 0));
    Mat blended, blendedMask;
    fb.blend(blended, blendedMask);
    EXPECT_EQ(blended.size(), Size(192, 96));

    UMat img8_1, img8_2;
    m1.convertTo(img8_1, CV_8UC3);
    m2.convertTo(img8_2, CV_8UC3, 0.8);
    detail::BlocksGainCompensator comp(16, 16);
    comp.feed(corners, {img8_1, img8_2},
              {{UMat(96, 128, CV_8U, Scalar(255)), 255}, {UMat(96, 128, CV_8U, Scalar(255)), 255}});

    Mat outMat = img8_2.getMat(ACCESS_READ).clone();
    UMat outUMat = img8_2.clone();
    comp.apply(1, corners[1], outMat, Mat());
    comp.apply(1, corners[1], outUMat, UMat());
    EXPECT_EQ(0.0, cvtest::norm(outMat, outUMat.getMat(ACCESS_READ), NORM_INF));
}

}} // namespace
