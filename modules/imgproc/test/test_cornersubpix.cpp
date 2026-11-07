/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015-2023, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

TEST(Imgproc_CornerSubPix, out_of_image_corners)
{
    const uint8_t image_pixels[] = {
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 3};

    cv::Mat image(cv::Size(7, 7), CV_8UC1, (void*)image_pixels, cv::Mat::AUTO_STEP);
    std::vector<cv::Point2f> corners = {cv::Point2f(5.25, 6.5)};
    cv::Size win(1, 1);
    cv::Size zeroZone(-1, -1);
    cv::TermCriteria criteria;
    cv::cornerSubPix(image, corners, win, zeroZone, criteria);

    ASSERT_EQ(corners.size(), 1u);
    ASSERT_TRUE(Rect(0, 0, image.cols, image.rows).contains(corners.front()));
}

// See https://github.com/opencv/opencv/issues/26016
TEST(Imgproc_CornerSubPix, corners_on_the_edge)
{
    cv::Mat image(500, 500, CV_8UC1);
    RNG& rng = TS::ptr()->get_rng();
    cvtest::randUni(rng, image, 0, 255);
    cv::Size win(1, 1);
    cv::Size zeroZone(-1, -1);
    cv::TermCriteria criteria;

    std::vector<cv::Point2f> cornersOK1 = { cv::Point2f(250, std::nextafter(499.5f, 499.5f - 1.0f)) };
    EXPECT_NO_THROW( cv::cornerSubPix(image, cornersOK1, win, zeroZone, criteria) ) << cornersOK1;

    std::vector<cv::Point2f> cornersOK2 = { cv::Point2f(250, 499.5f) };
    EXPECT_NO_THROW( cv::cornerSubPix(image, cornersOK2, win, zeroZone, criteria) ) << cornersOK2;

    std::vector<cv::Point2f> cornersOK3 = { cv::Point2f(250, std::nextafter(499.5f, 499.5f + 1.0f)) };
    EXPECT_NO_THROW( cv::cornerSubPix(image, cornersOK3, win, zeroZone, criteria) ) << cornersOK3;

    std::vector<cv::Point2f> cornersOK4 = { cv::Point2f(250, std::nextafter(500.0f, 500.0f - 1.0f)) };
    EXPECT_NO_THROW( cv::cornerSubPix(image, cornersOK4, win, zeroZone, criteria) ) << cornersOK4;

    std::vector<cv::Point2f> cornersNG1 = { cv::Point2f(250, 500.0f) };
    EXPECT_ANY_THROW( cv::cornerSubPix(image, cornersNG1, win, zeroZone, criteria) ) << cornersNG1;

    std::vector<cv::Point2f> cornersNG2 = { cv::Point2f(250, std::nextafter(500.0f, 500.0f + 1.0f)) };
    EXPECT_ANY_THROW( cv::cornerSubPix(image, cornersNG2, win, zeroZone, criteria) ) << cornersNG2;
}

typedef testing::TestWithParam<int> Imgproc_CornerSubPix_Parallel;

TEST_P(Imgproc_CornerSubPix_Parallel, many_corners_roi_single_vs_multi_thread)
{
    const int depth = GetParam();
    RNG& rng = TS::ptr()->get_rng();

    cv::Mat noise(480, 640, CV_8UC1), full;
    cvtest::randUni(rng, noise, 0, 255);
    cv::GaussianBlur(noise, full, cv::Size(7, 7), 2.0);
    if (depth != CV_8U)
        full.convertTo(full, depth);

    const cv::Rect roi(37, 23, 401, 317);
    cv::Mat image = full(roi);
    ASSERT_FALSE(image.isContinuous());

    std::vector<cv::Point2f> initial;
    for (int y = 8; y < image.rows - 8; y += 36)
        for (int x = 8; x < image.cols - 8; x += 36)
            initial.push_back(cv::Point2f(x + rng.uniform(-0.5f, 0.5f), y + rng.uniform(-0.5f, 0.5f)));
    ASSERT_GE(initial.size(), 50u);

    const cv::Size win(5, 5), zeroZone(-1, -1);
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);

    const int nthreads = cv::getNumThreads();
    std::vector<cv::Point2f> ref = initial, dst = initial;

    cv::setNumThreads(1);
    cv::cornerSubPix(image, ref, win, zeroZone, criteria);
    cv::setNumThreads(std::max(2, nthreads));
    cv::cornerSubPix(image, dst, win, zeroZone, criteria);

    ASSERT_EQ(ref.size(), dst.size());
    int moved = 0;
    for (size_t i = 0; i < ref.size(); i++)
    {
        if (ref[i] != initial[i])
            moved++;
        EXPECT_EQ(ref[i].x, dst[i].x) << "i=" << i << " initial=" << initial[i];
        EXPECT_EQ(ref[i].y, dst[i].y) << "i=" << i << " initial=" << initial[i];
    }
    EXPECT_GE(moved, (int)ref.size() / 4);
}

INSTANTIATE_TEST_CASE_P(/**/, Imgproc_CornerSubPix_Parallel, testing::Values(CV_8U, CV_32F));

}} // namespace
