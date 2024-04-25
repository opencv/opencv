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

BIGDATA_TEST(Imgproc_Threshold, huge)
{
    Mat m(65000, 40000, CV_8U);
    ASSERT_FALSE(m.isContinuous());

    uint64 i, n = (uint64)m.rows*m.cols;
    for( i = 0; i < n; i++ )
        m.data[i] = (uchar)(i & 255);

    cv::threshold(m, m, 127, 255, cv::THRESH_BINARY);
    int nz = cv::countNonZero(m);  // FIXIT 'int' is not enough here (overflow is possible with other inputs)
    ASSERT_EQ((uint64)nz, n / 2);
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_16085)
{
    Size sz(16, 16);
    Mat input(sz, CV_32F, Scalar::all(2));
    Mat result;
    cv::threshold(input, result, 2.0, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_21258)
{
    Size sz(16, 16);
    float val = nextafterf(16.0f, 0.0f);  // 0x417fffff, all bits in mantissa are 1
    Mat input(sz, CV_32F, Scalar::all(val));
    Mat result;
    cv::threshold(input, result, val, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_21258_Min)
{
    Size sz(16, 16);
    float min_val = -std::numeric_limits<float>::max();
    Mat input(sz, CV_32F, Scalar::all(min_val));
    Mat result;
    cv::threshold(input, result, min_val, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_Threshold, regression_THRESH_TOZERO_IPP_21258_Max)
{
    Size sz(16, 16);
    float max_val = std::numeric_limits<float>::max();
    Mat input(sz, CV_32F, Scalar::all(max_val));
    Mat result;
    cv::threshold(input, result, max_val, 0.0, THRESH_TOZERO);
    EXPECT_EQ(0, cv::norm(result, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, mean)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 8);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold1.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, mean_inv)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 8);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold1.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    gt = Mat(gt.rows, gt.cols, CV_8UC1, cv::Scalar(255)) - gt;
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, gauss)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 200, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, -5);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold2.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

TEST(Imgproc_AdaptiveThreshold, gauss_inv)
{
    const string input_path = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat input = imread(input_path, IMREAD_GRAYSCALE);
    Mat result;

    cv::adaptiveThreshold(input, result, 200, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, -5);

    const string gt_path = cvtest::findDataFile("../cv/imgproc/adaptive_threshold2.png");
    Mat gt = imread(gt_path, IMREAD_GRAYSCALE);
    gt = Mat(gt.rows, gt.cols, CV_8UC1, cv::Scalar(200)) - gt;
    EXPECT_EQ(0, cv::norm(result, gt, NORM_INF));
}

}} // namespace
