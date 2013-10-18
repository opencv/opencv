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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
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

#ifdef HAVE_OPENCL

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HoughCircles

PARAM_TEST_CASE(HoughCircles, cv::Size)
{
    static void drawCircles(cv::Mat& dst, const std::vector<cv::Vec3f>& circles, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));

        for (size_t i = 0; i < circles.size(); ++i)
            cv::circle(dst, cv::Point2f(circles[i][0], circles[i][1]), (int)circles[i][2], cv::Scalar::all(255), fill ? -1 : 1);
    }
};

OCL_TEST_P(HoughCircles, Accuracy)
{
    const cv::Size size = GET_PARAM(0);

    const float dp = 2.0f;
    const float minDist = 10.0f;
    const int minRadius = 10;
    const int maxRadius = 20;
    const int cannyThreshold = 100;
    const int votesThreshold = 15;

    std::vector<cv::Vec3f> circles_gold(4);
    circles_gold[0] = cv::Vec3i(20, 20, minRadius);
    circles_gold[1] = cv::Vec3i(90, 87, minRadius + 3);
    circles_gold[2] = cv::Vec3i(30, 70, minRadius + 8);
    circles_gold[3] = cv::Vec3i(80, 10, maxRadius);

    cv::Mat src(size, CV_8UC1);
    drawCircles(src, circles_gold, true);
    cv::ocl::oclMat d_src(src);

    cv::ocl::oclMat d_circles;
    cv::ocl::HoughCircles(d_src, d_circles, cv::HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
    ASSERT_TRUE(d_circles.rows > 0);

    cv::Mat circles;
    d_circles.download(circles);

    for (int i = 0; i < circles.cols; ++i)
    {
        cv::Vec3f cur = circles.at<cv::Vec3f>(i);

        bool found = false;

        for (size_t j = 0; j < circles_gold.size(); ++j)
        {
            cv::Vec3f gold = circles_gold[j];

            if (std::fabs(cur[0] - gold[0]) < minDist && std::fabs(cur[1] - gold[1]) < minDist && std::fabs(cur[2] - gold[2]) < minDist)
            {
                found = true;
                break;
            }
        }

        ASSERT_TRUE(found);
    }
}

INSTANTIATE_TEST_CASE_P(Hough, HoughCircles, DIFFERENT_SIZES);

#endif // HAVE_OPENCL
