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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
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
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "perf_precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cv;
using namespace perf;

//////////////////////////////////////////////////////////////////////
// HoughCircles

typedef std::tr1::tuple<cv::Size, float, float> Size_Dp_MinDist_t;
typedef perf::TestBaseWithParam<Size_Dp_MinDist_t> Size_Dp_MinDist;

PERF_TEST_P(Size_Dp_MinDist, OCL_HoughCircles,
            testing::Combine(
                testing::Values(perf::sz720p, perf::szSXGA, perf::sz1080p),
                testing::Values(1.0f, 2.0f, 4.0f),
                testing::Values(1.0f, 10.0f)))
{
    const Size_Dp_MinDist_t params = GetParam();
    const cv::Size size = std::tr1::get<0>(params);
    const float dp      = std::tr1::get<1>(params);
    const float minDist = std::tr1::get<2>(params);

    const int minRadius = 10;
    const int maxRadius = 30;
    const int cannyThreshold = 100;
    const int votesThreshold = 15;

    cv::RNG rng(123456789);

    cv::Mat src(size, CV_8UC1, cv::Scalar::all(0)), circles;

    const int numCircles = rng.uniform(50, 100);
    for (int i = 0; i < numCircles; ++i)
    {
        cv::Point center(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
        const int radius = rng.uniform(minRadius, maxRadius + 1);

        cv::circle(src, center, radius, cv::Scalar::all(255), -1);
    }

    declare.time(10.0).iterations(25);

    if (RUN_OCL_IMPL)
    {
        cv::ocl::oclMat ocl_src(src), ocl_circles;

        OCL_TEST_CYCLE() cv::ocl::HoughCircles(ocl_src, ocl_circles, HOUGH_GRADIENT, dp, minDist,
                                               cannyThreshold, votesThreshold, minRadius, maxRadius);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::HoughCircles(src, circles, HOUGH_GRADIENT, dp, minDist, cannyThreshold,
                                      votesThreshold, minRadius, maxRadius);
    }
    else
        OCL_PERF_ELSE

    int value = 0;
    SANITY_CHECK(value);
}

#endif // HAVE_OPENCL
