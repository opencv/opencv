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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "perf_precomp.hpp"

using namespace std;
using namespace testing;
using namespace perf;

//////////////////////////////////////////////////////////////////////
// Canny

DEF_PARAM_TEST(Image_AppertureSz_L2gradient, string, int, bool);

PERF_TEST_P(Image_AppertureSz_L2gradient, Canny,
            Combine(Values("perf/800x600.png", "perf/1280x1024.png", "perf/1680x1050.png"),
                    Values(3, 5),
                    Bool()))
{
    const string fileName = GET_PARAM(0);
    const int apperture_size = GET_PARAM(1);
    const bool useL2gradient = GET_PARAM(2);

    const cv::Mat image = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    const double low_thresh = 50.0;
    const double high_thresh = 100.0;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_image(image);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, useL2gradient);

        TEST_CYCLE() canny->detect(d_image, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::Canny(image, dst, low_thresh, high_thresh, apperture_size, useL2gradient);

        CPU_SANITY_CHECK(dst);
    }
}
