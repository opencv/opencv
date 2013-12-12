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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Peter Andreas Entschev, peter@entschev.com
//
//M*/

#include "perf_precomp.hpp"

using namespace perf;

///////////// FAST ////////////////////////

typedef std::tr1::tuple<std::string, int, bool> Image_Threshold_NonmaxSupression_t;
typedef perf::TestBaseWithParam<Image_Threshold_NonmaxSupression_t> Image_Threshold_NonmaxSupression;

PERF_TEST_P(Image_Threshold_NonmaxSupression, FAST,
            testing::Combine(testing::Values<string>("gpu/perf/aloe.png"),
                    testing::Values(20),
                    testing::Bool()))
{
    const Image_Threshold_NonmaxSupression_t params = GetParam();
    const std::string imgFile = std::tr1::get<0>(params);
    const int threshold = std::tr1::get<1>(params);
    const bool nonmaxSupression = std::tr1::get<2>(params);

    const cv::Mat img = imread(getDataPath(imgFile), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (RUN_OCL_IMPL)
    {
        cv::ocl::FAST_OCL fast(threshold, nonmaxSupression, 0.5);

        cv::ocl::oclMat d_img(img);
        cv::ocl::oclMat d_keypoints;

        OCL_TEST_CYCLE() fast(d_img, cv::ocl::oclMat(), d_keypoints);

        std::vector<cv::KeyPoint> ocl_keypoints;
        fast.downloadKeypoints(d_keypoints, ocl_keypoints);

        sortKeyPoints(ocl_keypoints);

        SANITY_CHECK_KEYPOINTS(ocl_keypoints);
    }
    else if (RUN_PLAIN_IMPL)
    {
        std::vector<cv::KeyPoint> cpu_keypoints;

        TEST_CYCLE() cv::FAST(img, cpu_keypoints, threshold, nonmaxSupression);

        SANITY_CHECK_KEYPOINTS(cpu_keypoints);
    }
    else
        OCL_PERF_ELSE;
}
