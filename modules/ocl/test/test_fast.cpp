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

#include "test_precomp.hpp"

#ifdef HAVE_OPENCL

////////////////////////////////////////////////////////
// FAST

namespace
{
    IMPLEMENT_PARAM_CLASS(FAST_Threshold, int)
    IMPLEMENT_PARAM_CLASS(FAST_NonmaxSupression, bool)
}

PARAM_TEST_CASE(FAST, FAST_Threshold, FAST_NonmaxSupression)
{
    int threshold;
    bool nonmaxSupression;

    virtual void SetUp()
    {
        threshold = GET_PARAM(0);
        nonmaxSupression = GET_PARAM(1);
    }
};

OCL_TEST_P(FAST, Accuracy)
{
    cv::Mat image = readImage("gpu/perf/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::ocl::FAST_OCL fast(threshold);
    fast.nonmaxSupression = nonmaxSupression;

    cv::ocl::oclMat ocl_image = cv::ocl::oclMat(image);

    std::vector<cv::KeyPoint> keypoints;
    fast(ocl_image, cv::ocl::oclMat(), keypoints);

    std::vector<cv::KeyPoint> keypoints_gold;
    cv::FAST(image, keypoints_gold, threshold, nonmaxSupression);

    ASSERT_KEYPOINTS_EQ(keypoints_gold, keypoints);
}

INSTANTIATE_TEST_CASE_P(OCL_Features2D, FAST, testing::Combine(
                        testing::Values(FAST_Threshold(25), FAST_Threshold(50)),
                        testing::Values(FAST_NonmaxSupression(false), FAST_NonmaxSupression(true))));

#endif
