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

namespace opencv_test { namespace {

//////////////////////////////////////////////////////////////////////
// BlendLinear

DEF_PARAM_TEST(Sz_Depth_Cn, cv::Size, MatDepth, MatCn);

PERF_TEST_P(Sz_Depth_Cn, BlendLinear,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_32F),
                    CUDA_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat img1(size, type);
    cv::Mat img2(size, type);
    declare.in(img1, img2, WARMUP_RNG);

    const cv::Mat weights1(size, CV_32FC1, cv::Scalar::all(0.5));
    const cv::Mat weights2(size, CV_32FC1, cv::Scalar::all(0.5));

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_img1(img1);
        const cv::cuda::GpuMat d_img2(img2);
        const cv::cuda::GpuMat d_weights1(weights1);
        const cv::cuda::GpuMat d_weights2(weights2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::blendLinear(d_img1, d_img2, d_weights1, d_weights2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

}} // namespace
