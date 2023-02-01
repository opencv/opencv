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
// MeanShiftFiltering

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, MeanShiftFiltering,
            Values<string>("gpu/meanshift/cones.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam());
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    const int sp = 50;
    const int sr = 50;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(rgba);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::meanShiftFiltering(d_src, dst, sp, sr);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::pyrMeanShiftFiltering(img, dst, sp, sr);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MeanShiftProc

PERF_TEST_P(Image, MeanShiftProc,
            Values<string>("gpu/meanshift/cones.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam());
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    const int sp = 50;
    const int sr = 50;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(rgba);
        cv::cuda::GpuMat dstr;
        cv::cuda::GpuMat dstsp;

        TEST_CYCLE() cv::cuda::meanShiftProc(d_src, dstr, dstsp, sp, sr);

        CUDA_SANITY_CHECK(dstr);
        CUDA_SANITY_CHECK(dstsp);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// MeanShiftSegmentation

PERF_TEST_P(Image, MeanShiftSegmentation,
            Values<string>("gpu/meanshift/cones.png"))
{
    declare.time(300.0);

    const cv::Mat img = readImage(GetParam());
    ASSERT_FALSE(img.empty());

    cv::Mat rgba;
    cv::cvtColor(img, rgba, cv::COLOR_BGR2BGRA);

    const int sp = 10;
    const int sr = 10;
    const int minsize = 20;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(rgba);
        cv::Mat dst;

        TEST_CYCLE() cv::cuda::meanShiftSegmentation(d_src, dst, sp, sr, minsize);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

}} // namespace
