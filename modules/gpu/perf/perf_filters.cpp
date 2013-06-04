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
// Blur

DEF_PARAM_TEST(Sz_Type_KernelSz, cv::Size, MatType, int);

PERF_TEST_P(Sz_Type_KernelSz, Filters_Blur,
            Combine(GPU_TYPICAL_MAT_SIZES,
                    Values(CV_8UC1, CV_8UC4),
                    Values(3, 5, 7)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::blur(d_src, dst, cv::Size(ksize, ksize));

        GPU_SANITY_CHECK(dst, 1);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::blur(src, dst, cv::Size(ksize, ksize));

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Sobel

PERF_TEST_P(Sz_Type_KernelSz, Filters_Sobel, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1), Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::Sobel(d_src, dst, -1, 1, 1, d_buf, ksize);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::Sobel(src, dst, -1, 1, 1, ksize);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Scharr

PERF_TEST_P(Sz_Type, Filters_Scharr, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::Scharr(d_src, dst, -1, 1, 0, d_buf);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::Scharr(src, dst, -1, 1, 0);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// GaussianBlur

PERF_TEST_P(Sz_Type_KernelSz, Filters_GaussianBlur, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1), Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::GaussianBlur(d_src, dst, cv::Size(ksize, ksize), d_buf, 0.5);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 0.5);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Laplacian

PERF_TEST_P(Sz_Type_KernelSz, Filters_Laplacian, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(1, 3)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::Laplacian(d_src, dst, -1, ksize);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::Laplacian(src, dst, -1, ksize);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Erode

PERF_TEST_P(Sz_Type, Filters_Erode, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::erode(d_src, dst, ker, d_buf);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::erode(src, dst, ker);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Dilate

PERF_TEST_P(Sz_Type, Filters_Dilate, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::dilate(d_src, dst, ker, d_buf);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::dilate(src, dst, ker);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MorphologyEx

CV_ENUM(MorphOp, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT)

DEF_PARAM_TEST(Sz_Type_Op, cv::Size, MatType, MorphOp);

PERF_TEST_P(Sz_Type_Op, Filters_MorphologyEx, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4), MorphOp::all()))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int morphOp = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;
        cv::gpu::GpuMat d_buf1;
        cv::gpu::GpuMat d_buf2;

        TEST_CYCLE() cv::gpu::morphologyEx(d_src, dst, morphOp, ker, d_buf1, d_buf2);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::morphologyEx(src, dst, morphOp, ker);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Filter2D

PERF_TEST_P(Sz_Type_KernelSz, Filters_Filter2D, Combine(GPU_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Mat kernel(ksize, ksize, CV_32FC1);
    declare.in(kernel, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_src(src);
        cv::gpu::GpuMat dst;

        TEST_CYCLE() cv::gpu::filter2D(d_src, dst, -1, kernel);

        GPU_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::filter2D(src, dst, -1, kernel);

        CPU_SANITY_CHECK(dst);
    }
}
