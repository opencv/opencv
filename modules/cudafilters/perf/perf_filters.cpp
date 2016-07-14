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

PERF_TEST_P(Sz_Type_KernelSz, Blur,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8UC1, CV_8UC4, CV_32FC1),
                    Values(3, 5, 7)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> blurFilter = cv::cuda::createBoxFilter(d_src.type(), -1, cv::Size(ksize, ksize));

        TEST_CYCLE() blurFilter->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst, 1);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::blur(src, dst, cv::Size(ksize, ksize));

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Filter2D

PERF_TEST_P(Sz_Type_KernelSz, Filter2D, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Mat kernel(ksize, ksize, CV_32FC1);
    declare.in(kernel, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> filter2D = cv::cuda::createLinearFilter(d_src.type(), -1, kernel);

        TEST_CYCLE() filter2D->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::filter2D(src, dst, -1, kernel);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Laplacian

PERF_TEST_P(Sz_Type_KernelSz, Laplacian, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4), Values(1, 3)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> laplacian = cv::cuda::createLaplacianFilter(d_src.type(), -1, ksize);

        TEST_CYCLE() laplacian->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::Laplacian(src, dst, -1, ksize);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Sobel

PERF_TEST_P(Sz_Type_KernelSz, Sobel, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1), Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> sobel = cv::cuda::createSobelFilter(d_src.type(), -1, 1, 1, ksize);

        TEST_CYCLE() sobel->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
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

PERF_TEST_P(Sz_Type, Scharr, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> scharr = cv::cuda::createScharrFilter(d_src.type(), -1, 1, 0);

        TEST_CYCLE() scharr->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
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

PERF_TEST_P(Sz_Type_KernelSz, GaussianBlur, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4, CV_32FC1), Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int ksize = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(d_src.type(), -1, cv::Size(ksize, ksize), 0.5);

        TEST_CYCLE() gauss->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), 0.5);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Erode

PERF_TEST_P(Sz_Type, Erode, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, src.type(), ker);

        TEST_CYCLE() erode->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
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

PERF_TEST_P(Sz_Type, Dilate, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, src.type(), ker);

        TEST_CYCLE() dilate->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
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

PERF_TEST_P(Sz_Type_Op, MorphologyEx, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1, CV_8UC4), MorphOp::all()))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int morphOp = GET_PARAM(2);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> morph = cv::cuda::createMorphologyFilter(morphOp, src.type(), ker);

        TEST_CYCLE() morph->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::morphologyEx(src, dst, morphOp, ker);

        CPU_SANITY_CHECK(dst);
    }
}
//////////////////////////////////////////////////////////////////////
// MedianFilter
//////////////////////////////////////////////////////////////////////
// Median

DEF_PARAM_TEST(Sz_KernelSz, cv::Size, int);

//PERF_TEST_P(Sz_Type_KernelSz, Median, Combine(CUDA_TYPICAL_MAT_SIZES, Values(CV_8UC1,CV_8UC1), Values(3, 5, 7, 9, 11, 13, 15)))
PERF_TEST_P(Sz_KernelSz, Median, Combine(CUDA_TYPICAL_MAT_SIZES, Values(3, 5, 7, 9, 11, 13, 15)))
{
    declare.time(20.0);

    const cv::Size size = GET_PARAM(0);
    // const int type = GET_PARAM(1);
    const int type = CV_8UC1;
    const int kernel = GET_PARAM(1);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::Filter> median = cv::cuda::createMedianFilter(d_src.type(), kernel);

        TEST_CYCLE() median->apply(d_src, dst);

        SANITY_CHECK_NOTHING();
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::medianBlur(src,dst,kernel);

        SANITY_CHECK_NOTHING();
    }
}