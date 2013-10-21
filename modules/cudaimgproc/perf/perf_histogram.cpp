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
// HistEvenC1

PERF_TEST_P(Sz_Depth, HistEvenC1,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_16S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;
        cv::cuda::GpuMat d_buf;

        TEST_CYCLE() cv::cuda::histEven(d_src, dst, d_buf, 30, 0, 180);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        const int hbins = 30;
        const float hranges[] = {0.0f, 180.0f};
        const int histSize[] = {hbins};
        const float* ranges[] = {hranges};
        const int channels[] = {0};

        cv::Mat dst;

        TEST_CYCLE() cv::calcHist(&src, 1, channels, cv::Mat(), dst, 1, histSize, ranges);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// HistEvenC4

PERF_TEST_P(Sz_Depth, HistEvenC4,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_16S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, CV_MAKE_TYPE(depth, 4));
    declare.in(src, WARMUP_RNG);

    int histSize[] = {30, 30, 30, 30};
    int lowerLevel[] = {0, 0, 0, 0};
    int upperLevel[] = {180, 180, 180, 180};

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat d_hist[4];
        cv::cuda::GpuMat d_buf;

        TEST_CYCLE() cv::cuda::histEven(d_src, d_hist, d_buf, histSize, lowerLevel, upperLevel);

        cv::Mat cpu_hist0, cpu_hist1, cpu_hist2, cpu_hist3;
        d_hist[0].download(cpu_hist0);
        d_hist[1].download(cpu_hist1);
        d_hist[2].download(cpu_hist2);
        d_hist[3].download(cpu_hist3);
        SANITY_CHECK(cpu_hist0);
        SANITY_CHECK(cpu_hist1);
        SANITY_CHECK(cpu_hist2);
        SANITY_CHECK(cpu_hist3);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// CalcHist

PERF_TEST_P(Sz, CalcHist,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::calcHist(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// EqualizeHist

PERF_TEST_P(Sz, EqualizeHist,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;
        cv::cuda::GpuMat d_buf;

        TEST_CYCLE() cv::cuda::equalizeHist(d_src, dst, d_buf);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::equalizeHist(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CLAHE

DEF_PARAM_TEST(Sz_ClipLimit, cv::Size, double);

PERF_TEST_P(Sz_ClipLimit, CLAHE,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(0.0, 40.0)))
{
    const cv::Size size = GET_PARAM(0);
    const double clipLimit = GET_PARAM(1);

    cv::Mat src(size, CV_8UC1);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(clipLimit);
        cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() clahe->apply(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit);
        cv::Mat dst;

        TEST_CYCLE() clahe->apply(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}
