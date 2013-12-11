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
// GEMM

CV_FLAGS(GemmFlags, 0, cv::GEMM_1_T, cv::GEMM_2_T, cv::GEMM_3_T)
#define ALL_GEMM_FLAGS Values(GemmFlags(0), GemmFlags(cv::GEMM_1_T), GemmFlags(cv::GEMM_2_T), GemmFlags(cv::GEMM_3_T), \
                              GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T | cv::GEMM_3_T))

DEF_PARAM_TEST(Sz_Type_Flags, cv::Size, MatType, GemmFlags);

PERF_TEST_P(Sz_Type_Flags, GEMM,
            Combine(Values(cv::Size(512, 512), cv::Size(1024, 1024)),
                    Values(CV_32FC1, CV_32FC2, CV_64FC1),
                    ALL_GEMM_FLAGS))
{
    const cv::Size size = GET_PARAM(0);
    const int type = GET_PARAM(1);
    const int flags = GET_PARAM(2);

    cv::Mat src1(size, type);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, type);
    declare.in(src2, WARMUP_RNG);

    cv::Mat src3(size, type);
    declare.in(src3, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        declare.time(5.0);

        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        const cv::cuda::GpuMat d_src3(src3);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, dst, flags);

        CUDA_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        declare.time(50.0);

        cv::Mat dst;

        TEST_CYCLE() cv::gemm(src1, src2, 1.0, src3, 1.0, dst, flags);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MulSpectrums

CV_FLAGS(DftFlags, 0, cv::DFT_INVERSE, cv::DFT_SCALE, cv::DFT_ROWS, cv::DFT_COMPLEX_OUTPUT, cv::DFT_REAL_OUTPUT)

DEF_PARAM_TEST(Sz_Flags, cv::Size, DftFlags);

PERF_TEST_P(Sz_Flags, MulSpectrums,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(0, DftFlags(cv::DFT_ROWS))))
{
    const cv::Size size = GET_PARAM(0);
    const int flag = GET_PARAM(1);

    cv::Mat a(size, CV_32FC2);
    cv::Mat b(size, CV_32FC2);
    declare.in(a, b, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_a(a);
        const cv::cuda::GpuMat d_b(b);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::mulSpectrums(d_a, d_b, dst, flag);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::mulSpectrums(a, b, dst, flag);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MulAndScaleSpectrums

PERF_TEST_P(Sz, MulAndScaleSpectrums,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    const float scale = 1.f / size.area();

    cv::Mat src1(size, CV_32FC2);
    cv::Mat src2(size, CV_32FC2);
    declare.in(src1,src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::mulAndScaleSpectrums(d_src1, d_src2, dst, cv::DFT_ROWS, scale, false);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Dft

PERF_TEST_P(Sz_Flags, Dft,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(0, DftFlags(cv::DFT_ROWS), DftFlags(cv::DFT_INVERSE))))
{
    declare.time(10.0);

    const cv::Size size = GET_PARAM(0);
    const int flag = GET_PARAM(1);

    cv::Mat src(size, CV_32FC2);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::dft(d_src, dst, size, flag);

        CUDA_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::dft(src, dst, flag);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Convolve

DEF_PARAM_TEST(Sz_KernelSz_Ccorr, cv::Size, int, bool);

PERF_TEST_P(Sz_KernelSz_Ccorr, Convolve,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(17, 27, 32, 64),
                    Bool()))
{
    declare.time(10.0);

    const cv::Size size = GET_PARAM(0);
    const int templ_size = GET_PARAM(1);
    const bool ccorr = GET_PARAM(2);

    const cv::Mat image(size, CV_32FC1);
    const cv::Mat templ(templ_size, templ_size, CV_32FC1);
    declare.in(image, templ, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        cv::cuda::GpuMat d_image = cv::cuda::createContinuous(size, CV_32FC1);
        d_image.upload(image);

        cv::cuda::GpuMat d_templ = cv::cuda::createContinuous(templ_size, templ_size, CV_32FC1);
        d_templ.upload(templ);

        cv::Ptr<cv::cuda::Convolution> convolution = cv::cuda::createConvolution();

        cv::cuda::GpuMat dst;

        TEST_CYCLE() convolution->convolve(d_image, d_templ, dst, ccorr);

        CUDA_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        if (ccorr)
            FAIL_NO_CPU();

        cv::Mat dst;

        TEST_CYCLE() cv::filter2D(image, dst, image.depth(), templ);

        CPU_SANITY_CHECK(dst);
    }
}
