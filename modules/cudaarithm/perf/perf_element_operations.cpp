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

#define ARITHM_MAT_DEPTH Values(CV_8U, CV_16U, CV_32F, CV_64F)

//////////////////////////////////////////////////////////////////////
// AddMat

DEF_PARAM_TEST(Sz_Depth, cv::Size, MatDepth);

PERF_TEST_P(Sz_Depth, AddMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::add(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::add(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// AddScalar

PERF_TEST_P(Sz_Depth, AddScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::add(d_src, s, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::add(src, s, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// SubtractMat

PERF_TEST_P(Sz_Depth, SubtractMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::subtract(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::subtract(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// SubtractScalar

PERF_TEST_P(Sz_Depth, SubtractScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::subtract(d_src, s, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::subtract(src, s, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MultiplyMat

PERF_TEST_P(Sz_Depth, MultiplyMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::multiply(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::multiply(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MultiplyScalar

PERF_TEST_P(Sz_Depth, MultiplyScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::multiply(d_src, s, dst);

        CUDA_SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::multiply(src, s, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// DivideMat

PERF_TEST_P(Sz_Depth, DivideMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::divide(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::divide(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// DivideScalar

PERF_TEST_P(Sz_Depth, DivideScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::divide(d_src, s, dst);

        CUDA_SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::divide(src, s, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// DivideScalarInv

PERF_TEST_P(Sz_Depth, DivideScalarInv,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::divide(s[0], d_src, dst);

        CUDA_SANITY_CHECK(dst, 1e-6);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::divide(s, src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// AbsDiffMat

PERF_TEST_P(Sz_Depth, AbsDiffMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::absdiff(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::absdiff(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// AbsDiffScalar

PERF_TEST_P(Sz_Depth, AbsDiffScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::absdiff(d_src, s, dst);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::absdiff(src, s, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Abs

PERF_TEST_P(Sz_Depth, Abs,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::abs(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Sqr

PERF_TEST_P(Sz_Depth, Sqr,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::sqr(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Sqrt

PERF_TEST_P(Sz_Depth, Sqrt,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    cv::randu(src, 0, 100000);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::sqrt(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::sqrt(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Log

PERF_TEST_P(Sz_Depth, Log,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    cv::randu(src, 0, 100000);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::log(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::log(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Exp

PERF_TEST_P(Sz_Depth, Exp,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    cv::randu(src, 0, 10);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::exp(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::exp(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// Pow

DEF_PARAM_TEST(Sz_Depth_Power, cv::Size, MatDepth, double);

PERF_TEST_P(Sz_Depth_Power, Pow,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16S, CV_32F),
                    Values(0.3, 2.0, 2.4)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const double power = GET_PARAM(2);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::pow(d_src, power, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::pow(src, power, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CompareMat

CV_ENUM(CmpCode, cv::CMP_EQ, cv::CMP_GT, cv::CMP_GE, cv::CMP_LT, cv::CMP_LE, cv::CMP_NE)

DEF_PARAM_TEST(Sz_Depth_Code, cv::Size, MatDepth, CmpCode);

PERF_TEST_P(Sz_Depth_Code, CompareMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH,
                    CmpCode::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int cmp_code = GET_PARAM(2);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::compare(d_src1, d_src2, dst, cmp_code);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::compare(src1, src2, dst, cmp_code);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CompareScalar

PERF_TEST_P(Sz_Depth_Code, CompareScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    ARITHM_MAT_DEPTH,
                    CmpCode::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int cmp_code = GET_PARAM(2);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::compare(d_src, s, dst, cmp_code);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::compare(src, s, dst, cmp_code);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseNot

PERF_TEST_P(Sz_Depth, BitwiseNot,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_not(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_not(src, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseAndMat

PERF_TEST_P(Sz_Depth, BitwiseAndMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_and(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_and(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseAndScalar

DEF_PARAM_TEST(Sz_Depth_Cn, cv::Size, MatDepth, MatCn);

PERF_TEST_P(Sz_Depth_Cn, BitwiseAndScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S),
                    CUDA_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);
    cv::Scalar_<int> is = s;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_and(d_src, is, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_and(src, is, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseOrMat

PERF_TEST_P(Sz_Depth, BitwiseOrMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_or(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_or(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseOrScalar

PERF_TEST_P(Sz_Depth_Cn, BitwiseOrScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S),
                    CUDA_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);
    cv::Scalar_<int> is = s;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_or(d_src, is, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_or(src, is, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseXorMat

PERF_TEST_P(Sz_Depth, BitwiseXorMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_xor(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_xor(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// BitwiseXorScalar

PERF_TEST_P(Sz_Depth_Cn, BitwiseXorScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S),
                    CUDA_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    cv::Scalar s;
    declare.in(s, WARMUP_RNG);
    cv::Scalar_<int> is = s;

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::bitwise_xor(d_src, is, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::bitwise_xor(src, is, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// RShift

PERF_TEST_P(Sz_Depth_Cn, RShift,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S),
                    CUDA_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Scalar_<int> val = cv::Scalar_<int>::all(4);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::rshift(d_src, val, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// LShift

PERF_TEST_P(Sz_Depth_Cn, LShift,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32S),
                    CUDA_CHANNELS_1_3_4))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int channels = GET_PARAM(2);

    const int type = CV_MAKE_TYPE(depth, channels);

    cv::Mat src(size, type);
    declare.in(src, WARMUP_RNG);

    const cv::Scalar_<int> val = cv::Scalar_<int>::all(4);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::lshift(d_src, val, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// MinMat

PERF_TEST_P(Sz_Depth, MinMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::min(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::min(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MinScalar

PERF_TEST_P(Sz_Depth, MinScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar val;
    declare.in(val, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::min(d_src, val[0], dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::min(src, val[0], dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MaxMat

PERF_TEST_P(Sz_Depth, MaxMat,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src1(size, depth);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::max(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::max(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MaxScalar

PERF_TEST_P(Sz_Depth, MaxScalar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    cv::Scalar val;
    declare.in(val, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::max(d_src, val[0], dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::max(src, val[0], dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// AddWeighted

DEF_PARAM_TEST(Sz_3Depth, cv::Size, MatDepth, MatDepth, MatDepth);

PERF_TEST_P(Sz_3Depth, AddWeighted,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(CV_8U, CV_16U, CV_32F, CV_64F),
                    Values(CV_8U, CV_16U, CV_32F, CV_64F),
                    Values(CV_8U, CV_16U, CV_32F, CV_64F)))
{
    const cv::Size size = GET_PARAM(0);
    const int depth1 = GET_PARAM(1);
    const int depth2 = GET_PARAM(2);
    const int dst_depth = GET_PARAM(3);

    cv::Mat src1(size, depth1);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, depth2);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::addWeighted(d_src1, 0.5, d_src2, 0.5, 10.0, dst, dst_depth);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::addWeighted(src1, 0.5, src2, 0.5, 10.0, dst, dst_depth);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeComplex

PERF_TEST_P(Sz, MagnitudeComplex,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC2);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::magnitude(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat xy[2];
        cv::split(src, xy);

        cv::Mat dst;

        TEST_CYCLE() cv::magnitude(xy[0], xy[1], dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeSqrComplex

PERF_TEST_P(Sz, MagnitudeSqrComplex,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src(size, CV_32FC2);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::magnitudeSqr(d_src, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Magnitude

PERF_TEST_P(Sz, Magnitude,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src1(size, CV_32FC1);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, CV_32FC1);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::magnitude(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::magnitude(src1, src2, dst);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// MagnitudeSqr

PERF_TEST_P(Sz, MagnitudeSqr,
            CUDA_TYPICAL_MAT_SIZES)
{
    const cv::Size size = GetParam();

    cv::Mat src1(size, CV_32FC1);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, CV_32FC1);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::magnitudeSqr(d_src1, d_src2, dst);

        CUDA_SANITY_CHECK(dst);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////////////////////
// Phase

DEF_PARAM_TEST(Sz_AngleInDegrees, cv::Size, bool);

PERF_TEST_P(Sz_AngleInDegrees, Phase,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Bool()))
{
    const cv::Size size = GET_PARAM(0);
    const bool angleInDegrees = GET_PARAM(1);

    cv::Mat src1(size, CV_32FC1);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, CV_32FC1);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::phase(d_src1, d_src2, dst, angleInDegrees);

        CUDA_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::phase(src1, src2, dst, angleInDegrees);

        CPU_SANITY_CHECK(dst);
    }
}

//////////////////////////////////////////////////////////////////////
// CartToPolar

PERF_TEST_P(Sz_AngleInDegrees, CartToPolar,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Bool()))
{
    const cv::Size size = GET_PARAM(0);
    const bool angleInDegrees = GET_PARAM(1);

    cv::Mat src1(size, CV_32FC1);
    declare.in(src1, WARMUP_RNG);

    cv::Mat src2(size, CV_32FC1);
    declare.in(src2, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src1(src1);
        const cv::cuda::GpuMat d_src2(src2);
        cv::cuda::GpuMat magnitude;
        cv::cuda::GpuMat angle;

        TEST_CYCLE() cv::cuda::cartToPolar(d_src1, d_src2, magnitude, angle, angleInDegrees);

        CUDA_SANITY_CHECK(magnitude);
        CUDA_SANITY_CHECK(angle, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat magnitude;
        cv::Mat angle;

        TEST_CYCLE() cv::cartToPolar(src1, src2, magnitude, angle, angleInDegrees);

        CPU_SANITY_CHECK(magnitude);
        CPU_SANITY_CHECK(angle);
    }
}

//////////////////////////////////////////////////////////////////////
// PolarToCart

PERF_TEST_P(Sz_AngleInDegrees, PolarToCart,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Bool()))
{
    const cv::Size size = GET_PARAM(0);
    const bool angleInDegrees = GET_PARAM(1);

    cv::Mat magnitude(size, CV_32FC1);
    declare.in(magnitude, WARMUP_RNG);

    cv::Mat angle(size, CV_32FC1);
    declare.in(angle, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_magnitude(magnitude);
        const cv::cuda::GpuMat d_angle(angle);
        cv::cuda::GpuMat x;
        cv::cuda::GpuMat y;

        TEST_CYCLE() cv::cuda::polarToCart(d_magnitude, d_angle, x, y, angleInDegrees);

        CUDA_SANITY_CHECK(x);
        CUDA_SANITY_CHECK(y);
    }
    else
    {
        cv::Mat x;
        cv::Mat y;

        TEST_CYCLE() cv::polarToCart(magnitude, angle, x, y, angleInDegrees);

        CPU_SANITY_CHECK(x);
        CPU_SANITY_CHECK(y);
    }
}

//////////////////////////////////////////////////////////////////////
// Threshold

CV_ENUM(ThreshOp, cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV)

DEF_PARAM_TEST(Sz_Depth_Op, cv::Size, MatDepth, ThreshOp);

PERF_TEST_P(Sz_Depth_Op, Threshold,
            Combine(CUDA_TYPICAL_MAT_SIZES,
            Values(CV_8U, CV_16U, CV_32F, CV_64F),
            ThreshOp::all()))
{
    const cv::Size size = GET_PARAM(0);
    const int depth = GET_PARAM(1);
    const int threshOp = GET_PARAM(2);

    cv::Mat src(size, depth);
    declare.in(src, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_src(src);
        cv::cuda::GpuMat dst;

        TEST_CYCLE() cv::cuda::threshold(d_src, dst, 100.0, 255.0, threshOp);

        CUDA_SANITY_CHECK(dst, 1e-10);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::threshold(src, dst, 100.0, 255.0, threshOp);

        CPU_SANITY_CHECK(dst);
    }
}

}} // namespace
