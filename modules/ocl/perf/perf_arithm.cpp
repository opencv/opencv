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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

using namespace perf;
using std::tr1::get;
using std::tr1::tuple;

///////////// Lut ////////////////////////

typedef Size_MatType LUTFixture;

PERF_TEST_P(LUTFixture, LUT,
          ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                             OCL_PERF_ENUM(CV_8UC1, CV_8UC3)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    // creating src data
    Mat src(srcSize, type), lut(1, 256, CV_8UC1);
    int dstType = CV_MAKETYPE(lut.depth(), src.channels());
    Mat dst(srcSize, dstType);

    randu(lut, 0, 2);
    declare.in(src, WARMUP_RNG).in(lut).out(dst);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclLut(lut), oclDst(srcSize, dstType);

        OCL_TEST_CYCLE() cv::ocl::LUT(oclSrc, oclLut, oclDst);
        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::LUT(src, lut, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Exp ////////////////////////

typedef TestBaseWithParam<Size> ExpFixture;

PERF_TEST_P(ExpFixture, Exp, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const double eps = 1e-6;

    // creating src data
    Mat src(srcSize, CV_32FC1), dst(srcSize, CV_32FC1);
    declare.in(src).out(dst);
    randu(src, 5, 16);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        OCL_TEST_CYCLE() cv::ocl::exp(oclSrc, oclDst);

        oclDst.download(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::exp(src, dst);
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK(dst, eps, ERROR_RELATIVE);
}

///////////// LOG ////////////////////////

typedef TestBaseWithParam<Size> LogFixture;

PERF_TEST_P(LogFixture, Log, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const double eps = 1e-6;

    // creating src data
    Mat src(srcSize, CV_32F), dst(srcSize, src.type());
    randu(src, 1, 10);
    declare.in(src).out(dst);

    if (srcSize == OCL_SIZE_4000)
        declare.time(3.6);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        OCL_TEST_CYCLE() cv::ocl::log(oclSrc, oclDst);

        oclDst.download(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::log(src, dst);
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK(dst, eps, ERROR_RELATIVE);
}

///////////// SQRT ///////////////////////

typedef TestBaseWithParam<Size> SqrtFixture;

PERF_TEST_P(SqrtFixture, Sqrt, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const double eps = 1e-6;

    // creating src data
    Mat src(srcSize, CV_32F), dst(srcSize, src.type());
    randu(src, 0, 10);
    declare.in(src).out(dst);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        OCL_TEST_CYCLE() cv::ocl::sqrt(oclSrc, oclDst);

        oclDst.download(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::sqrt(src, dst);
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK(dst, eps, ERROR_RELATIVE);
}

///////////// Add ////////////////////////

typedef Size_MatType AddFixture;

PERF_TEST_P(AddFixture, Add,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    randu(src1, 0, 1);
    randu(src2, 0, 1);
    declare.in(src1, src2).out(dst);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::add(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::add(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Mul ////////////////////////

typedef Size_MatType MulFixture;

PERF_TEST_P(MulFixture, Mul, ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                                                OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    randu(src1, 0, 256);
    randu(src2, 0, 256);
    declare.in(src1, src2).out(dst);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::multiply(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::multiply(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Div ////////////////////////

typedef Size_MatType DivFixture;

PERF_TEST_P(DivFixture, Div,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if ((srcSize == OCL_SIZE_4000 && type == CV_8UC1) ||
            (srcSize == OCL_SIZE_2000 && type == CV_8UC4))
        declare.time(4.2);
    else if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(16.6);

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::divide(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::divide(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Absdiff ////////////////////////

typedef Size_MatType AbsDiffFixture;

PERF_TEST_P(AbsDiffFixture, Absdiff,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).in(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::absdiff(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::absdiff(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// CartToPolar ////////////////////////

typedef TestBaseWithParam<Size> CartToPolarFixture;

PERF_TEST_P(CartToPolarFixture, CartToPolar, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const double eps = 8e-3;

    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst1(srcSize, CV_32FC1), dst2(srcSize, CV_32FC1);
    declare.in(src1, src2).out(dst1, dst2);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (srcSize == OCL_SIZE_4000)
        declare.time(3.6);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst1(srcSize, src1.type()), oclDst2(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::cartToPolar(oclSrc1, oclSrc2, oclDst1, oclDst2);

        oclDst1.download(dst1);
        oclDst2.download(dst2);

        SANITY_CHECK(dst1, eps);
        SANITY_CHECK(dst2, eps);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::cartToPolar(src1, src2, dst1, dst2);

        SANITY_CHECK(dst1, eps);
        SANITY_CHECK(dst2, eps);
    }
    else
        OCL_PERF_ELSE
}

///////////// PolarToCart ////////////////////////

typedef TestBaseWithParam<Size> PolarToCartFixture;

PERF_TEST_P(PolarToCartFixture, PolarToCart, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst1(srcSize, CV_32FC1), dst2(srcSize, CV_32FC1);
    declare.in(src1, src2).out(dst1, dst2);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (srcSize == OCL_SIZE_4000)
        declare.time(5.4);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst1(srcSize, src1.type()), oclDst2(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::polarToCart(oclSrc1, oclSrc2, oclDst1, oclDst2);

        oclDst1.download(dst1);
        oclDst2.download(dst2);

        SANITY_CHECK(dst1, 5e-5);
        SANITY_CHECK(dst2, 5e-5);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::polarToCart(src1, src2, dst1, dst2);

        SANITY_CHECK(dst1, 5e-5);
        SANITY_CHECK(dst2, 5e-5);
    }
    else
        OCL_PERF_ELSE
}

///////////// Magnitude ////////////////////////

typedef TestBaseWithParam<Size> MagnitudeFixture;

PERF_TEST_P(MagnitudeFixture, Magnitude, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst(srcSize, CV_32FC1);
    randu(src1, 0, 1);
    randu(src2, 0, 1);
    declare.in(src1, src2).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::magnitude(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-6);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::magnitude(src1, src2, dst);

        SANITY_CHECK(dst, 1e-6);
    }
    else
        OCL_PERF_ELSE
}

///////////// Transpose ////////////////////////

typedef Size_MatType TransposeFixture;

PERF_TEST_P(TransposeFixture, Transpose,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::transpose(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::transpose(src, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Flip ////////////////////////

typedef Size_MatType FlipFixture;

PERF_TEST_P(FlipFixture, Flip,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::flip(oclSrc, oclDst, 0);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::flip(src, dst, 0);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// minMax ////////////////////////

typedef Size_MatType minMaxFixture;

PERF_TEST_P(minMaxFixture, minMax,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type);
    declare.in(src, WARMUP_RNG);

    double min_val = std::numeric_limits<double>::max(),
            max_val = std::numeric_limits<double>::min();

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() cv::ocl::minMax(oclSrc, &min_val, &max_val);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
    else if (RUN_PLAIN_IMPL)
    {
        Point min_loc, max_loc;

        TEST_CYCLE() cv::minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
    else
        OCL_PERF_ELSE
}

///////////// minMaxLoc ////////////////////////

typedef Size_MatType minMaxLocFixture;

PERF_TEST_P(minMaxLocFixture, minMaxLoc,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type);
    randu(src, 0, 1);
    declare.in(src);

    double min_val = 0.0, max_val = 0.0;
    Point min_loc, max_loc;

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() cv::ocl::minMaxLoc(oclSrc, &min_val, &max_val, &min_loc, &max_loc);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
    else
        OCL_PERF_ELSE
}

///////////// Sum ////////////////////////

typedef Size_MatType SumFixture;

PERF_TEST_P(SumFixture, Sum,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32SC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type);
    Scalar result;
    randu(src, 0, 60);
    declare.in(src);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() result = cv::ocl::sum(oclSrc);

        SANITY_CHECK(result);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() result = cv::sum(src);

        SANITY_CHECK(result);
    }
    else
        OCL_PERF_ELSE
}

///////////// countNonZero ////////////////////////

typedef Size_MatType countNonZeroFixture;

PERF_TEST_P(countNonZeroFixture, countNonZero,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type);
    int result = 0;
    randu(src, 0, 256);
    declare.in(src);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() result = cv::ocl::countNonZero(oclSrc);

        SANITY_CHECK(result);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() result = cv::countNonZero(src);

        SANITY_CHECK(result);
    }
    else
        OCL_PERF_ELSE
}

///////////// Phase ////////////////////////

typedef TestBaseWithParam<Size> PhaseFixture;

PERF_TEST_P(PhaseFixture, Phase, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();

    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst(srcSize, CV_32FC1);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::phase(oclSrc1, oclSrc2, oclDst, 1);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-2);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::phase(src1, src2, dst, 1);

        SANITY_CHECK(dst, 1e-2);
    }
    else
        OCL_PERF_ELSE
}

///////////// bitwise_and////////////////////////

typedef Size_MatType BitwiseAndFixture;

PERF_TEST_P(BitwiseAndFixture, bitwise_and,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32SC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::bitwise_and(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::bitwise_and(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// bitwise_xor ////////////////////////

typedef Size_MatType BitwiseXorFixture;

PERF_TEST_P(BitwiseXorFixture, bitwise_xor,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32SC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::bitwise_xor(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::bitwise_xor(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// bitwise_or ////////////////////////

typedef Size_MatType BitwiseOrFixture;

PERF_TEST_P(BitwiseOrFixture, bitwise_or,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32SC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, src1.type());

        OCL_TEST_CYCLE() cv::ocl::bitwise_or(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::bitwise_or(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// bitwise_not////////////////////////

typedef Size_MatType BitwiseNotFixture;

PERF_TEST_P(BitwiseAndFixture, bitwise_not,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32SC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::bitwise_not(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::bitwise_not(src, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// compare////////////////////////

typedef Size_MatType CompareFixture;

PERF_TEST_P(CompareFixture, compare,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, CV_8UC1);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, CV_8UC1);

        OCL_TEST_CYCLE() cv::ocl::compare(oclSrc1, oclSrc2, oclDst, CMP_EQ);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::compare(src1, src2, dst, CMP_EQ);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// pow ////////////////////////

typedef TestBaseWithParam<Size> PowFixture;

PERF_TEST_P(PowFixture, pow, OCL_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const double eps = 1e-6;

    Mat src(srcSize, CV_32F), dst(srcSize, CV_32F);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        OCL_TEST_CYCLE() cv::ocl::pow(oclSrc, -2.0, oclDst);

        oclDst.download(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::pow(src, -2.0, dst);
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK(dst, eps, ERROR_RELATIVE);
}

///////////// AddWeighted////////////////////////

typedef Size_MatType AddWeightedFixture;

PERF_TEST_P(AddWeightedFixture, AddWeighted,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);
    double alpha = 2.0, beta = 1.0, gama = 3.0;

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::addWeighted(oclSrc1, alpha, oclSrc2, beta, gama, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::addWeighted(src1, alpha, src2, beta, gama, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Min ////////////////////////

typedef Size_MatType MinFixture;

PERF_TEST_P(MinFixture, Min,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::min(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() dst = cv::min(src1, src2);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Max ////////////////////////

typedef Size_MatType MaxFixture;

PERF_TEST_P(MaxFixture, Max,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::max(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() dst = cv::max(src1, src2);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Max ////////////////////////

typedef Size_MatType AbsFixture;

PERF_TEST_P(AbsFixture, Abs,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::abs(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() dst = cv::abs(src);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Repeat ////////////////////////

typedef Size_MatType RepeatFixture;

PERF_TEST_P(RepeatFixture, Repeat,
            ::testing::Combine(::testing::Values(OCL_SIZE_1000, OCL_SIZE_2000),
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int nx = 3, ny = 2;
    const Size dstSize(srcSize.width * nx, srcSize.height * ny);

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    Mat src(srcSize, type), dst(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(dstSize, type);

        OCL_TEST_CYCLE() cv::ocl::repeat(oclSrc, ny, nx, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::repeat(src, ny, nx, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}
