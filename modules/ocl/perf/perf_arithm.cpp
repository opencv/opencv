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

OCL_PERF_TEST_P(LUTFixture, LUT,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), cn = CV_MAT_CN(type);

    // creating src data
    Mat src(srcSize, CV_8UC(cn)), lut(1, 256, type);
    int dstType = CV_MAKETYPE(lut.depth(), src.channels());
    Mat dst(srcSize, dstType);

    declare.in(src, lut, WARMUP_RNG).out(dst);

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

typedef Size_MatType ExpFixture;

OCL_PERF_TEST_P(ExpFixture, Exp, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const double eps = 1e-6;

    // creating src data
    Mat src(srcSize, type), dst(srcSize, type);
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

///////////// Log ////////////////////////

typedef Size_MatType LogFixture;

OCL_PERF_TEST_P(LogFixture, Log, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    // getting params
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const double eps = 1e-6;

    // creating src data
    Mat src(srcSize, type), dst(srcSize, type);
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

///////////// Add ////////////////////////

typedef Size_MatType AddFixture;

OCL_PERF_TEST_P(AddFixture, Add,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

///////////// Subtract ////////////////////////

typedef Size_MatType SubtractFixture;

OCL_PERF_TEST_P(SubtractFixture, Subtract,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

        OCL_TEST_CYCLE() cv::ocl::subtract(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::subtract(src1, src2, dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}


///////////// Mul ////////////////////////

typedef Size_MatType MulFixture;

OCL_PERF_TEST_P(MulFixture, Multiply, ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

OCL_PERF_TEST_P(DivFixture, Divide,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

    // select implementation
    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::divide(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::divide(src1, src2, dst);

        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
        OCL_PERF_ELSE
}

///////////// Absdiff ////////////////////////

typedef Size_MatType AbsDiffFixture;

OCL_PERF_TEST_P(AbsDiffFixture, Absdiff,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

typedef Size_MatType CartToPolarFixture;

OCL_PERF_TEST_P(CartToPolarFixture, CartToPolar, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const double eps = 8e-3;

    Mat src1(srcSize, type), src2(srcSize, type),
            dst1(srcSize, type), dst2(srcSize, type);
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

typedef Size_MatType PolarToCartFixture;

OCL_PERF_TEST_P(PolarToCartFixture, PolarToCart, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type),
            dst1(srcSize, type), dst2(srcSize, type);
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

typedef Size_MatType MagnitudeFixture;

OCL_PERF_TEST_P(MagnitudeFixture, Magnitude, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, type), src2(srcSize, type),
            dst(srcSize, type);
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

OCL_PERF_TEST_P(TransposeFixture, Transpose, ::testing::Combine(
                OCL_TEST_SIZES, OCL_TEST_TYPES))
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

enum
{
    FLIP_BOTH = 0, FLIP_ROWS, FLIP_COLS
};

CV_ENUM(FlipType, FLIP_BOTH, FLIP_ROWS, FLIP_COLS)

typedef std::tr1::tuple<Size, MatType, FlipType> FlipParams;
typedef TestBaseWithParam<FlipParams> FlipFixture;

OCL_PERF_TEST_P(FlipFixture, Flip,
            ::testing::Combine(OCL_TEST_SIZES,
                               OCL_TEST_TYPES, FlipType::all()))
{
    const FlipParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int flipType = get<2>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::flip(oclSrc, oclDst, flipType - 1);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::flip(src, dst, flipType - 1);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// Phase ////////////////////////

typedef Size_MatType PhaseFixture;

OCL_PERF_TEST_P(PhaseFixture, Phase, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
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

OCL_PERF_TEST_P(BitwiseAndFixture, Bitwise_and,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

OCL_PERF_TEST_P(BitwiseXorFixture, Bitwise_xor,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

OCL_PERF_TEST_P(BitwiseOrFixture, Bitwise_or,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::bitwise_or(src1, src2, dst);
    }
    else
        OCL_PERF_ELSE

    if (CV_MAT_DEPTH(type) >= CV_32F)
        cv::patchNaNs(dst, 17);
    SANITY_CHECK(dst);
}

///////////// bitwise_not////////////////////////

typedef Size_MatType BitwiseNotFixture;

OCL_PERF_TEST_P(BitwiseNotFixture, Bitwise_not,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

///////////// SetIdentity ////////////////////////

typedef Size_MatType SetIdentityFixture;

OCL_PERF_TEST_P(SetIdentityFixture, SetIdentity,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type);
    Scalar s = Scalar::all(17);
    declare.in(src, WARMUP_RNG).out(src);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() cv::ocl::setIdentity(oclSrc, s);

        oclSrc.download(src);

        SANITY_CHECK(src);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::setIdentity(src, s);

        SANITY_CHECK(src);
    }
    else
        OCL_PERF_ELSE
}

///////////// compare////////////////////////

CV_ENUM(CmpCode, CMP_LT, CMP_LE, CMP_EQ, CMP_NE, CMP_GE, CMP_GT)

typedef std::tr1::tuple<Size, MatType, CmpCode> CompareParams;
typedef TestBaseWithParam<CompareParams> CompareFixture;

OCL_PERF_TEST_P(CompareFixture, Compare,
            ::testing::Combine(OCL_TEST_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1), CmpCode::all()))
{
    const CompareParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int cmpCode = get<2>(params);

    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, CV_8UC1);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, CV_8UC1);

        OCL_TEST_CYCLE() cv::ocl::compare(oclSrc1, oclSrc2, oclDst, cmpCode);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::compare(src1, src2, dst, cmpCode);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// pow ////////////////////////

typedef Size_MatType PowFixture;

OCL_PERF_TEST_P(PowFixture, Pow, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const double eps = 1e-6;

    Mat src(srcSize, type), dst(srcSize, type);
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

OCL_PERF_TEST_P(AddWeightedFixture, AddWeighted,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
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

OCL_PERF_TEST_P(MinFixture, Min,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
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

OCL_PERF_TEST_P(MaxFixture, Max,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
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

///////////// Abs ////////////////////////

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

OCL_PERF_TEST_P(RepeatFixture, Repeat,
            ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                               OCL_TEST_TYPES))
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
