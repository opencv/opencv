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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"
#include "opencv2/core/softfloat.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// Lut ////////////////////////

typedef Size_MatType LUTFixture;

OCL_PERF_TEST_P(LUTFixture, LUT,
          ::testing::Combine(OCL_TEST_SIZES,
                             OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), cn = CV_MAT_CN(type);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, CV_8UC(cn)), lut(1, 256, type);
    int dstType = CV_MAKETYPE(lut.depth(), src.channels());
    UMat dst(srcSize, dstType);

    declare.in(src, lut, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::LUT(src, lut, dst);

    SANITY_CHECK(dst);
}

///////////// Exp ////////////////////////

typedef Size_MatType ExpFixture;

OCL_PERF_TEST_P(ExpFixture, Exp, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src).out(dst);
    randu(src, 5, 16);

    OCL_TEST_CYCLE() cv::exp(src, dst);

    if (CV_MAT_DEPTH(type) >= CV_32F)
        SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
    else
        SANITY_CHECK(dst, 1);
}

///////////// Log ////////////////////////

typedef Size_MatType LogFixture;

OCL_PERF_TEST_P(LogFixture, Log, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    randu(src, 1, 10000);
    declare.in(src).out(dst);

    OCL_TEST_CYCLE() cv::log(src, dst);

    if (CV_MAT_DEPTH(type) >= CV_32F)
        SANITY_CHECK(dst, 2e-4, ERROR_RELATIVE);
    else
        SANITY_CHECK(dst, 1);
}

///////////// Add ////////////////////////

typedef Size_MatType AddFixture;

OCL_PERF_TEST_P(AddFixture, Add,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size srcSize = GET_PARAM(0);
    const int type = GET_PARAM(1);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::add(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// Subtract ////////////////////////

typedef Size_MatType SubtractFixture;

OCL_PERF_TEST_P(SubtractFixture, Subtract,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::subtract(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// Mul ////////////////////////

typedef Size_MatType MulFixture;

OCL_PERF_TEST_P(MulFixture, Multiply, ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::multiply(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// Div ////////////////////////

typedef Size_MatType DivFixture;

OCL_PERF_TEST_P(DivFixture, Divide,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // remove zeros from src2
    {
        Mat m2 = src2.getMat(ACCESS_RW);
        Mat zero_mask = m2 == 0;
        Mat fix;
        zero_mask.convertTo(fix, type); // 0 or 255
        cv::add(m2, fix, m2);
    }

    OCL_TEST_CYCLE() cv::divide(src1, src2, dst);

    if (CV_MAT_DEPTH(type) >= CV_32F)
        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    else
        SANITY_CHECK(dst, 1);
}

///////////// Absdiff ////////////////////////

typedef Size_MatType AbsDiffFixture;

OCL_PERF_TEST_P(AbsDiffFixture, Absdiff,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).in(dst);

    OCL_TEST_CYCLE() cv::absdiff(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// CartToPolar ////////////////////////

typedef Size_MatType CartToPolarFixture;

OCL_PERF_TEST_P(CartToPolarFixture, CartToPolar, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type),
            dst1(srcSize, type), dst2(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst1, dst2);

    OCL_TEST_CYCLE() cv::cartToPolar(src1, src2, dst1, dst2);

    SANITY_CHECK(dst1, 8e-3);
    SANITY_CHECK(dst2, 8e-3);
}

///////////// PolarToCart ////////////////////////

typedef Size_MatType PolarToCartFixture;

OCL_PERF_TEST_P(PolarToCartFixture, PolarToCart, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type),
            dst1(srcSize, type), dst2(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst1, dst2);

    OCL_TEST_CYCLE() cv::polarToCart(src1, src2, dst1, dst2);

    SANITY_CHECK(dst1, 5e-5);
    SANITY_CHECK(dst2, 5e-5);
}

///////////// Magnitude ////////////////////////

typedef Size_MatType MagnitudeFixture;

OCL_PERF_TEST_P(MagnitudeFixture, Magnitude, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type),
            dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::magnitude(src1, src2, dst);

    SANITY_CHECK(dst, 1e-6);
}

///////////// Transpose ////////////////////////

typedef Size_MatType TransposeFixture;

OCL_PERF_TEST_P(TransposeFixture, Transpose, ::testing::Combine(
                OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::transpose(src, dst);

    SANITY_CHECK(dst);
}

OCL_PERF_TEST_P(TransposeFixture, TransposeInplace, ::testing::Combine(
                OCL_PERF_ENUM(Size(640, 640), Size(1280, 1280), Size(2160, 2160)), OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    declare.in(src, WARMUP_RNG).out(src, WARMUP_NONE);

    OCL_TEST_CYCLE() cv::transpose(src, src);

    SANITY_CHECK_NOTHING();
}

///////////// Flip ////////////////////////

enum
{
    FLIP_BOTH = 0, FLIP_ROWS, FLIP_COLS
};

CV_ENUM(FlipType, FLIP_BOTH, FLIP_ROWS, FLIP_COLS)

typedef tuple<Size, MatType, FlipType> FlipParams;
typedef TestBaseWithParam<FlipParams> FlipFixture;

OCL_PERF_TEST_P(FlipFixture, Flip,
            ::testing::Combine(OCL_TEST_SIZES,
                               ::testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_32FC1, CV_32FC4),
                               FlipType::all()))
{
    const FlipParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int flipType = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::flip(src, dst, flipType - 1);

    SANITY_CHECK(dst);
}

///////////// Rotate ////////////////////////

enum
{
    ROTATE_90_CLOCKWISE = 0, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
};

CV_ENUM(RotateType, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE)

typedef tuple<Size, MatType, RotateType> RotateParams;
typedef TestBaseWithParam<RotateParams> RotateFixture;

OCL_PERF_TEST_P(RotateFixture, rotate,
                ::testing::Combine(OCL_TEST_SIZES,
                                   ::testing::Values(CV_8UC1, CV_8UC2, CV_8UC4, CV_32FC1, CV_32FC4),
                                   RotateType::all()))
{
    const RotateParams params = GetParam();
    const Size srcSize   = get<0>(params);
    const int type       = get<1>(params);
    const int rotateCode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::rotate(src, dst, rotateCode);

    SANITY_CHECK_NOTHING();
}

///////////// minMaxLoc ////////////////////////

typedef Size_MatType MinMaxLocFixture;

OCL_PERF_TEST_P(MinMaxLocFixture, MinMaxLoc,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    bool onecn = CV_MAT_CN(type) == 1;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);;
    declare.in(src, WARMUP_RNG);

    double min_val = 0.0, max_val = 0.0;
    Point min_loc, max_loc;

    OCL_TEST_CYCLE() cv::minMaxLoc(src, &min_val, &max_val, onecn ? &min_loc : NULL,
                                   onecn ? &max_loc : NULL);

    ASSERT_GE(max_val, min_val);
    SANITY_CHECK(min_val);
    SANITY_CHECK(max_val);

    int min_loc_x = min_loc.x, min_loc_y = min_loc.y, max_loc_x = max_loc.x,
            max_loc_y = max_loc.y;
    SANITY_CHECK(min_loc_x);
    SANITY_CHECK(min_loc_y);
    SANITY_CHECK(max_loc_x);
    SANITY_CHECK(max_loc_y);
}

///////////// Sum ////////////////////////

typedef Size_MatType SumFixture;

OCL_PERF_TEST_P(SumFixture, Sum,
            ::testing::Combine(OCL_TEST_SIZES,
                               OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), depth = CV_MAT_DEPTH(type);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    Scalar result;
    randu(src, 0, 60);
    declare.in(src);

    OCL_TEST_CYCLE() result = cv::sum(src);

    if (depth >= CV_32F)
        SANITY_CHECK(result, 1e-6, ERROR_RELATIVE);
    else
        SANITY_CHECK(result);
}

///////////// countNonZero ////////////////////////

typedef Size_MatType CountNonZeroFixture;

OCL_PERF_TEST_P(CountNonZeroFixture, CountNonZero,
                ::testing::Combine(OCL_TEST_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    int result = 0;
    randu(src, 0, 10);
    declare.in(src);

    OCL_TEST_CYCLE() result = cv::countNonZero(src);

    SANITY_CHECK(result);
}

///////////// countNonZero ////////////////////////

typedef Size_MatType HasNonZeroFixture;

OCL_PERF_TEST_P(HasNonZeroFixture, HasNonZero,
    ::testing::Combine(OCL_TEST_SIZES,
        OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    /*bool result = false;*/
    randu(src, 0, 10);
    declare.in(src);

    OCL_TEST_CYCLE() /*result =*/ cv::hasNonZero(src);

    SANITY_CHECK_NOTHING();
}

///////////// Phase ////////////////////////

typedef Size_MatType PhaseFixture;

OCL_PERF_TEST_P(PhaseFixture, Phase, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type),
            dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::phase(src1, src2, dst, 1);

    SANITY_CHECK(dst, 1e-2);
}

///////////// bitwise_and ////////////////////////

typedef Size_MatType BitwiseAndFixture;

OCL_PERF_TEST_P(BitwiseAndFixture, Bitwise_and,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::bitwise_and(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// bitwise_xor ////////////////////////

typedef Size_MatType BitwiseXorFixture;

OCL_PERF_TEST_P(BitwiseXorFixture, Bitwise_xor,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::bitwise_xor(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// bitwise_or ////////////////////////

typedef Size_MatType BitwiseOrFixture;

OCL_PERF_TEST_P(BitwiseOrFixture, Bitwise_or,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::bitwise_or(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// bitwise_not ////////////////////////

typedef Size_MatType BitwiseNotFixture;

OCL_PERF_TEST_P(BitwiseNotFixture, Bitwise_not,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::bitwise_not(src, dst);

    SANITY_CHECK(dst);
}

///////////// compare ////////////////////////

CV_ENUM(CmpCode, CMP_LT, CMP_LE, CMP_EQ, CMP_NE, CMP_GE, CMP_GT)

typedef tuple<Size, MatType, CmpCode> CompareParams;
typedef TestBaseWithParam<CompareParams> CompareFixture;

OCL_PERF_TEST_P(CompareFixture, Compare,
            ::testing::Combine(OCL_TEST_SIZES,
                               OCL_TEST_TYPES_134, CmpCode::all()))
{
    const CompareParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int cmpCode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, CV_8UC(CV_MAT_CN(type)));
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::compare(src1, src2, dst, cmpCode);

    SANITY_CHECK(dst);
}

OCL_PERF_TEST_P(CompareFixture, CompareScalar,
            ::testing::Combine(OCL_TEST_SIZES,
                               OCL_PERF_ENUM((MatType)CV_32FC1), // TODO: OCL_TEST_TYPES_134
                               CmpCode::all()))
{
    const CompareParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int cmpCode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), dst(srcSize, CV_8UC(CV_MAT_CN(type)));
    declare.in(src1, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::compare(src1, 32, dst, cmpCode);

    SANITY_CHECK(dst);
}

///////////// pow ////////////////////////

typedef Size_MatType PowFixture;

OCL_PERF_TEST_P(PowFixture, Pow, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    randu(src, 0, 100);
    declare.in(src).out(dst);

    OCL_TEST_CYCLE() cv::pow(src, 2.17, dst);

    SANITY_CHECK(dst, 1.5e-6, ERROR_RELATIVE);
}

///////////// iPow ////////////////////////
OCL_PERF_TEST_P(PowFixture, iPow, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_8UC3, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    randu(src, 0, 100);
    declare.in(src).out(dst);

    OCL_TEST_CYCLE() cv::pow(src, 3, dst);

    SANITY_CHECK_NOTHING();
}
///////////// AddWeighted////////////////////////

typedef Size_MatType AddWeightedFixture;

OCL_PERF_TEST_P(AddWeightedFixture, AddWeighted,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), depth = CV_MAT_DEPTH(type);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);
    double alpha = 2.0, beta = 1.0, gama = 3.0;

    OCL_TEST_CYCLE() cv::addWeighted(src1, alpha, src2, beta, gama, dst);

    if (depth >= CV_32F)
        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    else
        SANITY_CHECK(dst);
}

///////////// Sqrt ///////////////////////

typedef Size_MatType SqrtFixture;

OCL_PERF_TEST_P(SqrtFixture, Sqrt, ::testing::Combine(
                OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    randu(src, 0, 1000);
    declare.in(src).out(dst);

    OCL_TEST_CYCLE() cv::sqrt(src, dst);

    // To square root 32 bit floats we use native_sqrt, which has implementation
    // defined accuracy. We know intel devices have accurate native_sqrt, but
    // otherwise stick to a relaxed sanity check. For types larger than 32 bits
    // we can do the accuracy check for all devices as normal.
    if (CV_MAT_DEPTH(type) > CV_32F || !ocl::useOpenCL() ||
        ocl::Device::getDefault().isIntel())
        SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
    else
        SANITY_CHECK(dst, 1);
}

///////////// SetIdentity ////////////////////////

typedef Size_MatType SetIdentityFixture;

OCL_PERF_TEST_P(SetIdentityFixture, SetIdentity,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat dst(srcSize, type);
    declare.out(dst);

    OCL_TEST_CYCLE() cv::setIdentity(dst, cv::Scalar::all(181));

    SANITY_CHECK(dst);
}

///////////// MeanStdDev ////////////////////////

typedef Size_MatType MeanStdDevFixture;

OCL_PERF_TEST_P(MeanStdDevFixture, MeanStdDev,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const double eps = 2e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    Scalar mean, stddev;
    declare.in(src, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::meanStdDev(src, mean, stddev);

    double mean0 = mean[0], mean1 = mean[1], mean2 = mean[2], mean3 = mean[3];
    double stddev0 = stddev[0], stddev1 = stddev[1], stddev2 = stddev[2], stddev3 = stddev[3];

    SANITY_CHECK(mean0, eps, ERROR_RELATIVE);
    SANITY_CHECK(mean1, eps, ERROR_RELATIVE);
    SANITY_CHECK(mean2, eps, ERROR_RELATIVE);
    SANITY_CHECK(mean3, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev0, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev1, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev2, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev3, eps, ERROR_RELATIVE);
}

OCL_PERF_TEST_P(MeanStdDevFixture, MeanStdDevWithMask,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const double eps = 2e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), mask(srcSize, CV_8UC1);
    Scalar mean, stddev;
    declare.in(src, mask, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::meanStdDev(src, mean, stddev, mask);

    double mean0 = mean[0], mean1 = mean[1], mean2 = mean[2], mean3 = mean[3];
    double stddev0 = stddev[0], stddev1 = stddev[1], stddev2 = stddev[2], stddev3 = stddev[3];

    SANITY_CHECK(mean0, eps, ERROR_RELATIVE);
    SANITY_CHECK(mean1, eps, ERROR_RELATIVE);
    SANITY_CHECK(mean2, eps, ERROR_RELATIVE);
    SANITY_CHECK(mean3, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev0, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev1, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev2, eps, ERROR_RELATIVE);
    SANITY_CHECK(stddev3, eps, ERROR_RELATIVE);
}

///////////// Norm ////////////////////////

CV_ENUM(NormType, NORM_INF, NORM_L1, NORM_L2)

typedef tuple<Size, MatType, NormType> NormParams;
typedef TestBaseWithParam<NormParams> NormFixture;

OCL_PERF_TEST_P(NormFixture, Norm1Arg,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES_134, NormType::all()))
{
    const NormParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int normType = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type);
    double res;
    declare.in(src1, WARMUP_RNG);

    OCL_TEST_CYCLE() res = cv::norm(src1, normType);

    SANITY_CHECK(res, 1e-5, ERROR_RELATIVE);
}

OCL_PERF_TEST_P(NormFixture, Norm,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES_134, NormType::all()))
{
    const NormParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int normType = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type);
    double res;
    declare.in(src1, src2, WARMUP_RNG);

    OCL_TEST_CYCLE() res = cv::norm(src1, src2, normType);

    SANITY_CHECK(res, 1e-5, ERROR_RELATIVE);
}

OCL_PERF_TEST_P(NormFixture, NormRel,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES_134, NormType::all()))
{
    const NormParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int normType = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type);
    double res;
    declare.in(src1, src2, WARMUP_RNG);

    OCL_TEST_CYCLE() res = cv::norm(src1, src2, normType | cv::NORM_RELATIVE);

    SANITY_CHECK(res, 1e-5, ERROR_RELATIVE);
}

///////////// UMat::dot ////////////////////////

typedef Size_MatType UMatDotFixture;

OCL_PERF_TEST_P(UMatDotFixture, UMatDot,
            ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                               OCL_TEST_TYPES_134))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    double r = 0.0;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG);

    OCL_TEST_CYCLE() r = src1.dot(src2);

    SANITY_CHECK(r, 1e-5, ERROR_RELATIVE);
}

///////////// Repeat ////////////////////////

typedef Size_MatType RepeatFixture;

OCL_PERF_TEST_P(RepeatFixture, Repeat,
            ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3), OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), nx = 2, ny = 2;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(Size(srcSize.width * nx, srcSize.height * ny), type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::repeat(src, nx, ny, dst);

    SANITY_CHECK(dst);
}

///////////// Min ////////////////////////

typedef Size_MatType MinFixture;

OCL_PERF_TEST_P(MinFixture, Min,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::min(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// Max ////////////////////////

typedef Size_MatType MaxFixture;

OCL_PERF_TEST_P(MaxFixture, Max,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::max(src1, src2, dst);

    SANITY_CHECK(dst);
}

///////////// InRange ////////////////////////

typedef Size_MatType InRangeFixture;

OCL_PERF_TEST_P(InRangeFixture, InRange,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), lb(srcSize, type), ub(srcSize, type), dst(srcSize, CV_8UC1);
    declare.in(src, lb, ub, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::inRange(src, lb, ub, dst);

    SANITY_CHECK(dst);
}

///////////// Normalize ////////////////////////

CV_ENUM(NormalizeModes, NORM_MINMAX, NORM_L2, NORM_L1, NORM_INF)

typedef tuple<Size, MatType, NormalizeModes> NormalizeParams;
typedef TestBaseWithParam<NormalizeParams> NormalizeFixture;

OCL_PERF_TEST_P(NormalizeFixture, Normalize,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134,
                                   NormalizeModes::all()))
{
    const NormalizeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), mode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::normalize(src, dst, 10, 110, mode);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(NormalizeFixture, NormalizeWithMask,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1),
                                   NormalizeModes::all()))
{
    const NormalizeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), mode = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), mask(srcSize, CV_8UC1), dst(srcSize, type);
    declare.in(src, mask, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::normalize(src, dst, 10, 110, mode, -1, mask);

    SANITY_CHECK_NOTHING();
}

///////////// ConvertScaleAbs ////////////////////////

typedef Size_MatType ConvertScaleAbsFixture;

OCL_PERF_TEST_P(ConvertScaleAbsFixture, ConvertScaleAbs,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), cn = CV_MAT_CN(type);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, CV_8UC(cn));
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::convertScaleAbs(src, dst, 0.5, 2);

    SANITY_CHECK(dst, 1); // CV_8U
}

///////////// PatchNaNs ////////////////////////

template<typename _Tp>
_Tp randomNan(RNG& rng);

template<>
float randomNan(RNG& rng)
{
    uint32_t r = rng.next();
    Cv32suf v;
    v.u = r;
    // exp & set a bit to avoid zero mantissa
    v.u = v.u | 0x7f800001;
    return v.f;
}

template<>
double randomNan(RNG& rng)
{
    uint32_t r0 = rng.next();
    uint32_t r1 = rng.next();
    Cv64suf v;
    v.u = (uint64_t(r0) << 32) | uint64_t(r1);
    // exp &set a bit to avoid zero mantissa
    v.u = v.u | 0x7ff0000000000001;
    return v.f;
}

typedef Size_MatType PatchNaNsFixture;

OCL_PERF_TEST_P(PatchNaNsFixture, PatchNaNs,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC3, CV_64FC4)))
{
    const Size_MatType_t params = GetParam();
    Size srcSize = get<0>(params);
    const int type = get<1>(params), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    declare.in(src, WARMUP_RNG).out(src);

    // generating NaNs
    {
        Mat src_ = src.getMat(ACCESS_RW);
        srcSize.width *= cn;
        RNG& rng = theRNG();
        for (int y = 0; y < srcSize.height; ++y)
        {
            float  *const ptrf = src_.ptr<float>(y);
            double *const ptrd = src_.ptr<double>(y);
            for (int x = 0; x < srcSize.width; ++x)
            {
                if (depth == CV_32F)
                {
                    ptrf[x] = (x + y) % 2 == 0 ? randomNan<float >(rng) : ptrf[x];
                }
                else if (depth == CV_64F)
                {
                    ptrd[x] = (x + y) % 2 == 0 ? randomNan<double>(rng) : ptrd[x];
                }
            }
        }
    }

    OCL_TEST_CYCLE() cv::patchNaNs(src, 17.7);

    SANITY_CHECK(src);
}

////////////// finiteMask ////////////////////////

typedef Size_MatType FiniteMaskFixture;

OCL_PERF_TEST_P(FiniteMaskFixture, FiniteMask,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32FC1, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC3, CV_64FC4)))
{
    const Size_MatType_t params = GetParam();
    Size srcSize = get<0>(params);
    const int type = get<1>(params), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type);
    UMat mask(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(mask);

    // generating NaNs
    {
        Mat src_ = src.getMat(ACCESS_RW);
        srcSize.width *= cn;
        const softfloat  fpinf = softfloat ::inf();
        const softfloat  fninf = softfloat ::inf().setSign(true);
        const softdouble dpinf = softdouble::inf();
        const softdouble dninf = softdouble::inf().setSign(true);
        RNG& rng = theRNG();
        for (int y = 0; y < srcSize.height; ++y)
        {
            float  *const ptrf = src_.ptr<float>(y);
            double *const ptrd = src_.ptr<double>(y);
            for (int x = 0; x < srcSize.width; ++x)
            {
                int rem = (x + y) % 10;
                if (depth == CV_32F)
                {
                    ptrf[x] = rem <  4 ? randomNan<float >(rng) :
                              rem == 5 ? (float )((x + y)%2 ? fpinf : fninf) : ptrf[x];
                }
                else if (depth == CV_64F)
                {
                    ptrd[x] = rem <  4 ? randomNan<double>(rng) :
                              rem == 5 ? (double)((x + y)%2 ? dpinf : dninf) : ptrd[x];
                }
            }
        }
    }

    OCL_TEST_CYCLE() cv::finiteMask(src, mask);

    SANITY_CHECK(mask);
}

///////////// ScaleAdd ////////////////////////

typedef Size_MatType ScaleAddFixture;

OCL_PERF_TEST_P(ScaleAddFixture, ScaleAdd,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::scaleAdd(src1, 0.6, src2, dst);

    SANITY_CHECK(dst, 1e-6);
}

///////////// Transform ////////////////////////

typedef Size_MatType TransformFixture;

OCL_PERF_TEST_P(TransformFixture, Transform,
                ::testing::Combine(OCL_TEST_SIZES,
                ::testing::Values(CV_8UC3, CV_8SC3, CV_16UC3, CV_16SC3, CV_32SC3, CV_32FC3, CV_64FC3)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    const float transform[] = { 0.5f,           0.f, 0.86602540378f, 128,
                                0.f,            1.f, 0.f,            -64,
                                0.86602540378f, 0.f, 0.5f,            32,};
    Mat mtx(Size(4, 3), CV_32FC1, (void*)transform);

    UMat src(srcSize, type), dst(srcSize, type);
    randu(src, 0, 30);
    declare.in(src).out(dst);

    OCL_TEST_CYCLE() cv::transform(src, dst, mtx);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

///////////// PSNR ////////////////////////

typedef Size_MatType PSNRFixture;

OCL_PERF_TEST_P(PSNRFixture, PSNR,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    double psnr = 0;
    UMat src1(srcSize, type), src2(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG);

    OCL_TEST_CYCLE() psnr = cv::PSNR(src1, src2);

    SANITY_CHECK(psnr, 1e-4, ERROR_RELATIVE);
}

///////////// Reduce ////////////////////////

CV_ENUM(ReduceMinMaxOp, REDUCE_MIN, REDUCE_MAX)

typedef tuple<Size, std::pair<MatType, MatType>, int, ReduceMinMaxOp> ReduceMinMaxParams;
typedef TestBaseWithParam<ReduceMinMaxParams> ReduceMinMaxFixture;

OCL_PERF_TEST_P(ReduceMinMaxFixture, Reduce,
                ::testing::Combine(OCL_TEST_SIZES,
                                   OCL_PERF_ENUM(std::make_pair<MatType, MatType>(CV_8UC1, CV_8UC1),
                                                 std::make_pair<MatType, MatType>(CV_32FC4, CV_32FC4)),
                                   OCL_PERF_ENUM(0, 1),
                                   ReduceMinMaxOp::all()))
{
    const ReduceMinMaxParams params = GetParam();
    const std::pair<MatType, MatType> types = get<1>(params);
    const int stype = types.first, dtype = types.second,
            dim = get<2>(params), op = get<3>(params);
    const Size srcSize = get<0>(params),
            dstSize(dim == 0 ? srcSize.width : 1, dim == 0 ? 1 : srcSize.height);

    checkDeviceMaxMemoryAllocSize(srcSize, stype);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, stype), dst(dstSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::reduce(src, dst, dim, op, dtype);

    SANITY_CHECK_NOTHING();
}

CV_ENUM(ReduceAccOp, REDUCE_SUM, REDUCE_AVG, REDUCE_SUM2)

typedef tuple<Size, std::pair<MatType, MatType>, int, ReduceAccOp> ReduceAccParams;
typedef TestBaseWithParam<ReduceAccParams> ReduceAccFixture;

OCL_PERF_TEST_P(ReduceAccFixture, Reduce,
                ::testing::Combine(OCL_TEST_SIZES,
                                   OCL_PERF_ENUM(std::make_pair<MatType, MatType>(CV_8UC4, CV_32SC4),
                                                 std::make_pair<MatType, MatType>(CV_32FC1, CV_32FC1)),
                                   OCL_PERF_ENUM(0, 1),
                                   ReduceAccOp::all()))
{
    const ReduceAccParams params = GetParam();
    const std::pair<MatType, MatType> types = get<1>(params);
    const int stype = types.first, dtype = types.second,
            dim = get<2>(params), op = get<3>(params);
    const Size srcSize = get<0>(params),
            dstSize(dim == 0 ? srcSize.width : 1, dim == 0 ? 1 : srcSize.height);

    checkDeviceMaxMemoryAllocSize(srcSize, stype);
    checkDeviceMaxMemoryAllocSize(srcSize, dtype);

    UMat src(srcSize, stype), dst(dstSize, dtype);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::reduce(src, dst, dim, op, dtype);

    SANITY_CHECK_NOTHING();
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
