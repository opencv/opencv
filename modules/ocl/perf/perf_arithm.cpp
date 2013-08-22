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
//     and/or other oclMaterials provided with the distribution.
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

CV_ENUM(LUTMatTypes, CV_8UC1, CV_8UC3)

typedef tuple<Size, LUTMatTypes> LUTParams;
typedef TestBaseWithParam<LUTParams> LUTFixture;

PERF_TEST_P(LUTFixture, LUT,
          ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                             LUTMatTypes::all()))
{
    // getting params
    LUTParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    const std::string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type), lut(1, 256, CV_8UC1);
    int dstType = CV_MAKETYPE(lut.depth(), src.channels());
    Mat dst(srcSize, dstType);

    randu(lut, 0, 2);
    declare.in(src, WARMUP_RNG).in(lut).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclLut(lut), oclDst(srcSize, dstType);

        TEST_CYCLE() cv::ocl::LUT(oclSrc, oclLut, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::LUT(src, lut, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Exp ////////////////////////

typedef TestBaseWithParam<Size> ExpFixture;

PERF_TEST_P(ExpFixture, Exp, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();

    const std::string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, CV_32FC1), dst(srcSize, CV_32FC1);
    declare.in(src).out(dst);
    randu(src, 5, 16);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        TEST_CYCLE() cv::ocl::exp(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 0.3);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::exp(src, dst);

        SANITY_CHECK(dst, 0.3);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// LOG ////////////////////////

typedef TestBaseWithParam<Size> LogFixture;

PERF_TEST_P(LogFixture, Log, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const std::string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, CV_32F), dst(srcSize, src.type());
    randu(src, 1, 10);
    declare.in(src).out(dst);

    if (srcSize == OCL_SIZE_4000)
        declare.time(3.6);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        TEST_CYCLE() cv::ocl::log(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::log(src, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Add ////////////////////////

CV_ENUM(AddMatTypes, CV_8UC1, CV_32FC1)

typedef tuple<Size, AddMatTypes> AddParams;
typedef TestBaseWithParam<AddParams> AddFixture;

PERF_TEST_P(AddFixture, Add,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               AddMatTypes::all()))
{
    // getting params
    AddParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    randu(src1, 0, 1);
    randu(src2, 0, 1);
    declare.in(src1, src2).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::add(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::add(src1, src2, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Mul ////////////////////////

CV_ENUM(MulMatTypes, CV_8UC1, CV_8UC4)

typedef tuple<Size, MulMatTypes> MulParams;
typedef TestBaseWithParam<MulParams> MulFixture;

PERF_TEST_P(MulFixture, Mul, ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                                       MulMatTypes::all()))
{
    // getting params
    MulParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    randu(src1, 0, 256);
    randu(src2, 0, 256);
    declare.in(src1, src2).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::multiply(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::multiply(src1, src2, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Div ////////////////////////

typedef MulMatTypes DivMatTypes;
typedef tuple<Size, DivMatTypes> DivParams;
typedef TestBaseWithParam<DivParams> DivFixture;

PERF_TEST_P(DivFixture, Div, ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                                       DivMatTypes::all()))
{
    // getting params
    DivParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

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
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::divide(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::divide(src1, src2, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Absdiff ////////////////////////

typedef MulMatTypes AbsDiffMatTypes;
typedef tuple<Size, AbsDiffMatTypes> AbsDiffParams;
typedef TestBaseWithParam<AbsDiffParams> AbsDiffFixture;

PERF_TEST_P(AbsDiffFixture, Absdiff, ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                                               AbsDiffMatTypes::all()))
{
    // getting params
    AbsDiffParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).in(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::absdiff(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::absdiff(src1, src2, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// CartToPolar ////////////////////////

typedef TestBaseWithParam<Size> CartToPolarFixture;

PERF_TEST_P(CartToPolarFixture, CartToPolar, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst1(srcSize, CV_32FC1), dst2(srcSize, CV_32FC1);
    declare.in(src1, src2).out(dst1, dst2);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (srcSize == OCL_SIZE_4000)
        declare.time(3.6);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst1(srcSize, src1.type()), oclDst2(srcSize, src1.type());

        TEST_CYCLE() cv::ocl::cartToPolar(oclSrc1, oclSrc2, oclDst1, oclDst2);

        oclDst1.download(dst1);
        oclDst2.download(dst2);

        SANITY_CHECK(dst1, 5e-3);
        SANITY_CHECK(dst2, 5e-3);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::cartToPolar(src1, src2, dst1, dst2);

        SANITY_CHECK(dst1, 5e-3);
        SANITY_CHECK(dst2, 5e-3);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// PolarToCart ////////////////////////

typedef TestBaseWithParam<Size> PolarToCartFixture;

PERF_TEST_P(PolarToCartFixture, PolarToCart, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst1(srcSize, CV_32FC1), dst2(srcSize, CV_32FC1);
    declare.in(src1, src2).out(dst1, dst2);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    if (srcSize == OCL_SIZE_4000)
        declare.time(5.4);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst1(srcSize, src1.type()), oclDst2(srcSize, src1.type());

        TEST_CYCLE() cv::ocl::polarToCart(oclSrc1, oclSrc2, oclDst1, oclDst2);

        oclDst1.download(dst1);
        oclDst2.download(dst2);

        SANITY_CHECK(dst1, 5e-5);
        SANITY_CHECK(dst2, 5e-5);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::polarToCart(src1, src2, dst1, dst2);

        SANITY_CHECK(dst1, 5e-5);
        SANITY_CHECK(dst2, 5e-5);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Magnitude ////////////////////////

typedef TestBaseWithParam<Size> MagnitudeFixture;

PERF_TEST_P(MagnitudeFixture, Magnitude, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst(srcSize, CV_32FC1);
    randu(src1, 0, 1);
    randu(src2, 0, 1);
    declare.in(src1, src2).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst(srcSize, src1.type());

        TEST_CYCLE() cv::ocl::magnitude(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-6);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::magnitude(src1, src2, dst);

        SANITY_CHECK(dst, 1e-6);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Transpose ////////////////////////

typedef MulMatTypes TransposeMatTypes;
typedef tuple<Size, TransposeMatTypes> TransposeParams;
typedef TestBaseWithParam<TransposeParams> TransposeFixture;

PERF_TEST_P(TransposeFixture, Transpose,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               TransposeMatTypes::all()))
{
    // getting params
    TransposeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::transpose(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::transpose(src, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Flip ////////////////////////

typedef MulMatTypes FlipMatTypes;
typedef tuple<Size, FlipMatTypes> FlipParams;
typedef TestBaseWithParam<FlipParams> FlipFixture;

PERF_TEST_P(FlipFixture, Flip,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                                         FlipMatTypes::all()))
{
    // getting params
    TransposeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::flip(oclSrc, oclDst, 0);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::flip(src, dst, 0);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// minMax ////////////////////////

typedef AddMatTypes minMaxMatTypes;
typedef tuple<Size, minMaxMatTypes> minMaxParams;
typedef TestBaseWithParam<minMaxParams> minMaxFixture;

PERF_TEST_P(minMaxFixture, minMax,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               minMaxMatTypes::all()))
{
    // getting params
    minMaxParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type);
    declare.in(src, WARMUP_RNG);

    double min_val = 0.0, max_val = 0.0;

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src);

        TEST_CYCLE() cv::ocl::minMax(oclSrc, &min_val, &max_val);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
    else if (impl == "plain")
    {
        Point min_loc, max_loc;

        TEST_CYCLE() cv::minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// minMaxLoc ////////////////////////

typedef AddMatTypes minMaxLocMatTypes;
typedef tuple<Size, minMaxMatTypes> minMaxLocParams;
typedef TestBaseWithParam<minMaxLocParams> minMaxLocFixture;

PERF_TEST_P(minMaxLocFixture, minMaxLoc,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               minMaxLocMatTypes::all()))
{
    // getting params
    minMaxLocParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type);
    randu(src, 0, 1);
    declare.in(src);

    double min_val = 0.0, max_val = 0.0;
    Point min_loc, max_loc;

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src);

        TEST_CYCLE() cv::ocl::minMaxLoc(oclSrc, &min_val, &max_val, &min_loc, &max_loc);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::minMaxLoc(src, &min_val, &max_val, &min_loc, &max_loc);

        ASSERT_GE(max_val, min_val);
        SANITY_CHECK(min_val);
        SANITY_CHECK(max_val);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Sum ////////////////////////

CV_ENUM(SumMatTypes, CV_8UC1, CV_32SC1)

typedef tuple<Size, SumMatTypes> SumParams;
typedef TestBaseWithParam<SumParams> SumFixture;

PERF_TEST_P(SumFixture, Sum,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               SumMatTypes::all()))
{
    // getting params
    SumParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type);
    Scalar result;
    randu(src, 0, 60);
    declare.in(src);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src);

        TEST_CYCLE() result = cv::ocl::sum(oclSrc);

        SANITY_CHECK(result);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() result = cv::sum(src);

        SANITY_CHECK(result);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// countNonZero ////////////////////////

CV_ENUM(countNonZeroMatTypes, CV_8UC1, CV_32FC1)

typedef tuple<Size, countNonZeroMatTypes> countNonZeroParams;
typedef TestBaseWithParam<countNonZeroParams> countNonZeroFixture;

PERF_TEST_P(countNonZeroFixture, countNonZero,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               countNonZeroMatTypes::all()))
{
    // getting params
    countNonZeroParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type);
    int result = 0;
    randu(src, 0, 256);
    declare.in(src);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src);

        TEST_CYCLE() result = cv::ocl::countNonZero(oclSrc);

        SANITY_CHECK(result);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() result = cv::countNonZero(src);

        SANITY_CHECK(result);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Phase ////////////////////////

typedef TestBaseWithParam<Size> PhaseFixture;

PERF_TEST_P(PhaseFixture, Phase, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst(srcSize, CV_32FC1);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclDst(srcSize, src1.type());

        TEST_CYCLE() cv::ocl::phase(oclSrc1, oclSrc2, oclDst, 1);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-2);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::phase(src1, src2, dst, 1);

        SANITY_CHECK(dst, 1e-2);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// bitwise_and////////////////////////

typedef SumMatTypes BitwiseAndMatTypes;
typedef tuple<Size, BitwiseAndMatTypes> BitwiseAndParams;
typedef TestBaseWithParam<BitwiseAndParams> BitwiseAndFixture;

PERF_TEST_P(BitwiseAndFixture, bitwise_and,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               BitwiseAndMatTypes::all()))
{
    // getting params
    BitwiseAndParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2).out(dst);
    randu(src1, 0, 256);
    randu(src2, 0, 256);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, src1.type());

        TEST_CYCLE() cv::ocl::bitwise_and(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::bitwise_and(src1, src2, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// bitwise_not////////////////////////

typedef SumMatTypes BitwiseNotMatTypes;
typedef tuple<Size, BitwiseNotMatTypes> BitwiseNotParams;
typedef TestBaseWithParam<BitwiseNotParams> BitwiseNotFixture;

PERF_TEST_P(BitwiseAndFixture, bitwise_not,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               BitwiseAndMatTypes::all()))
{
    // getting params
    BitwiseNotParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::bitwise_not(oclSrc, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::bitwise_not(src, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// compare////////////////////////

typedef countNonZeroMatTypes CompareMatTypes;
typedef tuple<Size, CompareMatTypes> CompareParams;
typedef TestBaseWithParam<CompareParams> CompareFixture;

PERF_TEST_P(CompareFixture, compare,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               CompareMatTypes::all()))
{
    // getting params
    CompareParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, CV_8UC1);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, CV_8UC1);

        TEST_CYCLE() cv::ocl::compare(oclSrc1, oclSrc2, oclDst, CMP_EQ);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::compare(src1, src2, dst, CMP_EQ);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// pow ////////////////////////

typedef TestBaseWithParam<Size> PowFixture;

PERF_TEST_P(PowFixture, pow, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    // creating src data
    Mat src(srcSize, CV_32F), dst(srcSize, CV_32F);
    declare.in(src, WARMUP_RNG).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, src.type());

        TEST_CYCLE() cv::ocl::pow(oclSrc, -2.0, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 5e-2);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::pow(src, -2.0, dst);

        SANITY_CHECK(dst, 5e-2);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// MagnitudeSqr////////////////////////

typedef TestBaseWithParam<Size> MagnitudeSqrFixture;

PERF_TEST_P(MagnitudeSqrFixture, MagnitudeSqr, OCL_TYPICAL_MAT_SIZES)
{
    // getting params
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            dst(srcSize, CV_32FC1);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, src1.type());

        TEST_CYCLE() cv::ocl::magnitudeSqr(oclSrc1, oclSrc2, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else if (impl == "plain")
    {
        ASSERT_EQ(1, src1.channels());

        TEST_CYCLE()
        {
            for (int y = 0; y < srcSize.height; ++y)
            {
                const float * const src1Data = reinterpret_cast<float *>(src1.data + src1.step * y);
                const float * const src2Data = reinterpret_cast<float *>(src2.data + src2.step * y);
                float * const dstData = reinterpret_cast<float *>(dst.data + dst.step * y);
                for (int x = 0; x < srcSize.width; ++x)
                {
                    float t0 = src1Data[x] * src1Data[x];
                    float t1 = src2Data[x] * src2Data[x];
                    dstData[x] = t0 + t1;
                }
            }
        }

        SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// AddWeighted////////////////////////

typedef countNonZeroMatTypes AddWeightedMatTypes;
typedef tuple<Size, AddWeightedMatTypes> AddWeightedParams;
typedef TestBaseWithParam<AddWeightedParams> AddWeightedFixture;

PERF_TEST_P(AddWeightedFixture, AddWeighted,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               AddWeightedMatTypes::all()))
{
    // getting params
    AddWeightedParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const string impl = getSelectedImpl();

    // creating src data
    Mat src1(srcSize, type), src2(srcSize, type), dst(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG).out(dst);
    double alpha = 2.0, beta = 1.0, gama = 3.0;

    // select implementation
    if (impl == "ocl")
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::addWeighted(oclSrc1, alpha, oclSrc2, beta, gama, oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::addWeighted(src1, alpha, src2, beta, gama, dst);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}
