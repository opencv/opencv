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
using std::tr1::tuple;
using std::tr1::get;


///////////// MinMax ////////////////////////

typedef Size_MatType MinMaxFixture;

PERF_TEST_P(MinMaxFixture, MinMax,
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

///////////// MinMaxLoc ////////////////////////

typedef Size_MatType MinMaxLocFixture;

OCL_PERF_TEST_P(MinMaxLocFixture, MinMaxLoc,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
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

OCL_PERF_TEST_P(SumFixture, Sum,
                ::testing::Combine(OCL_TEST_SIZES,
                                   OCL_TEST_TYPES))
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

        SANITY_CHECK(result, 1e-6, ERROR_RELATIVE);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() result = cv::sum(src);

        SANITY_CHECK(result, 1e-6, ERROR_RELATIVE);
    }
    else
        OCL_PERF_ELSE
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

///////////// meanStdDev ////////////////////////

typedef Size_MatType MeanStdDevFixture;

OCL_PERF_TEST_P(MeanStdDevFixture, MeanStdDev,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type);
    Scalar mean, stddev;
    randu(src, 0, 256);
    declare.in(src);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() cv::ocl::meanStdDev(oclSrc, mean, stddev);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::meanStdDev(src, mean, stddev);
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK_NOTHING();
//    SANITY_CHECK(mean, 1e-6, ERROR_RELATIVE);
//    SANITY_CHECK(stddev, 1e-6, ERROR_RELATIVE);
}

///////////// norm////////////////////////

CV_ENUM(NormType, NORM_INF, NORM_L1, NORM_L2)

typedef std::tr1::tuple<Size, MatType, NormType> NormParams;
typedef TestBaseWithParam<NormParams> NormFixture;

OCL_PERF_TEST_P(NormFixture, Norm,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES, NormType::all()))
{
    const NormParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int normType = get<2>(params);
    perf::ERROR_TYPE errorType = type != NORM_INF ? ERROR_RELATIVE : ERROR_ABSOLUTE;
    double eps = 1e-5, value;

    Mat src1(srcSize, type), src2(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2);

        OCL_TEST_CYCLE() value = cv::ocl::norm(oclSrc1, oclSrc2, normType);

        SANITY_CHECK(value, eps, errorType);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() value = cv::norm(src1, src2, normType);

        SANITY_CHECK(value, eps, errorType);
    }
    else
        OCL_PERF_ELSE
}
