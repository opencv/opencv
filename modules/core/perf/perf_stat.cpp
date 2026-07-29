// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// Masked variants: 8U/16U/32F x C1/C3/C4.
#define TYPICAL_MAT_TYPES_STAT_MASK CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4
// sum/mean/meanStdDev additionally cover 16S.
#define TYPICAL_MAT_TYPES_STAT      TYPICAL_MAT_TYPES_STAT_MASK, CV_16SC1, CV_16SC3, CV_16SC4
// Single-channel-only reductions (shared by countNonZero/hasNonZero).
#define TYPICAL_MAT_TYPES_NONZERO   CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1

PERF_TEST_P(Size_MatType, sum, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ),
                testing::Values( TYPICAL_MAT_TYPES_STAT ) ))
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat arr(sz, type);
    Scalar s;

    declare.in(arr, WARMUP_RNG).out(s);

    TEST_CYCLE() s = sum(arr);

    SANITY_CHECK(s, 1e-6, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType, mean, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ),
                testing::Values( TYPICAL_MAT_TYPES_STAT ) ))
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src(sz, type);
    Scalar s;

    declare.in(src, WARMUP_RNG).out(s);

    TEST_CYCLE() s = cv::mean(src);

    SANITY_CHECK(s, 1e-5);
}

PERF_TEST_P(Size_MatType, mean_mask, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ),
                testing::Values( TYPICAL_MAT_TYPES_STAT_MASK ) ))
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src(sz, type);
    Mat mask = Mat::ones(src.size(), CV_8U);
    Scalar s;

    declare.in(src, WARMUP_RNG).in(mask).out(s);

    TEST_CYCLE() s = cv::mean(src, mask);

    SANITY_CHECK(s, 5e-5);
}

PERF_TEST_P(Size_MatType, meanStdDev, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ),
                testing::Values( TYPICAL_MAT_TYPES_STAT_MASK ) ))
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    Scalar mean;
    Scalar dev;

    declare.in(src, WARMUP_RNG).out(mean, dev);

    TEST_CYCLE() meanStdDev(src, mean, dev);

    SANITY_CHECK(mean, 1e-5, ERROR_RELATIVE);
    SANITY_CHECK(dev, 1e-5, ERROR_RELATIVE);
}

PERF_TEST_P(Size_MatType, meanStdDev_mask, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ),
                testing::Values( TYPICAL_MAT_TYPES_STAT_MASK ) ))
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    Mat mask = Mat::ones(sz, CV_8U);
    Scalar mean;
    Scalar dev;

    declare.in(src, WARMUP_RNG).in(mask).out(mean, dev);

    TEST_CYCLE() meanStdDev(src, mean, dev, mask);

    SANITY_CHECK(mean, 1e-5);
    SANITY_CHECK(dev, 1e-5);
}

PERF_TEST_P(Size_MatType, countNonZero, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( TYPICAL_MAT_TYPES_NONZERO ) ))
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    int cnt = 0;

    declare.in(src, WARMUP_RNG);

    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) cnt = countNonZero(src);

    SANITY_CHECK(cnt);
}

PERF_TEST_P(Size_MatType, hasNonZero, testing::Combine( testing::Values( TYPICAL_MAT_SIZES ), testing::Values( TYPICAL_MAT_TYPES_NONZERO ) ))
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    /*bool hnz = false;*/

    declare.in(src, WARMUP_RNG);

    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) /*hnz =*/ hasNonZero(src);

    SANITY_CHECK_NOTHING();
}

} // namespace
