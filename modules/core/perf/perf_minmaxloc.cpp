// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

PERF_TEST_P(Size_MatType, minMaxLoc, testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1,  CV_32FC1, CV_64FC1)
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    double minVal, maxVal;
    Point minLoc, maxLoc;

    if (matType == CV_8U)
        randu(src, 1, 254 /*do not include 0 and 255 to avoid early exit on 1 byte data*/);
    else if (matType == CV_8S)
        randu(src, -127, 126);
    else
        warmup(src, WARMUP_RNG);

    declare.in(src);

    TEST_CYCLE() minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

    SANITY_CHECK(minVal, 1e-12);
    SANITY_CHECK(maxVal, 1e-12);
}

// minMaxIdx (n-dim min/max + linear index), with the null-pointer combinations
// IPP exercises: which of {min, max, min-idx, max-idx} outputs are requested.
enum
{
    MMI_MIN     = 0x1,
    MMI_MAX     = 0x2,
    MMI_IDX     = 0x4,
    MMI_MIN_IDX = MMI_MIN | MMI_IDX,
    MMI_MAX_IDX = MMI_MAX | MMI_IDX,
    MMI_MIN_MAX = MMI_MIN | MMI_MAX,
    MMI_ALL     = MMI_MIN_IDX | MMI_MAX_IDX,
};
CV_ENUM(MinMaxType, MMI_ALL, MMI_MIN_MAX, MMI_MIN_IDX, MMI_MAX_IDX)

#define MINMAXIDX_TYPES CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1

typedef tuple<Size, MatType, MinMaxType> Size_MatType_MinMaxType_t;
typedef perf::TestBaseWithParam<Size_MatType_MinMaxType_t> Size_MatType_MinMaxType;

PERF_TEST_P(Size_MatType_MinMaxType, minMaxIdx, testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(MINMAXIDX_TYPES),
                 MinMaxType::all()
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int flags = get<2>(GetParam());

    Mat src(sz, matType);
    double minVal = 0, maxVal = 0;
    int minIdx[2] = {0}, maxIdx[2] = {0};

    double *pMinVal = (flags & MMI_MIN) ? &minVal : NULL;
    double *pMaxVal = (flags & MMI_MAX) ? &maxVal : NULL;
    int    *pMinIdx = ((flags & MMI_MIN) && (flags & MMI_IDX)) ? minIdx : NULL;
    int    *pMaxIdx = ((flags & MMI_MAX) && (flags & MMI_IDX)) ? maxIdx : NULL;

    // middle of the range to avoid early-exit short-cuts
    randu(src, 1, 155);
    declare.in(src);

    TEST_CYCLE() cv::minMaxIdx(src, pMinVal, pMaxVal, pMinIdx, pMaxIdx);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_MinMaxType, minMaxIdx_mask, testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(MINMAXIDX_TYPES),
                 MinMaxType::all()
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int flags = get<2>(GetParam());

    Mat src(sz, matType);
    Mat mask = Mat::ones(sz, CV_8UC1);
    double minVal = 0, maxVal = 0;
    int minIdx[2] = {0}, maxIdx[2] = {0};

    double *pMinVal = (flags & MMI_MIN) ? &minVal : NULL;
    double *pMaxVal = (flags & MMI_MAX) ? &maxVal : NULL;
    int    *pMinIdx = ((flags & MMI_MIN) && (flags & MMI_IDX)) ? minIdx : NULL;
    int    *pMaxIdx = ((flags & MMI_MAX) && (flags & MMI_IDX)) ? maxIdx : NULL;

    randu(src, 1, 155);
    declare.in(src).in(mask);

    TEST_CYCLE() cv::minMaxIdx(src, pMinVal, pMaxVal, pMinIdx, pMaxIdx, mask);

    SANITY_CHECK_NOTHING();
}

} // namespace
