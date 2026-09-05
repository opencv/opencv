// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {
using namespace perf;

typedef perf::TestBaseWithParam<Size> SizePrm;

PERF_TEST_P( SizePrm, LUT,
             testing::Values(szQVGA, szVGA, sz1080p)
           )
{
    Size sz = GetParam();

    int maxValue = 255;

    Mat src(sz, CV_8UC1);
    randu(src, 0, maxValue);
    Mat lut(1, 256, CV_8UC1);
    randu(lut, 0, maxValue);
    Mat dst(sz, CV_8UC1);

    TEST_CYCLE() LUT(src, lut, dst);

    SANITY_CHECK(dst, 0.1);
}

PERF_TEST_P( SizePrm, LUT_multi,
             testing::Values(szQVGA, szVGA, sz1080p)
           )
{
    Size sz = GetParam();

    int maxValue = 255;

    Mat src(sz, CV_8UC3);
    randu(src, 0, maxValue);
    Mat lut(1, 256, CV_8UC1);
    randu(lut, 0, maxValue);
    Mat dst(sz, CV_8UC3);

    TEST_CYCLE() LUT(src, lut, dst);

    SANITY_CHECK_NOTHING();
}

#define LUT_TYPES CV_8UC1, CV_8UC3, CV_8UC4, CV_32SC1

typedef perf::TestBaseWithParam< tuple<Size, MatType> > Size_MatType_LUT;

PERF_TEST_P( Size_MatType_LUT, LUT_types,
             testing::Combine(
                 testing::Values(szVGA, sz1080p),
                 testing::Values(LUT_TYPES)
             )
           )
{
    Size sz  = get<0>(GetParam());
    int  type = get<1>(GetParam());
    int  cn   = CV_MAT_CN(type);
    int  lutDepth = CV_MAT_DEPTH(type);

    Mat src(sz, CV_8UC(cn));
    randu(src, 0, 256);
    Mat lut(1, 256, CV_MAKETYPE(lutDepth, 1));
    randu(lut, 0, 256);
    Mat dst(sz, CV_MAKETYPE(lutDepth, cn));

    TEST_CYCLE() LUT(src, lut, dst);

    SANITY_CHECK_NOTHING();
}

}} // namespace
