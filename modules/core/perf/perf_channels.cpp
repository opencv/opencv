// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define CHANNELS_TYPES CV_8UC3, CV_8UC4, CV_16UC3, CV_16UC4, CV_32FC3, CV_32FC4

typedef tuple<Size, MatType> Size_MatType_t;
typedef perf::TestBaseWithParam<Size_MatType_t> Size_MatType_Channels;

PERF_TEST_P(Size_MatType_Channels, extractChannel,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CHANNELS_TYPES)
                ))
{
    Size sz  = get<0>(GetParam());
    int  type = get<1>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, CV_MAKETYPE(CV_MAT_DEPTH(type), 1));

    declare.in(src, WARMUP_RNG).out(dst);

    int coi = CV_MAT_CN(type) - 1;
    TEST_CYCLE_MULTIRUN(10) cv::extractChannel(src, dst, coi);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_Channels, insertChannel,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CHANNELS_TYPES)
                ))
{
    Size sz  = get<0>(GetParam());
    int  type = get<1>(GetParam());

    Mat src(sz, CV_MAKETYPE(CV_MAT_DEPTH(type), 1));
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).out(dst);

    int coi = CV_MAT_CN(type) - 1;
    TEST_CYCLE_MULTIRUN(10) cv::insertChannel(src, dst, coi);

    SANITY_CHECK_NOTHING();
}

} // namespace
