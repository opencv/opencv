// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define FILTERSPECKLES_TYPES CV_8UC1, CV_16SC1

typedef tuple<Size, MatType> Size_MatType_t;
typedef perf::TestBaseWithParam<Size_MatType_t> Size_MatType_FilterSpeckles;

PERF_TEST_P(Size_MatType_FilterSpeckles, filterSpeckles,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(FILTERSPECKLES_TYPES)
                ))
{
    Size sz  = get<0>(GetParam());
    int  type = get<1>(GetParam());

    // filterSpeckles works in place on a disparity map.
    Mat src(sz, type);
    Mat dst(sz, type);
    declare.in(src, WARMUP_RNG).out(dst);

    while (next())
    {
        src.copyTo(dst);
        startTimer();
        cv::filterSpeckles(dst, 0, 2, 2);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
