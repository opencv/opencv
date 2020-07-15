// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

#define TYPICAL_MAT_TYPES_MORPH  CV_8UC1, CV_8UC4
#define TYPICAL_MATS_MORPH       testing::Combine(SZ_ALL_GA, testing::Values(TYPICAL_MAT_TYPES_MORPH))

PERF_TEST_P(Size_MatType, erode, TYPICAL_MATS_MORPH)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).out(dst);

    int runs = (sz.width <= 320) ? 15 : 1;
    TEST_CYCLE_MULTIRUN(runs) erode(src, dst, noArray());

    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_MatType, dilate, TYPICAL_MATS_MORPH)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() dilate(src, dst, noArray());

    SANITY_CHECK(dst);
}

} // namespace
