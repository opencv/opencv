// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

#define TYPICAL_MAT_TYPES_MORPH  CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16SC1, CV_32FC1, CV_32FC3, CV_32FC4
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

// Parameterized kernel size + iterations (matches IPP perf coverage:
// rect structuring element of size 3/5/7/21, 1 and 3 iterations).
typedef tuple<Size, MatType, int, int> Size_MatType_kSize_iter_t;
typedef perf::TestBaseWithParam<Size_MatType_kSize_iter_t> Size_MatType_kSize_iter;

PERF_TEST_P(Size_MatType_kSize_iter, erode_kernel,
            testing::Combine(
                SZ_ALL_GA,
                testing::Values(TYPICAL_MAT_TYPES_MORPH),
                testing::Values(3, 5, 7, 21),
                testing::Values(1, 3)
                ))
{
    Size sz    = get<0>(GetParam());
    int  type  = get<1>(GetParam());
    int  ksize = get<2>(GetParam());
    int  iters = get<3>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() erode(src, dst, kernel, Point(-1, -1), iters);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Size_MatType_kSize_iter, dilate_kernel,
            testing::Combine(
                SZ_ALL_GA,
                testing::Values(TYPICAL_MAT_TYPES_MORPH),
                testing::Values(3, 5, 7, 21),
                testing::Values(1, 3)
                ))
{
    Size sz    = get<0>(GetParam());
    int  type  = get<1>(GetParam());
    int  ksize = get<2>(GetParam());
    int  iters = get<3>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() dilate(src, dst, kernel, Point(-1, -1), iters);

    SANITY_CHECK_NOTHING();
}

} // namespace
