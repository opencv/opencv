// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

enum PerfSqMatDepth{
    DEPTH_32S_32S = 0,
    DEPTH_32S_32F,
    DEPTH_32S_64F,
    DEPTH_32F_32F,
    DEPTH_32F_64F,
    DEPTH_64F_64F};

CV_ENUM(IntegralOutputDepths, DEPTH_32S_32S, DEPTH_32S_32F, DEPTH_32S_64F, DEPTH_32F_32F, DEPTH_32F_64F, DEPTH_64F_64F);

static int extraOutputDepths[6][2] = {{CV_32S, CV_32S}, {CV_32S, CV_32F}, {CV_32S, CV_64F}, {CV_32F, CV_32F}, {CV_32F, CV_64F}, {CV_64F, CV_64F}};

typedef tuple<Size, MatType, MatDepth> Size_MatType_OutMatDepth_t;
typedef perf::TestBaseWithParam<Size_MatType_OutMatDepth_t> Size_MatType_OutMatDepth;

typedef tuple<Size, MatType, IntegralOutputDepths> Size_MatType_OutMatDepthArray_t;
typedef perf::TestBaseWithParam<Size_MatType_OutMatDepthArray_t> Size_MatType_OutMatDepthArray;

PERF_TEST_P(Size_MatType_OutMatDepth, integral,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4),
                testing::Values(CV_32S, CV_32F, CV_64F)
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat src(sz, matType);
    Mat sum(sz, sdepth);

    declare.in(src, WARMUP_RNG).out(sum);

    TEST_CYCLE() integral(src, sum, sdepth);

    SANITY_CHECK(sum, 1e-6);
}

PERF_TEST_P(Size_MatType_OutMatDepth, integral_sqsum,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4),
                testing::Values(CV_32S, CV_32F, CV_64F)
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat src(sz, matType);
    Mat sum(sz, sdepth);
    Mat sqsum(sz, sdepth);

    declare.in(src, WARMUP_RNG).out(sum, sqsum);
    declare.time(100);

    TEST_CYCLE() integral(src, sum, sqsum, sdepth);

    SANITY_CHECK(sum, 1e-6);
    SANITY_CHECK(sqsum, 1e-6);
}

PERF_TEST_P(Size_MatType_OutMatDepthArray, DISABLED_integral_sqsum_full,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4),
                testing::Values(DEPTH_32S_32S, DEPTH_32S_32F, DEPTH_32S_64F, DEPTH_32F_32F, DEPTH_32F_64F, DEPTH_64F_64F)
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int *outputDepths = (int *)extraOutputDepths[get<2>(GetParam())];
    int sdepth = outputDepths[0];
    int sqdepth = outputDepths[1];

    Mat src(sz, matType);
    Mat sum(sz, sdepth);
    Mat sqsum(sz, sqdepth);

    declare.in(src, WARMUP_RNG).out(sum, sqsum);
    declare.time(100);

    TEST_CYCLE() integral(src, sum, sqsum, sdepth, sqdepth);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( Size_MatType_OutMatDepth, integral_sqsum_tilted,
             testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values( CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4 ),
                 testing::Values( CV_32S, CV_32F, CV_64F )
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat src(sz, matType);
    Mat sum(sz, sdepth);
    Mat sqsum(sz, sdepth);
    Mat tilted(sz, sdepth);

    declare.in(src, WARMUP_RNG).out(sum, sqsum, tilted);
    declare.time(100);

    TEST_CYCLE() integral(src, sum, sqsum, tilted, sdepth);

    SANITY_CHECK(sum, 1e-6);
    SANITY_CHECK(sqsum, 1e-6);
    SANITY_CHECK(tilted, 1e-6, tilted.depth() > CV_32S ? ERROR_RELATIVE : ERROR_ABSOLUTE);
}

} // namespace
