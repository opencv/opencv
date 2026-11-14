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

CV_ENUM(IntegralOutputDepths, DEPTH_32S_32S, DEPTH_32S_32F, DEPTH_32S_64F, DEPTH_32F_32F, DEPTH_32F_64F, DEPTH_64F_64F)

static int extraOutputDepths[6][2] = {{CV_32S, CV_32S}, {CV_32S, CV_32F}, {CV_32S, CV_64F}, {CV_32F, CV_32F}, {CV_32F, CV_64F}, {CV_64F, CV_64F}};

typedef tuple<Size, MatType, MatDepth> Size_MatType_OutMatDepth_t;
typedef perf::TestBaseWithParam<Size_MatType_OutMatDepth_t> Size_MatType_OutMatDepth;

typedef tuple<Size, std::tuple<MatType, IntegralOutputDepths>> Size_MatType_OutMatDepthArray_t;
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
    if (sdepth == CV_32F)
        src *= (1 << 23) / (double)(sz.area() * 256);  // FP32 calculations are not accurate (mantissa is 23-bit)

    TEST_CYCLE() integral(src, sum, sdepth);

    Mat src_roi; src(Rect(src.cols - 4, src.rows - 4, 4, 4)).convertTo(src_roi, sdepth);
    Mat restored_src_roi =
           sum(Rect(sum.cols - 4, sum.rows - 4, 4, 4)) + sum(Rect(sum.cols - 5, sum.rows - 5, 4, 4)) -
           sum(Rect(sum.cols - 4, sum.rows - 5, 4, 4)) - sum(Rect(sum.cols - 5, sum.rows - 4, 4, 4));
    EXPECT_EQ(0, cvtest::norm(restored_src_roi, src_roi, NORM_INF))
        << src_roi << endl << restored_src_roi << endl
        << sum(Rect(sum.cols - 4, sum.rows - 4, 4, 4));

    if (sdepth == CV_32F)
        SANITY_CHECK_NOTHING();
    else
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

    // A CV_32F sum/sqsum above 2^24 carries a float32 ULP that depends on the SIMD
    // width and accumulation order (the SIMD path now runs here where it used to fall
    // back to scalar), so compare those relatively; CV_32S/CV_64F outputs stay exact.
    // Mirrors the tilted check below.
    SANITY_CHECK(sum,   1e-6, sum.depth()   > CV_32S ? ERROR_RELATIVE : ERROR_ABSOLUTE);
    SANITY_CHECK(sqsum, 1e-6, sqsum.depth() > CV_32S ? ERROR_RELATIVE : ERROR_ABSOLUTE);
}

static std::vector<std::tuple<MatType, IntegralOutputDepths>> GetFullSqsumDepthPairs() {
    static int extraDepths[12][2] = {
        {CV_8U, DEPTH_32S_64F},
        {CV_8U, DEPTH_32S_32F},
        {CV_8U, DEPTH_32S_32S},
        {CV_8U, DEPTH_32F_64F},
        {CV_8U, DEPTH_32F_32F},
        {CV_8U, DEPTH_64F_64F},
        {CV_16U, DEPTH_64F_64F},
        {CV_16S, DEPTH_64F_64F},
        {CV_32F, DEPTH_32F_64F},
        {CV_32F, DEPTH_32F_32F},
        {CV_32F, DEPTH_64F_64F},
        {CV_64F, DEPTH_64F_64F}
    };
    std::vector<std::tuple<MatType, IntegralOutputDepths>> valid_pairs;
    for (size_t i = 0; i < 12; i++) {
        for (int cn = 1; cn <= 4; cn++) {
            valid_pairs.emplace_back(CV_MAKETYPE(extraDepths[i][0], cn), extraDepths[i][1]);
        }
    }
    return valid_pairs;
}

PERF_TEST_P(Size_MatType_OutMatDepthArray, DISABLED_integral_sqsum_full,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::ValuesIn(GetFullSqsumDepthPairs())
                )
            )
{
    Size sz = get<0>(GetParam());
    auto depths = get<1>(GetParam());
    int matType = get<0>(depths);
    int sdepth = extraOutputDepths[get<1>(depths)][0];
    int sqdepth = extraOutputDepths[get<1>(depths)][1];

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

// sum=CV_32S, sqsum=CV_64F: the default output types of cv::integral(src, sum, sqsum)
// for an 8-bit image. The integral_sqsum test above pairs sum and sqsum at the same
// depth, so this common combination was not measured.
PERF_TEST_P(Size_MatType_OutMatDepth, integral_sqsum_64f,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8UC1, CV_8UC3),
                testing::Values(CV_32S)
                )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat src(sz, matType);
    Mat sum(sz, sdepth);
    Mat sqsum(sz, CV_64F);

    declare.in(src, WARMUP_RNG).out(sum, sqsum);
    declare.time(100);

    TEST_CYCLE() integral(src, sum, sqsum, sdepth, CV_64F);

    // The CV_64F sqsum can exceed INT_MAX; round-tripping such whole-number doubles
    // through FileStorage is fixed by opencv#29368, so the regression baseline is usable.
    SANITY_CHECK(sum,   1e-6, sum.depth()   > CV_32S ? ERROR_RELATIVE : ERROR_ABSOLUTE);
    SANITY_CHECK(sqsum, 1e-6, sqsum.depth() > CV_32S ? ERROR_RELATIVE : ERROR_ABSOLUTE);
}

// Enabled coverage for the combinations the widened 8-bit integral sqsum path
// vectorizes: 8-bit source, every channel count, and every sum/sqsum output-depth
// pair that now has a SIMD path. (The full 12-combo matrix lives in the DISABLED
// test above; this curated subset is cheap enough to run by default.) Correctness
// is covered by the Imgproc_Integral.accuracy test.
static std::vector<std::tuple<MatType, IntegralOutputDepths>> GetSqsum8UDepthPairs() {
    static int pairs[6] = { DEPTH_32S_64F, DEPTH_32S_32F, DEPTH_32S_32S, DEPTH_32F_64F, DEPTH_32F_32F, DEPTH_64F_64F };
    std::vector<std::tuple<MatType, IntegralOutputDepths>> valid_pairs;
    for (int cn = 1; cn <= 4; cn++)
        for (int p = 0; p < 6; p++)
            valid_pairs.emplace_back(CV_MAKETYPE(CV_8U, cn), pairs[p]);
    return valid_pairs;
}

PERF_TEST_P(Size_MatType_OutMatDepthArray, integral_sqsum_8u,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::ValuesIn(GetSqsum8UDepthPairs())
                )
            )
{
    Size sz = get<0>(GetParam());
    auto depths = get<1>(GetParam());
    int matType = get<0>(depths);
    int sdepth  = extraOutputDepths[get<1>(depths)][0];
    int sqdepth = extraOutputDepths[get<1>(depths)][1];

    Mat src(sz, matType);
    Mat sum(sz, sdepth);
    Mat sqsum(sz, sqdepth);

    declare.in(src, WARMUP_RNG).out(sum, sqsum);
    declare.time(100);

    TEST_CYCLE() integral(src, sum, sqsum, sdepth, sqdepth);

    SANITY_CHECK_NOTHING();
}

} // namespace
