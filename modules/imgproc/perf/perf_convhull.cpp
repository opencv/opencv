// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// The perf framework divides the reported time by the run count, so results stay per-call.
static int multirun_count(int total) { return std::max(1, 100000 / total); }

// random points in a bounding box, a different input on every call. (points count, box side)
typedef tuple<int, int> ConvHullRandomParams;
typedef TestBaseWithParam<ConvHullRandomParams> ConvexHullRandomPerfTest;

PERF_TEST_P(ConvexHullRandomPerfTest, convexHull,
    testing::Combine(
        testing::Values(16, 64, 256, 1024, 4096, 16384, 50000),     // total points
        testing::Values(100, 1000, 10000)                           // box side; sparsity = side / total
    ))
{
    const int total = get<0>(GetParam());
    const int side  = get<1>(GetParam());
    const int runs  = multirun_count(total);

    RNG rng(0x12345678);      // fixed seed => identical input for comparison of different cases
    std::vector<std::vector<Point> > inputs(runs, std::vector<Point>(total));
    for (int k = 0; k < runs; ++k)
        for (int i = 0; i < total; ++i)
            inputs[k][i] = Point(rng.uniform(0, side), rng.uniform(0, side));

    std::vector<Point> hull_pts;
    declare.runs(runs);
    PERF_SAMPLE_BEGIN()
        for (int k = 0; k < runs; ++k)
            convexHull(inputs[k], hull_pts, false /*clockwise*/, true /*returnPoints*/);
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

// a noisy closed contour (simulate output of findContours), points ordered along the boundary,
// a different input on every call. (points count, step between neighbour points)
typedef tuple<int, double> ConvHullContourParams;
typedef TestBaseWithParam<ConvHullContourParams> ConvexHullContourPerfTest;

PERF_TEST_P_(ConvexHullContourPerfTest, convexHull)
{
    const int total   = get<0>(GetParam());
    const double step = get<1>(GetParam());
    const int runs    = multirun_count(total);

    // polar form r(theta) = R * (1 + noise); R ~ step*total/(2*pi)
    const double R = step * total / (2.0 * CV_PI);
    const double scaleY = 1.25;   // ellipse, taller than wide with any noise

    RNG rng(0x12345678);      // fixed seed => identical input for comparison of different cases
    std::vector<std::vector<Point> > inputs(runs, std::vector<Point>(total));
    for (int k = 0; k < runs; ++k)
        for (int i = 0; i < total; ++i)
        {
            const double theta = 2.0 * CV_PI * i / total;
            const double r = R * rng.uniform(0.9, 1.1);   // +-10% radial noise
            inputs[k][i] = Point(cvRound(r * std::cos(theta)), cvRound(scaleY * r * std::sin(theta)));
        }

    std::vector<Point> hull_pts;
    declare.runs(runs);
    PERF_SAMPLE_BEGIN()
        for (int k = 0; k < runs; ++k)
            convexHull(inputs[k], hull_pts, false /*clockwise*/, true /*returnPoints*/);
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

// sparsity ~ 0.35 * step
INSTANTIATE_TEST_CASE_P(/*none*/, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 1000, 4000, 16000, 50000),    // contour points
        testing::Values(1.0, 2.0, 4.0, 6.0, 8.0, 80.0)                      // ~pixels between neighbour points
    ));

// the two sweeps below were used to find the sort thresholds; run them with --gtest_also_run_disabled_tests

// sparsity sweep; sparsity ~ 0.35 * step
INSTANTIATE_TEST_CASE_P(DISABLED_Sparsity, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 1000, 4000, 16000, 50000),                    // contour points
        testing::Values(1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0)      // ~pixels between neighbour points
    ));

// points count sweep, sparse contours; step 8: x and y ranges below 256, step 80: above
INSTANTIATE_TEST_CASE_P(DISABLED_PointsCount, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(12, 16, 20, 24, 28, 32, 36, 40, 48, 64),                            // contour points
        testing::Values(8.0, 80.0)                                                          // ~pixels between neighbour points
    ));

}} // namespace
