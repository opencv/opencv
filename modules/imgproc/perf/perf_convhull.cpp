// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// Batch factor for TEST_CYCLE_MULTIRUN: keeps work per timed sample roughly constant (~100k point-ops) so small-n rows clear the timer floor.
// The perf framework divides the reported time by the run count, so results stay per-call.
static int multirun_count(int total) { return std::max(1, 100000 / total); }

// random points in a bounding box, a different input on every call. (points count, box side)
typedef tuple<int, int> ConvHullParams;
typedef TestBaseWithParam<ConvHullParams> ConvexHullPerfTest;

PERF_TEST_P(ConvexHullPerfTest, convexHull,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 50000),  // total points
        testing::Values(100, 1000, 10000)                                                         // box side; sparsity = side / total
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
typedef tuple<int, int> ConvHullContourParams;
typedef TestBaseWithParam<ConvHullContourParams> ConvexHullContourPerfTest;

PERF_TEST_P(ConvexHullContourPerfTest, convexHull,
    testing::Combine(
        testing::Values(16, 32, 64, 100, 300, 1000, 10000, 50000),  // contour points
        testing::Values(1, 2, 4, 8, 16)                 // ~pixels between neighbour points; sparsity ~ 0.35 * step
    ))
{
    const int total = get<0>(GetParam());
    const int step  = get<1>(GetParam());
    const int runs  = multirun_count(total);

    // polar form r(theta) = R * (1 + noise); R ~ step*total/(2*pi)
    const double R = step * total / (2.0 * CV_PI);
    const double scaleY = 1.25;   // ellipse, taller than wide with any noise

    RNG rng(0x12345678);      // fixed seed => identical input for both cases - bucketsort / std::sort
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

}} // namespace
