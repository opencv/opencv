// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// (points count, sparsity S = x-range / points count)
typedef tuple<int, double> ConvHullParams;
typedef TestBaseWithParam<ConvHullParams> ConvexHullPerfTest;

PERF_TEST_P(ConvexHullPerfTest, convexHull,
    testing::Combine(
        testing::Values(100, 1000, 10000, 50000),  // total points
        testing::Values(0.125, 0.5, 2.0)           // sparsity S = rangeX / total
        // S = 0.125 : dense, ~8 points per x-column, strong lo/hi pruning
        // S = 0.5   : typical contour density (circle-ish contour has S ~ 1/pi)
        // S = 2     : sparse but below MAX_SPARSITY_FACTOR(4) 
    ))
{
    const int total  = get<0>(GetParam());
    const double S   = get<1>(GetParam());
    const int rangeX = cvRound(S * total);

    RNG rng(0x12345678);      // fixed seed => identical input for both cases - bucketsort / std::sort
    std::vector<Point> points(total);
    for (int i = 0; i < total; ++i)
        points[i] = Point(rng.uniform(0, rangeX), rng.uniform(-100000, 100000));

    std::vector<Point> hull_pts;
    TEST_CYCLE() convexHull(points, hull_pts, false /*clockwise*/, true /*returnPoints*/);

    SANITY_CHECK_NOTHING();
}

// a noisy closed contour (simulate output of findContours), points ordered along the boundary.
typedef TestBaseWithParam<int> ConvexHullContourPerfTest;

PERF_TEST_P(ConvexHullContourPerfTest, convexHull,
    testing::Values(100, 1000, 10000, 50000))   // contour points
{
    const int total = GetParam();

    // polar form r(theta) = R * (1 + noise); R ~ total/(2*pi)
    const double R = total / (2.0 * CV_PI);

    RNG rng(0x12345678);      // fixed seed => identical input for both cases - bucketsort / std::sort
    std::vector<Point> points(total);
    for (int i = 0; i < total; ++i)
    {
        const double theta = 2.0 * CV_PI * i / total;
        const double r = R * rng.uniform(0.9, 1.1);   // +-10% radial noise
        points[i] = Point(cvRound(r * std::cos(theta)), cvRound(r * std::sin(theta)));
    }

    std::vector<Point> hull_pts;
    TEST_CYCLE() convexHull(points, hull_pts, false /*clockwise*/, true /*returnPoints*/);

    SANITY_CHECK_NOTHING();
}

}} // namespace
