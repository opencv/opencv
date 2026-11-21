// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

// The perf framework divides the reported time by the run count, so results stay per-call.
static int multirun_count(int total) { return std::max(1, 100000 / total); }

// random points in a bounding box, a different input on every call. (points count, box side, points type, returnPoints)
typedef tuple<int, int, MatDepth, bool> ConvHullRandomParams;
typedef TestBaseWithParam<ConvHullRandomParams> ConvexHullRandomPerfTest;

PERF_TEST_P_(ConvexHullRandomPerfTest, convexHull)
{
    const int total = get<0>(GetParam());
    const int side  = get<1>(GetParam());
    const int depth = get<2>(GetParam());
    const bool returnPoints = get<3>(GetParam());
    const int runs  = multirun_count(total);

    RNG& rng = theRNG();
    std::vector<Mat> inputs(runs);
    for (int k = 0; k < runs; ++k)
    {
        inputs[k].create(total, 1, CV_MAKETYPE(depth, 2));
        for (int i = 0; i < total; ++i)
            if (depth == CV_32S)
                inputs[k].at<Point>(i) = Point(rng.uniform(0, side), rng.uniform(0, side));
            else
                inputs[k].at<Point2f>(i) = Point2f(rng.uniform(0.f, (float)side), rng.uniform(0.f, (float)side));
    }

    std::vector<int> hull_idx;
    std::vector<Point> hull_pts;
    std::vector<Point2f> hull_ptsf;
    _OutputArray hull = !returnPoints ? _OutputArray(hull_idx) :
                        depth == CV_32S ? _OutputArray(hull_pts) : _OutputArray(hull_ptsf);
    declare.runs(runs);
    PERF_SAMPLE_BEGIN()
        for (int k = 0; k < runs; ++k)
            convexHull(inputs[k], hull, false /*clockwise*/, returnPoints);
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(IntPoints, ConvexHullRandomPerfTest,
    testing::Combine(
        testing::Values(16, 64, 256, 1024, 4096, 16384, 50000),     // total points
        testing::Values(100, 1000, 10000),                          // box side; sparsity = side / total
        testing::Values(CV_32S),
        testing::Values(true)                                       // returnPoints
    ));

// indices, dense and sparse points
INSTANTIATE_TEST_CASE_P(IntIndices, ConvexHullRandomPerfTest,
    testing::Combine(
        testing::Values(16, 64, 256, 1024, 4096, 16384, 50000),     // total points
        testing::Values(100, 10000),                                // box side; sparsity = side / total
        testing::Values(CV_32S),
        testing::Values(false)                                      // returnPoints
    ));

// float points are sparse for any box side
INSTANTIATE_TEST_CASE_P(FloatPoints, ConvexHullRandomPerfTest,
    testing::Combine(
        testing::Values(16, 64, 256, 1024, 4096, 16384, 50000),     // total points
        testing::Values(1000),                                      // box side
        testing::Values(CV_32F),
        testing::Values(true)                                       // returnPoints
    ));

INSTANTIATE_TEST_CASE_P(FloatIndices, ConvexHullRandomPerfTest,
    testing::Combine(
        testing::Values(16, 64, 256, 1024, 4096, 16384, 50000),     // total points
        testing::Values(1000),                                      // box side
        testing::Values(CV_32F),
        testing::Values(false)                                      // returnPoints
    ));

// a noisy closed contour (simulate output of findContours), points ordered along the boundary,
// a different input on every call. (points count, step between neighbour points, points type, returnPoints)
typedef tuple<int, double, MatDepth, bool> ConvHullContourParams;
typedef TestBaseWithParam<ConvHullContourParams> ConvexHullContourPerfTest;

PERF_TEST_P_(ConvexHullContourPerfTest, convexHull)
{
    const int total   = get<0>(GetParam());
    const double step = get<1>(GetParam());
    const int depth   = get<2>(GetParam());
    const bool returnPoints = get<3>(GetParam());
    const int runs    = multirun_count(total);

    // polar form r(theta) = R * (1 + noise); R ~ step*total/(2*pi)
    const double R = step * total / (2.0 * CV_PI);
    const double scaleY = 1.25;   // ellipse, taller than wide with any noise

    RNG& rng = theRNG();
    std::vector<Mat> inputs(runs);
    for (int k = 0; k < runs; ++k)
    {
        inputs[k].create(total, 1, CV_MAKETYPE(depth, 2));
        for (int i = 0; i < total; ++i)
        {
            const double theta = 2.0 * CV_PI * i / total;
            const double r = R * rng.uniform(0.9, 1.1);   // +-10% radial noise
            const double x = r * std::cos(theta), y = scaleY * r * std::sin(theta);
            if (depth == CV_32S)
                inputs[k].at<Point>(i) = Point(cvRound(x), cvRound(y));
            else
                inputs[k].at<Point2f>(i) = Point2f((float)x, (float)y);
        }
    }

    std::vector<int> hull_idx;
    std::vector<Point> hull_pts;
    std::vector<Point2f> hull_ptsf;
    _OutputArray hull = !returnPoints ? _OutputArray(hull_idx) :
                        depth == CV_32S ? _OutputArray(hull_pts) : _OutputArray(hull_ptsf);
    declare.runs(runs);
    PERF_SAMPLE_BEGIN()
        for (int k = 0; k < runs; ++k)
            convexHull(inputs[k], hull, false /*clockwise*/, returnPoints);
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

// sparsity ~ 0.35 * step
INSTANTIATE_TEST_CASE_P(IntPoints, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 1000, 4000, 16000, 50000),    // contour points
        testing::Values(1.0, 2.0, 4.0, 6.0, 8.0, 80.0),                     // ~pixels between neighbour points
        testing::Values(CV_32S),
        testing::Values(true)                                               // returnPoints
    ));

// indices; step 1: dense, with duplicate points; step 8: sparse
INSTANTIATE_TEST_CASE_P(IntIndices, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 1000, 4000, 16000, 50000),    // contour points
        testing::Values(1.0, 8.0),                                          // ~pixels between neighbour points
        testing::Values(CV_32S),
        testing::Values(false)                                              // returnPoints
    ));

// float points are sparse for any step
INSTANTIATE_TEST_CASE_P(FloatPoints, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 1000, 4000, 16000, 50000),    // contour points
        testing::Values(1.0),                                               // ~pixels between neighbour points
        testing::Values(CV_32F),
        testing::Values(true)                                               // returnPoints
    ));

INSTANTIATE_TEST_CASE_P(FloatIndices, ConvexHullContourPerfTest,
    testing::Combine(
        testing::Values(16, 32, 64, 128, 256, 1000, 4000, 16000, 50000),    // contour points
        testing::Values(1.0),                                               // ~pixels between neighbour points
        testing::Values(CV_32F),
        testing::Values(false)                                              // returnPoints
    ));

}} // namespace
