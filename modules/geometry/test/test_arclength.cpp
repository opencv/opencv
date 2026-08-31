// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<bool> Geometry_ArcLength;

template<typename T>
static double refArcLength(const std::vector<T>& curve, bool is_closed)
{
    double expected_length = 0.0;
    size_t npoints = curve.size();
    T prev_pt = curve[is_closed ? npoints - 1 : 0];
    int start_idx = is_closed ? 0 : 1;
    for (size_t i = start_idx; i < npoints; i++)
    {
        T pt = curve[i];
        expected_length += cv::norm(pt - prev_pt);
        prev_pt = pt;
    }

    return expected_length;
}

// Test arcLength accuracy
// Covers both open/closed curves and int/float coordinates
TEST_P(Geometry_ArcLength, accuracy_float)
{
    const bool is_closed = GetParam();

    RNG& rng = TS::ptr()->get_rng();

    // generate random points
    const int npoints = rng.uniform(3, 100);
    vector<Point2f> curvef;

    for (int i = 0; i < npoints; i++)
    {
        Point2f pt(rng.uniform(-1000.f, 1000.f), rng.uniform(-1000.f, 1000.f));
        curvef.push_back(pt);
    }

    double length = cv::arcLength(curvef, is_closed);
    double expected_length = refArcLength(curvef, is_closed);

    EXPECT_NEAR(expected_length, length, FLT_EPSILON * 100 * expected_length);
}

TEST_P(Geometry_ArcLength, accuracy_int)
{
    const bool is_closed = GetParam();

    RNG& rng = TS::ptr()->get_rng();

    // generate random points
    const int npoints = rng.uniform(3, 100);
    vector<Point2i> curvei;

    for (int i = 0; i < npoints; i++)
    {
        Point2f pt(rng.uniform(-1000.f, 1000.f), rng.uniform(-1000.f, 1000.f));
        curvei.push_back(Point2i((int)pt.x, (int)pt.y));
    }

    double length = cv::arcLength(curvei, is_closed);
    double expected_length = refArcLength(curvei, is_closed);

    EXPECT_NEAR(expected_length, length, FLT_EPSILON * expected_length);
}

// Simple test with a manually defined shape
TEST(Geometry_ArcLength, simple_square)
{
    // create a 10x10 square
    vector<Point2f> square;
    square.push_back(Point2f(0, 0));
    square.push_back(Point2f(10, 0));
    square.push_back(Point2f(10, 10));
    square.push_back(Point2f(0, 10));

    // closed perimeter
    EXPECT_DOUBLE_EQ(cv::arcLength(square, true), 40.0);

    // open perimeter (missing the edge from (0,10) to (0,0))
    EXPECT_DOUBLE_EQ(cv::arcLength(square, false), 30.0);
}

INSTANTIATE_TEST_CASE_P(
    Geometry,
    Geometry_ArcLength,
    testing::Bool()
);

}} // namespace opencv_test
