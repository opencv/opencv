// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<bool, int>> Geometry_ArcLength;

// Test arcLength accuracy
// Covers both open/closed curves and int/float coordinates
TEST_P(Geometry_ArcLength, accuracy)
{
    const bool is_closed = get<0>(GetParam());
    const int type = get<1>(GetParam());

    RNG& rng = TS::ptr()->get_rng();

    // generate random points
    const int npoints = rng.uniform(3, 100);
    vector<Point2f> curvef;
    vector<Point2i> curvei;

    for (int i = 0; i < npoints; i++)
    {
        Point2f pt(rng.uniform(-1000.f, 1000.f), rng.uniform(-1000.f, 1000.f));
        curvef.push_back(pt);
        curvei.push_back(Point2i((int)pt.x, (int)pt.y));
    }

    // test the specific type
    double length = 0.0;

    if (type == CV_32S)
    {
        length = cv::arcLength(curvei, is_closed);
    }
    else
    {
        length = cv::arcLength(curvef, is_closed);
    }

    double expected_length = 0.0;

    if (type == CV_32F)
    {
        Point2f prev_pt = curvef[is_closed ? npoints - 1 : 0];
        int start_idx = is_closed ? 0 : 1;
        for (int i = start_idx; i < npoints; i++)
        {
            Point2f pt = curvef[i];
            double dx = pt.x - prev_pt.x;
            double dy = pt.y - prev_pt.y;
            expected_length += sqrt(dx * dx + dy * dy);
            prev_pt = pt;
        }
    }
    else
    {
        Point2i prev_pt = curvei[is_closed ? npoints - 1 : 0];
        int start_idx = is_closed ? 0 : 1;
        for (int i = start_idx; i < npoints; i++)
        {
            Point2i pt = curvei[i];
            double dx = (double)pt.x - prev_pt.x;
            double dy = (double)pt.y - prev_pt.y;
            expected_length += sqrt(dx * dx + dy * dy);
            prev_pt = pt;
        }
    }

    EXPECT_NEAR(expected_length, length, FLT_EPSILON * 100 * expected_length);
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
    testing::Combine(
        testing::Bool(),
        testing::Values(CV_32S, CV_32F)
    )
);

}} // namespace opencv_test
