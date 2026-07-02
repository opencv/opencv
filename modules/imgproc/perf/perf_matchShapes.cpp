// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(MatchShapeMethod, CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3)

typedef perf::TestBaseWithParam< tuple<int, MatchShapeMethod> > MatchShapesFixture;

static void generateContour(int npoints, int seed, Mat& contour)
{
    RNG rng(seed);
    contour.create(npoints, 1, CV_32SC2);
    Point* pts = contour.ptr<Point>();
    const Point center(rng.uniform(200, 400), rng.uniform(200, 400));
    const int radius = rng.uniform(50, 150);
    for (int i = 0; i < npoints; ++i)
    {
        const double angle = 2 * CV_PI * i / npoints;
        pts[i].x = center.x + cvRound(radius * cos(angle));
        pts[i].y = center.y + cvRound(radius * sin(angle));
    }
}

PERF_TEST_P(MatchShapesFixture, matchShapes,
            testing::Combine(
                testing::Values(64, 256, 1024),
                MatchShapeMethod::all()))
{
    const int npoints = get<0>(GetParam());
    const int method = get<1>(GetParam());

    Mat contour1, contour2;
    generateContour(npoints, 0, contour1);
    generateContour(npoints, 1, contour2);

    declare.in(contour1, WARMUP_RNG).in(contour2, WARMUP_RNG);

    double dist = 0;
    TEST_CYCLE()
    {
        dist = cv::matchShapes(contour1, contour2, method, 0);
    }

    SANITY_CHECK(dist, 1e-3, ERROR_RELATIVE);
}

} // namespace opencv_test
