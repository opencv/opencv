// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

typedef tuple<int, MatType> PointsNum_MatType_t;
typedef perf::TestBaseWithParam<PointsNum_MatType_t> PointsNum_MatType;

PERF_TEST_P(PointsNum_MatType, convexHull,
            testing::Combine(
                testing::Values(100, 1000, 10000, 100000),
                testing::Values(CV_32S, CV_32F)
            )
            )
{
    int pointsNum = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat points(pointsNum, 1, CV_MAKETYPE(matType, 2));
    cv::randu(points, 0, 1000);

    Mat hull;
    declare.in(points);

    TEST_CYCLE() cv::convexHull(points, hull);

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
