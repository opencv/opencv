// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;
typedef TestBaseWithParam<tuple<int, int> > ORBOrientationRVV;
PERF_TEST_P(ORBOrientationRVV, detectAndCompute,
            testing::Combine(testing::Values(7, 15, 31, 65), testing::Values(500, 2000)))
{
    const int patch = get<0>(GetParam()), count = get<1>(GetParam());
    Mat src(480, 653, CV_8U), descriptors;
    RNG rng(30388); rng.fill(src, RNG::UNIFORM, 0, 256);
    Mat roi = src(Rect(3, 0, 640, 480));
    Ptr<ORB> orb = ORB::create(count, 1.2f, 8, patch + 8, 0, 2, ORB::HARRIS_SCORE, patch);
    std::vector<KeyPoint> points;
    declare.in(roi);
    TEST_CYCLE() orb->detectAndCompute(roi, noArray(), points, descriptors);
    ASSERT_FALSE(points.empty());
    SANITY_CHECK(descriptors);
}
} // namespace
