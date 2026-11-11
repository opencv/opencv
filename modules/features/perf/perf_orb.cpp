// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_feature2d.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<tuple<int, string> > ORB_PatchSize;
PERF_TEST_P(ORB_PatchSize, detect,
            testing::Combine(testing::Values(7, 15, 31, 65), TEST_IMAGES))
{
    const int patch = get<0>(GetParam());
    Mat image = imread(getDataPath(get<1>(GetParam())), IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, patch + 8, 0, 2, ORB::HARRIS_SCORE, patch);
    vector<KeyPoint> points;
    declare.in(image);
    TEST_CYCLE() orb->detect(image, points);
    SANITY_CHECK_NOTHING();
}

} // namespace
