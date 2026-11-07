// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {
typedef TestBaseWithParam<tuple<int, int> > RectSubPixSIMD;
PERF_TEST_P(RectSubPixSIMD, floatAnd8uControls,
            testing::Combine(testing::Values(7, 8, 9, 16, 31, 32, 33, 64, 128),
                             testing::Values(CV_32FC1, CV_32FC3, CV_8UC1)))
{
    const int width = get<0>(GetParam()), type = get<1>(GetParam());
    Mat src(480, 653, type), dst(width, width, CV_MAKETYPE(CV_32F, CV_MAT_CN(type)));
    RNG rng(30388); rng.fill(src, RNG::UNIFORM, 0, 255);
    Mat roi = src(Rect(3, 0, 640, 480));
    declare.in(roi).out(dst);
    TEST_CYCLE() getRectSubPix(roi, Size(width, width), Point2f(320.375f, 240.625f), dst, CV_32F);
    SANITY_CHECK(dst, 1e-3);
}
} // namespace
