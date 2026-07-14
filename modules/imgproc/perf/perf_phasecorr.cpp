// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef TestBaseWithParam<Size > CreateHanningWindowFixture;

PERF_TEST_P( CreateHanningWindowFixture, CreateHanningWindow, Values(szVGA, sz1080p))
{
    const Size size = GetParam();
    Mat dst(size, CV_32FC1);

    declare.in(dst, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::createHanningWindow(dst, size, CV_32FC1);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

} // namespace
