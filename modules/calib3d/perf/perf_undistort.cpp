// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test {

PERF_TEST(Undistort, InitUndistortMap)
{
    Size size_w_h(512 + 3, 512);
    Mat k(3, 3, CV_32FC1);
    Mat d(1, 14, CV_64FC1);
    Mat dst(size_w_h, CV_32FC2);
    declare.in(k, d, WARMUP_RNG).out(dst);
    TEST_CYCLE() initUndistortRectifyMap(k, d, noArray(), k, size_w_h, CV_32FC2, dst, noArray());
    SANITY_CHECK_NOTHING();
}

} // namespace
