// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test
{

typedef perf::TestBaseWithParam<Size> Perf_Decolor;

PERF_TEST_P(Perf_Decolor, decolor,
    testing::Values(::perf::szVGA,
                    ::perf::sz720p,
                    ::perf::sz1080p))
{
    Size sz = GetParam();

    Mat src(sz, CV_8UC3);
    randu(src, 0, 256);
    Mat grayscale(sz, CV_8UC1);
    Mat color_boost(sz, CV_8UC3);

    declare.in(src).out(grayscale, color_boost).time(120);

    TEST_CYCLE() cv::decolor(src, grayscale, color_boost);

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
