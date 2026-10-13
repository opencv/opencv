// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef TestBaseWithParam< tuple<Size, double, double, int> > TestPyrMeanShiftFiltering;

PERF_TEST_P( TestPyrMeanShiftFiltering, pyrMeanShiftFiltering,
             Combine(
                Values( szVGA, sz720p ),
                Values( 10. ),
                Values( 20., 40. ),
                Values( 2 )
             )
)
{
    Size sz;
    double sp, sr;
    int maxLevel;

    sz = get<0>(GetParam());
    sp = get<1>(GetParam());
    sr = get<2>(GetParam());
    maxLevel = get<3>(GetParam());

    Mat src(sz, CV_8UC3);
    Mat dst(sz, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst).time(60);

    TEST_CYCLE() pyrMeanShiftFiltering(src, dst, sp, sr, maxLevel);

    SANITY_CHECK_NOTHING();
}

} // namespace
