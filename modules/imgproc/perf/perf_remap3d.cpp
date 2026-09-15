// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef TestBaseWithParam<tuple<int, MatType, int> > Remap3DPerf;

PERF_TEST_P(Remap3DPerf, volume, Combine(Values(32, 64),
            Values(CV_8UC1, CV_32FC1, CV_32FC3), Values(INTER_NEAREST, INTER_LINEAR)))
{
    const int side = get<0>(GetParam()), type = get<1>(GetParam());
    const int interpolation = get<2>(GetParam()), size[] = {side, side, side};
    Mat src(3, size, type), dst(3, size, type), map(3, size, CV_32FC3);
    for (int z = 0; z < side; ++z)
        for (int y = 0; y < side; ++y)
            for (int x = 0; x < side; ++x)
                map.at<Vec3f>(z, y, x) = Vec3f(x + 0.3f, y - 0.2f, z + 0.1f * (x % 7));
    declare.in(src, WARMUP_RNG).in(map).out(dst);
    TEST_CYCLE() cv::remap3D(src, dst, map, interpolation, BORDER_REPLICATE);
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
