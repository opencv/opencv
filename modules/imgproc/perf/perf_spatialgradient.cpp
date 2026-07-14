// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<Size, int, int> Size_Ksize_BorderType_t;
typedef perf::TestBaseWithParam<Size_Ksize_BorderType_t> Size_Ksize_BorderType;

PERF_TEST_P( Size_Ksize_BorderType, spatialGradient,
    Combine(
        SZ_ALL_HD,
        Values( 3 ),
        Values( BORDER_DEFAULT, BORDER_REPLICATE )
    )
)
{
    Size size = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    int borderType = get<2>(GetParam());

    Mat src(size, CV_8UC1);
    Mat dx(size, CV_16SC1);
    Mat dy(size, CV_16SC1);

    declare.in(src, WARMUP_RNG).out(dx, dy);

    TEST_CYCLE() spatialGradient(src, dx, dy, ksize, borderType);

    SANITY_CHECK(dx);
    SANITY_CHECK(dy);
}

} // namespace
