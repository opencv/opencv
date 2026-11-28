// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<Size, int, int> LambdaConnected_Size_Conn_Type_t;
typedef perf::TestBaseWithParam<LambdaConnected_Size_Conn_Type_t> LambdaConnected_Size_Conn_Type;

PERF_TEST_P(LambdaConnected_Size_Conn_Type, lambdaConnected, Combine(
    testing::Values(Size(320, 240), Size(640, 480), Size(1280, 720)),
    testing::Values(4, 8),
    testing::Values((int)CV_8UC1, (int)CV_8UC3)
    ))
{
    Size size = get<0>(GetParam());
    int connectivity = get<1>(GetParam());
    int type = get<2>(GetParam());

    Mat image(size, type);
    declare.in(image, WARMUP_RNG);

    Point seed(size.width / 2, size.height / 2);
    Mat map;

    TEST_CYCLE()
    {
        lambdaConnected(image, map, seed, connectivity);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
