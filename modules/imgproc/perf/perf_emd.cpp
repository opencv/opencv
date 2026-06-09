// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"
#include <cmath>

namespace opencv_test { namespace {

typedef tuple<int, int> EMD_Size_Dim_t;
typedef perf::TestBaseWithParam<EMD_Size_Dim_t> EMD_Fixture;

PERF_TEST_P(EMD_Fixture, L1_Distance, testing::Combine(
            testing::Values(100, 500, 1000),
            testing::Values(3, 64)
))
{
    int size = get<0>(GetParam());
    int dims = get<1>(GetParam());

    Mat sign1(size, dims + 1, CV_32FC1);
    Mat sign2(size, dims + 1, CV_32FC1);

    theRNG().fill(sign1, RNG::UNIFORM, 0.1, 1.0);
    theRNG().fill(sign2, RNG::UNIFORM, 0.1, 1.0);

    declare.in(sign1, sign2);

    TEST_CYCLE()
    {
        cv::EMD(sign1, sign2, cv::DIST_L1);
    }

    SANITY_CHECK_NOTHING();
}

}}