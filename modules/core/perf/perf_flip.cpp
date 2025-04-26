// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {
using namespace perf;

enum
{
    FLIP_XY = 0,
    FLIP_X = 1,
    FLIP_Y = 2,
};

#define FLIP_SIZES szQVGA, szVGA, sz1080p
#define FLIP_TYPES CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8SC1, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_32SC1, CV_32FC1
#define FLIP_CODES FLIP_X, FLIP_Y, FLIP_XY

CV_FLAGS(FlipCode, FLIP_X, FLIP_Y, FLIP_XY);
typedef tuple<Size, MatType, FlipCode> Size_MatType_FlipCode_t;
typedef perf::TestBaseWithParam<Size_MatType_FlipCode_t> Size_MatType_FlipCode;

PERF_TEST_P(Size_MatType_FlipCode,
            flip,
            testing::Combine(testing::Values(FLIP_SIZES),
                             testing::Values(FLIP_TYPES),
                             testing::Values(FLIP_CODES)))
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int flipCode = get<2>(GetParam()) - 1;

    Mat src(sz, matType);
    Mat dst(sz, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::flip(src, dst, flipCode);

    SANITY_CHECK_NOTHING();
}

}}  // namespace opencv_test
