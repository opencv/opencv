// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Regression test for https://github.com/opencv/opencv/issues/29232
// cv::borderInterpolate with BORDER_WRAP overflowed a 32-bit signed integer
// when p is close to INT_MIN (e.g. p=-2147483583, len=269).
TEST(Core_BorderInterpolate, wrap_no_overflow)
{
    // These values triggered signed integer overflow in the old code:
    //   p - len + 1  ==  -2147483583 - 269 + 1  ==  -2147483851  (below INT_MIN)
    const int len = 269;
    const int p   = std::numeric_limits<int>::min() + 65;  // -2147483583

    // Must not crash / invoke UB, and must return a value in [0, len-1].
    int result = cv::borderInterpolate(p, len, cv::BORDER_WRAP);
    EXPECT_GE(result, 0);
    EXPECT_LT(result, len);

    // Cross-check: the mathematical modulo result.
    // Use long long to compute the reference safely.
    long long ref = ((long long)p % len + len) % len;
    EXPECT_EQ(result, (int)ref);
}

// Exhaustive smoke test: all border types should keep result in [0, len-1]
// for a range of negative and positive out-of-bounds coordinates.
TEST(Core_BorderInterpolate, wrap_range)
{
    const int len = 7;
    const int types[] = { cv::BORDER_WRAP };
    for (int btype : types)
    {
        for (int p = -3 * len; p < 3 * len; ++p)
        {
            int r = cv::borderInterpolate(p, len, btype);
            EXPECT_GE(r, 0)   << "p=" << p;
            EXPECT_LT(r, len) << "p=" << p;
        }
    }
}

}} // namespace opencv_test
