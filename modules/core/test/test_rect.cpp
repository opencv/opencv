// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

    TEST(Rect, rectUnionOperator) {

        Rect2f test = Rect();

        ASSERT_EQ(Rect(0, 0, 0, 0), Rect(0, 0, 1, 1) & Rect(2, 2, 1, 1));
        ASSERT_EQ(Rect(0, 0, 0, 0), Rect(2, 2, 1, 1) & Rect(0, 0, 1, 1));
        ASSERT_EQ(Rect(0, 0, 1, 1), Rect(0, 0, 1, 1) & Rect(0, 0, 1, 1));
        ASSERT_EQ(Rect(1, 1, 1, 1), Rect(0, 0, 2, 2) & Rect(1, 1, 1, 1));
        ASSERT_EQ(Rect(1, 1, 1, 1), Rect(0, 0, 2, 2) & Rect(1, 1, 2, 2));

        ASSERT_EQ(Rect(1, 0, 0, 1), Rect(0, 0, 1, 1) & Rect(1, 0, 1, 1));
        ASSERT_EQ(Rect(1, 1, 0, 0), Rect(0, 0, 1, 1) & Rect(1, 1, 1, 1));
    }

}} // namespace
