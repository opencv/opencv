// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(MatExpr_CompoundAssign, BasicOps)
{
    MatExpr expr = Mat1f::ones(3, 3);
    expr += Mat(Mat1f::ones(3, 3));
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 2.0f);

    expr = Mat1f::ones(3, 3) * 5;
    expr -= Mat(Mat1f::ones(3, 3) * 2);
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 3.0f);

    expr = Mat1f::ones(3, 3) * 3;
    expr *= 2;
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 6.0f);

    expr = Mat1f::ones(3, 3) * 8;
    expr /= 2;
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 4.0f);
}

TEST(MatExpr_CompoundAssign, ChainedOps)
{
    MatExpr expr = Mat1f::ones(3, 3);
    expr += Mat(Mat1f::ones(3, 3));  // 1 + 1 = 2
    expr *= 3;                  // 2 * 3 = 6
    expr -= Mat(Mat1f::ones(3, 3));  // 6 - 1 = 5
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 5.0f);
}

TEST(MatExpr_CompoundAssign, OriginalBugCase)
{
    auto a = Mat1f::ones(3,3);
    if (std::is_same<decltype(a), MatExpr>::value)
        a.operator += (Mat1f::ones(3,3));
    else
        a += Mat1f::ones(3,3);
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(a)(0, 0), 2.0f);
}
}} // namespace opencv_test
