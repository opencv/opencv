// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace cv;

TEST(MatExpr_Lazy, BasicFunctionality)
{
    MatExpr expr = Mat1f::ones(3, 3);
    expr.operator += (Mat1f::ones(3, 3));
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 2.0f);
}

TEST(MatExpr_Lazy, CompositionAndMaterialization)
{
    MatExpr expr = Mat1f::ones(3, 3) * 2; // lazy
    expr = expr + Mat1f::ones(3, 3);
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 3.0f);

    expr = expr * 2;
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 6.0f);
}

TEST(MatExpr_Lazy, CompoundAssignPreservesLazy)
{
    MatExpr expr = Mat1f::ones(3, 3) * 2;
    expr.operator += (Mat1f::ones(3, 3));
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 3.0f);
}

TEST(MatExpr_Lazy, AliasingSafety)
{
    MatExpr expr = Mat1f::ones(3, 3) * 2;
    expr.operator += (Mat1f::ones(3, 3));
    expr.operator *= (2);
    expr.operator -= (Mat1f::ones(3, 3));
    EXPECT_FLOAT_EQ(static_cast<Mat1f>(expr)(0, 0), 5.0f);
}
