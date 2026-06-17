// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Tests for Layer 4: the cv::expression() string frontend. Exercises placeholders, operator
// precedence, function calls, type casts, assignments and tuple (multi-) outputs. Limited to
// ops with kernels today (arithmetic / cast / pow); math funcs land with the #7 kernel matrix.

#include "../test_precomp.hpp"
#include "ew_parser.hpp"

namespace opencv_test { namespace {

static Mat expr1(const String& e, const std::vector<Mat>& in)
{
    std::vector<Mat> out;
    cv::ew::expression(e, in, out);
    return out[0];
}

TEST(Core_EW_Expr, add)
{
    Mat a(12, 15, CV_32F), b(12, 15, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("{0} + {1}", { a, b });
    Mat exp; cv::add(a, b, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Operator precedence: '*' binds tighter than '+', unary minus on a literal.
TEST(Core_EW_Expr, addweighted_precedence)
{
    Mat a(20, 16, CV_32F), b(20, 16, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("{0} * 2.5 + {1} * -1.5 + 7", { a, b });
    Mat exp; cv::addWeighted(a, 2.5, b, -1.5, 7.0, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Named temporary via ';' assignment.
TEST(Core_EW_Expr, assignment)
{
    Mat a(18, 22, CV_32F), b(18, 22, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("t = {0} * 2.5; t + {1}", { a, b });
    Mat exp; cv::addWeighted(a, 2.5, b, 1.0, 0.0, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Tuple -> several outputs.
TEST(Core_EW_Expr, tuple_outputs)
{
    Mat a(14, 19, CV_32F), b(14, 19, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    std::vector<Mat> out;
    cv::ew::expression("({0} + {1}, {0} - {1})", std::vector<Mat>{ a, b }, out);
    ASSERT_EQ(out.size(), 2u);
    Mat eadd, esub; cv::add(a, b, eadd); cv::subtract(a, b, esub);
    EXPECT_LE(cvtest::norm(out[0], eadd, NORM_INF), 1e-3) << "sum";
    EXPECT_LE(cvtest::norm(out[1], esub, NORM_INF), 1e-3) << "diff";
}

// Grouping parens (NOT a tuple) inside a larger expression.
TEST(Core_EW_Expr, grouping_parens)
{
    Mat a(11, 13, CV_32F), b(11, 13, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("({0} + {1}) * 2", { a, b });
    Mat exp; cv::add(a, b, exp); exp *= 2.0;
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Type-cast function: float -> uint8 (saturating).
TEST(Core_EW_Expr, cast_uint8)
{
    Mat a(23, 17, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, -50.f, 300.f);

    Mat got = expr1("uint8({0})", { a });
    Mat exp; a.convertTo(exp, CV_8U);
    ASSERT_EQ(got.type(), exp.type());
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));
}

// pow() function call with a scalar exponent.
TEST(Core_EW_Expr, pow_call)
{
    Mat a(16, 16, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 5.f);

    Mat got = expr1("pow({0}, 2)", { a });
    Mat exp; cv::pow(a, 2.0, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

}} // namespace
