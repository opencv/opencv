// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Tests for the public cv::texpr() string frontend. Exercises placeholders, operator precedence,
// function calls, type casts, assignments and tuple (multi-) outputs. Limited to ops with kernels
// today (arithmetic / cast / pow / min / max / absdiff).

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static Mat expr1(const String& e, const std::vector<Mat>& in)
{
    std::vector<Mat> out;
    cv::texpr(e, in, out);
    return out[0];
}

TEST(Core_TExpr, add)
{
    Mat a(12, 15, CV_32F), b(12, 15, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("{0} + {1}", { a, b });
    Mat exp; cv::add(a, b, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Built-in binary functions min/max/absdiff parsed and dispatched through emitBinary.
TEST(Core_TExpr, minmax_absdiff)
{
    Mat a(18, 21, CV_8U), b(18, 21, CV_8U);
    theRNG().fill(a, RNG::UNIFORM, 0, 255);
    theRNG().fill(b, RNG::UNIFORM, 0, 255);

    Mat gmin = expr1("min({0}, {1})", { a, b });
    Mat gmax = expr1("max({0}, {1})", { a, b });
    Mat gabs = expr1("absdiff({0}, {1})", { a, b });

    Mat emin, emax, eabs;
    cv::min(a, b, emin); cv::max(a, b, emax); cv::absdiff(a, b, eabs);
    EXPECT_EQ(0, cvtest::norm(gmin, emin, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(gmax, emax, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(gabs, eabs, NORM_INF));
}

// Operator precedence: '*' binds tighter than '+', unary minus on a literal.
TEST(Core_TExpr, addweighted_precedence)
{
    Mat a(20, 16, CV_32F), b(20, 16, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("{0} * 2.5 + {1} * -1.5 + 7", { a, b });
    Mat exp; cv::addWeighted(a, 2.5, b, -1.5, 7.0, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Named temporary via ';' assignment.
TEST(Core_TExpr, assignment)
{
    Mat a(18, 22, CV_32F), b(18, 22, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("t = {0} * 2.5; t + {1}", { a, b });
    Mat exp; cv::addWeighted(a, 2.5, b, 1.0, 0.0, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Tuple -> several outputs.
TEST(Core_TExpr, tuple_outputs)
{
    Mat a(14, 19, CV_32F), b(14, 19, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    std::vector<Mat> out;
    cv::texpr("({0} + {1}, {0} - {1})", std::vector<Mat>{ a, b }, out);
    ASSERT_EQ(out.size(), 2u);
    Mat eadd, esub; cv::add(a, b, eadd); cv::subtract(a, b, esub);
    EXPECT_LE(cvtest::norm(out[0], eadd, NORM_INF), 1e-3) << "sum";
    EXPECT_LE(cvtest::norm(out[1], esub, NORM_INF), 1e-3) << "diff";
}

// Grouping parens (NOT a tuple) inside a larger expression.
TEST(Core_TExpr, grouping_parens)
{
    Mat a(11, 13, CV_32F), b(11, 13, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = expr1("({0} + {1}) * 2", { a, b });
    Mat exp; cv::add(a, b, exp); exp *= 2.0;
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Type-cast function: float -> uint8 (saturating).
TEST(Core_TExpr, cast_uint8)
{
    Mat a(23, 17, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, -50.f, 300.f);

    Mat got = expr1("uint8({0})", { a });
    Mat exp; a.convertTo(exp, CV_8U);
    ASSERT_EQ(got.type(), exp.type());
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));
}

// pow() function call with a scalar exponent.
TEST(Core_TExpr, pow_call)
{
    Mat a(16, 16, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 5.f);

    Mat got = expr1("pow({0}, 2)", { a });
    Mat exp; cv::pow(a, 2.0, exp);
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

}} // namespace
