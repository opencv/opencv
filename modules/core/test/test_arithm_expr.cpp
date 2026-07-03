// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Tests for the public cv::texpr() string frontend. Exercises placeholders, operator precedence,
// function calls, type casts, assignments and tuple (multi-) outputs. Limited to ops with kernels
// today (arithmetic / cast / pow / min / max / absdiff).

#include "test_precomp.hpp"
#include "../src/arithm_expr.hpp"   // ew::absdiffResultDepth - the engine's absdiff auto-type rule

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

// pow over the interesting exponents: the special-cased 2/3/0.5/1/0, the general exp/log path
// (2.5, -1.5), and negative bases (integer exponent -> exact signed result, fractional -> NaN),
// against the double std::pow reference. Sizes chosen to exercise both the SIMD body and the tail.
TEST(Core_TExpr, pow_exponents)
{
    for (int depth : { CV_32F, CV_64F })
    {
        const double eps = depth == CV_32F ? 1e-6 : 1e-9;
        Mat a0(37, 41, CV_64F), a;
        theRNG().fill(a0, RNG::UNIFORM, 0.05, 9.);
        a0.convertTo(a, depth);
        for (double p : { 2., 3., 0.5, 1., 0., 2.5, -1.5 })
        {
            Mat got = expr1(cv::format("pow({0}, %.10g)", p), { a });
            ASSERT_EQ(got.depth(), depth) << "p=" << p;
            Mat ad, gd;
            a.convertTo(ad, CV_64F); got.convertTo(gd, CV_64F);
            double maxerr = 0;
            for (int y = 0; y < a.rows; y++)
                for (int x = 0; x < a.cols; x++)
                {
                    double r = std::pow(ad.at<double>(y, x), p);
                    maxerr = std::max(maxerr, std::abs(gd.at<double>(y, x) - r) / std::max(1.0, std::abs(r)));
                }
            EXPECT_LE(maxerr, eps) << "depth=" << depth << " p=" << p;
        }
    }
}

// negative bases: integer exponents keep exact signed results (scalar patch path), a fractional
// exponent yields NaN - both matching std::pow
TEST(Core_TExpr, pow_negative_base)
{
    Mat a(9, 13, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, -5.f, -1.f);

    Mat got3 = expr1("pow({0}, 3)", { a });
    for (int y = 0; y < a.rows; y++)
        for (int x = 0; x < a.cols; x++)
            ASSERT_NEAR(got3.at<float>(y, x), std::pow((double)a.at<float>(y, x), 3.), 1e-2);

    Mat gotf = expr1("pow({0}, 2.5)", { a });
    for (int y = 0; y < a.rows; y++)
        for (int x = 0; x < a.cols; x++)
            ASSERT_TRUE(cvIsNaN(gotf.at<float>(y, x))) << "pow(neg, frac) must be NaN";
}

// unary minus and abs() - compositions over the binary family (no dedicated kernels)
TEST(Core_TExpr, neg_abs)
{
    for (int depth : { CV_32F, CV_16S })
    {
        Mat a(15, 19, depth);
        theRNG().fill(a, RNG::UNIFORM, -100, 100);

        Mat gneg = expr1("-{0}", { a });
        Mat eneg; cv::subtract(Scalar(0), a, eneg);
        ASSERT_EQ(gneg.depth(), depth);
        EXPECT_EQ(0, cvtest::norm(gneg, eneg, NORM_INF)) << "neg depth=" << depth;

        // abs == absdiff(a, 0) INCLUDING the texpr auto result type rule: signed input -> the
        // UNSIGNED type of the same width (|SHRT_MIN| fits u16 exactly, no saturation). NB this
        // deliberately differs from the public cv::absdiff, whose auto depth keeps the source
        // type for 4.x compatibility - the VALUES agree, the depth rule is the engine's own.
        Mat gabs = expr1("abs({0})", { a });
        ASSERT_EQ(gabs.depth(), cv::ew::absdiffResultDepth(depth)) << "texpr absdiff type rule";
        Mat eabs; cv::absdiff(a, Scalar(0), eabs);
        eabs.convertTo(eabs, gabs.depth());
        EXPECT_EQ(0, cvtest::norm(gabs, eabs, NORM_INF)) << "abs depth=" << depth;
    }
}

// the abs(a - b) -> absdiff(a, b) peephole: on unsigned data the literal semantics (saturating
// subtract) would give max(a-b, 0) - the rewrite must give the true |a - b| everywhere
TEST(Core_TExpr, abs_sub_peephole)
{
    for (int depth : { CV_8U, CV_16S, CV_32F })
    {
        Mat a(23, 17, depth), b(23, 17, depth);
        theRNG().fill(a, RNG::UNIFORM, 0, 100);
        theRNG().fill(b, RNG::UNIFORM, 0, 100);

        Mat got = expr1("abs({0} - {1})", { a, b });
        ASSERT_EQ(got.depth(), cv::ew::absdiffResultDepth(depth)) << "depth=" << depth;
        Mat exp; cv::absdiff(a, b, exp);
        exp.convertTo(exp, got.depth());          // cv::absdiff auto KEEPS the source depth (4.x)
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "depth=" << depth;
    }
}

// clamp: scalar bounds (the common shape), array bounds, and type preservation
TEST(Core_TExpr, clamp)
{
    for (int depth : { CV_8U, CV_16S, CV_32F, CV_64F })
    {
        Mat a(25, 31, depth);
        theRNG().fill(a, RNG::UNIFORM, -100, 355);

        Mat got = expr1("clamp({0}, 10, 200)", { a });
        Mat emax, exp;
        cv::max(a, 10.0, emax); cv::min(emax, 200.0, exp);
        ASSERT_EQ(got.depth(), depth) << "clamp must keep the operand type";
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "depth=" << depth;
    }
    // array bounds
    Mat x(14, 22, CV_32F), lo(14, 22, CV_32F), hi(14, 22, CV_32F);
    theRNG().fill(x, RNG::UNIFORM, -10.f, 10.f);
    theRNG().fill(lo, RNG::UNIFORM, -5.f, 0.f);
    theRNG().fill(hi, RNG::UNIFORM, 0.f, 5.f);
    Mat got = expr1("clamp({0}, {1}, {2})", { x, lo, hi });
    Mat emax, exp;
    cv::max(x, lo, emax); cv::min(emax, hi, exp);
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));
}

// '**' operator: pow alias, binds tighter than '*', right-associative
TEST(Core_TExpr, pow_operator)
{
    Mat a(13, 18, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 0.5f, 2.f);

    Mat got = expr1("{0} ** 2", { a });
    Mat exp; cv::pow(a, 2.0, exp);
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));

    // precedence: 3 * a ** 2 == 3 * (a ** 2)
    Mat got2 = expr1("3 * {0} ** 2", { a });
    Mat exp2 = 3.0 * exp;
    EXPECT_LE(cvtest::norm(got2, exp2, NORM_INF), 1e-4);

    // right associativity: a ** 2 ** 3 == a ** (2 ** 3) == a ** 8
    Mat got3 = expr1("{0} ** 2 ** 3", { a });
    Mat exp3; cv::pow(a, 8.0, exp3);
    EXPECT_LE(cvtest::norm(got3, exp3, NORM_INF), 1e-4);
}

// '?:' conditional: select alias with the lowest precedence; right-associative chains
TEST(Core_TExpr, ternary_operator)
{
    Mat a(19, 23, CV_32F), b(19, 23, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 0.f, 100.f);
    theRNG().fill(b, RNG::UNIFORM, 0.f, 100.f);

    // max via ?: - the condition is a full comparison (lower precedence than '>')
    Mat got = expr1("{0} > {1} ? {0} : {1}", { a, b });
    Mat exp; cv::max(a, b, exp);
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));

    // arithmetic in every position without parentheses
    Mat got2 = expr1("{0} - {1} > 10 ? {0} + 1 : {1} * 2", { a, b });
    Mat mask = (a - b > 10), e1 = a + 1, e2 = b * 2, exp2 = e2.clone();
    e1.copyTo(exp2, mask);
    EXPECT_LE(cvtest::norm(got2, exp2, NORM_INF), 1e-4);

    // right-associative chain: c1 ? x : c2 ? y : z
    Mat got3 = expr1("{0} > 66 ? 1 : {0} > 33 ? 2 : 3", { a });
    Mat exp3(a.size(), CV_32F);
    for (int y = 0; y < a.rows; y++)
        for (int x = 0; x < a.cols; x++)
        {
            float v = a.at<float>(y, x);
            exp3.at<float>(y, x) = v > 66 ? 1.f : v > 33 ? 2.f : 3.f;
        }
    EXPECT_EQ(0, cvtest::norm(got3, exp3, NORM_INF));
}

// hypot(x, y) / its cv-flavored alias mag(x, y): naive sqrt(x^2 + y^2), matching cv::magnitude;
// kernels exist for the float depths only (f16/bf16/f32/f64)
TEST(Core_TExpr, hypot_mag)
{
    for (int depth : { CV_32F, CV_64F })
    {
        Mat x(23, 31, depth), y(23, 31, depth);
        theRNG().fill(x, RNG::UNIFORM, -100, 100);
        theRNG().fill(y, RNG::UNIFORM, -100, 100);

        Mat got = expr1("hypot({0}, {1})", { x, y });
        ASSERT_EQ(got.depth(), depth);
        Mat exp; cv::magnitude(x, y, exp);
        EXPECT_LE(cvtest::norm(got, exp, NORM_INF), depth == CV_32F ? 1e-4 : 1e-9);

        Mat got2 = expr1("mag({0}, {1})", { x, y });   // alias
        EXPECT_EQ(0, cvtest::norm(got2, got, NORM_INF));
    }
    // broadcast branch: hypot(array, scalar)
    Mat x(11, 17, CV_32F);
    theRNG().fill(x, RNG::UNIFORM, -10.f, 10.f);
    Mat got = expr1("hypot({0}, 3)", { x });
    for (int r = 0; r < x.rows; r++)
        for (int c = 0; c < x.cols; c++)
        {
            float v = x.at<float>(r, c);
            ASSERT_NEAR(got.at<float>(r, c), std::sqrt(v*v + 9.f), 1e-4) << r << "," << c;
        }
    // f16: T -> T through the native kernel (f32 hub inside)
    Mat xh, yh, y16(11, 17, CV_32F);
    x.convertTo(xh, CV_16F);
    theRNG().fill(y16, RNG::UNIFORM, -10.f, 10.f);
    y16.convertTo(yh, CV_16F);
    Mat goth = expr1("hypot({0}, {1})", { xh, yh });
    ASSERT_EQ(goth.depth(), CV_16F);
}

// per-element (array) exponent
TEST(Core_TExpr, pow_array_exponent)
{
    Mat a(21, 27, CV_32F), b(21, 27, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 0.1f, 5.f);
    theRNG().fill(b, RNG::UNIFORM, -2.f, 3.f);

    Mat got = expr1("pow({0}, {1})", { a, b });
    double maxerr = 0;
    for (int y = 0; y < a.rows; y++)
        for (int x = 0; x < a.cols; x++)
        {
            double r = std::pow((double)a.at<float>(y, x), (double)b.at<float>(y, x));
            maxerr = std::max(maxerr, std::abs(got.at<float>(y, x) - r) / std::max(1.0, std::abs(r)));
        }
    EXPECT_LE(maxerr, 1e-6);
}

// ---------------------------------------------------------------------------------- unary math
// Golden result = the double-precision std:: function applied per element (never the op under
// test); the tolerance is relative, scaled by the output depth's precision.

typedef double (*mathRef)(double);
struct MathOpRef { const char* name; mathRef ref; };
static const MathOpRef mathOps[] = {
    { "sqrt", std::sqrt }, { "exp", std::exp }, { "log", std::log },
    { "sin", std::sin }, { "cos", std::cos }, { "tanh", std::tanh }, { "erf", std::erf },
    { "relu", [](double x) { return x > 0 ? x : 0.; } },
};

// max |got - ref| / max(1, |ref|) over the array, both evaluated in f64
static double relErr(const Mat& got, const Mat& in, mathRef ref)
{
    Mat gotd, ind;
    got.convertTo(gotd, CV_64F);
    in.convertTo(ind, CV_64F);
    double maxerr = 0;
    for (int y = 0; y < ind.rows; y++)
        for (int x = 0; x < ind.cols; x++)
        {
            double r = ref(ind.at<double>(y, x));
            double e = std::abs(gotd.at<double>(y, x) - r) / std::max(1.0, std::abs(r));
            maxerr = std::max(maxerr, e);
        }
    return maxerr;
}

typedef testing::TestWithParam<MatDepth> Core_TExpr_Math;

TEST_P(Core_TExpr_Math, unary_accuracy)
{
    const int depth = GetParam();
    // eps: half-precision types are exact to ~2^-8/2^-11 per element; the f32 kernels are
    // polynomial approximations (a few ulp); f64 sqrt/exp/log are also vectorized polynomials
    const double eps = depth == CV_16F ? 2e-3 : depth == CV_16BF ? 1.6e-2
                     : depth == CV_32F ? 1e-6 : 1e-9;
    Mat a0(37, 41, CV_32F);
    theRNG().fill(a0, RNG::UNIFORM, 0.05f, 9.f);   // positive: one range serves log/sqrt too
    Mat a;
    a0.convertTo(a, depth);

    for (const MathOpRef& m : mathOps)
    {
        Mat got = expr1(cv::format("%s({0})", m.name), { a });
        ASSERT_EQ(got.depth(), depth) << m.name << ": math must be T -> T on float inputs";
        EXPECT_LE(relErr(got, a, m.ref), eps) << m.name << " depth=" << depth;
    }

    // negative inputs for the ops defined there (skip log/sqrt)
    Mat b0(37, 41, CV_32F), b;
    theRNG().fill(b0, RNG::UNIFORM, -8.f, 8.f);
    b0.convertTo(b, depth);
    for (const char* name : { "exp", "sin", "cos", "tanh", "erf", "relu" })
    {
        const MathOpRef* m = nullptr;
        for (const MathOpRef& c : mathOps) if (!strcmp(c.name, name)) m = &c;
        Mat got = expr1(cv::format("%s({0})", name), { b });
        EXPECT_LE(relErr(got, b, m->ref), eps) << name << "(neg) depth=" << depth;
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Core_TExpr_Math,
                        testing::Values(CV_16F, CV_16BF, CV_32F, CV_64F));

// integer input computes in the float domain and lands in f32
TEST(Core_TExpr, math_int_input)
{
    Mat a(19, 23, CV_8U);
    theRNG().fill(a, RNG::UNIFORM, 1, 100);
    Mat got = expr1("sqrt({0})", { a });
    ASSERT_EQ(got.depth(), CV_32F);
    EXPECT_LE(relErr(got, a, std::sqrt), 1e-6);
}

// in-place unary math (dst aliases src): the tail backoff must not re-apply the op
TEST(Core_TExpr, math_inplace)
{
    Mat a(21, 31, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 0.1f, 9.f);
    Mat ref = expr1("sqrt({0})", { a });
    std::vector<Mat> out{ a };                     // preallocated == input => in-place
    cv::texpr("sqrt({0})", std::vector<Mat>{ a }, out);
    EXPECT_EQ(0, cvtest::norm(out[0], ref, NORM_INF));
}

// ------------------------------------------------------------------------------------- select
TEST(Core_TExpr, select_basic)
{
    for (int type : { CV_8UC1, CV_16SC1, CV_32FC1, CV_64FC1 })
    {
        Mat a(25, 33, type), b(25, 33, type);
        theRNG().fill(a, RNG::UNIFORM, 0, 100);
        theRNG().fill(b, RNG::UNIFORM, 0, 100);

        Mat got = expr1("select({0} > {1}, {0}, {1})", { a, b });   // == max(a, b)
        Mat exp; cv::max(a, b, exp);
        ASSERT_EQ(got.type(), exp.type()) << "type=" << type;
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "type=" << type;
    }
}

// one branch is a scalar constant (broadcast stepx == 0 inside the kernel)
TEST(Core_TExpr, select_const_branch)
{
    Mat a(17, 29, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, -10.f, 10.f);
    Mat got = expr1("select({0} > 0, {0}, 0)", { a });   // == relu
    Mat exp = expr1("relu({0})", { a });
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));
}

// non-1-byte mask is normalized via `mask != 0`, not a value cast
TEST(Core_TExpr, select_float_mask)
{
    Mat m(15, 27, CV_32F), a(15, 27, CV_32F), b(15, 27, CV_32F);
    theRNG().fill(m, RNG::UNIFORM, -1.f, 1.f);
    theRNG().fill(a, RNG::UNIFORM, 0.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 0.f, 10.f);
    m.at<float>(3, 5) = 0.f;                       // exact zero -> must take branch b
    m.at<float>(7, 7) = 0.25f;                     // would round/saturate to 0 under a value cast

    Mat got = expr1("select({0}, {1}, {2})", { m, a, b });
    Mat mask = (m != 0), exp;
    exp = b.clone(); a.copyTo(exp, mask);
    EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF));
}

}} // namespace
