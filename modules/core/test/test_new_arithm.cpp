// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// White-box tests for the element-wise engine internals (cv::ew, declared in the module-internal
// src/arithm_expr.hpp - NOT part of the public API; the public surface is cv::add/... and cv::texpr,
// covered by test_new_arithm_extensive.cpp / test_arithm_expr.cpp). Two groups:
//   - the type-inference + cast-insertion policy (emitBinary) compiled and run through the executor;
//   - the single-op vertical slice (makeBinaryArithProgram / maybeAddCast) end-to-end.
// Both check against the classic cv:: ops.

#include "test_precomp.hpp"
#include "../src/arithm_expr.hpp"

namespace opencv_test { namespace {

using namespace cv::ew;

// emit shortcuts: a flexible literal, and a binary op over two slots.
static int K(TExpr& e, double v)              { return e.addConst(EW_DEPTH_NONE, Scalar(v), 1); }
static int bin(TExpr& e, TOp op, int a, int b){ return e.emitBinary(op, a, b); }

// Compile `e` and run it over the given inputs. The operands were already typed at build time
// (addInput carries each input's depth), so compile() just binds kernels + packs temp buffers.
static std::vector<Mat> run(TExpr& e, const std::vector<Mat>& inps)
{
    e.compile();
    std::vector<Mat> outs(e.noutputs);
    e.exec(inps.data(), outs.data());
    return outs;
}

// addWeighted(a,alpha,b,beta,gamma) = a*alpha + b*beta + gamma, built op-by-op via emitBinary
// (the temp buffers are allocated automatically by compile()'s liveness pass).
TEST(Core_EW_Compile, addweighted_f32)
{
    const int chans[] = { 1, 3 };
    double alpha = 2.5, beta = -1.5, gamma = 7.0;
    for (int ci = 0; ci < 2; ci++)
    {
        int H = 19, W = 23, cn = chans[ci];
        Mat a(H, W, CV_32FC(cn)), b(H, W, CV_32FC(cn));
        theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
        theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

        TExpr g;
        int ia = g.addInput(CV_32F), ib = g.addInput(CV_32F);
        int t0 = bin(g, OP_MUL, ia, K(g, alpha));
        int t1 = bin(g, OP_MUL, ib, K(g, beta));
        int t2 = bin(g, OP_ADD, t0, t1);
        g.output(bin(g, OP_ADD, t2, K(g, gamma)));

        std::vector<Mat> out = run(g, { a, b });

        Mat exp; cv::addWeighted(a, alpha, b, beta, gamma, exp);
        ASSERT_EQ(out[0].type(), exp.type());
        EXPECT_LE(cvtest::norm(out[0], exp, NORM_INF), 1e-3) << "cn=" << cn;
    }
}

// Mixed integer types: out = saturate_u8( saturate_u8(a*2.5) + b ), a,b are u8.
// emitBinary must insert u8->f32 input casts and f32->u8 result casts around each op (2.5 does not
// fit u8, so the direct u8 kernel is refused and the float working path is taken).
TEST(Core_EW_Compile, mixed_u8_inserts_casts)
{
    int H = 16, W = 24;
    Mat a(H, W, CV_8U), b(H, W, CV_8U);
    theRNG().fill(a, RNG::UNIFORM, 0, 60);
    theRNG().fill(b, RNG::UNIFORM, 0, 60);

    TExpr g;
    int ia = g.addInput(CV_8U), ib = g.addInput(CV_8U);
    int mul = bin(g, OP_MUL, ia, K(g, 2.5));   // -> u8 (natural)
    g.output(bin(g, OP_ADD, mul, ib));         // -> u8

    std::vector<Mat> out = run(g, { a, b });

    Mat t0, exp;
    a.convertTo(t0, CV_8U, 2.5);          // saturate_u8(a*2.5)
    cv::add(t0, b, exp);                  // saturate_u8(t0 + b)
    ASSERT_EQ(out[0].type(), exp.type());
    EXPECT_EQ(0, cvtest::norm(out[0], exp, NORM_INF));
}

// Tuple of two outputs from shared inputs: (a+b, a-b).
TEST(Core_EW_Compile, multi_output_tuple)
{
    int H = 14, W = 18;
    Mat a(H, W, CV_32F), b(H, W, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    TExpr g;
    int ia = g.addInput(CV_32F), ib = g.addInput(CV_32F);
    g.output(bin(g, OP_ADD, ia, ib));
    g.output(bin(g, OP_SUB, ia, ib));

    std::vector<Mat> out = run(g, { a, b });
    ASSERT_EQ(out.size(), 2u);

    Mat eadd, esub; cv::add(a, b, eadd); cv::subtract(a, b, esub);
    EXPECT_LE(cvtest::norm(out[0], eadd, NORM_INF), 1e-3) << "sum";
    EXPECT_LE(cvtest::norm(out[1], esub, NORM_INF), 1e-3) << "diff";
}

// Liveness: a linear chain of temps with disjoint lifetimes must share physical buffers.
// out = (((a+1)+1)+1)+1 == a+4 : the last add is redirected straight into the output slot, the
// three live temps share just 2 physical buffers.
TEST(Core_EW_Compile, temp_buffer_reuse)
{
    int H = 10, W = 13;
    Mat a(H, W, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);

    TExpr g;
    int x = g.addInput(CV_32F);
    for (int k = 0; k < 4; k++)
        x = bin(g, OP_ADD, x, K(g, 1.0));
    g.output(x);

    std::vector<Mat> out = run(g, { a });

    // last instruction writes straight into the OUTPUT slot (its producing temp was redirected)
    EXPECT_EQ(g.arginfo[g.prog[g.prog.size() - 1].result].kind, TExpr::OUTPUT);
    EXPECT_EQ(g.nbuffers, 2);        // disjoint lifetimes => only 2 physical buffers

    Mat exp; cv::add(a, Scalar(4.0), exp);
    EXPECT_LE(cvtest::norm(out[0], exp, NORM_INF), 1e-4);
}

// promoteArith is the auto result-depth rule (rdepth == -1). Checked against an INDEPENDENT hardcoded
// table (NOT computed from the engine): the extensive tests feed promoteArith to BOTH the engine and
// their own reference, so a wrong-but-consistent rule slips through there - this catches it. Also
// asserts commutativity, which a max-rank scheme silently breaks for mixed-sign / same-width floats.
TEST(Core_EW_Compile, promoteArith_rules)
{
    struct { int a, b, want; } cases[] = {
        // same signedness -> the wider one, sign kept
        { CV_8U, CV_8U, CV_8U }, { CV_8U, CV_16U, CV_16U }, { CV_8U, CV_64U, CV_64U },
        { CV_16S, CV_64S, CV_64S }, { CV_8S, CV_32S, CV_32S },
        // mixed sign, same width -> next-wider signed (64-bit has no wider int -> f64)
        { CV_8U, CV_8S, CV_16S }, { CV_16U, CV_16S, CV_32S },
        { CV_32U, CV_32S, CV_64S }, { CV_64U, CV_64S, CV_64F },
        // mixed sign, different width
        { CV_8S, CV_16U, CV_32S }, { CV_8U, CV_16S, CV_16S }, { CV_32S, CV_64U, CV_64F },
        // float + int -> smallest covering float
        { CV_16F, CV_8U, CV_16F }, { CV_16BF, CV_8U, CV_16BF }, { CV_16F, CV_16U, CV_32F },
        { CV_16F, CV_32S, CV_64F }, { CV_32F, CV_32S, CV_64F }, { CV_32F, CV_16S, CV_32F },
        // float + float
        { CV_16F, CV_32F, CV_32F }, { CV_16F, CV_16BF, CV_32F }, { CV_64F, CV_8U, CV_64F },
        // flexible operand (EW_DEPTH_NONE) does not force promotion
        { EW_DEPTH_NONE, CV_16U, CV_16U }, { CV_16U, EW_DEPTH_NONE, CV_16U },
        { EW_DEPTH_NONE, EW_DEPTH_NONE, EW_DEPTH_NONE },
    };
    for (auto& c : cases)
    {
        int got = promoteArith(c.a, c.b);
        EXPECT_EQ(got, c.want) << "promoteArith(" << c.a << "," << c.b << ")";
        EXPECT_EQ(promoteArith(c.b, c.a), got) << "not commutative at " << c.a << "," << c.b;
    }

    // exhaustive commutativity over all real depths
    const int depths[] = { CV_8U, CV_8S, CV_16U, CV_16S, CV_32U, CV_32S, CV_64U, CV_64S,
                           CV_16F, CV_16BF, CV_32F, CV_64F };
    for (int a : depths) for (int b : depths)
        EXPECT_EQ(promoteArith(a, b), promoteArith(b, a)) << "noncommutative at " << a << "," << b;
}

// ---------------------------------------------------------------------------
// Vertical slice: single-op programs (ADD/SUB/MUL/DIV/POW f32 and CAST) built via the hand builders
// (makeBinaryArithProgram / maybeAddCast) and run through the executor, checked against classic cv::.
// ---------------------------------------------------------------------------

// out = op(a, b), composed via the general binary-arith builder (the engine backing cv::add).
static Mat runBinary(TOp op, const Mat& a, const Mat& b, int rdepth)
{
    TExpr p; makeBinaryArithProgram(p, op, a.depth(), b.depth(), rdepth);
    Mat inps[] = {a, b}, out;
    p.exec(inps, &out);
    return out;
}

// out = cast(a), built through maybeAddCast (a single OP_CAST) and compiled.
static Mat runCast(const Mat& a, int rdepth)
{
    TExpr e;
    int s = e.addInput(a.depth());
    e.output(e.maybeAddCast(s, rdepth));
    e.compile();
    Mat out;
    e.exec(&a, &out);
    return out;
}

static void cvRef(TOp op, const Mat& a, const Mat& b, Mat& dst)
{
    switch (op)
    {
    case OP_ADD: cv::add(a, b, dst); break;
    case OP_SUB: cv::subtract(a, b, dst); break;
    case OP_MUL: cv::multiply(a, b, dst); break;
    case OP_DIV: cv::divide(a, b, dst); break;
    default: CV_Error(Error::StsBadArg, "unexpected op");
    }
}

// ADD/SUB/MUL/DIV on f32, single- and multi-channel, same shape.
TEST(Core_EW_Slice, binary_f32_same_shape)
{
    const TOp ops[] = { OP_ADD, OP_SUB, OP_MUL, OP_DIV };
    const int chans[] = { 1, 3, 4 };
    RNG& rng = theRNG();
    for (int oi = 0; oi < 4; oi++)
        for (int ci = 0; ci < 3; ci++)
        {
            int H = 17, W = 33, cn = chans[ci];
            Mat a(H, W, CV_32FC(cn)), b(H, W, CV_32FC(cn));
            rng.fill(a, RNG::UNIFORM, 1.f, 10.f);
            rng.fill(b, RNG::UNIFORM, 1.f, 10.f);

            Mat got = runBinary(ops[oi], a, b, CV_32F);
            Mat exp; cvRef(ops[oi], a, b, exp);

            ASSERT_EQ(got.size(), exp.size());
            ASSERT_EQ(got.type(), exp.type());
            EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3)
                << "op=" << opName(ops[oi]) << " cn=" << cn;
        }
}

// POW with a 1x1 (broadcast) exponent vs cv::pow(a, scalar).
TEST(Core_EW_Slice, pow_f32_scalar_exp)
{
    Mat a(20, 25, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 5.f);
    Mat e(1, 1, CV_32F, Scalar(2.0));

    Mat got = runBinary(OP_POW, a, e, CV_32F);
    Mat exp; cv::pow(a, 2.0, exp);

    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

typedef testing::TestWithParam<std::tuple<MatDepth, int, bool, bool, bool>> Core_EW_Pow;

TEST_P(Core_EW_Pow, fractional_simd_boundaries)
{
    const int depth = std::get<0>(GetParam());
    const int length = std::get<1>(GetParam());
    const bool special = std::get<2>(GetParam());
    const bool arrayExponent = std::get<3>(GetParam());
    const bool alias = std::get<4>(GetParam());
    const double positive[] = { 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0 };
    const double mixed[] = { 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
        -4.0, -1.0, -0.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), 16.0 };
    const double powers[] = { -1.5, 0.5, 1.5, 2.5 };
    Mat source64(1, length, CV_64F), powers64(1, length, CV_64F);
    for (int x = 0; x < length; x++)
    {
        source64.at<double>(x) = special ? mixed[x % 16] : positive[x % 8];
        powers64.at<double>(x) = arrayExponent ? powers[x % 4] : 1.5;
    }
    Mat image, exponents;
    source64.convertTo(image, depth);
    powers64.convertTo(exponents, depth);
    Mat result = alias ? image : Mat();
    if (arrayExponent)
    {
        TExpr program;
        makeBinaryArithProgram(program, OP_POW, depth, depth, depth);
        Mat inputs[] = { image, exponents };
        program.exec(inputs, &result);
    }
    else
    {
        cv::pow(image, 1.5, result);
    }
    ASSERT_EQ(result.type(), image.type());
    ASSERT_EQ(result.size(), image.size());
    if (alias)
    {
        EXPECT_EQ(result.data, image.data);
    }
    Mat actual;
    result.convertTo(actual, CV_64F);
    const double tolerance = depth == CV_16F ? 1e-3 :
        depth == CV_16BF ? 1e-2 : depth == CV_32F ? 2e-6 : 1e-12;
    for (int x = 0; x < length; x++)
    {
        const double expected = std::pow(source64.at<double>(x),
            powers64.at<double>(x));
        const double value = actual.at<double>(x);
        if (std::isnan(expected))
        {
            EXPECT_TRUE(std::isnan(value)) << "x=" << x;
        }
        else if (std::isinf(expected))
        {
            EXPECT_EQ(value, expected) << "x=" << x;
        }
        else
        {
            EXPECT_NEAR(value, expected, tolerance * std::abs(expected))
                << "x=" << x;
        }
    }
}

INSTANTIATE_TEST_CASE_P(Core, Core_EW_Pow,
    testing::Combine(testing::Values(CV_16F, CV_16BF, CV_32F, CV_64F),
        testing::Values(1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257),
        testing::Bool(), testing::Bool(), testing::Bool()));

// Row / column broadcasting (single channel) checked against repeat()+cv::add.
TEST(Core_EW_Slice, broadcast_row_col)
{
    int H = 12, W = 19;
    Mat a(H, W, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);

    {   // row vector broadcast over rows
        Mat brow(1, W, CV_32F);
        theRNG().fill(brow, RNG::UNIFORM, 1.f, 10.f);
        Mat got = runBinary(OP_ADD, a, brow, CV_32F);
        Mat bb, exp; cv::repeat(brow, H, 1, bb); cv::add(a, bb, exp);
        EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3) << "row";
    }
    {   // column vector broadcast over columns
        Mat bcol(H, 1, CV_32F);
        theRNG().fill(bcol, RNG::UNIFORM, 1.f, 10.f);
        Mat got = runBinary(OP_ADD, a, bcol, CV_32F);
        Mat bb, exp; cv::repeat(bcol, 1, W, bb); cv::add(a, bb, exp);
        EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3) << "col";
    }
}

// Channel broadcasting: HxWx3 * HxWx1 -> HxWx3.
TEST(Core_EW_Slice, broadcast_channel)
{
    int H = 15, W = 21;
    Mat a(H, W, CV_32FC3), b(H, W, CV_32FC1);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
    theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

    Mat got = runBinary(OP_MUL, a, b, CV_32F);

    std::vector<Mat> ach; cv::split(a, ach);
    for (size_t c = 0; c < ach.size(); c++) cv::multiply(ach[c], b, ach[c]);
    Mat exp; cv::merge(ach, exp);

    ASSERT_EQ(got.type(), exp.type());
    EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3);
}

// Saturating cast f32 <-> {u8, s32}.
TEST(Core_EW_Slice, cast_basic)
{
    Mat a(23, 31, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, -50.f, 300.f);   // exercise saturation for u8

    {
        Mat got = runCast(a, CV_8U);
        Mat exp; a.convertTo(exp, CV_8U);
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "f32->u8";
    }
    {
        Mat got = runCast(a, CV_32S);
        Mat exp; a.convertTo(exp, CV_32S);
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "f32->s32";
    }
    {
        Mat u; a.convertTo(u, CV_8U);
        Mat got = runCast(u, CV_32F);
        Mat exp; u.convertTo(exp, CV_32F);
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "u8->f32";
    }
}

// cv::add / cv::subtract on 32-bit ints SATURATE (SIMD via the local v_add_sat/v_sub_sat, scalar
// tail via the int64 work type) - uniform random data essentially never crosses the boundaries,
// so directed cases are mandatory (see the v_sat_arith brief): both rails, the 0 - INT_MIN case,
// mixed-sign non-overflow, and a random block checked against an exact int64 reference.
TEST(Core_EW_AddSub, saturation_s32_u32)
{
    const int W = 37;                       // odd width: SIMD body + scalar tail both covered
    {
        const int mx = INT_MAX, mn = INT_MIN;
        const int a[] = { mx,      mn,      mx, mn,  0,          -1, mx,      12345 };
        const int b[] = { 1,       -1,      mx, mn,  mn,         mx, -1,      -54321 };
        // exact int64 references
        Mat A(1, 8, CV_32S, (void*)a), B(1, 8, CV_32S, (void*)b), sum, dif;
        cv::add(A, B, sum);
        cv::subtract(A, B, dif);
        for (int i = 0; i < 8; i++)
        {
            int64_t rs = (int64_t)a[i] + b[i], rd = (int64_t)a[i] - b[i];
            EXPECT_EQ(sum.at<int>(i), (int)std::min<int64_t>(std::max<int64_t>(rs, mn), mx)) << "add s32 case " << i;
            EXPECT_EQ(dif.at<int>(i), (int)std::min<int64_t>(std::max<int64_t>(rd, mn), mx)) << "sub s32 case " << i;
        }
    }
    {
        const unsigned mx = UINT_MAX;
        const unsigned a[] = { mx, 0, mx, 5,  0, 100 };
        const unsigned b[] = { 1,  1, mx, 5,  0, 7 };
        Mat A(1, 6, CV_32U, (void*)a), B(1, 6, CV_32U, (void*)b), sum, dif;
        cv::add(A, B, sum);
        cv::subtract(A, B, dif);
        for (int i = 0; i < 6; i++)
        {
            uint64_t rs = (uint64_t)a[i] + b[i];
            int64_t  rd = (int64_t)a[i] - b[i];
            EXPECT_EQ(sum.at<unsigned>(i), (unsigned)std::min<uint64_t>(rs, mx)) << "add u32 case " << i;
            EXPECT_EQ(dif.at<unsigned>(i), (unsigned)std::max<int64_t>(rd, 0)) << "sub u32 case " << i;
        }
    }
    // random block spanning the full range (so saturation DOES occur), vs the int64 reference
    {
        Mat a(15, W, CV_32S), b(15, W, CV_32S), sum, dif;
        theRNG().fill(a, RNG::UNIFORM, INT_MIN, INT_MAX);
        theRNG().fill(b, RNG::UNIFORM, INT_MIN, INT_MAX);
        cv::add(a, b, sum);
        cv::subtract(a, b, dif);
        for (int y = 0; y < a.rows; y++)
            for (int x = 0; x < W; x++)
            {
                int64_t rs = (int64_t)a.at<int>(y, x) + b.at<int>(y, x);
                int64_t rd = (int64_t)a.at<int>(y, x) - b.at<int>(y, x);
                ASSERT_EQ(sum.at<int>(y, x), (int)std::min<int64_t>(std::max<int64_t>(rs, INT_MIN), INT_MAX)) << y << "," << x;
                ASSERT_EQ(dif.at<int>(y, x), (int)std::min<int64_t>(std::max<int64_t>(rd, INT_MIN), INT_MAX)) << y << "," << x;
            }
    }
}

// cv::multiply on 64-bit integers: the product is exact (a double holds only 53 bits) and saturates
// at the depth's range. Uniform random data never gets near the boundaries, so the cases are directed:
// products above 2^53, exactly at the limit, one past it, and the INT64_MIN edge.
TEST(Core_EW_Mul, exact_and_saturating_s64_u64)
{
    {
        // The last four cases sit in the band just below 2^64 where the double estimate cannot decide.
        const uint64_t a[] = { UINT64_MAX / 2, UINT64_MAX, 1ull << 32, (1ull << 32) + 1, 0, UINT64_MAX, 3, 3037000499ull, 9007199254740993ull, (1ull << 32) - 256, (1ull << 32) + 1, UINT64_MAX - 1000 };
        const uint64_t b[] = { 1, 1, 1ull << 32, (1ull << 32) - 1, UINT64_MAX, 2, UINT64_MAX / 3, 3037000499ull, 3, (1ull << 32) + 256, (1ull << 32) + 1, 1 };
        const uint64_t e[] = { 9223372036854775807ull, UINT64_MAX, UINT64_MAX, UINT64_MAX, 0, UINT64_MAX, UINT64_MAX, 9223372030926249001ull, 27021597764222979ull, 18446744073709486080ull, UINT64_MAX, UINT64_MAX - 1000 };
        const int n = (int)(sizeof(a)/sizeof(a[0]));
        Mat A(1, n, CV_64U, (void*)a), B(1, n, CV_64U, (void*)b), P;
        cv::multiply(A, B, P);
        ASSERT_EQ(P.type(), CV_64U);
        for (int i = 0; i < n; i++)
            EXPECT_EQ(P.at<uint64_t>(i), e[i]) << "mul u64 case " << i;
    }
    {
        // The last three cases sit in the band just below 2^63 where the double estimate cannot decide.
        const int64_t a[] = { INT64_MAX, INT64_MIN, INT64_MIN, -1, 3037000500ll, -3037000500ll, -(1ll << 31), 1ll << 32, 0, 9007199254740993ll, (1ll << 31) - 256, -(1ll << 31) + 256, INT64_MAX - 1000 };
        const int64_t b[] = { 1, 1, -1, 3074457345618258602ll, 3037000500ll, 3037000500ll, 1ll << 32, 1ll << 31, INT64_MIN, -3, (1ll << 32) + 512, (1ll << 32) + 512, 1 };
        const int64_t e[] = { INT64_MAX, INT64_MIN, INT64_MAX, -3074457345618258602ll, INT64_MAX, INT64_MIN, INT64_MIN, INT64_MAX, 0, -27021597764222979ll, 9223372036854644736ll, -9223372036854644736ll, INT64_MAX - 1000 };
        const int n = (int)(sizeof(a)/sizeof(a[0]));
        Mat A(1, n, CV_64S, (void*)a), B(1, n, CV_64S, (void*)b), P;
        cv::multiply(A, B, P);
        ASSERT_EQ(P.type(), CV_64S);
        for (int i = 0; i < n; i++)
            EXPECT_EQ(P.at<int64_t>(i), e[i]) << "mul s64 case " << i;
    }
}

// A scale other than 1 keeps the double computation.
TEST(Core_EW_Mul, scaled_s64)
{
    Mat A(1, 3, CV_64S), B(1, 3, CV_64S), P;
    A.at<int64_t>(0) = 3;         B.at<int64_t>(0) = 4;
    A.at<int64_t>(1) = INT64_MAX; B.at<int64_t>(1) = 2;
    A.at<int64_t>(2) = -5;        B.at<int64_t>(2) = 4;
    cv::multiply(A, B, P, 2.0);
    EXPECT_EQ(P.at<int64_t>(0), 24);
    EXPECT_EQ(P.at<int64_t>(1), INT64_MAX);
    EXPECT_EQ(P.at<int64_t>(2), -40);
}

// cv::divide on 64-bit integers: the quotient is exact (x / 1 == x above 2^53), rounded to nearest with
// ties to even like the narrower depths, divide-by-zero gives 0, and the result saturates at the range.
TEST(Core_EW_Div, exact_and_saturating_s64_u64)
{
    {
        const uint64_t a[] = { UINT64_MAX - 1, 9007199254740993ull, 9007199254740993ull, 7, 5, 1ull << 63, UINT64_MAX, UINT64_MAX, 1, 12345 };
        const uint64_t b[] = { 1, 1, 3, 2, 2, UINT64_MAX - 1, 3, UINT64_MAX, 3, 0 };
        const uint64_t e[] = { UINT64_MAX - 1, 9007199254740993ull, 3002399751580331ull, 4, 2, 1, 6148914691236517205ull, 1, 0, 0 };
        const int n = (int)(sizeof(a)/sizeof(a[0]));
        Mat A(1, n, CV_64U, (void*)a), B(1, n, CV_64U, (void*)b), Q;
        cv::divide(A, B, Q);
        ASSERT_EQ(Q.type(), CV_64U);
        for (int i = 0; i < n; i++)
            EXPECT_EQ(Q.at<uint64_t>(i), e[i]) << "div u64 case " << i;
    }
    {
        const int64_t a[] = { INT64_MIN, -INT64_MAX, 3074457345618258602ll, -9007199254740993ll, -5, -7, 7, INT64_MIN, INT64_MAX, INT64_MIN, -12345 };
        const int64_t b[] = { -1, 1, -2, 3, 2, 2, -2, 1, -1, 2, 0 };
        const int64_t e[] = { INT64_MAX, -INT64_MAX, -1537228672809129301ll, -3002399751580331ll, -2, -4, -4, INT64_MIN, -INT64_MAX, -4611686018427387904ll, 0 };
        const int n = (int)(sizeof(a)/sizeof(a[0]));
        Mat A(1, n, CV_64S, (void*)a), B(1, n, CV_64S, (void*)b), Q;
        cv::divide(A, B, Q);
        ASSERT_EQ(Q.type(), CV_64S);
        for (int i = 0; i < n; i++)
            EXPECT_EQ(Q.at<int64_t>(i), e[i]) << "div s64 case " << i;
    }
}

// ---- 64-bit multiply / divide on large arrays and with a Scalar operand ----
// The directed cases above are tiny. These run well over 100000 elements, so the vector loop, its
// tail and the parallel split all see values above 2^53, and the expected values come from a separate
// exact computation (a 128-bit schoolbook product, not the divide-based overflow check of the kernel).
static void ewMul128(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo)
{
    const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    lo = (p00 & 0xffffffffu) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}
static uint64_t ewMulRefU64(uint64_t a, uint64_t b)
{
    uint64_t hi, lo;
    ewMul128(a, b, hi, lo);
    return hi ? UINT64_MAX : lo;
}
static int64_t ewMulRefS64(int64_t a, int64_t b)
{
    const bool neg = (a < 0) != (b < 0);
    uint64_t hi, lo;
    ewMul128(a < 0 ? 0 - (uint64_t)a : (uint64_t)a, b < 0 ? 0 - (uint64_t)b : (uint64_t)b, hi, lo);
    const uint64_t limit = (uint64_t)INT64_MAX + (neg ? 1 : 0);
    if (hi || lo > limit)
        return neg ? INT64_MIN : INT64_MAX;
    return neg ? (int64_t)(0 - lo) : (int64_t)lo;
}
static uint64_t ewDivRefU64(uint64_t a, uint64_t b)
{
    if (b == 0)
        return 0;
    uint64_t q = a / b;
    const uint64_t r = a % b;
    return (r > b - r || (r == b - r && (q & 1))) ? q + 1 : q;
}
static int64_t ewDivRefS64(int64_t a, int64_t b)
{
    if (b == 0)
        return 0;
    const bool neg = (a < 0) != (b < 0);
    const uint64_t q = ewDivRefU64(a < 0 ? 0 - (uint64_t)a : (uint64_t)a, b < 0 ? 0 - (uint64_t)b : (uint64_t)b);
    if (q > (uint64_t)INT64_MAX + (neg ? 1 : 0))
        return neg ? INT64_MIN : INT64_MAX;
    return neg ? (int64_t)(0 - q) : (int64_t)q;
}
// A mix of magnitudes, every bit width from 0 to 64: both the small-operand fast path and the
// overflow check run, and many values sit above 2^53.
static uint64_t ewRandBits(RNG& rng)
{
    const uint64_t v = ((uint64_t)rng.next() << 32) | rng.next();
    return v >> rng.uniform(0, 64);
}
static int64_t ewRandSigned(RNG& rng)
{
    const int64_t v = (int64_t)(ewRandBits(rng) >> 1);
    return (rng.next() & 1) ? v : -v;
}

TEST(Core_EW_Mul, large_array_s64_u64)
{
    const int n = 200003;   // odd and well above 100000
    RNG rng(0x64b17);
    Mat Au(1, n, CV_64U), Bu(1, n, CV_64U), As(1, n, CV_64S), Bs(1, n, CV_64S), Pu, Ps;
    for (int i = 0; i < n; i++)
    {
        Au.at<uint64_t>(i) = ewRandBits(rng);
        Bu.at<uint64_t>(i) = ewRandBits(rng);
        As.at<int64_t>(i) = ewRandSigned(rng);
        Bs.at<int64_t>(i) = ewRandSigned(rng);
    }
    cv::multiply(Au, Bu, Pu);
    cv::multiply(As, Bs, Ps);
    int above53 = 0;
    for (int i = 0; i < n; i++)
    {
        ASSERT_EQ(Pu.at<uint64_t>(i), ewMulRefU64(Au.at<uint64_t>(i), Bu.at<uint64_t>(i))) << "u64 at " << i;
        ASSERT_EQ(Ps.at<int64_t>(i), ewMulRefS64(As.at<int64_t>(i), Bs.at<int64_t>(i))) << "s64 at " << i;
        above53 += Pu.at<uint64_t>(i) > (1ull << 53);
    }
    EXPECT_GT(above53, n / 4) << "the data should mostly exceed 2^53";
}

TEST(Core_EW_Div, large_array_s64_u64)
{
    const int n = 200003;
    RNG rng(0x64d17);
    Mat Au(1, n, CV_64U), Bu(1, n, CV_64U), As(1, n, CV_64S), Bs(1, n, CV_64S), Qu, Qs;
    for (int i = 0; i < n; i++)
    {
        Au.at<uint64_t>(i) = ewRandBits(rng);
        Bu.at<uint64_t>(i) = ewRandBits(rng);
        As.at<int64_t>(i) = ewRandSigned(rng);
        Bs.at<int64_t>(i) = ewRandSigned(rng);
    }
    cv::divide(Au, Bu, Qu);
    cv::divide(As, Bs, Qs);
    for (int i = 0; i < n; i++)
    {
        ASSERT_EQ(Qu.at<uint64_t>(i), ewDivRefU64(Au.at<uint64_t>(i), Bu.at<uint64_t>(i))) << "u64 at " << i;
        ASSERT_EQ(Qs.at<int64_t>(i), ewDivRefS64(As.at<int64_t>(i), Bs.at<int64_t>(i))) << "s64 at " << i;
    }
}

// Array by Scalar. A Scalar holds doubles, so the operand is a small integer that a double represents
// exactly; the array side is above 2^53.
TEST(Core_EW_MulDiv, scalar_operand_s64_u64)
{
    const int n = 200003;
    RNG rng(0x64c17);
    Mat Au(1, n, CV_64U), As(1, n, CV_64S), Pu, Ps, Qu, Qs;
    for (int i = 0; i < n; i++)
    {
        Au.at<uint64_t>(i) = ewRandBits(rng) | (1ull << 55);
        As.at<int64_t>(i) = ewRandSigned(rng) | (1ll << 55);
    }
    cv::multiply(Au, Scalar(3.0), Pu);
    cv::multiply(As, Scalar(-3.0), Ps);
    cv::divide(Au, Scalar(7.0), Qu);
    cv::divide(As, Scalar(-7.0), Qs);
    ASSERT_EQ(Pu.type(), CV_64U);
    ASSERT_EQ(Ps.type(), CV_64S);
    for (int i = 0; i < n; i++)
    {
        const uint64_t a = Au.at<uint64_t>(i);
        const int64_t b = As.at<int64_t>(i);
        ASSERT_EQ(Pu.at<uint64_t>(i), ewMulRefU64(a, 3)) << "mul u64 at " << i;
        ASSERT_EQ(Ps.at<int64_t>(i), ewMulRefS64(b, -3)) << "mul s64 at " << i;
        ASSERT_EQ(Qu.at<uint64_t>(i), ewDivRefU64(a, 7)) << "div u64 at " << i;
        ASSERT_EQ(Qs.at<int64_t>(i), ewDivRefS64(b, -7)) << "div s64 at " << i;
    }
}

TEST(Core_EW_AddSub, saturation_s64_u64)
{
    {
        const int64_t mx = INT64_MAX, mn = INT64_MIN;
        const int64_t a[]    = { mx, mn, mx, mn, 0,  -1,     mx,     mx, mn, 12345 };
        const int64_t b[]    = { 1,  -1, mx, mn, mn, mx,     -1,     mn, mx, -54321 };
        const int64_t rsum[] = { mx, mn, mx, mn, mn, mx - 1, mx - 1, -1, -1, -41976 };
        const int64_t rdif[] = { mx - 1, mn + 1, 0, 0, mx, mn, mx, mx, mn, 66666 };
        Mat A(1, 10, CV_64S, (void*)a), B(1, 10, CV_64S, (void*)b), sum, dif;
        cv::add(A, B, sum);
        cv::subtract(A, B, dif);
        for (int i = 0; i < 10; i++)
        {
            EXPECT_EQ(rsum[i], sum.at<int64_t>(i)) << "add s64 case " << i;
            EXPECT_EQ(rdif[i], dif.at<int64_t>(i)) << "sub s64 case " << i;
        }
    }
    {
        const uint64_t mx = UINT64_MAX, half = 1ULL << 63;
        const uint64_t a[]    = { mx, 0, mx, 5,  0, 100, half };
        const uint64_t b[]    = { 1,  1, mx, 5,  0, 7,   half };
        const uint64_t rsum[] = { mx, 1, mx, 10, 0, 107, mx };
        const uint64_t rdif[] = { mx - 1, 0, 0, 0, 0, 93, 0 };
        Mat A(1, 7, CV_64U, (void*)a), B(1, 7, CV_64U, (void*)b), sum, dif;
        cv::add(A, B, sum);
        cv::subtract(A, B, dif);
        for (int i = 0; i < 7; i++)
        {
            EXPECT_EQ(rsum[i], sum.at<uint64_t>(i)) << "add u64 case " << i;
            EXPECT_EQ(rdif[i], dif.at<uint64_t>(i)) << "sub u64 case " << i;
        }
    }
}

}} // namespace
