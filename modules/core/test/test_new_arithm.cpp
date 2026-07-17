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

}} // namespace
