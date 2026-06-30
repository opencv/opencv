// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Tests for Layer 2: type inference + cast insertion. Builds programs directly with emitBinary (the
// same policy layer the cv::expression parser and the hand builders use), compiles them (bind kernels
// + temp-buffer liveness) and runs them through the executor, checking against the classic cv:: ops.

#include "../test_precomp.hpp"
#include "ew_compile.hpp"
#include "ew_exec.hpp"

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

}} // namespace
