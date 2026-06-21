// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Tests for Layer 2: the graph compiler. Builds expression graphs by hand, compiles them to
// EwProgram (type inference + auto OP_CAST insertion + temp-buffer liveness) and runs them
// through the executor, checking against the classic cv:: ops.

#include "../test_precomp.hpp"
#include "ew_compile.hpp"
#include "ew_exec.hpp"

namespace opencv_test { namespace {

using namespace cv::ew;

static std::vector<Mat> run(const EwGraph& g,
                            const std::vector<Mat>& inps, EwProgram* outProg = nullptr)
{
    EwProgram buf, &p = outProg ? *outProg : buf;
    size_t ninputs = inps.size();
    AutoBuffer<int> depths(ninputs);
    for (size_t i = 0; i < ninputs; i++)
        depths[i] = inps[i].depth();
    compile(g, p, depths.data(), ninputs);
    std::vector<Mat> outs(p.noutputs);
    p.exec(inps.data(), outs.data());
    return outs;
}

// addWeighted(a,alpha,b,beta,gamma) = a*alpha + b*beta + gamma, built as a graph and compiled
// (the temp buffers that the previous test wired by hand are now allocated automatically).
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

        EwGraph g;
        int ia = g.input(0), ib = g.input(1);
        int t0 = g.binary(OP_MUL, ia, g.constant(Scalar(alpha)));
        int t1 = g.binary(OP_MUL, ib, g.constant(Scalar(beta)));
        int t2 = g.binary(OP_ADD, t0, t1);
        g.output(g.binary(OP_ADD, t2, g.constant(Scalar(gamma))));

        EwProgram prog;
        std::vector<Mat> out = run(g, { a, b }, &prog);

        Mat exp; cv::addWeighted(a, alpha, b, beta, gamma, exp);
        ASSERT_EQ(out[0].type(), exp.type());
        EXPECT_LE(cvtest::norm(out[0], exp, NORM_INF), 1e-3) << "cn=" << cn;
    }
}

// Mixed integer types: out = saturate_u8( saturate_u8(a*2.5) + b ), a,b are u8.
// The compiler must insert u8->f32 input casts and f32->u8 result casts around each op.
TEST(Core_EW_Compile, mixed_u8_inserts_casts)
{
    int H = 16, W = 24;
    Mat a(H, W, CV_8U), b(H, W, CV_8U);
    theRNG().fill(a, RNG::UNIFORM, 0, 60);
    theRNG().fill(b, RNG::UNIFORM, 0, 60);

    EwGraph g;
    int ia = g.input(0), ib = g.input(1);
    int mul = g.binary(OP_MUL, ia, g.constant(Scalar(2.5)));   // -> u8 (natural)
    g.output(g.binary(OP_ADD, mul, ib));                       // -> u8

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

    EwGraph g;
    int ia = g.input(0), ib = g.input(1);
    g.output(g.binary(OP_ADD, ia, ib));
    g.output(g.binary(OP_SUB, ia, ib));

    std::vector<Mat> out = run(g, { a, b });
    ASSERT_EQ(out.size(), 2u);

    Mat eadd, esub; cv::add(a, b, eadd); cv::subtract(a, b, esub);
    EXPECT_LE(cvtest::norm(out[0], eadd, NORM_INF), 1e-3) << "sum";
    EXPECT_LE(cvtest::norm(out[1], esub, NORM_INF), 1e-3) << "diff";
}

// Liveness: a linear chain of temps with disjoint lifetimes must share physical buffers.
// out = (((a+1)+1)+1)+1  == a+4 : 3 temps but only 2 buffers needed.
TEST(Core_EW_Compile, temp_buffer_reuse)
{
    int H = 10, W = 13;
    Mat a(H, W, CV_32F);
    theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);

    EwGraph g;
    int x = g.input(0);
    for (int k = 0; k < 4; k++)
        x = g.binary(OP_ADD, x, g.constant(Scalar(1.0)));
    g.output(x);

    EwProgram prog;
    std::vector<Mat> out = run(g, { a }, &prog);

    EXPECT_EQ(prog.ntemps, 3);       // last add writes straight to the output slot
    EXPECT_EQ(prog.nbuffers, 2);     // disjoint lifetimes => buffer reuse

    Mat exp; cv::add(a, Scalar(4.0), exp);
    EXPECT_LE(cvtest::norm(out[0], exp, NORM_INF), 1e-4);
}

}} // namespace
