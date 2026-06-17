// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// End-to-end test of the vertical slice: single-op programs (ADD/SUB/MUL/DIV/POW f32 and
// CAST) executed through the new element-wise engine, checked against the classic cv:: ops.

#include "../test_precomp.hpp"
#include "ew_exec.hpp"

namespace opencv_test { namespace {

using namespace cv::ew;

static Mat runBinary(ElemwiseOp op, const Mat& a, const Mat& b, int rdepth)
{
    EwProgram p = makeBinaryProgram(op, a.depth(), b.depth(), rdepth);
    std::vector<Mat> in, out;
    in.push_back(a); in.push_back(b);
    exec(p, in, out);
    return out[0];
}

static Mat runUnary(ElemwiseOp op, const Mat& a, int rdepth)
{
    EwProgram p = makeUnaryProgram(op, a.depth(), rdepth);
    std::vector<Mat> in, out;
    in.push_back(a);
    exec(p, in, out);
    return out[0];
}

static void cvRef(ElemwiseOp op, const Mat& a, const Mat& b, Mat& dst)
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
    const ElemwiseOp ops[] = { OP_ADD, OP_SUB, OP_MUL, OP_DIV };
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
        Mat got = runUnary(OP_CAST, a, CV_8U);
        Mat exp; a.convertTo(exp, CV_8U);
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "f32->u8";
    }
    {
        Mat got = runUnary(OP_CAST, a, CV_32S);
        Mat exp; a.convertTo(exp, CV_32S);
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "f32->s32";
    }
    {
        Mat u; a.convertTo(u, CV_8U);
        Mat got = runUnary(OP_CAST, u, CV_32F);
        Mat exp; u.convertTo(exp, CV_32F);
        EXPECT_EQ(0, cvtest::norm(got, exp, NORM_INF)) << "u8->f32";
    }
}

// ---- compound expression: addWeighted(a,alpha,b,beta,gamma) = a*alpha + b*beta + gamma ----
// Hand-built multi-instruction program; exercises the temp-buffer + const-arg machinery
// before the Layer-2 compiler exists.
static Mat runAddWeighted(const Mat& a, double alpha, const Mat& b, double beta,
                          double gamma, int rdepth)
{
    EwProgram p;
    p.ninputs = 2; p.noutputs = 1; p.ntemps = 3; p.nbuffers = 3;
    p.bufferOfTemp.assign(3, 0);
    p.bufferOfTemp[0] = 0; p.bufferOfTemp[1] = 1; p.bufferOfTemp[2] = 2;
    p.arginfo.resize(10);

    auto setA = [&](int s, ArgKind k, int depth, int index)
    { p.arginfo[s].kind = k; p.arginfo[s].depth = depth; p.arginfo[s].index = index; };

    setA(0, ARG_NONE,   EW_DEPTH_NONE, -1);
    setA(1, ARG_INPUT,  a.depth(), 0);
    setA(2, ARG_INPUT,  b.depth(), 1);
    setA(3, ARG_CONST,  CV_32F, -1); p.arginfo[3].cval = Scalar(alpha);
    setA(4, ARG_CONST,  CV_32F, -1); p.arginfo[4].cval = Scalar(beta);
    setA(5, ARG_CONST,  CV_32F, -1); p.arginfo[5].cval = Scalar(gamma);
    setA(6, ARG_TEMP,   CV_32F, 0);
    setA(7, ARG_TEMP,   CV_32F, 1);
    setA(8, ARG_TEMP,   CV_32F, 2);
    setA(9, ARG_OUTPUT, rdepth, 0);

    auto add_insn = [&](ElemwiseOp op, int a0, int a1, int r)
    {
        EwInsn ins; ins.op = op; ins.arg0 = a0; ins.arg1 = a1; ins.arg2 = 0; ins.result = r;
        ins.fptr = getElemwiseFunc(op, p.arginfo[a0].depth, p.arginfo[a1].depth,
                                   EW_DEPTH_NONE, p.arginfo[r].depth);
        CV_Assert(ins.fptr != nullptr);
        p.prog.push_back(ins);
    };
    add_insn(OP_MUL, 1, 3, 6);   // t0 = a * alpha
    add_insn(OP_MUL, 2, 4, 7);   // t1 = b * beta
    add_insn(OP_ADD, 6, 7, 8);   // t2 = t0 + t1
    add_insn(OP_ADD, 8, 5, 9);   // out = t2 + gamma

    std::vector<Mat> in, out;
    in.push_back(a); in.push_back(b);
    exec(p, in, out);
    return out[0];
}

TEST(Core_EW_Slice, addweighted_compound)
{
    const int chans[] = { 1, 3 };
    double alpha = 2.5, beta = -1.5, gamma = 7.0;
    for (int ci = 0; ci < 2; ci++)
    {
        int H = 19, W = 23, cn = chans[ci];
        Mat a(H, W, CV_32FC(cn)), b(H, W, CV_32FC(cn));
        theRNG().fill(a, RNG::UNIFORM, 1.f, 10.f);
        theRNG().fill(b, RNG::UNIFORM, 1.f, 10.f);

        Mat got = runAddWeighted(a, alpha, b, beta, gamma, CV_32F);
        Mat exp; cv::addWeighted(a, alpha, b, beta, gamma, exp);

        ASSERT_EQ(got.size(), exp.size());
        ASSERT_EQ(got.type(), exp.type());
        EXPECT_LE(cvtest::norm(got, exp, NORM_INF), 1e-3) << "cn=" << cn;
    }
}

}} // namespace
