// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Randomized property-based accuracy tests for the broadcasting element-wise ops, exercised through
// the PUBLIC cv:: entry points (add/subtract/multiply/divide/min/max/absdiff/compare with a mask,
// dtype, in-place aliasing and mixed input types). For each caseidx a deterministic splitmix64 seed
// picks random depths and broadcast-compatible shapes (ndims<=4, total<=100000) and every axis of each
// operand independently keeps its size or drops to 1, so full/row/col/channel broadcast are all hit.
// The reference decomposes each op into per-channel cv::broadcast + convertTo + the same op on aligned
// same-shape single-channel arrays. The module-internal header is included only for the promotion rule
// (promoteArith / absdiffResultDepth) the reference needs to predict each op's auto result depth.

#include "test_precomp.hpp"
#include "../src/arithm_expr.hpp"

namespace opencv_test { namespace {

using namespace cv::ew;

static inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static const uint64_t kSuiteSalt = 0x9ADD0CA57ULL;
static const int kNumCases = 1000;
static const int kMaxElems = 100000;

// engine-supported depths
static const int kDepths[] = { CV_8U, CV_8S, CV_16U, CV_16S, CV_32U, CV_32S,
                               CV_64U, CV_64S, CV_16F, CV_16BF, CV_32F, CV_64F };
static int sampleDepth(RNG& rng) { return kDepths[rng.uniform(0, (int)(sizeof(kDepths)/sizeof(kDepths[0])))]; }

static bool isFloat(int d) { return d==CV_16F || d==CV_16BF || d==CV_32F || d==CV_64F; }

// numpy-ish promotion (must mirror ew_exec.cpp's promote2 so the reference adds the same values)
static int promote2(int a, int b)
{
    if (a == b) return a;

    constexpr unsigned lbits = 3, lmask = (1u << lbits) - 1u;
    const uint64_t typelut = (uint64_t)((0ULL << CV_8U*lbits) | (0ULL << CV_8S*lbits) |
                                        (1ULL << CV_16U*lbits) | (1ULL << CV_16S*lbits) |
                                        (2ULL << CV_32U*lbits) | (2ULL << CV_32S*lbits) |
                                        (3ULL << CV_16F*lbits) | (3ULL << CV_16BF*lbits) |
                                        (3ULL << CV_32F*lbits) | (4ULL << CV_64F*lbits) |
                                        (4ULL << CV_64S*lbits) | (4ULL << CV_64U*lbits));
    unsigned pr_a = unsigned((typelut >> (a*lbits)) & lmask);
    unsigned pr_b = unsigned((typelut >> (b*lbits)) & lmask);
    unsigned max_pr = std::max(pr_a, pr_b);
    constexpr unsigned dbits = CV_CN_SHIFT, dmask = (1u << dbits) - 1u;
    const unsigned ctypelut = ((CV_16S << 0*dbits) | (CV_32S << 1*dbits) | (CV_64S << 2*dbits) |
                            (CV_32F << 3*dbits) | (CV_64F << 4*dbits));
    return int((ctypelut >> (max_pr*dbits)) & dmask);
}

// data range per depth: 8/16-bit wide enough to exercise saturation; 32/64-bit kept modest.
static void depthRange(int d, double& lo, double& hi)
{
    switch (d)
    {
    case CV_8U:  lo = 0;    hi = 255;   break;
    case CV_8S:  lo = -128; hi = 127;   break;
    case CV_16U: lo = 0;    hi = 65535; break;
    case CV_16S: lo = -32768; hi = 32767; break;
    default:     lo = -1000; hi = 1000; break;   // 32/64-bit ints and floats: no overflow
    }
}

static int sampleSize(RNG& rng, int lo, int hi)
{
    if (hi <= lo) return lo;
    if (rng.uniform(0.0, 1.0) < 0.35)
    {
        static const int cand[] = {1,2,3,4,7,8,15,16,17,31,32,33};
        int picks[16], n = 0;
        for (int c : cand) if (c >= lo && c <= hi) picks[n++] = c;
        picks[n++] = lo; picks[n++] = hi;
        return picks[rng.uniform(0, n)];
    }
    double v = std::exp(rng.uniform(std::log((double)lo), std::log((double)hi)));
    return std::min(hi, std::max(lo, cvRound(v)));
}

// a random shape (ndims in 1..4) with product <= kMaxElems
static std::vector<int> sampleShape(RNG& rng)
{
    int nd = rng.uniform(1, 5);
    std::vector<int> s(nd);
    long long prod = 1;
    for (int d = 0; d < nd; d++)
    {
        int hi = (int)std::min<long long>(512, std::max<long long>(1, kMaxElems / prod));
        s[d] = sampleSize(rng, 1, hi);
        prod *= s[d];
    }
    return s;
}

// build a random Mat of the given depth & shape, filled via a CV_64F master (randUni can't fill
// 16f/16bf/32u/64u/64s directly), values in the per-depth range. ~1/3 of the time the result is
// a NON-contiguous sub-array: the parent is padded by 1..2 on each edge of each axis and we
// return the inner view (gapped outer steps), to exercise the engine's non-continuous path.
static Mat makeRandom(RNG& rng, const std::vector<int>& shape, int cn, int depth,
                      double rlo = 1, double rhi = 0)
{
    const bool crop = rng.uniform(0, 3) == 0;
    const int nd = (int)shape.size();
    std::vector<int> pad(nd);
    std::vector<Range> ranges(nd);
    for (int d = 0; d < nd; d++)
    {
        int lo = crop ? rng.uniform(1, 3) : 0;   // 1..2
        int hi = crop ? rng.uniform(1, 3) : 0;
        pad[d] = shape[d] + lo + hi;
        ranges[d] = Range(lo, lo + shape[d]);
    }
    Mat m64(nd, pad.data(), CV_MAKETYPE(CV_64F, cn));
    double lo = rlo, hi = rhi;
    if (rlo > rhi) depthRange(depth, lo, hi);     // rlo>rhi (default) => per-depth range
    cvtest::randUni(rng, m64, Scalar::all(lo), Scalar::all(hi));
    Mat big; m64.convertTo(big, CV_MAKETYPE(depth, cn));
    return big(ranges);                          // full range when !crop => contiguous
}

static std::string shapeStr(const std::vector<int>& s)
{
    std::string r = "[";
    for (size_t i = 0; i < s.size(); i++) r += (i ? "x" : "") + std::to_string(s[i]);
    return r + "]";
}

// per-output-depth tolerance. `floatPath` = a float was involved on the way to an integer
// output, so a final float->int rounding tie may differ from cv:: by 1 (benign).
static void checkClose(const Mat& got, const Mat& ref, int rdepth, bool floatPath, const char* what)
{
    ASSERT_EQ(got.dims, ref.dims) << what;
    ASSERT_EQ(got.type(), ref.type()) << what;
    double n = cvtest::norm(got, ref, NORM_INF);
    if (!isFloat(rdepth))
    {
        EXPECT_LE(n, floatPath ? 1.0 : 0.0) << what;   // integer output: exact, or ±1 via float
    }
    else
    {
        double scale = std::max(1.0, cvtest::norm(ref, NORM_INF));
        double rel = rdepth==CV_16BF ? 1e-2 : rdepth==CV_16F ? 2e-3 : 1e-5;
        EXPECT_LE(n, rel*scale) << what << " (n=" << n << " scale=" << scale << ")";
    }
}

// ------------------------------------------------------------------------------- add / sub
// Parameterized on (op, caseidx): op 0 = ADD, 1 = SUB. The two ops share the same per-case data
// (seed depends only on caseidx), so they run on identical inputs.
class EW_Extensive_BinOp : public ::testing::TestWithParam<std::tuple<int,int>> {};

TEST_P(EW_Extensive_BinOp, accuracy)
{
    const int opSel = std::get<0>(GetParam());
    const int caseidx = std::get<1>(GetParam());
    const TOp op = opSel ? OP_SUB : OP_ADD;
    const char* opStr = opSel ? "sub" : "add";
    RNG rng(mix64(kSuiteSalt ^ (uint64_t)caseidx));

    std::vector<int> shape = sampleShape(rng);
    const int da = sampleDepth(rng), db = sampleDepth(rng);

    // channels: pick a base count (biased toward 1); each operand keeps it or drops to 1, so we
    // get C1+C1 (fold), Cn+Cn (fold) and the Cn+C1 / C1+Cn channel-broadcast (CH_DIM) mix.
    static const int cncand[] = { 1, 1, 2, 3, 4 };
    const int rcn = cncand[rng.uniform(0, 5)];
    const int cn_a = rng.uniform(0, 2) ? rcn : 1;
    const int cn_b = rng.uniform(0, 2) ? rcn : 1;
    const int ocn = std::max(cn_a, cn_b);

    // each operand independently keeps or broadcasts (->1) every axis
    std::vector<int> sa(shape.size()), sb(shape.size()), res(shape.size());
    for (size_t d = 0; d < shape.size(); d++)
    {
        sa[d] = rng.uniform(0, 2) ? shape[d] : 1;
        sb[d] = rng.uniform(0, 2) ? shape[d] : 1;
        res[d] = std::max(sa[d], sb[d]);
    }

    // in-place: ~1/3 of cases attempt it. When an input is spatially & channel "full" (its shape ==
    // the output shape res, channels == ocn), alias the output onto it and force Tr to that input's
    // depth so the buffer is truly reused. Exercises the executor's in-place handling AND the
    // kernels' dst==src aliasing (the halide-tail backoff). Otherwise Tr is a free random depth.
    int aliasIn = -1;
    if (rng.uniform(0, 3) == 0)
    {
        if      (sa == res && cn_a == ocn) aliasIn = 0;
        else if (sb == res && cn_b == ocn) aliasIn = 1;
    }
    const bool inplace = aliasIn >= 0;
    const int Tr = aliasIn == 0 ? da : aliasIn == 1 ? db : sampleDepth(rng);

    SCOPED_TRACE(cv::format("%s caseidx=%d da=%s db=%s Tr=%s a=%sC%d b=%sC%d inplace=%d",
                            opStr, caseidx, depthToString(da), depthToString(db), depthToString(Tr), shapeStr(sa).c_str(), cn_a,
                            shapeStr(sb).c_str(), cn_b, inplace ? aliasIn : -1));

    Mat a = makeRandom(rng, sa, cn_a, da), b = makeRandom(rng, sb, cn_b, db);

    // reference FIRST (an in-place exec may overwrite an input): per output channel pick a's/b's
    // channel (C1->Cn broadcast), spatial-broadcast, cast to common type C, op(Tr), then merge.
    int C = (da == db) ? da : promote2(da, db);
    std::vector<Mat> ach, bch; cv::split(a, ach); cv::split(b, bch);
    std::vector<Mat> refch(ocn);
    for (int c = 0; c < ocn; c++)
    {
        Mat apC, bpC;
        cvtest::convert(ach[cn_a == 1 ? 0 : c], apC, C);
        cvtest::convert(bch[cn_b == 1 ? 0 : c], bpC, C);
        Mat aB, bB; cv::broadcast(apC, res, aB); cv::broadcast(bpC, res, bB);
        cvtest::add(aB, 1, bB, op == OP_SUB ? -1 : 1, Scalar(), refch[c], Tr);
    }
    Mat ref; cv::merge(refch, ref);

    // public op: dst aliases input #aliasIn for the in-place case, else a fresh Mat.
    Mat outOwn;
    Mat& out = inplace ? (aliasIn == 0 ? a : b) : outOwn;
    if (op == OP_SUB) cv::subtract(a, b, out, noArray(), Tr);
    else              cv::add     (a, b, out, noArray(), Tr);

    checkClose(out, ref, Tr, isFloat(da) || isFloat(db), opStr);
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_BinOp,
    testing::Combine(testing::Values(0, 1), testing::Range(0, kNumCases)),
    [](const testing::TestParamInfo<std::tuple<int,int>>& ti) {
        return cv::format("%s_case%04d", std::get<0>(ti.param) ? "sub" : "add",
                          std::get<1>(ti.param));
    });

// ------------------------------------------------------------------- min / max / absdiff
// op 0 = MIN, 1 = MAX, 2 = ABSDIFF: operands promoted to a common type C; MIN/MAX result C, ABSDIFF
// result Cr = absdiffResultDepth(C) (unsigned same width for signed ints, since |a-b| can hit 2^w-1).
// cv::min/max/absdiff auto-promote mixed input types (no dtype arg), so this exercises the engine's
// promotion + cast insertion for a fresh family of ops through the public entry points.
class EW_Extensive_MinMax : public ::testing::TestWithParam<std::tuple<int,int>> {};

TEST_P(EW_Extensive_MinMax, accuracy)
{
    const int opSel = std::get<0>(GetParam());
    const int caseidx = std::get<1>(GetParam());
    const TOp op = opSel == 0 ? OP_MIN : opSel == 1 ? OP_MAX : OP_ABSDIFF;
    const char* opStr = opSel == 0 ? "min" : opSel == 1 ? "max" : "absdiff";
    RNG rng(mix64(kSuiteSalt ^ (uint64_t)(caseidx * 3 + opSel)));   // distinct stream per op

    std::vector<int> shape = sampleShape(rng);
    const int da = sampleDepth(rng), db = sampleDepth(rng);
    const int C = promoteArith(da, db);   // common compute/result type (auto), shared with the ref.
    const int Cr = C;   // min/max/absdiff all resolve to C (absdiff saturates its wide |a-b| back to C)

    static const int cncand[] = { 1, 1, 2, 3, 4 };
    const int rcn = cncand[rng.uniform(0, 5)];
    const int cn_a = rng.uniform(0, 2) ? rcn : 1;
    const int cn_b = rng.uniform(0, 2) ? rcn : 1;
    const int ocn = std::max(cn_a, cn_b);

    std::vector<int> sa(shape.size()), sb(shape.size()), res(shape.size());
    for (size_t d = 0; d < shape.size(); d++)
    {
        sa[d] = rng.uniform(0, 2) ? shape[d] : 1;
        sb[d] = rng.uniform(0, 2) ? shape[d] : 1;
        res[d] = std::max(sa[d], sb[d]);
    }

    SCOPED_TRACE(cv::format("%s caseidx=%d da=%d db=%d C=%d a=%sC%d b=%sC%d",
                            opStr, caseidx, da, db, C, shapeStr(sa).c_str(), cn_a,
                            shapeStr(sb).c_str(), cn_b));

    Mat a = makeRandom(rng, sa, cn_a, da), b = makeRandom(rng, sb, cn_b, db);

    // reference: per output channel pick a's/b's channel (C1->Cn), spatial-broadcast, cast to C, op.
    std::vector<Mat> ach, bch; cv::split(a, ach); cv::split(b, bch);
    std::vector<Mat> refch(ocn);
    for (int c = 0; c < ocn; c++)
    {
        Mat apC, bpC;
        cvtest::convert(ach[cn_a == 1 ? 0 : c], apC, C);
        cvtest::convert(bch[cn_b == 1 ? 0 : c], bpC, C);
        Mat aB, bB; cv::broadcast(apC, res, aB); cv::broadcast(bpC, res, bB);
        if (op == OP_MIN)      cvtest::min(aB, bB, refch[c]);
        else if (op == OP_MAX) cvtest::max(aB, bB, refch[c]);
        else                   cvtest::add(aB, 1, bB, -1, Scalar(), refch[c], Cr, /*calcAbs=*/true);
    }
    Mat ref; cv::merge(refch, ref);

    // public op: min/max/absdiff auto-promote mixed input types to C (= promoteArith(da,db)) and
    // broadcast, exactly like the reference; absdiff's result is the unsigned same-width Cr.
    Mat out;
    if (op == OP_MIN)      cv::min(a, b, out);
    else if (op == OP_MAX) cv::max(a, b, out);
    else                   cv::absdiff(a, b, out);

    checkClose(out, ref, Cr, isFloat(da) || isFloat(db), opStr);
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_MinMax,
    testing::Combine(testing::Values(0, 1, 2), testing::Range(0, kNumCases)),
    [](const testing::TestParamInfo<std::tuple<int,int>>& ti) {
        const int o = std::get<0>(ti.param);
        return cv::format("%s_case%04d", o == 0 ? "min" : o == 1 ? "max" : "absdiff",
                          std::get<1>(ti.param));
    });

// ----------------------------------------------------------------------------------- compare
// op 0 = CMP_EQ, 1 = CMP_GT: operands promoted to a common type C, result a u8 mask. Exercises
// emitBinary's compare branch (result forced to u8) and the optional mask value (0/255 default, or
// 0/1 set through TKernel::flags). Inputs are drawn from a small shared range so equality fires.
class EW_Extensive_Compare : public ::testing::TestWithParam<std::tuple<int,int>> {};

TEST_P(EW_Extensive_Compare, accuracy)
{
    const int opSel = std::get<0>(GetParam());   // 0 = EQ, 1 = GT
    const int caseidx = std::get<1>(GetParam());
    const int cmpop = opSel == 0 ? cv::CMP_EQ : cv::CMP_GT;
    const char* opStr = opSel == 0 ? "cmpEQ" : "cmpGT";
    RNG rng(mix64(kSuiteSalt ^ 0xC0FFEEULL ^ (uint64_t)(caseidx * 2 + opSel)));

    std::vector<int> shape = sampleShape(rng);
    const int da = sampleDepth(rng), db = sampleDepth(rng);
    const int C = (da == db) ? da : promote2(da, db);

    static const int cncand[] = { 1, 1, 2, 3, 4 };
    const int rcn = cncand[rng.uniform(0, 5)];
    const int cn_a = rng.uniform(0, 2) ? rcn : 1;
    const int cn_b = rng.uniform(0, 2) ? rcn : 1;
    const int ocn = std::max(cn_a, cn_b);

    std::vector<int> sa(shape.size()), sb(shape.size()), res(shape.size());
    for (size_t d = 0; d < shape.size(); d++)
    {
        sa[d] = rng.uniform(0, 2) ? shape[d] : 1;
        sb[d] = rng.uniform(0, 2) ? shape[d] : 1;
        res[d] = std::max(sa[d], sb[d]);
    }

    SCOPED_TRACE(cv::format("%s caseidx=%d da=%d db=%d C=%d a=%sC%d b=%sC%d",
                            opStr, caseidx, da, db, C, shapeStr(sa).c_str(), cn_a,
                            shapeStr(sb).c_str(), cn_b));

    // small shared range [0,12] (well within every depth) so EQ is hit on a healthy fraction
    Mat a = makeRandom(rng, sa, cn_a, da, 0, 12), b = makeRandom(rng, sb, cn_b, db, 0, 12);

    // reference: per channel, cast to C, compare in f64 (exact for these ranges) -> 0/255 mask
    std::vector<Mat> ach, bch; cv::split(a, ach); cv::split(b, bch);
    std::vector<Mat> refch(ocn);
    for (int c = 0; c < ocn; c++)
    {
        Mat apC, bpC;
        cvtest::convert(ach[cn_a == 1 ? 0 : c], apC, C);
        cvtest::convert(bch[cn_b == 1 ? 0 : c], bpC, C);
        Mat aB, bB; cv::broadcast(apC, res, aB); cv::broadcast(bpC, res, bB);
        Mat af, bf; cvtest::convert(aB, af, CV_64F); cvtest::convert(bB, bf, CV_64F);
        cvtest::compare(af, bf, refch[c], cmpop);      // 0 / 255
    }
    Mat ref255; cv::merge(refch, ref255);

    // public op: cv::compare auto-promotes mixed input types to the common type, broadcasts, and
    // yields a u8 0/255 mask per channel. (The engine's optional 0/1 mask is not exposed here.)
    Mat out; cv::compare(a, b, out, cmpop);
    ASSERT_EQ(out.type(), CV_8UC(ocn)) << opStr;
    EXPECT_EQ(0, cvtest::norm(out, ref255, NORM_INF)) << opStr;
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_Compare,
    testing::Combine(testing::Values(0, 1), testing::Range(0, kNumCases)),
    [](const testing::TestParamInfo<std::tuple<int,int>>& ti) {
        return cv::format("%s_case%04d", std::get<0>(ti.param) == 0 ? "cmpEQ" : "cmpGT",
                          std::get<1>(ti.param));
    });

// ------------------------------------------------------------------------------------- mul / div
// Parameterized on (op, caseidx): op 0 = MUL, 1 = DIV. Both compute in the float work type (float
// for <=16-bit, double for 32/64-bit), matching cv::multiply/divide; integer divide-by-zero => 0.
class EW_Extensive_MulDiv : public ::testing::TestWithParam<std::tuple<int,int>> {};

TEST_P(EW_Extensive_MulDiv, accuracy)
{
    const int opSel = std::get<0>(GetParam());
    const int caseidx = std::get<1>(GetParam());
    const TOp op = opSel ? OP_DIV : OP_MUL;
    const char* opStr = opSel ? "div" : "mul";
    RNG rng(mix64(kSuiteSalt ^ 0x3DD17ULL ^ (uint64_t)caseidx));

    std::vector<int> shape = sampleShape(rng);
    const int da = sampleDepth(rng), db = sampleDepth(rng);

    static const int cncand[] = { 1, 1, 2, 3, 4 };
    const int rcn = cncand[rng.uniform(0, 5)];
    const int cn_a = rng.uniform(0, 2) ? rcn : 1;
    const int cn_b = rng.uniform(0, 2) ? rcn : 1;
    const int ocn = std::max(cn_a, cn_b);

    std::vector<int> sa(shape.size()), sb(shape.size()), res(shape.size());
    for (size_t d = 0; d < shape.size(); d++)
    {
        sa[d] = rng.uniform(0, 2) ? shape[d] : 1;
        sb[d] = rng.uniform(0, 2) ? shape[d] : 1;
        res[d] = std::max(sa[d], sb[d]);
    }

    int aliasIn = -1;
    if (rng.uniform(0, 3) == 0)
    {
        if      (sa == res && cn_a == ocn) aliasIn = 0;
        else if (sb == res && cn_b == ocn) aliasIn = 1;
    }
    const bool inplace = aliasIn >= 0;
    const int Tr = aliasIn == 0 ? da : aliasIn == 1 ? db : sampleDepth(rng);

    // half the cases use a non-unit scale (mul: a*b*scale, div: a*scale/b), like cv::multiply/
    // divide. Kept in [1/256, 2] so it can shrink (e.g. 1/255) or modestly amplify without pushing
    // a product/quotient past the integer-output range (which would be float->int UB on both sides).
    double scale = 1.0;
    if (rng.uniform(0, 2)) scale = rng.uniform(1.0/256, 2.0);

    SCOPED_TRACE(cv::format("%s caseidx=%d da=%d db=%d Tr=%d a=%sC%d b=%sC%d inplace=%d scale=%.4f",
                            opStr, caseidx, da, db, Tr, shapeStr(sa).c_str(), cn_a,
                            shapeStr(sb).c_str(), cn_b, inplace ? aliasIn : -1, scale));

    // modest magnitudes: mul/div compute in a float work type, so a product/quotient that overflows
    // the integer output's range hits float->int UB (cv::multiply is UB there too). [-1000,1000]
    // keeps products <= 1e6 (no overflow), while still exercising saturation for small outputs.
    Mat a = makeRandom(rng, sa, cn_a, da, -1000, 1000), b = makeRandom(rng, sb, cn_b, db, -1000, 1000);

    const bool bothInt = !isFloat(da) && !isFloat(db);
    // Integer divide-by-zero is well-defined (=> 0) and IS exercised. Float-involved divide-by-zero
    // is UB (a/0 -> inf -> int), so avoid it here: make the divisor (b) nonzero for the float path.
    if (op == OP_DIV && !bothInt)
    {
        Mat b64; b.convertTo(b64, CV_64F);
        b64.setTo(1.0, b64 == 0.0);
        b64.convertTo(b, db);
    }

    // reference FIRST (in-place may overwrite an input): mirror the engine's spec - cast both to the
    // float work type Wf (float for <=16-bit common type, double for 32/64-bit), op in Wf, then cast
    // to Tr (same final cast the engine uses). For both-integer div, guard divide-by-zero -> 0.
    const int C = (da == db) ? da : promote2(da, db);
    const bool wide = (C==CV_32U || C==CV_32S || C==CV_64U || C==CV_64S || C==CV_64F);
    const int Wf = wide ? CV_64F : CV_32F;
    std::vector<Mat> ach, bch; cv::split(a, ach); cv::split(b, bch);
    std::vector<Mat> refch(ocn);
    for (int c = 0; c < ocn; c++)
    {
        Mat aWf, bWf;
        cvtest::convert(ach[cn_a == 1 ? 0 : c], aWf, Wf);
        cvtest::convert(bch[cn_b == 1 ? 0 : c], bWf, Wf);
        Mat aB, bB; cv::broadcast(aWf, res, aB); cv::broadcast(bWf, res, bB);
        Mat q;
        if (op == OP_DIV) { cvtest::divide(aB, bB, q, scale); if (bothInt) q.setTo(0, bB == 0); }
        else              cvtest::multiply(aB, bB, q, scale);
        cvtest::convert(q, refch[c], Tr);
    }
    Mat ref; cv::merge(refch, ref);

    // public op: dst aliases input #aliasIn for the in-place case, else a fresh Mat.
    Mat outOwn;
    Mat& out = inplace ? (aliasIn == 0 ? a : b) : outOwn;
    if (op == OP_DIV) cv::divide  (a, b, out, scale, Tr);
    else              cv::multiply(a, b, out, scale, Tr);

    checkClose(out, ref, Tr, true, opStr);   // float work => integer output may differ by <=1
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_MulDiv,
    testing::Combine(testing::Values(0, 1), testing::Range(0, kNumCases)),
    [](const testing::TestParamInfo<std::tuple<int,int>>& ti) {
        return cv::format("%s_case%04d", std::get<0>(ti.param) ? "div" : "mul",
                          std::get<1>(ti.param));
    });

// ------------------------------------------------------------------------------- masked add / sub
// add/sub with a write-mask. The data inputs share the output shape (a, b, out all `shape`-spatial,
// cn channels); the mask is single-channel, the output spatial shape, type bool/u8/s8. The output
// PRE-EXISTS (filled with random content): copyMask overwrites only the masked subset and leaves
// the rest unchanged (dst = mask ? op : dst). cn==1 exercises the per-element mask (CH_FOLD); cn>1
// the channel-axis broadcast (CH_DIM, mask stepx 0 => a whole n-channel row copied under one test).
class EW_Extensive_Mask : public ::testing::TestWithParam<std::tuple<int,int>> {};

TEST_P(EW_Extensive_Mask, accuracy)
{
    const int opSel = std::get<0>(GetParam());
    const int caseidx = std::get<1>(GetParam());
    const TOp op = opSel ? OP_SUB : OP_ADD;
    const char* opStr = opSel ? "sub" : "add";
    RNG rng(mix64(kSuiteSalt ^ 0x5A5C0DEULL ^ (uint64_t)caseidx));

    std::vector<int> shape = sampleShape(rng);
    const int da = sampleDepth(rng), db = sampleDepth(rng), Tr = sampleDepth(rng);

    static const int cncand[] = { 1, 1, 2, 3, 4 };
    const int cn = cncand[rng.uniform(0, 5)];

    static const int maskDepths[] = { CV_8U, CV_8S, CV_Bool };
    const int md = maskDepths[rng.uniform(0, 3)];

    SCOPED_TRACE(cv::format("%s caseidx=%d da=%d db=%d Tr=%d cn=%d md=%d shape=%s",
                            opStr, caseidx, da, db, Tr, cn, md, shapeStr(shape).c_str()));

    Mat a = makeRandom(rng, shape, cn, da), b = makeRandom(rng, shape, cn, db);

    // mask: single-channel, output spatial shape, ~half zero. Build a u8 0/1 master, convert it to
    // the chosen mask depth for the engine; the u8 master drives the reference copyTo.
    const int nd = (int)shape.size();
    Mat m8(nd, shape.data(), CV_8U);
    cvtest::randUni(rng, m8, Scalar::all(0), Scalar::all(2));   // 0 or 1
    Mat mask; m8.convertTo(mask, md);

    // pre-existing output content (preserved where mask==0): dst = mask ? op : dst.
    Mat init = makeRandom(rng, shape, cn, Tr).clone();   // contiguous Tr-typed dst

    // reference: full op per channel (cast to common type C, op to Tr), merge, then overwrite the
    // masked subset of `init` (the rest stays as the pre-existing content).
    int C = (da == db) ? da : promote2(da, db);
    std::vector<Mat> ach, bch; cv::split(a, ach); cv::split(b, bch);
    std::vector<Mat> refch(cn);
    for (int c = 0; c < cn; c++)
    {
        Mat apC, bpC; cvtest::convert(ach[c], apC, C); cvtest::convert(bch[c], bpC, C);
        cvtest::add(apC, 1, bpC, op == OP_SUB ? -1 : 1, Scalar(), refch[c], Tr);
    }
    Mat refFull; cv::merge(refch, refFull);
    Mat ref = init.clone();
    cvtest::copy(refFull, ref, m8);

    // public op with a write-mask: the pre-existing output is preserved where mask==0.
    Mat out = init.clone();
    if (op == OP_SUB) cv::subtract(a, b, out, mask, Tr);
    else              cv::add     (a, b, out, mask, Tr);

    checkClose(out, ref, Tr, isFloat(da) || isFloat(db), opStr);
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_Mask,
    testing::Combine(testing::Values(0, 1), testing::Range(0, kNumCases)),
    [](const testing::TestParamInfo<std::tuple<int,int>>& ti) {
        return cv::format("%s_case%04d", std::get<0>(ti.param) ? "sub" : "add",
                          std::get<1>(ti.param));
    });

// NOTE: a standalone cast group was dropped - the engine cast == cv::convertTo (comparing them would
// be a tautology), and mixed-type casts are already exercised inside the add/sub/mul/div groups above.

}} // namespace
