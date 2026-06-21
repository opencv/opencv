// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Randomized property-based accuracy tests for the element-wise engine (OP_ADD, OP_CAST).
// For each caseidx a deterministic splitmix64 seed picks random depths and broadcast-compatible
// shapes (ndims<=4, total elements<=100000), runs the engine, and compares with a cv:: reference
// (cv::broadcast + cv::add / convertTo). See ~/Downloads/extensive_test_harness_spec.md.

#include "../test_precomp.hpp"
#include "ew_exec.hpp"

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
static Mat makeRandom(RNG& rng, const std::vector<int>& shape, int cn, int depth)
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
    double lo, hi; depthRange(depth, lo, hi);
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
    const ElemwiseOp op = opSel ? OP_SUB : OP_ADD;
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

    SCOPED_TRACE(cv::format("%s caseidx=%d da=%d db=%d Tr=%d a=%sC%d b=%sC%d inplace=%d",
                            opStr, caseidx, da, db, Tr, shapeStr(sa).c_str(), cn_a,
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
        ach[cn_a == 1 ? 0 : c].convertTo(apC, C);
        bch[cn_b == 1 ? 0 : c].convertTo(bpC, C);
        Mat aB, bB; cv::broadcast(apC, res, aB); cv::broadcast(bpC, res, bB);
        if (op == OP_SUB) cv::subtract(aB, bB, refch[c], noArray(), Tr);
        else              cv::add     (aB, bB, refch[c], noArray(), Tr);
    }
    Mat ref; cv::merge(refch, ref);

    // engine (output aliases input #aliasIn for the in-place case, else a fresh Mat)
    EwProgram p;
    makeBinaryArithProgram(p, op, da, db, Tr);
    Mat inps[] = {a, b}, outOwn;
    Mat* outPtr = inplace ? &inps[aliasIn] : &outOwn;
    p.exec(inps, outPtr);

    checkClose(*outPtr, ref, Tr, isFloat(da) || isFloat(db), opStr);
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_BinOp,
    testing::Combine(testing::Values(0, 1), testing::Range(0, kNumCases)),
    [](const testing::TestParamInfo<std::tuple<int,int>>& info) {
        return cv::format("%s_case%04d", std::get<0>(info.param) ? "sub" : "add",
                          std::get<1>(info.param));
    });

// ---------------------------------------------------------------------------------- cast
class EW_Extensive_Cast : public ::testing::TestWithParam<int> {};

TEST_P(EW_Extensive_Cast, accuracy)
{
    const int caseidx = GetParam();
    RNG rng(mix64(kSuiteSalt ^ 0xCA57ULL ^ (uint64_t)caseidx));

    std::vector<int> shape = sampleShape(rng);
    const int sd = sampleDepth(rng), dd = sampleDepth(rng);

    SCOPED_TRACE(cv::format("caseidx=%d sd=%d dd=%d shape=%s", caseidx, sd, dd, shapeStr(shape).c_str()));

    Mat a = makeRandom(rng, shape, 1, sd), out;

    EwProgram p = makeUnaryProgram(OP_CAST, sd, dd);
    p.exec(&a, &out);

    Mat ref; a.convertTo(ref, dd);
    checkClose(out, ref, dd, isFloat(sd), "cast");
}

INSTANTIATE_TEST_CASE_P(Core_EW, EW_Extensive_Cast, testing::Range(0, kNumCases),
    [](const testing::TestParamInfo<int>& info){ return cv::format("case%04d", info.param); });

}} // namespace
