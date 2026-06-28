// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Ad-hoc perf comparison for the element-wise engine vs classic cv::add. Lives in
// opencv_test_core for now (move to opencv_perf_core later). Each (type-combo, size) is run
// 10..30 times; the minimum getTickCount() time is reported as the most stable metric.

#include "../test_precomp.hpp"
#include "ew_exec.hpp"
#include "ew_parser.hpp"
#include <iostream>
#include <iomanip>

namespace opencv_test { namespace {

using namespace cv::ew;

static Mat randMat(const std::vector<int>& shape, int cn, int depth, double lo, double hi)
{
    Mat m64((int)shape.size(), shape.data(), CV_MAKETYPE(CV_64F, cn));
    cvtest::randUni(theRNG(), m64, Scalar::all(lo), Scalar::all(hi));
    Mat m; m64.convertTo(m, CV_MAKETYPE(depth, cn));
    return m;
}

// single-channel 0/1 mask of the given spatial shape
static Mat randMask(const std::vector<int>& shape)
{
    Mat m((int)shape.size(), shape.data(), CV_8U);
    cvtest::randUni(theRNG(), m, Scalar::all(0), Scalar::all(2));
    return m;
}

// Min over `iters` trials of the per-call time in MICROSECONDS. Each trial runs f() `ninner`
// times inside one timed region and divides by ninner, so the timer's coarse resolution is
// amortized across many calls - essential for sub-microsecond operations.
template<typename F>
static double minUs(F&& f, int iters, int ninner)
{
    f();  // warmup (allocates reused output, warms caches)
    double best = DBL_MAX;
    for (int i = 0; i < iters; i++)
    {
        int64 t0 = getTickCount();
        for (int j = 0; j < ninner; j++) f();
        double us = (getTickCount() - t0) * 1e6 / getTickFrequency() / ninner;
        best = std::min(best, us);
    }
    return best;
}

struct Combo { int da, db, Tr; std::string name; };
struct Sz    { std::vector<int> shape; int cn; int ninner; const char* name; };

// Shared add/sub sweep over (type-combo x size). `masked` adds a single-channel write-mask: the
// engine builds the op-into-temp + copyMask program; the cv:: reference times cv::add/subtract with
// the mask (only for same-type combos - mixed-type + mask isn't compared). Engine correctness is
// always checked against a deterministic zero + copyTo(mask) reference.
static void perfBinOp(TOp op, const char* title, bool masked)
{
    const std::string opname = opName(op);
    const Combo combos[] = {
        { CV_8U,  CV_8U,  CV_8U,  opname + "(u8,  u8)->u8   " },
        { CV_16F, CV_16F, CV_16F, opname + "(f16, f16)->f16 " },
        { CV_32F, CV_32F, CV_32F, opname + "(f32, f32)->f32 " },
        { CV_8U,  CV_16F, CV_16F, opname + "(u8,  f16)->f16 " },
    };
    const Sz sizes[] = {
        { {10,10,10},   1, 5000, "10x10x10    " },
        { {165,121},    1, 2000, "165x121     " },
        { {1024,1024},  3,    4, "1024x1024x3 " },
    };

    std::cout << "\n[ew-perf] " << title << "  (min us per call over 30 trials)\n";
    std::cout << "  combo                size           engine     cv::op     speedup\n";
    std::cout << "  -----------------------------------------------------------------\n";

    for (const Combo& c : combos)
        for (const Sz& s : sizes)
        {
            Mat a = randMat(s.shape, s.cn, c.da, 0, 100);
            Mat b = randMat(s.shape, s.cn, c.db, 0, 100);
            // div: make the divisor nonzero (the 0..100 data includes 0 for integer b); float-div by
            // zero is UB, integer-div by zero is a separate (accuracy-tested) corner not timed here.
            if (op == OP_DIV) { Mat b64; b.convertTo(b64, CV_64F); b64.setTo(1.0, b64 == 0.0); b64.convertTo(b, c.db); }
            Mat mask = masked ? randMask(s.shape) : Mat();
            Mat init = masked ? randMat(s.shape, s.cn, c.Tr, 0, 50) : Mat();   // pre-existing dst
            Mat in2[] = {a, b}, in3[] = {a, b, mask}, out = masked ? init.clone() : Mat();
            Mat* inps = masked ? in3 : in2;
            const int mdepth = masked ? CV_8U : EW_DEPTH_NONE;
            // u8*u8 mul: exercise a realistic non-unit scale (1/255, the normalized-blend case) -
            // checks the specialized u8 mul branch still holds up when scale != 1.
            const double scale = (op == OP_MUL && c.da == CV_8U && c.db == CV_8U) ? 1.0/255 : 1.0;

            // Full per-call path (matches a future cv:: op): build the program every call.
            // (masked preserves the pre-filled `out` where mask==0, so repeated calls are idempotent.)
            double te = minUs([&]{ TExpr p; makeBinaryArithProgram(p, op, c.da, c.db, c.Tr, mdepth, scale);
                                   p.exec(inps, &out); }, 30, s.ninner);

            // engine correctness sanity: add/sub vs cv:: directly; mul/div vs a double reference (the
            // extensive test owns exactness, so a generous tolerance here just guards against garbage).
            const bool fp = (op == OP_MUL || op == OP_DIV);
            Mat ref;
            if (op == OP_ADD)      cv::add     (a, b, ref, noArray(), c.Tr);
            else if (op == OP_SUB) cv::subtract(a, b, ref, noArray(), c.Tr);
            else if (op == OP_MIN || op == OP_MAX || op == OP_ABSDIFF) {
                   Mat aT, bT; a.convertTo(aT, c.Tr); b.convertTo(bT, c.Tr);
                   if (op == OP_MIN) cv::min(aT, bT, ref);
                   else if (op == OP_MAX) cv::max(aT, bT, ref);
                   else cv::absdiff(aT, bT, ref); }
            else { Mat aD, bD, q; a.convertTo(aD, CV_64F); b.convertTo(bD, CV_64F);
                   if (op == OP_MUL) cv::multiply(aD, bD, q, scale); else cv::divide(aD, bD, q);
                   q.convertTo(ref, c.Tr); }
            if (masked) { Mat full = ref; ref = init.clone(); full.copyTo(ref, mask); }
            double n = cvtest::norm(out, ref, NORM_INF);
            double sc = std::max(1.0, cvtest::norm(ref, NORM_INF));
            double tol = (c.Tr==CV_16F||c.Tr==CV_16BF) ? (fp ? 1e-2*sc : 1.0)
                       : c.Tr==CV_32F                  ? (fp ? 1e-3*sc : 1e-3)
                       : (fp ? 1.0 : 0.0);
            EXPECT_LE(n, tol) << title << " " << c.name << " " << s.name;

            // cv:: timing reference (skip for mixed-type masked / mul / div, where the cv:: array op
            // needs same-type inputs). min/max/absdiff have no dtype arg and require identical input
            // types, so they are only timed against cv:: when da == db.
            const bool mm = (op == OP_MIN || op == OP_MAX || op == OP_ABSDIFF);
            double tc = -1;
            if (c.da == c.db || (!masked && !fp && !mm))
            {
                Mat tmp;
                InputArray m = masked ? InputArray(mask) : noArray();
                if      (op == OP_ADD) tc = minUs([&]{ cv::add     (a, b, tmp, m, c.Tr); }, 30, s.ninner);
                else if (op == OP_SUB) tc = minUs([&]{ cv::subtract(a, b, tmp, m, c.Tr); }, 30, s.ninner);
                else if (op == OP_MUL) tc = minUs([&]{ cv::multiply(a, b, tmp, scale, c.Tr); }, 30, s.ninner);
                else if (op == OP_MIN) tc = minUs([&]{ cv::min     (a, b, tmp); }, 30, s.ninner);
                else if (op == OP_MAX) tc = minUs([&]{ cv::max     (a, b, tmp); }, 30, s.ninner);
                else if (op == OP_ABSDIFF) tc = minUs([&]{ cv::absdiff(a, b, tmp); }, 30, s.ninner);
                else                   tc = minUs([&]{ cv::divide  (a, b, tmp, 1.0, c.Tr); }, 30, s.ninner);
            }

            std::cout << "  " << c.name << "  " << s.name << "  "
                      << std::fixed << std::setprecision(3) << std::setw(8) << te << "   ";
            if (tc >= 0)
                std::cout << std::setw(8) << tc << "   " << std::setprecision(2) << std::setw(6) << (tc/te) << "x";
            else
                std::cout << "      -          -  ";
            std::cout << "\n";
        }
}

TEST(Core_EW_Perf, add)
{
    perfBinOp(OP_ADD, "add", false);

    // per-channel scalar broadcast: (1024x1024) 8UC3 + (1x1) 8UC3  -> exercises the broadcast path
    Mat a = randMat({1024,1024}, 3, CV_8U, 0, 100);
    Mat b = randMat({1,1},      3, CV_8U, 0, 100);
    Mat inps[] = {a, b}, out;

    double te = minUs([&]{ TExpr p; makeAddProgram(p, CV_8U, CV_8U, CV_8U);
                           p.exec(inps, &out); }, 30, 4);

    Vec3b bv = b.at<Vec3b>(0, 0);
    Scalar sb(bv[0], bv[1], bv[2]);
    Mat ref;
    double tc = minUs([&]{ cv::add(a, sb, ref); }, 30, 4);
    EXPECT_EQ(0.0, cvtest::norm(out, ref, NORM_INF)) << "u8+scalar broadcast";

    std::cout << "\n[ew-perf] addScalar  u8 +u8 ->u8   1024x1024x3 + (1x1)x3   "
              << std::fixed << std::setprecision(3) << std::setw(8) << te << "   "
              << std::setw(8) << tc << "   " << std::setprecision(2) << std::setw(6) << (tc/te) << "x\n";
    std::cout << std::endl;
}

// Diagnostic: split the per-call cost into program BUILD (makeXxx + compile) vs EXEC (the run).
// For each op we time (build+exec) and (exec-only, program built once). The gap = build overhead,
// which a program cache would remove. 165x121 c1 (small, so overhead dominates).
TEST(Core_EW_Perf, buildVsExec)
{
    const std::vector<int> shape{165,121};
    std::cout << "\n[ew-perf] build-vs-exec  165x121 c1   (min us per call over 30 trials)\n";
    std::cout << "  op                build+exec   exec-only    build    cv::\n";
    std::cout << "  --------------------------------------------------------------\n";

    auto row = [&](const char* name, TOp op, int da, int db, int Tr,
                   std::function<void()> cvref)
    {
        Mat a = randMat(shape, 1, da, 1, 100);
        Mat b = randMat(shape, 1, db, 1, 100);
        if (op == OP_DIV) { Mat t; b.convertTo(t, CV_64F); t.setTo(1.0, t==0.0); t.convertTo(b, db); }
        Mat in[] = {a, b}, out;

        double tFull = minUs([&]{ TExpr p; makeBinaryArithProgram(p, op, da, db, Tr); p.exec(in, &out); }, 30, 2000);
        TExpr p; makeBinaryArithProgram(p, op, da, db, Tr);
        double tExec = minUs([&]{ p.exec(in, &out); }, 30, 2000);
        double tBuild = minUs([&]{ TExpr q; makeBinaryArithProgram(q, op, da, db, Tr); }, 30, 2000);
        double tcv = cvref ? minUs(cvref, 30, 2000) : -1;

        std::cout << "  " << std::left << std::setw(18) << name << std::right << std::fixed << std::setprecision(3)
                  << std::setw(8) << tFull << "   " << std::setw(8) << tExec << "   "
                  << std::setw(7) << tBuild << "  ";
        if (tcv >= 0) std::cout << std::setw(7) << tcv; else std::cout << "      -";
        std::cout << "\n";
    };

    Mat tmp;
    Mat a8 = randMat(shape,1,CV_8U,1,100), b8 = randMat(shape,1,CV_8U,1,100);
    row("add u8->u8",  OP_ADD, CV_8U, CV_8U, CV_8U, [&]{ cv::add(a8,b8,tmp); });
    row("mul u8->u8",  OP_MUL, CV_8U, CV_8U, CV_8U, [&]{ cv::multiply(a8,b8,tmp); });
    row("div u8->u8",  OP_DIV, CV_8U, CV_8U, CV_8U, nullptr);
    Mat af = randMat(shape,1,CV_32F,1,100), bf = randMat(shape,1,CV_32F,1,100);
    row("mul f32->f32", OP_MUL, CV_32F, CV_32F, CV_32F, [&]{ cv::multiply(af,bf,tmp); });
    std::cout << std::endl;
}

TEST(Core_EW_Perf, sub)
{
    perfBinOp(OP_SUB, "sub", false);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, mul)
{
    perfBinOp(OP_MUL, "mul", false);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, div)
{
    perfBinOp(OP_DIV, "div", false);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, min)
{
    perfBinOp(OP_MIN, "min", false);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, max)
{
    perfBinOp(OP_MAX, "max", false);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, absdiff)
{
    perfBinOp(OP_ABSDIFF, "absdiff", false);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, addMask)
{
    perfBinOp(OP_ADD, "add+mask", true);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, subMask)
{
    perfBinOp(OP_SUB, "sub+mask", true);
    std::cout << std::endl;
}

TEST(Core_EW_Perf, addWeighted)
{
    // fused: addWeighted(a,alpha,b,beta,gamma) = a*alpha + b*beta + gamma. Two convert_scale MACs +
    // an add => 2 temp buffers => exercises the body's L1 column-fragmentation.
    std::cout << "\n[ew-perf] addWeighted  (min us per call over 30 trials)\n";
    std::cout << "  combo         size           engine    cv::aW     speedup\n";

    const double alpha = 1.5, beta = -0.75, gamma = 12.0;
    struct Sz2 { std::vector<int> shape; int cn; int ninner; const char* name; };
    const Sz2 sizes2[] = {
        { {10,10,10},  1, 2000, "10x10x10    " },
        { {165,121},   1, 1000, "165x121     " },
        { {1024,1024}, 1,    8, "1024x1024   " },
    };
    for (const Sz2& s : sizes2)
    {
        Mat a = randMat(s.shape, s.cn, CV_32F, -100, 100);
        Mat b = randMat(s.shape, s.cn, CV_32F, -100, 100);
        Mat inps[] = {a, b}, out;
        double te = minUs([&]{ TExpr p; makeAddWeightedProgram(p, CV_32F, CV_32F, CV_32F, alpha, beta, gamma);
                               p.exec(inps, &out); }, 30, s.ninner);
        Mat ref;
        double tc = minUs([&]{ cv::addWeighted(a, alpha, b, beta, gamma, ref); }, 30, s.ninner);
        EXPECT_LE(cvtest::norm(out, ref, NORM_INF), 1e-2) << "addWeighted " << s.name;

        std::cout << "  f32 aW->f32   " << s.name << "  " << std::fixed << std::setprecision(3)
                  << std::setw(8) << te << "   " << std::setw(8) << tc << "   "
                  << std::setprecision(2) << std::setw(6) << (tc/te) << "x\n";
    }
    std::cout << std::endl;
}

}} // namespace
