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

TEST(Core_EW_Perf, add)
{
    struct Combo { int da, db, Tr; const char* name; };
    const Combo combos[] = {
        { CV_8U,  CV_8U,  CV_8U,  "u8 +u8 ->u8 " },
        { CV_16F, CV_16F, CV_16F, "f16+f16->f16" },
        { CV_32F, CV_32F, CV_32F, "f32+f32->f32" },
        { CV_8U,  CV_16F, CV_16F, "u8 +f16->f16" },
    };
    // ninner = inner repeats per timed trial (more for cheap/small cases, fewer for big ones).
    struct Sz { std::vector<int> shape; int cn; int ninner; const char* name; };
    const Sz sizes[] = {
        { {10,10,10},   1, 5000, "10x10x10    " },
        { {165,121},    1, 2000, "165x121     " },
        { {1024,1024},  3,   4, "1024x1024x3 " },
    };

    struct Op { ElemwiseOp op; const char* name; };
    const Op ops[] = { { OP_ADD, "add" }, { OP_SUB, "sub" } };

    for (const Op& o : ops)
    {
        std::cout << "\n[ew-perf] " << o.name << "  (min us per call over 30 trials)\n";
        std::cout << "  combo         size           engine    cv::op     speedup\n";

        for (const Combo& c : combos)
            for (const Sz& s : sizes)
            {
                Mat a = randMat(s.shape, s.cn, c.da, 0, 100);
                Mat b = randMat(s.shape, s.cn, c.db, 0, 100);
                Mat inps[] = {a, b}, out;

                // Full per-call path (matches a future cv::add/subtract): build the program every call.
                double te = minUs([&]{ EwProgram p; makeBinaryArithProgram(p, o.op, c.da, c.db, c.Tr);
                                       p.exec(inps, &out); }, 30, s.ninner);
                //double te = minUs([&]{ expression("{0} + {1}", inps, outs); }, 30, s.ninner);

                // cv:: reference only for equal input types (its array-op-array path needs that)
                double tc = -1;
                //if (c.da == c.db)
                {
                    Mat ref;
                    if (o.op == OP_SUB)
                        tc = minUs([&]{ cv::subtract(a, b, ref, noArray(), c.Tr); }, 30, s.ninner);
                    else
                        tc = minUs([&]{ cv::add(a, b, ref, noArray(), c.Tr); }, 30, s.ninner);
                    double n = cvtest::norm(out, ref, NORM_INF);
                    double tol = (c.Tr==CV_16F||c.Tr==CV_16BF) ? 1.0 : c.Tr==CV_32F ? 1e-3 : 0.0;
                    EXPECT_LE(n, tol) << o.name << " " << c.name << " " << s.name;
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

    // per-channel scalar broadcast: (1024x1024) 8UC3 + (1x1) 8UC3  -> exercises the broadcast path
    {
        Mat a = randMat({1024,1024}, 3, CV_8U, 0, 100);
        Mat b = randMat({1,1},      3, CV_8U, 0, 100);
        Mat inps[] = {a, b}, out;

        double te = minUs([&]{ EwProgram p; makeAddProgram(p, CV_8U, CV_8U, CV_8U);
                               p.exec(inps, &out); }, 30, 4);

        Vec3b bv = b.at<Vec3b>(0, 0);
        Scalar sb(bv[0], bv[1], bv[2]);
        Mat ref;
        double tc = minUs([&]{ cv::add(a, sb, ref); }, 30, 4);
        EXPECT_EQ(0.0, cvtest::norm(out, ref, NORM_INF)) << "u8+scalar broadcast";

        std::cout << "\n[ew-perf] addScalar  u8 +u8 ->u8   1024x1024x3 + (1x1)x3   "
                  << std::fixed << std::setprecision(3) << std::setw(8) << te << "   "
                  << std::setw(8) << tc << "   " << std::setprecision(2) << std::setw(6) << (tc/te) << "x\n";
    }

    // fused: addWeighted(a,alpha,b,beta,gamma) = a*alpha + b*beta + gamma. Two convert_scale MACs +
    // an add => 2 temp buffers => exercises the body's L1 column-fragmentation.
    std::cout << "\n[ew-perf] addWeighted  (min us per call over 30 trials)\n";
    std::cout << "  combo         size           engine    cv::aW     speedup\n";
    {
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
            double te = minUs([&]{ EwProgram p; makeAddWeightedProgram(p, CV_32F, CV_32F, CV_32F, alpha, beta, gamma);
                                   p.exec(inps, &out); }, 30, s.ninner);
            Mat ref;
            double tc = minUs([&]{ cv::addWeighted(a, alpha, b, beta, gamma, ref); }, 30, s.ninner);
            EXPECT_LE(cvtest::norm(out, ref, NORM_INF), 1e-2) << "addWeighted " << s.name;

            std::cout << "  f32 aW->f32   " << s.name << "  " << std::fixed << std::setprecision(3)
                      << std::setw(8) << te << "   " << std::setw(8) << tc << "   "
                      << std::setprecision(2) << std::setw(6) << (tc/te) << "x\n";
        }
    }
    std::cout << std::endl;
}

}} // namespace
