// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Ad-hoc perf comparison for the element-wise engine vs classic cv::add. Lives in
// opencv_test_core for now (move to opencv_perf_core later). Each (type-combo, size) is run
// 10..30 times; the minimum getTickCount() time is reported as the most stable metric.

#include "../test_precomp.hpp"
#include "ew_exec.hpp"
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

template<typename F>
static double minMs(F&& f, int iters)
{
    f();  // warmup
    double best = DBL_MAX;
    for (int i = 0; i < iters; i++)
    {
        int64 t0 = getTickCount();
        f();
        double ms = (getTickCount() - t0) * 1000.0 / getTickFrequency();
        best = std::min(best, ms);
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
    struct Sz { std::vector<int> shape; int cn; const char* name; };
    const Sz sizes[] = {
        { {10,10,10},   1, "10x10x10    " },
        { {165,121},    1, "165x121     " },
        { {1024,1024},  3, "1024x1024x3 " },
    };

    std::cout << "\n[ew-perf] add  (min ms over 30 runs)\n";
    std::cout << "  combo         size           engine     cv::add    speedup\n";

    for (const Combo& c : combos)
        for (const Sz& s : sizes)
        {
            Mat a = randMat(s.shape, s.cn, c.da, 0, 100);
            Mat b = randMat(s.shape, s.cn, c.db, 0, 100);

            EwProgram p = makeAddProgram(c.da, c.db, c.Tr);
            std::vector<Mat> out;
            double te = minMs([&]{ exec(p, { a, b }, out); }, 30);

            // cv::add only for equal input types (its array-op-array path needs that)
            double tc = -1;
            if (c.da == c.db)
            {
                Mat ref;
                tc = minMs([&]{ cv::add(a, b, ref, noArray(), c.Tr); }, 30);
                // sanity: engine matches cv::add
                double n = cvtest::norm(out[0], ref, NORM_INF);
                double tol = (c.Tr==CV_16F||c.Tr==CV_16BF) ? 1.0 : c.Tr==CV_32F ? 1e-3 : 0.0;
                EXPECT_LE(n, tol) << c.name << " " << s.name;
            }

            std::cout << "  " << c.name << "  " << s.name << "  "
                      << std::fixed << std::setprecision(3) << std::setw(8) << te << "   ";
            if (tc >= 0)
                std::cout << std::setw(8) << tc << "   " << std::setprecision(2) << std::setw(6) << (tc/te) << "x";
            else
                std::cout << "      -          -  ";
            std::cout << "\n";
        }

    // per-channel scalar broadcast: (1024x1024) 8UC3 + (1x1) 8UC3  -> exercises the broadcast path
    {
        Mat a = randMat({1024,1024}, 3, CV_8U, 0, 100);
        Mat b = randMat({1,1},      3, CV_8U, 0, 100);

        EwProgram p = makeAddProgram(CV_8U, CV_8U, CV_8U);
        std::vector<Mat> out;
        double te = minMs([&]{ exec(p, { a, b }, out); }, 30);

        Vec3b bv = b.at<Vec3b>(0, 0);
        Scalar sb(bv[0], bv[1], bv[2]);
        Mat ref;
        double tc = minMs([&]{ cv::add(a, sb, ref); }, 30);
        EXPECT_EQ(0.0, cvtest::norm(out[0], ref, NORM_INF)) << "u8+scalar broadcast";

        std::cout << "  u8 +u8 ->u8   1024x1024x3 + (1x1)x3   "
                  << std::fixed << std::setprecision(3) << std::setw(8) << te << "   "
                  << std::setw(8) << tc << "   " << std::setprecision(2) << std::setw(6) << (tc/te) << "x\n";
    }
    std::cout << std::endl;
}

}} // namespace
