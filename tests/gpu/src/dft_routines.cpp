/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "gputest.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

struct CV_GpuMulSpectrumsTest: CvTest
{
    CV_GpuMulSpectrumsTest(): CvTest("GPU-MulSpectrumsTest", "mulSpectrums") {}

    void run(int)
    {
        try
        {
            test(0);
            testConj(0);
            testScaled(0);
            testScaledConj(0);
            test(DFT_ROWS);
            testConj(DFT_ROWS);
            testScaled(DFT_ROWS);
            testScaledConj(DFT_ROWS);
        }
        catch (const Exception& e)
        {
            ts->printf(CvTS::CONSOLE, e.what());
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    void gen(int cols, int rows, Mat& mat)
    {
        RNG rng;
        mat.create(rows, cols, CV_32FC2);
        rng.fill(mat, RNG::UNIFORM, Scalar::all(0.f), Scalar::all(10.f));
    }

    bool cmp(const Mat& gold, const Mat& mine, float max_err=1e-3f)
    {
        if (gold.size() != mine.size())
        {
            ts->printf(CvTS::CONSOLE, "bad sizes: gold: %d d%, mine: %d %d\n", gold.cols, gold.rows, mine.cols, mine.rows);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        if (gold.type() != mine.type())
        {
            ts->printf(CvTS::CONSOLE, "bad types: gold=%d, mine=%d\n", gold.type(), mine.type());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        for (int i = 0; i < gold.rows; ++i)
        {
            for (int j = 0; j < gold.cols * 2; ++j)
            {
                float gold_ = gold.at<float>(i, j);
                float mine_ = mine.at<float>(i, j);
                if (fabs(gold_ - mine_) > max_err)
                {
                    ts->printf(CvTS::CONSOLE, "bad values at %d %d: gold=%f, mine=%f\n", j, i, gold_, mine_);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                }
            }
        }
        return true;
    }

    bool cmpScaled(const Mat& gold, const Mat& mine, float scale, float max_err=1e-3f)
    {
        if (gold.size() != mine.size())
        {
            ts->printf(CvTS::CONSOLE, "bad sizes: gold: %d d%, mine: %d %d\n", gold.cols, gold.rows, mine.cols, mine.rows);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        if (gold.type() != mine.type())
        {
            ts->printf(CvTS::CONSOLE, "bad types: gold=%d, mine=%d\n", gold.type(), mine.type());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        for (int i = 0; i < gold.rows; ++i)
        {
            for (int j = 0; j < gold.cols * 2; ++j)
            {
                float gold_ = gold.at<float>(i, j) * scale;
                float mine_ = mine.at<float>(i, j);
                if (fabs(gold_ - mine_) > max_err)
                {
                    ts->printf(CvTS::CONSOLE, "bad values at %d %d: gold=%f, mine=%f\n", j, i, gold_, mine_);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                }
            }
        }
        return true;
    }

    void test(int flags)
    {
        int cols = 1 + rand() % 100, rows = 1 + rand() % 1000;

        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);

        Mat c_gold;
        mulSpectrums(a, b, c_gold, flags, false);

        GpuMat d_c;
        mulSpectrums(GpuMat(a), GpuMat(b), d_c, flags, false);

        if (!cmp(c_gold, Mat(d_c)))
            ts->printf(CvTS::CONSOLE, "test failed: cols=%d, rows=%d, flags=%d\n", cols, rows, flags);
    }

    void testConj(int flags)
    {
        int cols = 1 + rand() % 100, rows = 1 + rand() % 1000;

        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);

        Mat c_gold;
        mulSpectrums(a, b, c_gold, flags, true);

        GpuMat d_c;
        mulSpectrums(GpuMat(a), GpuMat(b), d_c, flags, true);

        if (!cmp(c_gold, Mat(d_c)))
            ts->printf(CvTS::CONSOLE, "testConj failed: cols=%d, rows=%d, flags=%d\n", cols, rows, flags);
    }

    void testScaled(int flags)
    {
        int cols = 1 + rand() % 100, rows = 1 + rand() % 1000;

        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);
        float scale = 1.f / a.size().area();

        Mat c_gold;
        mulSpectrums(a, b, c_gold, flags, false);

        GpuMat d_c;
        mulAndScaleSpectrums(GpuMat(a), GpuMat(b), d_c, flags, scale, false);

        if (!cmpScaled(c_gold, Mat(d_c), scale))
            ts->printf(CvTS::CONSOLE, "testScaled failed: cols=%d, rows=%d, flags=%d\n", cols, rows, flags);
    }

    void testScaledConj(int flags)
    {
        int cols = 1 + rand() % 100, rows = 1 + rand() % 1000;

        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);
        float scale = 1.f / a.size().area();

        Mat c_gold;
        mulSpectrums(a, b, c_gold, flags, true);

        GpuMat d_c;
        mulAndScaleSpectrums(GpuMat(a), GpuMat(b), d_c, flags, scale, true);

        if (!cmpScaled(c_gold, Mat(d_c), scale))
            ts->printf(CvTS::CONSOLE, "testScaledConj failed: cols=%d, rows=%d, flags=%D\n", cols, rows, flags);
    }
} CV_GpuMulSpectrumsTest_inst;


struct CV_GpuDftTest: CvTest
{
    CV_GpuDftTest(): CvTest("GPU-DftTest", "dft") {}

    void run(int)
    {
        try
        {
            int cols = 1 + rand() % 100, rows = 1 + rand() % 100;

            testC2C(cols, rows, 0, "no flags");
            testC2C(cols, rows + 1, 0, "no flags 0 1");
            testC2C(cols, rows + 1, 0, "no flags 1 0");
            testC2C(cols + 1, rows, 0, "no flags 1 1");
            testC2C(cols, rows, DFT_INVERSE, "DFT_INVERSE");
            testC2C(cols, rows, DFT_ROWS, "DFT_ROWS");
            testC2C(1, rows, 0, "single col");
            testC2C(cols, 1, 0, "single row");
            testC2C(1, rows, DFT_INVERSE, "single col inversed");
            testC2C(cols, 1, DFT_INVERSE, "single row inversed");
            testC2C(cols, 1, DFT_ROWS, "single row DFT_ROWS");
            testC2C(1, 2, 0, "size 1 2");
            testC2C(2, 1, 0, "size 2 1");

            testR2CThenC2R(cols, rows, "sanity");
            testR2CThenC2R(cols, rows + 1, "sanity 0 1");
            testR2CThenC2R(cols + 1, rows, "sanity 1 0");
            testR2CThenC2R(cols + 1, rows + 1, "sanity 1 1");
            testR2CThenC2R(1, rows, "single col");
            testR2CThenC2R(1, rows + 1, "single col 1");
            testR2CThenC2R(cols, 1, "single row" );;
            testR2CThenC2R(cols + 1, 1, "single row 1" );;
        }
        catch (const Exception& e)
        {
            ts->printf(CvTS::CONSOLE, e.what());
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    void gen(int cols, int rows, int cn, Mat& mat)
    {
        RNG rng;
        mat.create(rows, cols, CV_MAKETYPE(CV_32F, cn));
        rng.fill(mat, RNG::UNIFORM, Scalar::all(0.f), Scalar::all(10.f));
    }

    bool cmp(const Mat& gold, const Mat& mine, float max_err=1e-3f, float scale=1.f)
    {
        if (gold.size() != mine.size())
        {
            ts->printf(CvTS::CONSOLE, "bad sizes: gold: %d %d, mine: %d %d\n", gold.cols, gold.rows, mine.cols, mine.rows);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        if (gold.depth() != mine.depth())
        {
            ts->printf(CvTS::CONSOLE, "bad depth: gold=%d, mine=%d\n", gold.depth(), mine.depth());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        if (gold.channels() != mine.channels())
        {
            ts->printf(CvTS::CONSOLE, "bad channel count: gold=%d, mine=%d\n", gold.channels(), mine.channels());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }
        for (int i = 0; i < gold.rows; ++i)
        {
            for (int j = 0; j < gold.cols * gold.channels(); ++j)
            {
                float gold_ = gold.at<float>(i, j);
                float mine_ = mine.at<float>(i, j) * scale;
                if (fabs(gold_ - mine_) > max_err)
                {
                    ts->printf(CvTS::CONSOLE, "bad values at %d %d: gold=%f, mine=%f\n", j / gold.channels(), i, gold_, mine_);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                }
            }
        }
        return true;
    }

    void testC2C(int cols, int rows, int flags, const std::string& hint)
    {
        Mat a;
        gen(cols, rows, 2, a);

        Mat b_gold;
        dft(a, b_gold, flags);

        GpuMat d_b;
        dft(GpuMat(a), d_b, flags);

        bool ok = true;
        if (ok && d_b.depth() != CV_32F)
        {
            ts->printf(CvTS::CONSOLE, "bad depth: %d\n", d_b.depth());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            ok = false;
        }
        if (ok && d_b.channels() != 2)
        {
            ts->printf(CvTS::CONSOLE, "bad channel count: %d\n", d_b.channels());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            ok = false;
        }
        if (ok) ok = cmp(b_gold, Mat(d_b), rows * cols * 1e-5f);
        if (!ok) 
            ts->printf(CvTS::CONSOLE, "testC2C failed: hint=%s, cols=%d, rows=%d, flags=%d\n", hint.c_str(), cols, rows, flags);
    }

    void testR2CThenC2R(int cols, int rows, const std::string& hint)
    {
        Mat a;
        gen(cols, rows, 1, a);

        bool odd = false;
        if (a.cols == 1) odd = a.rows % 2 == 1;
        else odd = a.cols % 2 == 1;
        bool ok = true;

        GpuMat d_b;
        GpuMat d_c;
        dft(GpuMat(a), d_b, 0);
        dft(d_b, d_c, DFT_REAL_OUTPUT, 0, odd);

        if (ok && d_c.depth() != CV_32F)
        {
            ts->printf(CvTS::CONSOLE, "bad depth: %d\n", d_c.depth());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            ok = false;
        }
        if (ok && d_c.channels() != 1)
        {
            ts->printf(CvTS::CONSOLE, "bad channel count: %d\n", d_c.channels());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            ok = false;
        }
        if (ok) ok = cmp(a, Mat(d_c), rows * cols * 1e-5f, 1.f / (rows * cols));
        if (!ok) 
            ts->printf(CvTS::CONSOLE, "testR2CThenC2R failed: hint=%s, cols=%d, rows=%d\n", hint.c_str(), cols, rows);
    }
} CV_GpuDftTest_inst;