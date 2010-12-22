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
            if (!test(1 + rand() % 100, 1 + rand() % 1000)) return;
            if (!testConj(1 + rand() % 100, 1 + rand() % 1000)) return;
            if (!testScaled(1 + rand() % 100, 1 + rand() % 1000)) return;
            if (!testScaledConj(1 + rand() % 100, 1 + rand() % 1000)) return;
        }
        catch (const Exception& e)
        {
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

    bool test(int cols, int rows)
    {
        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);

        Mat c_gold;
        mulSpectrums(a, b, c_gold, 0, false);

        GpuMat d_c;
        mulSpectrums(GpuMat(a), GpuMat(b), d_c, 0, false);

        return cmp(c_gold, Mat(d_c)) 
            || (ts->printf(CvTS::CONSOLE, "test failed: cols=%d, rows=%d\n", cols, rows), false);
    }

    bool testConj(int cols, int rows)
    {
        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);

        Mat c_gold;
        mulSpectrums(a, b, c_gold, 0, true);

        GpuMat d_c;
        mulSpectrums(GpuMat(a), GpuMat(b), d_c, 0, true);

        return cmp(c_gold, Mat(d_c)) 
            || (ts->printf(CvTS::CONSOLE, "testConj failed: cols=%d, rows=%d\n", cols, rows), false);
    }

    bool testScaled(int cols, int rows)
    {
        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);
        float scale = 1.f / a.size().area();

        Mat c_gold;
        mulSpectrums(a, b, c_gold, 0, false);

        GpuMat d_c;
        mulAndScaleSpectrums(GpuMat(a), GpuMat(b), d_c, 0, scale, false);

        return cmpScaled(c_gold, Mat(d_c), scale) 
            || (ts->printf(CvTS::CONSOLE, "testScaled failed: cols=%d, rows=%d\n", cols, rows), false);
    }

    bool testScaledConj(int cols, int rows)
    {
        Mat a, b;
        gen(cols, rows, a);
        gen(cols, rows, b);
        float scale = 1.f / a.size().area();

        Mat c_gold;
        mulSpectrums(a, b, c_gold, 0, true);

        GpuMat d_c;
        mulAndScaleSpectrums(GpuMat(a), GpuMat(b), d_c, 0, scale, true);

        return cmpScaled(c_gold, Mat(d_c), scale) 
            || (ts->printf(CvTS::CONSOLE, "testScaledConj failed: cols=%d, rows=%d\n", cols, rows), false);
    }
} CV_GpuMulSpectrumsTest_inst;