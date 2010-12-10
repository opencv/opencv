/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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
#include <string>
#include <iostream>

//#define SHOW_TIME

#ifdef SHOW_TIME
#include <ctime>
#define F(x)
#else
#define F(x)
#endif

using namespace cv;
using namespace std;

struct CV_GpuMatchTemplateTest: CvTest 
{
    CV_GpuMatchTemplateTest(): CvTest("GPU-MatchTemplateTest", "matchTemplate") {}

    void run(int)
    {
        try
        {
            Mat image, templ;
            Mat dst_gold;
            gpu::GpuMat dst;
            int n, m, h, w;
            F(clock_t t;)

            for (int i = 0; i < 3; ++i)
            {
                n = 1 + rand() % 2000;
                m = 1 + rand() % 1000;
                do h = 1 + rand() % 30; while (h > n);
                do w = 1 + rand() % 30; while (w > m);

                gen(image, n, m, CV_8U);
                gen(templ, h, w, CV_8U);
                F(t = clock();)
                matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF);
                F(cout << "cpu:" << clock() - t << endl;)
                F(t = clock();)
                gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                F(cout << "gpu_block: " << clock() - t << endl;)
                if (!check(dst_gold, Mat(dst), 5 * h * w * 1e-5f)) return;

                gen(image, n, m, CV_8U);
                gen(templ, h, w, CV_8U);
                F(t = clock();)
                matchTemplate(image, templ, dst_gold, CV_TM_CCORR);
                F(cout << "cpu:" << clock() - t << endl;)
                F(t = clock();)
                gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR);
                F(cout << "gpu_block: " << clock() - t << endl;)
                if (!check(dst_gold, Mat(dst), 5 * h * w * 1e-5f)) return;

                gen(image, n, m, CV_32F);
                gen(templ, h, w, CV_32F);
                F(t = clock();)
                matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF);
                F(cout << "cpu:" << clock() - t << endl;)
                F(t = clock();)
                gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                F(cout << "gpu_block: " << clock() - t << endl;)
                if (!check(dst_gold, Mat(dst), 0.25f * h * w * 1e-5f)) return;

                //gen(image, n, m, CV_32F);
                //gen(templ, h, w, CV_32F);
                //F(t = clock();)
                //matchTemplate(image, templ, dst_gold, CV_TM_CCORR);
                //F(cout << "cpu:" << clock() - t << endl;)
                //F(t = clock();)
                //gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR);
                //F(cout << "gpu_block: " << clock() - t << endl;)
                //if (!check(dst_gold, Mat(dst), 0.25f * h * w * 1e-5f)) return;
            }
        }
        catch (const Exception& e)
        {
            ts->printf(CvTS::CONSOLE, e.what());
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    void gen(Mat& a, int rows, int cols, int type)
    {
        RNG rng;
        a.create(rows, cols, type);
        if (type == CV_8U)
            rng.fill(a, RNG::UNIFORM, Scalar(0), Scalar(10));
        else if (type == CV_32F)
            rng.fill(a, RNG::UNIFORM, Scalar(0.f), Scalar(1.f));
    }

    bool check(const Mat& a, const Mat& b, float max_err)
    {
        if (a.size() != b.size())
        {
            ts->printf(CvTS::CONSOLE, "bad size");
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }

        float err = (float)norm(a, b, NORM_INF);
        if (err > max_err)
        {
            ts->printf(CvTS::CONSOLE, "bad accuracy: %f\n", err);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }

        return true;
    }

    //void match_template_naive_SQDIFF(const Mat& a, const Mat& b, Mat& c)
    //{
    //    c.create(a.rows - b.rows + 1, a.cols - b.cols + 1, CV_32F);         
    //    for (int i = 0; i < c.rows; ++i)
    //    {
    //        for (int j = 0; j < c.cols; ++j)
    //        {
    //            float delta;
    //            float sum = 0.f;
    //            for (int y = 0; y < b.rows; ++y)
    //            {
    //                const unsigned char* arow = a.ptr(i + y);
    //                const unsigned char* brow = b.ptr(y);
    //                for (int x = 0; x < b.cols; ++x)
    //                {
    //                    delta = (float)(arow[j + x] - brow[x]);
    //                    sum += delta * delta;
    //                }
    //            }
    //            c.at<float>(i, j) = sum;
    //        }
    //    }
    //}

    //void match_template_naive_CCORR(const Mat& a, const Mat& b, Mat& c)
    //{
    //    c.create(a.rows - b.rows + 1, a.cols - b.cols + 1, CV_32F);         
    //    for (int i = 0; i < c.rows; ++i)
    //    {
    //        for (int j = 0; j < c.cols; ++j)
    //        {
    //            float sum = 0.f;
    //            for (int y = 0; y < b.rows; ++y)
    //            {
    //                const float* arow = a.ptr<float>(i + y);
    //                const float* brow = b.ptr<float>(y);
    //                for (int x = 0; x < b.cols; ++x)
    //                    sum += arow[j + x] * brow[x];
    //            }
    //            c.at<float>(i, j) = sum;
    //        }
    //    }
    //}
} match_template_test;

