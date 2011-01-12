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
#define F(x) x
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

            for (int cn = 1; cn <= 4; ++cn)
            {
                F(ts->printf(CvTS::CONSOLE, "cn: %d\n", cn);)
                for (int i = 0; i <= 0; ++i)
                {
                    n = 1 + rand() % 1000;
                    m = 1 + rand() % 1000;
                    do h = 1 + rand() % 100; while (h > n);
                    do w = 1 + rand() % 100; while (w > m);

                    //cout << "w: " << w << " h: " << h << endl;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 5 * h * w * 1e-4f)) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF_NORMED);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF_NORMED);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), h * w * 1e-5f)) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCORR);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 5 * h * w * cn * cn * 1e-5f)) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCORR_NORMED);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR_NORMED);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), h * w * 1e-6f)) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCOEFF);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCOEFF);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 5 * h * w * cn * cn * 1e-5f)) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCOEFF_NORMED);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCOEFF_NORMED);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), h * w * 1e-6f)) return;

                    gen(image, n, m, CV_32F, cn);
                    gen(templ, h, w, CV_32F, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF);
                    F(cout << "depth: 32F cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 0.25f * h * w * 1e-5f)) return;

                    gen(image, n, m, CV_32F, cn);
                    gen(templ, h, w, CV_32F, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCORR);
                    F(cout << "depth: 32F cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 0.25f * h * w * 1e-5f)) return;
                }
            }
        }
        catch (const Exception& e)
        {
            ts->printf(CvTS::CONSOLE, e.what());
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    void gen(Mat& a, int rows, int cols, int depth, int cn)
    {
        RNG rng;
        a.create(rows, cols, CV_MAKETYPE(depth, cn));
        if (depth == CV_8U)
            rng.fill(a, RNG::UNIFORM, Scalar::all(1), Scalar::all(10));
        else if (depth == CV_32F)
            rng.fill(a, RNG::UNIFORM, Scalar::all(0.001f), Scalar::all(1.f));
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
} match_template_test;

struct CV_GpuMatchTemplateFindPatternInBlackTest: CvTest 
{
    CV_GpuMatchTemplateFindPatternInBlackTest()
            : CvTest("GPU-MatchTemplateFindPatternInBlackTest", "matchTemplate") {}

    void run(int)
    {
        try
        {
            Mat image = imread(std::string(ts->get_data_path()) + "matchtemplate/black.jpg");
            if (image.empty())
            {
                ts->printf(CvTS::CONSOLE, "can't open file '%s'", (std::string(ts->get_data_path()) 
                                                                   + "matchtemplate/black.jpg").c_str());
                ts->set_failed_test_info(CvTS::FAIL_INVALID_TEST_DATA);
                return;
            }

            Mat pattern = imread(std::string(ts->get_data_path()) + "matchtemplate/cat.jpg");
            if (pattern.empty())
            {
                ts->printf(CvTS::CONSOLE, "can't open file '%s'", (std::string(ts->get_data_path()) 
                                                                   + "matchtemplate/cat.jpg").c_str());
                ts->set_failed_test_info(CvTS::FAIL_INVALID_TEST_DATA);
                return;
            }

            gpu::GpuMat d_image(image);
            gpu::GpuMat d_pattern(pattern);
            gpu::GpuMat d_result;

            double maxValue;
            Point maxLoc;
            Point maxLocGold(284, 12);

            gpu::matchTemplate(d_image, d_pattern, d_result, CV_TM_CCOEFF_NORMED);
            gpu::minMaxLoc(d_result, NULL, &maxValue, NULL, &maxLoc );
            if (maxLoc != maxLocGold)
            {
                ts->printf(CvTS::CONSOLE, "bad match (CV_TM_CCOEFF_NORMED): %d %d, must be at: %d %d", 
                           maxLoc.x, maxLoc.y, maxLocGold.x, maxLocGold.y);
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }

            gpu::matchTemplate(d_image, d_pattern, d_result, CV_TM_CCORR_NORMED);
            gpu::minMaxLoc(d_result, NULL, &maxValue, NULL, &maxLoc );
            if (maxLoc != maxLocGold)
            {
                ts->printf(CvTS::CONSOLE, "bad match (CV_TM_CCORR_NORMED): %d %d, must be at: %d %d", 
                           maxLoc.x, maxLoc.y, maxLocGold.x, maxLocGold.y);
                ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
        catch (const Exception& e)
        {
            ts->printf(CvTS::CONSOLE, e.what());
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }
} match_templet_find_bordered_pattern_test;