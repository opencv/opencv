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
            bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                             gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
            if (!double_ok)
            {
                // For sqrIntegral
                ts->printf(CvTS::CONSOLE, "\nCode and device double support is required (CC >= 1.3)");
                ts->set_failed_test_info(CvTS::FAIL_GENERIC);
                return;
            }

            Mat image, templ;
            Mat dst_gold;
            gpu::GpuMat dst;
            int n, m, h, w;
            F(clock_t t;)

            RNG rng(*ts->get_rng());

            for (int cn = 1; cn <= 4; ++cn)
            {
                F(ts->printf(CvTS::CONSOLE, "cn: %d\n", cn);)
                for (int i = 0; i <= 0; ++i)
                {
                    n = rng.uniform(30, 100);
                    m = rng.uniform(30, 100);
                    h = rng.uniform(5, n - 1);
                    w = rng.uniform(5, m - 1);

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 5 * h * w * 1e-4f, "SQDIFF 8U")) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF_NORMED);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF_NORMED);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), h * w * 1e-5f, "SQDIFF_NOREMD 8U")) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCORR);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 5 * h * w * cn * cn * 1e-5f, "CCORR 8U")) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCORR_NORMED);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR_NORMED);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), h * w * 1e-6f, "CCORR_NORMED 8U")) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCOEFF);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCOEFF);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 5 * h * w * cn * cn * 1e-5f, "CCOEFF 8U")) return;

                    gen(image, n, m, CV_8U, cn);
                    gen(templ, h, w, CV_8U, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCOEFF_NORMED);
                    F(cout << "depth: 8U cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCOEFF_NORMED);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), h * w * 1e-6f, "CCOEFF_NORMED 8U")) return;

                    gen(image, n, m, CV_32F, cn);
                    gen(templ, h, w, CV_32F, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_SQDIFF);
                    F(cout << "depth: 32F cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 0.25f * h * w * 1e-5f, "SQDIFF 32F")) return;

                    gen(image, n, m, CV_32F, cn);
                    gen(templ, h, w, CV_32F, cn);
                    F(t = clock();)
                    matchTemplate(image, templ, dst_gold, CV_TM_CCORR);
                    F(cout << "depth: 32F cn: " << cn << " n: " << n << " m: " << m << " w: " << w << " h: " << h << endl;)
                    F(cout << "cpu:" << clock() - t << endl;)
                    F(t = clock();)
                    gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_CCORR);
                    F(cout << "gpu_block: " << clock() - t << endl;)
                    if (!check(dst_gold, Mat(dst), 0.25f * h * w * 1e-5f, "CCORR 32F")) return;
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

    bool check(const Mat& a, const Mat& b, float max_err, const string& method="")
    {
        if (a.size() != b.size())
        {
            ts->printf(CvTS::CONSOLE, "bad size, method=%s\n", method.c_str());
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }

        //for (int i = 0; i < a.rows; ++i)
        //{
        //    for (int j = 0; j < a.cols; ++j)
        //    {
        //        float a_ = a.at<float>(i, j);
        //        float b_ = b.at<float>(i, j);
        //        if (fabs(a_ - b_) > max_err)
        //        {
        //            ts->printf(CvTS::CONSOLE, "a=%f, b=%f, i=%d, j=%d\n", a_, b_, i, j);
        //            cin.get();
        //        }
        //    }
        //}

        float err = (float)norm(a, b, NORM_INF);
        if (err > max_err)
        {
            ts->printf(CvTS::CONSOLE, "bad accuracy: %f, method=%s\n", err, method.c_str());
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
            bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) && 
                             gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
            if (!double_ok)
            {
                // For sqrIntegral
                ts->printf(CvTS::CONSOLE, "\nCode and device double support is required (CC >= 1.3)");
                ts->set_failed_test_info(CvTS::FAIL_GENERIC);
                return;
            }

            Mat image = imread(std::string(ts->get_data_path()) + "matchtemplate/black.png");
            if (image.empty())
            {
                ts->printf(CvTS::CONSOLE, "can't open file '%s'", (std::string(ts->get_data_path()) 
                                                                   + "matchtemplate/black.png").c_str());
                ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
                return;
            }

            Mat pattern = imread(std::string(ts->get_data_path()) + "matchtemplate/cat.png");
            if (pattern.empty())
            {
                ts->printf(CvTS::CONSOLE, "can't open file '%s'", (std::string(ts->get_data_path()) 
                                                                   + "matchtemplate/cat.png").c_str());
                ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
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