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

            for (int i = 0; i < 4; ++i)
            {
                n = 1 + rand() % 100;
                m = 1 + rand() % 100;
                do h = 1 + rand() % 20; while (h > n);
                do w = 1 + rand() % 20; while (w > m);
                gen(image, n, m, CV_8U);
                gen(templ, h, w, CV_8U);

                match_template_naive(image, templ, dst_gold);
                gpu::matchTemplate(gpu::GpuMat(image), gpu::GpuMat(templ), dst, CV_TM_SQDIFF);
                if (!check8U(dst_gold, Mat(dst))) return;
            }
        }
        catch (const Exception& e)
        {
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
    }

    // Naive version for unsigned char
    // Time complexity is O(a.size().area() * b.size().area()).
    void match_template_naive(const Mat& a, const Mat& b, Mat& c)
    {
        c.create(a.rows - b.rows + 1, a.cols - b.cols + 1, CV_32F);         
        for (int i = 0; i < c.rows; ++i)
        {
            for (int j = 0; j < c.cols; ++j)
            {
                float delta;
                float sum = 0.f;
                for (int y = 0; y < b.rows; ++y)
                {
                    const unsigned char* arow = a.ptr(i + y);
                    const unsigned char* brow = b.ptr(y);
                    for (int x = 0; x < b.cols; ++x)
                    {
                        delta = (float)(arow[j + x] - brow[x]);
                        sum += delta * delta;
                    }
                }
                c.at<float>(i, j) = sum;
            }
        }
    }


    bool check8U(const Mat& a, const Mat& b)
    {
        if (a.size() != b.size())
        {
            ts->printf(CvTS::CONSOLE, "bad size");
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            return false;
        }

        for (int i = 0; i < a.rows; ++i)
        {
            for (int j = 0; j < a.cols; ++j)
            {
                float v1 = a.at<float>(i, j);
                float v2 = b.at<float>(i, j);
                if (fabs(v1 - v2) > 1e-3f)
                {
                    ts->printf(CvTS::CONSOLE, "(gold)%f != %f, pos: (%d, %d) size: (%d, %d)\n", 
                               v1, v2, j, i, a.cols, a.rows);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                }
            }
        }

        return true;
    }
} match_template_test;
