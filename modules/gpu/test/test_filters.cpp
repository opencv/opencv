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

#include <iostream>
#include <cmath>
#include <limits>
#include "test_precomp.hpp"

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuNppFilterTest : public cvtest::BaseTest
{
public:
    CV_GpuNppFilterTest(const char* test_name, const char* test_funcs) {}
    virtual ~CV_GpuNppFilterTest() {}

protected:
    void run(int);
    virtual int test(const Mat& img) = 0;
    
    int test8UC1(const Mat& img)
    {
        cv::Mat img_C1;
        cvtColor(img, img_C1, CV_BGR2GRAY);
        return test(img_C1);
    }

    int test8UC4(const Mat& img)
    {
        cv::Mat img_C4;
        cvtColor(img, img_C4, CV_BGR2BGRA);
        return test(img_C4);
    }
        
    int CheckNorm(const Mat& m1, const Mat& m2, const Size& ksize)
    {
        Rect roi = Rect(ksize.width, ksize.height, m1.cols - 2 * ksize.width, m1.rows - 2 * ksize.height);
        Mat m1ROI = m1(roi);
        Mat m2ROI = m2(roi);

        double res = norm(m1ROI, m2ROI, NORM_INF);

        // Max difference (2.0) in GaussianBlur
        if (res <= 2)
            return cvtest::TS::OK;
        
        ts->printf(cvtest::TS::LOG, "Norm: %f\n", res);
        return cvtest::TS::FAIL_GENERIC;
    }
};

void CV_GpuNppFilterTest::run( int )
{    
    cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");
    
    if (img.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        return;
    }

    //run tests
    int testResult = cvtest::TS::OK;

    if (test8UC1(img) != cvtest::TS::OK)
        testResult = cvtest::TS::FAIL_GENERIC;

    if (test8UC4(img) != cvtest::TS::OK)
        testResult = cvtest::TS::FAIL_GENERIC;

    ts->set_failed_test_info(testResult);

    ts->set_failed_test_info(cvtest::TS::OK);
}

////////////////////////////////////////////////////////////////////////////////
// blur
struct CV_GpuNppImageBlurTest : public CV_GpuNppFilterTest
{
    CV_GpuNppImageBlurTest() : CV_GpuNppFilterTest( "GPU-NppImageBlur", "blur" ) {}

    int test(const Mat& img)
    {
        int ksizes[] = {3, 5, 7};
        int ksizes_num = sizeof(ksizes) / sizeof(int);

        int test_res = cvtest::TS::OK;

        for (int i = 0; i < ksizes_num; ++i)
        {
            for (int j = 0; j < ksizes_num; ++j)
            {
                Size ksize(ksizes[i], ksizes[j]);

                ts->printf(cvtest::TS::LOG, "\nksize = (%dx%d)\n", ksizes[i], ksizes[j]);

                Mat cpudst;
                cv::blur(img, cpudst, ksize);

                GpuMat gpu1(img);
                GpuMat gpudst;
                cv::gpu::blur(gpu1, gpudst, ksize);

                if (CheckNorm(cpudst, gpudst, ksize) != cvtest::TS::OK)
                    test_res = cvtest::TS::FAIL_GENERIC;
            }
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Sobel
struct CV_GpuNppImageSobelTest : public CV_GpuNppFilterTest
{
    CV_GpuNppImageSobelTest() : CV_GpuNppFilterTest( "GPU-NppImageSobel", "Sobel" ) {}

    int test(const Mat& img)
    {
        int ksizes[] = {3, 5, 7};
        int ksizes_num = sizeof(ksizes) / sizeof(int);

        int dx = 1, dy = 0;

        int test_res = cvtest::TS::OK;

        for (int i = 0; i < ksizes_num; ++i)
        {
            ts->printf(cvtest::TS::LOG, "\nksize = %d\n", ksizes[i]);

            Mat cpudst;
            cv::Sobel(img, cpudst, -1, dx, dy, ksizes[i]);

            GpuMat gpu1(img);
            GpuMat gpudst;
            cv::gpu::Sobel(gpu1, gpudst, -1, dx, dy, ksizes[i]);

            if (CheckNorm(cpudst, gpudst, Size(ksizes[i], ksizes[i])) != cvtest::TS::OK)
                test_res = cvtest::TS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Scharr
struct CV_GpuNppImageScharrTest : public CV_GpuNppFilterTest
{
    CV_GpuNppImageScharrTest() : CV_GpuNppFilterTest( "GPU-NppImageScharr", "Scharr" ) {}

    int test(const Mat& img)
    {
        int dx = 1, dy = 0;

        Mat cpudst;
        cv::Scharr(img, cpudst, -1, dx, dy);

        GpuMat gpu1(img);
        GpuMat gpudst;
        cv::gpu::Scharr(gpu1, gpudst, -1, dx, dy);
                
        return CheckNorm(cpudst, gpudst, Size(3, 3));
    }
};


////////////////////////////////////////////////////////////////////////////////
// GaussianBlur
struct CV_GpuNppImageGaussianBlurTest : public CV_GpuNppFilterTest
{
    CV_GpuNppImageGaussianBlurTest() : CV_GpuNppFilterTest( "GPU-NppImageGaussianBlur", "GaussianBlur" ) {}

    int test(const Mat& img)
    {
        int ksizes[] = {3, 5, 7};
        int ksizes_num = sizeof(ksizes) / sizeof(int);

        int test_res = cvtest::TS::OK;

        const double sigma1 = 3.0;

        for (int i = 0; i < ksizes_num; ++i)
        {
            for (int j = 0; j < ksizes_num; ++j)
            {
                cv::Size ksize(ksizes[i], ksizes[j]);

                ts->printf(cvtest::TS::LOG, "ksize = (%dx%d)\t\n", ksizes[i], ksizes[j]);

                Mat cpudst;
                cv::GaussianBlur(img, cpudst, ksize, sigma1);

                GpuMat gpu1(img);
                GpuMat gpudst;
                cv::gpu::GaussianBlur(gpu1, gpudst, ksize, sigma1);

                if (CheckNorm(cpudst, gpudst, ksize) != cvtest::TS::OK)
                    test_res = cvtest::TS::FAIL_GENERIC;
            }
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Laplacian
struct CV_GpuNppImageLaplacianTest : public CV_GpuNppFilterTest
{
    CV_GpuNppImageLaplacianTest() : CV_GpuNppFilterTest( "GPU-NppImageLaplacian", "Laplacian" ) {}

    int test(const Mat& img)
    {
        int ksizes[] = {1, 3};
        int ksizes_num = sizeof(ksizes) / sizeof(int);

        int test_res = cvtest::TS::OK;

        for (int i = 0; i < ksizes_num; ++i)
        {
            ts->printf(cvtest::TS::LOG, "\nksize = %d\n", ksizes[i]);

            Mat cpudst;
            cv::Laplacian(img, cpudst, -1, ksizes[i]);

            GpuMat gpu1(img);
            GpuMat gpudst;
            cv::gpu::Laplacian(gpu1, gpudst, -1, ksizes[i]);

            if (CheckNorm(cpudst, gpudst, Size(3, 3)) != cvtest::TS::OK)
                test_res = cvtest::TS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Erode
class CV_GpuErodeTest : public CV_GpuNppFilterTest
{
public:
    CV_GpuErodeTest() : CV_GpuNppFilterTest( "GPU-NppErode", "erode" ) {} 

protected:
	virtual int test(const Mat& img)
    {
        Mat kernel(Mat::ones(3, 3, CV_8U));

	    cv::Mat cpuRes;
        cv::erode(img, cpuRes, kernel);

	    GpuMat gpuRes;
        cv::gpu::erode(GpuMat(img), gpuRes, kernel);

	    return CheckNorm(cpuRes, gpuRes, Size(3, 3));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Dilate
class CV_GpuDilateTest : public CV_GpuNppFilterTest
{
public:
    CV_GpuDilateTest() : CV_GpuNppFilterTest( "GPU-NppDilate", "dilate" ) {} 

protected:
	virtual int test(const Mat& img)
    {
        Mat kernel(Mat::ones(3, 3, CV_8U));

	    cv::Mat cpuRes;
        cv::dilate(img, cpuRes, kernel);

	    GpuMat gpuRes;
        cv::gpu::dilate(GpuMat(img), gpuRes, kernel);
	
	    return CheckNorm(cpuRes, gpuRes, Size(3, 3));
    }
};

////////////////////////////////////////////////////////////////////////////////
// MorphologyEx
class CV_GpuMorphExTest : public CV_GpuNppFilterTest
{
public:
    CV_GpuMorphExTest() : CV_GpuNppFilterTest( "GPU-NppMorphologyEx", "morphologyEx" ) {} 

protected:
	virtual int test(const Mat& img)
    {
        static int ops[] = { MORPH_OPEN, CV_MOP_CLOSE, CV_MOP_GRADIENT, CV_MOP_TOPHAT, CV_MOP_BLACKHAT};
        const char *names[] = { "MORPH_OPEN", "CV_MOP_CLOSE", "CV_MOP_GRADIENT", "CV_MOP_TOPHAT", "CV_MOP_BLACKHAT"};
        int num = sizeof(ops)/sizeof(ops[0]);

        GpuMat kernel(Mat::ones(3, 3, CV_8U));

        int res = cvtest::TS::OK;

        for(int i = 0; i < num; ++i)
        {
            ts->printf(cvtest::TS::LOG, "Tesing %s\n", names[i]);

	        cv::Mat cpuRes;
            cv::morphologyEx(img, cpuRes, ops[i], kernel);

	        GpuMat gpuRes;
            cv::gpu::morphologyEx(GpuMat(img), gpuRes, ops[i], kernel);

            if (cvtest::TS::OK != CheckNorm(cpuRes, gpuRes, Size(4, 4)))
                res = cvtest::TS::FAIL_GENERIC;
        }
        return res;
    }
};


TEST(blur, accuracy) { CV_GpuNppImageBlurTest test; test.safe_run(); }
TEST(sobel, accuracy) { CV_GpuNppImageSobelTest test; test.safe_run(); }
TEST(scharr, accuracy) { CV_GpuNppImageScharrTest test; test.safe_run(); }
TEST(gaussianBlur, accuracy) { CV_GpuNppImageGaussianBlurTest test; test.safe_run(); }
TEST(laplcaian, accuracy) { CV_GpuNppImageLaplacianTest test; test.safe_run(); }
TEST(erode, accuracy) { CV_GpuErodeTest test; test.safe_run(); }
TEST(dilate, accuracy) { CV_GpuDilateTest test; test.safe_run(); }
TEST(morphEx, accuracy) { CV_GpuMorphExTest test;  test.safe_run(); }
