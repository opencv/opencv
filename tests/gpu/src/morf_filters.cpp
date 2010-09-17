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
#include "gputest.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuNppMorphogyTest : public CvTest
{
public:
    CV_GpuNppMorphogyTest(const char* test_name, const char* test_funcs) : CvTest(test_name, test_funcs) {}
    virtual ~CV_GpuNppMorphogyTest() {}

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
        
    int CheckNorm(const Mat& m1, const Mat& m2)
    {
        double res = norm(m1, m2, NORM_INF);

        if (res < std::numeric_limits<double>::epsilon())
            return CvTS::OK;
        
        ts->printf(CvTS::LOG, "\nNorm: %f\n", res);
        return CvTS::FAIL_GENERIC;
    }
};

void CV_GpuNppMorphogyTest::run( int )
{    
    cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");
    
    if (img.empty())
    {
        ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
        return;
    }

    try
    {
        //run tests
        int testResult = test8UC1(img);
        if (testResult != CvTS::OK)
        {
            ts->set_failed_test_info(testResult);
            return;
        }

        testResult = test8UC4(img);
        if (testResult != CvTS::OK)
        {
            ts->set_failed_test_info(testResult);
            return;
        }        
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;        
    }

    ts->set_failed_test_info(CvTS::OK);
}

////////////////////////////////////////////////////////////////////////////////
// Erode
class CV_GpuErodeTest : public CV_GpuNppMorphogyTest
{
public:
    CV_GpuErodeTest() : CV_GpuNppMorphogyTest( "GPU-NppErode", "erode" ) {} 

protected:
	virtual int test(const Mat& img)
    {
        Mat kernel(3, 3, CV_8U, Scalar(1));
        Point anchor(1,1);
        int iters = 3;

	    cv::Mat cpuRes;
        cv::erode(img, cpuRes, kernel, anchor, iters);

	    GpuMat gpuRes;
        cv::gpu::erode(GpuMat(img), gpuRes, kernel, anchor, iters);
	
	    return CheckNorm(cpuRes, gpuRes);
    }
};

CV_GpuErodeTest CV_GpuErode_test;

////////////////////////////////////////////////////////////////////////////////
// Dilate
class CV_GpuDilateTest : public CV_GpuNppMorphogyTest
{
public:
    CV_GpuDilateTest() : CV_GpuNppMorphogyTest( "GPU-NppDilate", "dilate" ) {} 

protected:
	virtual int test(const Mat& img)
    {
        Mat kernel(3, 3, CV_8U, Scalar(1));
        Point anchor(1,1);
        int iters = 3;

	    cv::Mat cpuRes;
        cv::dilate(img, cpuRes, kernel, anchor, iters);

	    GpuMat gpuRes;
        cv::gpu::dilate(GpuMat(img), gpuRes, kernel, anchor, iters);
	
	    return CheckNorm(cpuRes, gpuRes);
    }
};

CV_GpuDilateTest CV_GpuDilate_test;


////////////////////////////////////////////////////////////////////////////////
// Dilate
class CV_GpuMorphExTest : public CV_GpuNppMorphogyTest
{
public:
    CV_GpuMorphExTest() : CV_GpuNppMorphogyTest( "GPU-NppMorphologyEx", "dmorphologyExilate" ) {} 

protected:
	virtual int test(const Mat& img)
    {
        static int ops[] = { MORPH_OPEN, CV_MOP_CLOSE, CV_MOP_GRADIENT, CV_MOP_TOPHAT, CV_MOP_BLACKHAT};
        const char *names[] = { "MORPH_OPEN", "CV_MOP_CLOSE", "CV_MOP_GRADIENT", "CV_MOP_TOPHAT", "CV_MOP_BLACKHAT"};
        int num = sizeof(ops)/sizeof(ops[0]);

        Mat kernel(3, 3, CV_8U, Scalar(1));
        Point anchor(1,1);
        int iters = 3;

        for(int i = 0; i < num; ++i)
        {
            ts->printf(CvTS::LOG, "Tesing %s\n", names[i]);

	        cv::Mat cpuRes;
            cv::morphologyEx(img, cpuRes, ops[i], kernel, anchor, iters);

	        GpuMat gpuRes;
            cv::gpu::morphologyEx(GpuMat(img), gpuRes, ops[i], kernel, anchor, iters);
	
	        int res = CheckNorm(cpuRes, gpuRes);
            if (CvTS::OK != res)
                return res;
        }
        return CvTS::OK;
    }
};

CV_GpuMorphExTest CV_GpuMorphEx_test;
