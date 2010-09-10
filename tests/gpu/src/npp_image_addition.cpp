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
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuNppImageAdditionTest : public CvTest
{
public:
    CV_GpuNppImageAdditionTest();
    ~CV_GpuNppImageAdditionTest();

protected:
    void run(int);
    
    int test8UC1(const Mat& imgL, const Mat& imgR);
    int test8UC4(const Mat& imgL, const Mat& imgR);
    int test32FC1(const Mat& imgL, const Mat& imgR);

    int test(const Mat& imgL, const Mat& imgR);
    int CheckNorm(const Mat& m1, const Mat& m2);
};

CV_GpuNppImageAdditionTest::CV_GpuNppImageAdditionTest(): CvTest( "GPU-NppImageAddition", "add" )
{
}

CV_GpuNppImageAdditionTest::~CV_GpuNppImageAdditionTest() {}

int CV_GpuNppImageAdditionTest::test8UC1(const Mat& imgL, const Mat& imgR)
{
    cv::Mat imgL_C1;
    cv::Mat imgR_C1;
    cvtColor(imgL, imgL_C1, CV_BGR2GRAY);
    cvtColor(imgR, imgR_C1, CV_BGR2GRAY);

    return test(imgL_C1, imgR_C1);
}

int CV_GpuNppImageAdditionTest::test8UC4(const Mat& imgL, const Mat& imgR)
{
    cv::Mat imgL_C4;
    cv::Mat imgR_C4;
    cvtColor(imgL, imgL_C4, CV_BGR2BGRA);
    cvtColor(imgR, imgR_C4, CV_BGR2BGRA);

    return test(imgL_C4, imgR_C4);
}

int CV_GpuNppImageAdditionTest::test32FC1( const Mat& imgL, const Mat& imgR )
{
    cv::Mat imgL_C1;
    cv::Mat imgR_C1;
    cvtColor(imgL, imgL_C1, CV_BGR2GRAY);
    cvtColor(imgR, imgR_C1, CV_BGR2GRAY);
    
    imgL_C1.convertTo(imgL_C1, CV_32F);
    imgR_C1.convertTo(imgR_C1, CV_32F);

    return test(imgL_C1, imgR_C1);
}

int CV_GpuNppImageAdditionTest::test( const Mat& imgL, const Mat& imgR )
{
    cv::Mat cpuAdd;
    cv::add(imgL, imgR, cpuAdd);

    GpuMat gpuL(imgL);
    GpuMat gpuR(imgR);
    GpuMat gpuAdd;
    cv::gpu::add(gpuL, gpuR, gpuAdd);

    return CheckNorm(cpuAdd, gpuAdd);
}

int CV_GpuNppImageAdditionTest::CheckNorm(const Mat& m1, const Mat& m2)
{
    double ret = norm(m1, m2);

    if (ret < 1.0)
    {
        return CvTS::OK;
    }
    else
    {
        ts->printf(CvTS::LOG, "\nNorm: %f\n", ret);
        return CvTS::FAIL_GENERIC;
    }
}

void CV_GpuNppImageAdditionTest::run( int )
{
    //load images
    cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-L.png");
    cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-R.png");

    if (img_l.empty() || img_r.empty())
    {
        ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
        return;
    }

    //run tests
    int testResult = test8UC1(img_l, img_r);
    if (testResult != CvTS::OK)
    {
        ts->set_failed_test_info(testResult);
        return;
    }

    testResult = test8UC4(img_l, img_r);
    if (testResult != CvTS::OK)
    {
        ts->set_failed_test_info(testResult);
        return;
    }

    testResult = test32FC1(img_l, img_r);
    if (testResult != CvTS::OK)
    {
        ts->set_failed_test_info(testResult);
        return;
    }

    ts->set_failed_test_info(CvTS::OK);
}

CV_GpuNppImageAdditionTest CV_GpuNppImageAddition_test;