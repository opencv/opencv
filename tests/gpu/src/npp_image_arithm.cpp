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
#include "gputest.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuNppImageArithmTest : public CvTest
{
public:
    CV_GpuNppImageArithmTest(const char* test_name, const char* test_funcs);
    virtual ~CV_GpuNppImageArithmTest();

protected:
    void run(int);
    
    int test8UC1(const Mat& cpu1, const Mat& cpu2);
    int test8UC4(const Mat& cpu1, const Mat& cpu2);
    int test32FC1(const Mat& cpu1, const Mat& cpu2);

    virtual int test(const Mat& cpu1, const Mat& cpu2) = 0;
    int CheckNorm(const Mat& m1, const Mat& m2);
};

CV_GpuNppImageArithmTest::CV_GpuNppImageArithmTest(const char* test_name, const char* test_funcs): CvTest(test_name, test_funcs)
{
}

CV_GpuNppImageArithmTest::~CV_GpuNppImageArithmTest() {}

int CV_GpuNppImageArithmTest::test8UC1(const Mat& cpu1, const Mat& cpu2)
{
    cv::Mat imgL_C1;
    cv::Mat imgR_C1;
    cvtColor(cpu1, imgL_C1, CV_BGR2GRAY);
    cvtColor(cpu2, imgR_C1, CV_BGR2GRAY);

    return test(imgL_C1, imgR_C1);
}

int CV_GpuNppImageArithmTest::test8UC4(const Mat& cpu1, const Mat& cpu2)
{
    cv::Mat imgL_C4;
    cv::Mat imgR_C4;
    cvtColor(cpu1, imgL_C4, CV_BGR2BGRA);
    cvtColor(cpu2, imgR_C4, CV_BGR2BGRA);

    return test(imgL_C4, imgR_C4);
}

int CV_GpuNppImageArithmTest::test32FC1( const Mat& cpu1, const Mat& cpu2 )
{
    cv::Mat imgL_C1;
    cv::Mat imgR_C1;
    cvtColor(cpu1, imgL_C1, CV_BGR2GRAY);
    cvtColor(cpu2, imgR_C1, CV_BGR2GRAY);
    
    imgL_C1.convertTo(imgL_C1, CV_32F);
    imgR_C1.convertTo(imgR_C1, CV_32F);

    return test(imgL_C1, imgR_C1);
}

int CV_GpuNppImageArithmTest::CheckNorm(const Mat& m1, const Mat& m2)
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

void CV_GpuNppImageArithmTest::run( int )
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

////////////////////////////////////////////////////////////////////////////////
// Add
class CV_GpuNppImageAddTest : public CV_GpuNppImageArithmTest
{
public:
	CV_GpuNppImageAddTest();

protected:
	virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageAddTest::CV_GpuNppImageAddTest(): CV_GpuNppImageArithmTest( "GPU-NppImageAdd", "add" )
{
}

int CV_GpuNppImageAddTest::test( const Mat& cpu1, const Mat& cpu2 )
{
	cv::Mat cpuRes;
	cv::add(cpu1, cpu2, cpuRes);

	GpuMat gpu1(cpu1);
	GpuMat gpu2(cpu2);
	GpuMat gpuRes;
	cv::gpu::add(gpu1, gpu2, gpuRes);

	return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageAddTest CV_GpuNppImageAdd_test;

////////////////////////////////////////////////////////////////////////////////
// Sub
class CV_GpuNppImageSubtractTest : public CV_GpuNppImageArithmTest
{
public:
	CV_GpuNppImageSubtractTest();

protected:
	virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageSubtractTest::CV_GpuNppImageSubtractTest(): CV_GpuNppImageArithmTest( "GPU-NppImageSubtract", "subtract" )
{
}

int CV_GpuNppImageSubtractTest::test( const Mat& cpu1, const Mat& cpu2 )
{
	cv::Mat cpuRes;
	cv::subtract(cpu1, cpu2, cpuRes);

	GpuMat gpu1(cpu1);
	GpuMat gpu2(cpu2);
	GpuMat gpuRes;
	cv::gpu::subtract(gpu1, gpu2, gpuRes);    

	return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageSubtractTest CV_GpuNppImageSubtract_test;

////////////////////////////////////////////////////////////////////////////////
// multiply
class CV_GpuNppImageMultiplyTest : public CV_GpuNppImageArithmTest
{
public:
	CV_GpuNppImageMultiplyTest();

protected:
	virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageMultiplyTest::CV_GpuNppImageMultiplyTest(): CV_GpuNppImageArithmTest( "GPU-NppImageMultiply", "multiply" )
{
}

int CV_GpuNppImageMultiplyTest::test( const Mat& cpu1, const Mat& cpu2 )
{
	cv::Mat cpuRes;
	cv::multiply(cpu1, cpu2, cpuRes);

	GpuMat gpu1(cpu1);
	GpuMat gpu2(cpu2);
	GpuMat gpuRes;
	cv::gpu::multiply(gpu1, gpu2, gpuRes);

	return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageMultiplyTest CV_GpuNppImageMultiply_test;

////////////////////////////////////////////////////////////////////////////////
// divide
class CV_GpuNppImageDivideTest : public CV_GpuNppImageArithmTest
{
public:
	CV_GpuNppImageDivideTest();

protected:
	virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageDivideTest::CV_GpuNppImageDivideTest(): CV_GpuNppImageArithmTest( "GPU-NppImageDivide", "divide" )
{
}

int CV_GpuNppImageDivideTest::test( const Mat& cpu1, const Mat& cpu2 )
{
	cv::Mat cpuRes;
	cv::divide(cpu1, cpu2, cpuRes);

	GpuMat gpu1(cpu1);
	GpuMat gpu2(cpu2);
	GpuMat gpuRes;
	cv::gpu::divide(gpu1, gpu2, gpuRes);

	return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageDivideTest CV_GpuNppImageDivide_test;

////////////////////////////////////////////////////////////////////////////////
// transpose
class CV_GpuNppImageTransposeTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageTransposeTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageTransposeTest::CV_GpuNppImageTransposeTest(): CV_GpuNppImageArithmTest( "GPU-NppImageTranspose", "transpose" )
{
}

int CV_GpuNppImageTransposeTest::test( const Mat& cpu1, const Mat& )
{
    if (!((cpu1.depth() == CV_8U) && cpu1.channels() == 1))
        return CvTS::OK;

    cv::Mat cpuRes;
    cv::transpose(cpu1, cpuRes);

    GpuMat gpu1(cpu1);
    GpuMat gpuRes;
    cv::gpu::transpose(gpu1, gpuRes);

    return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageTransposeTest CV_GpuNppImageTranspose_test;

////////////////////////////////////////////////////////////////////////////////
// absdiff
class CV_GpuNppImageAbsdiffTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageAbsdiffTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageAbsdiffTest::CV_GpuNppImageAbsdiffTest(): CV_GpuNppImageArithmTest( "GPU-NppImageAbsdiff", "absdiff" )
{
}

int CV_GpuNppImageAbsdiffTest::test( const Mat& cpu1, const Mat& cpu2 )
{
    if (!((cpu1.depth() == CV_8U || cpu1.depth() == CV_32F) && cpu1.channels() == 1))
        return CvTS::OK;

    cv::Mat cpuRes;
    cv::absdiff(cpu1, cpu2, cpuRes);

    GpuMat gpu1(cpu1);
    GpuMat gpu2(cpu2);
    GpuMat gpuRes;
    cv::gpu::absdiff(gpu1, gpu2, gpuRes);

    return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageAbsdiffTest CV_GpuNppImageAbsdiff_test;

////////////////////////////////////////////////////////////////////////////////
// threshold
class CV_GpuNppImageThresholdTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageThresholdTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageThresholdTest::CV_GpuNppImageThresholdTest(): CV_GpuNppImageArithmTest( "GPU-NppImageThreshold", "threshold" )
{
}

int CV_GpuNppImageThresholdTest::test( const Mat& cpu1, const Mat& )
{
    if (!((cpu1.depth() == CV_32F) && cpu1.channels() == 1))
        return CvTS::OK;

    const double thresh = 0.5;
    const double maxval = 0.0;

    cv::Mat cpuRes;
    cv::threshold(cpu1, cpuRes, thresh, maxval, THRESH_TRUNC);

    GpuMat gpu1(cpu1);
    GpuMat gpuRes;
    cv::gpu::threshold(gpu1, gpuRes, thresh, maxval, THRESH_TRUNC);

    return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageThresholdTest CV_GpuNppImageThreshold_test;

////////////////////////////////////////////////////////////////////////////////
// compare
class CV_GpuNppImageCompareTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageCompareTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageCompareTest::CV_GpuNppImageCompareTest(): CV_GpuNppImageArithmTest( "GPU-NppImageCompare", "compare" )
{
}

int CV_GpuNppImageCompareTest::test( const Mat& cpu1, const Mat& cpu2 )
{
    if (cpu1.type() != CV_32FC1)
        return CvTS::OK;

    cv::Mat cpuRes;
    cv::compare(cpu1, cpu2, cpuRes, CMP_GT);

    GpuMat gpu1(cpu1);
    GpuMat gpu2(cpu2);
    GpuMat gpuRes;
    cv::gpu::compare(gpu1, gpu2, gpuRes, CMP_GT);

    return CheckNorm(cpuRes, gpuRes);
}

CV_GpuNppImageCompareTest CV_GpuNppImageCompare_test;

////////////////////////////////////////////////////////////////////////////////
// meanStdDev
class CV_GpuNppImageMeanStdDevTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageMeanStdDevTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageMeanStdDevTest::CV_GpuNppImageMeanStdDevTest(): CV_GpuNppImageArithmTest( "GPU-NppImageMeanStdDev", "meanStdDev" )
{
}

int CV_GpuNppImageMeanStdDevTest::test( const Mat& cpu1, const Mat& )
{
    if (cpu1.type() != CV_8UC1)
        return CvTS::OK;

    Scalar cpumean; 
    Scalar cpustddev;
    cv::meanStdDev(cpu1, cpumean, cpustddev);

    GpuMat gpu1(cpu1);
    Scalar gpumean; 
    Scalar gpustddev;
    cv::gpu::meanStdDev(gpu1, gpumean, gpustddev);

    return (cpumean == gpumean && cpustddev == gpustddev) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

CV_GpuNppImageMeanStdDevTest CV_GpuNppImageMeanStdDev_test;

////////////////////////////////////////////////////////////////////////////////
// norm
class CV_GpuNppImageNormTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageNormTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageNormTest::CV_GpuNppImageNormTest(): CV_GpuNppImageArithmTest( "GPU-NppImageNorm", "norm" )
{
}

int CV_GpuNppImageNormTest::test( const Mat& cpu1, const Mat& cpu2 )
{
    if (cpu1.type() != CV_8UC1)
        return CvTS::OK;

    double cpu_norm_inf = cv::norm(cpu1, cpu2, NORM_INF);
    double cpu_norm_L1 = cv::norm(cpu1, cpu2, NORM_L1);
    double cpu_norm_L2 = cv::norm(cpu1, cpu2, NORM_L2);

    GpuMat gpu1(cpu1);
    GpuMat gpu2(cpu2);
    double gpu_norm_inf = cv::gpu::norm(gpu1, gpu2, NORM_INF);
    double gpu_norm_L1 = cv::gpu::norm(gpu1, gpu2, NORM_L1);
    double gpu_norm_L2 = cv::gpu::norm(gpu1, gpu2, NORM_L2);

    return (cpu_norm_inf == gpu_norm_inf && cpu_norm_L1 == gpu_norm_L1 && cpu_norm_L2 == gpu_norm_L2) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

CV_GpuNppImageNormTest CV_GpuNppImageNorm_test;