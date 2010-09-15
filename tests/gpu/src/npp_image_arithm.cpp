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

class CV_GpuNppImageArithmTest : public CvTest
{
public:
    CV_GpuNppImageArithmTest(const char* test_name, const char* test_funcs);
    virtual ~CV_GpuNppImageArithmTest();

protected:
    void run(int);
    
    int test8UC1(const Mat& cpu1, const Mat& cpu2);
    int test8UC4(const Mat& cpu1, const Mat& cpu2);
    int test32SC1(const Mat& cpu1, const Mat& cpu2);
    int test32FC1(const Mat& cpu1, const Mat& cpu2);

    virtual int test(const Mat& cpu1, const Mat& cpu2) = 0;
    int CheckNorm(const Mat& m1, const Mat& m2);
    int CheckNorm(const Scalar& s1, const Scalar& s2);
    int CheckNorm(double d1, double d2);
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

int CV_GpuNppImageArithmTest::test32SC1( const Mat& cpu1, const Mat& cpu2 )
{
    cv::Mat imgL_C1;
    cv::Mat imgR_C1;
    cvtColor(cpu1, imgL_C1, CV_BGR2GRAY);
    cvtColor(cpu2, imgR_C1, CV_BGR2GRAY);
    
    imgL_C1.convertTo(imgL_C1, CV_32S);
    imgR_C1.convertTo(imgR_C1, CV_32S);

    return test(imgL_C1, imgR_C1);
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
    double ret = norm(m1, m2, NORM_INF);

    if (ret < std::numeric_limits<double>::epsilon())
    {
        return CvTS::OK;
    }
    else
    {
        ts->printf(CvTS::LOG, "\nNorm: %f\n", ret);
        return CvTS::FAIL_GENERIC;
    }
}

int CV_GpuNppImageArithmTest::CheckNorm(const Scalar& s1, const Scalar& s2)
{
    double ret0 = CheckNorm(s1[0], s2[0]), ret1 = CheckNorm(s1[1], s2[1]), ret2 = CheckNorm(s1[2], s2[2]), ret3 = CheckNorm(s1[3], s2[3]);

    return (ret0 == CvTS::OK && ret1 == CvTS::OK && ret2 == CvTS::OK && ret3 == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

int CV_GpuNppImageArithmTest::CheckNorm(double d1, double d2)
{
    double ret = ::fabs(d1 - d2);

    if (ret < std::numeric_limits<double>::epsilon())
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
    //cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-L.png");
    //cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobm/aloe-R.png");

    cv::Mat img_l = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");
    cv::Mat img_r = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-R.png");

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

    testResult = test32SC1(img_l, img_r);
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
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4 && cpu1.type() != CV_32FC1)
        return CvTS::OK;

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
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4 && cpu1.type() != CV_32FC1)
        return CvTS::OK;

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
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4 && cpu1.type() != CV_32FC1)
        return CvTS::OK;

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
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4 && cpu1.type() != CV_32FC1)
        return CvTS::OK;

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
    if (cpu1.type() != CV_8UC1)
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
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_32FC1)
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
    if (cpu1.type() != CV_32FC1)
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

    return (CheckNorm(cpumean, gpumean) == CvTS::OK && CheckNorm(cpustddev, gpustddev) == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
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

    return (CheckNorm(cpu_norm_inf, gpu_norm_inf) == CvTS::OK  
            && CheckNorm(cpu_norm_L1, gpu_norm_L1) == CvTS::OK 
            && CheckNorm(cpu_norm_L2, gpu_norm_L2) == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

CV_GpuNppImageNormTest CV_GpuNppImageNorm_test;

////////////////////////////////////////////////////////////////////////////////
// flip
class CV_GpuNppImageFlipTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageFlipTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageFlipTest::CV_GpuNppImageFlipTest(): CV_GpuNppImageArithmTest( "GPU-NppImageFlip", "flip" )
{
}

int CV_GpuNppImageFlipTest::test( const Mat& cpu1, const Mat& )
{
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4)
        return CvTS::OK;

    Mat cpux, cpuy, cpub;
    cv::flip(cpu1, cpux, 0);
    cv::flip(cpu1, cpuy, 1);
    cv::flip(cpu1, cpub, -1);

    GpuMat gpu1(cpu1);
    GpuMat gpux, gpuy, gpub;
    cv::gpu::flip(gpu1, gpux, 0);
    cv::gpu::flip(gpu1, gpuy, 1);
    cv::gpu::flip(gpu1, gpub, -1);

    return (CheckNorm(cpux, gpux) == CvTS::OK && 
            CheckNorm(cpuy, gpuy) == CvTS::OK && 
            CheckNorm(cpub, gpub) == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

CV_GpuNppImageFlipTest CV_GpuNppImageFlip_test;

////////////////////////////////////////////////////////////////////////////////
// resize
class CV_GpuNppImageResizeTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageResizeTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageResizeTest::CV_GpuNppImageResizeTest(): CV_GpuNppImageArithmTest( "GPU-NppImageResize", "resize" )
{
}

int CV_GpuNppImageResizeTest::test( const Mat& cpu1, const Mat& )
{
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4)
        return CvTS::OK;

    Mat cpunn, cpulin, cpucub, cpulanc;
    cv::resize(cpu1, cpunn, Size(), 0.5, 0.5, INTER_NEAREST);
    cv::resize(cpu1, cpulin, Size(), 0.5, 0.5, INTER_LINEAR);
    cv::resize(cpu1, cpucub, Size(), 0.5, 0.5, INTER_CUBIC);
    cv::resize(cpu1, cpulanc, Size(), 0.5, 0.5, INTER_LANCZOS4);

    GpuMat gpu1(cpu1);
    GpuMat gpunn, gpulin, gpucub, gpulanc;
    cv::gpu::resize(gpu1, gpunn, Size(), 0.5, 0.5, INTER_NEAREST);
    cv::gpu::resize(gpu1, gpulin, Size(), 0.5, 0.5, INTER_LINEAR);
    cv::gpu::resize(gpu1, gpucub, Size(), 0.5, 0.5, INTER_CUBIC);
    cv::gpu::resize(gpu1, gpulanc, Size(), 0.5, 0.5, INTER_LANCZOS4);

    int nnres =CheckNorm(cpunn, gpunn);
    int linres = CheckNorm(cpulin, gpulin);
    int cubres = CheckNorm(cpucub, gpucub);
    int lancres = CheckNorm(cpulanc, gpulanc);

    return (nnres == CvTS::OK && linres == CvTS::OK && cubres == CvTS::OK && lancres == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

CV_GpuNppImageResizeTest CV_GpuNppImageResize_test;

////////////////////////////////////////////////////////////////////////////////
// sum
class CV_GpuNppImageSumTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageSumTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageSumTest::CV_GpuNppImageSumTest(): CV_GpuNppImageArithmTest( "GPU-NppImageSum", "sum" )
{
}

int CV_GpuNppImageSumTest::test( const Mat& cpu1, const Mat& )
{
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4)
        return CvTS::OK;

    Scalar cpures = cv::sum(cpu1);

    GpuMat gpu1(cpu1);
    Scalar gpures = cv::gpu::sum(gpu1);

    return CheckNorm(cpures, gpures);
}

CV_GpuNppImageSumTest CV_GpuNppImageSum_test;

////////////////////////////////////////////////////////////////////////////////
// minNax
class CV_GpuNppImageMinNaxTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageMinNaxTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageMinNaxTest::CV_GpuNppImageMinNaxTest(): CV_GpuNppImageArithmTest( "GPU-NppImageMinNax", "minNax" )
{
}

int CV_GpuNppImageMinNaxTest::test( const Mat& cpu1, const Mat& )
{
    if (cpu1.type() != CV_8UC1)
        return CvTS::OK;

    double cpumin, cpumax;
    cv::minMaxLoc(cpu1, &cpumin, &cpumax);

    GpuMat gpu1(cpu1);
    double gpumin, gpumax;
    cv::gpu::minMax(gpu1, &gpumin, &gpumax);

    return (CheckNorm(cpumin, gpumin) == CvTS::OK && CheckNorm(cpumax, gpumax) == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

CV_GpuNppImageMinNaxTest CV_GpuNppImageMinNax_test;

////////////////////////////////////////////////////////////////////////////////
// copyConstBorder
class CV_GpuNppImageCopyConstBorderTest : public CV_GpuNppImageArithmTest
{
public:
    CV_GpuNppImageCopyConstBorderTest();

protected:
    virtual int test(const Mat& cpu1, const Mat& cpu2);
};

CV_GpuNppImageCopyConstBorderTest::CV_GpuNppImageCopyConstBorderTest(): CV_GpuNppImageArithmTest( "GPU-NppImageCopyConstBorder", "copyConstBorder" )
{
}

int CV_GpuNppImageCopyConstBorderTest::test( const Mat& cpu1, const Mat& )
{
    if (cpu1.type() != CV_8UC1 && cpu1.type() != CV_8UC4 && cpu1.type() != CV_32SC1)
        return CvTS::OK;

    Mat cpudst;
    cv::copyMakeBorder(cpu1, cpudst, 5, 5, 5, 5, BORDER_CONSTANT);

    GpuMat gpu1(cpu1);
    GpuMat gpudst;    
    cv::gpu::copyConstBorder(gpu1, gpudst, 5, 5, 5, 5);

    return CheckNorm(cpudst, gpudst);
}

CV_GpuNppImageCopyConstBorderTest CV_GpuNppImageCopyConstBorder_test;