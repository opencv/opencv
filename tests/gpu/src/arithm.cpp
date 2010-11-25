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

using namespace cv;
using namespace std;
using namespace gpu;

#define CHECK(pred, err) if (!(pred)) { \
    ts->printf(CvTS::LOG, "Fail: \"%s\" at line: %d\n", #pred, __LINE__); \
    ts->set_failed_test_info(err); \
    return; }

class CV_GpuArithmTest : public CvTest
{
public:
    CV_GpuArithmTest(const char* test_name, const char* test_funcs) : CvTest(test_name, test_funcs) {}
    virtual ~CV_GpuArithmTest() {}

protected:
    void run(int);

    int test(int type);

    virtual int test(const Mat& mat1, const Mat& mat2) = 0;

    int CheckNorm(const Mat& m1, const Mat& m2);
    int CheckNorm(const Scalar& s1, const Scalar& s2);
    int CheckNorm(double d1, double d2);
};

int CV_GpuArithmTest::test(int type)
{
    cv::Size sz(200, 200);
    cv::Mat mat1(sz, type), mat2(sz, type);
    cv::RNG rng(*ts->get_rng());
    rng.fill(mat1, cv::RNG::UNIFORM, cv::Scalar::all(1), cv::Scalar::all(20));
    rng.fill(mat2, cv::RNG::UNIFORM, cv::Scalar::all(1), cv::Scalar::all(20));

    return test(mat1, mat2);
}

int CV_GpuArithmTest::CheckNorm(const Mat& m1, const Mat& m2)
{
    double ret = norm(m1, m2, NORM_INF);

    if (ret < 1e-5)
        return CvTS::OK;

    ts->printf(CvTS::LOG, "\nNorm: %f\n", ret);
    return CvTS::FAIL_GENERIC;
}

int CV_GpuArithmTest::CheckNorm(const Scalar& s1, const Scalar& s2)
{
    double ret0 = CheckNorm(s1[0], s2[0]), ret1 = CheckNorm(s1[1], s2[1]), ret2 = CheckNorm(s1[2], s2[2]), ret3 = CheckNorm(s1[3], s2[3]);

    return (ret0 == CvTS::OK && ret1 == CvTS::OK && ret2 == CvTS::OK && ret3 == CvTS::OK) ? CvTS::OK : CvTS::FAIL_GENERIC;
}

int CV_GpuArithmTest::CheckNorm(double d1, double d2)
{
    double ret = ::fabs(d1 - d2);

    if (ret < 1e-5)
        return CvTS::OK;

    ts->printf(CvTS::LOG, "\nNorm: %f\n", ret);
    return CvTS::FAIL_GENERIC;
}

void CV_GpuArithmTest::run( int )
{
    int testResult = CvTS::OK;
    try
    {
        const int types[] = {CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1};
        const char* type_names[] = {"CV_8UC1", "CV_8UC3", "CV_8UC4", "CV_32FC1"};
        const int type_count = sizeof(types)/sizeof(types[0]);

        //run tests
        for (int t = 0; t < type_count; ++t)
        {
            ts->printf(CvTS::LOG, "========Start test %s========\n", type_names[t]);

            if (CvTS::OK == test(types[t]))
                ts->printf(CvTS::LOG, "SUCCESS\n");
            else
            {
                ts->printf(CvTS::LOG, "FAIL\n");
                testResult = CvTS::FAIL_MISMATCH;
            }
        }
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;
        return;
    }

    ts->set_failed_test_info(testResult);
}

////////////////////////////////////////////////////////////////////////////////
// Add

struct CV_GpuNppImageAddTest : public CV_GpuArithmTest
{
    CV_GpuNppImageAddTest() : CV_GpuArithmTest( "GPU-NppImageAdd", "add" ) {}

        virtual int test(const Mat& mat1, const Mat& mat2)
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4 && mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::add(mat1, mat2, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuRes;
        cv::gpu::add(gpu1, gpu2, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Sub
struct CV_GpuNppImageSubtractTest : public CV_GpuArithmTest
{
    CV_GpuNppImageSubtractTest() : CV_GpuArithmTest( "GPU-NppImageSubtract", "subtract" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4 && mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::subtract(mat1, mat2, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuRes;
        cv::gpu::subtract(gpu1, gpu2, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// multiply
struct CV_GpuNppImageMultiplyTest : public CV_GpuArithmTest
{
    CV_GpuNppImageMultiplyTest() : CV_GpuArithmTest( "GPU-NppImageMultiply", "multiply" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4 && mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

	    cv::Mat cpuRes;
	    cv::multiply(mat1, mat2, cpuRes);

	    GpuMat gpu1(mat1);
	    GpuMat gpu2(mat2);
	    GpuMat gpuRes;
	    cv::gpu::multiply(gpu1, gpu2, gpuRes);

            return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// divide
struct CV_GpuNppImageDivideTest : public CV_GpuArithmTest
{
    CV_GpuNppImageDivideTest() : CV_GpuArithmTest( "GPU-NppImageDivide", "divide" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4 && mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

	    cv::Mat cpuRes;
	    cv::divide(mat1, mat2, cpuRes);

	    GpuMat gpu1(mat1);
	    GpuMat gpu2(mat2);
	    GpuMat gpuRes;
	    cv::gpu::divide(gpu1, gpu2, gpuRes);

            return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// transpose
struct CV_GpuNppImageTransposeTest : public CV_GpuArithmTest
{
    CV_GpuNppImageTransposeTest() : CV_GpuArithmTest( "GPU-NppImageTranspose", "transpose" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_8UC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::transpose(mat1, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpuRes;
        cv::gpu::transpose(gpu1, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// absdiff
struct CV_GpuNppImageAbsdiffTest : public CV_GpuArithmTest
{
    CV_GpuNppImageAbsdiffTest() : CV_GpuArithmTest( "GPU-NppImageAbsdiff", "absdiff" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4 && mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::absdiff(mat1, mat2, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuRes;
        cv::gpu::absdiff(gpu1, gpu2, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// compare
struct CV_GpuNppImageCompareTest : public CV_GpuArithmTest
{
    CV_GpuNppImageCompareTest() : CV_GpuArithmTest( "GPU-NppImageCompare", "compare" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        int cmp_codes[] = {CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE};
        const char* cmp_str[] = {"CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE"};
        int cmp_num = sizeof(cmp_codes) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < cmp_num; ++i)
        {
            ts->printf(CvTS::LOG, "\nCompare operation: %s\n", cmp_str[i]);

            cv::Mat cpuRes;
            cv::compare(mat1, mat2, cpuRes, cmp_codes[i]);

            GpuMat gpu1(mat1);
            GpuMat gpu2(mat2);
            GpuMat gpuRes;
            cv::gpu::compare(gpu1, gpu2, gpuRes, cmp_codes[i]);

            if (CheckNorm(cpuRes, gpuRes) != CvTS::OK)
                test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// meanStdDev
struct CV_GpuNppImageMeanStdDevTest : public CV_GpuArithmTest
{
    CV_GpuNppImageMeanStdDevTest() : CV_GpuArithmTest( "GPU-NppImageMeanStdDev", "meanStdDev" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_8UC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        Scalar cpumean;
        Scalar cpustddev;
        cv::meanStdDev(mat1, cpumean, cpustddev);

        GpuMat gpu1(mat1);
        Scalar gpumean;
        Scalar gpustddev;
        cv::gpu::meanStdDev(gpu1, gpumean, gpustddev);

        int test_res = CvTS::OK;

        if (CheckNorm(cpumean, gpumean) != CvTS::OK)
        {
            ts->printf(CvTS::LOG, "\nMean FAILED\n");
            test_res = CvTS::FAIL_GENERIC;
        }

        if (CheckNorm(cpustddev, gpustddev) != CvTS::OK)
        {
            ts->printf(CvTS::LOG, "\nStdDev FAILED\n");
            test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// norm
struct CV_GpuNppImageNormTest : public CV_GpuArithmTest
{
    CV_GpuNppImageNormTest() : CV_GpuArithmTest( "GPU-NppImageNorm", "norm" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_8UC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        int norms[] = {NORM_INF, NORM_L1, NORM_L2};
        const char* norms_str[] = {"NORM_INF", "NORM_L1", "NORM_L2"};
        int norms_num = sizeof(norms) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < norms_num; ++i)
        {
            ts->printf(CvTS::LOG, "\nNorm type: %s\n", norms_str[i]);

            double cpu_norm = cv::norm(mat1, mat2, norms[i]);

            GpuMat gpu1(mat1);
            GpuMat gpu2(mat2);
            double gpu_norm = cv::gpu::norm(gpu1, gpu2, norms[i]);

            if (CheckNorm(cpu_norm, gpu_norm) != CvTS::OK)
                test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// flip
struct CV_GpuNppImageFlipTest : public CV_GpuArithmTest
{
    CV_GpuNppImageFlipTest() : CV_GpuArithmTest( "GPU-NppImageFlip", "flip" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        int flip_codes[] = {0, 1, -1};
        const char* flip_axis[] = {"X", "Y", "Both"};
        int flip_codes_num = sizeof(flip_codes) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < flip_codes_num; ++i)
        {
            ts->printf(CvTS::LOG, "\nFlip Axis: %s\n", flip_axis[i]);

            Mat cpu_res;
            cv::flip(mat1, cpu_res, flip_codes[i]);

            GpuMat gpu1(mat1);
            GpuMat gpu_res;
            cv::gpu::flip(gpu1, gpu_res, flip_codes[i]);

            if (CheckNorm(cpu_res, gpu_res) != CvTS::OK)
                test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// sum
struct CV_GpuNppImageSumTest : public CV_GpuArithmTest
{
    CV_GpuNppImageSumTest() : CV_GpuArithmTest( "GPU-NppImageSum", "sum" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        Scalar cpures = cv::sum(mat1);

        GpuMat gpu1(mat1);
        Scalar gpures = cv::gpu::sum(gpu1);

        return CheckNorm(cpures, gpures);
    }
};

////////////////////////////////////////////////////////////////////////////////
// LUT
struct CV_GpuNppImageLUTTest : public CV_GpuArithmTest
{
    CV_GpuNppImageLUTTest() : CV_GpuArithmTest( "GPU-NppImageLUT", "LUT" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC3)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat lut(1, 256, CV_8UC1);
        cv::RNG rng(*ts->get_rng());
        rng.fill(lut, cv::RNG::UNIFORM, cv::Scalar::all(100), cv::Scalar::all(200));

        cv::Mat cpuRes;
        cv::LUT(mat1, lut, cpuRes);

        cv::gpu::GpuMat gpuRes;
        cv::gpu::LUT(GpuMat(mat1), lut, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// exp
struct CV_GpuNppImageExpTest : public CV_GpuArithmTest
{
    CV_GpuNppImageExpTest() : CV_GpuArithmTest( "GPU-NppImageExp", "exp" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::exp(mat1, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpuRes;
        cv::gpu::exp(gpu1, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// log
struct CV_GpuNppImageLogTest : public CV_GpuArithmTest
{
    CV_GpuNppImageLogTest() : CV_GpuArithmTest( "GPU-NppImageLog", "log" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::log(mat1, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpuRes;
        cv::gpu::log(gpu1, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// magnitude
struct CV_GpuNppImageMagnitudeTest : public CV_GpuArithmTest
{
    CV_GpuNppImageMagnitudeTest() : CV_GpuArithmTest( "GPU-NppImageMagnitude", "magnitude" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::magnitude(mat1, mat2, cpuRes);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuRes;
        cv::gpu::magnitude(gpu1, gpu2, gpuRes);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// phase
struct CV_GpuNppImagePhaseTest : public CV_GpuArithmTest
{
    CV_GpuNppImagePhaseTest() : CV_GpuArithmTest( "GPU-NppImagePhase", "phase" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuRes;
        cv::phase(mat1, mat2, cpuRes, true);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuRes;
        cv::gpu::phase(gpu1, gpu2, gpuRes, true);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// cartToPolar
struct CV_GpuNppImageCartToPolarTest : public CV_GpuArithmTest
{
    CV_GpuNppImageCartToPolarTest() : CV_GpuArithmTest( "GPU-NppImageCartToPolar", "cartToPolar" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuMag, cpuAngle;
        cv::cartToPolar(mat1, mat2, cpuMag, cpuAngle);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuMag, gpuAngle;
        cv::gpu::cartToPolar(gpu1, gpu2, gpuMag, gpuAngle);

        int magRes = CheckNorm(cpuMag, gpuMag);
        int angleRes = CheckNorm(cpuAngle, gpuAngle);

        return magRes == CvTS::OK && angleRes == CvTS::OK ? CvTS::OK : CvTS::FAIL_GENERIC;
    }
};

////////////////////////////////////////////////////////////////////////////////
// polarToCart
struct CV_GpuNppImagePolarToCartTest : public CV_GpuArithmTest
{
    CV_GpuNppImagePolarToCartTest() : CV_GpuArithmTest( "GPU-NppImagePolarToCart", "polarToCart" ) {}

    int test( const Mat& mat1, const Mat& mat2 )
    {
        if (mat1.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::Mat cpuX, cpuY;
        cv::polarToCart(mat1, mat2, cpuX, cpuY);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuX, gpuY;
        cv::gpu::polarToCart(gpu1, gpu2, gpuX, gpuY);

        int xRes = CheckNorm(cpuX, gpuX);
        int yRes = CheckNorm(cpuY, gpuY);

        return xRes == CvTS::OK && yRes == CvTS::OK ? CvTS::OK : CvTS::FAIL_GENERIC;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Min max

struct CV_GpuMinMaxTest: public CvTest
{
    CV_GpuMinMaxTest(): CvTest("GPU-MinMaxTest", "minMax") {}

    cv::gpu::GpuMat buf;

    void run(int)
    {
        int depth_end;
        int major, minor;
        cv::gpu::getComputeCapability(getDevice(), major, minor);
        minor = 0;
        if (minor >= 1) depth_end = CV_64F; else depth_end = CV_32F;

        for (int cn = 1; cn <= 4; ++cn)
            for (int depth = CV_8U; depth <= depth_end; ++depth)
            {
                int rows = 1, cols = 3;
                test(rows, cols, cn, depth);
                for (int i = 0; i < 4; ++i)
                {
                    int rows = 1 + rand() % 1000;
                    int cols = 1 + rand() % 1000;
                    test(rows, cols, cn, depth);
                }
            }
    }

    void test(int rows, int cols, int cn, int depth)
    {
        cv::Mat src(rows, cols, CV_MAKE_TYPE(depth, cn));
        cv::RNG rng;
        for (int i = 0; i < src.rows; ++i)
        { 
            Mat row(1, src.cols * src.elemSize(), CV_8U, src.ptr(i));
            rng.fill(row, RNG::UNIFORM, Scalar(0), Scalar(255));
        }

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        Mat src_ = src.reshape(1);
        if (depth != CV_8S)
        {
            cv::minMaxLoc(src_, &minVal, &maxVal, &minLoc, &maxLoc);
        }
        else 
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type 
            minVal = std::numeric_limits<double>::max();
            maxVal = std::numeric_limits<double>::min();
            for (int i = 0; i < src_.rows; ++i)
                for (int j = 0; j < src_.cols; ++j)
                {
                    char val = src_.at<char>(i, j);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
        }

        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;        
        cv::gpu::minMax(cv::gpu::GpuMat(src), &minVal_, &maxVal_, buf);
       
        if (abs(minVal - minVal_) > 1e-3f)
        {
            ts->printf(CvTS::CONSOLE, "\nfail: minVal=%f minVal_=%f rows=%d cols=%d depth=%d cn=%d\n", minVal, minVal_, rows, cols, depth, cn);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
        }
        if (abs(maxVal - maxVal_) > 1e-3f)
        {
            ts->printf(CvTS::CONSOLE, "\nfail: maxVal=%f maxVal_=%f rows=%d cols=%d depth=%d cn=%d\n", maxVal, maxVal_, rows, cols, depth, cn);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
        }
    }  
};


////////////////////////////////////////////////////////////////////////////////
// Min max loc

struct CV_GpuMinMaxLocTest: public CvTest
{
    CV_GpuMinMaxLocTest(): CvTest("GPU-MinMaxLocTest", "minMaxLoc") {}

    void run(int)
    {
        int depth_end;
        int major, minor;
        cv::gpu::getComputeCapability(getDevice(), major, minor);
        if (minor >= 1) depth_end = CV_64F; else depth_end = CV_32F;
        for (int depth = CV_8U; depth <= depth_end; ++depth)
        {
            int rows = 1, cols = 3;
            test(rows, cols, depth);
            for (int i = 0; i < 4; ++i)
            {
                int rows = 1 + rand() % 1000;
                int cols = 1 + rand() % 1000;
                test(rows, cols, depth);
            }
        }
    }

    void test(int rows, int cols, int depth)
    {
        cv::Mat src(rows, cols, depth);
        cv::RNG rng;
        for (int i = 0; i < src.rows; ++i)
        { 
            Mat row(1, src.cols * src.elemSize(), CV_8U, src.ptr(i));
            rng.fill(row, RNG::UNIFORM, Scalar(0), Scalar(255));
        }

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        if (depth != CV_8S)       
            cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
        else 
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type 
            minVal = std::numeric_limits<double>::max();
            maxVal = std::numeric_limits<double>::min();
            for (int i = 0; i < src.rows; ++i)
                for (int j = 0; j < src.cols; ++j)
                {
                    char val = src.at<char>(i, j);
                    if (val < minVal) { minVal = val; minLoc = cv::Point(j, i); }
                    if (val > maxVal) { maxVal = val; maxLoc = cv::Point(j, i); }
                }
        }

        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;        
        cv::gpu::minMaxLoc(cv::gpu::GpuMat(src), &minVal_, &maxVal_, &minLoc_, &maxLoc_);
       
        CHECK(minVal == minVal_, CvTS::FAIL_INVALID_OUTPUT);
        CHECK(maxVal == maxVal_, CvTS::FAIL_INVALID_OUTPUT);
        CHECK(0 == memcmp(src.ptr(minLoc.y) + minLoc.x * src.elemSize(), src.ptr(minLoc_.y) + minLoc_.x * src.elemSize(), src.elemSize()),  
              CvTS::FAIL_INVALID_OUTPUT);
        CHECK(0 == memcmp(src.ptr(maxLoc.y) + maxLoc.x * src.elemSize(), src.ptr(maxLoc_.y) + maxLoc_.x * src.elemSize(), src.elemSize()),  
              CvTS::FAIL_INVALID_OUTPUT);
    }  
};


/////////////////////////////////////////////////////////////////////////////
/////////////////// tests registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// If we comment some tests, we may foget/miss to uncomment it after.
// Placing all test definitions in one place
// makes us know about what tests are commented.

CV_GpuNppImageAddTest CV_GpuNppImageAdd_test;
CV_GpuNppImageSubtractTest CV_GpuNppImageSubtract_test;
CV_GpuNppImageMultiplyTest CV_GpuNppImageMultiply_test;
CV_GpuNppImageDivideTest CV_GpuNppImageDivide_test;
CV_GpuNppImageTransposeTest CV_GpuNppImageTranspose_test;
CV_GpuNppImageAbsdiffTest CV_GpuNppImageAbsdiff_test;
CV_GpuNppImageCompareTest CV_GpuNppImageCompare_test;
CV_GpuNppImageMeanStdDevTest CV_GpuNppImageMeanStdDev_test;
CV_GpuNppImageNormTest CV_GpuNppImageNorm_test;
CV_GpuNppImageFlipTest CV_GpuNppImageFlip_test;
CV_GpuNppImageSumTest CV_GpuNppImageSum_test;
CV_GpuNppImageLUTTest CV_GpuNppImageLUT_test;
CV_GpuNppImageExpTest CV_GpuNppImageExp_test;
CV_GpuNppImageLogTest CV_GpuNppImageLog_test;
CV_GpuNppImageMagnitudeTest CV_GpuNppImageMagnitude_test;
CV_GpuNppImagePhaseTest CV_GpuNppImagePhase_test;
CV_GpuNppImageCartToPolarTest CV_GpuNppImageCartToPolar_test;
CV_GpuNppImagePolarToCartTest CV_GpuNppImagePolarToCart_test;
CV_GpuMinMaxTest CV_GpuMinMaxTest_test;
CV_GpuMinMaxLocTest CV_GpuMinMaxLocTest_test;
