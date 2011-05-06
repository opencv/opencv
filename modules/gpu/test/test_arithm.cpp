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

#define CHECK(pred, err) if (!(pred)) { \
    ts->printf(cvtest::TS::CONSOLE, "Fail: \"%s\" at line: %d\n", #pred, __LINE__); \
    ts->set_failed_test_info(err); \
    return; }

class CV_GpuArithmTest : public cvtest::BaseTest
{
public:
    CV_GpuArithmTest(const char* /*test_name*/, const char* /*test_funcs*/){}
    virtual ~CV_GpuArithmTest() {}

protected:
    void run(int);

    int test(int type);

    virtual int test(const Mat& mat1, const Mat& mat2) = 0;

    int CheckNorm(const Mat& m1, const Mat& m2, double eps = 1e-5);
    int CheckNorm(const Scalar& s1, const Scalar& s2, double eps = 1e-5);
    int CheckNorm(double d1, double d2, double eps = 1e-5);
};

int CV_GpuArithmTest::test(int type)
{
    cv::Size sz(200, 200);
    cv::Mat mat1(sz, type), mat2(sz, type);
    
    cv::RNG& rng = ts->get_rng();

    if (type != CV_32FC1)
    {
        rng.fill(mat1, cv::RNG::UNIFORM, cv::Scalar::all(1), cv::Scalar::all(20));
        rng.fill(mat2, cv::RNG::UNIFORM, cv::Scalar::all(1), cv::Scalar::all(20));
    }
    else
    {
        rng.fill(mat1, cv::RNG::UNIFORM, cv::Scalar::all(0.1), cv::Scalar::all(1.0));
        rng.fill(mat2, cv::RNG::UNIFORM, cv::Scalar::all(0.1), cv::Scalar::all(1.0));
    }

    return test(mat1, mat2);
}

int CV_GpuArithmTest::CheckNorm(const Mat& m1, const Mat& m2, double eps)
{
    double ret = norm(m1, m2, NORM_INF);

    if (ret < eps)
        return cvtest::TS::OK;

    ts->printf(cvtest::TS::LOG, "\nNorm: %f\n", ret);
    return cvtest::TS::FAIL_GENERIC;
}

int CV_GpuArithmTest::CheckNorm(const Scalar& s1, const Scalar& s2, double eps)
{
    int ret0 = CheckNorm(s1[0], s2[0], eps), 
        ret1 = CheckNorm(s1[1], s2[1], eps), 
        ret2 = CheckNorm(s1[2], s2[2], eps), 
        ret3 = CheckNorm(s1[3], s2[3], eps);

    return (ret0 == cvtest::TS::OK && ret1 == cvtest::TS::OK && ret2 == cvtest::TS::OK && ret3 == cvtest::TS::OK) ? cvtest::TS::OK : cvtest::TS::FAIL_GENERIC;
}

int CV_GpuArithmTest::CheckNorm(double d1, double d2, double eps)
{
    double ret = ::fabs(d1 - d2);

    if (ret < eps)
        return cvtest::TS::OK;

    ts->printf(cvtest::TS::LOG, "\nNorm: %f\n", ret);
    return cvtest::TS::FAIL_GENERIC;
}

void CV_GpuArithmTest::run( int )
{
    int testResult = cvtest::TS::OK;

    const int types[] = {CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1};
    const char* type_names[] = {"CV_8UC1 ", "CV_8UC3 ", "CV_8UC4 ", "CV_32FC1"};
    const int type_count = sizeof(types)/sizeof(types[0]);

    //run tests
    for (int t = 0; t < type_count; ++t)
    {
        ts->printf(cvtest::TS::LOG, "Start testing %s", type_names[t]);

        if (cvtest::TS::OK == test(types[t]))
            ts->printf(cvtest::TS::LOG, "SUCCESS\n");
        else
        {
            ts->printf(cvtest::TS::LOG, "FAIL\n");
            testResult = cvtest::TS::FAIL_MISMATCH;
        }
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

	    cv::Mat cpuRes;
	    cv::divide(mat1, mat2, cpuRes);

	    GpuMat gpu1(mat1);
	    GpuMat gpu2(mat2);
	    GpuMat gpuRes;
	    cv::gpu::divide(gpu1, gpu2, gpuRes);

        return CheckNorm(cpuRes, gpuRes, 1.01f);
    }
};

////////////////////////////////////////////////////////////////////////////////
// transpose
struct CV_GpuNppImageTransposeTest : public CV_GpuArithmTest
{
    CV_GpuNppImageTransposeTest() : CV_GpuArithmTest( "GPU-NppImageTranspose", "transpose" ) {}

    int test( const Mat& mat1, const Mat& )
    {
        if (mat1.type() != CV_8UC1 && mat1.type() != CV_8UC4 && mat1.type() != CV_32FC1)
        {
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        int cmp_codes[] = {CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE};
        const char* cmp_str[] = {"CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE"};
        int cmp_num = sizeof(cmp_codes) / sizeof(int);

        int test_res = cvtest::TS::OK;

        for (int i = 0; i < cmp_num; ++i)
        {
            ts->printf(cvtest::TS::LOG, "\nCompare operation: %s\n", cmp_str[i]);

            cv::Mat cpuRes;
            cv::compare(mat1, mat2, cpuRes, cmp_codes[i]);

            GpuMat gpu1(mat1);
            GpuMat gpu2(mat2);
            GpuMat gpuRes;
            cv::gpu::compare(gpu1, gpu2, gpuRes, cmp_codes[i]);

            if (CheckNorm(cpuRes, gpuRes) != cvtest::TS::OK)
                test_res = cvtest::TS::FAIL_GENERIC;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        Scalar cpumean;
        Scalar cpustddev;
        cv::meanStdDev(mat1, cpumean, cpustddev);

        GpuMat gpu1(mat1);
        Scalar gpumean;
        Scalar gpustddev;
        cv::gpu::meanStdDev(gpu1, gpumean, gpustddev);

        int test_res = cvtest::TS::OK;

        if (CheckNorm(cpumean, gpumean) != cvtest::TS::OK)
        {
            ts->printf(cvtest::TS::LOG, "\nMean FAILED\n");
            test_res = cvtest::TS::FAIL_GENERIC;
        }

        if (CheckNorm(cpustddev, gpustddev) != cvtest::TS::OK)
        {
            ts->printf(cvtest::TS::LOG, "\nStdDev FAILED\n");
            test_res = cvtest::TS::FAIL_GENERIC;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        int norms[] = {NORM_INF, NORM_L1, NORM_L2};
        const char* norms_str[] = {"NORM_INF", "NORM_L1", "NORM_L2"};
        int norms_num = sizeof(norms) / sizeof(int);

        int test_res = cvtest::TS::OK;

        for (int i = 0; i < norms_num; ++i)
        {
            ts->printf(cvtest::TS::LOG, "\nNorm type: %s\n", norms_str[i]);

            double cpu_norm = cv::norm(mat1, mat2, norms[i]);

            GpuMat gpu1(mat1);
            GpuMat gpu2(mat2);
            double gpu_norm = cv::gpu::norm(gpu1, gpu2, norms[i]);

            if (CheckNorm(cpu_norm, gpu_norm) != cvtest::TS::OK)
                test_res = cvtest::TS::FAIL_GENERIC;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        int flip_codes[] = {0, 1, -1};
        const char* flip_axis[] = {"X", "Y", "Both"};
        int flip_codes_num = sizeof(flip_codes) / sizeof(int);

        int test_res = cvtest::TS::OK;

        for (int i = 0; i < flip_codes_num; ++i)
        {
            ts->printf(cvtest::TS::LOG, "\nFlip Axis: %s\n", flip_axis[i]);

            Mat cpu_res;
            cv::flip(mat1, cpu_res, flip_codes[i]);

            GpuMat gpu1(mat1);
            GpuMat gpu_res;
            cv::gpu::flip(gpu1, gpu_res, flip_codes[i]);

            if (CheckNorm(cpu_res, gpu_res) != cvtest::TS::OK)
                test_res = cvtest::TS::FAIL_GENERIC;
        }

        return test_res;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        cv::Mat lut(1, 256, CV_8UC1);
        cv::RNG& rng = ts->get_rng();
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        cv::Mat cpuRes;
        cv::phase(mat1, mat2, cpuRes, true);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuRes;
        cv::gpu::phase(gpu1, gpu2, gpuRes, true);

        return CheckNorm(cpuRes, gpuRes, 0.3f);
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        cv::Mat cpuMag, cpuAngle;
        cv::cartToPolar(mat1, mat2, cpuMag, cpuAngle);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuMag, gpuAngle;
        cv::gpu::cartToPolar(gpu1, gpu2, gpuMag, gpuAngle);

        int magRes = CheckNorm(cpuMag, gpuMag);
        int angleRes = CheckNorm(cpuAngle, gpuAngle, 0.005f);

        return magRes == cvtest::TS::OK && angleRes == cvtest::TS::OK ? cvtest::TS::OK : cvtest::TS::FAIL_GENERIC;
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
            ts->printf(cvtest::TS::LOG, "\tUnsupported type\t");
            return cvtest::TS::OK;
        }

        cv::Mat cpuX, cpuY;
        cv::polarToCart(mat1, mat2, cpuX, cpuY);

        GpuMat gpu1(mat1);
        GpuMat gpu2(mat2);
        GpuMat gpuX, gpuY;
        cv::gpu::polarToCart(gpu1, gpu2, gpuX, gpuY);

        int xRes = CheckNorm(cpuX, gpuX);
        int yRes = CheckNorm(cpuY, gpuY);

        return xRes == cvtest::TS::OK && yRes == cvtest::TS::OK ? cvtest::TS::OK : cvtest::TS::FAIL_GENERIC;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Min max

struct CV_GpuMinMaxTest: public cvtest::BaseTest
{
    CV_GpuMinMaxTest() {}

    cv::gpu::GpuMat buf;

    void run(int)
    {
        bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) &&
                         gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
        int depth_end = double_ok ? CV_64F : CV_32F;

        for (int depth = CV_8U; depth <= depth_end; ++depth)
        {
            for (int i = 0; i < 3; ++i)
            {
                int rows = 1 + rand() % 1000;
                int cols = 1 + rand() % 1000;
                test(rows, cols, 1, depth);
                test_masked(rows, cols, 1, depth);
            }
        }
    }

    void test(int rows, int cols, int cn, int depth)
    {
        cv::Mat src(rows, cols, CV_MAKE_TYPE(depth, cn));
        cv::RNG& rng = ts->get_rng();
        rng.fill(src, RNG::UNIFORM, Scalar(0), Scalar(255));

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        if (depth != CV_8S)
        {
            cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
        }
        else 
        {
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src.rows; ++i)
                for (int j = 0; j < src.cols; ++j)
                {
                    signed char val = src.at<signed char>(i, j);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
        }

        double minVal_, maxVal_;
        cv::gpu::minMax(cv::gpu::GpuMat(src), &minVal_, &maxVal_, cv::gpu::GpuMat(), buf);
       
        if (abs(minVal - minVal_) > 1e-3f)
        {
            ts->printf(cvtest::TS::CONSOLE, "\nfail: minVal=%f minVal_=%f rows=%d cols=%d depth=%d cn=%d\n", minVal, minVal_, rows, cols, depth, cn);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
        if (abs(maxVal - maxVal_) > 1e-3f)
        {
            ts->printf(cvtest::TS::CONSOLE, "\nfail: maxVal=%f maxVal_=%f rows=%d cols=%d depth=%d cn=%d\n", maxVal, maxVal_, rows, cols, depth, cn);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
    }  

    void test_masked(int rows, int cols, int cn, int depth)
    {
        cv::Mat src(rows, cols, CV_MAKE_TYPE(depth, cn));
        cv::RNG& rng = ts->get_rng();
        rng.fill(src, RNG::UNIFORM, Scalar(0), Scalar(255));

        cv::Mat mask(src.size(), CV_8U);
        rng.fill(mask, RNG::UNIFORM, Scalar(0), Scalar(2));

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        Mat src_ = src.reshape(1);
        if (depth != CV_8S)
        {
            cv::minMaxLoc(src_, &minVal, &maxVal, &minLoc, &maxLoc, mask);
        }
        else 
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type 
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src_.rows; ++i)
                for (int j = 0; j < src_.cols; ++j)
                {
                    char val = src_.at<char>(i, j);
                    if (mask.at<unsigned char>(i, j)) { if (val < minVal) minVal = val; }
                    if (mask.at<unsigned char>(i, j)) { if (val > maxVal) maxVal = val; }
                }
        }

        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;        
        cv::gpu::minMax(cv::gpu::GpuMat(src), &minVal_, &maxVal_, cv::gpu::GpuMat(mask), buf);
       
        if (abs(minVal - minVal_) > 1e-3f)
        {
            ts->printf(cvtest::TS::CONSOLE, "\nfail: minVal=%f minVal_=%f rows=%d cols=%d depth=%d cn=%d\n", minVal, minVal_, rows, cols, depth, cn);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
        if (abs(maxVal - maxVal_) > 1e-3f)
        {
            ts->printf(cvtest::TS::CONSOLE, "\nfail: maxVal=%f maxVal_=%f rows=%d cols=%d depth=%d cn=%d\n", maxVal, maxVal_, rows, cols, depth, cn);
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
    }  
};


////////////////////////////////////////////////////////////////////////////////
// Min max loc

struct CV_GpuMinMaxLocTest: public cvtest::BaseTest
{
    CV_GpuMinMaxLocTest() {}

    GpuMat valbuf, locbuf;

    void run(int)
    {
        bool double_ok = gpu::TargetArchs::builtWith(gpu::NATIVE_DOUBLE) &&
                         gpu::DeviceInfo().supports(gpu::NATIVE_DOUBLE);
        int depth_end = double_ok ? CV_64F : CV_32F;

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
        cv::RNG& rng = ts->get_rng();
        rng.fill(src, RNG::UNIFORM, Scalar(0), Scalar(255));

        cv::Mat mask(src.size(), CV_8U);
        rng.fill(mask, RNG::UNIFORM, Scalar(0), Scalar(2));

        // At least one of the mask elements must be non zero as OpenCV returns 0
        // in such case, when our implementation returns maximum or minimum value
        mask.at<unsigned char>(0, 0) = 1;

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        if (depth != CV_8S)       
            cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask);
        else 
        {
            // OpenCV's minMaxLoc doesn't support CV_8S type 
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src.rows; ++i)
                for (int j = 0; j < src.cols; ++j)
                {
                    char val = src.at<char>(i, j);
                    if (mask.at<unsigned char>(i, j))
                    {
                        if (val < minVal) { minVal = val; minLoc = cv::Point(j, i); }
                        if (val > maxVal) { maxVal = val; maxLoc = cv::Point(j, i); }
                    }
                }
        }

        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;        
        cv::gpu::minMaxLoc(cv::gpu::GpuMat(src), &minVal_, &maxVal_, &minLoc_, &maxLoc_, cv::gpu::GpuMat(mask), valbuf, locbuf);

        CHECK(minVal == minVal_, cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(maxVal == maxVal_, cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(0 == memcmp(src.ptr(minLoc.y) + minLoc.x * src.elemSize(), src.ptr(minLoc_.y) + minLoc_.x * src.elemSize(), src.elemSize()),  
              cvtest::TS::FAIL_INVALID_OUTPUT);
        CHECK(0 == memcmp(src.ptr(maxLoc.y) + maxLoc.x * src.elemSize(), src.ptr(maxLoc_.y) + maxLoc_.x * src.elemSize(), src.elemSize()),  
              cvtest::TS::FAIL_INVALID_OUTPUT);
    }  
};

////////////////////////////////////////////////////////////////////////////
// Count non zero
struct CV_GpuCountNonZeroTest: cvtest::BaseTest 
{
    CV_GpuCountNonZeroTest(){}

    void run(int) 
    {
        int depth_end;
        if (cv::gpu::DeviceInfo().supports(cv::gpu::NATIVE_DOUBLE))
            depth_end = CV_64F;
        else
            depth_end = CV_32F;
        for (int depth = CV_8U; depth <= CV_32F; ++depth)
        {
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
        if (depth == 5)
            rng.fill(src, RNG::UNIFORM, Scalar(-1000.f), Scalar(1000.f));
        else if (depth == 6)
            rng.fill(src, RNG::UNIFORM, Scalar(-1000.), Scalar(1000.));
        else
            for (int i = 0; i < src.rows; ++i)
            { 
                Mat row(1, src.cols * src.elemSize(), CV_8U, src.ptr(i));
                rng.fill(row, RNG::UNIFORM, Scalar(0), Scalar(256));
            }

        int n_gold = cv::countNonZero(src);
        int n = cv::gpu::countNonZero(cv::gpu::GpuMat(src));

        if (n != n_gold)
        {
            ts->printf(cvtest::TS::LOG, "%d %d %d %d %d\n", n, n_gold, depth, cols, rows);
            n_gold = cv::countNonZero(src);
        }

        CHECK(n == n_gold, cvtest::TS::FAIL_INVALID_OUTPUT);
    }
};


//////////////////////////////////////////////////////////////////////////////
// sum

struct CV_GpuSumTest: cvtest::BaseTest 
{
    CV_GpuSumTest() {}

    void run(int) 
    {
        Mat src;
        Scalar a, b;
        double max_err = 1e-5;

        int typemax = CV_32F;
        for (int type = CV_8U; type <= typemax; ++type)
        {
            //
            // sum
            //

            gen(1 + rand() % 500, 1 + rand() % 500, CV_MAKETYPE(type, 1), src);
            a = sum(src);
            b = sum(GpuMat(src));
            if (abs(a[0] - b[0]) > src.size().area() * max_err)
            {
                ts->printf(cvtest::TS::CONSOLE, "1 cols: %d, rows: %d, expected: %f, actual: %f\n", src.cols, src.rows, a[0], b[0]);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            gen(1 + rand() % 500, 1 + rand() % 500, CV_MAKETYPE(type, 2), src);
            a = sum(src);
            b = sum(GpuMat(src));
            if (abs(a[0] - b[0]) + abs(a[1] - b[1]) > src.size().area() * max_err)
            {
                ts->printf(cvtest::TS::CONSOLE, "2 cols: %d, rows: %d, expected: %f, actual: %f\n", src.cols, src.rows, a[1], b[1]);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            gen(1 + rand() % 500, 1 + rand() % 500, CV_MAKETYPE(type, 3), src);
            a = sum(src);
            b = sum(GpuMat(src));
            if (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])> src.size().area() * max_err)
            {
                ts->printf(cvtest::TS::CONSOLE, "3 cols: %d, rows: %d, expected: %f, actual: %f\n", src.cols, src.rows, a[2], b[2]);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            gen(1 + rand() % 500, 1 + rand() % 500, CV_MAKETYPE(type, 4), src);
            a = sum(src);
            b = sum(GpuMat(src));
            if (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])> src.size().area() * max_err)
            {
                ts->printf(cvtest::TS::CONSOLE, "4 cols: %d, rows: %d, expected: %f, actual: %f\n", src.cols, src.rows, a[3], b[3]);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            gen(1 + rand() % 500, 1 + rand() % 500, type, src);
            a = sum(src);
            b = sum(GpuMat(src));
            if (abs(a[0] - b[0]) > src.size().area() * max_err)
            {
                ts->printf(cvtest::TS::CONSOLE, "cols: %d, rows: %d, expected: %f, actual: %f\n", src.cols, src.rows, a[0], b[0]);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            //
            // absSum
            //

            gen(1 + rand() % 200, 1 + rand() % 200, CV_MAKETYPE(type, 1), src);
            b = absSum(GpuMat(src));
            a = norm(src, NORM_L1);
            if (abs(a[0] - b[0]) > src.size().area() * max_err)
            {
                ts->printf(cvtest::TS::CONSOLE, "type: %d, cols: %d, rows: %d, expected: %f, actual: %f\n", type, src.cols, src.rows, a[0], b[0]);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            //
            // sqrSum
            //

            if (type != CV_8S)
            {
                gen(1 + rand() % 200, 1 + rand() % 200, CV_MAKETYPE(type, 1), src);
                b = sqrSum(GpuMat(src));
                Mat sqrsrc;
                multiply(src, src, sqrsrc);
                a = sum(sqrsrc);
                if (abs(a[0] - b[0]) > src.size().area() * max_err)
                {
                    ts->printf(cvtest::TS::CONSOLE, "type: %d, cols: %d, rows: %d, expected: %f, actual: %f\n", type, src.cols, src.rows, a[0], b[0]);
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
                gen(1 + rand() % 200, 1 + rand() % 200, CV_MAKETYPE(type, 2), src);
                b = sqrSum(GpuMat(src));
                multiply(src, src, sqrsrc);
                a = sum(sqrsrc);
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])> src.size().area() * max_err * 2)
                {
                    ts->printf(cvtest::TS::CONSOLE, "type: %d, cols: %d, rows: %d, expected: %f, actual: %f\n", type, src.cols, src.rows, a[0], b[0]);
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
                gen(1 + rand() % 200, 1 + rand() % 200, CV_MAKETYPE(type, 3), src);
                b = sqrSum(GpuMat(src));
                multiply(src, src, sqrsrc);
                a = sum(sqrsrc);
                if (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])> src.size().area() * max_err * 3)
                {
                    ts->printf(cvtest::TS::CONSOLE, "type: %d, cols: %d, rows: %d, expected: %f, actual: %f\n", type, src.cols, src.rows, a[0], b[0]);
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
                gen(1 + rand() % 200, 1 + rand() % 200, CV_MAKETYPE(type, 4), src);
                b = sqrSum(GpuMat(src));
                multiply(src, src, sqrsrc);
                a = sum(sqrsrc);
                if (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3])> src.size().area() * max_err * 4)
                {
                    ts->printf(cvtest::TS::CONSOLE, "type: %d, cols: %d, rows: %d, expected: %f, actual: %f\n", type, src.cols, src.rows, a[0], b[0]);
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }
    }

    void gen(int cols, int rows, int type, Mat& m)
    {
        m.create(rows, cols, type);
        RNG rng;
        rng.fill(m, RNG::UNIFORM, Scalar::all(0), Scalar::all(16));

    }
};

TEST(add, accuracy) { CV_GpuNppImageAddTest test; test.safe_run(); }
TEST(subtract, accuracy) { CV_GpuNppImageSubtractTest test; test.safe_run(); }
TEST(multiply, accuracy) { CV_GpuNppImageMultiplyTest test; test.safe_run(); }
TEST(divide, accuracy) { CV_GpuNppImageDivideTest test; test.safe_run(); }
TEST(transpose, accuracy) { CV_GpuNppImageTransposeTest test; test.safe_run(); }
TEST(absdiff, accuracy) { CV_GpuNppImageAbsdiffTest test; test.safe_run(); }
TEST(compare, accuracy) { CV_GpuNppImageCompareTest test; test.safe_run(); }
TEST(meanStdDev, accuracy) { CV_GpuNppImageMeanStdDevTest test; test.safe_run(); }
TEST(normDiff, accuracy) { CV_GpuNppImageNormTest test; test.safe_run(); }
TEST(flip, accuracy) { CV_GpuNppImageFlipTest test; test.safe_run(); }
TEST(LUT, accuracy) { CV_GpuNppImageLUTTest test; test.safe_run(); }
TEST(exp, accuracy) { CV_GpuNppImageExpTest test; test.safe_run(); }
TEST(log, accuracy) { CV_GpuNppImageLogTest test; test.safe_run(); }
TEST(magnitude, accuracy) { CV_GpuNppImageMagnitudeTest test; test.safe_run(); }
TEST(phase, accuracy) { CV_GpuNppImagePhaseTest test; test.safe_run(); }
TEST(cartToPolar, accuracy) { CV_GpuNppImageCartToPolarTest test; test.safe_run(); }
TEST(polarToCart, accuracy) { CV_GpuNppImagePolarToCartTest test; test.safe_run(); }
TEST(minMax, accuracy) { CV_GpuMinMaxTest test; test.safe_run(); }
TEST(minMaxLoc, accuracy) { CV_GpuMinMaxLocTest test; test.safe_run(); }
TEST(countNonZero, accuracy) { CV_GpuCountNonZeroTest test; test.safe_run(); }
TEST(sum, accuracy) { CV_GpuSumTest test; test.safe_run(); }
