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
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace gpu;

class CV_GpuImageProcTest : public CvTest
{
public:
    CV_GpuImageProcTest(const char* test_name, const char* test_funcs) : CvTest(test_name, test_funcs) {}
    virtual ~CV_GpuImageProcTest() {}

protected:
    void run(int);
    
    int test8UC1 (const Mat& img);
    int test8UC4 (const Mat& img);
    int test32SC1(const Mat& img);
    int test32FC1(const Mat& img);

    virtual int test(const Mat& img) = 0;

    int CheckNorm(const Mat& m1, const Mat& m2);
};


int CV_GpuImageProcTest::test8UC1(const Mat& img)
{
    cv::Mat img_C1;
    cvtColor(img, img_C1, CV_BGR2GRAY);

    return test(img_C1);
}

int CV_GpuImageProcTest::test8UC4(const Mat& img)
{
    cv::Mat img_C4;
    cvtColor(img, img_C4, CV_BGR2BGRA);

    return test(img_C4);
}

int CV_GpuImageProcTest::test32SC1(const Mat& img)
{
    cv::Mat img_C1;
    cvtColor(img, img_C1, CV_BGR2GRAY);    
    img_C1.convertTo(img_C1, CV_32S);

    return test(img_C1);
}

int CV_GpuImageProcTest::test32FC1(const Mat& img)
{
    cv::Mat temp, img_C1;
    img.convertTo(temp, CV_32F);
    cvtColor(temp, img_C1, CV_BGR2GRAY);

    return test(img_C1);
}

int CV_GpuImageProcTest::CheckNorm(const Mat& m1, const Mat& m2)
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

void CV_GpuImageProcTest::run( int )
{
    //load image
    cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");

    if (img.empty())
    {
        ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
        return;
    }

    int testResult = CvTS::OK;
    try
    {
        //run tests
        ts->printf(CvTS::LOG, "\n========Start test 8UC1========\n");
        if (test8UC1(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "\nSUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "\nFAIL\n");
            testResult = CvTS::FAIL_GENERIC;
        }

        ts->printf(CvTS::LOG, "\n========Start test 8UC4========\n");
        if (test8UC4(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "\nSUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "\nFAIL\n");
            testResult = CvTS::FAIL_GENERIC;
        }

        ts->printf(CvTS::LOG, "\n========Start test 32SC1========\n");
        if (test32SC1(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "\nSUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "\nFAIL\n");
            testResult = CvTS::FAIL_GENERIC;
        }

        ts->printf(CvTS::LOG, "\n========Start test 32FC1========\n");
        if (test32FC1(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "\nSUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "\nFAIL\n");
            testResult = CvTS::FAIL_GENERIC;
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
// threshold
struct CV_GpuNppImageThresholdTest : public CV_GpuImageProcTest
{
public:
    CV_GpuNppImageThresholdTest() : CV_GpuImageProcTest( "GPU-NppImageThreshold", "threshold" ) {}

    int test(const Mat& img)
    {
        if (img.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::RNG rng(*ts->get_rng());
        const double thresh = rng;

        cv::Mat cpuRes;
        cv::threshold(img, cpuRes, thresh, 0.0, THRESH_TRUNC);

        GpuMat gpu1(img);
        GpuMat gpuRes;
        cv::gpu::threshold(gpu1, gpuRes, thresh);

        return CheckNorm(cpuRes, gpuRes);
    }
};

////////////////////////////////////////////////////////////////////////////////
// resize
struct CV_GpuNppImageResizeTest : public CV_GpuImageProcTest
{
    CV_GpuNppImageResizeTest() : CV_GpuImageProcTest( "GPU-NppImageResize", "resize" ) {}
    int test(const Mat& img)
    {
        if (img.type() != CV_8UC1 && img.type() != CV_8UC4)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        int interpolations[] = {INTER_NEAREST, INTER_LINEAR, /*INTER_CUBIC,*/ /*INTER_LANCZOS4*/};
        const char* interpolations_str[] = {"INTER_NEAREST", "INTER_LINEAR", /*"INTER_CUBIC",*/ /*"INTER_LANCZOS4"*/};
        int interpolations_num = sizeof(interpolations) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < interpolations_num; ++i)
        {
            ts->printf(CvTS::LOG, "\nInterpolation type: %s\n", interpolations_str[i]);

            Mat cpu_res;
            cv::resize(img, cpu_res, Size(), 0.5, 0.5, interpolations[i]);

            GpuMat gpu1(img), gpu_res;
            cv::gpu::resize(gpu1, gpu_res, Size(), 0.5, 0.5, interpolations[i]);

            if (CheckNorm(cpu_res, gpu_res) != CvTS::OK)
                test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// copyMakeBorder
struct CV_GpuNppImageCopyMakeBorderTest : public CV_GpuImageProcTest
{
    CV_GpuNppImageCopyMakeBorderTest() : CV_GpuImageProcTest( "GPU-NppImageCopyMakeBorder", "copyMakeBorder" ) {}

    int test(const Mat& img)
    {
        if (img.type() != CV_8UC1 && img.type() != CV_8UC4 && img.type() != CV_32SC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        cv::RNG rng(*ts->get_rng());
        int top = rng.uniform(1, 10);
        int botton = rng.uniform(1, 10);
        int left = rng.uniform(1, 10);
        int right = rng.uniform(1, 10);
        cv::Scalar val(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        Mat cpudst;
        cv::copyMakeBorder(img, cpudst, top, botton, left, right, BORDER_CONSTANT, val);

        GpuMat gpu1(img);
        GpuMat gpudst;    
        cv::gpu::copyMakeBorder(gpu1, gpudst, top, botton, left, right, val);

        return CheckNorm(cpudst, gpudst);
    }
};

////////////////////////////////////////////////////////////////////////////////
// warpAffine
struct CV_GpuNppImageWarpAffineTest : public CV_GpuImageProcTest
{
    CV_GpuNppImageWarpAffineTest() : CV_GpuImageProcTest( "GPU-NppImageWarpAffine", "warpAffine" ) {}

    int test(const Mat& img)
    {
        if (img.type() == CV_32SC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }
        
        static const double coeffs[2][3] = 
        { 
            {cos(3.14 / 6), -sin(3.14 / 6), 100.0}, 
            {sin(3.14 / 6), cos(3.14 / 6), -100.0}
        };
        Mat M(2, 3, CV_64F, (void*)coeffs);

        int flags[] = {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_NEAREST | WARP_INVERSE_MAP, INTER_LINEAR | WARP_INVERSE_MAP, INTER_CUBIC | WARP_INVERSE_MAP};
        const char* flags_str[] = {"INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_NEAREST | WARP_INVERSE_MAP", "INTER_LINEAR | WARP_INVERSE_MAP", "INTER_CUBIC | WARP_INVERSE_MAP"};
        int flags_num = sizeof(flags) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < flags_num; ++i)
        {
            ts->printf(CvTS::LOG, "\nFlags: %s\n", flags_str[i]);

            Mat cpudst;
            cv::warpAffine(img, cpudst, M, img.size(), flags[i]);

            GpuMat gpu1(img);
            GpuMat gpudst;
            cv::gpu::warpAffine(gpu1, gpudst, M, gpu1.size(), flags[i]);
            
            if (CheckNorm(cpudst, gpudst) != CvTS::OK)
                test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// warpPerspective
struct CV_GpuNppImageWarpPerspectiveTest : public CV_GpuImageProcTest
{
    CV_GpuNppImageWarpPerspectiveTest() : CV_GpuImageProcTest( "GPU-NppImageWarpPerspective", "warpPerspective" ) {}


    int test(const Mat& img)
    {
        if (img.type() == CV_32SC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }
        
        static const double coeffs[3][3] = 
        {
            {cos(3.14 / 6), -sin(3.14 / 6), 100.0}, 
            {sin(3.14 / 6), cos(3.14 / 6), -100.0}, 
            {0.0, 0.0, 1.0}
        };
        Mat M(3, 3, CV_64F, (void*)coeffs);

        int flags[] = {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_NEAREST | WARP_INVERSE_MAP, INTER_LINEAR | WARP_INVERSE_MAP, INTER_CUBIC | WARP_INVERSE_MAP};
        const char* flags_str[] = {"INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_NEAREST | WARP_INVERSE_MAP", "INTER_LINEAR | WARP_INVERSE_MAP", "INTER_CUBIC | WARP_INVERSE_MAP"};
        int flags_num = sizeof(flags) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < flags_num; ++i)
        {
            ts->printf(CvTS::LOG, "\nFlags: %s\n", flags_str[i]);

            Mat cpudst;
            cv::warpPerspective(img, cpudst, M, img.size(), flags[i]);

            GpuMat gpu1(img);
            GpuMat gpudst;
            cv::gpu::warpPerspective(gpu1, gpudst, M, gpu1.size(), flags[i]);
            
            if (CheckNorm(cpudst, gpudst) != CvTS::OK)
                test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// integral
struct CV_GpuNppImageIntegralTest : public CV_GpuImageProcTest
{
    CV_GpuNppImageIntegralTest() : CV_GpuImageProcTest( "GPU-NppImageIntegral", "integral" ) {}

    int test(const Mat& img)
    {
        if (img.type() != CV_8UC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        Mat cpusum, cpusqsum;
        cv::integral(img, cpusum, cpusqsum, CV_32S);

        GpuMat gpu1(img);
        GpuMat gpusum, gpusqsum;
        cv::gpu::integral(gpu1, gpusum, gpusqsum);

        gpusqsum.convertTo(gpusqsum, CV_64F);

        int test_res = CvTS::OK;

        if (CheckNorm(cpusum, gpusum) != CvTS::OK)
        {
            ts->printf(CvTS::LOG, "\nSum failed\n");
            test_res = CvTS::FAIL_GENERIC;
        }
        if (CheckNorm(cpusqsum, gpusqsum) != CvTS::OK)
        {
            ts->printf(CvTS::LOG, "\nSquared sum failed\n");
            test_res = CvTS::FAIL_GENERIC;
        }

        return test_res;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Canny
struct CV_GpuNppImageCannyTest : public CV_GpuImageProcTest
{
    CV_GpuNppImageCannyTest() : CV_GpuImageProcTest( "GPU-NppImageCanny", "Canny" ) {}

    int test(const Mat& img)
    {
        if (img.type() != CV_8UC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        const double threshold1 = 1.0, threshold2 = 10.0;

        Mat cpudst;
        cv::Canny(img, cpudst, threshold1, threshold2);

        GpuMat gpu1(img);
        GpuMat gpudst;
        cv::gpu::Canny(gpu1, gpudst, threshold1, threshold2);

        return CheckNorm(cpudst, gpudst);
    }
};

////////////////////////////////////////////////////////////////////////////////
// cvtColor
class CV_GpuCvtColorTest : public CvTest
{
public:
    CV_GpuCvtColorTest() : CvTest("GPU-CvtColor", "cvtColor") {}
    ~CV_GpuCvtColorTest() {};

protected:
    void run(int);
    
    int CheckNorm(const Mat& m1, const Mat& m2);
};


int CV_GpuCvtColorTest::CheckNorm(const Mat& m1, const Mat& m2)
{
    double ret = norm(m1, m2, NORM_INF);

    if (ret <= 2)
    {
        return CvTS::OK;
    }
    else
    {
        ts->printf(CvTS::LOG, "\nNorm: %f\n", ret);
        return CvTS::FAIL_GENERIC;
    }
}

void CV_GpuCvtColorTest::run( int )
{
    cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");

    if (img.empty())
    {
        ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
        return;
    }

    int testResult = CvTS::OK;
    cv::Mat cpuRes;
    cv::gpu::GpuMat gpuImg(img), gpuRes;
    try
    {
        int codes[] = { CV_BGR2RGB, CV_RGB2BGRA, CV_BGRA2RGB,
                        CV_RGB2BGR555, CV_BGR5552BGR, CV_BGR2BGR565, CV_BGR5652RGB, 
                        CV_RGB2YCrCb, CV_YCrCb2BGR, CV_BGR2YUV, CV_YUV2RGB,
                        CV_RGB2XYZ, CV_XYZ2BGR, CV_BGR2XYZ, CV_XYZ2RGB,
                        CV_RGB2HSV, CV_HSV2BGR, CV_BGR2HSV_FULL, CV_HSV2RGB_FULL,
                        CV_RGB2HLS, CV_HLS2BGR, CV_BGR2HLS_FULL, CV_HLS2RGB_FULL,
                        CV_RGB2GRAY, CV_GRAY2BGRA, CV_BGRA2GRAY,
                        CV_GRAY2BGR555, CV_BGR5552GRAY, CV_GRAY2BGR565, CV_BGR5652GRAY};
        const char* codes_str[] = { "CV_BGR2RGB", "CV_RGB2BGRA", "CV_BGRA2RGB",
                                    "CV_RGB2BGR555", "CV_BGR5552BGR", "CV_BGR2BGR565", "CV_BGR5652RGB", 
                                    "CV_RGB2YCrCb", "CV_YCrCb2BGR", "CV_BGR2YUV", "CV_YUV2RGB",
                                    "CV_RGB2XYZ", "CV_XYZ2BGR", "CV_BGR2XYZ", "CV_XYZ2RGB",
                                    "CV_RGB2HSV", "CV_HSV2RGB", "CV_BGR2HSV_FULL", "CV_HSV2RGB_FULL",
                                    "CV_RGB2HLS", "CV_HLS2RGB", "CV_BGR2HLS_FULL", "CV_HLS2RGB_FULL",
                                    "CV_RGB2GRAY", "CV_GRAY2BGRA", "CV_BGRA2GRAY",
                                    "CV_GRAY2BGR555", "CV_BGR5552GRAY", "CV_GRAY2BGR565", "CV_BGR5652GRAY"};
        int codes_num = sizeof(codes) / sizeof(int);

        for (int i = 0; i < codes_num; ++i)
        {
            ts->printf(CvTS::LOG, "\n%s\n", codes_str[i]);

            cv::cvtColor(img, cpuRes, codes[i]);
            cv::gpu::cvtColor(gpuImg, gpuRes, codes[i]);

            if (CheckNorm(cpuRes, gpuRes) == CvTS::OK)
                ts->printf(CvTS::LOG, "\nSUCCESS\n");
            else
            {
                ts->printf(CvTS::LOG, "\nFAIL\n");
                testResult = CvTS::FAIL_GENERIC;
            }

            img = cpuRes;
            gpuImg = gpuRes;
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
// Histograms
class CV_GpuHistogramsTest : public CvTest
{
public:
    CV_GpuHistogramsTest() : CvTest("GPU-Histograms", "histEven") {}
    ~CV_GpuHistogramsTest() {};

protected:
    void run(int);

    int CheckNorm(const Mat& m1, const Mat& m2)
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
};

void CV_GpuHistogramsTest::run( int )
{
    //load image
    cv::Mat img = cv::imread(std::string(ts->get_data_path()) + "stereobp/aloe-L.png");

    if (img.empty())
    {
        ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
        return;
    }

    try
    {
        Mat hsv;
        cv::cvtColor(img, hsv, CV_BGR2HSV);

        int hbins = 30;
        int histSize[] = {hbins};

        float hranges[] = {0, 180};
        const float* ranges[] = {hranges};

        MatND hist;
        
        int channels[] = {0};
        calcHist(&hsv, 1, channels, Mat(), hist, 1, histSize, ranges);

        GpuMat gpuHsv(hsv);
        std::vector<GpuMat> srcs;
        cv::gpu::split(gpuHsv, srcs);
        GpuMat gpuHist;
        histEven(srcs[0], gpuHist, hbins, (int)hranges[0], (int)hranges[1]);

        Mat cpuHist = hist;
        cpuHist = cpuHist.t();
        cpuHist.convertTo(cpuHist, CV_32S);

        ts->set_failed_test_info(CheckNorm(cpuHist, gpuHist));
    }
    catch(const cv::Exception& e)
    {
        if (!check_and_treat_gpu_exception(e, ts))
            throw;
        return;
    }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////// tests registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// If we comment some tests, we may foget/miss to uncomment it after.
// Placing all test definitions in one place 
// makes us know about what tests are commented.

CV_GpuNppImageThresholdTest CV_GpuNppImageThreshold_test;
CV_GpuNppImageResizeTest CV_GpuNppImageResize_test;
CV_GpuNppImageCopyMakeBorderTest CV_GpuNppImageCopyMakeBorder_test;
CV_GpuNppImageWarpAffineTest CV_GpuNppImageWarpAffine_test;
CV_GpuNppImageWarpPerspectiveTest CV_GpuNppImageWarpPerspective_test;
CV_GpuNppImageIntegralTest CV_GpuNppImageIntegral_test;
CV_GpuNppImageCannyTest CV_GpuNppImageCanny_test;
CV_GpuCvtColorTest CV_GpuCvtColor_test;
CV_GpuHistogramsTest CV_GpuHistograms_test;