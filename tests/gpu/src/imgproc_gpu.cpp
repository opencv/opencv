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

    // Checks whether two images are similar enough using normalized
    // cross-correlation as an error measure
    int CheckSimilarity(const Mat& m1, const Mat& m2, float max_err=1e-3f);
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
    img.convertTo(temp, CV_32F, 1.f / 255.f);
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
        ts->printf(CvTS::LOG, "Norm: %f\n", ret);
        return CvTS::FAIL_GENERIC;
    }
}

int CV_GpuImageProcTest::CheckSimilarity(const Mat& m1, const Mat& m2, float max_err)
{
    Mat diff;
    cv::matchTemplate(m1, m2, diff, CV_TM_CCORR_NORMED);

    float err = abs(diff.at<float>(0, 0) - 1.f);

    if (err > max_err)
        return CvTS::FAIL_INVALID_OUTPUT;

    return CvTS::OK;
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
            ts->printf(CvTS::LOG, "SUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "FAIL\n");
            testResult = CvTS::FAIL_GENERIC;
        }

        ts->printf(CvTS::LOG, "\n========Start test 8UC4========\n");
        if (test8UC4(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "SUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "FAIL\n");
            testResult = CvTS::FAIL_GENERIC;
        }

        ts->printf(CvTS::LOG, "\n========Start test 32SC1========\n");
        if (test32SC1(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "SUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "FAIL\n");
            testResult = CvTS::FAIL_GENERIC;
        }

        ts->printf(CvTS::LOG, "\n========Start test 32FC1========\n");
        if (test32FC1(img) == CvTS::OK)
            ts->printf(CvTS::LOG, "SUCCESS\n");
        else
        {
            ts->printf(CvTS::LOG, "FAIL\n");
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
struct CV_GpuImageThresholdTest : public CV_GpuImageProcTest
{
public:
    CV_GpuImageThresholdTest() : CV_GpuImageProcTest( "GPU-ImageThreshold", "threshold" ) {}

    int test(const Mat& img)
    {
        if (img.type() != CV_8UC1 && img.type() != CV_32FC1)
        {
            ts->printf(CvTS::LOG, "\nUnsupported type\n");
            return CvTS::OK;
        }

        const double maxVal = img.type() == CV_8UC1 ? 255 : 1.0;

        cv::RNG rng(*ts->get_rng());

        int res = CvTS::OK;

        for (int type = THRESH_BINARY; type <= THRESH_TOZERO_INV; ++type)
        {
            const double thresh = rng.uniform(0.0, maxVal);

            cv::Mat cpuRes;
            cv::threshold(img, cpuRes, thresh, maxVal, type);

            GpuMat gpu1(img);
            GpuMat gpuRes;
            cv::gpu::threshold(gpu1, gpuRes, thresh, maxVal, type);

            if (CheckNorm(cpuRes, gpuRes) != CvTS::OK)
                res = CvTS::FAIL_GENERIC;
        }

        return res;
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
            ts->printf(CvTS::LOG, "Unsupported type\n");
            return CvTS::OK;
        }

        int interpolations[] = {INTER_NEAREST, INTER_LINEAR, /*INTER_CUBIC,*/ /*INTER_LANCZOS4*/};
        const char* interpolations_str[] = {"INTER_NEAREST", "INTER_LINEAR", /*"INTER_CUBIC",*/ /*"INTER_LANCZOS4"*/};
        int interpolations_num = sizeof(interpolations) / sizeof(int);

        int test_res = CvTS::OK;

        for (int i = 0; i < interpolations_num; ++i)
        {
            ts->printf(CvTS::LOG, "Interpolation: %s\n", interpolations_str[i]);

            Mat cpu_res1, cpu_res2;
            cv::resize(img, cpu_res1, Size(), 2.0, 2.0, interpolations[i]);
            cv::resize(cpu_res1, cpu_res2, Size(), 0.5, 0.5, interpolations[i]);

            GpuMat gpu1(img), gpu_res1, gpu_res2;
            cv::gpu::resize(gpu1, gpu_res1, Size(), 2.0, 2.0, interpolations[i]);
            cv::gpu::resize(gpu_res1, gpu_res2, Size(), 0.5, 0.5, interpolations[i]);

            if (CheckSimilarity(cpu_res2, gpu_res2) != CvTS::OK)
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

        static double reflect[2][3] = { {-1, 0, 0},
                                        { 0, -1, 0} };
        reflect[0][2] = img.cols;
        reflect[1][2] = img.rows;

        Mat M(2, 3, CV_64F, (void*)reflect);

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

            // Check inner parts (ignoring 1 pixel width border)
            if (CheckSimilarity(cpudst.rowRange(1, cpudst.rows - 1).colRange(1, cpudst.cols - 1),
                                gpudst.rowRange(1, gpudst.rows - 1).colRange(1, gpudst.cols - 1)) != CvTS::OK)
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

        static double reflect[3][3] = { { -1, 0, 0},
                                        { 0, -1, 0},
                                        { 0, 0, 1 }};
        reflect[0][2] = img.cols;
        reflect[1][2] = img.rows;
        Mat M(3, 3, CV_64F, (void*)reflect);

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

            // Check inner parts (ignoring 1 pixel width border)
            if (CheckSimilarity(cpudst.rowRange(1, cpudst.rows - 1).colRange(1, cpudst.cols - 1),
                                gpudst.rowRange(1, gpudst.rows - 1).colRange(1, gpudst.cols - 1)) != CvTS::OK)
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

        Mat cpusum;
        cv::integral(img, cpusum, CV_32S);

        GpuMat gpu1(img);
        GpuMat gpusum;
        cv::gpu::integral(gpu1, gpusum);

        return CheckNorm(cpusum, gpusum) == CvTS::OK ? CvTS::OK : CvTS::FAIL_GENERIC;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Canny
//struct CV_GpuNppImageCannyTest : public CV_GpuImageProcTest
//{
//    CV_GpuNppImageCannyTest() : CV_GpuImageProcTest( "GPU-NppImageCanny", "Canny" ) {}
//
//    int test(const Mat& img)
//    {
//        if (img.type() != CV_8UC1)
//        {
//            ts->printf(CvTS::LOG, "\nUnsupported type\n");
//            return CvTS::OK;
//        }
//
//        const double threshold1 = 1.0, threshold2 = 10.0;
//
//        Mat cpudst;
//        cv::Canny(img, cpudst, threshold1, threshold2);
//
//        GpuMat gpu1(img);
//        GpuMat gpudst;
//        cv::gpu::Canny(gpu1, gpudst, threshold1, threshold2);
//
//        return CheckNorm(cpudst, gpudst);
//    }
//};

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

    if (ret <= 3)
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

////////////////////////////////////////////////////////////////////////
// Corner Harris feature detector

struct CV_GpuCornerHarrisTest: CvTest 
{
    CV_GpuCornerHarrisTest(): CvTest("GPU-CornerHarrisTest", "cornerHarris") {}

    void run(int)
    {
        try
        {
            for (int i = 0; i < 5; ++i)
            {
                int rows = 25 + rand() % 300, cols = 25 + rand() % 300;
                if (!compareToCpuTest(rows, cols, CV_32F, 1 + rand() % 5, 1 + 2 * (rand() % 4))) return;
                if (!compareToCpuTest(rows, cols, CV_32F, 1 + rand() % 5, -1)) return;
                if (!compareToCpuTest(rows, cols, CV_8U, 1 + rand() % 5, 1 + 2 * (rand() % 4))) return;
                if (!compareToCpuTest(rows, cols, CV_8U, 1 + rand() % 5, -1)) return;
            }
        }
        catch (const Exception& e)
        {
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    bool compareToCpuTest(int rows, int cols, int depth, int blockSize, int apertureSize)
    {
        RNG rng;
        cv::Mat src(rows, cols, depth);
        if (depth == CV_32F) 
            rng.fill(src, RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));
        else if (depth == CV_8U)
            rng.fill(src, RNG::UNIFORM, cv::Scalar(0), cv::Scalar(256));

        double k = 0.1;

        cv::Mat dst_gold;
        cv::gpu::GpuMat dst;
        cv::Mat dsth;
        int borderType;

        borderType = BORDER_REFLECT101;
        cv::cornerHarris(src, dst_gold, blockSize, apertureSize, k, borderType); 
        cv::gpu::cornerHarris(cv::gpu::GpuMat(src), dst, blockSize, apertureSize, k, borderType);

        dsth = dst;
        for (int i = 0; i < dst.rows; ++i)
        {
            for (int j = 0; j < dst.cols; ++j)
            {
                float a = dst_gold.at<float>(i, j);
                float b = dsth.at<float>(i, j);
                if (fabs(a - b) > 1e-3f) 
                {
                    ts->printf(CvTS::CONSOLE, "%d %d %f %f %d\n", i, j, a, b, apertureSize);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                };
            }
        }

        borderType = BORDER_REPLICATE;
        cv::cornerHarris(src, dst_gold, blockSize, apertureSize, k, borderType); 
        cv::gpu::cornerHarris(cv::gpu::GpuMat(src), dst, blockSize, apertureSize, k, borderType);

        dsth = dst;
        for (int i = 0; i < dst.rows; ++i)
        {
            for (int j = 0; j < dst.cols; ++j)
            {
                float a = dst_gold.at<float>(i, j);
                float b = dsth.at<float>(i, j);
                if (fabs(a - b) > 1e-3f) 
                {
                    ts->printf(CvTS::CONSOLE, "%d %d %f %f %d\n", i, j, a, b, apertureSize);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                };
            }
        }
        return true;
    }
};

////////////////////////////////////////////////////////////////////////
// Corner Min Eigen Val

struct CV_GpuCornerMinEigenValTest: CvTest 
{
    CV_GpuCornerMinEigenValTest(): CvTest("GPU-CornerMinEigenValTest", "cornerMinEigenVal") {}

    void run(int)
    {
        try
        {
            for (int i = 0; i < 3; ++i)
            {
                int rows = 25 + rand() % 300, cols = 25 + rand() % 300;
                if (!compareToCpuTest(rows, cols, CV_32F, 1 + rand() % 5, -1)) return;
                if (!compareToCpuTest(rows, cols, CV_32F, 1 + rand() % 5, 1 + 2 * (rand() % 4))) return;
                if (!compareToCpuTest(rows, cols, CV_8U, 1 + rand() % 5, -1)) return;
                if (!compareToCpuTest(rows, cols, CV_8U, 1 + rand() % 5, 1 + 2 * (rand() % 4))) return;
            }
        }
        catch (const Exception& e)
        {
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    bool compareToCpuTest(int rows, int cols, int depth, int blockSize, int apertureSize)
    {
        RNG rng;
        cv::Mat src(rows, cols, depth);
        if (depth == CV_32F) 
            rng.fill(src, RNG::UNIFORM, cv::Scalar(0), cv::Scalar(1));
        else if (depth == CV_8U)
            rng.fill(src, RNG::UNIFORM, cv::Scalar(0), cv::Scalar(256));

        cv::Mat dst_gold;
        cv::gpu::GpuMat dst;
        cv::Mat dsth;

        int borderType;

        borderType = BORDER_REFLECT101;
        cv::cornerMinEigenVal(src, dst_gold, blockSize, apertureSize, borderType); 
        cv::gpu::cornerMinEigenVal(cv::gpu::GpuMat(src), dst, blockSize, apertureSize, borderType);      

        dsth = dst;
        for (int i = 0; i < dst.rows; ++i)
        {
            for (int j = 0; j < dst.cols; ++j)
            {
                float a = dst_gold.at<float>(i, j);
                float b = dsth.at<float>(i, j);
                if (fabs(a - b) > 1e-2f) 
                {
                    ts->printf(CvTS::CONSOLE, "%d %d %f %f %d %d\n", i, j, a, b, apertureSize, blockSize);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                };
            }
        }

        borderType = BORDER_REPLICATE;
        cv::cornerMinEigenVal(src, dst_gold, blockSize, apertureSize, borderType); 
        cv::gpu::cornerMinEigenVal(cv::gpu::GpuMat(src), dst, blockSize, apertureSize, borderType);      

        dsth = dst;
        for (int i = 0; i < dst.rows; ++i)
        {
            for (int j = 0; j < dst.cols; ++j)
            {
                float a = dst_gold.at<float>(i, j);
                float b = dsth.at<float>(i, j);
                if (fabs(a - b) > 1e-2f) 
                {
                    ts->printf(CvTS::CONSOLE, "%d %d %f %f %d %d\n", i, j, a, b, apertureSize, blockSize);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return false;
                };
            }
        }

        return true;
    }
};

struct CV_GpuColumnSumTest: CvTest 
{
    CV_GpuColumnSumTest(): CvTest("GPU-ColumnSumTest", "columnSum") {}

    void run(int)
    {
        try
        {
            int cols = 375;
            int rows = 1072;

            Mat src(rows, cols, CV_32F);
            RNG rng(1);
            rng.fill(src, RNG::UNIFORM, Scalar(0), Scalar(1));

            GpuMat d_dst;
            columnSum(GpuMat(src), d_dst);
            
            Mat dst = d_dst;
            for (int j = 0; j < src.cols; ++j)
            {
                float a = src.at<float>(0, j);
                float b = dst.at<float>(0, j);
                if (fabs(a - b) > 0.5f)
                {
                    ts->printf(CvTS::CONSOLE, "big diff at %d %d: %f %f\n", 0, j, a, b);
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
            for (int i = 1; i < src.rows; ++i)
            {
                for (int j = 0; j < src.cols; ++j)
                {
                    float a = src.at<float>(i, j) += src.at<float>(i - 1, j);
                    float b = dst.at<float>(i, j);
                    if (fabs(a - b) > 0.5f)
                    {
                        ts->printf(CvTS::CONSOLE, "big diff at %d %d: %f %f\n", i, j, a, b);
                        ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                        return;
                    }
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
};

struct CV_GpuNormTest : CvTest 
{
    CV_GpuNormTest() : CvTest("GPU-Norm", "norm") {}

    void run(int)
    {
        try
        {
            RNG rng(0);

            int rows = rng.uniform(1, 500);
            int cols = rng.uniform(1, 500);

            for (int cn = 1; cn <= 4; ++cn)
            {
                test(NORM_L1, rows, cols, CV_8U, cn, Scalar::all(0), Scalar::all(10));
                test(NORM_L1, rows, cols, CV_8S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_L1, rows, cols, CV_16U, cn, Scalar::all(0), Scalar::all(10));
                test(NORM_L1, rows, cols, CV_16S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_L1, rows, cols, CV_32S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_L1, rows, cols, CV_32F, cn, Scalar::all(0), Scalar::all(1));

                test(NORM_L2, rows, cols, CV_8U, cn, Scalar::all(0), Scalar::all(10));
                test(NORM_L2, rows, cols, CV_8S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_L2, rows, cols, CV_16U, cn, Scalar::all(0), Scalar::all(10));
                test(NORM_L2, rows, cols, CV_16S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_L2, rows, cols, CV_32S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_L2, rows, cols, CV_32F, cn, Scalar::all(0), Scalar::all(1));

                test(NORM_INF, rows, cols, CV_8U, cn, Scalar::all(0), Scalar::all(10));
                test(NORM_INF, rows, cols, CV_8S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_INF, rows, cols, CV_16U, cn, Scalar::all(0), Scalar::all(10));
                test(NORM_INF, rows, cols, CV_16S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_INF, rows, cols, CV_32S, cn, Scalar::all(-10), Scalar::all(10));
                test(NORM_INF, rows, cols, CV_32F, cn, Scalar::all(0), Scalar::all(1));
            }
        }
        catch (const cv::Exception& e)
        {
            ts->printf(CvTS::CONSOLE, e.what());
            if (!check_and_treat_gpu_exception(e, ts)) throw;
            return;
        }
    }

    void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high)
    {
        mat.create(rows, cols, type);
        RNG rng(0);
        rng.fill(mat, RNG::UNIFORM, low, high);
    }

    void test(int norm_type, int rows, int cols, int depth, int cn, Scalar low, Scalar high)
    {
        int type = CV_MAKE_TYPE(depth, cn);

        Mat src;
        gen(src, rows, cols, type, low, high);

        double gold = norm(src, norm_type);
        double mine = norm(GpuMat(src), norm_type);

        if (abs(gold - mine) > 1e-3)
        {
            ts->printf(CvTS::CONSOLE, "failed test: gold=%f, mine=%f, norm_type=%d, rows=%d, "
                       "cols=%d, depth=%d, cn=%d\n", gold, mine, norm_type, rows, cols, depth, cn);
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D
class CV_GpuReprojectImageTo3DTest : public CvTest
{
public:
    CV_GpuReprojectImageTo3DTest() : CvTest("GPU-ReprojectImageTo3D", "reprojectImageTo3D") {}

protected:
    void run(int)
    {
        Mat disp(320, 240, CV_8UC1);

        RNG rng(*ts->get_rng());
        rng.fill(disp, RNG::UNIFORM, Scalar(5), Scalar(30));

        Mat Q(4, 4, CV_32FC1);
        rng.fill(Q, RNG::UNIFORM, Scalar(0.1), Scalar(1));

        Mat cpures;
        GpuMat gpures;

        reprojectImageTo3D(disp, cpures, Q, false);
        reprojectImageTo3D(GpuMat(disp), gpures, Q);

        Mat temp = gpures;

        for (int y = 0; y < cpures.rows; ++y)
        {
            const Vec3f* cpu_row = cpures.ptr<Vec3f>(y);
            const Vec4f* gpu_row = temp.ptr<Vec4f>(y);
            for (int x = 0; x < cpures.cols; ++x)
            {
                Vec3f a = cpu_row[x];
                Vec4f b = gpu_row[x];

                if (fabs(a[0] - b[0]) > 1e-5 || fabs(a[1] - b[1]) > 1e-5 || fabs(a[2] - b[2]) > 1e-5)
                {
                    ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////
/////////////////// tests registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// If we comment some tests, we may foget/miss to uncomment it after.
// Placing all test definitions in one place
// makes us know about what tests are commented.

CV_GpuImageThresholdTest CV_GpuImageThreshold_test;
CV_GpuNppImageResizeTest CV_GpuNppImageResize_test;
CV_GpuNppImageCopyMakeBorderTest CV_GpuNppImageCopyMakeBorder_test;
CV_GpuNppImageWarpAffineTest CV_GpuNppImageWarpAffine_test;
CV_GpuNppImageWarpPerspectiveTest CV_GpuNppImageWarpPerspective_test;
CV_GpuNppImageIntegralTest CV_GpuNppImageIntegral_test;
//CV_GpuNppImageCannyTest CV_GpuNppImageCanny_test;
CV_GpuCvtColorTest CV_GpuCvtColor_test;
CV_GpuHistogramsTest CV_GpuHistograms_test;
CV_GpuCornerHarrisTest CV_GpuCornerHarris_test;
CV_GpuCornerMinEigenValTest CV_GpuCornerMinEigenVal_test;
CV_GpuColumnSumTest CV_GpuColumnSum_test;
CV_GpuNormTest CV_GpuNormTest_test;
CV_GpuReprojectImageTo3DTest CV_GpuReprojectImageTo3D_test;
