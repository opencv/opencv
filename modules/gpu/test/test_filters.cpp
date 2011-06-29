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

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

struct FilterTest
{
    static cv::Mat img_rgba;
    static cv::Mat img_gray;

    static void SetUpTestCase() 
    {
        cv::Mat img = readImage("stereobp/aloe-L.png");
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    }

    static void TearDownTestCase() 
    {
        img_rgba.release();
        img_gray.release();
    }
};

cv::Mat FilterTest::img_rgba;
cv::Mat FilterTest::img_gray;

static double checkNorm(const cv::Mat& m1, const cv::Mat& m2, const cv::Size& ksize)
{
    cv::Rect roi(ksize.width, ksize.height, m1.cols - 2 * ksize.width, m1.rows - 2 * ksize.height);
    cv::Mat m1ROI = m1(roi);
    cv::Mat m2ROI = m2(roi);
    return checkNorm(m1ROI, m2ROI);
}

static double checkNorm(const cv::Mat& m1, const cv::Mat& m2, int ksize)
{
    return checkNorm(m1, m2, cv::Size(ksize, ksize));
}

#define EXPECT_MAT_NEAR_KSIZE(mat1, mat2, ksize, eps) \
    { \
        ASSERT_EQ(mat1.type(), mat2.type()); \
        ASSERT_EQ(mat1.size(), mat2.size()); \
        EXPECT_LE(checkNorm(mat1, mat2, ksize), eps); \
    }

/////////////////////////////////////////////////////////////////////////////////////////////////
// blur

struct Blur : FilterTest, testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        ksize = cv::Size(std::tr1::get<1>(GetParam()), std::tr1::get<2>(GetParam()));

        cv::gpu::setDevice(devInfo.deviceID());

        cv::blur(img_rgba, dst_gold_rgba, ksize);
        cv::blur(img_gray, dst_gold_gray, ksize);
    }
};

TEST_P(Blur, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);
    PRINT_PARAM(ksize);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::blur(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, ksize);
        cv::gpu::blur(cv::gpu::GpuMat(img_gray), dev_dst_gray, ksize);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, ksize, 1.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, ksize, 1.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Blur, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(3, 5, 7), 
                        testing::Values(3, 5, 7)));

/////////////////////////////////////////////////////////////////////////////////////////////////
// sobel

struct Sobel : FilterTest, testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, std::pair<int, int> > >
{
    cv::gpu::DeviceInfo devInfo;
    int ksize;
    int dx, dy;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        ksize = std::tr1::get<1>(GetParam());
        std::pair<int, int> d = std::tr1::get<2>(GetParam());
        dx = d.first; dy = d.second;

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Sobel(img_rgba, dst_gold_rgba, -1, dx, dy, ksize);
        cv::Sobel(img_gray, dst_gold_gray, -1, dx, dy, ksize);
    }
};

TEST_P(Sobel, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);
    PRINT_PARAM(ksize);
    PRINT_PARAM(dx);
    PRINT_PARAM(dy);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::Sobel(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, -1, dx, dy, ksize);
        cv::gpu::Sobel(cv::gpu::GpuMat(img_gray), dev_dst_gray, -1, dx, dy, ksize);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, ksize, 0.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, ksize, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Sobel, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(3, 5, 7), 
                        testing::Values(std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(1, 1), std::make_pair(2, 0), std::make_pair(2, 1), std::make_pair(0, 2), std::make_pair(1, 2), std::make_pair(2, 2))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// scharr

struct Scharr : FilterTest, testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, std::pair<int, int> > >
{
    cv::gpu::DeviceInfo devInfo;
    int dx, dy;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        std::pair<int, int> d = std::tr1::get<1>(GetParam());
        dx = d.first; dy = d.second;

        cv::gpu::setDevice(devInfo.deviceID());

        cv::Scharr(img_rgba, dst_gold_rgba, -1, dx, dy);
        cv::Scharr(img_gray, dst_gold_gray, -1, dx, dy);
    }
};

TEST_P(Scharr, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);
    PRINT_PARAM(dx);
    PRINT_PARAM(dy);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::Scharr(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, -1, dx, dy);
        cv::gpu::Scharr(cv::gpu::GpuMat(img_gray), dev_dst_gray, -1, dx, dy);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Scharr, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(std::make_pair(1, 0), std::make_pair(0, 1))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// gaussianBlur

struct GaussianBlur : FilterTest, testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;

    double sigma1, sigma2;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        ksize = cv::Size(std::tr1::get<1>(GetParam()), std::tr1::get<2>(GetParam()));

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        sigma1 = rng.uniform(0.1, 1.0); 
        sigma2 = rng.uniform(0.1, 1.0);
        
        cv::GaussianBlur(img_rgba, dst_gold_rgba, ksize, sigma1, sigma2);
        cv::GaussianBlur(img_gray, dst_gold_gray, ksize, sigma1, sigma2);
    }
};

TEST_P(GaussianBlur, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);
    PRINT_PARAM(ksize);
    PRINT_PARAM(sigma1);
    PRINT_PARAM(sigma2);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::GaussianBlur(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, ksize, sigma1, sigma2);
        cv::gpu::GaussianBlur(cv::gpu::GpuMat(img_gray), dev_dst_gray, ksize, sigma1, sigma2);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, ksize, 3.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, ksize, 3.0);
}

INSTANTIATE_TEST_CASE_P(Filter, GaussianBlur, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(3, 5, 7), 
                        testing::Values(3, 5, 7)));

/////////////////////////////////////////////////////////////////////////////////////////////////
// laplacian

struct Laplacian : FilterTest, testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int ksize;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        ksize = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::Laplacian(img_rgba, dst_gold_rgba, -1, ksize);
        cv::Laplacian(img_gray, dst_gold_gray, -1, ksize);
    }
};

TEST_P(Laplacian, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);
    PRINT_PARAM(ksize);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::Laplacian(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, -1, ksize);
        cv::gpu::Laplacian(cv::gpu::GpuMat(img_gray), dev_dst_gray, -1, ksize);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Laplacian, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(1, 3)));

/////////////////////////////////////////////////////////////////////////////////////////////////
// erode

struct Erode : FilterTest, testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Mat kernel;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        kernel = cv::Mat::ones(3, 3, CV_8U);

        cv::erode(img_rgba, dst_gold_rgba, kernel);
        cv::erode(img_gray, dst_gold_gray, kernel);
    }
};

TEST_P(Erode, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::erode(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, kernel);
        cv::gpu::erode(cv::gpu::GpuMat(img_gray), dev_dst_gray, kernel);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Erode, testing::ValuesIn(devices()));

/////////////////////////////////////////////////////////////////////////////////////////////////
// dilate

struct Dilate : FilterTest, testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Mat kernel;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        kernel = cv::Mat::ones(3, 3, CV_8U);

        cv::dilate(img_rgba, dst_gold_rgba, kernel);
        cv::dilate(img_gray, dst_gold_gray, kernel);
    }
};

TEST_P(Dilate, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    PRINT_PARAM(devInfo);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::dilate(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, kernel);
        cv::gpu::dilate(cv::gpu::GpuMat(img_gray), dev_dst_gray, kernel);

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Dilate, testing::ValuesIn(devices()));

/////////////////////////////////////////////////////////////////////////////////////////////////
// morphEx

static const int morphOps[] = {cv::MORPH_OPEN, CV_MOP_CLOSE, CV_MOP_GRADIENT, CV_MOP_TOPHAT, CV_MOP_BLACKHAT};
static const char* morphOps_str[] = {"MORPH_OPEN", "MOP_CLOSE", "MOP_GRADIENT", "MOP_TOPHAT", "MOP_BLACKHAT"};

struct MorphEx : FilterTest, testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int morphOpsIdx;

    cv::Mat kernel;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;

    using FilterTest::SetUpTestCase;
    using FilterTest::TearDownTestCase;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        morphOpsIdx = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        kernel = cv::Mat::ones(3, 3, CV_8U);

        cv::morphologyEx(img_rgba, dst_gold_rgba, morphOps[morphOpsIdx], kernel);
        cv::morphologyEx(img_gray, dst_gold_gray, morphOps[morphOpsIdx], kernel);
    }
};

TEST_P(MorphEx, Accuracy)
{
    ASSERT_TRUE(!img_rgba.empty() && !img_gray.empty());

    const char* morphOpStr = morphOps_str[morphOpsIdx];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(morphOpStr);

    cv::Mat dst_rgba;
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::morphologyEx(cv::gpu::GpuMat(img_rgba), dev_dst_rgba, morphOps[morphOpsIdx], cv::gpu::GpuMat(kernel));
        cv::gpu::morphologyEx(cv::gpu::GpuMat(img_gray), dev_dst_gray, morphOps[morphOpsIdx], cv::gpu::GpuMat(kernel));

        dev_dst_rgba.download(dst_rgba);
        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 4, 0.0);
    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 4, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, MorphEx, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Range(0, 5)));

#endif // HAVE_CUDA
