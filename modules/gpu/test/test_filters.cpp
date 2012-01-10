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

using namespace cvtest;
using namespace testing;

namespace
{
    double checkNorm(const cv::Mat& m1, const cv::Mat& m2, const cv::Size& ksize)
    {
        cv::Rect roi(ksize.width, ksize.height, m1.cols - 2 * ksize.width, m1.rows - 2 * ksize.height);
        cv::Mat m1ROI = m1(roi);
        cv::Mat m2ROI = m2(roi);
        return ::checkNorm(m1ROI, m2ROI);
    }

    double checkNorm(const cv::Mat& m1, const cv::Mat& m2, int ksize)
    {
        return checkNorm(m1, m2, cv::Size(ksize, ksize));
    }
}

#define EXPECT_MAT_NEAR_KSIZE(mat1, mat2, ksize, eps) \
    { \
        ASSERT_EQ(mat1.type(), mat2.type()); \
        ASSERT_EQ(mat1.size(), mat2.size()); \
        EXPECT_LE(checkNorm(mat1, mat2, ksize), eps); \
    }

/////////////////////////////////////////////////////////////////////////////////////////////////
// blur

PARAM_TEST_CASE(Blur, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
                
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        cv::blur(img_rgba, dst_gold_rgba, ksize);
        cv::blur(img_gray, dst_gold_gray, ksize);
    }
};

TEST_P(Blur, Rgba)
{
    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::blur(loadMat(img_rgba, useRoi), dev_dst_rgba, ksize);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, ksize, 1.0);
}

TEST_P(Blur, Gray)
{
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::blur(loadMat(img_gray, useRoi), dev_dst_gray, ksize);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, ksize, 1.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Blur, Combine(
                        ALL_DEVICES, 
                        Values(cv::Size(3, 3), cv::Size(5, 5), cv::Size(7, 7)),
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// sobel

PARAM_TEST_CASE(Sobel, cv::gpu::DeviceInfo, int, int, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int ksize;
    int dx;
    int dy;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        dx = GET_PARAM(2);
        dy = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        if (dx == 0 && dy == 0)
            return;

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
        
        cv::Sobel(img_rgba, dst_gold_rgba, -1, dx, dy, ksize);
        cv::Sobel(img_gray, dst_gold_gray, -1, dx, dy, ksize);
    }
};

TEST_P(Sobel, Rgba)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::Sobel(loadMat(img_rgba, useRoi), dev_dst_rgba, -1, dx, dy, ksize);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, ksize, 0.0);
}

TEST_P(Sobel, Gray)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::Sobel(loadMat(img_gray, useRoi), dev_dst_gray, -1, dx, dy, ksize);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, ksize, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Sobel, Combine(
                        ALL_DEVICES, 
                        Values(3, 5, 7), 
                        Values(0, 1, 2),
                        Values(0, 1, 2),
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// scharr

PARAM_TEST_CASE(Scharr, cv::gpu::DeviceInfo, int, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int dx;
    int dy;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        dx = GET_PARAM(1);
        dy = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        if (dx + dy != 1)
            return;

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        cv::Scharr(img_rgba, dst_gold_rgba, -1, dx, dy);
        cv::Scharr(img_gray, dst_gold_gray, -1, dx, dy);
    }
};

TEST_P(Scharr, Rgba)
{
    if (dx + dy != 1)
        return;

    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::Scharr(loadMat(img_rgba, useRoi), dev_dst_rgba, -1, dx, dy);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
}

TEST_P(Scharr, Gray)
{
    if (dx + dy != 1)
        return;

    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::Scharr(loadMat(img_gray, useRoi), dev_dst_gray, -1, dx, dy);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Scharr, Combine(
                        ALL_DEVICES, 
                        Values(0, 1),
                        Values(0, 1),
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// gaussianBlur

PARAM_TEST_CASE(GaussianBlur, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    double sigma1, sigma2;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
        
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        sigma1 = rng.uniform(0.1, 1.0); 
        sigma2 = rng.uniform(0.1, 1.0);
        
        cv::GaussianBlur(img_rgba, dst_gold_rgba, ksize, sigma1, sigma2);
        cv::GaussianBlur(img_gray, dst_gold_gray, ksize, sigma1, sigma2);
    }
};

TEST_P(GaussianBlur, Rgba)
{
    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::GaussianBlur(loadMat(img_rgba, useRoi), dev_dst_rgba, ksize, sigma1, sigma2);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, ksize, 3.0);
}

TEST_P(GaussianBlur, Gray)
{
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::GaussianBlur(loadMat(img_gray, useRoi), dev_dst_gray, ksize, sigma1, sigma2);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, ksize, 3.0);
}

INSTANTIATE_TEST_CASE_P(Filter, GaussianBlur, Combine(
                        ALL_DEVICES, 
                        Values(cv::Size(3, 3), cv::Size(5, 5), cv::Size(7, 7)),
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// laplacian

PARAM_TEST_CASE(Laplacian, cv::gpu::DeviceInfo, int, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int ksize;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        cv::Laplacian(img_rgba, dst_gold_rgba, -1, ksize);
        cv::Laplacian(img_gray, dst_gold_gray, -1, ksize);
    }
};

TEST_P(Laplacian, Rgba)
{
    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::Laplacian(loadMat(img_rgba, useRoi), dev_dst_rgba, -1, ksize);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
}

TEST_P(Laplacian, Gray)
{
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::Laplacian(loadMat(img_gray, useRoi), dev_dst_gray, -1, ksize);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Laplacian, Combine(
                        ALL_DEVICES,
                        Values(1, 3),
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// erode

PARAM_TEST_CASE(Erode, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat kernel;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        kernel = cv::Mat::ones(3, 3, CV_8U);
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        cv::erode(img_rgba, dst_gold_rgba, kernel);
        cv::erode(img_gray, dst_gold_gray, kernel);
    }
};

TEST_P(Erode, Rgba)
{
    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::erode(loadMat(img_rgba, useRoi), dev_dst_rgba, kernel);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
}

TEST_P(Erode, Gray)
{
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::erode(loadMat(img_gray, useRoi), dev_dst_gray, kernel);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Erode, Combine(
                        ALL_DEVICES,
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// dilate

PARAM_TEST_CASE(Dilate, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat kernel;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        kernel = cv::Mat::ones(3, 3, CV_8U);
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        cv::dilate(img_rgba, dst_gold_rgba, kernel);
        cv::dilate(img_gray, dst_gold_gray, kernel);
    }
};

TEST_P(Dilate, Rgba)
{
    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::dilate(loadMat(img_rgba, useRoi), dev_dst_rgba, kernel);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 3, 0.0);
}

TEST_P(Dilate, Gray)
{
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::dilate(loadMat(img_gray, useRoi), dev_dst_gray, kernel);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 3, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, Dilate, Combine(
                        ALL_DEVICES,
                        USE_ROI));

/////////////////////////////////////////////////////////////////////////////////////////////////
// morphEx

PARAM_TEST_CASE(MorphEx, cv::gpu::DeviceInfo, MorphOp, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int morphOp;
    bool useRoi;
    
    cv::Mat img_rgba;
    cv::Mat img_gray;

    cv::Mat kernel;

    cv::Mat dst_gold_rgba;
    cv::Mat dst_gold_gray;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        morphOp = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, img_rgba, CV_BGR2BGRA);
        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

        kernel = cv::Mat::ones(3, 3, CV_8U);
        
        cv::morphologyEx(img_rgba, dst_gold_rgba, morphOp, kernel);
        cv::morphologyEx(img_gray, dst_gold_gray, morphOp, kernel);
    }
};

TEST_P(MorphEx, Rgba)
{
    cv::Mat dst_rgba;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_rgba;

        cv::gpu::morphologyEx(loadMat(img_rgba, useRoi), dev_dst_rgba, morphOp, kernel);

        dev_dst_rgba.download(dst_rgba);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_rgba, dst_rgba, 4, 0.0);
}

TEST_P(MorphEx, Gray)
{
    cv::Mat dst_gray;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst_gray;

        cv::gpu::morphologyEx(loadMat(img_gray, useRoi), dev_dst_gray, morphOp, kernel);

        dev_dst_gray.download(dst_gray);
    );

    EXPECT_MAT_NEAR_KSIZE(dst_gold_gray, dst_gray, 4, 0.0);
}

INSTANTIATE_TEST_CASE_P(Filter, MorphEx, Combine(
                        ALL_DEVICES,
                        Values((int)cv::MORPH_OPEN, (int)cv::MORPH_CLOSE, (int)cv::MORPH_GRADIENT, (int)cv::MORPH_TOPHAT, (int)cv::MORPH_BLACKHAT),
                        USE_ROI));

#endif // HAVE_CUDA
