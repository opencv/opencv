/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

////////////////////////////////////////////////////////
// BilateralFilter

PARAM_TEST_CASE(BilateralFilter, cv::gpu::DeviceInfo, cv::Size, MatType)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int kernel_size;
    float sigma_color;
    float sigma_spatial;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);

        kernel_size = 5;
        sigma_color = 10.f;
        sigma_spatial = 3.5f;

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(BilateralFilter, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    src.convertTo(src, type);
    cv::gpu::GpuMat dst;

    cv::gpu::bilateralFilter(loadMat(src), dst, kernel_size, sigma_color, sigma_spatial);

    cv::Mat dst_gold;
    cv::bilateralFilter(src, dst_gold, kernel_size, sigma_color, sigma_spatial);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-3 : 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Denoising, BilateralFilter, testing::Combine(
    ALL_DEVICES,
    testing::Values(cv::Size(128, 128), cv::Size(113, 113), cv::Size(639, 481)),
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_32FC1), MatType(CV_32FC3))
    ));


////////////////////////////////////////////////////////
// Brute Force Non local means

struct BruteForceNonLocalMeans: testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();
        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(BruteForceNonLocalMeans, Regression)
{
    using cv::gpu::GpuMat;

    cv::Mat bgr  = readImage("denoising/lena_noised_gaussian_sigma=20_multi_0.png", cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());
    cv::resize(bgr, bgr, cv::Size(256, 256));

    cv::Mat gray;
    cv::cvtColor(bgr, gray, CV_BGR2GRAY);

    GpuMat dbgr, dgray;
    cv::gpu::nonLocalMeans(GpuMat(bgr),  dbgr, 20);
    cv::gpu::nonLocalMeans(GpuMat(gray), dgray, 20);

#if 0
    dumpImage("denoising/nlm_denoised_lena_bgr.png", cv::Mat(dbgr));
    dumpImage("denoising/nlm_denoised_lena_gray.png", cv::Mat(dgray));
#endif

    cv::Mat bgr_gold  = readImage("denoising/nlm_denoised_lena_bgr.png", cv::IMREAD_COLOR);
    cv::Mat gray_gold  = readImage("denoising/nlm_denoised_lena_gray.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(bgr_gold.empty() || gray_gold.empty());
    cv::resize(bgr_gold, bgr_gold, cv::Size(256, 256));
    cv::resize(gray_gold, gray_gold, cv::Size(256, 256));

    EXPECT_MAT_NEAR(bgr_gold, dbgr, 1);
    EXPECT_MAT_NEAR(gray_gold, dgray, 1);
}

INSTANTIATE_TEST_CASE_P(GPU_Denoising, BruteForceNonLocalMeans, ALL_DEVICES);

////////////////////////////////////////////////////////
// Fast Force Non local means

struct FastNonLocalMeans: testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();
        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(FastNonLocalMeans, Regression)
{
    using cv::gpu::GpuMat;

    cv::Mat bgr  = readImage("denoising/lena_noised_gaussian_sigma=20_multi_0.png", cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());

    cv::Mat gray;
    cv::cvtColor(bgr, gray, CV_BGR2GRAY);

    GpuMat dbgr, dgray;
    cv::gpu::FastNonLocalMeansDenoising fnlmd;

    fnlmd.simpleMethod(GpuMat(gray),  dgray, 20);
    fnlmd.labMethod(GpuMat(bgr),  dbgr, 20, 10);

#if 0
    dumpImage("denoising/fnlm_denoised_lena_bgr.png", cv::Mat(dbgr));
    dumpImage("denoising/fnlm_denoised_lena_gray.png", cv::Mat(dgray));
#endif

    cv::Mat bgr_gold  = readImage("denoising/fnlm_denoised_lena_bgr.png", cv::IMREAD_COLOR);
    cv::Mat gray_gold  = readImage("denoising/fnlm_denoised_lena_gray.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(bgr_gold.empty() || gray_gold.empty());

    EXPECT_MAT_NEAR(bgr_gold, dbgr, 1);
    EXPECT_MAT_NEAR(gray_gold, dgray, 1);
}

INSTANTIATE_TEST_CASE_P(GPU_Denoising, FastNonLocalMeans, ALL_DEVICES);

#endif // HAVE_CUDA
