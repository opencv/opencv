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

#include "opencv2/photo/cuda.hpp"
#include "opencv2/ts/cuda_test.hpp"

#include "opencv2/opencv_modules.hpp"
#include "cvconfig.h"

#if defined (HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAIMGPROC)

using namespace cvtest;

////////////////////////////////////////////////////////
// Brute Force Non local means

TEST(CUDA_BruteForceNonLocalMeans, Regression)
{
    using cv::cuda::GpuMat;

    cv::Mat bgr  = readImage("../gpu/denoising/lena_noised_gaussian_sigma=20_multi_0.png", cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    GpuMat dbgr, dgray;
    cv::cuda::nonLocalMeans(GpuMat(bgr),  dbgr, 20);
    cv::cuda::nonLocalMeans(GpuMat(gray), dgray, 20);

#if 0
    dumpImage("../gpu/denoising/nlm_denoised_lena_bgr.png", cv::Mat(dbgr));
    dumpImage("../gpu/denoising/nlm_denoised_lena_gray.png", cv::Mat(dgray));
#endif

    cv::Mat bgr_gold  = readImage("../gpu/denoising/nlm_denoised_lena_bgr.png", cv::IMREAD_COLOR);
    cv::Mat gray_gold  = readImage("../gpu/denoising/nlm_denoised_lena_gray.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(bgr_gold.empty() || gray_gold.empty());

    EXPECT_MAT_NEAR(bgr_gold, dbgr, 1e-4);
    EXPECT_MAT_NEAR(gray_gold, dgray, 1e-4);
}

////////////////////////////////////////////////////////
// Fast Force Non local means

TEST(CUDA_FastNonLocalMeans, Regression)
{
    using cv::cuda::GpuMat;

    cv::Mat bgr  = readImage("../gpu/denoising/lena_noised_gaussian_sigma=20_multi_0.png", cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    GpuMat dbgr, dgray;
    cv::cuda::FastNonLocalMeansDenoising fnlmd;

    fnlmd.simpleMethod(GpuMat(gray),  dgray, 20);
    fnlmd.labMethod(GpuMat(bgr),  dbgr, 20, 10);

#if 0
    dumpImage("../gpu/denoising/fnlm_denoised_lena_bgr.png", cv::Mat(dbgr));
    dumpImage("../gpu/denoising/fnlm_denoised_lena_gray.png", cv::Mat(dgray));
#endif

    cv::Mat bgr_gold  = readImage("../gpu/denoising/fnlm_denoised_lena_bgr.png", cv::IMREAD_COLOR);
    cv::Mat gray_gold  = readImage("../gpu/denoising/fnlm_denoised_lena_gray.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(bgr_gold.empty() || gray_gold.empty());

    EXPECT_MAT_NEAR(bgr_gold, dbgr, 1);
    EXPECT_MAT_NEAR(gray_gold, dgray, 1);
}

#endif // HAVE_CUDA
