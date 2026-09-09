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

namespace opencv_test { namespace {

////////////////////////////////////////////////////////
// Brute Force Non local means

TEST(CUDA_BruteForceNonLocalMeans, Regression)
{
    using cv::cuda::GpuMat;

    cv::Mat bgr  = readImage("../gpu/denoising/lena_noised_gaussian_sigma=20_multi_0.png", cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());
    cv::resize(bgr, bgr, cv::Size(256, 256));

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
    cv::resize(bgr_gold, bgr_gold, cv::Size(256, 256));
    cv::resize(gray_gold, gray_gold, cv::Size(256, 256));

    EXPECT_MAT_NEAR(bgr_gold, dbgr, 1);
    EXPECT_MAT_NEAR(gray_gold, dgray, 1);
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

    cv::cuda::fastNlMeansDenoising(GpuMat(gray),  dgray, 20);
    cv::cuda::fastNlMeansDenoisingColored(GpuMat(bgr),  dbgr, 20, 10);

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

TEST(CUDA_Photo_FastNlMeans, regression_16u)
{
    // 1. Generate random 16-bit data on the CPU (512x512 is sufficient for testing)
    cv::Mat src_host(512, 512, CV_16U);
    cv::randu(src_host, Scalar::all(0), Scalar::all(16));

    // 2. Upload data to the GPU
    cv::cuda::GpuMat src_device;
    src_device.upload(src_host);
    cv::cuda::GpuMat dst_device;

    float h = 3.0f;
    int search_window = 21;
    int block_size = 7;

    // 3. Execute the function and ensure it does not throw any exceptions
    EXPECT_NO_THROW({
        cv::cuda::fastNlMeansDenoising(src_device, dst_device, h, search_window, block_size);
    });

    ASSERT_FALSE(dst_device.empty()) << "Error: GPU output matrix should not be empty!";

    // 4. Download data back to the CPU for analysis
    cv::Mat dst_host;
    dst_device.download(dst_host);

    // CHECK 1: Did the output become completely zero? (Black image check)
    double minVal, maxVal;
    cv::minMaxLoc(dst_host, &minVal, &maxVal);
    EXPECT_GT(maxVal, 0) << "Error: Maximum value in the output image is 0 (Completely black)!";

    // CHECK 2: Did the algorithm actually modify the image?
    cv::Mat diff;
    cv::absdiff(src_host, dst_host, diff); // Get the absolute difference between input and output
    int changed_pixels = cv::countNonZero(diff);
    EXPECT_GT(changed_pixels, 0) << "Error: The algorithm did not change the image at all!";
}

TEST(CUDA_Photo_FastNlMeans, accuracy_16u)
{
    // 1. Load the standard OpenCV test image from opencv_extra
    std::string path = "shared/lena.png";
    string img_path = cvtest::findDataFile(path);
    Mat img_8u = imread(img_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img_8u.empty()) << "Test image not found!";

    // 2. Convert the 8-bit image to 16-bit and scale it to the full range (0-255 becomes 0-65280)
    Mat src_host;
    img_8u.convertTo(src_host, CV_16U, 256.0);

    // 3. Add synthetic Gaussian noise safely
    Mat noise(src_host.size(), CV_16S);
    randn(noise, 0, 1500); // Standard deviation suitable for 16-bit scale

    Mat noisy_host;
    // cv::add automatically handles clamping between 0-65535 for CV_16U
    cv::add(src_host, noise, noisy_host, noArray(), CV_16U);

    // 4. Upload data to the GPU
    cuda::GpuMat d_noisy, d_dst;
    d_noisy.upload(noisy_host);

    // 5. Execute the 16-bit NLM algorithm
    float h = 3000.0f;
    int search_window = 21;
    int block_size = 7;

    EXPECT_NO_THROW({
        cuda::fastNlMeansDenoising(d_noisy, d_dst, h, search_window, block_size);
    });

    Mat dst_host;
    d_dst.download(dst_host);

    // 6. Accuracy Check (PSNR) - IMPORTANT: Max value must be explicitly set to 65535.0 for 16-bit!
    double max_val_16u = 65535.0;
    double psnr_noisy = cv::PSNR(src_host, noisy_host, max_val_16u);
    double psnr_denoised = cv::PSNR(src_host, dst_host, max_val_16u);

    // The PSNR of the denoised image must be HIGHER than the noisy image (Higher = Better quality)
    EXPECT_GT(psnr_denoised, psnr_noisy)
        << "Accuracy Failed: Denoised image quality is worse than the noisy input!";

    // Logical minimum threshold for PSNR
    EXPECT_GT(psnr_denoised, 30.0)
        << "Accuracy Failed: PSNR is too low, check accumulator types or scaling!";
}

}} // namespace
#endif // HAVE_CUDA
