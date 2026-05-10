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

namespace opencv_test { namespace {

//#define DUMP_RESULTS

#ifdef DUMP_RESULTS
#  define DUMP(image, path) imwrite(path, image)
#else
#  define DUMP(image, path)
#endif


TEST(Photo_DenoisingGrayscale, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    string expected_path = folder + "lena_noised_denoised_grayscale_tw=7_sw=21_h=10.png";

    Mat original = imread(original_path, IMREAD_GRAYSCALE);
    Mat expected = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    Mat result;
    fastNlMeansDenoising(original, result, 10);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingColored, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    string expected_path = folder + "lena_noised_denoised_lab12_tw=7_sw=21_h=10_h2=10.png";

    Mat original = imread(original_path, IMREAD_COLOR);
    Mat expected = imread(expected_path, IMREAD_COLOR);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    Mat result;
    fastNlMeansDenoisingColored(original, result, 10, 10);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingGrayscaleMulti, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    string expected_path = folder + "lena_noised_denoised_multi_tw=7_sw=21_h=15.png";
    Mat expected = imread(expected_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    vector<Mat> original(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original[i] = imread(original_path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(original[i].empty()) << "Could not load input image " << original_path;
    }

    Mat result;
    fastNlMeansDenoisingMulti(original, result, imgs_count / 2, imgs_count, 15);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingColoredMulti, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    string expected_path = folder + "lena_noised_denoised_multi_lab12_tw=7_sw=21_h=10_h2=15.png";
    Mat expected = imread(expected_path, IMREAD_COLOR);
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    vector<Mat> original(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original[i] = imread(original_path, IMREAD_COLOR);
        ASSERT_FALSE(original[i].empty()) << "Could not load input image " << original_path;
    }

    Mat result;
    fastNlMeansDenoisingColoredMulti(original, result, imgs_count / 2, imgs_count, 10, 15);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_White, issue_2646)
{
    cv::Mat img(50, 50, CV_8UC1, cv::Scalar::all(255));
    cv::Mat filtered;
    cv::fastNlMeansDenoising(img, filtered);

    int nonWhitePixelsCount = (int)img.total() - cv::countNonZero(filtered == img);

    ASSERT_EQ(0, nonWhitePixelsCount);
}

TEST(Photo_Denoising, speed)
{
    string imgname = string(cvtest::TS::ptr()->get_data_path()) + "shared/5MP.png";
    Mat src = imread(imgname, IMREAD_GRAYSCALE), dst;

    double t = (double)getTickCount();
    fastNlMeansDenoising(src, dst, 5, 7, 21);
    t = (double)getTickCount() - t;
    printf("execution time: %gms\n", t*1000./getTickFrequency());
}

// Related issue : https://github.com/opencv/opencv/issues/26582
TEST(Photo_DenoisingGrayscaleMulti16bitL1, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    vector<Mat> original_8u(imgs_count);
    vector<Mat> original_16u(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original_8u[i] = imread(original_path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(original_8u[i].empty()) << "Could not load input image " << original_path;
        original_8u[i].convertTo(original_16u[i], CV_16U);
    }

    Mat result_8u, result_16u;
    std::vector<float> h = {15};
    fastNlMeansDenoisingMulti(original_8u, result_8u, /*imgToDenoiseIndex*/ imgs_count / 2, /*temporalWindowSize*/ imgs_count, h, 7, 21, NORM_L1);
    fastNlMeansDenoisingMulti(original_16u, result_16u, /*imgToDenoiseIndex*/ imgs_count / 2, /*temporalWindowSize*/ imgs_count, h, 7, 21, NORM_L1);
    DUMP(result_8u, "8u.res.png");
    DUMP(result_16u, "16u.res.png");

    cv::Mat expected;
    result_8u.convertTo(expected, CV_16U);

}

TEST(Photo_DenoisingGrayscale16bitL2, accuracy)
{
    // Generate a programmatic image with some structure
    Mat original_8u(100, 100, CV_8UC1);
    for (int r = 0; r < 100; ++r)
    {
        for (int c = 0; c < 100; ++c)
        {
            original_8u.at<uchar>(r, c) = (uchar)((r * 2 + c) % 256);
        }
    }

    // Add noise
    Mat noised_8u = original_8u.clone();
    for (int r = 10; r < 90; ++r)
    {
        for (int c = 10; c < 90; ++c)
        {
            noised_8u.at<uchar>(r, c) = (uchar)std::min(255, std::max(0, (int)noised_8u.at<uchar>(r, c) + ((r + c) % 15 - 7)));
        }
    }

    Mat noised_16u;
    noised_8u.convertTo(noised_16u, CV_16U);

    Mat result_8u, result_16u;
    std::vector<float> h = {10};
    fastNlMeansDenoising(noised_8u, result_8u, h, 7, 21, NORM_L2);
    fastNlMeansDenoising(noised_16u, result_16u, h, 7, 21, NORM_L2);

    Mat expected;
    result_8u.convertTo(expected, CV_16U);

    EXPECT_MAT_NEAR(result_16u, expected, 1);
}

TEST(Photo_DenoisingGrayscaleMulti16bitL2, accuracy)
{
    const int imgs_count = 3;
    vector<Mat> original_8u(imgs_count);
    vector<Mat> original_16u(imgs_count);

    for (int i = 0; i < imgs_count; ++i)
    {
        original_8u[i] = Mat(100, 100, CV_8UC1);
        for (int r = 0; r < 100; ++r)
        {
            for (int c = 0; c < 100; ++c)
            {
                original_8u[i].at<uchar>(r, c) = (uchar)((r * 2 + c + i * 5) % 256);
            }
        }
        original_8u[i].convertTo(original_16u[i], CV_16U);
    }

    Mat result_8u, result_16u;
    std::vector<float> h = {15};
    fastNlMeansDenoisingMulti(original_8u, result_8u, /*imgToDenoiseIndex*/ imgs_count / 2, /*temporalWindowSize*/ imgs_count, h, 7, 21, NORM_L2);
    fastNlMeansDenoisingMulti(original_16u, result_16u, /*imgToDenoiseIndex*/ imgs_count / 2, /*temporalWindowSize*/ imgs_count, h, 7, 21, NORM_L2);

    cv::Mat expected;
    result_8u.convertTo(expected, CV_16U);

    EXPECT_MAT_NEAR(result_16u, expected, 1);
}

}} // namespace
