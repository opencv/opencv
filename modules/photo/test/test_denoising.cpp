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
TEST(Photo_DenoisingGrayscaleMulti16Bit, ComprehensiveRegression)
{
    auto computePSNR = [](const Mat& I1, const Mat& I2) -> double {
        CV_Assert(I1.type() == I2.type() && I1.size() == I2.size());

        Mat s1;
        absdiff(I1, I2, s1);
        s1.convertTo(s1, CV_32F);
        s1 = s1.mul(s1);

        Scalar s = sum(s1);

        double mse = s[0] / static_cast<double>(I1.total());
        if (mse == 0) {
            return INFINITY;
        }
        double max_pixel = 65535.0;
        double psnr = 10.0 * log10((max_pixel * max_pixel) / mse);
        return psnr;
    };

    const int imgs_count = 5;
    const int width = 512;
    const int height = 512;
    std::vector<Mat> original(imgs_count);

    for (int i = 0; i < imgs_count; i++)
    {
        original[i] = Mat::ones(height, width, CV_16UC1) * 10000;
        randu(original[i], Scalar::all(9500), Scalar::all(10500));
    }

    int templateWindowSize = 7;
    int searchWindowSize = 21;
    float h = 15.0f;
    std::vector<float> h_vec = {h};

    Mat result;

    try
    {
        cv::fastNlMeansDenoisingMulti(
            original,
            result,
            imgs_count / 2,
            imgs_count,
            h_vec,
            templateWindowSize,
            searchWindowSize,
            cv::NORM_L1);
    }
    catch (const cv::Exception &e)
    {
        FAIL() << "fastNlMeansDenoisingMulti threw an exception with 16-bit images: " << e.what();
    }

    ASSERT_FALSE(result.empty()) << "Denoising result is empty.";
    ASSERT_EQ(result.type(), CV_16UC1) << "Denoising result has incorrect type.";
    ASSERT_EQ(result.size(), original[0].size()) << "Denoising result has incorrect size.";

    double minVal, maxVal;
    minMaxLoc(result, &minVal, &maxVal);
    ASSERT_GE(minVal, 0) << "Denoised image has negative values.";
    ASSERT_LE(maxVal, 65535) << "Denoised image has values exceeding 16-bit maximum.";

    Mat groundTruth = Mat::ones(height, width, CV_16UC1) * 10000;
    double psnr = computePSNR(result, groundTruth);
    ASSERT_GT(psnr, 30.0) << "PSNR is too low, denoising may not be effective.";

    Mat maxImage = Mat::ones(height, width, CV_16UC1) * 65535;
    original = std::vector<Mat>(imgs_count, maxImage);
    cv::fastNlMeansDenoisingMulti(
        original,
        result,
        imgs_count / 2,
        imgs_count,
        h_vec,
        templateWindowSize,
        searchWindowSize,
        cv::NORM_L1);
    minMaxLoc(result, &minVal, &maxVal);
    ASSERT_EQ(maxVal, 65535) << "Denoised max image altered maximum pixel values.";

    Mat zeroImage = Mat::zeros(height, width, CV_16UC1);
    original = std::vector<Mat>(imgs_count, zeroImage);
    cv::fastNlMeansDenoisingMulti(
        original,
        result,
        imgs_count / 2,
        imgs_count,
        h_vec,
        templateWindowSize,
        searchWindowSize,
        cv::NORM_L1);
    minMaxLoc(result, &minVal, &maxVal);
    ASSERT_EQ(minVal, 0) << "Denoised zero image altered minimum pixel values.";
    ASSERT_EQ(maxVal, 0) << "Denoised zero image altered maximum pixel values.";
}

}} // namespace
