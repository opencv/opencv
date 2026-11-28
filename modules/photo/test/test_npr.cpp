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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

static const double numerical_precision = 100.;

TEST(Photo_NPR_EdgePreserveSmoothing_RecursiveFilter, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    edgePreservingFilter(source,result,1);

    Mat reference = imread(folder + "smoothened_RF_reference.png");

    double psnr = cvtest::PSNR(reference, result);
    EXPECT_GT(psnr, 60.0);
}

TEST(Photo_NPR_EdgePreserveSmoothing_NormConvFilter, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    edgePreservingFilter(source,result,2);

    Mat reference = imread(folder + "smoothened_NCF_reference.png");

    double psnr = cvtest::PSNR(reference, result);
    EXPECT_GT(psnr, 60.0);
}

TEST(Photo_NPR_EdgePreserveSmoothing_NormConvFilter, large_image)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB, CV_TEST_TAG_DEBUG_LONG);

    const int rows = 2400, cols = 2400, ref_cols = 256, band_h = 3;
    Mat bands(rows / band_h, 1, CV_8UC3);
    theRNG().fill(bands, RNG::UNIFORM, 0, 256);

    Mat row_colors(rows, 1, CV_8UC3);
    for (int i = 0; i < rows; i++)
        bands.row(i / band_h).copyTo(row_colors.row(i));

    Mat src = repeat(row_colors, 1, cols);
    Mat ref_src = repeat(row_colors, 1, ref_cols);

    Mat result, ref_result;
    edgePreservingFilter(src, result, NORMCONV_FILTER);
    edgePreservingFilter(ref_src, ref_result, NORMCONV_FILTER);
    EXPECT_LE(cvtest::norm(result, repeat(ref_result.col(ref_cols / 2), 1, cols), NORM_INF), 1);

    Mat stylized, ref_stylized;
    stylization(src, stylized);
    stylization(ref_src, ref_stylized);
    EXPECT_LE(cvtest::norm(stylized, repeat(ref_stylized.col(ref_cols / 2), 1, cols), NORM_INF), 1);
}

TEST(Photo_NPR_DetailEnhance, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    detailEnhance(source,result);

    Mat reference = imread(folder + "detail_enhanced_reference.png");
    double psnr = cvtest::PSNR(reference, result);
    EXPECT_GT(psnr, 60.0);
}

TEST(Photo_NPR_PencilSketch, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat pencil_result, color_pencil_result;
    pencilSketch(source,pencil_result, color_pencil_result, 10, 0.1f, 0.03f);

    Mat pencil_reference = imread(folder + "pencil_sketch_reference.png", 0 /* == grayscale*/);
    double pencil_error = cvtest::norm(pencil_reference, pencil_result, NORM_L1);
    EXPECT_LE(pencil_error, numerical_precision);

    Mat color_pencil_reference = imread(folder + "color_pencil_sketch_reference.png");
    double color_pencil_error = cvtest::norm(color_pencil_reference, color_pencil_result, NORM_L1);
    EXPECT_LE(color_pencil_error, numerical_precision);
}

TEST(Photo_NPR_Stylization, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "npr/";
    string original_path = folder + "test1.png";

    Mat source = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load input image " << original_path;

    Mat result;
    stylization(source,result);

    Mat stylized_reference = imread(folder + "stylized_reference.png");
    double stylized_error = cvtest::norm(stylized_reference, result, NORM_L1);
    EXPECT_LE(stylized_error, numerical_precision);

}

// The filters work on 3-channel images. More channels used to overflow a heap buffer.
typedef testing::TestWithParam<int> Photo_NPR_MoreThanThreeChannels;

TEST_P(Photo_NPR_MoreThanThreeChannels, throws)
{
    Mat src = Mat::zeros(24, 24, CV_8UC(GetParam())), dst1, dst2;
    EXPECT_THROW(edgePreservingFilter(src, dst1, RECURS_FILTER), cv::Exception);
    EXPECT_THROW(edgePreservingFilter(src, dst1, NORMCONV_FILTER), cv::Exception);
    EXPECT_THROW(pencilSketch(src, dst1, dst2), cv::Exception);
    EXPECT_THROW(stylization(src, dst1), cv::Exception);
}

INSTANTIATE_TEST_CASE_P(/**/, Photo_NPR_MoreThanThreeChannels, testing::Values(4, 6, 8));

// An image with a zero width or height crashed.
typedef testing::TestWithParam<Size> Photo_NPR_EmptyInput;

TEST_P(Photo_NPR_EmptyInput, throws)
{
    Mat src(GetParam(), CV_8UC3), dst1, dst2;
    EXPECT_THROW(edgePreservingFilter(src, dst1), cv::Exception);
    EXPECT_THROW(pencilSketch(src, dst1, dst2), cv::Exception);
    EXPECT_THROW(stylization(src, dst1), cv::Exception);
}

INSTANTIATE_TEST_CASE_P(/**/, Photo_NPR_EmptyInput, testing::Values(Size(0, 0), Size(0, 5), Size(5, 0)));

TEST(Photo_NPR, three_channel_input_still_works)
{
    Mat src(24, 24, CV_8UC3), dst1, dst2;
    theRNG().fill(src, RNG::UNIFORM, 0, 256);

    EXPECT_NO_THROW(edgePreservingFilter(src, dst1));
    EXPECT_EQ(src.size(), dst1.size());
    EXPECT_NO_THROW(pencilSketch(src, dst1, dst2));
    EXPECT_EQ(src.size(), dst1.size());
    EXPECT_EQ(src.size(), dst2.size());
    EXPECT_NO_THROW(stylization(src, dst1));
    EXPECT_EQ(src.size(), dst1.size());
    EXPECT_EQ(CV_8UC3, dst1.type());
}

TEST(Photo_NPR, edge_preserving_filter_gray_input_still_works)
{
    Mat src(24, 24, CV_8UC1), dst;
    theRNG().fill(src, RNG::UNIFORM, 0, 256);
    EXPECT_NO_THROW(edgePreservingFilter(src, dst));
    EXPECT_EQ(src.size(), dst.size());
}

// The normalized convolution filter is written for exactly 3 channels and reads past the end of
// a row for narrow 1 and 2 channel images, so it needs 3 channels. The recursive filter does not.
typedef testing::TestWithParam<tuple<int, Size> > Photo_NPR_NormConvChannels;

TEST_P(Photo_NPR_NormConvChannels, requires_three_channels)
{
    Mat src = Mat::zeros(get<1>(GetParam()), CV_8UC(get<0>(GetParam()))), dst;
    EXPECT_THROW(edgePreservingFilter(src, dst, NORMCONV_FILTER), cv::Exception);
    EXPECT_NO_THROW(edgePreservingFilter(src, dst, RECURS_FILTER));
}

INSTANTIATE_TEST_CASE_P(/**/, Photo_NPR_NormConvChannels,
    testing::Combine(testing::Values(1, 2), testing::Values(Size(24, 24), Size(1, 24), Size(2, 24), Size(24, 1))));

}} // namespace
