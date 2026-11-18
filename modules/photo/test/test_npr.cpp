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

static Mat makeZeroAndSmallSigmaInput(RNG& rng)
{
    Mat full(54, 104, CV_8UC3);
    // Left half: high-frequency transitions plus zero-gradient pairs so ct_H exceeds 2^29 at sigma_r = 1e-6
    Mat left = full.colRange(0, 52);
    rng.fill(left, RNG::UNIFORM, 24, 232);
    for (int c = 4; c < left.cols; c += 6)
        left.col(c - 1).copyTo(left.col(c));
    // Right half: 8x8 flat color patches so stylization preserves bright patch interiors alongside edges
    Mat blocks(7, 7, CV_8UC3);
    rng.fill(blocks, RNG::UNIFORM, 24, 232);
    Mat right;
    resize(blocks, right, Size(52, 54), 0, 0, INTER_NEAREST);
    right.copyTo(full.colRange(52, 104));
    return full(Rect(4, 3, 96, 48)); // non-continuous ROI
}

TEST(Photo_NPR, zero_sigma_returns_unsmoothed)
{
    Mat src = makeZeroAndSmallSigmaInput(theRNG());
    Mat dst, refStylized;
    stylization(src, refStylized, 1e-4f, 0.45f);

    edgePreservingFilter(src, dst, NORMCONV_FILTER, 0.f, 0.4f);
    EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));

    stylization(src, dst, 0.f, 0.45f);
    Scalar mean0, std0;
    meanStdDev(dst, mean0, std0);
    EXPECT_GT(cv::norm(std0), 10.0);
    EXPECT_LE(cvtest::norm(dst, refStylized, NORM_INF), 1.0);

    edgePreservingFilter(src, dst, RECURS_FILTER, 60.f, 0.f);
    EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));

    edgePreservingFilter(src, dst, NORMCONV_FILTER, 60.f, 0.f);
    EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));

    detailEnhance(src, dst, 10.f, 0.f);
    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 2.0);

    Mat stylR0;
    stylization(src, stylR0, 60.f, 0.f);
    Scalar meanR0, stdR0;
    meanStdDev(stylR0, meanR0, stdR0);
    EXPECT_GT(cv::norm(stdR0), 10.0);
    EXPECT_LE(cvtest::norm(stylR0, refStylized, NORM_INF), 1.0);

    edgePreservingFilter(src, dst, NORMCONV_FILTER, 0.f, 0.f);
    EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));

    edgePreservingFilter(src, dst, RECURS_FILTER, 0.f, 0.f);
    EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));
}

TEST(Photo_NPR, small_positive_sigma_not_flat)
{
    Mat src = makeZeroAndSmallSigmaInput(theRNG());
    Mat dst, refStylized;
    stylization(src, refStylized, 1e-4f, 0.45f);

    const float smallSigmaS[] = { 1e-6f, 1e-5f };
    for (int i = 0; i < 2; ++i)
    {
        const float ss = smallSigmaS[i];
        edgePreservingFilter(src, dst, NORMCONV_FILTER, ss, 0.4f);
        EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF)) << "sigma_s=" << ss;

        stylization(src, dst, ss, 0.45f);
        Scalar m, s;
        meanStdDev(dst, m, s);
        EXPECT_GT(cv::norm(s), 10.0) << "sigma_s=" << ss;
        EXPECT_LE(cvtest::norm(dst, refStylized, NORM_INF), 1.0) << "sigma_s=" << ss;
    }

    const float minNormal = std::numeric_limits<float>::min();
    const float smallSigmaR[] = { 1e-6f, minNormal };
    for (int i = 0; i < 2; ++i)
    {
        const float sr = smallSigmaR[i];
        edgePreservingFilter(src, dst, NORMCONV_FILTER, 60.f, sr);
        EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF)) << "sigma_r=" << sr;

        stylization(src, dst, 60.f, sr);
        Scalar m, s;
        meanStdDev(dst, m, s);
        EXPECT_GT(cv::norm(s), 10.0) << "sigma_r=" << sr;
        EXPECT_LE(cvtest::norm(dst, refStylized, NORM_INF), 1.0) << "sigma_r=" << sr;
    }

    edgePreservingFilter(src, dst, RECURS_FILTER, 60.f, minNormal);
    EXPECT_EQ(0.0, cvtest::norm(dst, src, NORM_INF));

    detailEnhance(src, dst, 10.f, minNormal);
    EXPECT_LE(cvtest::norm(dst, src, NORM_INF), 2.0);
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

TEST(Photo_NPR_EdgePreserveSmoothing_NormConvFilter, small_flat_image)
{
    RNG& rng = theRNG();
    const Size sizes[] = {
        Size(1, 1), Size(1, 5), Size(5, 1), Size(3, 3),
        Size(64, 64), Size(40, 200), Size(200, 40), Size(91, 91)
    };
    for (const Size& sz : sizes)
    {
        Scalar color(rng.uniform(16, 240), rng.uniform(16, 240), rng.uniform(16, 240));
        Mat src(sz, CV_8UC3, color), result;

        edgePreservingFilter(src, result, NORMCONV_FILTER);
        ASSERT_EQ(result.size(), src.size());
        ASSERT_EQ(result.type(), src.type());
        EXPECT_LE(cvtest::norm(result, src, NORM_INF), 1) << "edgePreservingFilter " << sz;

        stylization(src, result);
        ASSERT_EQ(result.size(), src.size());
        ASSERT_EQ(result.type(), src.type());
        EXPECT_LE(cvtest::norm(result, src, NORM_INF), 1) << "stylization " << sz;

        if (sz.width <= 91 && sz.height <= 91)
        {
            Mat sketch, colorSketch;
            pencilSketch(src, sketch, colorSketch);
            ASSERT_EQ(sketch.size(), src.size());
            ASSERT_EQ(colorSketch.size(), src.size());
            const int expectedShade = saturate_cast<uchar>((sz.width + sz.height) * 0.02f * 255.0f);
            EXPECT_LE(cvtest::norm(sketch, Mat(sz, CV_8UC1, Scalar(expectedShade)), NORM_INF), 1) << "pencilSketch " << sz;
        }
    }

    Mat colSrc(16, 1, CV_8UC3), colRes, wideRes;
    rng.fill(colSrc, RNG::UNIFORM, 32, 224);
    edgePreservingFilter(colSrc, colRes, NORMCONV_FILTER);
    edgePreservingFilter(repeat(colSrc, 1, 96), wideRes, NORMCONV_FILTER);
    ASSERT_EQ(colRes.size(), colSrc.size());
    EXPECT_LE(cvtest::norm(colRes, wideRes.col(0), NORM_INF), 1);

    Mat rowSrc = colSrc.t(), rowRes, tallRes;
    edgePreservingFilter(rowSrc, rowRes, NORMCONV_FILTER);
    edgePreservingFilter(repeat(rowSrc, 96, 1), tallRes, NORMCONV_FILTER);
    ASSERT_EQ(rowRes.size(), rowSrc.size());
    EXPECT_LE(cvtest::norm(rowRes, tallRes.row(0), NORM_INF), 1);
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

}} // namespace
