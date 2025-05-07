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

namespace opencv_test { namespace {

TEST(Imgproc_Hist_Calc, calcHist_regression_11544)
{
    cv::Mat1w m = cv::Mat1w::zeros(10, 10);
    int n_images = 1;
    int channels[] = { 0 };
    cv::Mat mask;
    cv::MatND hist1, hist2;
    cv::MatND hist1_opt, hist2_opt;
    int dims = 1;
    int hist_size[] = { 1000 };
    float range1[] = { 0, 900 };
    float range2[] = { 0, 1000 };
    const float* ranges1[] = { range1 };
    const float* ranges2[] = { range2 };

    setUseOptimized(false);
    cv::calcHist(&m, n_images, channels, mask, hist1, dims, hist_size, ranges1);
    cv::calcHist(&m, n_images, channels, mask, hist2, dims, hist_size, ranges2);

    setUseOptimized(true);
    cv::calcHist(&m, n_images, channels, mask, hist1_opt, dims, hist_size, ranges1);
    cv::calcHist(&m, n_images, channels, mask, hist2_opt, dims, hist_size, ranges2);

    for(int i = 0; i < 1000; i++)
    {
        EXPECT_EQ(hist1.at<float>(i), hist1_opt.at<float>(i)) << i;
        EXPECT_EQ(hist2.at<float>(i), hist2_opt.at<float>(i)) << i;
    }
}

TEST(Imgproc_Hist_Calc, badarg)
{
    const int channels[] = {0};
    float range1[] = {0, 10};
    float range2[] = {10, 20};
    const float * ranges[] = {range1, range2};
    Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    Mat imgInt = cv::Mat::zeros(10, 10, CV_32SC1);
    Mat hist;
    const int hist_size[] = { 100, 100 };
    // base run
    EXPECT_NO_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, 1, hist_size, ranges, true));
    // bad parameters
    EXPECT_THROW(cv::calcHist(NULL, 1, channels, noArray(), hist, 1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 0, channels, noArray(), hist, 1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, NULL, noArray(), hist, 2, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, channels, noArray(), noArray(), 1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, -1, hist_size, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, 1, NULL, ranges, true), cv::Exception);
    EXPECT_THROW(cv::calcHist(&imgInt, 1, channels, noArray(), hist, 1, hist_size, NULL, true), cv::Exception);
    // special case
    EXPECT_NO_THROW(cv::calcHist(&img, 1, channels, noArray(), hist, 1, hist_size, NULL, true));

    Mat backProj;
    // base run
    EXPECT_NO_THROW(cv::calcBackProject(&img, 1, channels, hist, backProj, ranges, 1, true));
    // bad parameters
    EXPECT_THROW(cv::calcBackProject(NULL, 1, channels, hist, backProj, ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&img, 0, channels, hist, backProj, ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&img, 1, channels, noArray(), backProj, ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&img, 1, channels, hist, noArray(), ranges, 1, true), cv::Exception);
    EXPECT_THROW(cv::calcBackProject(&imgInt, 1, channels, hist, backProj, NULL, 1, true), cv::Exception);
    // special case
    EXPECT_NO_THROW(cv::calcBackProject(&img, 1, channels, hist, backProj, NULL, 1, true));
}

TEST(Imgproc_Hist_Calc, IPP_ranges_with_equal_exponent_21595)
{
    const int channels[] = { 0 };
    float range1[] = { -0.5f, 1.5f };
    const float* ranges[] = { range1 };
    const int hist_size[] = { 2 };

    uint8_t m[1][6] = { { 0, 1, 0, 1 , 1, 1 } };
    cv::Mat images_u = Mat(1, 6, CV_8UC1, m);
    cv::Mat histogram_u;
    cv::calcHist(&images_u, 1, channels, noArray(), histogram_u, 1, hist_size, ranges);

    ASSERT_EQ(histogram_u.at<float>(0), 2.f) << "0 not counts correctly, res: " << histogram_u.at<float>(0);
    ASSERT_EQ(histogram_u.at<float>(1), 4.f) << "1 not counts correctly, res: " << histogram_u.at<float>(0);
}

TEST(Imgproc_Hist_Calc, IPP_ranges_with_nonequal_exponent_21595)
{
    const int channels[] = { 0 };
    float range1[] = { -1.3f, 1.5f };
    const float* ranges[] = { range1 };
    const int hist_size[] = { 3 };

    uint8_t m[1][6] = { { 0, 1, 0, 1 , 1, 1 } };
    cv::Mat images_u = Mat(1, 6, CV_8UC1, m);
    cv::Mat histogram_u;
    cv::calcHist(&images_u, 1, channels, noArray(), histogram_u, 1, hist_size, ranges);

    ASSERT_EQ(histogram_u.at<float>(0), 0.f) << "not equal to zero, res: " << histogram_u.at<float>(0);
    ASSERT_EQ(histogram_u.at<float>(1), 2.f) << "0 not counts correctly, res: " << histogram_u.at<float>(1);
    ASSERT_EQ(histogram_u.at<float>(2), 4.f) << "1 not counts correctly, res: " << histogram_u.at<float>(2);
}

////////////////////////////////////////// equalizeHist() /////////////////////////////////////////

void equalizeHistReference(const Mat& src, Mat& dst)
{
    std::vector<int> hist(256, 0);
    for (int y = 0; y < src.rows; y++)
    {
        const uchar* srow = src.ptr(y);
        for (int x = 0; x < src.cols; x++)
        {
            hist[srow[x]]++;
        }
    }

    int first = 0;
    while (!hist[first]) ++first;

    int total = (int)src.total();
    if (hist[first] == total)
    {
        dst.setTo(first);
        return;
    }

    std::vector<uchar> lut(256);
    lut[first] = 0;
    float scale = (255.f)/(total - hist[first]);

    int sum = 0;
    for (int i = first + 1; i < 256; ++i)
    {
        sum += hist[i];
        lut[i] = saturate_cast<uchar>(sum * scale);
    }

    cv::LUT(src, lut, dst);
}

typedef ::testing::TestWithParam<std::tuple<cv::Size, int>> Imgproc_Equalize_Hist;

TEST_P(Imgproc_Equalize_Hist, accuracy)
{
    auto p = GetParam();
    cv::Size size = std::get<0>(p);
    int idx = std::get<1>(p);

    RNG &rng = cvtest::TS::ptr()->get_rng();
    rng.state += idx;

    cv::Mat src(size, CV_8U);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    cv::Mat dst, gold;

    equalizeHistReference(src, gold);

    cv::equalizeHist(src, dst);

    ASSERT_EQ(CV_8UC1, dst.type());
    ASSERT_EQ(gold.size(), dst.size());

    EXPECT_MAT_NEAR(dst, gold, 1);
    EXPECT_MAT_N_DIFF(dst, gold, 0.05 * size.area()); // The 5% range could be accomodated to HAL
}

INSTANTIATE_TEST_CASE_P(Imgproc_Hist, Imgproc_Equalize_Hist, ::testing::Combine(
                        ::testing::Values(cv::Size(123, 321), cv::Size(256, 256), cv::Size(1024, 768)),
                        ::testing::Range(0, 10)));

// See https://github.com/opencv/opencv/issues/24757
TEST(Imgproc_Hist_Compare, intersect_regression_24757)
{
    cv::Mat src1 = cv::Mat::zeros(128,1, CV_32FC1);
    cv::Mat src2 = cv::Mat(128,1, CV_32FC1, cv::Scalar(std::numeric_limits<double>::max()));

                                             // Ideal result        Wrong result
    src1.at<float>(32 * 0,0) = +1.0f;        // work = +1.0         +1.0
    src1.at<float>(32 * 1,0) = +55555555.5f; // work = +55555556.5  +55555555.5
    src1.at<float>(32 * 2,0) = -55555555.5f; // work = +1.0         0.0
    src1.at<float>(32 * 3,0) = -1.0f;        // work = 0.0          -1.0

    EXPECT_DOUBLE_EQ(compareHist(src1, src2, cv::HISTCMP_INTERSECT), 0.0);
}

}} // namespace
/* End Of File */
