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

// Multi-threaded calcHist must match a row-by-row single-threaded baseline.
// The non-contiguous ROI covers row splitting, the contiguous clone covers
// column splitting; the 1D mask keeps IPP/HAL fast paths out of the way.
typedef testing::TestWithParam< testing::tuple<int, int, Size> > Imgproc_Hist_Calc_Parallel;

TEST_P(Imgproc_Hist_Calc_Parallel, consistency)
{
    int type = testing::get<0>(GetParam());
    int dims = testing::get<1>(GetParam());
    Size sz  = testing::get<2>(GetParam());

    int cn = dims;

    Mat buf(Size(sz.width + 1, sz.height), CV_MAKETYPE(type, cn));
    randu(buf, Scalar::all(0), Scalar::all(256));
    Mat src = buf(Rect(0, 0, sz.width, sz.height));

    Mat mask;
    if( dims == 1 )
    {
        mask.create(sz, CV_8UC1);
        randu(mask, Scalar::all(0), Scalar::all(2));
    }

    std::vector<int> ch(dims), hs(dims);
    std::vector<std::vector<float> > rv(dims);
    std::vector<const float*> rp(dims);
    for( int i = 0; i < dims; i++ )
    {
        ch[i] = i;
        hs[i] = (dims == 1) ? 64 : (dims == 2) ? 8 : 4;
        rv[i].assign(2, 0.f);
        rv[i][1] = 256.f;
        rp[i] = &rv[i][0];
    }

    // Row-by-row: every call stays below the parallel threshold.
    Mat baseline;
    for( int y = 0; y < src.rows; y++ )
    {
        Mat row = src.row(y);
        Mat mask_row = (dims == 1) ? mask.row(y) : Mat();
        bool acc = (y > 0);
        calcHist(&row, 1, ch.data(), mask_row, baseline, dims, hs.data(), rp.data(), true, acc);
    }

    setNumThreads(8);
    Mat parallel_result;
    calcHist(&src, 1, ch.data(), mask, parallel_result, dims, hs.data(), rp.data(), true, false);

    // Flattened to one row: column-split path.
    Mat contiguous = src.clone();
    Mat contiguous_result;
    calcHist(&contiguous, 1, ch.data(), mask, contiguous_result, dims, hs.data(), rp.data(), true, false);
    setNumThreads(-1);

    EXPECT_EQ(0.0, cv::norm(baseline, parallel_result, NORM_INF))
        << "row-split, type=" << type << " dims=" << dims << " size=" << sz;
    EXPECT_EQ(0.0, cv::norm(baseline, contiguous_result, NORM_INF))
        << "column-split, type=" << type << " dims=" << dims << " size=" << sz;
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgproc_Hist_Calc_Parallel,
    testing::Combine(
        testing::Values(CV_8U, CV_16U, CV_32F),
        testing::Values(1, 2, 3),
        testing::Values(Size(640, 480), Size(1920, 1080), Size(7, 13))
    )
);

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

typedef ::testing::TestWithParam<std::tuple<int, bool>> Imgproc_Hist_CompareSparseChiSquare;

TEST_P(Imgproc_Hist_CompareSparseChiSquare, accuracy)
{
    const int dims = std::get<0>(GetParam());
    const bool explicitZeros = std::get<1>(GetParam());
    int sizes[] = {2, 2, 2};
    const float histograms[][2][4] = {
        {{1, 0, 0, 0}, {1, 2, 0, 0}}, // second histogram has an additional bin
        {{1, 0, 0, 0}, {0, 3, 0, 0}}, // disjoint support
        {{1, 2, 0, 0}, {3, 0, 4, 0}}, // partially overlapping support
        {{0, 0, 0, 0}, {1, 3, 0, 0}}, // empty first histogram
        {{0, 0, 0, 0}, {0, 0, 0, 0}}, // both empty
        {{1, 2, 0, 0}, {1, 2, 0, 0}}, // identical histograms
        {{0, 0, 0, 0}, {1e-20f, 0, 0, 0}} // denominator below the comparison threshold
    };
    const double expected[] = {4, 8, 14, 8, 0, 0, 0};

    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
    {
        SCOPED_TRACE(i);
        Mat dense1(dims, sizes, CV_32F, Scalar(0));
        Mat dense2(dims, sizes, CV_32F, Scalar(0));
        for (int j = 0; j < 4; ++j)
        {
            dense1.ptr<float>()[j] = histograms[i][0][j];
            dense2.ptr<float>()[j] = histograms[i][1][j];
        }
        SparseMat sparse1(dense1), sparse2(dense2);
        if (explicitZeros)
        {
            for (int j = 0; j < 4; ++j)
            {
                const int idx[] = {0, j / 2, j % 2};
                sparse1.ref<float>(idx + (3 - dims)) = histograms[i][0][j];
                sparse2.ref<float>(idx + (3 - dims)) = histograms[i][1][j];
            }
        }

        const size_t count1 = sparse1.nzcount(), count2 = sparse2.nzcount();

        EXPECT_DOUBLE_EQ(expected[i], compareHist(dense1, dense2, HISTCMP_CHISQR_ALT));
        EXPECT_DOUBLE_EQ(expected[i], compareHist(sparse1, sparse2, HISTCMP_CHISQR_ALT));
        EXPECT_DOUBLE_EQ(expected[i], compareHist(sparse2, sparse1, HISTCMP_CHISQR_ALT));
        EXPECT_DOUBLE_EQ(compareHist(dense1, dense2, HISTCMP_CHISQR),
                         compareHist(sparse1, sparse2, HISTCMP_CHISQR));
        EXPECT_DOUBLE_EQ(compareHist(dense2, dense1, HISTCMP_CHISQR),
                         compareHist(sparse2, sparse1, HISTCMP_CHISQR));
        EXPECT_EQ(count1, sparse1.nzcount());
        EXPECT_EQ(count2, sparse2.nzcount());
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_Hist_CompareSparseChiSquare,
                       testing::Combine(testing::Values(2, 3), testing::Bool()));

}} // namespace
/* End Of File */
