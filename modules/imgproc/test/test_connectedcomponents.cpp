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

namespace opencv_test {
namespace {

class CV_ConnectedComponentsTest : public cvtest::BaseTest
{
public:
    CV_ConnectedComponentsTest();
    ~CV_ConnectedComponentsTest();
protected:
    void run(int);
};

CV_ConnectedComponentsTest::CV_ConnectedComponentsTest() {}
CV_ConnectedComponentsTest::~CV_ConnectedComponentsTest() {}

// This function force a row major order for the labels
void normalizeLabels(Mat1i& imgLabels, int iNumLabels) {
    vector<int> vecNewLabels(iNumLabels + 1, 0);
    int iMaxNewLabel = 0;

    for (int r = 0; r < imgLabels.rows; ++r) {
        for (int c = 0; c < imgLabels.cols; ++c) {
            int iCurLabel = imgLabels(r, c);
            if (iCurLabel > 0) {
                if (vecNewLabels[iCurLabel] == 0) {
                    vecNewLabels[iCurLabel] = ++iMaxNewLabel;
                }
                imgLabels(r, c) = vecNewLabels[iCurLabel];
            }
        }
    }
}

void CV_ConnectedComponentsTest::run(int /* start_from */)
{

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    string exp_path = string(ts->get_data_path()) + "connectedcomponents/ccomp_exp.png";
    Mat exp = imread(exp_path, IMREAD_GRAYSCALE);
    Mat orig = imread(string(ts->get_data_path()) + "connectedcomponents/concentric_circles.png", 0);

    if (orig.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    Mat bw = orig > 128;

    for (uint cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt)
    {

        Mat1i labelImage;
        int nLabels = connectedComponents(bw, labelImage, 8, CV_32S, ccltype[cclt]);

        normalizeLabels(labelImage, nLabels);

        // Validate test results
        for (int r = 0; r < labelImage.rows; ++r) {
            for (int c = 0; c < labelImage.cols; ++c) {
                int l = labelImage.at<int>(r, c);
                bool pass = l >= 0 && l <= nLabels;
                if (!pass) {
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    return;
                }
            }
        }

        if (exp.empty() || orig.size() != exp.size())
        {
            imwrite(exp_path, labelImage);
            exp = labelImage;
        }

        if (0 != cvtest::norm(labelImage > 0, exp > 0, NORM_INF))
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            return;
        }
        if (nLabels != cvtest::norm(labelImage, NORM_INF) + 1)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            return;
        }

    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Imgproc_ConnectedComponents, regression) { CV_ConnectedComponentsTest test; test.safe_run(); }

TEST(Imgproc_ConnectedComponents, grana_buffer_overflow)
{
    cv::Mat darkMask;
    darkMask.create(31, 87, CV_8U);
    darkMask = 0;

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    int nbComponents = cv::connectedComponentsWithStats(darkMask, labels, stats, centroids, 8, CV_32S, cv::CCL_GRANA);
    EXPECT_EQ(1, nbComponents);
}

static cv::Mat createCrashMat(int numThreads) {
    const int h = numThreads * 4 * 2 + 8;
    const double nParallelStripes = std::max(1, std::min(h / 2, numThreads * 4));
    const int w = 4;

    const int nstripes = cvRound(nParallelStripes <= 0 ? h : MIN(MAX(nParallelStripes, 1.), h));
    const cv::Range stripeRange(0, nstripes);
    const cv::Range wholeRange(0, h);

    cv::Mat m(h, w, CV_8U);
    m = 0;

    // Look for a range that starts with odd value and ends with even value
    cv::Range bugRange;
    for (int s = stripeRange.start; s < stripeRange.end; s++) {
        cv::Range sr(s, s + 1);
        cv::Range r;
        r.start = (int)(wholeRange.start +
            ((uint64)sr.start * (wholeRange.end - wholeRange.start) + nstripes / 2) / nstripes);
        r.end = sr.end >= nstripes ?
            wholeRange.end :
            (int)(wholeRange.start +
                ((uint64)sr.end * (wholeRange.end - wholeRange.start) + nstripes / 2) / nstripes);

        if (r.start > 0 && r.start % 2 == 1 && r.end % 2 == 0 && r.end >= r.start + 2) {
            bugRange = r;
            break;
        }
    }

    if (bugRange.empty()) { // Could not create a buggy range
        return m;
    }

    // Fill in bug Range
    for (int x = 1; x < w; x++) {
        m.at<char>(bugRange.start - 1, x) = 1;
    }

    m.at<char>(bugRange.start + 0, 0) = 1;
    m.at<char>(bugRange.start + 0, 1) = 1;
    m.at<char>(bugRange.start + 0, 3) = 1;
    m.at<char>(bugRange.start + 1, 1) = 1;
    m.at<char>(bugRange.start + 2, 1) = 1;
    m.at<char>(bugRange.start + 2, 3) = 1;
    m.at<char>(bugRange.start + 3, 0) = 1;
    m.at<char>(bugRange.start + 3, 1) = 1;

    return m;
}

TEST(Imgproc_ConnectedComponents, parallel_wu_labels)
{
    cv::Mat mat = createCrashMat(cv::getNumThreads());
    if (mat.empty()) {
        return;
    }

    const int nbPixels = cv::countNonZero(mat);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int nb = 0;
    EXPECT_NO_THROW(nb = cv::connectedComponentsWithStats(mat, labels, stats, centroids, 8, CV_32S, cv::CCL_WU));

    int area = 0;
    for (int i = 1; i < nb; ++i) {
        area += stats.at<int32_t>(i, cv::CC_STAT_AREA);
    }

    EXPECT_EQ(nbPixels, area);
}

TEST(Imgproc_ConnectedComponents, missing_background_pixels)
{
    cv::Mat m = Mat::ones(10, 10, CV_8U);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    EXPECT_NO_THROW(cv::connectedComponentsWithStats(m, labels, stats, centroids, 8, CV_32S, cv::CCL_WU));
    EXPECT_EQ(stats.at<int32_t>(0, cv::CC_STAT_WIDTH), 0);
    EXPECT_EQ(stats.at<int32_t>(0, cv::CC_STAT_HEIGHT), 0);
    EXPECT_EQ(stats.at<int32_t>(0, cv::CC_STAT_LEFT), -1);
    EXPECT_TRUE(std::isnan(centroids.at<double>(0, 0)));
    EXPECT_TRUE(std::isnan(centroids.at<double>(0, 1)));
}

TEST(Imgproc_ConnectedComponents, spaghetti_bbdt_sauf_stats)
{
    cv::Mat1b img(16, 16);
    img << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    cv::Mat1i labels;
    cv::Mat1i stats;
    cv::Mat1d centroids;

    int ccltype[] = { cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    for (uint cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(cv::connectedComponentsWithStats(img, labels, stats, centroids, 8, CV_32S, ccltype[cclt]));
        EXPECT_EQ(stats(0, cv::CC_STAT_LEFT), 0);
        EXPECT_EQ(stats(0, cv::CC_STAT_TOP), 0);
        EXPECT_EQ(stats(0, cv::CC_STAT_WIDTH), 16);
        EXPECT_EQ(stats(0, cv::CC_STAT_HEIGHT), 15);
        EXPECT_EQ(stats(0, cv::CC_STAT_AREA), 144);

        EXPECT_EQ(stats(1, cv::CC_STAT_LEFT), 1);
        EXPECT_EQ(stats(1, cv::CC_STAT_TOP), 1);
        EXPECT_EQ(stats(1, cv::CC_STAT_WIDTH), 3);
        EXPECT_EQ(stats(1, cv::CC_STAT_HEIGHT), 3);
        EXPECT_EQ(stats(1, cv::CC_STAT_AREA), 9);

        EXPECT_EQ(stats(2, cv::CC_STAT_LEFT), 1);
        EXPECT_EQ(stats(2, cv::CC_STAT_TOP), 1);
        EXPECT_EQ(stats(2, cv::CC_STAT_WIDTH), 8);
        EXPECT_EQ(stats(2, cv::CC_STAT_HEIGHT), 7);
        EXPECT_EQ(stats(2, cv::CC_STAT_AREA), 40);

        EXPECT_EQ(stats(3, cv::CC_STAT_LEFT), 10);
        EXPECT_EQ(stats(3, cv::CC_STAT_TOP), 2);
        EXPECT_EQ(stats(3, cv::CC_STAT_WIDTH), 5);
        EXPECT_EQ(stats(3, cv::CC_STAT_HEIGHT), 2);
        EXPECT_EQ(stats(3, cv::CC_STAT_AREA), 8);

        EXPECT_EQ(stats(4, cv::CC_STAT_LEFT), 11);
        EXPECT_EQ(stats(4, cv::CC_STAT_TOP), 5);
        EXPECT_EQ(stats(4, cv::CC_STAT_WIDTH), 3);
        EXPECT_EQ(stats(4, cv::CC_STAT_HEIGHT), 3);
        EXPECT_EQ(stats(4, cv::CC_STAT_AREA), 9);

        EXPECT_EQ(stats(5, cv::CC_STAT_LEFT), 2);
        EXPECT_EQ(stats(5, cv::CC_STAT_TOP), 9);
        EXPECT_EQ(stats(5, cv::CC_STAT_WIDTH), 1);
        EXPECT_EQ(stats(5, cv::CC_STAT_HEIGHT), 1);
        EXPECT_EQ(stats(5, cv::CC_STAT_AREA), 1);

        EXPECT_EQ(stats(6, cv::CC_STAT_LEFT), 12);
        EXPECT_EQ(stats(6, cv::CC_STAT_TOP), 9);
        EXPECT_EQ(stats(6, cv::CC_STAT_WIDTH), 1);
        EXPECT_EQ(stats(6, cv::CC_STAT_HEIGHT), 1);
        EXPECT_EQ(stats(6, cv::CC_STAT_AREA), 1);

        // Labels' order could be different!
        if (cclt == cv::CCL_WU || cclt == cv::CCL_SAUF) {
            // CCL_SAUF, CCL_WU
            EXPECT_EQ(stats(9, cv::CC_STAT_LEFT), 1);
            EXPECT_EQ(stats(9, cv::CC_STAT_TOP), 11);
            EXPECT_EQ(stats(9, cv::CC_STAT_WIDTH), 4);
            EXPECT_EQ(stats(9, cv::CC_STAT_HEIGHT), 2);
            EXPECT_EQ(stats(9, cv::CC_STAT_AREA), 8);

            EXPECT_EQ(stats(7, cv::CC_STAT_LEFT), 6);
            EXPECT_EQ(stats(7, cv::CC_STAT_TOP), 10);
            EXPECT_EQ(stats(7, cv::CC_STAT_WIDTH), 4);
            EXPECT_EQ(stats(7, cv::CC_STAT_HEIGHT), 2);
            EXPECT_EQ(stats(7, cv::CC_STAT_AREA), 8);

            EXPECT_EQ(stats(8, cv::CC_STAT_LEFT), 0);
            EXPECT_EQ(stats(8, cv::CC_STAT_TOP), 10);
            EXPECT_EQ(stats(8, cv::CC_STAT_WIDTH), 16);
            EXPECT_EQ(stats(8, cv::CC_STAT_HEIGHT), 6);
            EXPECT_EQ(stats(8, cv::CC_STAT_AREA), 21);
        }
        else {
            // CCL_BBDT, CCL_GRANA, CCL_SPAGHETTI, CCL_BOLELLI
            EXPECT_EQ(stats(7, cv::CC_STAT_LEFT), 1);
            EXPECT_EQ(stats(7, cv::CC_STAT_TOP), 11);
            EXPECT_EQ(stats(7, cv::CC_STAT_WIDTH), 4);
            EXPECT_EQ(stats(7, cv::CC_STAT_HEIGHT), 2);
            EXPECT_EQ(stats(7, cv::CC_STAT_AREA), 8);

            EXPECT_EQ(stats(8, cv::CC_STAT_LEFT), 6);
            EXPECT_EQ(stats(8, cv::CC_STAT_TOP), 10);
            EXPECT_EQ(stats(8, cv::CC_STAT_WIDTH), 4);
            EXPECT_EQ(stats(8, cv::CC_STAT_HEIGHT), 2);
            EXPECT_EQ(stats(8, cv::CC_STAT_AREA), 8);

            EXPECT_EQ(stats(9, cv::CC_STAT_LEFT), 0);
            EXPECT_EQ(stats(9, cv::CC_STAT_TOP), 10);
            EXPECT_EQ(stats(9, cv::CC_STAT_WIDTH), 16);
            EXPECT_EQ(stats(9, cv::CC_STAT_HEIGHT), 6);
            EXPECT_EQ(stats(9, cv::CC_STAT_AREA), 21);
        }
        EXPECT_EQ(stats(10, cv::CC_STAT_LEFT), 9);
        EXPECT_EQ(stats(10, cv::CC_STAT_TOP), 12);
        EXPECT_EQ(stats(10, cv::CC_STAT_WIDTH), 5);
        EXPECT_EQ(stats(10, cv::CC_STAT_HEIGHT), 2);
        EXPECT_EQ(stats(10, cv::CC_STAT_AREA), 7);
    }
}

TEST(Imgproc_ConnectedComponents, chessboard_even)
{
    cv::Size size(16, 16);
    cv::Mat1b input(size);
    cv::Mat1i output_8c(size);
    cv::Mat1i output_4c(size);

    // Chessboard image with even number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;

        output_8c <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;

        output_4c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
            0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0,
            0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 0,
            0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0,
            0, 57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64,
            65, 0, 66, 0, 67, 0, 68, 0, 69, 0, 70, 0, 71, 0, 72, 0,
            0, 73, 0, 74, 0, 75, 0, 76, 0, 77, 0, 78, 0, 79, 0, 80,
            81, 0, 82, 0, 83, 0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0,
            0, 89, 0, 90, 0, 91, 0, 92, 0, 93, 0, 94, 0, 95, 0, 96,
            97, 0, 98, 0, 99, 0, 100, 0, 101, 0, 102, 0, 103, 0, 104, 0,
            0, 105, 0, 106, 0, 107, 0, 108, 0, 109, 0, 110, 0, 111, 0, 112,
            113, 0, 114, 0, 115, 0, 116, 0, 117, 0, 118, 0, 119, 0, 120, 0,
            0, 121, 0, 122, 0, 123, 0, 124, 0, 125, 0, 126, 0, 127, 0, 128;
    }

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    cv::Mat1i labels;
    cv::Mat diff;
    int nLabels = 0;
    for (size_t cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 8, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_8c;
        EXPECT_EQ(cv::countNonZero(diff), 0);


        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 4, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_4c;
        EXPECT_EQ(cv::countNonZero(diff), 0);
    }

}

TEST(Imgproc_ConnectedComponents, chessboard_odd)
{
    cv::Size size(15, 15);
    cv::Mat1b input(size);
    cv::Mat1i output_8c(size);
    cv::Mat1i output_4c(size);

    // Chessboard image with odd number of rows and cols
    // Note that this is the maximum number of labels for 4-way connectivity
    {
        input <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;

        output_8c <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;

        output_4c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
            0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0,
            16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23,
            0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0,
            31, 0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38,
            0, 39, 0, 40, 0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0,
            46, 0, 47, 0, 48, 0, 49, 0, 50, 0, 51, 0, 52, 0, 53,
            0, 54, 0, 55, 0, 56, 0, 57, 0, 58, 0, 59, 0, 60, 0,
            61, 0, 62, 0, 63, 0, 64, 0, 65, 0, 66, 0, 67, 0, 68,
            0, 69, 0, 70, 0, 71, 0, 72, 0, 73, 0, 74, 0, 75, 0,
            76, 0, 77, 0, 78, 0, 79, 0, 80, 0, 81, 0, 82, 0, 83,
            0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0, 89, 0, 90, 0,
            91, 0, 92, 0, 93, 0, 94, 0, 95, 0, 96, 0, 97, 0, 98,
            0, 99, 0, 100, 0, 101, 0, 102, 0, 103, 0, 104, 0, 105, 0,
            106, 0, 107, 0, 108, 0, 109, 0, 110, 0, 111, 0, 112, 0, 113;
    }

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    cv::Mat1i labels;
    cv::Mat diff;
    int nLabels = 0;
    for (size_t cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 8, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_8c;
        EXPECT_EQ(cv::countNonZero(diff), 0);


        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 4, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_4c;
        EXPECT_EQ(cv::countNonZero(diff), 0);
    }

}

TEST(Imgproc_ConnectedComponents, maxlabels_8conn_even)
{
    cv::Size size(16, 16);
    cv::Mat1b input(size);
    cv::Mat1i output_8c(size);
    cv::Mat1i output_4c(size);

    {
        input <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        output_8c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        output_4c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    cv::Mat1i labels;
    cv::Mat diff;
    int nLabels = 0;
    for (size_t cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 8, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_8c;
        EXPECT_EQ(cv::countNonZero(diff), 0);


        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 4, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_4c;
        EXPECT_EQ(cv::countNonZero(diff), 0);
    }

}

TEST(Imgproc_ConnectedComponents, maxlabels_8conn_odd)
{
    cv::Size size(15, 15);
    cv::Mat1b input(size);
    cv::Mat1i output_8c(size);
    cv::Mat1i output_4c(size);

    {
        input <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;

        output_8c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64;

        output_4c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0, 32,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 40,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63, 0, 64;
    }

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    cv::Mat1i labels;
    cv::Mat diff;
    int nLabels = 0;
    for (size_t cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 8, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_8c;
        EXPECT_EQ(cv::countNonZero(diff), 0);


        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 4, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_4c;
        EXPECT_EQ(cv::countNonZero(diff), 0);
    }

}

TEST(Imgproc_ConnectedComponents, single_row)
{
    cv::Size size(1, 15);
    cv::Mat1b input(size);
    cv::Mat1i output_8c(size);
    cv::Mat1i output_4c(size);

    {
        input <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;


        output_8c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8;


        output_4c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8;

    }

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    cv::Mat1i labels;
    cv::Mat diff;
    int nLabels = 0;
    for (size_t cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 8, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_8c;
        EXPECT_EQ(cv::countNonZero(diff), 0);


        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 4, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_4c;
        EXPECT_EQ(cv::countNonZero(diff), 0);
    }

}

TEST(Imgproc_ConnectedComponents, single_column)
{
    cv::Size size(15, 1);
    cv::Mat1b input(size);
    cv::Mat1i output_8c(size);
    cv::Mat1i output_4c(size);

    {
        input <<
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;


        output_8c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8;


        output_4c <<
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8;

    }

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    cv::Mat1i labels;
    cv::Mat diff;
    int nLabels = 0;
    for (size_t cclt = 0; cclt < sizeof(ccltype) / sizeof(int); ++cclt) {

        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 8, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_8c;
        EXPECT_EQ(cv::countNonZero(diff), 0);


        EXPECT_NO_THROW(nLabels = cv::connectedComponents(input, labels, 4, CV_32S, ccltype[cclt]));
        normalizeLabels(labels, nLabels);

        diff = labels != output_4c;
        EXPECT_EQ(cv::countNonZero(diff), 0);
    }

}


TEST(Imgproc_ConnectedComponents, 4conn_regression_21366)
{
    Mat src = Mat::zeros(Size(10, 10), CV_8UC1);
    {
        Mat labels, stats, centroids;
        EXPECT_NO_THROW(cv::connectedComponentsWithStats(src, labels, stats, centroids, 4));
    }
}

TEST(Imgproc_ConnectedComponents, regression_27568)
{
    Mat image = Mat::zeros(Size(1000, 1000), CV_8UC1);
    for (int row = 0; row < image.rows; row += 2)
    {
        for (int col = 0; col < image.cols; col += 2)
        {
            image.at<uint8_t>(row, col) = 1;
        }
    }

    for (const int connectivity : {4, 8})
    {
        for (const int ccltype : {CCL_DEFAULT, CCL_WU, CCL_GRANA, CCL_BOLELLI, CCL_SAUF, CCL_BBDT, CCL_SPAGHETTI})
        {
            {
                Mat labels, stats, centroids;
                try
                {
                    connectedComponentsWithStats(
                        image, labels, stats, centroids, connectivity, CV_16U, ccltype);
                    ADD_FAILURE();
                }
                catch (const Exception& exception)
                {
                    EXPECT_TRUE(
                        strstr(
                            exception.what(),
                            "Total number of labels overflowed label type. Try using CV_32S instead of CV_16U as ltype"));
                }
            }

            {
                Mat labels, stats, centroids;
                EXPECT_NO_THROW(
                    connectedComponentsWithStats(
                        image, labels, stats, centroids, connectivity, CV_32S, ccltype));
            }
        }
    }
}

}
} // namespace
