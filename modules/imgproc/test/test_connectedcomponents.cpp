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

    for (int r = 0; r<imgLabels.rows; ++r) {
        for (int c = 0; c<imgLabels.cols; ++c) {
            int iCurLabel = imgLabels(r, c);
            if (iCurLabel>0) {
                if (vecNewLabels[iCurLabel] == 0) {
                    vecNewLabels[iCurLabel] = ++iMaxNewLabel;
                }
                imgLabels(r, c) = vecNewLabels[iCurLabel];
            }
        }
    }
}

void CV_ConnectedComponentsTest::run( int /* start_from */)
{

    int ccltype[] = { cv::CCL_DEFAULT, cv::CCL_WU, cv::CCL_GRANA, cv::CCL_BOLELLI, cv::CCL_SAUF, cv::CCL_BBDT, cv::CCL_SPAGHETTI };

    string exp_path = string(ts->get_data_path()) + "connectedcomponents/ccomp_exp.png";
    Mat exp = imread(exp_path, 0);
    Mat orig = imread(string(ts->get_data_path()) + "connectedcomponents/concentric_circles.png", 0);

    if (orig.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    Mat bw = orig > 128;

    for (uint cclt = 0; cclt < sizeof(ccltype)/sizeof(int); ++cclt)
    {

        Mat1i labelImage;
        int nLabels = connectedComponents(bw, labelImage, 8, CV_32S, ccltype[cclt]);

        normalizeLabels(labelImage, nLabels);

        // Validate test results
        for (int r = 0; r < labelImage.rows; ++r){
            for (int c = 0; c < labelImage.cols; ++c){
                int l = labelImage.at<int>(r, c);
                bool pass = l >= 0 && l <= nLabels;
                if (!pass){
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
        r.start = (int) (wholeRange.start +
                         ((uint64) sr.start * (wholeRange.end - wholeRange.start) + nstripes / 2) / nstripes);
        r.end = sr.end >= nstripes ?
                    wholeRange.end :
                    (int) (wholeRange.start +
                           ((uint64) sr.end * (wholeRange.end - wholeRange.start) + nstripes / 2) / nstripes);

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
    if(mat.empty()) {
        return;
    }

    const int nbPixels = cv::countNonZero(mat);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int nb = 0;
    EXPECT_NO_THROW( nb = cv::connectedComponentsWithStats(mat, labels, stats, centroids, 8, CV_32S, cv::CCL_WU) );

    int area = 0;
    for(int i=1; i<nb; ++i) {
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
    EXPECT_NO_THROW(cv::connectedComponentsWithStats(m, labels, stats, centroids, 8, CV_32S, cv::CCL_WU) );
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

}} // namespace
