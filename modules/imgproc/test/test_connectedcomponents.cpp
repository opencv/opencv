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

    int ccltype[] = { cv::CCL_WU, cv::CCL_DEFAULT, cv::CCL_GRANA };

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

}} // namespace
