/*M///////////////////////////////////////////////////////////////////////////////////////
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
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

CV_ENUM(Method, RANSAC, LMEDS)
typedef TestWithParam<Method> EstimateTranslation2D;

static float rngIn(float from, float to) { return from + (to - from) * (float)theRNG(); }

// build a pure translation 2x3 matrix
static cv::Mat rngTranslationMat()
{
    double tx = rngIn(-20.f, 20.f);
    double ty = rngIn(-20.f, 20.f);
    double t[2*3] = { 1.0, 0.0, tx,
                      0.0, 1.0, ty };
    return cv::Mat(2, 3, CV_64F, t).clone();
}

static inline cv::Vec2d getTxTy(const cv::Mat& T)
{
    CV_Assert(T.rows == 2 && T.cols == 3 && T.type() == CV_64F);
    return cv::Vec2d(T.at<double>(0,2), T.at<double>(1,2));
}

TEST_P(EstimateTranslation2D, test1Point)
{
    // minimal sample is 1 point
    for (size_t i = 0; i < 500; ++i)
    {
        cv::Mat T = rngTranslationMat();
        cv::Vec2d T_ref = getTxTy(T);

        cv::Mat fpts(1, 1, CV_32FC2);
        cv::Mat tpts(1, 1, CV_32FC2);

        fpts.at<cv::Point2f>(0) = cv::Point2f(rngIn(1,2), rngIn(5,6));
        transform(fpts, tpts, T);

        std::vector<uchar> inliers;
        cv::Vec2d T_est = estimateTranslation2D(fpts, tpts, inliers, GetParam() /* method */);

        EXPECT_NEAR(T_est[0], T_ref[0], 1e-6);
        EXPECT_NEAR(T_est[1], T_ref[1], 1e-6);
        EXPECT_EQ((int)inliers.size(), 1);
        EXPECT_EQ((int)inliers[0], 1);
    }
}

TEST_P(EstimateTranslation2D, testNPoints)
{
    for (size_t i = 0; i < 500; ++i)
    {
        cv::Mat T = rngTranslationMat();
        cv::Vec2d T_ref = getTxTy(T);

        const int method = GetParam();
        const int n = 100;
        int m;
        // LMEDS can't handle more than 50% outliers (by design)
        if (method == LMEDS)
            m = 3*n/5;
        else
            m = 2*n/5;

        const float shift_outl = 15.f;
        const float noise_level = 20.f;

        cv::Mat fpts(1, n, CV_32FC2);
        cv::Mat tpts(1, n, CV_32FC2);

        randu(fpts, 0.f, 100.f);
        transform(fpts, tpts, T);

        /* adding noise to some points (make last n-m points outliers) */
        cv::Mat outliers = tpts.colRange(m, n);
        outliers.reshape(1) += shift_outl;

        cv::Mat noise(outliers.size(), outliers.type());
        randu(noise, 0.f, noise_level);
        outliers += noise;

        std::vector<uchar> inliers;
        cv::Vec2d T_est = estimateTranslation2D(fpts, tpts, inliers, method);

        // Check estimation produced finite values
        ASSERT_TRUE(std::isfinite(T_est[0]) && std::isfinite(T_est[1]));

        EXPECT_NEAR(T_est[0], T_ref[0], 1e-4);
        EXPECT_NEAR(T_est[1], T_ref[1], 1e-4);

        bool inliers_good = std::count(inliers.begin(), inliers.end(), 1) == m &&
            m == std::accumulate(inliers.begin(), inliers.begin() + m, 0);
        EXPECT_TRUE(inliers_good);
    }
}

// test conversion from other datatypes than float
TEST_P(EstimateTranslation2D, testConversion)
{
    cv::Mat T = rngTranslationMat();
    T.convertTo(T, CV_32S); // convert to int to transform ints properly

    std::vector<cv::Point> fpts(3);
    std::vector<cv::Point> tpts(3);

    fpts[0] = cv::Point2f(rngIn(1,2), rngIn(5,6));
    fpts[1] = cv::Point2f(rngIn(3,4), rngIn(3,4));
    fpts[2] = cv::Point2f(rngIn(1,2), rngIn(3,4));

    transform(fpts, tpts, T);

    std::vector<uchar> inliers;
    cv::Vec2d T_est = estimateTranslation2D(fpts, tpts, inliers, GetParam() /* method */);

    ASSERT_TRUE(std::isfinite(T_est[0]) && std::isfinite(T_est[1]));

    T.convertTo(T, CV_64F); // convert back for reference extraction
    cv::Vec2d T_ref = getTxTy(T);

    EXPECT_NEAR(T_est[0], T_ref[0], 1e-3);
    EXPECT_NEAR(T_est[1], T_ref[1], 1e-3);

    // all must be inliers
    EXPECT_EQ(countNonZero(inliers), 3);
}

INSTANTIATE_TEST_CASE_P(Calib3d, EstimateTranslation2D, Method::all());

// "don't change inputs" regression, mirroring affine partial test
TEST(EstimateTranslation2D, dont_change_inputs)
{
    /*const static*/ float pts0_[10] = {
            0.0f, 0.0f,
            0.0f, 8.0f,
            4.0f, 0.0f, // outlier
            8.0f, 8.0f,
            8.0f, 0.0f
    };
    /*const static*/ float pts1_[10] = {
            0.1f, 0.1f,
            0.1f, 8.1f,
            0.0f, 4.0f, // outlier
            8.1f, 8.1f,
            8.1f, 0.1f
    };

    cv::Mat pts0(cv::Size(1, 5), CV_32FC2, (void*)pts0_);
    cv::Mat pts1(cv::Size(1, 5), CV_32FC2, (void*)pts1_);

    cv::Mat pts0_copy = pts0.clone();
    cv::Mat pts1_copy = pts1.clone();

    cv::Mat inliers;

    cv::Vec2d T = cv::estimateTranslation2D(pts0, pts1, inliers);

    for (int i = 0; i < pts0.rows; ++i)
        EXPECT_EQ(pts0_copy.at<cv::Vec2f>(i), pts0.at<cv::Vec2f>(i)) << "pts0: i=" << i;

    for (int i = 0; i < pts1.rows; ++i)
        EXPECT_EQ(pts1_copy.at<cv::Vec2f>(i), pts1.at<cv::Vec2f>(i)) << "pts1: i=" << i;

    EXPECT_EQ(0, (int)inliers.at<uchar>(2));

    // sanity: estimated translation should be finite
    EXPECT_TRUE(std::isfinite(T[0]) && std::isfinite(T[1]));
}

}} // namespace
