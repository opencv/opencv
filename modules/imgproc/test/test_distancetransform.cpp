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
#include <numeric>

namespace opencv_test { namespace {

BIGDATA_TEST(Imgproc_DistanceTransform, large_image_12218)
{
    const int lls_maxcnt = 79992000;   // labels's maximum count
    const int lls_mincnt = 1;          // labels's minimum count
    int i, j, nz;
    Mat src(8000, 20000, CV_8UC1), dst, labels;
    for( i = 0; i < src.rows; i++ )
        for( j = 0; j < src.cols; j++ )
            src.at<uchar>(i, j) = (j > (src.cols / 2)) ? 0 : 255;

    distanceTransform(src, dst, labels, cv::DIST_L2, cv::DIST_MASK_3, DIST_LABEL_PIXEL);

    double scale = (double)lls_mincnt / (double)lls_maxcnt;
    labels.convertTo(labels, CV_32SC1, scale);
    Size size = labels.size();
    nz = cv::countNonZero(labels);
    EXPECT_EQ(nz, (size.height*size.width / 2));
}

TEST(Imgproc_DistanceTransform, wide_image_22732)
{
    Mat src = Mat::zeros(1, 4099, CV_8U); // 4099 or larger used to be bad
    Mat dist(src.rows, src.cols, CV_32F);
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE, CV_32F);
    int nz = countNonZero(dist);
    EXPECT_EQ(nz, 0);
}

TEST(Imgproc_DistanceTransform, large_square_22732)
{
    Mat src = Mat::zeros(8000, 8005, CV_8U), dist;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE, CV_32F);
    int nz = countNonZero(dist);
    EXPECT_EQ(dist.size(), src.size());
    EXPECT_EQ(dist.type(), CV_32F);
    EXPECT_EQ(nz, 0);

    Point p0(src.cols-1, src.rows-1);
    src.setTo(1);
    src.at<uchar>(p0) = 0;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE, CV_32F);
    EXPECT_EQ(dist.size(), src.size());
    EXPECT_EQ(dist.type(), CV_32F);
    bool first = true;
    int nerrs = 0;
    for (int y = 0; y < dist.rows; y++)
        for (int x = 0; x < dist.cols; x++) {
            float d = dist.at<float>(y, x);
            double dx = (double)(x - p0.x), dy = (double)(y - p0.y);
            float d0 = (float)sqrt(dx*dx + dy*dy);
            if (std::abs(d0 - d) > 1) {
                if (first) {
                    printf("y=%d, x=%d. dist_ref=%.2f, dist=%.2f\n", y, x, d0, d);
                    first = false;
                }
                nerrs++;
            }
        }
    EXPECT_EQ(0, nerrs) << "reference distance map is different from computed one at " << nerrs << " pixels\n";
}

BIGDATA_TEST(Imgproc_DistanceTransform, issue_23895_3x3)
{
    Mat src = Mat::zeros(50000, 50000, CV_8U), dist;
    distanceTransform(src.col(0), dist, DIST_L2, DIST_MASK_3);
    int nz = countNonZero(dist);
    EXPECT_EQ(nz, 0);
}

BIGDATA_TEST(Imgproc_DistanceTransform, issue_23895_5x5)
{
    Mat src = Mat::zeros(50000, 50000, CV_8U), dist;
    distanceTransform(src.col(0), dist, DIST_L2, DIST_MASK_5);
    int nz = countNonZero(dist);
    EXPECT_EQ(nz, 0);
}

BIGDATA_TEST(Imgproc_DistanceTransform, issue_23895_5x5_labels)
{
    Mat src = Mat::zeros(50000, 50000, CV_8U), dist, labels;
    distanceTransform(src.col(0), dist, labels, DIST_L2, DIST_MASK_5);
    int nz = countNonZero(dist);
    EXPECT_EQ(nz, 0);
}

TEST(Imgproc_DistanceTransform, max_distance_3x3)
{
    Mat src = Mat::ones(1, 70000, CV_8U), dist;
    src.at<uint8_t>(0, 0) = 0;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_3);

    double minVal, maxVal;
    minMaxLoc(dist, &minVal, &maxVal);
    EXPECT_GE(maxVal, 65533);
}

TEST(Imgproc_DistanceTransform, max_distance_5x5)
{
    Mat src = Mat::ones(1, 70000, CV_8U), dist;
    src.at<uint8_t>(0, 0) = 0;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_5);

    double minVal, maxVal;
    minMaxLoc(dist, &minVal, &maxVal);
    EXPECT_GE(maxVal, 65533);
}

TEST(Imgproc_DistanceTransform, max_distance_5x5_labels)
{
    Mat src = Mat::ones(1, 70000, CV_8U), dist, labels;
    src.at<uint8_t>(0, 0) = 0;
    distanceTransform(src, dist, labels, DIST_L2, DIST_MASK_5);

    double minVal, maxVal;
    minMaxLoc(dist, &minVal, &maxVal);
    EXPECT_GE(maxVal, 65533);
}

TEST(Imgproc_DistanceTransform, precise_long_dist)
{
    static const int maxDist = 1 << 16;
    Mat src = Mat::ones(1, 70000, CV_8U), dist;
    src.at<uint8_t>(0, 0) = 0;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE, CV_32F);

    Mat expected(src.size(), CV_32F);
    std::iota(expected.begin<float>(), expected.end<float>(), 0.f);
    expected.colRange(maxDist, expected.cols).setTo(maxDist);

    EXPECT_EQ(cv::norm(expected, dist, NORM_INF), 0);
}

TEST(Imgproc_DistanceTransform, ipp_deterministic_corner)
{
    setNumThreads(1);

    Mat src(1, 4096, CV_8U, Scalar(255)), dist;
    src.at<uint8_t>(0, 0) = 0;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE);
    for (int i = 0; i < src.cols; ++i)
    {
        float expected = static_cast<float>(i);
        ASSERT_EQ(expected, dist.at<float>(0, i)) << cv::format("diff: %e", expected - dist.at<float>(0, i));
    }
}

TEST(Imgproc_DistanceTransform, ipp_deterministic)
{
    setNumThreads(1);
    RNG& rng = TS::ptr()->get_rng();
    Mat src(1, 800, CV_8U, Scalar(255)), dist;
    int p1 = cvtest::randInt(rng) % src.cols;
    int p2 = cvtest::randInt(rng) % src.cols;
    int p3 = cvtest::randInt(rng) % src.cols;
    src.at<uint8_t>(0, p1) = 0;
    src.at<uint8_t>(0, p2) = 0;
    src.at<uint8_t>(0, p3) = 0;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE);
    for (int i = 0; i < src.cols; ++i)
    {
        float expected = static_cast<float>(min(min(abs(i - p1), abs(i - p2)), abs(i - p3)));
        ASSERT_EQ(expected, dist.at<float>(0, i)) << cv::format("diff: %e", expected - dist.at<float>(0, i));
    }
}

}} // namespace
