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
#include "opencv2/ts/ocl_test.hpp"

namespace opencv_test { namespace {

class CV_SmallContourMomentTest : public cvtest::BaseTest
{
public:
    CV_SmallContourMomentTest() {}
    ~CV_SmallContourMomentTest() {}
protected:
    void run(int)
    {
        try
        {
            vector<Point> points;
            points.push_back(Point(50, 56));
            points.push_back(Point(53, 53));
            points.push_back(Point(46, 54));
            points.push_back(Point(49, 51));

            Moments m = moments(points, false);
            double area = contourArea(points);

            CV_Assert( m.m00 == 0 && m.m01 == 0 && m.m10 == 0 && area == 0 );
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Imgproc_ContourMoment, small) { CV_SmallContourMomentTest test; test.safe_run(); }

TEST(Imgproc_Moments, degenerateContours)
{
    std::vector<cv::Point> c1;
    c1.push_back(cv::Point(10,10));
    cv::Moments m1 = cv::moments(c1, false);
    EXPECT_EQ(m1.m00, 0);

    std::vector<cv::Point> c2;
    c2.push_back(cv::Point(0,0));
    c2.push_back(cv::Point(5,5));
    c2.push_back(cv::Point(10,10));
    cv::Moments m2 = cv::moments(c2, false);
    EXPECT_EQ(m2.m00, 0);
}

TEST(Geometry_MatchShapes, reflectedContour)
{
    const std::vector<Point> triangle = {{0, 0}, {5, 0}, {1, 2}};
    const std::vector<Point> reflection = {{0, 0}, {-5, 0}, {-1, 2}};

    // Reflection changes only the sign of h7, whose magnitude is 8 / 4100625.
    const double logMagnitude = std::abs(std::log10(8.0 / 4100625));
    const double distances[] = {2 / logMagnitude, 2 * logMagnitude, 2};
    double hu[7];
    HuMoments(moments(triangle), hu);

    for (int method : {CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3})
    {
        SCOPED_TRACE(method);
        EXPECT_NEAR(0.0, matchShapes(triangle, reflection, method, 0), 1e-12);
        EXPECT_NEAR(0.0, matchShapes(triangle, reflection, method, 0, 1e-5), 1e-12);
        EXPECT_NEAR(0.0, matchShapes(triangle, reflection, method, 0, std::abs(hu[6])), 1e-12);
        for (double eps : {1e-6, 0.0})
            EXPECT_NEAR(distances[method - CONTOURS_MATCH_I1], matchShapes(triangle, reflection, method, 0, eps), 1e-12);
    }
}

TEST(Geometry_MatchShapes, pairwiseThreshold)
{
    const std::vector<Point> square = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    const std::vector<Point> triangle = {{0, 0}, {5, 0}, {1, 2}};
    // Only h1 is nonzero in the square. The two h1 values are 1/6 and 5/18.
    const double expected = std::log10(5.0 / 3.0);
    const int method = CONTOURS_MATCH_I2;
    const double legacy = matchShapes(square, triangle, method, 1);
    EXPECT_NEAR(expected, legacy, 1e-12);
    EXPECT_EQ(legacy, matchShapes(square, triangle, method, 0, 1e-5));
    EXPECT_NEAR(expected, matchShapes(square, triangle, method, 1, 0.0), 1e-12);
    for (int mode : {CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3})
    {
        SCOPED_TRACE(mode);
        EXPECT_EQ(0.0, matchShapes(square, triangle, mode, 0, 0.2));
        EXPECT_EQ(0.0, matchShapes(triangle, square, mode, 0, 0.2));
    }
}

TEST(Geometry_MatchShapes, degenerateContours)
{
    const std::vector<Point> line = {{0, 0}, {1, 1}, {2, 2}};
    const std::vector<Point> triangle = {{0, 0}, {5, 0}, {1, 2}};
    for (int method : {CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3})
    {
        SCOPED_TRACE(method);
        for (double eps : {0.0, 1e-5, 1.0})
        {
            SCOPED_TRACE(eps);
            EXPECT_EQ(0.0, matchShapes(line, line, method, 0, eps));
            EXPECT_EQ(DBL_MAX, matchShapes(line, triangle, method, 0, eps));
            EXPECT_EQ(DBL_MAX, matchShapes(triangle, line, method, 0, eps));
        }
    }
}

TEST(Geometry_MatchShapes, invalidThreshold)
{
    const std::vector<Point> triangle = {{0, 0}, {5, 0}, {1, 2}};
    for (double eps : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(matchShapes(triangle, triangle, CONTOURS_MATCH_I2, 0, eps), cv::Exception);
}

}} // namespace
