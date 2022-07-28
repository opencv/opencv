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
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//      Nghia Ho, nghiaho12@yahoo.com
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
//   * The name of OpenCV Foundation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
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

#define ACCURACY 0.00001

// See pics/intersection.png for the scenarios we are testing

// Test the following scenarios:
// 1 - no intersection
// 2 - partial intersection, rectangle translated
// 3 - partial intersection, rectangle rotated 45 degree on the corner, forms a triangle intersection
// 4 - full intersection, rectangles of same size directly on top of each other
// 5 - partial intersection, rectangle on top rotated 45 degrees
// 6 - partial intersection, rectangle on top of different size
// 7 - full intersection, rectangle fully enclosed in the other
// 8 - partial intersection, rectangle corner just touching. point contact
// 9 - partial intersection. rectangle side by side, line contact

static void compare(const std::vector<Point2f>& test, const std::vector<Point2f>& target)
{
    ASSERT_EQ(test.size(), target.size());
    ASSERT_TRUE(test.size() < 4 || isContourConvex(test));
    ASSERT_TRUE(target.size() < 4 || isContourConvex(target));
    for( size_t i = 0; i < test.size(); i++ )
    {
        double r = sqrt(normL2Sqr<double>(test[i] - target[i]));
        ASSERT_LT(r, ACCURACY);
    }
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_1)
{
    // no intersection
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 12.0f);
    RotatedRect rect2(Point2f(10, 10), Size2f(2, 2), 34.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_NONE);
    CV_Assert(vertices.empty());
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_2)
{
    // partial intersection, rectangles translated
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(1, 1), Size2f(2, 2), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(4);
    targetVertices[0] = Point2f(1.0f, 0.0f);
    targetVertices[1] = Point2f(1.0f, 1.0f);
    targetVertices[2] = Point2f(0.0f, 1.0f);
    targetVertices[3] = Point2f(0.0f, 0.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_3)
{
    // partial intersection, rectangles rotated 45 degree on the corner, forms a triangle intersection
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(1, 1), Size2f(sqrt(2.0f), 20), 45.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(3);
    targetVertices[0] = Point2f(1.0f, 0.0f);
    targetVertices[1] = Point2f(1.0f, 1.0f);
    targetVertices[2] = Point2f(0.0f, 1.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_4)
{
    // full intersection, rectangles of same size directly on top of each other
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(0, 0), Size2f(2, 2), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_FULL);

    vector<Point2f> targetVertices(4);
    targetVertices[0] = Point2f(-1.0f, 1.0f);
    targetVertices[1] = Point2f(-1.0f, -1.0f);
    targetVertices[2] = Point2f(1.0f, -1.0f);
    targetVertices[3] = Point2f(1.0f, 1.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_5)
{
    // partial intersection, rectangle on top rotated 45 degrees
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(0, 0), Size2f(2, 2), 45.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(8);
    targetVertices[0] = Point2f(-1.0f, -0.414214f);
    targetVertices[1] = Point2f(-0.414214f, -1.0f);
    targetVertices[2] = Point2f(0.414214f, -1.0f);
    targetVertices[3] = Point2f(1.0f, -0.414214f);
    targetVertices[4] = Point2f(1.0f, 0.414214f);
    targetVertices[5] = Point2f(0.414214f, 1.0f);
    targetVertices[6] = Point2f(-0.414214f, 1.0f);
    targetVertices[7] = Point2f(-1.0f, 0.414214f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_6)
{
    // 6 - partial intersection, rectangle on top of different size
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(0, 0), Size2f(2, 10), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(4);
    targetVertices[0] = Point2f(-1.0f, -1.0f);
    targetVertices[1] = Point2f(1.0f, -1.0f);
    targetVertices[2] = Point2f(1.0f, 1.0f);
    targetVertices[3] = Point2f(-1.0f, 1.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_7)
{
    // full intersection, rectangle fully enclosed in the other
    RotatedRect rect1(Point2f(0, 0), Size2f(12.34f, 56.78f), 0.0f);
    RotatedRect rect2(Point2f(0, 0), Size2f(2, 2), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_FULL);

    vector<Point2f> targetVertices(4);
    targetVertices[0] = Point2f(-1.0f, 1.0f);
    targetVertices[1] = Point2f(-1.0f, -1.0f);
    targetVertices[2] = Point2f(1.0f, -1.0f);
    targetVertices[3] = Point2f(1.0f, 1.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_8)
{
    // intersection by a single vertex
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(2, 2), Size2f(2, 2), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    compare(vertices, vector<Point2f>(1, Point2f(1.0f, 1.0f)));
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_9)
{
    // full intersection, rectangle fully enclosed in the other
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(2, 0), Size2f(2, 123.45f), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(2);
    targetVertices[0] = Point2f(1.0f, -1.0f);
    targetVertices[1] = Point2f(1.0f, 1.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_10)
{
    // three points of rect2 are inside rect1.
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(0, 0.5), Size2f(1, 1), 45.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(5);
    targetVertices[0] = Point2f(0.207107f, 1.0f);
    targetVertices[1] = Point2f(-0.207107f, 1.0f);
    targetVertices[2] = Point2f(-0.707107f, 0.5f);
    targetVertices[3] = Point2f(0.0f, -0.207107f);
    targetVertices[4] = Point2f(0.707107f, 0.5f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_11)
{
    RotatedRect rect1(Point2f(0, 0), Size2f(4, 2), 0.0f);
    RotatedRect rect2(Point2f(0, 0), Size2f(2, 2), -45.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(6);
    targetVertices[0] = Point2f(-0.414214f, -1.0f);
    targetVertices[1] = Point2f(0.414213f, -1.0f);
    targetVertices[2] = Point2f(1.41421f, 0.0f);
    targetVertices[3] = Point2f(0.414214f, 1.0f);
    targetVertices[4] = Point2f(-0.414213f, 1.0f);
    targetVertices[5] = Point2f(-1.41421f, 0.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_12)
{
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(0, 1), Size2f(1, 1), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(4);
    targetVertices[0] = Point2f(-0.5f, 1.0f);
    targetVertices[1] = Point2f(-0.5f, 0.5f);
    targetVertices[2] = Point2f(0.5f, 0.5f);
    targetVertices[3] = Point2f(0.5f, 1.0f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_13)
{
    RotatedRect rect1(Point2f(0, 0), Size2f(1, 3), 0.0f);
    RotatedRect rect2(Point2f(0, 1), Size2f(3, 1), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);

    vector<Point2f> targetVertices(4);
    targetVertices[0] = Point2f(-0.5f, 0.5f);
    targetVertices[1] = Point2f(0.5f, 0.5f);
    targetVertices[2] = Point2f(0.5f, 1.5f);
    targetVertices[3] = Point2f(-0.5f, 1.5f);
    compare(vertices, targetVertices);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_14)
{
    const int kNumTests = 100;
    const float kWidth = 5;
    const float kHeight = 5;
    RotatedRect rects[2];
    std::vector<Point2f> inter;
    cv::RNG& rng = cv::theRNG();
    for (int i = 0; i < kNumTests; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            rects[j].center = Point2f(rng.uniform(0.0f, kWidth), rng.uniform(0.0f, kHeight));
            rects[j].size = Size2f(rng.uniform(1.0f, kWidth), rng.uniform(1.0f, kHeight));
            rects[j].angle = rng.uniform(0.0f, 360.0f);
        }
        int res = rotatedRectangleIntersection(rects[0], rects[1], inter);
        EXPECT_TRUE(res == INTERSECT_NONE || res == INTERSECT_PARTIAL || res == INTERSECT_FULL) << res;
        ASSERT_TRUE(inter.size() < 4 || isContourConvex(inter)) << inter;
    }
}

TEST(Imgproc_RotatedRectangleIntersection, regression_12221_1)
{
    RotatedRect r1(
        Point2f(259.65081787109375, 51.58895492553711),
        Size2f(5487.8779296875, 233.8921661376953),
        -29.488616943359375);
    RotatedRect r2(
        Point2f(293.70465087890625, 112.10154724121094),
        Size2f(5487.8896484375, 234.87368774414062),
        -31.27001953125);

    std::vector<Point2f> intersections;
    int interType = cv::rotatedRectangleIntersection(r1, r2, intersections);
    EXPECT_EQ(INTERSECT_PARTIAL, interType);
    EXPECT_LE(intersections.size(), (size_t)8);
}

TEST(Imgproc_RotatedRectangleIntersection, regression_12221_2)
{
    RotatedRect r1(
        Point2f(239.78500366210938, 515.72021484375),
        Size2f(70.23420715332031, 39.74684524536133),
        -42.86162567138672);
    RotatedRect r2(
        Point2f(242.4205322265625, 510.1195373535156),
        Size2f(66.85948944091797, 61.46455383300781),
        -9.840961456298828);

    std::vector<Point2f> intersections;
    int interType = cv::rotatedRectangleIntersection(r1, r2, intersections);
    EXPECT_EQ(INTERSECT_PARTIAL, interType);
    EXPECT_LE(intersections.size(), (size_t)8);
}

TEST(Imgproc_RotatedRectangleIntersection, accuracy_21659)
{
    float scaleFactor = 1000;//to challenge the normalizationScale in the algorithm
    cv::RectanglesIntersectTypes intersectionResult = cv::RectanglesIntersectTypes::INTERSECT_NONE;
    std::vector<cv::Point2f> intersection;
    double intersectionArea = 0;
    cv::RotatedRect r1 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    cv::RotatedRect r2;

    r2 = cv::RotatedRect(cv::Point2f(-2.f, -2.f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_NONE, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-0), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(1.5f, .5f)*scaleFactor, cv::Size2f(1.f, 2.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_PARTIAL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-0), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(1.5f, 1.5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_PARTIAL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-0), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_FULL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-r2.size.area()), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(.5f, .5f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_FULL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-r2.size.area()), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(2.f, .5f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_PARTIAL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-500000), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 45);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_PARTIAL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-828427), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(1.f, 1.f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 45);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_PARTIAL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-250000), 1e-1);

    //see #21659
    r1 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545063f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    r2 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545235f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(cv::RectanglesIntersectTypes::INTERSECT_PARTIAL, intersectionResult);
    ASSERT_LE(std::abs(intersectionArea-r1.size.area()), 1e-3);

    r1 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545063f + 0.01f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    r2 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545235f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_LE(std::abs(intersectionArea-r1.size.area()), 1e-1);

    r1 = cv::RotatedRect(cv::Point2f(45.0715866f, 39.8825722f), cv::Size2f(3.0f, 3.0f), 0.10067749f);
    r2 = cv::RotatedRect(cv::Point2f(45.0715866f, 39.8825874f), cv::Size2f(3.0f, 3.0f), 0.10067749f);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_LE(std::abs(intersectionArea-r1.size.area()), 1e-3);
}

TEST(Imgproc_RotatedRectangleIntersection, regression_18520)
{
    RotatedRect rr_empty(
        Point2f(2, 2),
        Size2f(0, 0),  // empty
        0);
    RotatedRect rr(
        Point2f(50, 50),
        Size2f(4, 4),
        0);

    {
        std::vector<Point2f> intersections;
        int interType = cv::rotatedRectangleIntersection(rr_empty, rr, intersections);
        EXPECT_EQ(INTERSECT_NONE, interType) << "rr_empty, rr";
        EXPECT_EQ((size_t)0, intersections.size()) << "rr_empty, rr";
    }
    {
        std::vector<Point2f> intersections;
        int interType = cv::rotatedRectangleIntersection(rr, rr_empty, intersections);
        EXPECT_EQ(INTERSECT_NONE, interType) << "rr, rr_empty";
        EXPECT_EQ((size_t)0, intersections.size()) << "rr, rr_empty";
    }
}

TEST(Imgproc_RotatedRectangleIntersection, regression_19824)
{
    RotatedRect r1(
        Point2f(246805.033f, 4002326.94f),
        Size2f(26.40587f, 6.20026f),
        -62.10156f);
    RotatedRect r2(
        Point2f(246805.122f, 4002326.59f),
        Size2f(27.4821f, 8.5361f),
        -56.33761f);

    std::vector<Point2f> intersections;
    int interType = cv::rotatedRectangleIntersection(r1, r2, intersections);
    EXPECT_EQ(INTERSECT_PARTIAL, interType);
    EXPECT_LE(intersections.size(), (size_t)7);
}

}} // namespace
