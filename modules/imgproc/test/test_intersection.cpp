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

class CV_RotatedRectangleIntersectionTest: public cvtest::ArrayTest
{
public:

protected:
    void run (int);

private:
    void test1();
    void test2();
    void test3();
    void test4();
    void test5();
    void test6();
    void test7();
    void test8();
    void test9();
    void test10();
    void test11();
    void test12();
    void test13();
    void test14();
};

static void compare(const std::vector<Point2f>& test, const std::vector<Point2f>& target)
{
    ASSERT_EQ(test.size(), target.size());
    ASSERT_TRUE(test.size() < 4 || isContourConvex(test));
    ASSERT_TRUE(target.size() < 4 || isContourConvex(target));
    for( size_t i = 0; i < test.size(); i++ )
    {
        double dx = test[i].x - target[i].x;
        double dy = test[i].y - target[i].y;
        double r = sqrt(dx*dx + dy*dy);
        ASSERT_LT(r, ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::run(int)
{
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
    // 9 - partial intersetion. rectangle side by side, line contact

    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();
    test11();
    test12();
    test13();
    test14();
}

void CV_RotatedRectangleIntersectionTest::test1()
{
    // no intersection
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 12.0f);
    RotatedRect rect2(Point2f(10, 10), Size2f(2, 2), 34.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_NONE);
    CV_Assert(vertices.empty());
}

void CV_RotatedRectangleIntersectionTest::test2()
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

void CV_RotatedRectangleIntersectionTest::test3()
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

void CV_RotatedRectangleIntersectionTest::test4()
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

void CV_RotatedRectangleIntersectionTest::test5()
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

void CV_RotatedRectangleIntersectionTest::test6()
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

void CV_RotatedRectangleIntersectionTest::test7()
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

void CV_RotatedRectangleIntersectionTest::test8()
{
    // intersection by a single vertex
    RotatedRect rect1(Point2f(0, 0), Size2f(2, 2), 0.0f);
    RotatedRect rect2(Point2f(2, 2), Size2f(2, 2), 0.0f);

    vector<Point2f> vertices;
    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    compare(vertices, vector<Point2f>(1, Point2f(1.0f, 1.0f)));
}

void CV_RotatedRectangleIntersectionTest::test9()
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

void CV_RotatedRectangleIntersectionTest::test10()
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

void CV_RotatedRectangleIntersectionTest::test11()
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

void CV_RotatedRectangleIntersectionTest::test12()
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

void CV_RotatedRectangleIntersectionTest::test13()
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

void CV_RotatedRectangleIntersectionTest::test14()
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

TEST (Imgproc_RotatedRectangleIntersection, accuracy) { CV_RotatedRectangleIntersectionTest test; test.safe_run(); }

}} // namespace
