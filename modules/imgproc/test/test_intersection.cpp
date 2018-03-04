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
};

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
}

void CV_RotatedRectangleIntersectionTest::test1()
{
    // no intersection

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 12.0f;

    rect2.center.x = 10;
    rect2.center.y = 10;
    rect2.size.width = 2;
    rect2.size.height = 2;
    rect2.angle = 34.0f;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_NONE);
    CV_Assert(vertices.empty());
}

void CV_RotatedRectangleIntersectionTest::test2()
{
    // partial intersection, rectangles translated

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 1;
    rect2.center.y = 1;
    rect2.size.width = 2;
    rect2.size.height = 2;
    rect2.angle = 0;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    CV_Assert(vertices.size() == 4);

    vector<Point2f> possibleVertices(4);

    possibleVertices[0] = Point2f(0.0f, 0.0f);
    possibleVertices[1] = Point2f(1.0f, 1.0f);
    possibleVertices[2] = Point2f(0.0f, 1.0f);
    possibleVertices[3] = Point2f(1.0f, 0.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::test3()
{
    // partial intersection, rectangles rotated 45 degree on the corner, forms a triangle intersection
    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 1;
    rect2.center.y = 1;
    rect2.size.width = sqrt(2.0f);
    rect2.size.height = 20;
    rect2.angle = 45.0f;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    CV_Assert(vertices.size() == 3);

    vector<Point2f> possibleVertices(3);

    possibleVertices[0] = Point2f(1.0f, 1.0f);
    possibleVertices[1] = Point2f(0.0f, 1.0f);
    possibleVertices[2] = Point2f(1.0f, 0.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::test4()
{
    // full intersection, rectangles of same size directly on top of each other

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 0;
    rect2.center.y = 0;
    rect2.size.width = 2;
    rect2.size.height = 2;
    rect2.angle = 0;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_FULL);
    CV_Assert(vertices.size() == 4);

    vector<Point2f> possibleVertices(4);

    possibleVertices[0] = Point2f(-1.0f, 1.0f);
    possibleVertices[1] = Point2f(1.0f, -1.0f);
    possibleVertices[2] = Point2f(-1.0f, -1.0f);
    possibleVertices[3] = Point2f(1.0f, 1.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::test5()
{
    // partial intersection, rectangle on top rotated 45 degrees

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 0;
    rect2.center.y = 0;
    rect2.size.width = 2;
    rect2.size.height = 2;
    rect2.angle = 45.0f;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    CV_Assert(vertices.size() == 8);

    vector<Point2f> possibleVertices(8);

    possibleVertices[0] = Point2f(-1.0f, -0.414214f);
    possibleVertices[1] = Point2f(-1.0f, 0.414214f);
    possibleVertices[2] = Point2f(-0.414214f, -1.0f);
    possibleVertices[3] = Point2f(0.414214f, -1.0f);
    possibleVertices[4] = Point2f(1.0f, -0.414214f);
    possibleVertices[5] = Point2f(1.0f, 0.414214f);
    possibleVertices[6] = Point2f(0.414214f, 1.0f);
    possibleVertices[7] = Point2f(-0.414214f, 1.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::test6()
{
    // 6 - partial intersection, rectangle on top of different size

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 0;
    rect2.center.y = 0;
    rect2.size.width = 2;
    rect2.size.height = 10;
    rect2.angle = 0;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    CV_Assert(vertices.size() == 4);

    vector<Point2f> possibleVertices(4);

    possibleVertices[0] = Point2f(1.0f, 1.0f);
    possibleVertices[1] = Point2f(1.0f, -1.0f);
    possibleVertices[2] = Point2f(-1.0f, -1.0f);
    possibleVertices[3] = Point2f(-1.0f, 1.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::test7()
{
    // full intersection, rectangle fully enclosed in the other

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 12.34f;
    rect1.size.height = 56.78f;
    rect1.angle = 0;

    rect2.center.x = 0;
    rect2.center.y = 0;
    rect2.size.width = 2;
    rect2.size.height = 2;
    rect2.angle = 0;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_FULL);
    CV_Assert(vertices.size() == 4);

    vector<Point2f> possibleVertices(4);

    possibleVertices[0] = Point2f(1.0f, 1.0f);
    possibleVertices[1] = Point2f(1.0f, -1.0f);
    possibleVertices[2] = Point2f(-1.0f, -1.0f);
    possibleVertices[3] = Point2f(-1.0f, 1.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

void CV_RotatedRectangleIntersectionTest::test8()
{
    // full intersection, rectangle fully enclosed in the other

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 2;
    rect2.center.y = 2;
    rect2.size.width = 2;
    rect2.size.height = 2;
    rect2.angle = 0;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    CV_Assert(vertices.size() == 1);

    double dx = vertices[0].x - 1;
    double dy = vertices[0].y - 1;
    double r = sqrt(dx*dx + dy*dy);

    CV_Assert(r < ACCURACY);
}

void CV_RotatedRectangleIntersectionTest::test9()
{
    // full intersection, rectangle fully enclosed in the other

    RotatedRect rect1, rect2;

    rect1.center.x = 0;
    rect1.center.y = 0;
    rect1.size.width = 2;
    rect1.size.height = 2;
    rect1.angle = 0;

    rect2.center.x = 2;
    rect2.center.y = 0;
    rect2.size.width = 2;
    rect2.size.height = 123.45f;
    rect2.angle = 0;

    vector<Point2f> vertices;

    int ret = rotatedRectangleIntersection(rect1, rect2, vertices);

    CV_Assert(ret == INTERSECT_PARTIAL);
    CV_Assert(vertices.size() == 2);

    vector<Point2f> possibleVertices(2);

    possibleVertices[0] = Point2f(1.0f, 1.0f);
    possibleVertices[1] = Point2f(1.0f, -1.0f);

    for( size_t i = 0; i < vertices.size(); i++ )
    {
        double bestR = DBL_MAX;

        for( size_t j = 0; j < possibleVertices.size(); j++ )
        {
            double dx = vertices[i].x - possibleVertices[j].x;
            double dy = vertices[i].y - possibleVertices[j].y;
            double r = sqrt(dx*dx + dy*dy);

            bestR = std::min(bestR, r);
        }

        CV_Assert(bestR < ACCURACY);
    }
}

TEST (Imgproc_RotatedRectangleIntersection, accuracy) { CV_RotatedRectangleIntersectionTest test; test.safe_run(); }

}} // namespace
