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

class Core_RotatedRectConstructorTest : public cvtest::BaseTest
{
public:
    Core_RotatedRectConstructorTest();
protected:
    int prepare_test_case( int );
    void run_func();
    int validate_test_results( int );
    float MAX_COORD_VAL;
    Point2f a, b, c;
    RotatedRect rec;
};

Core_RotatedRectConstructorTest::Core_RotatedRectConstructorTest()
{
    test_case_count = 100;
    MAX_COORD_VAL = 1000.0f;
}

int Core_RotatedRectConstructorTest::prepare_test_case( int test_case_idx )
{
    cvtest::BaseTest::prepare_test_case( test_case_idx );
    RNG& rng = ts->get_rng();
    a = Point2f( rng.uniform(-MAX_COORD_VAL, MAX_COORD_VAL), rng.uniform(-MAX_COORD_VAL, MAX_COORD_VAL) );
    do
    {
        b = Point2f( rng.uniform(-MAX_COORD_VAL, MAX_COORD_VAL), rng.uniform(-MAX_COORD_VAL, MAX_COORD_VAL) );
    }
    while( cv::norm(a - b) <= FLT_EPSILON );
    Vec2f along(a - b);
    Vec2f perp = Vec2f(-along[1], along[0]);
    double d = (double) rng.uniform(1.0f, 5.0f);
    if( cvtest::randInt(rng) % 2 == 0 ) d = -d;
    c = Point2f( (float) ((double) b.x + d * perp[0]), (float) ((double) b.y + d * perp[1]) );
    return 1;
}

void Core_RotatedRectConstructorTest::run_func()
{
    rec = RotatedRect(a, b, c);
}

int Core_RotatedRectConstructorTest::validate_test_results( int )
{
    Point2f vertices[4];
    rec.points(vertices);
    int count_match = 0;
    for( int i = 0; i < 4; i++ )
    {
        if( cv::norm(vertices[i] - a) <= 0.001 ) count_match++;
        else if( cv::norm(vertices[i] - b) <= 0.001 ) count_match++;
        else if( cv::norm(vertices[i] - c) <= 0.001 ) count_match++;
    }
    if( count_match == 3 )
        return cvtest::TS::OK;
    ts->printf( cvtest::TS::LOG, "RotatedRect end points don't match those supplied in constructor");
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    return cvtest::TS::OK;
}

TEST(Core_RotatedRect, three_point_constructor) { Core_RotatedRectConstructorTest test; test.safe_run(); }

TEST(Core_RotatedRect, intersection)
{
    float scaleFactor = 1000;//to challenge the numericalScalingFactor in the algorithm
    cv::RectanglesIntersectTypes intersectionResult = cv::RectanglesIntersectTypes::INTERSECT_NONE;
    std::vector<cv::Point2f> intersection;
    double intersectionArea = 0;
    cv::RotatedRect r1 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    cv::RotatedRect r2;

    r2 = cv::RotatedRect(cv::Point2f(-2.f, -2.f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_NONE);
    ASSERT_LE(std::abs(intersectionArea-0), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(1.5f, .5f)*scaleFactor, cv::Size2f(1.f, 2.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_PARTIAL);
    ASSERT_LE(std::abs(intersectionArea-0), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(1.5f, 1.5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_PARTIAL);
    ASSERT_LE(std::abs(intersectionArea-0), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_FULL);
    ASSERT_LE(std::abs(intersectionArea-r2.size.area()), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(.5f, .5f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_FULL);
    ASSERT_LE(std::abs(intersectionArea-r2.size.area()), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(2.f, .5f)*scaleFactor, 0);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_PARTIAL);
    ASSERT_LE(std::abs(intersectionArea-500000), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(.5f, .5f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 45);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_PARTIAL);
    ASSERT_LE(std::abs(intersectionArea-828427), 1e-1);

    r2 = cv::RotatedRect(cv::Point2f(1.f, 1.f)*scaleFactor, cv::Size2f(1.f, 1.f)*scaleFactor, 45);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_PARTIAL);
    ASSERT_LE(std::abs(intersectionArea-250000), 1e-1);

    //see #21659
    r1 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545063f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    r2 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545235f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_EQ(intersectionResult, cv::RectanglesIntersectTypes::INTERSECT_PARTIAL);
    ASSERT_LE(std::abs(intersectionArea-r1.size.area()), 1e-3);

    r1 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545063f + 0.01f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    r2 = cv::RotatedRect(cv::Point2f(4.48589373f, 12.5545235f), cv::Size2f(4.0f, 4.0f), 0.0347290039f);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_LE(std::abs(intersectionArea-r1.size.area()), 1e-1);

    r1 = cv::RotatedRect(cv::Point2f(45.0715866, 39.8825722), cv::Size2f(3.0f, 3.0f), 0.10067749f);
    r2 = cv::RotatedRect(cv::Point2f(45.0715866, 39.8825874), cv::Size2f(3.0f, 3.0f), 0.10067749f);
    intersectionResult = (cv::RectanglesIntersectTypes) cv::rotatedRectangleIntersection(r1, r2, intersection);
    intersectionArea = (intersection.size() <= 2) ? 0. : cv::contourArea(intersection);
    ASSERT_LE(std::abs(intersectionArea-r1.size.area()), 1e-3);
}

}} // namespace
