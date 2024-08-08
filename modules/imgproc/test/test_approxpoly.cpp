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

//
// TODO!!!:
//  check_slice (and/or check) seem(s) to be broken, or this is a bug in function
//  (or its inability to handle possible self-intersections in the generated contours).
//
//  At least, if // return TotalErrors;
//  is uncommented in check_slice, the test fails easily.
//  So, now (and it looks like since 0.9.6)
//  we only check that the set of vertices of the approximated polygon is
//  a subset of vertices of the original contour.
//

//Tests to make sure that unreasonable epsilon (error)
//values never get passed to the Douglas-Peucker algorithm.
TEST(Imgproc_ApproxPoly, bad_epsilon)
{
    std::vector<Point2f> inputPoints;
    inputPoints.push_back(Point2f(0.0f, 0.0f));
    std::vector<Point2f> outputPoints;

    double eps = std::numeric_limits<double>::infinity();
    ASSERT_ANY_THROW(approxPolyDP(inputPoints, outputPoints, eps, false));

    eps = 9e99;
    ASSERT_ANY_THROW(approxPolyDP(inputPoints, outputPoints, eps, false));

    eps = -1e-6;
    ASSERT_ANY_THROW(approxPolyDP(inputPoints, outputPoints, eps, false));

    eps = NAN;
    ASSERT_ANY_THROW(approxPolyDP(inputPoints, outputPoints, eps, false));
}

struct ApproxPolyN: public testing::Test
{
    void SetUp()
    {
        vector<vector<Point>> inputPoints = {
            {  {87, 103}, {100, 112}, {96, 138}, {80, 169}, {60, 183}, {38, 176}, {41, 145}, {56, 118}, {76, 104} },
            {  {196, 102}, {205, 118}, {174, 196}, {152, 207}, {102, 194}, {100, 175}, {131, 109} },
            {  {372, 101}, {377, 119}, {337, 238}, {324, 248}, {240, 229}, {199, 214}, {232, 123}, {245, 103} },
            {  {463, 86}, {563, 112}, {574, 135}, {596, 221}, {518, 298}, {412, 266}, {385, 164}, {462, 86} }
        };

        Mat image(600, 600, CV_8UC1, Scalar(0));

        for (vector<Point>& polygon : inputPoints) {
            polylines(image, { polygon }, true, Scalar(255), 1);
        }

        findContours(image, contours, RETR_LIST, CHAIN_APPROX_NONE);
    }

    vector<vector<Point>> contours;
};

TEST_F(ApproxPolyN, accuracyInt)
{
    vector<vector<Point>> rightCorners = {
        { {72, 187}, {37, 176}, {42, 127}, {133, 64} },
        { {168, 212}, {92, 192}, {131, 109}, {213, 100} },
        { {72, 187}, {37, 176}, {42, 127}, {133, 64} },
        { {384, 100}, {333, 251}, {197, 220}, {239, 103} },
        { {168, 212}, {92, 192}, {131, 109}, {213, 100} },
        { {333, 251}, {197, 220}, {239, 103}, {384, 100} },
        { {542, 6}, {596, 221}, {518, 299}, {312, 236} },
        { {596, 221}, {518, 299}, {312, 236}, {542, 6} }
    };
    EXPECT_EQ(rightCorners.size(), contours.size());

    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<Point> corners;
        approxPolyN(contours[i], corners, 4, -1, true);
        ASSERT_EQ(rightCorners[i], corners );
    }
}

TEST_F(ApproxPolyN, accuracyFloat)
{
    vector<vector<Point2f>> rightCorners = {
        { {72.f, 187.f}, {37.f, 176.f}, {42.f, 127.f}, {133.f, 64.f} },
        { {168.f, 212.f}, {92.f, 192.f}, {131.f, 109.f}, {213.f, 100.f} },
        { {72.f, 187.f}, {37.f, 176.f}, {42.f, 127.f}, {133.f, 64.f} },
        { {384.f, 100.f}, {333.f, 251.f}, {197.f, 220.f}, {239.f, 103.f} },
        { {168.f, 212.f}, {92.f, 192.f}, {131.f, 109.f}, {213.f, 100.f} },
        { {333.f, 251.f}, {197.f, 220.f}, {239.f, 103.f}, {384.f, 100.f} },
        { {542.f, 6.f}, {596.f, 221.f}, {518.f, 299.f}, {312.f, 236.f} },
        { {596.f, 221.f}, {518.f, 299.f}, {312.f, 236.f}, {542.f, 6.f} }
    };
    EXPECT_EQ(rightCorners.size(), contours.size());

    for (size_t i = 0; i < contours.size(); ++i) {
        std::vector<Point2f> corners;
        approxPolyN(contours[i], corners, 4, -1, true);
        EXPECT_LT(cvtest::norm(rightCorners[i], corners, NORM_INF), .5f);
    }
}

TEST_F(ApproxPolyN, bad_args)
{
    Mat contour(10, 1, CV_32FC2);
    vector<vector<Point>> bad_contours;
    vector<Point> corners;
    ASSERT_ANY_THROW(approxPolyN(contour, corners, 0));
    ASSERT_ANY_THROW(approxPolyN(contour, corners, 3, 0));
    ASSERT_ANY_THROW(approxPolyN(bad_contours, corners, 4));
}

}} // namespace
