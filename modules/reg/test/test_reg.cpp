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
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include "opencv2/reg/mapaffine.hpp"
#include "opencv2/reg/mapshift.hpp"
#include "opencv2/reg/mapprojec.hpp"
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mappergradeuclid.hpp"
#include "opencv2/reg/mappergradsimilar.hpp"
#include "opencv2/reg/mappergradaffine.hpp"
#include "opencv2/reg/mappergradproj.hpp"
#include "opencv2/reg/mapperpyramid.hpp"

using namespace std;
using namespace cv;
using namespace cv::reg;


class RegTest : public testing::Test
{
public:
    void loadImage();

    void testShift();
    void testEuclidean();
    void testSimilarity();
    void testAffine();
    void testProjective();
private:
    Mat img1;
};

void RegTest::testShift()
{
    Mat img2;

    // Warp original image
    Vec<double, 2> shift(5., 5.);
    MapShift mapTest(shift);
    mapTest.warp(img1, img2);

    // Register
    MapperGradShift mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapShift* mapShift = dynamic_cast<MapShift*>(mapPtr.obj);
    cout << endl << "--- Testing shift mapper ---" << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapShift->getShift()) << endl;

    // Check accuracy
    Ptr<Map> mapInv;
    mapShift->inverseMap(mapInv);
    mapTest.compose(*mapInv.obj);
    double shNorm = norm(mapTest.getShift());
    EXPECT_LE(shNorm, 0.1);
}

void RegTest::testEuclidean()
{
    Mat img2;

    // Warp original image
    double theta = 3*CV_PI/180;
    double cosT = cos(theta);
    double sinT = sin(theta);
    Matx<double, 2, 2> linTr(cosT, -sinT, sinT, cosT);
    Vec<double, 2> shift(5., 5.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);

    // Register
    MapperGradEuclid mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    cout << endl << "--- Testing Euclidean mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Check accuracy
    Ptr<Map> mapInv;
    mapAff->inverseMap(mapInv);
    mapTest.compose(*mapInv.obj);
    double shNorm = norm(mapTest.getShift());
    EXPECT_LE(shNorm, 0.1);
    double linTrNorm = norm(mapTest.getLinTr());
    EXPECT_LE(linTrNorm, sqrt(2.) + 0.01);
    EXPECT_GE(linTrNorm, sqrt(2.) - 0.01);
}

void RegTest::testSimilarity()
{
    Mat img2;

    // Warp original image
    double theta = 3*CV_PI/180;
    double scale = 0.95;
    double a = scale*cos(theta);
    double b = scale*sin(theta);
    Matx<double, 2, 2> linTr(a, -b, b, a);
    Vec<double, 2> shift(5., 5.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);

    // Register
    MapperGradSimilar mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    cout << endl << "--- Testing similarity mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Check accuracy
    Ptr<Map> mapInv;
    mapAff->inverseMap(mapInv);
    mapTest.compose(*mapInv.obj);
    double shNorm = norm(mapTest.getShift());
    EXPECT_LE(shNorm, 0.1);
    double linTrNorm = norm(mapTest.getLinTr());
    EXPECT_LE(linTrNorm, sqrt(2.) + 0.01);
    EXPECT_GE(linTrNorm, sqrt(2.) - 0.01);
}

void RegTest::testAffine()
{
    Mat img2;

    // Warp original image
    Matx<double, 2, 2> linTr(1., 0.1, -0.01, 1.);
    Vec<double, 2> shift(1., 1.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);

    // Register
    MapperGradAffine mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    cout << endl << "--- Testing affine mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Check accuracy
    Ptr<Map> mapInv;
    mapAff->inverseMap(mapInv);
    mapTest.compose(*mapInv.obj);
    double shNorm = norm(mapTest.getShift());
    EXPECT_LE(shNorm, 0.1);
    double linTrNorm = norm(mapTest.getLinTr());
    EXPECT_LE(linTrNorm, sqrt(2.) + 0.01);
    EXPECT_GE(linTrNorm, sqrt(2.) - 0.01);
}


void RegTest::testProjective()
{
    Mat img2;

    // Warp original image
    Matx<double, 3, 3> projTr(1., 0., 0., 0., 1., 0., 0.0001, 0.0001, 1);
    MapProjec mapTest(projTr);
    mapTest.warp(img1, img2);

    // Register
    MapperGradProj mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.obj);
    mapProj->normalize();
    cout << endl << "--- Testing projective transformation mapper ---" << endl;
    cout << Mat(projTr) << endl;
    cout << Mat(mapProj->getProjTr()) << endl;

    // Check accuracy
    Ptr<Map> mapInv;
    mapProj->inverseMap(mapInv);
    mapTest.compose(*mapInv.obj);
    double projNorm = norm(mapTest.getProjTr());
    EXPECT_LE(projNorm, sqrt(3.) + 0.01);
    EXPECT_GE(projNorm, sqrt(3.) - 0.01);
}

void RegTest::loadImage()
{
    const string imageName = cvtest::TS::ptr()->get_data_path() + "home.png";

    img1 = imread(imageName, -1);
    ASSERT_TRUE(img1.data != 0);
    // Convert to double, 3 channels
    img1.convertTo(img1, CV_64FC3);
}


TEST_F(RegTest, shift)
{
    loadImage();
    testShift();
}

TEST_F(RegTest, euclidean)
{
    loadImage();
    testEuclidean();
}

TEST_F(RegTest, similarity)
{
    loadImage();
    testSimilarity();
}

TEST_F(RegTest, affine)
{
    loadImage();
    testAffine();
}

TEST_F(RegTest, projective)
{
    loadImage();
    testProjective();
}
