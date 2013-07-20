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

#include "perf_precomp.hpp"
#include "opencv2/ts.hpp"
#include "opencv2/ts/gpu_perf.hpp"

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
using namespace std::tr1;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::reg;


Vec<double, 2> perfShift(const Mat& img1)
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

    MapShift* mapShift = dynamic_cast<MapShift*>(mapPtr.obj);
    return mapShift->getShift();
}

Matx<double, 2, 6> perfEuclidean(const Mat& img1)
{
    Mat img2;
    Matx<double, 2, 6> transf;

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

    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    Matx<double, 2, 2> resLinTr = mapAff->getLinTr();
    transf(0, 0) = resLinTr(0, 0), transf(0, 1) = resLinTr(0, 1);
    transf(1, 0) = resLinTr(1, 0), transf(1, 1) = resLinTr(1, 1);
    Vec<double, 2> resShift = mapAff->getShift();
    transf(0, 2) = resShift(0);
    transf(1, 2) = resShift(1);
    return transf;
}

Matx<double, 2, 6> perfSimilarity(const Mat& img1)
{
    Mat img2;
    Matx<double, 2, 6> transf;

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

    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    Matx<double, 2, 2> resLinTr = mapAff->getLinTr();
    transf(0, 0) = resLinTr(0, 0), transf(0, 1) = resLinTr(0, 1);
    transf(1, 0) = resLinTr(1, 0), transf(1, 1) = resLinTr(1, 1);
    Vec<double, 2> resShift = mapAff->getShift();
    transf(0, 2) = resShift(0);
    transf(1, 2) = resShift(1);
    return transf;
}

Matx<double, 2, 6> perfAffine(const Mat& img1)
{
    Mat img2;
    Matx<double, 2, 6> transf;

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

    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    Matx<double, 2, 2> resLinTr = mapAff->getLinTr();
    transf(0, 0) = resLinTr(0, 0), transf(0, 1) = resLinTr(0, 1);
    transf(1, 0) = resLinTr(1, 0), transf(1, 1) = resLinTr(1, 1);
    Vec<double, 2> resShift = mapAff->getShift();
    transf(0, 2) = resShift(0);
    transf(1, 2) = resShift(1);
    return transf;
}

Matx<double, 3, 3> perfProjective(const Mat& img1)
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

    MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.obj);
    mapProj->normalize();
    return mapProj->getProjTr();
}


PERF_TEST_P(Size_MatType, Registration_Shift,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_64FC1), MatType(CV_64FC3))))
{
    declare.time(60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    Vec<double, 2> shift;
    declare.in(frame, WARMUP_RNG).out(shift);

    TEST_CYCLE() shift = perfShift(frame);

    SANITY_CHECK(shift);
}

PERF_TEST_P(Size_MatType, Registration_Euclidean,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_64FC1), MatType(CV_64FC3))))
{
    declare.time(60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    Matx<double, 2, 6> result;
    declare.in(frame, WARMUP_RNG).out(result);

    TEST_CYCLE() result = perfEuclidean(frame);

    SANITY_CHECK(result);
}

PERF_TEST_P(Size_MatType, Registration_Similarity,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_64FC1), MatType(CV_64FC3))))
{
    declare.time(60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    Matx<double, 2, 6> result;
    declare.in(frame, WARMUP_RNG).out(result);

    TEST_CYCLE() result = perfSimilarity(frame);

    SANITY_CHECK(result);
}

PERF_TEST_P(Size_MatType, Registration_Affine,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_64FC1), MatType(CV_64FC3))))
{
    declare.time(60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    Matx<double, 2, 6> result;
    declare.in(frame, WARMUP_RNG).out(result);

    TEST_CYCLE() result = perfAffine(frame);

    SANITY_CHECK(result);
}

PERF_TEST_P(Size_MatType, Registration_Projective,
            Combine(Values(szSmall64, szSmall128),
                    Values(MatType(CV_64FC1), MatType(CV_64FC3))))
{
    declare.time(60);

    const Size size = get<0>(GetParam());
    const int type = get<1>(GetParam());

    Mat frame(size, type);
    Matx<double, 3, 3> result;
    declare.in(frame, WARMUP_RNG).out(result);

    TEST_CYCLE() result = perfProjective(frame);

    SANITY_CHECK(result);
}
