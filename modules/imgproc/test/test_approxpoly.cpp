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

}} // namespace
