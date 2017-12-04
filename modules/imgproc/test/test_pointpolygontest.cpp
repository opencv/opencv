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

using namespace cv;
using namespace std;

class CV_PointPolygonTestTest: public cvtest::ArrayTest
{
public:
    CV_PointPolygonTestTest();
    ~CV_PointPolygonTestTest();

protected:
    void run (int);
};

CV_PointPolygonTestTest::CV_PointPolygonTestTest() {}
CV_PointPolygonTestTest::~CV_PointPolygonTestTest() {}

template <typename T> vector<T> generate_contour()
{
    return vector<T>{
        T(0, 0),
        T(0, 100000),
        T(100000, 100000),
        T(100000, 50000),
        T(100000, 0)
    };
}

Point2f generate_point()
{
    return Point2f(40000, 40000);
}

template <typename T> bool CV_PointPolygonTestTest::require_is_inside()
{
    const auto contour = generate_contour<T>();
    const auto point = generate_point();
    const auto result = cv::pointPolygonTest(contour, point, false);

    if (result <= 0) {
        CV_Error(1, "Desired result: point is inside polygon - actual result: point is not inside polygon");
        return false;
    }

    return true;
}

void CV_PointPolygonTestTest::run(int)
{
    if (!require_is_inside<Point>()) return;
    require_is_inside<Point2f>();
}

TEST (Imgproc_PointPolygonTest, accuracy) { CV_PointPolygonTestTest test; test.safe_run(); }
