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

namespace cvtest
{

/// phase correlation
class CV_PhaseCorrelatorTest : public cvtest::ArrayTest
{
public:
    CV_PhaseCorrelatorTest();
protected:
    void run( int );
};

CV_PhaseCorrelatorTest::CV_PhaseCorrelatorTest() {}

void CV_PhaseCorrelatorTest::run( int )
{
    ts->set_failed_test_info(cvtest::TS::OK);

    Mat r1 = Mat::ones(Size(129, 128), CV_64F);
    Mat r2 = Mat::ones(Size(129, 128), CV_64F);

    double expectedShiftX = -10.0;
    double expectedShiftY = -20.0;

    // draw 10x10 rectangles @ (100, 100) and (90, 80) should see ~(-10, -20) shift here...
    cv::rectangle(r1, Point(100, 100), Point(110, 110), Scalar(0, 0, 0), CV_FILLED);
    cv::rectangle(r2, Point(90, 80), Point(100, 90), Scalar(0, 0, 0), CV_FILLED);

    Mat hann;
    createHanningWindow(hann, r1.size(), CV_64F);
    Point2d phaseShift = phaseCorrelate(r1, r2, hann);

    // test accuracy should be less than 1 pixel...
    if((expectedShiftX - phaseShift.x) >= 1 || (expectedShiftY - phaseShift.y) >= 1)
    {
         ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    }
}

TEST(Imgproc_PhaseCorrelatorTest, accuracy) { CV_PhaseCorrelatorTest test; test.safe_run(); }

}
