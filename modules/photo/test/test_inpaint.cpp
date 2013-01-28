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
#include <string>

using namespace cv;

class CV_InpaintTest : public cvtest::BaseTest
{
public:
    CV_InpaintTest();
    ~CV_InpaintTest();
protected:
    void run(int);
};

CV_InpaintTest::CV_InpaintTest()
{
}
CV_InpaintTest::~CV_InpaintTest() {}

void CV_InpaintTest::run( int )
{
    string folder = string(ts->get_data_path()) + "inpaint/";
    Mat orig = imread(folder + "orig.png");
    Mat exp1 = imread(folder + "exp1.png");
    Mat exp2 = imread(folder + "exp2.png");
    Mat mask = imread(folder + "mask.png");

    if (orig.empty() || exp1.empty() || exp2.empty() || mask.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat inv_mask;
    mask.convertTo(inv_mask, CV_8UC3, -1.0, 255.0);

    Mat mask1ch;
    cv::cvtColor(mask, mask1ch, CV_BGR2GRAY);

    Mat test = orig.clone();
    test.setTo(Scalar::all(255), mask1ch);

    Mat res1, res2;
    inpaint( test, mask1ch, res1, 5, CV_INPAINT_NS );
    inpaint( test, mask1ch, res2, 5, CV_INPAINT_TELEA );

    Mat diff1, diff2;
    absdiff( orig, res1, diff1 );
    absdiff( orig, res2, diff2 );

    double n1 = norm(diff1.reshape(1), NORM_INF, inv_mask.reshape(1));
    double n2 = norm(diff2.reshape(1), NORM_INF, inv_mask.reshape(1));

    if (n1 != 0 || n2 != 0)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        return;
    }

    absdiff( exp1, res1, diff1 );
    absdiff( exp2, res2, diff2 );

    n1 = norm(diff1.reshape(1), NORM_INF, mask.reshape(1));
    n2 = norm(diff2.reshape(1), NORM_INF, mask.reshape(1));

    const int jpeg_thres = 3;
    if (n1 > jpeg_thres || n2 > jpeg_thres)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        return;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Photo_Inpaint, regression) { CV_InpaintTest test; test.safe_run(); }
