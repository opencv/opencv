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

namespace opencv_test { namespace {

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
    cv::cvtColor(mask, mask1ch, COLOR_BGR2GRAY);

    Mat test = orig.clone();
    test.setTo(Scalar::all(255), mask1ch);

    Mat res1, res2;
    inpaint( test, mask1ch, res1, 5, INPAINT_NS );
    inpaint( test, mask1ch, res2, 5, INPAINT_TELEA );

    Mat diff1, diff2;
    absdiff( orig, res1, diff1 );
    absdiff( orig, res2, diff2 );

    double n1 = cvtest::norm(diff1.reshape(1), NORM_INF, inv_mask.reshape(1));
    double n2 = cvtest::norm(diff2.reshape(1), NORM_INF, inv_mask.reshape(1));

    if (n1 != 0 || n2 != 0)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        return;
    }

    absdiff( exp1, res1, diff1 );
    absdiff( exp2, res2, diff2 );

    n1 = cvtest::norm(diff1.reshape(1), NORM_INF, mask.reshape(1));
    n2 = cvtest::norm(diff2.reshape(1), NORM_INF, mask.reshape(1));

    const int jpeg_thres = 3;
    if (n1 > jpeg_thres || n2 > jpeg_thres)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        return;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Photo_Inpaint, regression) { CV_InpaintTest test; test.safe_run(); }

typedef testing::TestWithParam<tuple<perf::MatType> > formats;

TEST_P(formats, basic)
{
    const int type = get<0>(GetParam());
    Mat src(100, 100, type);
    src.setTo(Scalar::all(128));
    Mat ref = src.clone();
    Mat dst, mask = Mat::zeros(src.size(), CV_8U);

    circle(src, Point(50, 50), 5, Scalar::all(200), 6);
    circle(mask, Point(50, 50), 5, Scalar::all(200), 6);
    inpaint(src, mask, dst, 10, INPAINT_NS);

    Mat dst2;
    inpaint(src, mask, dst2, 10, INPAINT_TELEA);

    ASSERT_EQ(cv::norm(dst, ref, NORM_INF), 0.);
    ASSERT_EQ(cv::norm(dst2, ref, NORM_INF), 0.);
}

INSTANTIATE_TEST_CASE_P(Photo_Inpaint, formats, testing::Values(CV_32FC1, CV_16UC1, CV_8UC1, CV_8UC3));

TEST(Photo_InpaintBorders, regression)
{
    Mat img(64, 64, CV_8U);
    img = 128;
    img(Rect(0, 0, 16, 64)) = 0;

    Mat mask(64, 64, CV_8U);
    mask = 0;
    mask(Rect(0, 0, 16, 64)) = 255;

    Mat inpainted;
    inpaint(img, mask, inpainted, 1, INPAINT_TELEA);

    Mat diff;
    cv::absdiff(inpainted, 128*Mat::ones(inpainted.size(), inpainted.type()), diff);
    ASSERT_TRUE(countNonZero(diff) == 0);
}

typedef testing::TestWithParam<tuple<perf::MatType>> Photo_InpaintSmallBorders;

TEST_P(Photo_InpaintSmallBorders, regression)
{
    int type = get<0>(GetParam());
    Mat img(5, 5, type, Scalar::all(128));
    Mat expected = img.clone();

    Mat mask = Mat::zeros(5, 5, CV_8U);
    mask(Rect(1, 1, 3, 3)) = 255;

    img.setTo(Scalar::all(0), mask);

    Mat inpainted, diff;

    inpaint(img, mask, inpainted, 1, INPAINT_TELEA);
    cv::absdiff(inpainted, expected, diff);
    ASSERT_EQ(countNonZero(diff.reshape(1)), 0);

    inpaint(img, mask, inpainted, 1, INPAINT_NS);
    cv::absdiff(inpainted, expected, diff);
    ASSERT_EQ(countNonZero(diff.reshape(1)), 0);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Photo_InpaintSmallBorders,  Values(CV_8UC1, CV_8UC3));

}} // namespace
