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
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

using namespace std;
using namespace cv;

class CV_FuzzyImageTest : public cvtest::BaseTest
{
public:
    CV_FuzzyImageTest();
    ~CV_FuzzyImageTest();
protected:
    void run(int);
};

CV_FuzzyImageTest::CV_FuzzyImageTest()
{
}
CV_FuzzyImageTest::~CV_FuzzyImageTest() {}

void CV_FuzzyImageTest::run( int )
{
    string folder = string(ts->get_data_path()) + "fuzzy/";
    Mat orig = imread(folder + "orig.png");
    Mat exp1 = imread(folder + "exp1.png");
    Mat exp2 = imread(folder + "exp2.png");
    Mat exp3 = imread(folder + "exp3.png");
    Mat mask1 = imread(folder + "mask1.png");
    Mat mask2 = imread(folder + "mask2.png");

    if (orig.empty() || exp1.empty() || exp2.empty() || mask1.empty() || mask2.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    // Conversion because of comparison.

    orig.convertTo(orig, CV_32F);
    exp1.convertTo(exp1, CV_32F);
    exp2.convertTo(exp2, CV_32F);
    exp3.convertTo(exp3, CV_32F);

    Mat res1, res2,res3;
    ft::inpaint(orig, mask1, res1, 2, ft::LINEAR, ft::ONE_STEP);
    ft::inpaint(orig, mask2, res2, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(orig, mask2, res3, 2, ft::LINEAR, ft::ITERATIVE);

    Mat diff1, diff2, diff3;
    absdiff(orig, res1, diff1);
    absdiff(orig, res2, diff2);
    absdiff(orig, res3, diff3);

    double n1 = cvtest::norm(diff1.reshape(1), NORM_INF, mask1.reshape(1));
    double n2 = cvtest::norm(diff2.reshape(1), NORM_INF, mask2.reshape(1));
    double n3 = cvtest::norm(diff3.reshape(1), NORM_INF, mask2.reshape(1));

    if (n1 != 0 || n2 != 0 || n3 != 0)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        return;
    }

    absdiff(exp1, res1, diff1);
    absdiff(exp2, res2, diff2);
    absdiff(exp3, res3, diff3);

    n1 = cvtest::norm(diff1.reshape(1), NORM_INF, mask1.reshape(1));
    n2 = cvtest::norm(diff2.reshape(1), NORM_INF, mask2.reshape(1));
    n3 = cvtest::norm(diff3.reshape(1), NORM_INF, mask2.reshape(1));

    const int jpeg_thres = 3;
    if (n1 > jpeg_thres || n2 > jpeg_thres || n3 > jpeg_thres)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
        return;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Fuzzy_image, regression) { CV_FuzzyImageTest test; test.safe_run(); }
