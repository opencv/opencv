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
using namespace std;

class CV_ConnectedComponentsTest : public cvtest::BaseTest
{
public:
    CV_ConnectedComponentsTest();
    ~CV_ConnectedComponentsTest();
protected:
    void run(int);
};

CV_ConnectedComponentsTest::CV_ConnectedComponentsTest() {}
CV_ConnectedComponentsTest::~CV_ConnectedComponentsTest() {}

void CV_ConnectedComponentsTest::run( int /* start_from */)
{
    string exp_path = string(ts->get_data_path()) + "connectedcomponents/ccomp_exp.png";
    Mat exp = imread(exp_path, 0);
    Mat orig = imread(string(ts->get_data_path()) + "connectedcomponents/concentric_circles.png", 0);

    if (orig.empty())
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat bw = orig > 128;
    Mat labelImage;
    int nLabels = connectedComponents(bw, labelImage, 8, CV_32S);

    for(int r = 0; r < labelImage.rows; ++r){
        for(int c = 0; c < labelImage.cols; ++c){
            int l = labelImage.at<int>(r, c);
            bool pass = l >= 0 && l <= nLabels;
            if(!pass){
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
        }
    }

    if( exp.empty() || orig.size() != exp.size() )
    {
        imwrite(exp_path, labelImage);
        exp = labelImage;
    }

    if (0 != norm(labelImage > 0, exp > 0, NORM_INF))
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        return;
    }
    if (nLabels != norm(labelImage, NORM_INF)+1)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        return;
    }
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Imgproc_ConnectedComponents, regression) { CV_ConnectedComponentsTest test; test.safe_run(); }
