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

class CV_Adaptivethresh : public cvtest::BaseTest
{
public:
    CV_Adaptivethresh();
    ~CV_Adaptivethresh();
protected:
    void run(int);
};

CV_Adaptivethresh::CV_Adaptivethresh() {}
CV_Adaptivethresh::~CV_Adaptivethresh() {}

void CV_Adaptivethresh::run( int /* start_from */)
{
    string exp_path = string(ts->get_data_path()) + "adaptivethresh/lena_orig.png";
    Mat lena = imread(exp_path, 0); // CV_LOAD_IMAGE_GRAYSCALE=0
    if (lena.empty() )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_MISSING_TEST_DATA );
        return;
    }
    int sum=0;
    for (int i = 0; i < lena.rows; i++)
    {
        unsigned char *ptr = lena.ptr(i);
        for (int j=0;j<lena.cols;j++,ptr++)
            sum+=*ptr;
    }
    if (sum!=31910861)
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }
    int windowSize[9] = {3,9,11,17,21,25,29,37,47};
    int expectedValueMean[9] = {96138,121836,124499,129096,130538,131330,131743,131616,131223};
    int expectedValueGaussNew[9] = {86308,112910,116197,122117,124672,126488,127855,129377,130387};
    int expectedValueGaussOld[9] = {88583,81365,154081,98049,149357,106414,179701,168433,90250};
    Mat im;
    bool failed=false;
    for(int i = 0; i<9; ++i )
    {
        adaptiveThreshold( lena, im, 255,cv::ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,windowSize[i],0);
        int numberWhite=countNonZero(im);
        if (numberWhite != expectedValueMean[i])
        {
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
            return;
        }
        adaptiveThreshold( lena, im, 255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,windowSize[i],0);
        if (numberWhite != expectedValueGaussNew[i])
        {   

            if (numberWhite != expectedValueGaussOld[i])
                ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
            else
                ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
    }
    if (failed)
        ts->set_failed_test_info(cvtest::TS::OK);
    else
        ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Imgproc_Adaptivethresh, regression) { CV_Adaptivethresh test; test.safe_run(); }
