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

#include "cvtest.h"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

class CV_GrabcutTest : public CvTest
{
public:
    CV_GrabcutTest();
    ~CV_GrabcutTest();    
protected:    
    void run(int);    
};

CV_GrabcutTest::CV_GrabcutTest(): CvTest( "segmentation-grabcut", "cv::grabCut" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}
CV_GrabcutTest::~CV_GrabcutTest() {}

bool verify(const Mat& mask, const Mat& exp)
{    
    if (0 == norm(mask, exp, NORM_INF))
        return true;

    Mat mask_dilated, exp_dilated;

    const int inter_num = 2;
    dilate(mask, mask_dilated, Mat(), Point(-1, -1), inter_num);
    dilate(exp, exp_dilated, Mat(), Point(-1, -1), inter_num);
    
    return countNonZero(mask-exp_dilated) + countNonZero(mask_dilated-exp) == 0;
}

void CV_GrabcutTest::run( int /* start_from */)
{       
    DefaultRngAuto defRng;
        
    Mat img = imread(string(ts->get_data_path()) + "shared/airplane.jpg");    
    Mat mask_prob = imread(string(ts->get_data_path()) + "grabcut/mask_prob.png", 0);
    Mat exp_mask1 = imread(string(ts->get_data_path()) + "grabcut/exp_mask1.png", 0);
    Mat exp_mask2 = imread(string(ts->get_data_path()) + "grabcut/exp_mask2.png", 0);
    
    if (img.empty() || mask_prob.empty() || exp_mask1.empty() || exp_mask2.empty() ||
        img.size() != mask_prob.size() || mask_prob.size() != exp_mask1.size() || 
        exp_mask1.size() != exp_mask2.size())
    {
         ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);         
         return;
    }
    
    Rect rect(Point(24, 126), Point(483, 294));
    Mat exp_bgdModel, exp_fgdModel;

    Mat mask;
    mask = Scalar(0);
    Mat bgdModel, fgdModel;
    grabCut( img, mask, rect, bgdModel, fgdModel, 0, GC_INIT_WITH_RECT );    
    grabCut( img, mask, rect, bgdModel, fgdModel, 2, GC_EVAL );

    //imwrite(string(ts->get_data_path()) + "grabcut/mask_prob.png", mask_prob);
    //imwrite(string(ts->get_data_path()) + "grabcut/exp_mask1.png", mask);
    
    if (!verify(mask & 1, exp_mask1))
    {        
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);        
        return;
    }
    
    mask = mask_prob;
    bgdModel.release();
    fgdModel.release(); 
    rect = Rect();
    grabCut( img, mask, rect, bgdModel, fgdModel, 0, GC_INIT_WITH_MASK );
    grabCut( img, mask, rect, bgdModel, fgdModel, 1, GC_EVAL );

    //imwrite(string(ts->get_data_path()) + "grabcut/exp_mask2.png", mask);
    
    if (!verify(mask & 1, exp_mask2))
    {
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);        
        return;
    }                    
    ts->set_failed_test_info(CvTS::OK);    
}

CV_GrabcutTest grabcut_test;
