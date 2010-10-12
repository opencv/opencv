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

#include "cxcoretest.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>

using namespace cv;
using namespace std;

class CV_MatrOpTest : public CvTest
{
public:
    CV_MatrOpTest();
    ~CV_MatrOpTest();    
protected:
    void run(int);    

    bool TestMat();
    bool TestMatND();
    bool TestSparseMat();


    bool checkMatSetError(const Mat& m1, const Mat& m2);
};

CV_MatrOpTest::CV_MatrOpTest(): CvTest( "matrix-operations", "?" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}
CV_MatrOpTest::~CV_MatrOpTest() {}

bool CV_MatrOpTest::checkMatSetError(const Mat& m1, const Mat& m2)
{
    if (norm(m1, m2, NORM_INF) == 0)
        return true;
    
    ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
    return false;    
}

bool CV_MatrOpTest::TestMat()
{
    Mat one_3x1(3, 1, CV_32F, Scalar(1.0));
    Mat shi_3x1(3, 1, CV_32F, Scalar(1.2));
    Mat shi_2x1(2, 1, CV_32F, Scalar(-1));
    Scalar shift = Scalar::all(15);

    float data[] = { sqrt(2.f)/2, -sqrt(2.f)/2, 1.f, sqrt(2.f)/2, sqrt(2.f)/2, 10.f };
    Mat rot_2x3(2, 3, CV_32F, data);
       
    Mat res = rot_2x3 * (one_3x1 + shi_3x1 + shi_3x1 + shi_3x1) - shi_2x1 + shift;

    Mat tmp, res2;
    add(one_3x1, shi_3x1, tmp);
    add(tmp, shi_3x1, tmp);
    add(tmp, shi_3x1, tmp);
    gemm(rot_2x3, tmp, 1, shi_2x1, -1, res2, 0);
    add(res2, Mat(2, 1, CV_32F, shift), res2);
    
    if (!checkMatSetError(res, res2))
        return false;
    
    Mat mat4x4(4, 4, CV_32F);
    randu(mat4x4, Scalar(0), Scalar(10));

    Mat roi1 = mat4x4(Rect(Point(1, 1), Size(2, 2)));
    Mat roi2 = mat4x4(Range(1, 3), Range(1, 3));

    if (!checkMatSetError(roi1, roi2))
        return false;

    if (!checkMatSetError(mat4x4, mat4x4(Rect(Point(0,0), mat4x4.size()))))
        return false;

    
    return true;
}

bool CV_MatrOpTest::TestMatND()
{  
    int sizes[] = { 3, 3, 3};
    cv::MatND nd(3, sizes, CV_32F);

   /* MatND res = nd * nd + nd;    
    MatND res2;
    cv::gemm(nd, nd, 1, nd, 1, res2);
    
    if (!checkMatSetError(res1, res2))
        return false;*/

    return true;
}

bool CV_MatrOpTest::TestSparseMat()
{  
    int sizes[] = { 10, 10, 10};
    SparseMat mat(sizeof(sizes)/sizeof(sizes[0]), sizes, CV_32F);

    return true;
}



void CV_MatrOpTest::run( int /* start_from */)
{
    if (!TestMat())
        return;

    if (!TestMatND())
        return;

    if (!TestSparseMat())
        return;
         
    ts->set_failed_test_info(CvTS::OK);
}

CV_MatrOpTest cv_MatrOp_test;


