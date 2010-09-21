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
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include "cvaux.h"

using namespace cv;
using namespace std;


class CV_OperationsTest : public CvTest
{
public:
    CV_OperationsTest();
    ~CV_OperationsTest();    
protected:
    void run(int);    

    struct test_excep
    {
        test_excep(const string& _s=string("")) : s(_s) {};
        string s;
    };

    bool SomeMatFunctions();
    bool TestMat();
    bool TestTemplateMat();
    bool TestMatND();
    bool TestSparseMat();
    bool operations1();

    void checkDiff(const Mat& m1, const Mat& m2, const string& s) { if (norm(m1, m2, NORM_INF) != 0) throw test_excep(s); }
    void checkDiffF(const Mat& m1, const Mat& m2, const string& s) { if (norm(m1, m2, NORM_INF) > 1e-5) throw test_excep(s); }

};

CV_OperationsTest::CV_OperationsTest(): CvTest( "operations", "?" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}
CV_OperationsTest::~CV_OperationsTest() {}

#define STR(a) STR2(a)
#define STR2(a) #a

#define CHECK_DIFF(a, b) checkDiff(a, b, "(" #a ")  !=  (" #b ")  at l." STR(__LINE__))
#define CHECK_DIFF_FLT(a, b) checkDiffF(a, b, "(" #a ")  !=(eps)  (" #b ")  at l." STR(__LINE__))

#if defined _MSC_VER && _MSC_VER < 1400
#define MSVC_OLD 1
#else
#define MSVC_OLD 0
#endif

bool CV_OperationsTest::TestMat()
{
    try
    {
        Mat one_3x1(3, 1, CV_32F, Scalar(1.0));
        Mat shi_3x1(3, 1, CV_32F, Scalar(1.2));
        Mat shi_2x1(2, 1, CV_32F, Scalar(-1));
        Scalar shift = Scalar::all(15);

        float data[] = { sqrt(2.f)/2, -sqrt(2.f)/2, 1.f, sqrt(2.f)/2, sqrt(2.f)/2, 10.f };
        Mat rot_2x3(2, 3, CV_32F, data);
        
		Mat res = one_3x1 + shi_3x1 + shi_3x1 + shi_3x1;
        res = Mat(Mat(2 * rot_2x3) * res - shi_2x1) + shift;

        Mat tmp, res2;
        add(one_3x1, shi_3x1, tmp);
        add(tmp, shi_3x1, tmp);
        add(tmp, shi_3x1, tmp);
        gemm(rot_2x3, tmp, 2, shi_2x1, -1, res2, 0);
        add(res2, Mat(2, 1, CV_32F, shift), res2);
        
        CHECK_DIFF(res, res2);
            
        Mat mat4x4(4, 4, CV_32F);
        randu(mat4x4, Scalar(0), Scalar(10));

        Mat roi1 = mat4x4(Rect(Point(1, 1), Size(2, 2)));
        Mat roi2 = mat4x4(Range(1, 3), Range(1, 3));
        
        CHECK_DIFF(roi1, roi2);
        CHECK_DIFF(mat4x4, mat4x4(Rect(Point(0,0), mat4x4.size())));        

        Mat intMat10(3, 3, CV_32S, Scalar(10));
        Mat intMat11(3, 3, CV_32S, Scalar(11));
        Mat resMat(3, 3, CV_8U, Scalar(255));
                        
        CHECK_DIFF(resMat, intMat10 == intMat10);
        CHECK_DIFF(resMat, intMat10 <  intMat11);
        CHECK_DIFF(resMat, intMat11 >  intMat10);
        CHECK_DIFF(resMat, intMat10 <= intMat11);
        CHECK_DIFF(resMat, intMat11 >= intMat10);
        CHECK_DIFF(resMat, intMat11 != intMat10);

        CHECK_DIFF(resMat, intMat10 == 10.0);
        CHECK_DIFF(resMat, 10.0 == intMat10);
        CHECK_DIFF(resMat, intMat10 <  11.0);
        CHECK_DIFF(resMat, 11.0 > intMat10);
        CHECK_DIFF(resMat, 10.0 < intMat11);
        CHECK_DIFF(resMat, 11.0 >= intMat10);
        CHECK_DIFF(resMat, 10.0 <= intMat11);
        CHECK_DIFF(resMat, 10.0 != intMat11);
        CHECK_DIFF(resMat, intMat11 != 10.0);

        Mat eye =  Mat::eye(3, 3, CV_16S);
        Mat maskMat4(3, 3, CV_16S, Scalar(4));
        Mat maskMat1(3, 3, CV_16S, Scalar(1));
        Mat maskMat5(3, 3, CV_16S, Scalar(5));
        Mat maskMat0(3, 3, CV_16S, Scalar(0));

        CHECK_DIFF(maskMat0, maskMat4 & maskMat1);
        CHECK_DIFF(maskMat0, Scalar(1) & maskMat4);
        CHECK_DIFF(maskMat0, maskMat4 & Scalar(1));
        
        Mat m;
        m = maskMat4.clone(); m &= maskMat1; CHECK_DIFF(maskMat0, m);
        m = maskMat4.clone(); m &= maskMat1 | maskMat1; CHECK_DIFF(maskMat0, m);
        m = maskMat4.clone(); m &= (2* maskMat1 - maskMat1); CHECK_DIFF(maskMat0, m);

        m = maskMat4.clone(); m &= Scalar(1); CHECK_DIFF(maskMat0, m);
        m = maskMat4.clone(); m |= maskMat1; CHECK_DIFF(maskMat5, m);
        m = maskMat5.clone(); m ^= maskMat1; CHECK_DIFF(maskMat4, m);
        m = maskMat4.clone(); m |= (2* maskMat1 - maskMat1); CHECK_DIFF(maskMat5, m);
        m = maskMat5.clone(); m ^= (2* maskMat1 - maskMat1); CHECK_DIFF(maskMat4, m);

        m = maskMat4.clone(); m |= Scalar(1); CHECK_DIFF(maskMat5, m);
        m = maskMat5.clone(); m ^= Scalar(1); CHECK_DIFF(maskMat4, m);

           
        
        CHECK_DIFF(maskMat0, (maskMat4 | maskMat4) & (maskMat1 | maskMat1));
        CHECK_DIFF(maskMat0, (maskMat4 | maskMat4) & maskMat1);
        CHECK_DIFF(maskMat0, maskMat4 & (maskMat1 | maskMat1));
        CHECK_DIFF(maskMat0, (maskMat1 | maskMat1) & Scalar(4));
        CHECK_DIFF(maskMat0, Scalar(4) & (maskMat1 | maskMat1));
        
        CHECK_DIFF(maskMat0, maskMat5 ^ (maskMat4 | maskMat1));
        CHECK_DIFF(maskMat0, (maskMat4 | maskMat1) ^ maskMat5);
        CHECK_DIFF(maskMat0, (maskMat4 + maskMat1) ^ (maskMat4 + maskMat1));
        CHECK_DIFF(maskMat0, Scalar(5) ^ (maskMat4 | Scalar(1)));
        CHECK_DIFF(maskMat1, Scalar(5) ^ maskMat4);
        CHECK_DIFF(maskMat0, Scalar(5) ^ (maskMat4 + maskMat1));
        CHECK_DIFF(maskMat5, Scalar(5) | (maskMat4 + maskMat1));
        CHECK_DIFF(maskMat0, (maskMat4 + maskMat1) ^ Scalar(5));

        CHECK_DIFF(maskMat5, maskMat5 | (maskMat4 ^ maskMat1));
        CHECK_DIFF(maskMat5, (maskMat4 ^ maskMat1) | maskMat5);        
        CHECK_DIFF(maskMat5, maskMat5 | (maskMat4 ^ Scalar(1)));
        CHECK_DIFF(maskMat5, (maskMat4 | maskMat4) | Scalar(1));
        CHECK_DIFF(maskMat5, Scalar(1) | (maskMat4 | maskMat4));
        CHECK_DIFF(maskMat5, Scalar(1) | maskMat4);
        CHECK_DIFF(maskMat5, (maskMat5 | maskMat5) | (maskMat4 ^ maskMat1));

        CHECK_DIFF(maskMat1, min(maskMat1, maskMat5));
        CHECK_DIFF(maskMat1, min(Mat(maskMat1 | maskMat1), maskMat5 | maskMat5));
        CHECK_DIFF(maskMat5, max(maskMat1, maskMat5));
        CHECK_DIFF(maskMat5, max(Mat(maskMat1 | maskMat1), maskMat5 | maskMat5));

        CHECK_DIFF(maskMat1, min(maskMat1, maskMat5 | maskMat5));
        CHECK_DIFF(maskMat1, min(maskMat1 | maskMat1, maskMat5));
        CHECK_DIFF(maskMat5, max(maskMat1 | maskMat1, maskMat5));
        CHECK_DIFF(maskMat5, max(maskMat1, maskMat5 | maskMat5));

        CHECK_DIFF(~maskMat1, maskMat1 ^ -1);
        CHECK_DIFF(~(maskMat1 | maskMat1), maskMat1 ^ -1); 

        CHECK_DIFF(maskMat1, maskMat4/4.0);   

        /////////////////////////////

        CHECK_DIFF(1.0 - (maskMat5 | maskMat5), -maskMat4);
        CHECK_DIFF((maskMat4 | maskMat4) * 1.0 + 1.0, maskMat5);
        CHECK_DIFF(1.0 + (maskMat4 | maskMat4) * 1.0, maskMat5);
        CHECK_DIFF((maskMat5 | maskMat5) * 1.0 - 1.0, maskMat4);
        CHECK_DIFF(5.0 - (maskMat4 | maskMat4) * 1.0, maskMat1);
        CHECK_DIFF((maskMat4 | maskMat4) * 1.0 + 0.5 + 0.5, maskMat5);
        CHECK_DIFF(0.5 + ((maskMat4 | maskMat4) * 1.0 + 0.5), maskMat5);
        CHECK_DIFF(((maskMat4 | maskMat4) * 1.0 + 2.0) - 1.0, maskMat5);
        CHECK_DIFF(5.0 - ((maskMat1 | maskMat1) * 1.0 + 3.0), maskMat1);
        CHECK_DIFF( ( (maskMat1 | maskMat1) * 2.0 + 2.0) * 1.25, maskMat5);
        CHECK_DIFF( 1.25 * ( (maskMat1 | maskMat1) * 2.0 + 2.0), maskMat5);
        CHECK_DIFF( -( (maskMat1 | maskMat1) * (-2.0) + 1.0), maskMat1);      
        CHECK_DIFF( maskMat1 * 1.0 + maskMat4 * 0.5 + 2.0, maskMat5);         
        CHECK_DIFF( 1.0 + (maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0), maskMat5);         
        CHECK_DIFF( (maskMat1 * 1.0 + maskMat4 * 0.5 + 2.0) - 1.0, maskMat4);         
        CHECK_DIFF(5.0 -  (maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0), maskMat1);         
        CHECK_DIFF((maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0)*1.25, maskMat5);         
        CHECK_DIFF(1.25 * (maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0), maskMat5);         
        CHECK_DIFF(-(maskMat1 * 2.0 + maskMat4 * (-1) + 1.0), maskMat1);         
        CHECK_DIFF((maskMat1 * 1.0 + maskMat4), maskMat5);         
        CHECK_DIFF((maskMat4 + maskMat1 * 1.0), maskMat5);         
        CHECK_DIFF((maskMat1 * 3.0 + 1.0) + maskMat1, maskMat5);         
        CHECK_DIFF(maskMat1 + (maskMat1 * 3.0 + 1.0), maskMat5);         
        CHECK_DIFF(maskMat1*4.0 + (maskMat1 | maskMat1), maskMat5);         
        CHECK_DIFF((maskMat1 | maskMat1) + maskMat1*4.0, maskMat5);         
        CHECK_DIFF((maskMat1*3.0 + 1.0) + (maskMat1 | maskMat1), maskMat5);         
        CHECK_DIFF((maskMat1 | maskMat1) + (maskMat1*3.0 + 1.0), maskMat5);
        CHECK_DIFF(maskMat1*4.0 + maskMat4*2.0, maskMat1 * 12);
        CHECK_DIFF((maskMat1*3.0 + 1.0) + maskMat4*2.0, maskMat1 * 12);
        CHECK_DIFF(maskMat4*2.0 + (maskMat1*3.0 + 1.0), maskMat1 * 12);
        CHECK_DIFF((maskMat1*3.0 + 1.0) + (maskMat1*2.0 + 2.0), maskMat1 * 8);
                                                     
        CHECK_DIFF(maskMat5*1.0 - maskMat4, maskMat1);
        CHECK_DIFF(maskMat5 - maskMat1 * 4.0, maskMat1);
        CHECK_DIFF((maskMat4 * 1.0 + 4.0)- maskMat4, maskMat4);
        CHECK_DIFF(maskMat5 - (maskMat1 * 2.0 + 2.0), maskMat1);
        CHECK_DIFF(maskMat5*1.0 - (maskMat4 | maskMat4), maskMat1);
        CHECK_DIFF((maskMat5 | maskMat5) - maskMat1 * 4.0, maskMat1);                
        CHECK_DIFF((maskMat4 * 1.0 + 4.0)- (maskMat4 | maskMat4), maskMat4);
        CHECK_DIFF((maskMat5 | maskMat5) - (maskMat1 * 2.0 + 2.0), maskMat1);
        CHECK_DIFF(maskMat1*5.0 - maskMat4 * 1.0, maskMat1);
        CHECK_DIFF((maskMat1*5.0 + 3.0)- maskMat4 * 1.0, maskMat4);
        CHECK_DIFF(maskMat4 * 2.0 - (maskMat1*4.0 + 3.0), maskMat1);
        CHECK_DIFF((maskMat1 * 2.0 + 3.0) - (maskMat1*3.0 + 1.0), maskMat1);

        CHECK_DIFF((maskMat5 - maskMat4)* 4.0, maskMat4);
        CHECK_DIFF(4.0 * (maskMat5 - maskMat4), maskMat4);
        
        CHECK_DIFF(-((maskMat4 | maskMat4) - (maskMat5 | maskMat5)), maskMat1);

        CHECK_DIFF(4.0 * (maskMat1 | maskMat1), maskMat4);
        CHECK_DIFF((maskMat4 | maskMat4)/4.0, maskMat1);

#if !MSVC_OLD
        CHECK_DIFF(2.0 * (maskMat1 * 2.0) , maskMat4);
#endif
        CHECK_DIFF((maskMat4 / 2.0) / 2.0 , maskMat1);
        CHECK_DIFF(-(maskMat4 - maskMat5) , maskMat1);
        CHECK_DIFF(-((maskMat4 - maskMat5) * 1.0), maskMat1);        
                      
                                    
        /////////////////////////////
        CHECK_DIFF(maskMat4 /  maskMat4, maskMat1);

        ///// Element-wise multiplication

        CHECK_DIFF(maskMat4.mul(maskMat4, 0.25), maskMat4);
        CHECK_DIFF(maskMat4.mul(maskMat1 * 4, 0.25), maskMat4);
        CHECK_DIFF(maskMat4.mul(maskMat4 / 4), maskMat4);
        CHECK_DIFF(maskMat4.mul(maskMat4 / 4), maskMat4);
        CHECK_DIFF(maskMat4.mul(maskMat4) * 0.25, maskMat4);
        CHECK_DIFF(0.25 * maskMat4.mul(maskMat4), maskMat4);
      
        ////// Element-wise division

        CHECK_DIFF(maskMat4 / maskMat4, maskMat1);
        CHECK_DIFF((maskMat4 & maskMat4) / (maskMat1 * 4), maskMat1);

        CHECK_DIFF((maskMat4 & maskMat4) / maskMat4, maskMat1);
        CHECK_DIFF(maskMat4 / (maskMat4 & maskMat4), maskMat1);
        CHECK_DIFF((maskMat1 * 4) / maskMat4, maskMat1);

        CHECK_DIFF(maskMat4 / (maskMat1 * 4), maskMat1);
        CHECK_DIFF((maskMat4 * 0.5 )/ (maskMat1 * 2), maskMat1);

        CHECK_DIFF(maskMat4 / maskMat4.mul(maskMat1), maskMat1);
        CHECK_DIFF((maskMat4 & maskMat4) / maskMat4.mul(maskMat1), maskMat1);

        CHECK_DIFF(4.0 / maskMat4, maskMat1);        
        CHECK_DIFF(4.0 / (maskMat4 | maskMat4), maskMat1);        
        CHECK_DIFF(4.0 / (maskMat1 * 4.0), maskMat1);
        CHECK_DIFF(4.0 / (maskMat4 / maskMat1), maskMat1);

        m = maskMat4.clone(); m/=4.0; CHECK_DIFF(m, maskMat1);
        m = maskMat4.clone(); m/=maskMat4; CHECK_DIFF(m, maskMat1);
        m = maskMat4.clone(); m/=(maskMat1 * 4.0); CHECK_DIFF(m, maskMat1);
        m = maskMat4.clone(); m/=(maskMat4 / maskMat1); CHECK_DIFF(m, maskMat1);
      
        /////////////////////////////        
        float matrix_data[] = { 3, 1, -4, -5, 1, 0, 0, 1.1f, 1.5f};        
        Mat mt(3, 3, CV_32F, matrix_data);
        Mat mi = mt.inv();
        Mat d1 = Mat::eye(3, 3, CV_32F);
        Mat d2 = d1 * 2;
        MatExpr mt_tr = mt.t();
        MatExpr mi_tr = mi.t();
        Mat mi2 = mi * 2;


        CHECK_DIFF_FLT( mi2 * mt, d2 );
        CHECK_DIFF_FLT( mi * mt, d1 );
        CHECK_DIFF_FLT( mt_tr * mi_tr, d1 );

        m = mi.clone(); m*=mt;  CHECK_DIFF_FLT(m, d1);
        m = mi.clone(); m*= (2 * mt - mt) ;  CHECK_DIFF_FLT(m, d1);

        m = maskMat4.clone(); m+=(maskMat1 * 1.0); CHECK_DIFF(m, maskMat5);
        m = maskMat5.clone(); m-=(maskMat1 * 4.0); CHECK_DIFF(m, maskMat1);

        m = maskMat1.clone(); m+=(maskMat1 * 3.0 + 1.0); CHECK_DIFF(m, maskMat5);
        m = maskMat5.clone(); m-=(maskMat1 * 3.0 + 1.0); CHECK_DIFF(m, maskMat1);
#if !MSVC_OLD
        m = mi.clone(); m+=(3.0 * mi * mt + d1); CHECK_DIFF_FLT(m, mi + d1 * 4);
        m = mi.clone(); m-=(3.0 * mi * mt + d1); CHECK_DIFF_FLT(m, mi - d1 * 4);
        m = mi.clone(); m*=(mt * 1.0); CHECK_DIFF_FLT(m, d1);
        m = mi.clone(); m*=(mt * 1.0 + Mat::eye(m.size(), m.type())); CHECK_DIFF_FLT(m, d1 + mi);
        m = mi.clone(); m*=mt_tr.t(); CHECK_DIFF_FLT(m, d1);

        CHECK_DIFF_FLT( (mi * 2) * mt, d2);
        CHECK_DIFF_FLT( mi * (2 * mt), d2);           
        CHECK_DIFF_FLT( mt.t() * mi_tr, d1 );
        CHECK_DIFF_FLT( mt_tr * mi.t(), d1 );           
        CHECK_DIFF_FLT( (mi * 0.4) * (mt * 5), d2);

        CHECK_DIFF_FLT( mt.t() * (mi_tr * 2), d2 );
        CHECK_DIFF_FLT( (mt_tr * 2) * mi.t(), d2 );           

        CHECK_DIFF_FLT(mt.t() * mi.t(), d1);
        CHECK_DIFF_FLT( (mi * mt) * 2.0, d2);
        CHECK_DIFF_FLT( 2.0 * (mi * mt), d2);
        CHECK_DIFF_FLT( -(mi * mt), -d1);

        CHECK_DIFF_FLT( (mi * mt) / 2.0, d1 / 2);

        Mat mt_mul_2_plus_1;
        gemm(mt, d1, 2, Mat::ones(3, 3, CV_32F), 1, mt_mul_2_plus_1);
        
        CHECK_DIFF( (mt * 2.0 + 1.0) * mi, mt_mul_2_plus_1 * mi);        // (A*alpha + beta)*B
        CHECK_DIFF( mi * (mt * 2.0 + 1.0), mi * mt_mul_2_plus_1);        // A*(B*alpha + beta)            
        CHECK_DIFF( (mt * 2.0 + 1.0) * (mi * 2), mt_mul_2_plus_1 * mi2); // (A*alpha + beta)*(B*gamma)
        CHECK_DIFF( (mi *2)* (mt * 2.0 + 1.0), mi2 * mt_mul_2_plus_1);   // (A*gamma)*(B*alpha + beta)
        CHECK_DIFF_FLT( (mt * 2.0 + 1.0) * mi.t(), mt_mul_2_plus_1 * mi_tr); // (A*alpha + beta)*B^t
        CHECK_DIFF_FLT( mi.t() * (mt * 2.0 + 1.0), mi_tr * mt_mul_2_plus_1); // A^t*(B*alpha + beta)

        CHECK_DIFF_FLT( (mi * mt + d2)*5, d1 * 3 * 5);
        CHECK_DIFF_FLT( mi * mt + d2, d1 * 3);
        CHECK_DIFF_FLT( -(mi * mt) + d2, d1);
        CHECK_DIFF_FLT( (mi * mt) + d1, d2);
        CHECK_DIFF_FLT( d1 + (mi * mt), d2);
        CHECK_DIFF_FLT( (mi * mt) - d2, -d1);
        CHECK_DIFF_FLT( d2 - (mi * mt), d1);

        CHECK_DIFF_FLT( (mi * mt) + d2 * 0.5, d2);
        CHECK_DIFF_FLT( d2 * 0.5 + (mi * mt), d2);
        CHECK_DIFF_FLT( (mi * mt) - d1 * 2, -d1);
        CHECK_DIFF_FLT( d1 * 2 - (mi * mt), d1);      

        CHECK_DIFF_FLT( (mi * mt) + mi.t(), mi_tr + d1);
        CHECK_DIFF_FLT( mi.t() + (mi * mt), mi_tr + d1);
        CHECK_DIFF_FLT( (mi * mt) - mi.t(), d1 - mi_tr);
        CHECK_DIFF_FLT( mi.t() - (mi * mt), mi_tr - d1);

        CHECK_DIFF_FLT( 2.0 *(mi * mt + d2), d1 * 6);
        CHECK_DIFF_FLT( -(mi * mt + d2), d1 * -3);

        CHECK_DIFF_FLT(mt.inv() * mt, d1);

        CHECK_DIFF_FLT(mt.inv() * (2*mt - mt), d1);               
#endif
    }
    catch (const test_excep& e)
    {
        ts->printf(CvTS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

bool CV_OperationsTest::SomeMatFunctions()
{
    try
    {
        Mat rgba( 10, 10, CV_8UC4, Scalar(1,2,3,4) );
        Mat bgr( rgba.rows, rgba.cols, CV_8UC3 );
        Mat alpha( rgba.rows, rgba.cols, CV_8UC1 );        
        Mat out[] = { bgr, alpha };
        // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
        // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
        int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
        mixChannels( &rgba, 1, out, 2, from_to, 4 );        

        Mat bgr_exp( rgba.size(), CV_8UC3, Scalar(3,2,1));
        Mat alpha_exp( rgba.size(), CV_8UC1, Scalar(4));

        CHECK_DIFF(bgr_exp, bgr);      
        CHECK_DIFF(alpha_exp, alpha);      
    }
    catch (const test_excep& e)
    {
        ts->printf(CvTS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
        return false;
    }
    return true;

}


bool CV_OperationsTest::TestTemplateMat()
{  
    try
    {
        Mat_<float> one_3x1(3, 1, 1.0f);
        Mat_<float> shi_3x1(3, 1, 1.2f);
        Mat_<float> shi_2x1(2, 1, -2);
        Scalar shift = Scalar::all(15);

        float data[] = { sqrt(2.f)/2, -sqrt(2.f)/2, 1.f, sqrt(2.f)/2, sqrt(2.f)/2, 10.f };
        Mat_<float> rot_2x3(2, 3, data);
               
        Mat_<float> res = Mat(Mat(2 * rot_2x3) * Mat(one_3x1 + shi_3x1 + shi_3x1 + shi_3x1) - shi_2x1) + shift;
        Mat_<float> resS = rot_2x3 * one_3x1;

        Mat_<float> tmp, res2, resS2;
        add(one_3x1, shi_3x1, tmp);
        add(tmp, shi_3x1, tmp);
        add(tmp, shi_3x1, tmp);
        gemm(rot_2x3, tmp, 2, shi_2x1, -1, res2, 0);
        add(res2, Mat(2, 1, CV_32F, shift), res2);
        
        gemm(rot_2x3, one_3x1, 1, shi_2x1, 0, resS2, 0);
        CHECK_DIFF(res, res2);        
        CHECK_DIFF(resS, resS2);

            
        Mat_<float> mat4x4(4, 4);
        randu(mat4x4, Scalar(0), Scalar(10));

        Mat_<float> roi1 = mat4x4(Rect(Point(1, 1), Size(2, 2)));
        Mat_<float> roi2 = mat4x4(Range(1, 3), Range(1, 3));
        
        CHECK_DIFF(roi1, roi2);
        CHECK_DIFF(mat4x4, mat4x4(Rect(Point(0,0), mat4x4.size())));        

        Mat_<int> intMat10(3, 3, 10);
        Mat_<int> intMat11(3, 3, 11);
        Mat_<uchar> resMat(3, 3, 255);
                
        CHECK_DIFF(resMat, intMat10 == intMat10);
        CHECK_DIFF(resMat, intMat10 <  intMat11);
        CHECK_DIFF(resMat, intMat11 >  intMat10);
        CHECK_DIFF(resMat, intMat10 <= intMat11);
        CHECK_DIFF(resMat, intMat11 >= intMat10);

        CHECK_DIFF(resMat, intMat10 == 10.0);
        CHECK_DIFF(resMat, intMat10 <  11.0);
        CHECK_DIFF(resMat, intMat11 >  10.0);
        CHECK_DIFF(resMat, intMat10 <= 11.0);
        CHECK_DIFF(resMat, intMat11 >= 10.0);

        Mat_<uchar> maskMat4(3, 3, 4);
        Mat_<uchar> maskMat1(3, 3, 1);
        Mat_<uchar> maskMat5(3, 3, 5);
        Mat_<uchar> maskMat0(3, 3, (uchar)0);

        CHECK_DIFF(maskMat0, maskMat4 & maskMat1);        
        CHECK_DIFF(maskMat0, Scalar(1) & maskMat4);
        CHECK_DIFF(maskMat0, maskMat4 & Scalar(1));
                        
        Mat_<uchar> m;
        m = maskMat4.clone(); m&=maskMat1; CHECK_DIFF(maskMat0, m);
        m = maskMat4.clone(); m&=Scalar(1); CHECK_DIFF(maskMat0, m);

        m = maskMat4.clone(); m|=maskMat1; CHECK_DIFF(maskMat5, m);
        m = maskMat4.clone(); m^=maskMat1; CHECK_DIFF(maskMat5, m);
        
        CHECK_DIFF(maskMat0, (maskMat4 | maskMat4) & (maskMat1 | maskMat1));
        CHECK_DIFF(maskMat0, (maskMat4 | maskMat4) & maskMat1);
        CHECK_DIFF(maskMat0, maskMat4 & (maskMat1 | maskMat1));

        CHECK_DIFF(maskMat0, maskMat5 ^ (maskMat4 | maskMat1));
        CHECK_DIFF(maskMat0, Scalar(5) ^ (maskMat4 | Scalar(1)));

        CHECK_DIFF(maskMat5, maskMat5 | (maskMat4 ^ maskMat1));
        CHECK_DIFF(maskMat5, maskMat5 | (maskMat4 ^ Scalar(1)));

        CHECK_DIFF(~maskMat1, maskMat1 ^ 0xFF);
        CHECK_DIFF(~(maskMat1 | maskMat1), maskMat1 ^ 0xFF); 

        CHECK_DIFF(maskMat1 + maskMat4, maskMat5);
        CHECK_DIFF(maskMat1 + Scalar(4), maskMat5);
        CHECK_DIFF(Scalar(4) + maskMat1, maskMat5);
        CHECK_DIFF(Scalar(4) + (maskMat1 & maskMat1), maskMat5);

        CHECK_DIFF(maskMat1 + 4.0, maskMat5);
        CHECK_DIFF((maskMat1 & 0xFF) + 4.0, maskMat5);
        CHECK_DIFF(4.0 + maskMat1, maskMat5);

        m = maskMat4.clone(); m+=Scalar(1); CHECK_DIFF(m, maskMat5);
        m = maskMat4.clone(); m+=maskMat1; CHECK_DIFF(m, maskMat5);
        m = maskMat4.clone(); m+=(maskMat1 | maskMat1); CHECK_DIFF(m, maskMat5);

        CHECK_DIFF(maskMat5 - maskMat1, maskMat4);
        CHECK_DIFF(maskMat5 - Scalar(1), maskMat4);
        CHECK_DIFF((maskMat5 | maskMat5) - Scalar(1), maskMat4);
        CHECK_DIFF(maskMat5 - 1, maskMat4);
        CHECK_DIFF((maskMat5 | maskMat5) - 1, maskMat4);
        CHECK_DIFF((maskMat5 | maskMat5) - (maskMat1 | maskMat1), maskMat4);

        CHECK_DIFF(maskMat1, min(maskMat1, maskMat5));
        CHECK_DIFF(maskMat5, max(maskMat1, maskMat5));
        
        m = maskMat5.clone(); m-=Scalar(1); CHECK_DIFF(m, maskMat4);
        m = maskMat5.clone(); m-=maskMat1; CHECK_DIFF(m, maskMat4);
        m = maskMat5.clone(); m-=(maskMat1 | maskMat1); CHECK_DIFF(m, maskMat4);

        m = maskMat4.clone(); m |= Scalar(1); CHECK_DIFF(maskMat5, m);
        m = maskMat5.clone(); m ^= Scalar(1); CHECK_DIFF(maskMat4, m);

        CHECK_DIFF(maskMat1, maskMat4/4.0);       

        Mat_<float> negf(3, 3, -3.0);                
        Mat_<float> posf = -negf;
        Mat_<float> posf2 = posf * 2;
        Mat_<int> negi(3, 3, -3);        

        CHECK_DIFF(abs(negf), -negf);         
        CHECK_DIFF(abs(posf - posf2), -negf);         
        CHECK_DIFF(abs(negi), -(negi & negi));

        CHECK_DIFF(5.0 - maskMat4, maskMat1);
        

        CHECK_DIFF(maskMat4.mul(maskMat4, 0.25), maskMat4);
        CHECK_DIFF(maskMat4.mul(maskMat1 * 4, 0.25), maskMat4);
        CHECK_DIFF(maskMat4.mul(maskMat4 / 4), maskMat4);

          
        ////// Element-wise division

        CHECK_DIFF(maskMat4 / maskMat4, maskMat1);
        CHECK_DIFF(4.0 / maskMat4, maskMat1);
        m = maskMat4.clone(); m/=4.0; CHECK_DIFF(m, maskMat1);
        
        ////////////////////////////////

        typedef Mat_<int> TestMat_t;

        const TestMat_t cnegi = negi.clone();

        TestMat_t::iterator beg = negi.begin();
        TestMat_t::iterator end = negi.end();
        
        TestMat_t::const_iterator cbeg = cnegi.begin();
        TestMat_t::const_iterator cend = cnegi.end();

        int sum = 0;
        for(; beg!=end; ++beg)
            sum+=*beg;

        for(; cbeg!=cend; ++cbeg)
            sum-=*cbeg;

        if (sum != 0) throw test_excep();

        CHECK_DIFF(negi.col(1), negi.col(2));
        CHECK_DIFF(negi.row(1), negi.row(2));
        CHECK_DIFF(negi.col(1), negi.diag());
        
        if (Mat_<Point2f>(1, 1).elemSize1() != sizeof(float)) throw test_excep();
        if (Mat_<Point2f>(1, 1).elemSize() != 2 * sizeof(float)) throw test_excep();
        if (Mat_<Point2f>(1, 1).depth() != CV_32F) throw test_excep();
        if (Mat_<float>(1, 1).depth() != CV_32F) throw test_excep();
        if (Mat_<int>(1, 1).depth() != CV_32S) throw test_excep();
        if (Mat_<double>(1, 1).depth() != CV_64F) throw test_excep();
        if (Mat_<Point3d>(1, 1).depth() != CV_64F) throw test_excep();        
        if (Mat_<signed char>(1, 1).depth() != CV_8S) throw test_excep();
        if (Mat_<unsigned short>(1, 1).depth() != CV_16U) throw test_excep();
        if (Mat_<unsigned short>(1, 1).channels() != 1) throw test_excep();
        if (Mat_<Point2f>(1, 1).channels() != 2) throw test_excep();
        if (Mat_<Point3f>(1, 1).channels() != 3) throw test_excep();
        if (Mat_<Point3d>(1, 1).channels() != 3) throw test_excep();

        Mat_<uchar> eye = Mat_<uchar>::zeros(2, 2); CHECK_DIFF(Mat_<uchar>::zeros(Size(2, 2)), eye);
        eye.at<uchar>(Point(0,0)) = 1; eye.at<uchar>(1, 1) = 1;
                
        CHECK_DIFF(Mat_<uchar>::eye(2, 2), eye);
        CHECK_DIFF(eye, Mat_<uchar>::eye(Size(2,2)));        
        
        Mat_<uchar> ones(2, 2, (uchar)1);
        CHECK_DIFF(ones, Mat_<uchar>::ones(Size(2,2)));
        CHECK_DIFF(Mat_<uchar>::ones(2, 2), ones);

        Mat_<Point2f> pntMat(2, 2, Point2f(1, 0));
        if(pntMat.stepT() != 2) throw test_excep();

        uchar uchar_data[] = {1, 0, 0, 1};

        Mat_<uchar> matFromData(1, 4, uchar_data);
        const Mat_<uchar> mat2 = matFromData.clone();
        CHECK_DIFF(matFromData, eye.reshape(1));
        if (matFromData(Point(0,0)) != uchar_data[0])throw test_excep();
        if (mat2(Point(0,0)) != uchar_data[0]) throw test_excep();

        if (matFromData(0,0) != uchar_data[0])throw test_excep();
        if (mat2(0,0) != uchar_data[0]) throw test_excep();
                
        Mat_<uchar> rect(eye, Rect(0, 0, 1, 1));
        if (rect.cols != 1 || rect.rows != 1 || rect(0,0) != uchar_data[0]) throw test_excep();

        //cv::Mat_<_Tp>::adjustROI(int,int,int,int)
        //cv::Mat_<_Tp>::cross(const Mat_&) const	        
        //cv::Mat_<_Tp>::Mat_(const vector<_Tp>&,bool)
        //cv::Mat_<_Tp>::Mat_(int,int,_Tp*,size_t)
        //cv::Mat_<_Tp>::Mat_(int,int,const _Tp&)	
        //cv::Mat_<_Tp>::Mat_(Size,const _Tp&)	
        //cv::Mat_<_Tp>::mul(const Mat_<_Tp>&,double) const	
        //cv::Mat_<_Tp>::mul(const MatExpr_<MatExpr_Op2_<Mat_<_Tp>,double,Mat_<_Tp>,MatOp_DivRS_<Mat> >,Mat_<_Tp> >&,double) const	
        //cv::Mat_<_Tp>::mul(const MatExpr_<MatExpr_Op2_<Mat_<_Tp>,double,Mat_<_Tp>,MatOp_Scale_<Mat> >,Mat_<_Tp> >&,double) const	
        //cv::Mat_<_Tp>::operator Mat_<T2>() const	
        //cv::Mat_<_Tp>::operator MatExpr_<Mat_<_Tp>,Mat_<_Tp> >() const	
        //cv::Mat_<_Tp>::operator()(const Range&,const Range&) const	
        //cv::Mat_<_Tp>::operator()(const Rect&) const

        //cv::Mat_<_Tp>::operator=(const MatExpr_Base&)
        //cv::Mat_<_Tp>::operator[](int) const


        ///////////////////////////////

        float matrix_data[] = { 3, 1, -4, -5, 1, 0, 0, 1.1f, 1.5f};        
        Mat_<float> mt(3, 3, matrix_data);
        Mat_<float> mi = mt.inv();
        Mat_<float> d1 = Mat_<float>::eye(3, 3);
        Mat_<float> d2 = d1 * 2;
        Mat_<float> mt_tr = mt.t();
        Mat_<float> mi_tr = mi.t();
        Mat_<float> mi2 = mi * 2;

        CHECK_DIFF_FLT( mi2 * mt, d2 );
        CHECK_DIFF_FLT( mi * mt, d1 );
        CHECK_DIFF_FLT( mt_tr * mi_tr, d1 );

        Mat_<float> mf;
        mf = mi.clone(); mf*=mt;  CHECK_DIFF_FLT(mf, d1);

        ////// typedefs //////

        if (Mat1b(1, 1).elemSize() != sizeof(uchar)) throw test_excep();
        if (Mat2b(1, 1).elemSize() != 2 * sizeof(uchar)) throw test_excep();
        if (Mat3b(1, 1).elemSize() != 3 * sizeof(uchar)) throw test_excep();
        if (Mat1f(1, 1).elemSize() != sizeof(float)) throw test_excep();
        if (Mat2f(1, 1).elemSize() != 2 * sizeof(float)) throw test_excep();
        if (Mat3f(1, 1).elemSize() != 3 * sizeof(float)) throw test_excep();
        if (Mat1f(1, 1).depth() != CV_32F) throw test_excep();
        if (Mat3f(1, 1).depth() != CV_32F) throw test_excep();
        if (Mat3f(1, 1).type() != CV_32FC3) throw test_excep();
        if (Mat1i(1, 1).depth() != CV_32S) throw test_excep();
        if (Mat1d(1, 1).depth() != CV_64F) throw test_excep();
        if (Mat1b(1, 1).depth() != CV_8U) throw test_excep();
        if (Mat3b(1, 1).type() != CV_8UC3) throw test_excep();
        if (Mat1w(1, 1).depth() != CV_16U) throw test_excep();
        if (Mat1s(1, 1).depth() != CV_16S) throw test_excep();
        if (Mat1f(1, 1).channels() != 1) throw test_excep();
        if (Mat1b(1, 1).channels() != 1) throw test_excep();
        if (Mat1i(1, 1).channels() != 1) throw test_excep();
        if (Mat1w(1, 1).channels() != 1) throw test_excep();
        if (Mat1s(1, 1).channels() != 1) throw test_excep();
        if (Mat2f(1, 1).channels() != 2) throw test_excep();
        if (Mat2b(1, 1).channels() != 2) throw test_excep();
        if (Mat2i(1, 1).channels() != 2) throw test_excep();
        if (Mat2w(1, 1).channels() != 2) throw test_excep();
        if (Mat2s(1, 1).channels() != 2) throw test_excep();
        if (Mat3f(1, 1).channels() != 3) throw test_excep();
        if (Mat3b(1, 1).channels() != 3) throw test_excep();
        if (Mat3i(1, 1).channels() != 3) throw test_excep();
        if (Mat3w(1, 1).channels() != 3) throw test_excep();
        if (Mat3s(1, 1).channels() != 3) throw test_excep();

    }
    catch (const test_excep& e)
    {
        ts->printf(CvTS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

bool CV_OperationsTest::TestMatND()
{  
    int sizes[] = { 3, 3, 3};
    cv::MatND nd(3, sizes, CV_32F);

    return true;
}

bool CV_OperationsTest::TestSparseMat()
{  
    try
    {
        int sizes[] = { 10, 10, 10};
        int dims = sizeof(sizes)/sizeof(sizes[0]);
        SparseMat mat(dims, sizes, CV_32FC2);

        if (mat.dims() != dims) throw test_excep();
        if (mat.channels() != 2) throw test_excep();
        if (mat.depth() != CV_32F) throw test_excep();

        SparseMat mat2 = mat.clone();
    }
    catch (const test_excep&)
    {
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

bool CV_OperationsTest::operations1()
{    
    try 
    {
        Point3d p1(1, 1, 1), p2(2, 2, 2), p4(4, 4, 4);    
        p1*=2;    
        if (!(p1     == p2)) throw test_excep();
        if (!(p2 * 2 == p4)) throw test_excep();
        if (!(p2 * 2.f == p4)) throw test_excep();
        if (!(p2 * 2.f == p4)) throw test_excep();

        Point2d pi1(1, 1), pi2(2, 2), pi4(4, 4);    
        pi1*=2;
        if (!(pi1     == pi2)) throw test_excep();
        if (!(pi2 * 2 == pi4)) throw test_excep();
        if (!(pi2 * 2.f == pi4)) throw test_excep();
        if (!(pi2 * 2.f == pi4)) throw test_excep();
        
        Vec2d v12(1, 1), v22(2, 2);
        v12*=2.0;
        if (!(v12 == v22)) throw test_excep();
        
        Vec3d v13(1, 1, 1), v23(2, 2, 2);
        v13*=2.0;
        if (!(v13 == v23)) throw test_excep();

        Vec4d v14(1, 1, 1, 1), v24(2, 2, 2, 2);
        v14*=2.0;
        if (!(v14 == v24)) throw test_excep();
        
        Size sz(10, 20);
        if (sz.area() != 200) throw test_excep();
        if (sz.width != 10 || sz.height != 20) throw test_excep();
        if (((CvSize)sz).width != 10 || ((CvSize)sz).height != 20) throw test_excep();
        
        Vec<double, 5> v5d(1, 1, 1, 1, 1);
        Vec<double, 6> v6d(1, 1, 1, 1, 1, 1);
        Vec<double, 7> v7d(1, 1, 1, 1, 1, 1, 1);
        Vec<double, 8> v8d(1, 1, 1, 1, 1, 1, 1, 1);
        Vec<double, 9> v9d(1, 1, 1, 1, 1, 1, 1, 1, 1);
        Vec<double,10> v10d(1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        Vec<double,10> v10dzero;
        for (int ii = 0; ii < 10; ++ii) {
            if (!v10dzero[ii] == 0.0)
                throw test_excep();
        }
    }
    catch(const test_excep&)
    {
        ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

void CV_OperationsTest::run( int /* start_from */)
{
    if (!TestMat())
        return;

    if (!SomeMatFunctions())
        return;

    if (!TestTemplateMat())
        return;

 /*   if (!TestMatND())
        return;*/

    if (!TestSparseMat())
        return;

    if (!operations1())
        return;

    ts->set_failed_test_info(CvTS::OK);
}

CV_OperationsTest cv_Operations_test;
