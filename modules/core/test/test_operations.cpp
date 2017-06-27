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
#include "opencv2/ts/ocl_test.hpp" // T-API like tests

namespace cvtest {
namespace {

class CV_OperationsTest : public cvtest::BaseTest
{
public:
    CV_OperationsTest();
    ~CV_OperationsTest();
protected:
    void run(int);

    struct test_excep
    {
        test_excep(const string& _s=string("")) : s(_s) { }
        string s;
    };

    bool SomeMatFunctions();
    bool TestMat();
    template<typename _Tp> void TestType(Size sz, _Tp value);
    bool TestTemplateMat();
    bool TestMatND();
    bool TestSparseMat();
    bool TestVec();
    bool TestMatxMultiplication();
    bool TestMatxElementwiseDivison();
    bool TestSubMatAccess();
    bool TestExp();
    bool TestSVD();
    bool operations1();

    void checkDiff(const Mat& m1, const Mat& m2, const string& s)
    {
        if (cvtest::norm(m1, m2, NORM_INF) != 0) throw test_excep(s);
    }
    void checkDiffF(const Mat& m1, const Mat& m2, const string& s)
    {
        if (cvtest::norm(m1, m2, NORM_INF) > 1e-5) throw test_excep(s);
    }
};

CV_OperationsTest::CV_OperationsTest()
{
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

template<typename _Tp> void CV_OperationsTest::TestType(Size sz, _Tp value)
{
    cv::Mat_<_Tp> m(sz);
    CV_Assert(m.cols == sz.width && m.rows == sz.height && m.depth() == DataType<_Tp>::depth &&
              m.channels() == DataType<_Tp>::channels &&
              m.elemSize() == sizeof(_Tp) && m.step == m.elemSize()*m.cols);
    for( int y = 0; y < sz.height; y++ )
        for( int x = 0; x < sz.width; x++ )
        {
            m(y,x) = value;
        }

    double s = sum(Mat(m).reshape(1))[0];
    CV_Assert( s == (double)sz.width*sz.height );
}

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
        ts->printf(cvtest::TS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
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
        ts->printf(cvtest::TS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;

}


bool CV_OperationsTest::TestSubMatAccess()
{
    try
    {
        Mat_<float> T_bs(4,4);
        Vec3f cdir(1.f, 1.f, 0.f);
        Vec3f ydir(1.f, 0.f, 1.f);
        Vec3f fpt(0.1f, 0.7f, 0.2f);
        T_bs.setTo(0);
        T_bs(Range(0,3),Range(2,3)) = 1.0*Mat(cdir); // wierd OpenCV stuff, need to do multiply
        T_bs(Range(0,3),Range(1,2)) = 1.0*Mat(ydir);
        T_bs(Range(0,3),Range(0,1)) = 1.0*Mat(cdir.cross(ydir));
        T_bs(Range(0,3),Range(3,4)) = 1.0*Mat(fpt);
        T_bs(3,3) = 1.0;
        //std::cout << "[Nav Grok] S frame =" << std::endl << T_bs << std::endl;

        // set up display coords, really just the S frame
        std::vector<float>coords;

        for (int i=0; i<16; i++)
        {
            coords.push_back(T_bs(i));
            //std::cout << T_bs1(i) << std::endl;
        }
        CV_Assert( cvtest::norm(coords, T_bs.reshape(1,1), NORM_INF) == 0 );
    }
    catch (const test_excep& e)
    {
        ts->printf(cvtest::TS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
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
        CHECK_DIFF(matFromData, eye.reshape(1, 1));
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

        vector<Mat_<float> > mvf, mvf2;
        Mat_<Vec2f> mf2;
        mvf.push_back(Mat_<float>::ones(4, 3));
        mvf.push_back(Mat_<float>::zeros(4, 3));
        merge(mvf, mf2);
        split(mf2, mvf2);
        CV_Assert( cvtest::norm(mvf2[0], mvf[0], CV_C) == 0 &&
                  cvtest::norm(mvf2[1], mvf[1], CV_C) == 0 );

        {
        Mat a(2,2,CV_32F,1.f);
        Mat b(1,2,CV_32F,1.f);
        Mat c = (a*b.t()).t();
        CV_Assert( cvtest::norm(c, CV_L1) == 4. );
        }

        bool badarg_catched = false;
        try
        {
            Mat m1 = Mat::zeros(1, 10, CV_8UC1);
            Mat m2 = Mat::zeros(10, 10, CV_8UC3);
            m1.copyTo(m2.row(1));
        }
        catch(const Exception&)
        {
            badarg_catched = true;
        }
        CV_Assert( badarg_catched );

        Size size(2, 5);
        TestType<float>(size, 1.f);
        cv::Vec3f val1 = 1.f;
        TestType<cv::Vec3f>(size, val1);
        cv::Matx31f val2 = 1.f;
        TestType<cv::Matx31f>(size, val2);
        cv::Matx41f val3 = 1.f;
        TestType<cv::Matx41f>(size, val3);
        cv::Matx32f val4 = 1.f;
        TestType<cv::Matx32f>(size, val4);
    }
    catch (const test_excep& e)
    {
        ts->printf(cvtest::TS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
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
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}


bool CV_OperationsTest::TestMatxMultiplication()
{
    try
    {
        Matx33f mat(1, 1, 1, 0, 1, 1, 0, 0, 1); // Identity matrix
        Point2f pt(3, 4);
        Point3f res = mat * pt; // Correctly assumes homogeneous coordinates

        Vec3f res2 = mat*Vec3f(res.x, res.y, res.z);

        if(res.x != 8.0) throw test_excep();
        if(res.y != 5.0) throw test_excep();
        if(res.z != 1.0) throw test_excep();

        if(res2[0] != 14.0) throw test_excep();
        if(res2[1] != 6.0) throw test_excep();
        if(res2[2] != 1.0) throw test_excep();

        Matx44f mat44f(1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1);
        Matx44d mat44d(1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1);
        Scalar s(4, 3, 2, 1);
        Scalar sf = mat44f*s;
        Scalar sd = mat44d*s;

        if(sf[0] != 10.0) throw test_excep();
        if(sf[1] != 6.0) throw test_excep();
        if(sf[2] != 3.0) throw test_excep();
        if(sf[3] != 1.0) throw test_excep();

        if(sd[0] != 10.0) throw test_excep();
        if(sd[1] != 6.0) throw test_excep();
        if(sd[2] != 3.0) throw test_excep();
        if(sd[3] != 1.0) throw test_excep();
    }
    catch(const test_excep&)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return false;
    }
    return true;
}

bool CV_OperationsTest::TestMatxElementwiseDivison()
{
    try
    {
        Matx22f mat(2, 4, 6, 8);
        Matx22f mat2(2, 2, 2, 2);

        Matx22f res = mat.div(mat2);

        if(res(0, 0) != 1.0) throw test_excep();
        if(res(0, 1) != 2.0) throw test_excep();
        if(res(1, 0) != 3.0) throw test_excep();
        if(res(1, 1) != 4.0) throw test_excep();
    }
    catch(const test_excep&)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return false;
    }
    return true;
}


bool CV_OperationsTest::TestVec()
{
    try
    {
        cv::Mat hsvImage_f(5, 5, CV_32FC3), hsvImage_b(5, 5, CV_8UC3);
        int i = 0,j = 0;
        cv::Vec3f a;

        //these compile
        cv::Vec3b b = a;
        hsvImage_f.at<cv::Vec3f>(i,j) = cv::Vec3f((float)i,0,1);
        hsvImage_b.at<cv::Vec3b>(i,j) = cv::Vec3b(cv::Vec3f((float)i,0,1));

        //these don't
        b = cv::Vec3f(1,0,0);
        cv::Vec3b c;
        c = cv::Vec3f(0,0,1);
        hsvImage_b.at<cv::Vec3b>(i,j) = cv::Vec3f((float)i,0,1);
        hsvImage_b.at<cv::Vec3b>(i,j) = a;
        hsvImage_b.at<cv::Vec3b>(i,j) = cv::Vec3f(1,2,3);
    }
    catch(const test_excep&)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
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
            if (v10dzero[ii] != 0.0)
                throw test_excep();
        }

        Mat A(1, 32, CV_32F), B;
        for( int i = 0; i < A.cols; i++ )
            A.at<float>(i) = (float)(i <= 12 ? i : 24 - i);
        transpose(A, B);

        int minidx[2] = {0, 0}, maxidx[2] = {0, 0};
        double minval = 0, maxval = 0;
        minMaxIdx(A, &minval, &maxval, minidx, maxidx);

        if( !(minidx[0] == 0 && minidx[1] == 31 && maxidx[0] == 0 && maxidx[1] == 12 &&
                  minval == -7 && maxval == 12))
            throw test_excep();

        minMaxIdx(B, &minval, &maxval, minidx, maxidx);

        if( !(minidx[0] == 31 && minidx[1] == 0 && maxidx[0] == 12 && maxidx[1] == 0 &&
              minval == -7 && maxval == 12))
            throw test_excep();

        Matx33f b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f);
        Mat c;
        add(Mat::zeros(3, 3, CV_32F), b, c);
        CV_Assert( cvtest::norm(b, c, CV_C) == 0 );

        add(Mat::zeros(3, 3, CV_64F), b, c, noArray(), c.type());
        CV_Assert( cvtest::norm(b, c, CV_C) == 0 );

        add(Mat::zeros(6, 1, CV_64F), 1, c, noArray(), c.type());
        CV_Assert( cvtest::norm(Matx61f(1.f, 1.f, 1.f, 1.f, 1.f, 1.f), c, CV_C) == 0 );

        vector<Point2f> pt2d(3);
        vector<Point3d> pt3d(2);

        CV_Assert( Mat(pt2d).checkVector(2) == 3 && Mat(pt2d).checkVector(3) < 0 &&
                   Mat(pt3d).checkVector(2) < 0 && Mat(pt3d).checkVector(3) == 2 );

        Matx44f m44(0.8147f, 0.6324f, 0.9575f, 0.9572f,
                0.9058f, 0.0975f, 0.9649f, 0.4854f,
                0.1270f, 0.2785f, 0.1576f, 0.8003f,
                0.9134f, 0.5469f, 0.9706f, 0.1419f);
        double d = determinant(m44);
        CV_Assert( fabs(d - (-0.0262)) <= 0.001 );

        Cv32suf z;
        z.i = 0x80000000;
        CV_Assert( cvFloor(z.f) == 0 && cvCeil(z.f) == 0 && cvRound(z.f) == 0 );
    }
    catch(const test_excep&)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}


bool CV_OperationsTest::TestExp()
{
    Mat1f tt = Mat1f::ones(4,2);
    Mat1f outs;
    exp(-tt, outs);
    Mat1f tt2 = Mat1f::ones(4,1), outs2;
    exp(-tt2, outs2);
    return true;
}


bool CV_OperationsTest::TestSVD()
{
    try
    {
        Mat A = (Mat_<double>(3,4) << 1, 2, -1, 4, 2, 4, 3, 5, -1, -2, 6, 7);
        Mat x;
        SVD::solveZ(A,x);
        if( cvtest::norm(A*x, CV_C) > FLT_EPSILON )
            throw test_excep();

        SVD svd(A, SVD::FULL_UV);
        if( cvtest::norm(A*svd.vt.row(3).t(), CV_C) > FLT_EPSILON )
            throw test_excep();

        Mat Dp(3,3,CV_32FC1);
        Mat Dc(3,3,CV_32FC1);
        Mat Q(3,3,CV_32FC1);
        Mat U,Vt,R,T,W;

        Dp.at<float>(0,0)=0.86483884f; Dp.at<float>(0,1)= -0.3077251f; Dp.at<float>(0,2)=-0.55711365f;
        Dp.at<float>(1,0)=0.49294353f; Dp.at<float>(1,1)=-0.24209651f; Dp.at<float>(1,2)=-0.25084701f;
        Dp.at<float>(2,0)=0;           Dp.at<float>(2,1)=0;            Dp.at<float>(2,2)=0;

        Dc.at<float>(0,0)=0.75632739f; Dc.at<float>(0,1)= -0.38859656f; Dc.at<float>(0,2)=-0.36773083f;
        Dc.at<float>(1,0)=0.9699229f;  Dc.at<float>(1,1)=-0.49858192f;  Dc.at<float>(1,2)=-0.47134098f;
        Dc.at<float>(2,0)=0.10566688f; Dc.at<float>(2,1)=-0.060333252f; Dc.at<float>(2,2)=-0.045333147f;

        Q=Dp*Dc.t();
        SVD decomp;
        decomp=SVD(Q);
        U=decomp.u;
        Vt=decomp.vt;
        W=decomp.w;
        Mat I = Mat::eye(3, 3, CV_32F);

        if( cvtest::norm(U*U.t(), I, CV_C) > FLT_EPSILON ||
            cvtest::norm(Vt*Vt.t(), I, CV_C) > FLT_EPSILON ||
            W.at<float>(2) < 0 || W.at<float>(1) < W.at<float>(2) ||
            W.at<float>(0) < W.at<float>(1) ||
            cvtest::norm(U*Mat::diag(W)*Vt, Q, CV_C) > FLT_EPSILON )
            throw test_excep();
    }
    catch(const test_excep&)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
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

    if (!TestMatND())
        return;

    if (!TestSparseMat())
        return;

    if (!TestVec())
        return;

    if (!TestMatxMultiplication())
        return;

    if (!TestMatxElementwiseDivison())
        return;

    if (!TestSubMatAccess())
        return;

    if (!TestExp())
        return;

    if (!TestSVD())
        return;

    if (!operations1())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Core_Array, expressions) { CV_OperationsTest test; test.safe_run(); }

class CV_SparseMatTest : public cvtest::BaseTest
{
public:
    CV_SparseMatTest() {}
    ~CV_SparseMatTest() {}
protected:
    void run(int)
    {
        try
        {
            RNG& rng = theRNG();
            const int MAX_DIM=3;
            int sizes[MAX_DIM], idx[MAX_DIM];
            for( int iter = 0; iter < 100; iter++ )
            {
                ts->printf(cvtest::TS::LOG, ".");
                ts->update_context(this, iter, true);
                int k, dims = rng.uniform(1, MAX_DIM+1), p = 1;
                for( k = 0; k < dims; k++ )
                {
                    sizes[k] = rng.uniform(1, 30);
                    p *= sizes[k];
                }
                int j, nz = rng.uniform(0, (p+2)/2), nz0 = 0;
                SparseMat_<int> v(dims,sizes);

                CV_Assert( (int)v.nzcount() == 0 );

                SparseMatIterator_<int> it = v.begin();
                SparseMatIterator_<int> it_end = v.end();

                for( k = 0; it != it_end; ++it, ++k )
                    ;
                CV_Assert( k == 0 );

                int sum0 = 0, sum = 0;
                for( j = 0; j < nz; j++ )
                {
                    int val = rng.uniform(1, 100);
                    for( k = 0; k < dims; k++ )
                        idx[k] = rng.uniform(0, sizes[k]);
                    if( dims == 1 )
                    {
                        CV_Assert( v.ref(idx[0]) == v(idx[0]) );
                    }
                    else if( dims == 2 )
                    {
                        CV_Assert( v.ref(idx[0], idx[1]) == v(idx[0], idx[1]) );
                    }
                    else if( dims == 3 )
                    {
                        CV_Assert( v.ref(idx[0], idx[1], idx[2]) == v(idx[0], idx[1], idx[2]) );
                    }
                    CV_Assert( v.ref(idx) == v(idx) );
                    v.ref(idx) += val;
                    if( v(idx) == val )
                        nz0++;
                    sum0 += val;
                }

                CV_Assert( (int)v.nzcount() == nz0 );

                it = v.begin();
                it_end = v.end();

                for( k = 0; it != it_end; ++it, ++k )
                    sum += *it;
                CV_Assert( k == nz0 && sum == sum0 );

                v.clear();
                CV_Assert( (int)v.nzcount() == 0 );

                it = v.begin();
                it_end = v.end();

                for( k = 0; it != it_end; ++it, ++k )
                    ;
                CV_Assert( k == 0 );
            }
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Core_SparseMat, iterations) { CV_SparseMatTest test; test.safe_run(); }

TEST(MatTestRoi, adjustRoiOverflow)
{
    Mat m(15, 10, CV_32S);
    Mat roi(m, cv::Range(2, 10), cv::Range(3,6));
    int rowsInROI = roi.rows;
    roi.adjustROI(1, 0, 0, 0);

    ASSERT_EQ(roi.rows, rowsInROI + 1);

    roi.adjustROI(-m.rows, -m.rows, 0, 0);

    ASSERT_EQ(roi.rows, m.rows);
}


CV_ENUM(SortRowCol, SORT_EVERY_COLUMN, SORT_EVERY_ROW)
CV_ENUM(SortOrder, SORT_ASCENDING, SORT_DESCENDING)

PARAM_TEST_CASE(sortIdx, MatDepth, SortRowCol, SortOrder, Size, bool)
{
    int type;
    Size size;
    int flags;
    bool use_roi;

    Mat src, src_roi;
    Mat dst, dst_roi;

    virtual void SetUp()
    {
        int depth = GET_PARAM(0);
        int rowFlags = GET_PARAM(1);
        int orderFlags = GET_PARAM(2);
        size = GET_PARAM(3);
        use_roi = GET_PARAM(4);

        type = CV_MAKE_TYPE(depth, 1);

        flags = rowFlags | orderFlags;
    }

    void generateTestData()
    {
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, size, srcBorder, type, -100, 100);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, size, dstBorder, CV_32S, 5, 16);
    }

    template<typename T>
    void check_(const cv::Mat& values_, const cv::Mat_<int>& idx_)
    {
        cv::Mat_<T>& values = (cv::Mat_<T>&)values_;
        cv::Mat_<int>& idx = (cv::Mat_<int>&)idx_;
        size_t N = values.total();
        std::vector<bool> processed(N, false);
        int prevIdx = idx(0);
        T prevValue = values(prevIdx);
        processed[prevIdx] = true;
        for (size_t i = 1; i < N; i++)
        {
            int nextIdx = idx((int)i);
            T value = values(nextIdx);
            ASSERT_EQ(false, processed[nextIdx]) << "Indexes must be unique. i=" << i << " idx=" << nextIdx << std::endl << idx;
            processed[nextIdx] = true;
            if ((flags & SORT_DESCENDING) == SORT_DESCENDING)
                ASSERT_GE(prevValue, value) << "i=" << i << " prevIdx=" << prevIdx << " idx=" << nextIdx;
            else
                ASSERT_LE(prevValue, value) << "i=" << i << " prevIdx=" << prevIdx << " idx=" << nextIdx;
            prevValue = value;
            prevIdx = nextIdx;
        }
    }

    void validate()
    {
        ASSERT_EQ(CV_32SC1, dst_roi.type());
        ASSERT_EQ(size, dst_roi.size());
        bool isColumn = (flags & SORT_EVERY_COLUMN) == SORT_EVERY_COLUMN;
        size_t N = isColumn ? src_roi.cols : src_roi.rows;
        Mat values_row((int)N, 1, type), idx_row((int)N, 1, CV_32S);
        for (size_t i = 0; i < N; i++)
        {
            SCOPED_TRACE(cv::format("row/col=%d", (int)i));
            if (isColumn)
            {
                src_roi.col((int)i).copyTo(values_row);
                dst_roi.col((int)i).copyTo(idx_row);
            }
            else
            {
                src_roi.row((int)i).copyTo(values_row);
                dst_roi.row((int)i).copyTo(idx_row);
            }
            switch(type)
            {
            case CV_8U: check_<uchar>(values_row, idx_row); break;
            case CV_8S: check_<char>(values_row, idx_row); break;
            case CV_16S: check_<short>(values_row, idx_row); break;
            case CV_32S: check_<int>(values_row, idx_row); break;
            case CV_32F: check_<float>(values_row, idx_row); break;
            case CV_64F: check_<double>(values_row, idx_row); break;
            default: ASSERT_FALSE(true) << "Unsupported type: " << type;
            }
        }
    }
};

TEST_P(sortIdx, simple)
{
    for (int j = 0; j < 5; j++)
    {
        generateTestData();

        cv::sortIdx(src_roi, dst_roi, flags);
        validate();
    }
}

INSTANTIATE_TEST_CASE_P(Core, sortIdx, Combine(
        Values(CV_8U, CV_8S, CV_16S, CV_32S, CV_32F, CV_64F), // depth
        Values(SORT_EVERY_COLUMN, SORT_EVERY_ROW),
        Values(SORT_ASCENDING, SORT_DESCENDING),
        Values(Size(3, 3), Size(16, 8)),
        ::testing::Bool()
));


TEST(Core_sortIdx, regression_8941)
{
    cv::Mat src = (cv::Mat_<int>(3, 3) <<
        1, 2, 3,
        0, 9, 5,
        8, 1, 6
    );
    cv::Mat expected = (cv::Mat_<int>(3, 1) <<
        1,
        0,
        2
    );

    cv::Mat result;
    cv::sortIdx(src.col(0), result, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
#if 0
    std::cout << src.col(0) << std::endl;
    std::cout << result << std::endl;
#endif
    ASSERT_EQ(expected.size(), result.size());
    EXPECT_EQ(0, cvtest::norm(expected, result, NORM_INF)) <<
        "result=" << std::endl << result << std::endl <<
        "expected=" << std::endl << expected;
}

}} // namespace
