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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
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
#include <iostream>
#include "opencv2/core/ocl.hpp"

using namespace cv;
using namespace std;

class CV_UMatTest :
        public cvtest::BaseTest
{
public:
    CV_UMatTest() {}
    ~CV_UMatTest() {}
protected:
    void run(int);

    struct test_excep
    {
        test_excep(const string& _s=string("")) : s(_s) { }
        string s;
    };

    bool TestUMat();

    void checkDiff(const Mat& m1, const Mat& m2, const string& s)
    {
        if (norm(m1, m2, NORM_INF) != 0)
            throw test_excep(s);
    }
    void checkDiffF(const Mat& m1, const Mat& m2, const string& s)
    {
        if (norm(m1, m2, NORM_INF) > 1e-5)
            throw test_excep(s);
    }
};

#define STR(a) STR2(a)
#define STR2(a) #a

#define CHECK_DIFF(a, b) checkDiff(a, b, "(" #a ")  !=  (" #b ")  at l." STR(__LINE__))
#define CHECK_DIFF_FLT(a, b) checkDiffF(a, b, "(" #a ")  !=(eps)  (" #b ")  at l." STR(__LINE__))


bool CV_UMatTest::TestUMat()
{
    try
    {
        Mat a(100, 100, CV_16SC2), b, c;
        randu(a, Scalar::all(-100), Scalar::all(100));
        Rect roi(1, 3, 5, 4);
        Mat ra(a, roi), rb, rc, rc0;
        UMat ua, ura, ub, urb, uc, urc;
        a.copyTo(ua);
        ua.copyTo(b);
        CHECK_DIFF(a, b);

        ura = ua(roi);
        ura.copyTo(rb);

        CHECK_DIFF(ra, rb);

        ra += Scalar::all(1.f);
        {
            Mat temp = ura.getMat(ACCESS_RW);
            temp += Scalar::all(1.f);
        }
        ra.copyTo(rb);
        CHECK_DIFF(ra, rb);

        b = a.clone();
        ra = a(roi);
        rb = b(roi);
        randu(b, Scalar::all(-100), Scalar::all(100));
        b.copyTo(ub);
        urb = ub(roi);

        /*std::cout << "==============================================\nbefore op (CPU):\n";
        std::cout << "ra: " << ra << std::endl;
        std::cout << "rb: " << rb << std::endl;*/

        ra.copyTo(ura);
        rb.copyTo(urb);
        ra.release();
        rb.release();
        ura.copyTo(ra);
        urb.copyTo(rb);

        /*std::cout << "==============================================\nbefore op (GPU):\n";
        std::cout << "ra: " << ra << std::endl;
        std::cout << "rb: " << rb << std::endl;*/

        cv::max(ra, rb, rc);
        cv::max(ura, urb, urc);
        urc.copyTo(rc0);

        /*std::cout << "==============================================\nafter op:\n";
        std::cout << "rc: " << rc << std::endl;
        std::cout << "rc0: " << rc0 << std::endl;*/

        CHECK_DIFF(rc0, rc);

        {
            UMat tmp = rc0.getUMat(ACCESS_WRITE);
            cv::max(ura, urb, tmp);
        }
        CHECK_DIFF(rc0, rc);

        ura.copyTo(urc);
        cv::max(urc, urb, urc);
        urc.copyTo(rc0);
        CHECK_DIFF(rc0, rc);

        rc = ra ^ rb;
        cv::bitwise_xor(ura, urb, urc);
        urc.copyTo(rc0);

        /*std::cout << "==============================================\nafter op:\n";
        std::cout << "ra: " << rc0 << std::endl;
        std::cout << "rc: " << rc << std::endl;*/

        CHECK_DIFF(rc0, rc);

        rc = ra + rb;
        cv::add(ura, urb, urc);
        urc.copyTo(rc0);

        CHECK_DIFF(rc0, rc);

        cv::subtract(ra, Scalar::all(5), rc);
        cv::subtract(ura, Scalar::all(5), urc);
        urc.copyTo(rc0);

        CHECK_DIFF(rc0, rc);
    }
    catch (const test_excep& e)
    {
        ts->printf(cvtest::TS::LOG, "%s\n", e.s.c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return false;
    }
    return true;
}

void CV_UMatTest::run( int /* start_from */)
{
    printf("Use OpenCL: %s\nHave OpenCL: %s\n",
           ocl::useOpenCL() ? "TRUE" : "FALSE",
           ocl::haveOpenCL() ? "TRUE" : "FALSE" );

    if (!TestUMat())
        return;

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Core_UMat, base) { CV_UMatTest test; test.safe_run(); }

TEST(Core_UMat, getUMat)
{
    {
        int a[3] = { 1, 2, 3 };
        Mat m = Mat(1, 1, CV_32SC3, a);
        UMat u = m.getUMat(ACCESS_READ);
        EXPECT_NE((void*)NULL, u.u);
    }

    {
        Mat m(10, 10, CV_8UC1), ref;
        for (int y = 0; y < m.rows; ++y)
        {
            uchar * const ptr = m.ptr<uchar>(y);
            for (int x = 0; x < m.cols; ++x)
                ptr[x] = (uchar)(x + y * 2);
        }

        ref = m.clone();
        Rect r(1, 1, 8, 8);
        ref(r).setTo(17);

        {
            UMat um = m(r).getUMat(ACCESS_WRITE);
            um.setTo(17);
        }

        double err = norm(m, ref, NORM_INF);
        if (err > 0)
        {
            std::cout << "m: " << std::endl << m << std::endl;
            std::cout << "ref: " << std::endl << ref << std::endl;
        }
        EXPECT_EQ(0., err);
    }
}

TEST(UMat, Sync)
{
    UMat um(10, 10, CV_8UC1);

    {
        Mat m = um.getMat(ACCESS_WRITE);
        m.setTo(cv::Scalar::all(17));
    }

    um.setTo(cv::Scalar::all(19));

    EXPECT_EQ(0, cv::norm(um.getMat(ACCESS_READ), cv::Mat(um.size(), um.type(), 19), NORM_INF));
}

#define EXPECT_MAT_NEAR(m1, m2) ASSERT_EQ(0, cv::norm(m1, m1, cv::NORM_INF))

TEST(UMat, setOpenCL)
{
    // save the current state
    bool useOCL = ocl::useOpenCL();

    Mat m = (Mat_<uchar>(3,3)<<0,1,2,3,4,5,6,7,8);

    ocl::setUseOpenCL(true);
    UMat um1;
    m.copyTo(um1);

    ocl::setUseOpenCL(false);
    UMat um2;
    m.copyTo(um2);

    ocl::setUseOpenCL(true);
    countNonZero(um1);
    countNonZero(um2);

    um1.copyTo(um2);
    EXPECT_MAT_NEAR(um1, um2);
    EXPECT_MAT_NEAR(um1, m);
    um2.copyTo(um1);
    EXPECT_MAT_NEAR(um1, m);
    EXPECT_MAT_NEAR(um1, um2);

    ocl::setUseOpenCL(false);
    countNonZero(um1);
    countNonZero(um2);

    um1.copyTo(um2);
    EXPECT_MAT_NEAR(um1, um2);
    EXPECT_MAT_NEAR(um1, m);
    um2.copyTo(um1);
    EXPECT_MAT_NEAR(um1, um2);
    EXPECT_MAT_NEAR(um1, m);

    // reset state to the previous one
    ocl::setUseOpenCL(useOCL);
}
