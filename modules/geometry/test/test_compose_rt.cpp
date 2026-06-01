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

class Differential
{
public:
    typedef Mat_<double> mat_t;

    Differential(double eps_, const mat_t& rv1_, const mat_t& tv1_, const mat_t& rv2_, const mat_t& tv2_)
        : rv1(rv1_), tv1(tv1_), rv2(rv2_), tv2(tv2_), eps(eps_), ev(3, 1) {}

    void dRv1(mat_t& dr3_dr1, mat_t& dt3_dr1)
    {
        dr3_dr1.create(3, 3);     dt3_dr1.create(3, 3);

        for(int i = 0; i < 3; ++i)
        {
            ev.setTo(Scalar(0));    ev(i, 0) = eps;

            composeRT( rv1 + ev, tv1, rv2, tv2, rv3_p, tv3_p);
            composeRT( rv1 - ev, tv1, rv2, tv2, rv3_m, tv3_m);

            dr3_dr1.col(i) = rv3_p - rv3_m;
            dt3_dr1.col(i) = tv3_p - tv3_m;
        }
        dr3_dr1 /= 2 * eps;       dt3_dr1 /= 2 * eps;
    }

    void dRv2(mat_t& dr3_dr2, mat_t& dt3_dr2)
    {
        dr3_dr2.create(3, 3);     dt3_dr2.create(3, 3);

        for(int i = 0; i < 3; ++i)
        {
            ev.setTo(Scalar(0));    ev(i, 0) = eps;

            composeRT( rv1, tv1, rv2 + ev, tv2, rv3_p, tv3_p);
            composeRT( rv1, tv1, rv2 - ev, tv2, rv3_m, tv3_m);

            dr3_dr2.col(i) = rv3_p - rv3_m;
            dt3_dr2.col(i) = tv3_p - tv3_m;
        }
        dr3_dr2 /= 2 * eps;       dt3_dr2 /= 2 * eps;
    }

    void dTv1(mat_t& drt3_dt1, mat_t& dt3_dt1)
    {
        drt3_dt1.create(3, 3);     dt3_dt1.create(3, 3);

        for(int i = 0; i < 3; ++i)
        {
            ev.setTo(Scalar(0));    ev(i, 0) = eps;

            composeRT( rv1, tv1 + ev, rv2, tv2, rv3_p, tv3_p);
            composeRT( rv1, tv1 - ev, rv2, tv2, rv3_m, tv3_m);

            drt3_dt1.col(i) = rv3_p - rv3_m;
            dt3_dt1.col(i) = tv3_p - tv3_m;
        }
        drt3_dt1 /= 2 * eps;       dt3_dt1 /= 2 * eps;
    }

    void dTv2(mat_t& dr3_dt2, mat_t& dt3_dt2)
    {
        dr3_dt2.create(3, 3);     dt3_dt2.create(3, 3);

        for(int i = 0; i < 3; ++i)
        {
            ev.setTo(Scalar(0));    ev(i, 0) = eps;

            composeRT( rv1, tv1, rv2, tv2 + ev, rv3_p, tv3_p);
            composeRT( rv1, tv1, rv2, tv2 - ev, rv3_m, tv3_m);

            dr3_dt2.col(i) = rv3_p - rv3_m;
            dt3_dt2.col(i) = tv3_p - tv3_m;
        }
        dr3_dt2 /= 2 * eps;       dt3_dt2 /= 2 * eps;
    }

private:
    const mat_t& rv1, tv1, rv2, tv2;
    double eps;
    Mat_<double> ev;

    Differential& operator=(const Differential&);
    Mat rv3_m, tv3_m, rv3_p, tv3_p;
};

class CV_composeRT_Test : public cvtest::BaseTest
{
public:
    CV_composeRT_Test() {}
    ~CV_composeRT_Test() {}
protected:

    void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);

        Mat_<double> rvec1(3, 1), tvec1(3, 1), rvec2(3, 1), tvec2(3, 1);

        randu(rvec1, Scalar(0), Scalar(6.29));
        randu(rvec2, Scalar(0), Scalar(6.29));

        randu(tvec1, Scalar(-2), Scalar(2));
        randu(tvec2, Scalar(-2), Scalar(2));

        Mat rvec3, tvec3;
        composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3);

        Mat rvec3_exp, tvec3_exp;

        Mat rmat1, rmat2;
        cv::Rodrigues(rvec1, rmat1); // TODO cvtest
        cv::Rodrigues(rvec2, rmat2); // TODO cvtest
        cv::Rodrigues(rmat2 * rmat1, rvec3_exp); // TODO cvtest

        tvec3_exp = rmat2 * tvec1 + tvec2;

        const double thres = 1e-5;
        if (cv::norm(rvec3_exp, rvec3) > thres ||  cv::norm(tvec3_exp, tvec3) > thres)
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);

        const double eps = 1e-3;
        Differential diff(eps, rvec1, tvec1, rvec2, tvec2);

        Mat dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2;

        composeRT(rvec1, tvec1, rvec2, tvec2, rvec3, tvec3,
            dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2);

        Mat_<double> dr3_dr1, dt3_dr1;
           diff.dRv1(dr3_dr1, dt3_dr1);

        if (cv::norm(dr3_dr1, dr3dr1) > thres || cv::norm(dt3_dr1, dt3dr1) > thres)
        {
            ts->printf( cvtest::TS::LOG, "Invalid derivates by r1\n" );
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }

        Mat_<double> dr3_dr2, dt3_dr2;
           diff.dRv2(dr3_dr2, dt3_dr2);

        if (cv::norm(dr3_dr2, dr3dr2) > thres || cv::norm(dt3_dr2, dt3dr2) > thres)
        {
            ts->printf( cvtest::TS::LOG, "Invalid derivates by r2\n" );
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }

        Mat_<double> dr3_dt1, dt3_dt1;
           diff.dTv1(dr3_dt1, dt3_dt1);

        if (cv::norm(dr3_dt1, dr3dt1) > thres || cv::norm(dt3_dt1, dt3dt1) > thres)
        {
            ts->printf( cvtest::TS::LOG, "Invalid derivates by t1\n" );
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }

        Mat_<double> dr3_dt2, dt3_dt2;
           diff.dTv2(dr3_dt2, dt3_dt2);

        if (cv::norm(dr3_dt2, dr3dt2) > thres || cv::norm(dt3_dt2, dt3dt2) > thres)
        {
            ts->printf( cvtest::TS::LOG, "Invalid derivates by t2\n" );
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }
    }
};

TEST(Calib3d_ComposeRT, accuracy) { CV_composeRT_Test test; test.safe_run(); }

}} // namespace
