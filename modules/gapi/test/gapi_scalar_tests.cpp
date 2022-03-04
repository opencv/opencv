// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <iostream>

namespace opencv_test
{

TEST(GAPI_Scalar, Argument)
{
    cv::Size sz(2, 2);
    cv::Mat in_mat(sz, CV_8U);
    cv::randn(in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::GComputationT<cv::GMat (cv::GMat, cv::GScalar)> mulS([](cv::GMat in, cv::GScalar c)
    {
        return in*c;
    });

    cv::Mat out_mat(sz, CV_8U);
    mulS.apply(in_mat, cv::Scalar(2), out_mat);

    cv::Mat reference = in_mat*2;
    EXPECT_EQ(0, cvtest::norm(out_mat, reference, NORM_INF));
}

TEST(GAPI_Scalar, ReturnValue)
{
    const cv::Size sz(2, 2);
    cv::Mat in_mat(sz, CV_8U, cv::Scalar(1));

    cv::GComputationT<cv::GScalar (cv::GMat)> sum_of_sum([](cv::GMat in)
    {
        return cv::gapi::sum(in + in);
    });

    cv::Scalar out;
    sum_of_sum.apply(in_mat, out);

    EXPECT_EQ(8, out[0]);
}

TEST(GAPI_Scalar, TmpScalar)
{
    const cv::Size sz(2, 2);
    cv::Mat in_mat(sz, CV_8U, cv::Scalar(1));

    cv::GComputationT<cv::GMat (cv::GMat)> mul_by_sum([](cv::GMat in)
    {
        return in * cv::gapi::sum(in);
    });

    cv::Mat out_mat(sz, CV_8U);
    mul_by_sum.apply(in_mat, out_mat);

    cv::Mat reference = cv::Mat(sz, CV_8U, cv::Scalar(4));
    EXPECT_EQ(0, cvtest::norm(out_mat, reference, NORM_INF));
}

TEST(GAPI_ScalarWithValue, Simple_Arithmetic_Pipeline)
{
    GMat in;
    GMat out = (in + 1) * 2;
    cv::GComputation comp(in, out);

    cv::Mat in_mat  = cv::Mat::eye(3, 3, CV_8UC1);
    cv::Mat ref_mat, out_mat;

    ref_mat = (in_mat + 1) * 2;
    comp.apply(in_mat, out_mat);

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_ScalarWithValue, GScalar_Initilization)
{
    cv::Scalar sc(2);
    cv::GMat in;
    cv::GScalar s(sc);
    cv::GComputation comp(in, cv::gapi::mulC(in, s));

    cv::Mat in_mat = cv::Mat::eye(3, 3, CV_8UC1);
    cv::Mat ref_mat, out_mat;
    cv::multiply(in_mat, sc, ref_mat, 1, CV_8UC1);
    comp.apply(cv::gin(in_mat), cv::gout(out_mat));

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_ScalarWithValue, Constant_GScalar_In_Middle_Graph)
{
    cv::Scalar  sc(5);
    cv::GMat    in1;
    cv::GScalar in2;
    cv::GScalar s(sc);

    auto add_out = cv::gapi::addC(in1, in2);
    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(cv::gapi::mulC(add_out, s)));

    cv::Mat    in_mat = cv::Mat::eye(3, 3, CV_8UC1);
    cv::Scalar in_scalar(3);

    cv::Mat ref_mat, out_mat, add_mat;
    cv::add(in_mat, in_scalar, add_mat);
    cv::multiply(add_mat, sc, ref_mat, 1, CV_8UC1);
    comp.apply(cv::gin(in_mat, in_scalar), cv::gout(out_mat));

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

} // namespace opencv_test
