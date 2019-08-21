// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

namespace opencv_test
{

namespace
{
    cv::Mat diff(cv::Mat m1, cv::Mat m2, int t)
    {
        return cv::abs(m1-m2) > t;
    }

    int non_zero3(cv::Mat m3c)
    {
        std::vector<cv::Mat> mm(3);
        cv::split(m3c, mm);
        return (  cv::countNonZero(mm[0])
                + cv::countNonZero(mm[1])
                + cv::countNonZero(mm[2]));
    }
}

TEST(GAPI_Typed, UnaryOp)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat         (sz, CV_8UC3),
        out_mat_untyped(sz, CV_8UC3),
        out_mat_typed1 (sz, CV_8UC3),
        out_mat_typed2 (sz, CV_8UC3),
        out_mat_cv     (sz, CV_8UC3);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in;
        cv::GMat out = cv::gapi::RGB2YUV(in);
        return cv::GComputation(in, out);
    });
    cvtU.apply(in_mat, out_mat_untyped);

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GMat (cv::GMat)> cvtT(cv::gapi::RGB2YUV);
    auto cvtTComp = cvtT.compile(cv::descr_of(in_mat));

    cvtT.apply(in_mat, out_mat_typed1);
    cvtTComp(in_mat, out_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    cv::cvtColor(in_mat, out_mat_cv, cv::COLOR_RGB2YUV);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    cv::Mat
        diff_u  = diff(out_mat_cv, out_mat_untyped, 0),
        diff_t  = diff(out_mat_cv, out_mat_typed1,  0),
        diff_tc = diff(out_mat_cv, out_mat_typed2,  0);

    EXPECT_EQ(0, non_zero3(diff_u));
    EXPECT_EQ(0, non_zero3(diff_t));
    EXPECT_EQ(0, non_zero3(diff_tc));
}

TEST(GAPI_Typed, BinaryOp)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat1        (sz, CV_8UC1),
        in_mat2        (sz, CV_8UC1),
        out_mat_untyped(sz, CV_8UC1),
        out_mat_typed1 (sz, CV_8UC1),
        out_mat_typed2 (sz, CV_8UC1),
        out_mat_cv     (sz, CV_8UC1);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in1, in2;
        cv::GMat out = cv::gapi::add(in1, in2);
        return cv::GComputation({in1, in2}, {out});
    });
    std::vector<cv::Mat> u_ins  = {in_mat1, in_mat2};
    std::vector<cv::Mat> u_outs = {out_mat_untyped};
    cvtU.apply(u_ins, u_outs);

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GMat (cv::GMat, cv::GMat)> cvtT([](cv::GMat m1, cv::GMat m2)
    {
        return m1+m2;
    });
    auto cvtTC =  cvtT.compile(cv::descr_of(in_mat1),
                               cv::descr_of(in_mat2));

    cvtT.apply(in_mat1, in_mat2, out_mat_typed1);
    cvtTC(in_mat1, in_mat2, out_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    cv::add(in_mat1, in_mat2, out_mat_cv);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    cv::Mat
        diff_u  = diff(out_mat_cv, out_mat_untyped, 0),
        diff_t  = diff(out_mat_cv, out_mat_typed1,  0),
        diff_tc = diff(out_mat_cv, out_mat_typed2,  0);

    EXPECT_EQ(0, cv::countNonZero(diff_u));
    EXPECT_EQ(0, cv::countNonZero(diff_t));
    EXPECT_EQ(0, cv::countNonZero(diff_tc));
}


TEST(GAPI_Typed, MultipleOuts)
{
    // Initialization //////////////////////////////////////////////////////////
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat        (sz, CV_8UC1),
        out_mat_unt1  (sz, CV_8UC1),
        out_mat_unt2  (sz, CV_8UC1),
        out_mat_typed1(sz, CV_8UC1),
        out_mat_typed2(sz, CV_8UC1),
        out_mat_comp1 (sz, CV_8UC1),
        out_mat_comp2 (sz, CV_8UC1),
        out_mat_cv1   (sz, CV_8UC1),
        out_mat_cv2   (sz, CV_8UC1);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in;
        cv::GMat out1 = in * 2.f;
        cv::GMat out2 = in * 4.f;
        return cv::GComputation({in}, {out1, out2});
    });
    std::vector<cv::Mat> u_ins  = {in_mat};
    std::vector<cv::Mat> u_outs = {out_mat_unt1, out_mat_unt2};
    cvtU.apply(u_ins, u_outs);

    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<std::tuple<cv::GMat, cv::GMat> (cv::GMat)> cvtT([](cv::GMat in)
    {
        return std::make_tuple(in*2.f, in*4.f);
    });
    auto cvtTC =  cvtT.compile(cv::descr_of(in_mat));

    cvtT.apply(in_mat, out_mat_typed1, out_mat_typed2);
    cvtTC(in_mat, out_mat_comp1, out_mat_comp2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_mat_cv1 = in_mat * 2.f;
    out_mat_cv2 = in_mat * 4.f;

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    cv::Mat
        diff_u1 = diff(out_mat_cv1, out_mat_unt1,   0),
        diff_u2 = diff(out_mat_cv2, out_mat_unt2,   0),
        diff_t1 = diff(out_mat_cv1, out_mat_typed1, 0),
        diff_t2 = diff(out_mat_cv2, out_mat_typed2, 0),
        diff_c1 = diff(out_mat_cv1, out_mat_comp1,  0),
        diff_c2 = diff(out_mat_cv2, out_mat_comp2,  0);

    EXPECT_EQ(0, cv::countNonZero(diff_u1));
    EXPECT_EQ(0, cv::countNonZero(diff_u2));
    EXPECT_EQ(0, cv::countNonZero(diff_t1));
    EXPECT_EQ(0, cv::countNonZero(diff_t2));
    EXPECT_EQ(0, cv::countNonZero(diff_c1));
    EXPECT_EQ(0, cv::countNonZero(diff_c2));
}

} // opencv_test
