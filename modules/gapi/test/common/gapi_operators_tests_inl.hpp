// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPERATOR_TESTS_INL_COMMON_HPP
#define OPENCV_GAPI_OPERATOR_TESTS_INL_COMMON_HPP

#include "gapi_operators_tests.hpp"

namespace opencv_test
{
TEST_P(MathOperatorMatScalarTest, OperatorAccuracyTest )
{
    compare_f cmpF;
    g_api_ocv_pair_mat_scalar op;
    int type = 0, dtype = 0;
    cv::Size sz;
    bool initOutMatr = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, op, type, sz, dtype, initOutMatr, compile_args) = GetParam();
    initMatsRandU(type, sz, dtype, initOutMatr);

    auto fun_gapi = op.g_api_function;
    auto fun_ocv = op.ocv_function ;

    // G-API code & corresponding OpenCV code ////////////////////////////////

    cv::GMat in1;
    cv::GScalar in2;
    auto out = fun_gapi(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    fun_ocv(in_mat1, sc, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(MathOperatorMatMatTest, OperatorAccuracyTest )
{
    compare_f cmpF;
    g_api_ocv_pair_mat_mat op;
    int type = 0, dtype = 0;
    cv::Size sz;
    bool initOutMatr = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, op, type, sz, dtype, initOutMatr, compile_args) = GetParam();
    initMatsRandU(type, sz, dtype, initOutMatr);

    auto fun_gapi = op.g_api_function;
    auto fun_ocv = op.ocv_function ;

    // G-API code & corresponding OpenCV code ////////////////////////////////

    cv::GMat in1;
    cv::GMat in2;
    auto out = fun_gapi(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    fun_ocv(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(NotOperatorTest, OperatorAccuracyTest)
{
    cv::Size sz_in = std::get<1>(GetParam());
    initMatrixRandU(std::get<0>(GetParam()), sz_in, std::get<0>(GetParam()), std::get<2>(GetParam()));
    cv::GCompileArgs compile_args;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = ~in;
    cv::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv =~in_mat1;
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}
} // opencv_test

#endif // OPENCV_GAPI_OPERATOR_TESTS_INL_COMMON_HPP
