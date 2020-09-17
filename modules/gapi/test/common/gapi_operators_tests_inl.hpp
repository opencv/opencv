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
    g_api_ocv_pair_mat_scalar funcs(op);
    auto fun_gapi = funcs.g_api_function;
    auto fun_ocv  = funcs.ocv_function;

    if (op == DIVR)
        in_mat1.setTo(1, in_mat1 == 0);                               // avoiding zeros in divide input data
    if (op == DIV)
        sc += Scalar(sc[0] == 0, sc[1] == 0, sc[2] == 0, sc[3] == 0); // avoiding zeros in divide input data

    // G-API code & corresponding OpenCV code ////////////////////////////////

    cv::GMat in1;
    cv::GScalar in2;
    auto out = fun_gapi(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());

    fun_ocv(in_mat1, sc, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(MathOperatorMatMatTest, OperatorAccuracyTest )
{
    g_api_ocv_pair_mat_mat funcs(op);
    auto fun_gapi = funcs.g_api_function;
    auto fun_ocv  = funcs.ocv_function;

    if (op == DIV)
        in_mat2.setTo(1, in_mat2 == 0); // avoiding zeros in divide input data

    // G-API code & corresponding OpenCV code ////////////////////////////////

    cv::GMat in1;
    cv::GMat in2;
    auto out = fun_gapi(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    fun_ocv(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(NotOperatorTest, OperatorAccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = ~in;
    cv::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv =~in_mat1;
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

namespace for_test
{
class Foo {};

inline int operator&(Foo, int) { return 1; }
inline int operator|(Foo, int) { return 1; }
inline int operator^(Foo, int) { return 1; }
inline int operator~(Foo)      { return 1; }

inline int operator+(Foo, int) { return 1; }
inline int operator-(Foo, int) { return 1; }
inline int operator*(Foo, int) { return 1; }
inline int operator/(Foo, int) { return 1; }

inline int operator> (Foo, int) { return 1; }
inline int operator>=(Foo, int) { return 1; }
inline int operator< (Foo, int) { return 1; }
inline int operator<=(Foo, int) { return 1; }
inline int operator==(Foo, int) { return 1; }
inline int operator!=(Foo, int) { return 1; }

TEST(CVNamespaceOperatorsTest, OperatorCompilationTest)
{
    cv::GScalar sc;
    cv::GMat mat_in1, mat_in2;

    cv::GMat op_not = ~ mat_in1;

    cv::GMat op_mat_mat1  = mat_in1 &  mat_in2;
    cv::GMat op_mat_mat2  = mat_in1 |  mat_in2;
    cv::GMat op_mat_mat3  = mat_in1 ^  mat_in2;
    cv::GMat op_mat_mat4  = mat_in1 +  mat_in2;
    cv::GMat op_mat_mat5  = mat_in1 -  mat_in2;
    cv::GMat op_mat_mat6  = mat_in1 /  mat_in2;
    cv::GMat op_mat_mat7  = mat_in1 >  mat_in2;
    cv::GMat op_mat_mat8  = mat_in1 >= mat_in2;
    cv::GMat op_mat_mat9  = mat_in1 <  mat_in2;
    cv::GMat op_mat_mat10 = mat_in1 <= mat_in2;
    cv::GMat op_mat_mat11 = mat_in1 == mat_in2;
    cv::GMat op_mat_mat12 = mat_in1 != mat_in2;

    cv::GMat op_mat_sc1  = mat_in1 &  sc;
    cv::GMat op_mat_sc2  = mat_in1 |  sc;
    cv::GMat op_mat_sc3  = mat_in1 ^  sc;
    cv::GMat op_mat_sc4  = mat_in1 +  sc;
    cv::GMat op_mat_sc5  = mat_in1 -  sc;
    cv::GMat op_mat_sc6  = mat_in1 *  sc;
    cv::GMat op_mat_sc7  = mat_in1 /  sc;
    cv::GMat op_mat_sc8  = mat_in1 >  sc;
    cv::GMat op_mat_sc9  = mat_in1 >= sc;
    cv::GMat op_mat_sc10 = mat_in1 <  sc;
    cv::GMat op_mat_sc11 = mat_in1 <= sc;
    cv::GMat op_mat_sc12 = mat_in1 == sc;
    cv::GMat op_mat_sc13 = mat_in1 != sc;

    cv::GMat op_sc_mat1  = sc &  mat_in2;
    cv::GMat op_sc_mat2  = sc |  mat_in2;
    cv::GMat op_sc_mat3  = sc ^  mat_in2;
    cv::GMat op_sc_mat4  = sc +  mat_in2;
    cv::GMat op_sc_mat5  = sc -  mat_in2;
    cv::GMat op_sc_mat6  = sc *  mat_in2;
    cv::GMat op_sc_mat7  = sc /  mat_in2;
    cv::GMat op_sc_mat8  = sc >  mat_in2;
    cv::GMat op_sc_mat9  = sc >= mat_in2;
    cv::GMat op_sc_mat10 = sc <  mat_in2;
    cv::GMat op_sc_mat11 = sc <= mat_in2;
    cv::GMat op_sc_mat12 = sc == mat_in2;
    cv::GMat op_sc_mat13 = sc != mat_in2;

    cv::GMat mul_mat_float1 = mat_in1 * 1.0f;
    cv::GMat mul_mat_float2 = 1.0f * mat_in2;
    // No compilation errors expected
}
} // for_test
} // opencv_test

#endif // OPENCV_GAPI_OPERATOR_TESTS_INL_COMMON_HPP
