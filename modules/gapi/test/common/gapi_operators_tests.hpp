// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP
#define OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"

namespace opencv_test
{

struct g_api_ocv_pair_mat_scalar {
    using g_api_function_t = std::function<cv::GMat(cv::GMat,cv::GScalar)>;
    using ocv_function_t   = std::function<void(cv::Mat const&, cv::Scalar, cv::Mat&)>;

    std::string      name;
    g_api_function_t g_api_function;
    ocv_function_t   ocv_function;


    g_api_ocv_pair_mat_scalar(std::string const& n, g_api_function_t const& g, ocv_function_t const& o)
    : name(n), g_api_function(g), ocv_function(o) {}

    g_api_ocv_pair_mat_scalar() = default;

    friend std::ostream& operator<<(std::ostream& o, const g_api_ocv_pair_mat_scalar& p)
    {
        return o<<p.name;
    }
};

struct g_api_ocv_pair_mat_mat {
    using g_api_function_t = std::function<cv::GMat(cv::GMat,cv::GMat)>;
    using ocv_function_t   = std::function<void(cv::Mat const&, cv::Mat const&, cv::Mat&)>;

    std::string      name;
    g_api_function_t g_api_function;
    ocv_function_t   ocv_function;


    g_api_ocv_pair_mat_mat(std::string const& n, g_api_function_t const& g, ocv_function_t const& o)
    : name(n), g_api_function(g), ocv_function(o) {}

    g_api_ocv_pair_mat_mat() = default;

    friend std::ostream& operator<<(std::ostream& o, const g_api_ocv_pair_mat_mat& p)
    {
        return o<<p.name;
    }
};

////////////////////////////////////////////////////////////////////////////////
//
// FIXME: Please refactor this test to a template test (T,U) with enum (OP)
//
////////////////////////////////////////////////////////////////////////////////
namespace
{


//declare test cases for matrix and scalar operators
g_api_ocv_pair_mat_scalar opPlus =  {std::string{"operator+"},
                                    [](cv::GMat in,cv::GScalar c){return in+c;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::add(in, c, out);}};
g_api_ocv_pair_mat_scalar opPlusR = {std::string{"rev_operator+"},
                                    [](cv::GMat in,cv::GScalar c){return c+in;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::add(c, in, out);}};
g_api_ocv_pair_mat_scalar opMinus = {std::string{"operator-"},
                                    [](cv::GMat in,cv::GScalar c){return in-c;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::subtract(in, c, out);}};
g_api_ocv_pair_mat_scalar opMinusR = {std::string{"rev_operator-"},
                                    [](cv::GMat in,cv::GScalar c){return c-in;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::subtract(c, in, out);}};
g_api_ocv_pair_mat_scalar opMul =   {std::string{"operator*"},
                                    [](cv::GMat in,cv::GScalar c){return in*c;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::multiply(in, c, out);}};
g_api_ocv_pair_mat_scalar opMulR =  {std::string{"rev_operator*"},
                                    [](cv::GMat in,cv::GScalar c){return c*in;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::multiply(c, in, out);}};
g_api_ocv_pair_mat_scalar opDiv =   {std::string{"operator/"},
                                    [](cv::GMat in,cv::GScalar c){return in/c;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::divide(in, c, out);}};
g_api_ocv_pair_mat_scalar opDivR =  {std::string{"rev_operator/"},
                                    [](cv::GMat in,cv::GScalar c){return c/in;},
                                    [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::divide(c, in, out);}};

g_api_ocv_pair_mat_scalar opGT = {std::string{"operator>"},
                                            [](cv::GMat in,cv::GScalar c){return in>c;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_GT);}};
g_api_ocv_pair_mat_scalar opLT = {std::string{"operator<"},
                                            [](cv::GMat in,cv::GScalar c){return in<c;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_LT);}};
g_api_ocv_pair_mat_scalar opGE = {std::string{"operator>="},
                                            [](cv::GMat in,cv::GScalar c){return in>=c;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_GE);}};
g_api_ocv_pair_mat_scalar opLE = {std::string{"operator<="},
                                            [](cv::GMat in,cv::GScalar c){return in<=c;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_LE);}};
g_api_ocv_pair_mat_scalar opEQ = {std::string{"operator=="},
                                            [](cv::GMat in,cv::GScalar c){return in==c;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_EQ);}};
g_api_ocv_pair_mat_scalar opNE = {std::string{"operator!="},
                                            [](cv::GMat in,cv::GScalar c){return in!=c;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_NE);}};
g_api_ocv_pair_mat_scalar opGTR = {std::string{"rev_operator>"},
                                            [](cv::GMat in,cv::GScalar c){return c>in;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_GT);}};
g_api_ocv_pair_mat_scalar opLTR = {std::string{"rev_operator<"},
                                            [](cv::GMat in,cv::GScalar c){return c<in;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_LT);}};
g_api_ocv_pair_mat_scalar opGER = {std::string{"rev_operator>="},
                                            [](cv::GMat in,cv::GScalar c){return c>=in;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_GE);}};
g_api_ocv_pair_mat_scalar opLER = {std::string{"rev_operator<="},
                                            [](cv::GMat in,cv::GScalar c){return c<=in;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_LE);}};
g_api_ocv_pair_mat_scalar opEQR = {std::string{"rev_operator=="},
                                            [](cv::GMat in,cv::GScalar c){return c==in;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_EQ);}};
g_api_ocv_pair_mat_scalar opNER = {std::string{"rev_operator!="},
                                            [](cv::GMat in,cv::GScalar c){return c!=in;},
                                            [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_NE);}};

g_api_ocv_pair_mat_scalar opAND = {std::string{"operator&"},
                                        [](cv::GMat in1,cv::GScalar in2){return in1&in2;},
                                        [](const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out){cv::bitwise_and(in1, in2, out);}};
g_api_ocv_pair_mat_scalar opOR = {std::string{"operator|"},
                                        [](cv::GMat in1,cv::GScalar in2){return in1|in2;},
                                        [](const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out){cv::bitwise_or(in1, in2, out);}};
g_api_ocv_pair_mat_scalar opXOR = {std::string{"operator^"},
                                        [](cv::GMat in1,cv::GScalar in2){return in1^in2;},
                                        [](const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out){cv::bitwise_xor(in1, in2, out);}};
g_api_ocv_pair_mat_scalar opANDR = {std::string{"rev_operator&"},
                                        [](cv::GMat in1,cv::GScalar in2){return in2&in1;},
                                        [](const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out){cv::bitwise_and(in2, in1, out);}};
g_api_ocv_pair_mat_scalar opORR = {std::string{"rev_operator|"},
                                        [](cv::GMat in1,cv::GScalar in2){return in2|in1;},
                                        [](const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out){cv::bitwise_or(in2, in1, out);}};
g_api_ocv_pair_mat_scalar opXORR = {std::string{"rev_operator^"},
                                        [](cv::GMat in1,cv::GScalar in2){return in2^in1;},
                                        [](const cv::Mat& in1, const cv::Scalar& in2, cv::Mat& out){cv::bitwise_xor(in2, in1, out);}};

// declare test cases for matrix and matrix operators
g_api_ocv_pair_mat_mat opPlusM =  {std::string{"operator+"},
                                            [](cv::GMat in1,cv::GMat in2){return in1+in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::add(in1, in2, out);}};
g_api_ocv_pair_mat_mat opMinusM = {std::string{"operator-"},
                                            [](cv::GMat in,cv::GMat in2){return in-in2;},
                                            [](const cv::Mat& in, const cv::Mat& in2, cv::Mat& out){cv::subtract(in, in2, out);}};
g_api_ocv_pair_mat_mat opDivM = {std::string{"operator/"},
                                            [](cv::GMat in,cv::GMat in2){return in/in2;},
                                            [](const cv::Mat& in, const cv::Mat& in2, cv::Mat& out){cv::divide(in, in2, out);}};
g_api_ocv_pair_mat_mat opGreater =  {std::string{"operator>"},
                                            [](cv::GMat in1,cv::GMat in2){return in1>in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_GT);}};
g_api_ocv_pair_mat_mat opGreaterEq = {std::string{"operator>="},
                                            [](cv::GMat in1,cv::GMat in2){return in1>=in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_GE);}};
g_api_ocv_pair_mat_mat opLess = {std::string{"operator<"},
                                            [](cv::GMat in1,cv::GMat in2){return in1<in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_LT);}};
g_api_ocv_pair_mat_mat opLessEq = {std::string{"operator<="},
                                            [](cv::GMat in1,cv::GMat in2){return in1<=in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_LE);}};
g_api_ocv_pair_mat_mat opEq = {std::string{"operator=="},
                                            [](cv::GMat in1,cv::GMat in2){return in1==in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_EQ);}};
g_api_ocv_pair_mat_mat opNotEq = {std::string{"operator!="},
                                            [](cv::GMat in1,cv::GMat in2){return in1!=in2;},
                                            [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_NE);}};

g_api_ocv_pair_mat_mat opAnd = {std::string{"operator&"},
                                        [](cv::GMat in1,cv::GMat in2){return in1&in2;},
                                        [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::bitwise_and(in1, in2, out);}};
g_api_ocv_pair_mat_mat opOr = {std::string{"operator|"},
                                        [](cv::GMat in1,cv::GMat in2){return in1|in2;},
                                        [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::bitwise_or(in1, in2, out);}};
g_api_ocv_pair_mat_mat opXor = {std::string{"operator^"},
                                        [](cv::GMat in1,cv::GMat in2){return in1^in2;},
                                        [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::bitwise_xor(in1, in2, out);}};

} // anonymous namespace
struct MathOperatorMatScalarTest : public TestParams<std::tuple<compare_f, g_api_ocv_pair_mat_scalar,int,cv::Size,int,bool,cv::GCompileArgs>>{};
struct MathOperatorMatMatTest : public TestParams<std::tuple<compare_f, g_api_ocv_pair_mat_mat,int,cv::Size,int,bool,cv::GCompileArgs>>{};
struct NotOperatorTest : public TestParams<std::tuple<int,cv::Size,bool,cv::GCompileArgs>> {};
} // opencv_test

#endif // OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP
