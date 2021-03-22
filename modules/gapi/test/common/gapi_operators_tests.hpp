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
enum operation
{
    ADD,  SUB,  MUL,  DIV,
    ADDR, SUBR, MULR, DIVR,
    GT,  LT,  GE,  LE,  EQ,  NE,
    GTR, LTR, GER, LER, EQR, NER,
    AND,  OR,  XOR,
    ANDR, ORR, XORR
};

// Note: namespace must match the namespace of the type of the printed object
inline std::ostream& operator<<(std::ostream& os, operation op)
{
#define CASE(v) case operation::v: os << #v; break
    switch (op)
    {
        CASE(ADD);  CASE(SUB);  CASE(MUL);  CASE(DIV);
        CASE(ADDR); CASE(SUBR); CASE(MULR); CASE(DIVR);
        CASE(GT);  CASE(LT);  CASE(GE);  CASE(LE);  CASE(EQ);  CASE(NE);
        CASE(GTR); CASE(LTR); CASE(GER); CASE(LER); CASE(EQR); CASE(NER);
        CASE(AND);  CASE(OR);  CASE(XOR);
        CASE(ANDR); CASE(ORR); CASE(XORR);
        default: GAPI_Assert(false && "unknown operation value");
    }
#undef CASE
    return os;
}

namespace
{
// declare test cases for matrix and scalar operators
auto opADD_gapi  = [](cv::GMat in,cv::GScalar c){return in + c;};
auto opADD_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::add(in, c, out);};

auto opADDR_gapi = [](cv::GMat in,cv::GScalar c){return c + in;};
auto opADDR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::add(c, in, out);};

auto opSUB_gapi  = [](cv::GMat in,cv::GScalar c){return in - c;};
auto opSUB_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::subtract(in, c, out);};

auto opSUBR_gapi = [](cv::GMat in,cv::GScalar c){return c - in;};
auto opSUBR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::subtract(c, in, out);};

auto opMUL_gapi  = [](cv::GMat in,cv::GScalar c){return in * c;};
auto opMUL_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::multiply(in, c, out);};

auto opMULR_gapi = [](cv::GMat in,cv::GScalar c){return c * in;};
auto opMULR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::multiply(c, in, out);};

auto opDIV_gapi  = [](cv::GMat in,cv::GScalar c){return in / c;};
auto opDIV_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::divide(in, c, out);};

auto opDIVR_gapi = [](cv::GMat in,cv::GScalar c){return c / in;};
auto opDIVR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::divide(c, in, out);};


auto opGT_gapi  = [](cv::GMat in,cv::GScalar c){return in > c;};
auto opGT_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_GT);};

auto opGTR_gapi = [](cv::GMat in,cv::GScalar c){return c > in;};
auto opGTR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_GT);};

auto opLT_gapi  = [](cv::GMat in,cv::GScalar c){return in < c;};
auto opLT_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_LT);};

auto opLTR_gapi = [](cv::GMat in,cv::GScalar c){return c < in;};
auto opLTR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_LT);};

auto opGE_gapi  = [](cv::GMat in,cv::GScalar c){return in >= c;};
auto opGE_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_GE);};

auto opGER_gapi = [](cv::GMat in,cv::GScalar c){return c >= in;};
auto opGER_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_GE);};

auto opLE_gapi  = [](cv::GMat in,cv::GScalar c){return in <= c;};
auto opLE_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_LE);};

auto opLER_gapi = [](cv::GMat in,cv::GScalar c){return c <= in;};
auto opLER_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_LE);};

auto opEQ_gapi  = [](cv::GMat in,cv::GScalar c){return in == c;};
auto opEQ_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_EQ);};

auto opEQR_gapi = [](cv::GMat in,cv::GScalar c){return c == in;};
auto opEQR_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_EQ);};

auto opNE_gapi  = [](cv::GMat in,cv::GScalar c){return in != c;};
auto opNE_ocv   = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(in, c, out,cv::CMP_NE);};

auto opNER_gapi = [](cv::GMat in,cv::GScalar c){return c != in;};
auto opNER_ocv  = [](const cv::Mat& in, cv::Scalar c, cv::Mat& out){cv::compare(c, in, out,cv::CMP_NE);};


auto opAND_gapi  = [](cv::GMat in,cv::GScalar c){return in & c;};
auto opAND_ocv   = [](const cv::Mat& in, const cv::Scalar& c, cv::Mat& out){cv::bitwise_and(in, c, out);};

auto opOR_gapi   = [](cv::GMat in,cv::GScalar c){return in | c;};
auto opOR_ocv    = [](const cv::Mat& in, const cv::Scalar& c, cv::Mat& out){cv::bitwise_or(in, c, out);};

auto opXOR_gapi  = [](cv::GMat in,cv::GScalar c){return in ^ c;};
auto opXOR_ocv   = [](const cv::Mat& in, const cv::Scalar& c, cv::Mat& out){cv::bitwise_xor(in, c, out);};

auto opANDR_gapi = [](cv::GMat in,cv::GScalar c){return c & in;};
auto opANDR_ocv  = [](const cv::Mat& in, const cv::Scalar& c, cv::Mat& out){cv::bitwise_and(c, in, out);};

auto opORR_gapi  = [](cv::GMat in,cv::GScalar c){return c | in;};
auto opORR_ocv   = [](const cv::Mat& in, const cv::Scalar& c, cv::Mat& out){cv::bitwise_or(c, in, out);};

auto opXORR_gapi = [](cv::GMat in,cv::GScalar c){return c ^ in;};
auto opXORR_ocv  = [](const cv::Mat& in, const cv::Scalar& c, cv::Mat& out){cv::bitwise_xor(c, in, out);};

// declare test cases for matrix and matrix operators
auto opADDM_gapi = [](cv::GMat in1,cv::GMat in2){return in1 + in2;};
auto opADDM_ocv  = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::add(in1, in2, out);};

auto opSUBM_gapi = [](cv::GMat in1,cv::GMat in2){return in1 - in2;};
auto opSUBM_ocv  = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::subtract(in1, in2, out);};

auto opDIVM_gapi = [](cv::GMat in1,cv::GMat in2){return in1 / in2;};
auto opDIVM_ocv  = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::divide(in1, in2, out);};


auto opGTM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 > in2;};
auto opGTM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_GT);};

auto opGEM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 >= in2;};
auto opGEM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_GE);};

auto opLTM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 < in2;};
auto opLTM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_LT);};

auto opLEM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 <= in2;};
auto opLEM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_LE);};

auto opEQM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 == in2;};
auto opEQM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_EQ);};

auto opNEM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 != in2;};
auto opNEM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::compare(in1, in2, out, cv::CMP_NE);};


auto opANDM_gapi = [](cv::GMat in1,cv::GMat in2){return in1 & in2;};
auto opANDM_ocv  = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::bitwise_and(in1, in2, out);};

auto opORM_gapi  = [](cv::GMat in1,cv::GMat in2){return in1 | in2;};
auto opORM_ocv   = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::bitwise_or(in1, in2, out);};

auto opXORM_gapi = [](cv::GMat in1,cv::GMat in2){return in1 ^ in2;};
auto opXORM_ocv  = [](const cv::Mat& in1, const cv::Mat& in2, cv::Mat& out){cv::bitwise_xor(in1, in2, out);};
} // anonymous namespace

struct g_api_ocv_pair_mat_scalar {
    using g_api_function_t = std::function<cv::GMat(cv::GMat,cv::GScalar)>;
    using ocv_function_t   = std::function<void(cv::Mat const&, cv::Scalar, cv::Mat&)>;

    g_api_function_t g_api_function;
    ocv_function_t   ocv_function;

    g_api_ocv_pair_mat_scalar() = default;

#define CASE(v) case operation::v: \
    g_api_function = op##v##_gapi; \
    ocv_function   = op##v##_ocv;  \
    break

    g_api_ocv_pair_mat_scalar(operation op)
    {
        switch (op)
        {
            CASE(ADD);  CASE(SUB);  CASE(MUL);  CASE(DIV);
            CASE(ADDR); CASE(SUBR); CASE(MULR); CASE(DIVR);
            CASE(GT);  CASE(LT);  CASE(GE);  CASE(LE);  CASE(EQ);  CASE(NE);
            CASE(GTR); CASE(LTR); CASE(GER); CASE(LER); CASE(EQR); CASE(NER);
            CASE(AND);  CASE(OR);  CASE(XOR);
            CASE(ANDR); CASE(ORR); CASE(XORR);
            default: GAPI_Assert(false && "unknown operation value");
        }
    }
#undef CASE
};

struct g_api_ocv_pair_mat_mat {
    using g_api_function_t = std::function<cv::GMat(cv::GMat,cv::GMat)>;
    using ocv_function_t   = std::function<void(cv::Mat const&, cv::Mat const&, cv::Mat&)>;

    g_api_function_t g_api_function;
    ocv_function_t   ocv_function;

    g_api_ocv_pair_mat_mat() = default;

#define CASE(v) case operation::v:  \
    g_api_function = op##v##M_gapi; \
    ocv_function   = op##v##M_ocv;  \
    break

    g_api_ocv_pair_mat_mat(operation op)
    {
        switch (op)
        {
            CASE(ADD);  CASE(SUB);  CASE(DIV);
            CASE(GT); CASE(LT); CASE(GE); CASE(LE); CASE(EQ); CASE(NE);
            CASE(AND); CASE(OR); CASE(XOR);
            default: GAPI_Assert(false && "unknown operation value");
        }
    }
#undef CASE
};

// Create new value-parameterized test fixture:
// MathOperatorMatScalarTest - fixture name
// initMatsRandU - function that is used to initialize input/output data
// FIXTURE_API(CompareMats, g_api_ocv_pair_mat_scalar) - test-specific parameters (types)
// 2 - number of test-specific parameters
// cmpF, op - test-spcific parameters (names)
//
// We get:
// 1. Default parameters: int type, cv::Size sz, int dtype, getCompileArgs() function
//      - available in test body
// 2. Input/output matrices will be initialized by initMatsRandU (in this fixture)
// 3. Specific parameters: cmpF, op of corresponding types
//      - created (and initialized) automatically
//      - available in test body
// Note: all parameter _values_ (e.g. type CV_8UC3) are set via INSTANTIATE_TEST_CASE_P macro
GAPI_TEST_FIXTURE(MathOperatorMatScalarTest, initMatsRandU,
    FIXTURE_API(CompareMats, operation), 2, cmpF, op)
GAPI_TEST_FIXTURE(MathOperatorMatMatTest, initMatsRandU,
    FIXTURE_API(CompareMats, operation), 2, cmpF, op)
GAPI_TEST_FIXTURE(NotOperatorTest, initMatrixRandU, <>, 0)
} // opencv_test

#endif // OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP
