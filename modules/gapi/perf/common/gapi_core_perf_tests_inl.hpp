// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP

#include <iostream>

#include "gapi_core_perf_tests.hpp"

namespace opencv_test
{
using namespace perf;

//------------------------------------------------------------------------------

PERF_TEST_P_(AddPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::add(in_mat1, in_mat2, out_mat_ocv, cv::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::add(in1, in2, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(AddCPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::add(in_mat1, sc, out_mat_ocv, cv::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::addC(in1, sc1, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SubPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::subtract(in_mat1, in_mat2, out_mat_ocv, cv::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::sub(in1, in2, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SubCPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::subtract(in_mat1, sc, out_mat_ocv, cv::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::subC(in1, sc1, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SubRCPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::subtract(sc, in_mat1, out_mat_ocv, cv::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::subRC(sc1, in1, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MulPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv, 1.0, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2, 1.0, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MulDoublePerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    auto& rng = cv::theRNG();
    double d = rng.uniform(0.0, 10.0);
    initMatrixRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::multiply(in_mat1, d, out_mat_ocv, 1, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    out = cv::gapi::mulC(in1, d, dtype);
    cv::GComputation c(in1, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MulCPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::multiply(in_mat1, sc, out_mat_ocv, 1, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::mulC(in1, sc1, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(DivPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    int dtype = get<3>(GetParam());
    cv::GCompileArgs compile_args = get<4>(GetParam());

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::divide(in_mat1, in_mat2, out_mat_ocv, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::div(in1, in2, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(DivCPerfTest, TestPerformance)
{
    Size sz = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int dtype = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::divide(in_mat1, sc, out_mat_ocv, 1.0, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::divC(in1, sc1, 1.0, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(DivRCPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    int dtype = get<3>(GetParam());
    cv::GCompileArgs compile_args = get<4>(GetParam());

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);

    // FIXIT Unstable input data for divide, don't process zeros
    sc += Scalar::all(1);
    in_mat1 += 1;

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::divide(sc, in_mat1, out_mat_ocv, 1.0, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::divRC(sc1, in1, 1.0, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MaskPerfTest, TestPerformance)
{
    Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandU(type, sz_in, type, false);
    in_mat2 = cv::Mat(sz_in, CV_8UC1);
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    in_mat2 = in_mat2 > 128;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_mat_ocv = cv::Mat::zeros(in_mat1.size(), in_mat1.type());
    in_mat1.copyTo(out_mat_ocv, in_mat2);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in, m;
    auto out = cv::gapi::mask(in, m);
    cv::GComputation c(cv::GIn(in, m), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MeanPerfTest, TestPerformance)
{
    Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandU(type, sz_in, false);
    cv::Scalar out_norm;
    cv::Scalar out_norm_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_norm_ocv = cv::mean(in_mat1);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::mean(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_norm), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_norm), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(out_norm[0], out_norm_ocv[0]);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Polar2CartPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz_in = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatsRandU(CV_32FC1, sz_in, CV_32FC1, false);
    cv::Mat out_mat2;
    cv::Mat out_mat_ocv2;

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::polarToCart(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = cv::gapi::polarToCart(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out1, out2));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2), std::move(compile_args));
    }
    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_TRUE(cmpF(out_mat_ocv2, out_mat2));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Cart2PolarPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz_in = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatsRandU(CV_32FC1, sz_in, CV_32FC1, false);
    cv::Mat out_mat2(sz_in, CV_32FC1);
    cv::Mat out_mat_ocv2(sz_in, CV_32FC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::cartToPolar(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = cv::gapi::cartToPolar(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out1, out2));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_TRUE(cmpF(out_mat_ocv2, out_mat2));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CmpPerfTest, TestPerformance)
{
    CmpTypes opType = get<0>(GetParam());
    cv::Size sz = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, CV_8U, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::compare(in_mat1, in_mat2, out_mat_ocv, opType);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    switch (opType)
    {
    case CMP_EQ: out = cv::gapi::cmpEQ(in1, in2); break;
    case CMP_GT: out = cv::gapi::cmpGT(in1, in2); break;
    case CMP_GE: out = cv::gapi::cmpGE(in1, in2); break;
    case CMP_LT: out = cv::gapi::cmpLT(in1, in2); break;
    case CMP_LE: out = cv::gapi::cmpLE(in1, in2); break;
    case CMP_NE: out = cv::gapi::cmpNE(in1, in2); break;
    default: FAIL() << "no such compare operation type for two matrices!";
    }
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CmpWithScalarPerfTest, TestPerformance)
{
    CmpTypes opType = get<0>(GetParam());
    cv::Size sz = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, CV_8U, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::compare(in_mat1, sc, out_mat_ocv, opType);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar in2;
    switch (opType)
    {
    case CMP_EQ: out = cv::gapi::cmpEQ(in1, in2); break;
    case CMP_GT: out = cv::gapi::cmpGT(in1, in2); break;
    case CMP_GE: out = cv::gapi::cmpGE(in1, in2); break;
    case CMP_LT: out = cv::gapi::cmpLT(in1, in2); break;
    case CMP_LE: out = cv::gapi::cmpLE(in1, in2); break;
    case CMP_NE: out = cv::gapi::cmpNE(in1, in2); break;
    default: FAIL() << "no such compare operation type for matrix and scalar!";
    }
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BitwisePerfTest, TestPerformance)
{
    bitwiseOp opType = get<0>(GetParam());
    cv::Size sz = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz, type, false);

    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1, in2, out;
    switch (opType)
    {
    case AND:
    {
        out = cv::gapi::bitwise_and(in1, in2);
        cv::bitwise_and(in_mat1, in_mat2, out_mat_ocv);
        break;
    }
    case OR:
    {
        out = cv::gapi::bitwise_or(in1, in2);
        cv::bitwise_or(in_mat1, in_mat2, out_mat_ocv);
        break;
    }
    case XOR:
    {
        out = cv::gapi::bitwise_xor(in1, in2);
        cv::bitwise_xor(in_mat1, in_mat2, out_mat_ocv);
        break;
    }
    default:
    {
        FAIL() << "no such bitwise operation type!";
    }
    }
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BitwiseNotPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::bitwise_not(in_mat1, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in, out;
    out = cv::gapi::bitwise_not(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SelectPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatsRandU(type, sz_in, type, false);
    cv::Mat in_mask(sz_in, CV_8UC1);
    cv::randu(in_mask, cv::Scalar::all(0), cv::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    in_mat2.copyTo(out_mat_ocv);
    in_mat1.copyTo(out_mat_ocv, in_mask);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3, out;
    out = cv::gapi::select(in1, in2, in3);
    cv::GComputation c(GIn(in1, in2, in3), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MinPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());


    initMatsRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::min(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::min(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MaxPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());


    initMatsRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::max(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::max(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(AbsDiffPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());


    initMatsRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::absdiff(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::absDiff(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(AbsDiffCPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());


    initMatsRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::absdiff(in_mat1, sc, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::absDiffC(in1, sc1);
    cv::GComputation c(cv::GIn(in1, sc1), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SumPerfTest, TestPerformance)
{
    compare_scalar_f cmpF = get<0>(GetParam());
    cv::Size sz_in = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());


    initMatrixRandU(type, sz_in, type, false);
    cv::Scalar out_sum;
    cv::Scalar out_sum_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_sum_ocv = cv::sum(in_mat1);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sum(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_sum), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_sum), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_sum, out_sum_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(AddWeightedPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    cv::Size sz_in = get<1>(GetParam());
    MatType type = get<2>(GetParam());
    int dtype = get<3>(GetParam());
    cv::GCompileArgs compile_args = get<4>(GetParam());

    auto& rng = cv::theRNG();
    double alpha = rng.uniform(0.0, 1.0);
    double beta = rng.uniform(0.0, 1.0);
    double gamma = rng.uniform(0.0, 1.0);
    initMatsRandU(type, sz_in, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::addWeighted(in_mat1, alpha, in_mat2, beta, gamma, out_mat_ocv, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::addWeighted(in1, alpha, in2, beta, gamma, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);


    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(NormPerfTest, TestPerformance)
{
    compare_scalar_f cmpF = get<0>(GetParam());
    NormTypes opType = get<1>(GetParam());
    cv::Size sz = get<2>(GetParam());
    MatType type = get<3>(GetParam());
    cv::GCompileArgs compile_args = get<4>(GetParam());


    initMatrixRandU(type, sz, type, false);
    cv::Scalar out_norm;
    cv::Scalar out_norm_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_norm_ocv = cv::norm(in_mat1, opType);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar out;
    switch (opType)
    {
    case NORM_L1: out = cv::gapi::normL1(in1); break;
    case NORM_L2: out = cv::gapi::normL2(in1); break;
    case NORM_INF: out = cv::gapi::normInf(in1); break;
    default: FAIL() << "no such norm operation type!";
    }
    cv::GComputation c(GIn(in1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1), gout(out_norm), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1), gout(out_norm), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_norm, out_norm_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(IntegralPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());


    MatType type_out = (type == CV_8U) ? CV_32SC1 : CV_64FC1;


    in_mat1 = cv::Mat(sz_in, type);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Size sz_out = cv::Size(sz_in.width + 1, sz_in.height + 1);
    cv::Mat out_mat1(sz_out, type_out);
    cv::Mat out_mat_ocv1(sz_out, type_out);

    cv::Mat out_mat2(sz_out, CV_64FC1);
    cv::Mat out_mat_ocv2(sz_out, CV_64FC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::integral(in_mat1, out_mat_ocv1, out_mat_ocv2);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2;
    std::tie(out1, out2) = cv::gapi::integral(in1, type_out, CV_64FC1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_mat1, out_mat2), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_mat1, out_mat2), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_ocv1 != out_mat1));
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ThresholdPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int tt = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    cv::Scalar thr = initScalarRandU(50);
    cv::Scalar maxval = initScalarRandU(50) + cv::Scalar(50, 50, 50, 50);
    initMatrixRandU(type, sz_in, type, false);
    cv::Scalar out_scalar;

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::threshold(in_mat1, out_mat_ocv, thr.val[0], maxval.val[0], tt);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar th1, mv1;
    out = cv::gapi::threshold(in1, th1, mv1, tt);
    cv::GComputation c(GIn(in1, th1, mv1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, thr, maxval), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, thr, maxval), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ThresholdOTPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int tt = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    cv::Scalar maxval = initScalarRandU(50) + cv::Scalar(50, 50, 50, 50);
    initMatrixRandU(type, sz_in, type, false);
    cv::Scalar out_gapi_scalar;
    double ocv_res;

    // OpenCV code ///////////////////////////////////////////////////////////
    ocv_res = cv::threshold(in_mat1, out_mat_ocv, maxval.val[0], maxval.val[0], tt);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar mv1, scout;
    std::tie<cv::GMat, cv::GScalar>(out, scout) = cv::gapi::threshold(in1, mv1, tt);
    cv::GComputation c(cv::GIn(in1, mv1), cv::GOut(out, scout));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);
    EXPECT_EQ(ocv_res, out_gapi_scalar.val[0]);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(InRangePerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    cv::Scalar thrLow = initScalarRandU(100);
    cv::Scalar thrUp = initScalarRandU(100) + cv::Scalar(100, 100, 100, 100);
    initMatrixRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::inRange(in_mat1, thrLow, thrUp, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar th1, mv1;
    auto out = cv::gapi::inRange(in1, th1, mv1);
    cv::GComputation c(GIn(in1, th1, mv1), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Split3PerfTest, TestPerformance)
{
    Size sz_in = get<0>(GetParam());
    cv::GCompileArgs compile_args = get<1>(GetParam());


    initMatrixRandU(CV_8UC3, sz_in, CV_8UC1);
    cv::Mat out_mat2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat3 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv3 = cv::Mat(sz_in, CV_8UC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<cv::Mat> out_mats_ocv = { out_mat_ocv, out_mat_ocv2, out_mat_ocv3 };
    cv::split(in_mat1, out_mats_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3;
    std::tie(out1, out2, out3) = cv::gapi::split3(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(0, cv::norm(out_mat_ocv2, out_mat2, NORM_INF));
    EXPECT_EQ(0, cv::norm(out_mat_ocv3, out_mat3, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Split4PerfTest, TestPerformance)
{
    Size sz_in = get<0>(GetParam());
    cv::GCompileArgs compile_args = get<1>(GetParam());

    initMatrixRandU(CV_8UC4, sz_in, CV_8UC1);
    cv::Mat out_mat2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat3 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat4 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv3 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv4 = cv::Mat(sz_in, CV_8UC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<cv::Mat> out_mats_ocv = { out_mat_ocv, out_mat_ocv2, out_mat_ocv3, out_mat_ocv4 };
    cv::split(in_mat1, out_mats_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4) = cv::gapi::split4(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3, out4));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3, out_mat4), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3, out_mat4), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(0, cv::norm(out_mat_ocv2, out_mat2, NORM_INF));
    EXPECT_EQ(0, cv::norm(out_mat_ocv3, out_mat3, NORM_INF));
    EXPECT_EQ(0, cv::norm(out_mat_ocv4, out_mat4, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Merge3PerfTest, TestPerformance)
{
    Size sz_in = get<0>(GetParam());
    cv::GCompileArgs compile_args = get<1>(GetParam());

    initMatsRandU(CV_8UC1, sz_in, CV_8UC3);
    cv::Mat in_mat3(sz_in, CV_8UC1);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);
    cv::randn(in_mat3, mean, stddev);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<cv::Mat> in_mats_ocv = { in_mat1, in_mat2, in_mat3 };
    cv::merge(in_mats_ocv, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3;
    auto out = cv::gapi::merge3(in1, in2, in3);
    cv::GComputation c(cv::GIn(in1, in2, in3), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1, in_mat2, in_mat3), cv::gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1, in_mat2, in_mat3), cv::gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Merge4PerfTest, TestPerformance)
{
    Size sz_in = get<0>(GetParam());
    cv::GCompileArgs compile_args = get<1>(GetParam());

    initMatsRandU(CV_8UC1, sz_in, CV_8UC3);
    cv::Mat in_mat3(sz_in, CV_8UC1);
    cv::Mat in_mat4(sz_in, CV_8UC1);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);
    cv::randn(in_mat3, mean, stddev);
    cv::randn(in_mat4, mean, stddev);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<cv::Mat> in_mats_ocv = { in_mat1, in_mat2, in_mat3, in_mat4 };
    cv::merge(in_mats_ocv, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3, in4;
    auto out = cv::gapi::merge4(in1, in2, in3, in4);
    cv::GComputation c(cv::GIn(in1, in2, in3, in4), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1, in_mat2, in_mat3, in_mat4), cv::gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1, in_mat2, in_mat3, in_mat4), cv::gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(RemapPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandU(type, sz_in, type, false);
    cv::Mat in_map1(sz_in, CV_16SC2);
    cv::Mat in_map2 = cv::Mat();
    cv::randu(in_map1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Scalar bv = cv::Scalar();

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::remap(in_mat1, out_mat_ocv, in_map1, in_map2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, bv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1;
    auto out = cv::gapi::remap(in1, in_map1, in_map2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, bv);
    cv::GComputation c(in1, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(FlipPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int flipCode = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatrixRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::flip(in_mat1, out_mat_ocv, flipCode);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::flip(in, flipCode);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CropPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::Rect rect_to = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatrixRandU(type, sz_in, type, false);
    cv::Size sz_out = cv::Size(rect_to.width, rect_to.height);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::Mat(in_mat1, rect_to).copyTo(out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::crop(in, rect_to);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_out);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConcatHorPerfTest, TestPerformance)
{
    cv::Size sz_out = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    int wpart = sz_out.width / 4;

    cv::Size sz_in1 = cv::Size(wpart, sz_out.height);
    cv::Size sz_in2 = cv::Size(sz_out.width - wpart, sz_out.height);

    in_mat1 = cv::Mat(sz_in1, type);
    in_mat2 = cv::Mat(sz_in2, type);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    out_mat_gapi = cv::Mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::hconcat(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::concatHor(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConcatHorVecPerfTest, TestPerformance)
{
    cv::Size sz_out = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    int wpart1 = sz_out.width / 3;
    int wpart2 = sz_out.width / 2;

    cv::Size sz_in1 = cv::Size(wpart1, sz_out.height);
    cv::Size sz_in2 = cv::Size(wpart2, sz_out.height);
    cv::Size sz_in3 = cv::Size(sz_out.width - wpart1 - wpart2, sz_out.height);

    in_mat1 = cv::Mat(sz_in1, type);
    in_mat2 = cv::Mat(sz_in2, type);
    cv::Mat in_mat3(sz_in3, type);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);
    cv::randn(in_mat3, mean, stddev);

    out_mat_gapi = cv::Mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    std::vector <cv::Mat> cvmats = { in_mat1, in_mat2, in_mat3 };

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::hconcat(cvmats, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <cv::GMat> mats(3);
    auto out = cv::gapi::concatHor(mats);
    cv::GComputation c({ mats[0], mats[1], mats[2] }, { out });

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConcatVertPerfTest, TestPerformance)
{
    cv::Size sz_out = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    int hpart = sz_out.height * 2 / 3;

    cv::Size sz_in1 = cv::Size(sz_out.width, hpart);
    cv::Size sz_in2 = cv::Size(sz_out.width, sz_out.height - hpart);

    in_mat1 = cv::Mat(sz_in1, type);
    in_mat2 = cv::Mat(sz_in2, type);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    out_mat_gapi = cv::Mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::vconcat(in_mat1, in_mat2, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::concatVert(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConcatVertVecPerfTest, TestPerformance)
{
    cv::Size sz_out = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    int hpart1 = sz_out.height * 2 / 5;
    int hpart2 = sz_out.height / 5;

    cv::Size sz_in1 = cv::Size(sz_out.width, hpart1);
    cv::Size sz_in2 = cv::Size(sz_out.width, hpart2);
    cv::Size sz_in3 = cv::Size(sz_out.width, sz_out.height - hpart1 - hpart2);

    in_mat1 = cv::Mat(sz_in1, type);
    in_mat2 = cv::Mat(sz_in2, type);
    cv::Mat in_mat3(sz_in3, type);

    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);
    cv::randn(in_mat3, mean, stddev);

    out_mat_gapi = cv::Mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    std::vector <cv::Mat> cvmats = { in_mat1, in_mat2, in_mat3 };

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::vconcat(cvmats, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <cv::GMat> mats(3);
    auto out = cv::gapi::concatVert(mats);
    cv::GComputation c({ mats[0], mats[1], mats[2] }, { out });

    // Warm-up graph engine:
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(LUTPerfTest, TestPerformance)
{
    MatType type_mat = get<0>(GetParam());
    MatType type_lut = get<1>(GetParam());
    MatType type_out = CV_MAKETYPE(CV_MAT_DEPTH(type_lut), CV_MAT_CN(type_mat));
    cv::Size sz_in = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatrixRandU(type_mat, sz_in, type_out);
    cv::Size sz_lut = cv::Size(1, 256);
    cv::Mat in_lut(sz_lut, type_lut);
    cv::randu(in_lut, cv::Scalar::all(0), cv::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::LUT(in_mat1, in_lut, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUT(in, in_lut);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConvertToPerfTest, TestPerformance)
{
    MatType type_mat = get<0>(GetParam());
    int depth_to = get<1>(GetParam());
    cv::Size sz_in = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());
    MatType type_out = CV_MAKETYPE(depth_to, CV_MAT_CN(type_mat));

    initMatrixRandU(type_mat, sz_in, type_out);

    // OpenCV code ///////////////////////////////////////////////////////////
    in_mat1.convertTo(out_mat_ocv, depth_to);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::convertTo(in, depth_to);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ResizePerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int interp = get<2>(GetParam());
    cv::Size sz_in = get<3>(GetParam());
    cv::Size sz_out = get<4>(GetParam());
    cv::GCompileArgs compile_args = get<5>(GetParam());

    in_mat1 = cv::Mat(sz_in, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);
    cv::randn(in_mat1, mean, stddev);
    out_mat_gapi = cv::Mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::resize(in_mat1, out_mat_ocv, sz_out, 0.0, 0.0, interp);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::resize(in, sz_out, 0.0, 0.0, interp);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ResizeFxFyPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    int interp = get<2>(GetParam());
    cv::Size sz_in = get<3>(GetParam());
    double fx = get<4>(GetParam());
    double fy = get<5>(GetParam());
    cv::GCompileArgs compile_args = get<6>(GetParam());

    in_mat1 = cv::Mat(sz_in, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);
    cv::randn(in_mat1, mean, stddev);
    cv::Size sz_out = cv::Size(saturate_cast<int>(sz_in.width *fx), saturate_cast<int>(sz_in.height*fy));
    out_mat_gapi = cv::Mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::resize(in_mat1, out_mat_ocv, sz_out, fx, fy, interp);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::resize(in, sz_out, fx, fy, interp);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

}
#endif // OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP
