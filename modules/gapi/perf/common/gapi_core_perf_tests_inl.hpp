// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP

#include <iostream>

#include "gapi_core_perf_tests.hpp"

#include "../../test/common/gapi_core_tests_common.hpp"

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

    // There is no need to qualify gin, gout, descr_of with namespace (cv::)
    // as they are in the same namespace as their actual argument (i.e. cv::Mat)
    // and thus are found via ADL, as in the examples below.
    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MulPerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    cv::GCompileArgs compile_args;

    std::tie(cmpF, sz, type, dtype, scale, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2, scale, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
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
    double scale = get<4>(GetParam());
    cv::GCompileArgs compile_args = get<5>(GetParam());

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);

    //This condition need to workaround bug in OpenCV.
    //It reinitializes divider matrix without zero values.
    if (dtype == CV_16S && dtype != type)
        cv::randu(in_mat2, cv::Scalar::all(1), cv::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::divide(in_mat1, in_mat2, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::div(in1, in2, scale, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
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

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::divide(sc, in_mat1, out_mat_ocv, 1.0, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar sc1;
    out = cv::gapi::divRC(sc1, in1, 1.0, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    // FIXIT unrealiable check: EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_norm));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_norm));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CmpWithScalarPerfTest, TestPerformance)
{
    MatType type    = -1;
    CmpTypes opType = CMP_EQ;
    cv::Size sz;
    compare_f cmpF;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, opType, sz, type, compile_args) = GetParam();

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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(out_mat_gapi.size(), sz);
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BitwisePerfTest, TestPerformance)
{
    MatType   type           = -1;
    bitwiseOp opType         = AND;
    bool      testWithScalar = false;
    cv::Size sz;
    cv::GCompileArgs compile_args;

    std::tie(opType, testWithScalar, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1, in2, out;
    if( testWithScalar )
    {
        cv::GScalar sc1;
        switch (opType)
        {
        case AND:
            out = cv::gapi::bitwise_and(in1, sc1);
            cv::bitwise_and(in_mat1, sc, out_mat_ocv);
            break;
        case OR:
            out = cv::gapi::bitwise_or(in1, sc1);
            cv::bitwise_or(in_mat1, sc, out_mat_ocv);
            break;
        case XOR:
            out = cv::gapi::bitwise_xor(in1, sc1);
            cv::bitwise_xor(in_mat1, sc, out_mat_ocv);
            break;
        default:
            FAIL() << "no such bitwise operation type!";
        }
        cv::GComputation c(GIn(in1, sc1), GOut(out));

        // Warm-up graph engine:
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

        TEST_CYCLE()
        {
            c.apply(gin(in_mat1, sc), gout(out_mat_gapi));
        }
    }
    else
    {
        switch (opType)
        {
        case AND:
            out = cv::gapi::bitwise_and(in1, in2);
            cv::bitwise_and(in_mat1, in_mat2, out_mat_ocv);
            break;
        case OR:
            out = cv::gapi::bitwise_or(in1, in2);
            cv::bitwise_or(in_mat1, in_mat2, out_mat_ocv);
            break;
        case XOR:
            out = cv::gapi::bitwise_xor(in1, in2);
            cv::bitwise_xor(in_mat1, in_mat2, out_mat_ocv);
            break;
        default:
            FAIL() << "no such bitwise operation type!";
        }
        cv::GComputation c(GIn(in1, in2), GOut(out));

        // Warm-up graph engine:
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

        TEST_CYCLE()
        {
            c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi));
        }
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mask)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_sum));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_sum));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_sum, out_sum_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------
#pragma push_macro("countNonZero")
#undef countNonZero
PERF_TEST_P_(CountNonZeroPerfTest, TestPerformance)
{
    compare_scalar_f cmpF;
    cv::Size sz_in;
    MatType type = -1;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz_in, type, compile_args) = GetParam();

    initMatrixRandU(type, sz_in, type, false);
    int out_cnz_gapi, out_cnz_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_cnz_ocv = cv::countNonZero(in_mat1);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::countNonZero(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_cnz_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_cnz_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_cnz_gapi, out_cnz_ocv));
    }

    SANITY_CHECK_NOTHING();
}
#pragma pop_macro("countNonZero")
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_norm));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_norm));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat1, out_mat2));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat1, out_mat2));
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
    auto cc = c.compile(descr_of(gin(in_mat1, thr, maxval)),
                        std::move(compile_args));
    cc(gin(in_mat1, thr, maxval), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, thr, maxval), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, maxval)),
                        std::move(compile_args));
    cc(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar));
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
    auto cc = c.compile(descr_of(gin(in_mat1, thrLow, thrUp)),
                        std::move(compile_args));
    cc(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3, out_mat4));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3, out_mat4));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3, in_mat4)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3, in_mat4), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3, in_mat4), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_out);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CopyPerfTest, TestPerformance)
{
    cv::Size sz_in = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandU(type, sz_in, type, false);
    cv::Size sz_out = sz_in;

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::Mat(in_mat1).copyTo(out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::copy(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConvertToPerfTest, TestPerformance)
{
    int depth_to     = -1;
    MatType type_mat = -1;
    double alpha = 0., beta = 0.;
    cv::Size sz_in;
    compare_f cmpF;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type_mat, depth_to, sz_in, alpha, beta, compile_args) = GetParam();
    MatType type_out = CV_MAKETYPE(depth_to, CV_MAT_CN(type_mat));

    initMatrixRandU(type_mat, sz_in, type_out);

    // OpenCV code ///////////////////////////////////////////////////////////
    in_mat1.convertTo(out_mat_ocv, depth_to, alpha, beta);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::convertTo(in, depth_to, alpha, beta);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz_in);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(KMeansNDPerfTest, TestPerformance)
{
    cv::Size sz;
    CompareMats cmpF;
    int K = -1;
    cv::KmeansFlags flags = cv::KMEANS_RANDOM_CENTERS;
    cv::GCompileArgs compile_args;
    std::tie(sz, cmpF, K, flags, compile_args) = GetParam();

    MatType2 type = CV_32FC1;
    initMatrixRandU(type, sz, -1, false);

    double compact_gapi = -1.;
    cv::Mat labels_gapi, centers_gapi;
    if (flags & cv::KMEANS_USE_INITIAL_LABELS)
    {
        const int amount = sz.height;
        cv::Mat bestLabels(cv::Size{1, amount}, CV_32SC1);
        cv::randu(bestLabels, 0, K);

        cv::GComputation c(kmeansTestGAPI(in_mat1, bestLabels, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(cv::gin(in_mat1, bestLabels),
                    cv::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestOpenCVCompare(in_mat1, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi, cmpF);
    }
    else
    {
        cv::GComputation c(kmeansTestGAPI(in_mat1, K, flags, std::move(compile_args), compact_gapi,
                                          labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(cv::gin(in_mat1), cv::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestValidate(sz, type, K, compact_gapi, labels_gapi, centers_gapi);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KMeans2DPerfTest, TestPerformance)
{
    int amount = -1;
    int K = -1;
    cv::KmeansFlags flags = cv::KMEANS_RANDOM_CENTERS;
    cv::GCompileArgs compile_args;
    std::tie(amount, K, flags, compile_args) = GetParam();

    std::vector<cv::Point2f> in_vector{};
    initPointsVectorRandU(amount, in_vector);

    double compact_gapi = -1.;
    std::vector<int> labels_gapi{};
    std::vector<cv::Point2f> centers_gapi{};
    if (flags & cv::KMEANS_USE_INITIAL_LABELS)
    {
        std::vector<int> bestLabels(amount);
        cv::randu(bestLabels, 0, K);

        cv::GComputation c(kmeansTestGAPI(in_vector, bestLabels, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(cv::gin(in_vector, bestLabels),
                    cv::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestOpenCVCompare(in_vector, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi);
    }
    else
    {
        cv::GComputation c(kmeansTestGAPI(in_vector, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(cv::gin(in_vector), cv::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestValidate({-1, amount}, -1, K, compact_gapi, labels_gapi, centers_gapi);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KMeans3DPerfTest, TestPerformance)
{
    int amount = -1;
    int K = -1;
    cv::KmeansFlags flags = cv::KMEANS_RANDOM_CENTERS;
    cv::GCompileArgs compile_args;
    std::tie(amount, K, flags, compile_args) = GetParam();

    std::vector<cv::Point3f> in_vector{};
    initPointsVectorRandU(amount, in_vector);

    double compact_gapi = -1.;
    std::vector<int> labels_gapi;
    std::vector<cv::Point3f> centers_gapi;
    if (flags & cv::KMEANS_USE_INITIAL_LABELS)
    {
        std::vector<int> bestLabels(amount);
        cv::randu(bestLabels, 0, K);

        cv::GComputation c(kmeansTestGAPI(in_vector, bestLabels, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(cv::gin(in_vector, bestLabels),
                    cv::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestOpenCVCompare(in_vector, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi);
    }
    else
    {
        cv::GComputation c(kmeansTestGAPI(in_vector, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(cv::gin(in_vector), cv::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestValidate({-1, amount}, -1, K, compact_gapi, labels_gapi, centers_gapi);
    }
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(TransposePerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz_in;
    MatType type = -1;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz_in, type, compile_args) = GetParam();

    initMatrixRandU(type, sz_in, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::transpose(in_mat1, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::transpose(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
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
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }
    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

// This test cases were created to control performance result of test scenario mentioned here:
// https://stackoverflow.com/questions/60629331/opencv-gapi-performance-not-good-as-expected

PERF_TEST_P_(BottleneckKernelsConstInputPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    std::string fileName = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    in_mat1 = cv::imread(findDataFile(fileName));

    cv::Mat cvvga;
    cv::Mat cvgray;
    cv::Mat cvblurred;

    cv::resize(in_mat1, cvvga, cv::Size(), 0.5, 0.5);
    cv::cvtColor(cvvga, cvgray, cv::COLOR_BGR2GRAY);
    cv::blur(cvgray, cvblurred, cv::Size(3, 3));
    cv::Canny(cvblurred, out_mat_ocv, 32, 128, 3);

    cv::GMat in;
    cv::GMat vga = cv::gapi::resize(in, cv::Size(), 0.5, 0.5, INTER_LINEAR);
    cv::GMat gray = cv::gapi::BGR2Gray(vga);
    cv::GMat blurred = cv::gapi::blur(gray, cv::Size(3, 3));
    cv::GMat out = cv::gapi::Canny(blurred, 32, 128, 3);
    cv::GComputation ac(in, out);

    auto cc = ac.compile(descr_of(gin(in_mat1)),
        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ResizeInSimpleGraphPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    cv::Size sz_in = get<2>(GetParam());
    cv::GCompileArgs compile_args = get<3>(GetParam());

    initMatsRandU(type, sz_in, type, false);

    cv::Mat add_res_ocv;

    cv::add(in_mat1, in_mat2, add_res_ocv);
    cv::resize(add_res_ocv, out_mat_ocv, cv::Size(), 0.5, 0.5);

    cv::GMat in1, in2;
    cv::GMat add_res_gapi = cv::gapi::add(in1, in2);
    cv::GMat out = cv::gapi::resize(add_res_gapi, cv::Size(), 0.5, 0.5, INTER_LINEAR);
    cv::GComputation ac(GIn(in1, in2), GOut(out));

    auto cc = ac.compile(descr_of(gin(in_mat1, in_mat2)),
                         std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ParseSSDBLPerfTest, TestPerformance)
{
    cv::Size sz;
    float confidence_threshold = 0.0f;
    int filter_label = 0;
    cv::GCompileArgs compile_args;
    std::tie(sz, confidence_threshold, filter_label, compile_args) = GetParam();
    cv::Mat in_mat = generateSSDoutput(sz);
    std::vector<cv::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // Reference code //////////////////////////////////////////////////////////
    parseSSDBLref(in_mat, sz, confidence_threshold, filter_label, boxes_ref, labels_ref);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    cv::GOpaque<cv::Size> op_sz;
    auto out = cv::gapi::parseSSD(in, op_sz, confidence_threshold, filter_label);
    cv::GComputation c(cv::GIn(in, op_sz), cv::GOut(std::get<0>(out), std::get<1>(out)));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat), descr_of(sz), std::move(compile_args));
    cc(cv::gin(in_mat, sz), cv::gout(boxes_gapi, labels_gapi));

    TEST_CYCLE()
    {
        cc(cv::gin(in_mat, sz), cv::gout(boxes_gapi, labels_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(boxes_gapi == boxes_ref);
        EXPECT_TRUE(labels_gapi == labels_ref);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ParseSSDPerfTest, TestPerformance)
{
    cv::Size sz;
    float confidence_threshold = 0;
    bool alignment_to_square = false, filter_out_of_bounds = false;
    cv::GCompileArgs compile_args;
    std::tie(sz, confidence_threshold, alignment_to_square, filter_out_of_bounds, compile_args) = GetParam();
    cv::Mat in_mat = generateSSDoutput(sz);
    std::vector<cv::Rect> boxes_gapi, boxes_ref;

    // Reference code //////////////////////////////////////////////////////////
    parseSSDref(in_mat, sz, confidence_threshold, alignment_to_square, filter_out_of_bounds, boxes_ref);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    cv::GOpaque<cv::Size> op_sz;
    auto out = cv::gapi::parseSSD(in, op_sz, confidence_threshold, alignment_to_square, filter_out_of_bounds);
    cv::GComputation c(cv::GIn(in, op_sz), cv::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat), descr_of(sz), std::move(compile_args));
    cc(cv::gin(in_mat, sz), cv::gout(boxes_gapi));

    TEST_CYCLE()
    {
        cc(cv::gin(in_mat, sz), cv::gout(boxes_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(boxes_gapi == boxes_ref);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ParseYoloPerfTest, TestPerformance)
{
    cv::Size sz;
    float confidence_threshold = 0.0f, nms_threshold = 0.0f;
    int num_classes = 0;
    cv::GCompileArgs compile_args;
    std::tie(sz, confidence_threshold, nms_threshold, num_classes, compile_args) = GetParam();
    cv::Mat in_mat = generateYoloOutput(num_classes);
    auto anchors = cv::gapi::nn::parsers::GParseYolo::defaultAnchors();
    std::vector<cv::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // Reference code //////////////////////////////////////////////////////////
    parseYoloRef(in_mat, sz, confidence_threshold, nms_threshold, num_classes, anchors, boxes_ref, labels_ref);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    cv::GOpaque<cv::Size> op_sz;
    auto out = cv::gapi::parseYolo(in, op_sz, confidence_threshold, nms_threshold, anchors);
    cv::GComputation c(cv::GIn(in, op_sz), cv::GOut(std::get<0>(out), std::get<1>(out)));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat), descr_of(sz), std::move(compile_args));
    cc(cv::gin(in_mat, sz), cv::gout(boxes_gapi, labels_gapi));

    TEST_CYCLE()
    {
        cc(cv::gin(in_mat, sz), cv::gout(boxes_gapi, labels_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(boxes_gapi == boxes_ref);
        EXPECT_TRUE(labels_gapi == labels_ref);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SizePerfTest, TestPerformance)
{
    MatType type;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(type, sz, compile_args) = GetParam();
    in_mat1 = cv::Mat(sz, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::streaming::size(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    cv::Size out_sz;

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat1), std::move(compile_args));
    cc(cv::gin(in_mat1), cv::gout(out_sz));

    TEST_CYCLE()
    {
        cc(cv::gin(in_mat1), cv::gout(out_sz));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(out_sz, sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SizeRPerfTest, TestPerformance)
{
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(sz, compile_args) = GetParam();
    cv::Rect rect(cv::Point(0,0), sz);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GOpaque<cv::Rect> op_rect;
    auto out = cv::gapi::streaming::size(op_rect);
    cv::GComputation c(cv::GIn(op_rect), cv::GOut(out));
    cv::Size out_sz;

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(rect), std::move(compile_args));
    cc(cv::gin(rect), cv::gout(out_sz));

    TEST_CYCLE()
    {
        cc(cv::gin(rect), cv::gout(out_sz));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(out_sz, sz);
    }

    SANITY_CHECK_NOTHING();
}

}
#endif // OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP
