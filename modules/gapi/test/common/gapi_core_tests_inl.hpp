// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_TESTS_INL_HPP

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include "gapi_core_tests.hpp"

#include "gapi_core_tests_common.hpp"

namespace opencv_test
{
TEST_P(MathOpTest, MatricesAccuracyTest)
{
    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1, in2, out;
    if( testWithScalar )
    {
        cv::GScalar sc1;
        switch(opType)
        {
        case (ADD):
        {
            out = cv::gapi::addC(in1, sc1, dtype);
            cv::add(in_mat1, sc, out_mat_ocv, cv::noArray(), dtype);
            break;
        }
        case (SUB):
        {
            if( doReverseOp )
            {
                out = cv::gapi::subRC(sc1, in1, dtype);
                cv::subtract(sc, in_mat1, out_mat_ocv, cv::noArray(), dtype);
            }
            else
            {
                out = cv::gapi::subC(in1, sc1, dtype);
                cv::subtract(in_mat1, sc, out_mat_ocv, cv::noArray(), dtype);
            }
            break;
        }
        case (DIV):
        {
            if( doReverseOp )
            {
                in_mat1.setTo(1, in_mat1 == 0);  // avoiding zeros in divide input data
                out = cv::gapi::divRC(sc1, in1, scale, dtype);
                cv::divide(sc, in_mat1, out_mat_ocv, scale, dtype);
                break;
            }
            else
            {
                sc += Scalar(sc[0] == 0, sc[1] == 0, sc[2] == 0, sc[3] == 0);  // avoiding zeros in divide input data
                out = cv::gapi::divC(in1, sc1, scale, dtype);
                cv::divide(in_mat1, sc, out_mat_ocv, scale, dtype);
                break;
            }
        }
        case (MUL):
        {
            // FIXME: add `scale` parameter to mulC
            out = cv::gapi::mulC(in1, sc1, /* scale, */ dtype);
            cv::multiply(in_mat1, sc, out_mat_ocv, 1., dtype);
            break;
        }
        default:
        {
            FAIL() << "no such math operation type for scalar and matrix!";
        }
        }
        cv::GComputation c(GIn(in1, sc1), GOut(out));
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());
    }
    else
    {
        switch(opType)
        {
        case (ADD):
        {
            out = cv::gapi::add(in1, in2, dtype);
            cv::add(in_mat1, in_mat2, out_mat_ocv, cv::noArray(), dtype);
            break;
        }
        case (SUB):
        {
            out = cv::gapi::sub(in1, in2, dtype);
            cv::subtract(in_mat1, in_mat2, out_mat_ocv, cv::noArray(), dtype);
            break;
        }
        case (DIV):
        {
            in_mat2.setTo(1, in_mat2 == 0);  // avoiding zeros in divide input data
            out = cv::gapi::div(in1, in2, scale, dtype);
            cv::divide(in_mat1, in_mat2, out_mat_ocv, scale, dtype);
            break;
        }
        case (MUL):
        {
            out = cv::gapi::mul(in1, in2, scale, dtype);
            cv::multiply(in_mat1, in_mat2, out_mat_ocv, scale, dtype);
            break;
        }
        default:
        {
            FAIL() << "no such math operation type for matrix and matrix!";
        }}
        cv::GComputation c(GIn(in1, in2), GOut(out));
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
    // TODO: make threshold vs bit-exact criteria be driven by testing parameter
    #if 1
        if (CV_MAT_DEPTH(out_mat_ocv.type()) != CV_32F &&
            CV_MAT_DEPTH(out_mat_ocv.type()) != CV_64F)
        {
            // integral: allow 1% of differences, and no diffs by >1 unit
            EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF), 1);  // check: abs(a[i] - b[i]) <= 1
            float tolerance = 0.01f;
#if defined(__arm__) || defined(__aarch64__)
            if (opType == DIV)
                tolerance = 0.05f;
#endif
            EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_L1), tolerance*out_mat_ocv.total());
        }
        else
        {
            // floating-point: expect 6 decimal digits - best we expect of F32
            EXPECT_LE(cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF | NORM_RELATIVE), 1e-6);
        }
    #else
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    #endif
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(MulDoubleTest, AccuracyTest)
{
    auto& rng = cv::theRNG();
    double d = rng.uniform(0.0, 10.0);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    out = cv::gapi::mulC(in1, d, dtype);
    cv::GComputation c(in1, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::multiply(in_mat1, d, out_mat_ocv, 1, dtype);

    // Comparison ////////////////////////////////////////////////////////////
#if 1
    if (CV_MAT_DEPTH(out_mat_ocv.type()) != CV_32F &&
        CV_MAT_DEPTH(out_mat_ocv.type()) != CV_64F)
    {
        // integral: allow 1% of differences, and no diffs by >1 unit
        EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF), 1);
        EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_L1 | NORM_RELATIVE), 0.01);
    }
    else
    {
        // floating-point: expect 6 decimal digits - best we expect of F32
        EXPECT_LE(cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF | NORM_RELATIVE), 1e-6);
    }
#else
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
#endif
    EXPECT_EQ(out_mat_gapi.size(), sz);
}

TEST_P(DivTest, DISABLED_DivByZeroTest)  // https://github.com/opencv/opencv/pull/12826
{
    in_mat2 = cv::Mat(sz, type);
    in_mat2.setTo(cv::Scalar::all(0));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::div(in1, in2, 1.0, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::divide(in_mat1, in_mat2, out_mat_ocv, 1.0, dtype);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(DivCTest, DISABLED_DivByZeroTest)  // https://github.com/opencv/opencv/pull/12826
{
    sc = cv::Scalar::all(0);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar sc1;
    auto out = cv::gapi::divC(in1, sc1, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::divide(in_mat1, sc, out_mat_ocv, dtype);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        cv::Mat zeros = cv::Mat::zeros(sz, type);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, zeros, NORM_INF));
    }
}

TEST_P(MeanTest, AccuracyTest)
{
    cv::Scalar out_norm;
    cv::Scalar out_norm_ocv;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::mean(in);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_norm), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_norm_ocv = cv::mean(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(out_norm[0], out_norm_ocv[0]);
    }
}

TEST_P(MaskTest, AccuracyTest)
{
    in_mat2 = cv::Mat(sz, CV_8UC1);
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    in_mat2 = in_mat2 > 128;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in, m;
    auto out = cv::gapi::mask(in, m);

    cv::GComputation c(cv::GIn(in, m), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = cv::Mat::zeros(in_mat1.size(), in_mat1.type());
        in_mat1.copyTo(out_mat_ocv, in_mat2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(Polar2CartTest, AccuracyTest)
{
    cv::Mat out_mat2;
    cv::Mat out_mat_ocv2;
    if (dtype != -1)
    {
        out_mat2 = cv::Mat(sz, dtype);
        out_mat_ocv2 = cv::Mat(sz, dtype);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = cv::gapi::polarToCart(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out1, out2));
    c.apply(gin(in_mat1,in_mat2), gout(out_mat_gapi, out_mat2), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::polarToCart(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        // Note that we cannot rely on bit-exact sin/cos functions used for this
        // transform, so we need a threshold for verifying results vs reference.
        //
        // Relative threshold like 1e-6 is very restrictive, nearly best we can
        // expect of single-precision elementary functions implementation.
        //
        // However, good idea is making such threshold configurable: parameter
        // of this test - which a specific test instantiation could setup.
        //
        // Note that test instantiation for the OpenCV back-end could even let
        // the threshold equal to zero, as CV back-end calls the same kernel.
        //
        // TODO: Make threshold a configurable parameter of this test (ADE-221)

        ASSERT_EQ(out_mat_gapi.size(), sz);

        cv::Mat &outx = out_mat_gapi,
                &outy = out_mat2;
        cv::Mat &refx = out_mat_ocv,
                &refy = out_mat_ocv2;

        EXPECT_LE(cvtest::norm(refx, outx, NORM_L1 | NORM_RELATIVE), 1e-6);
        EXPECT_LE(cvtest::norm(refy, outy, NORM_L1 | NORM_RELATIVE), 1e-6);
    }
}

TEST_P(Cart2PolarTest, AccuracyTest)
{
    cv::Mat out_mat2(sz, dtype);
    cv::Mat out_mat_ocv2(sz, dtype);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = cv::gapi::cartToPolar(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out1, out2));
    c.apply(gin(in_mat1,in_mat2), gout(out_mat_gapi, out_mat2));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cartToPolar(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        // Note that we cannot rely on bit-exact sin/cos functions used for this
        // transform, so we need a threshold for verifying results vs reference.
        //
        // Relative threshold like 1e-6 is very restrictive, nearly best we can
        // expect of single-precision elementary functions implementation.
        //
        // However, good idea is making such threshold configurable: parameter
        // of this test - which a specific test instantiation could setup.
        //
        // Note that test instantiation for the OpenCV back-end could even let
        // the threshold equal to zero, as CV back-end calls the same kernel.
        //
        // TODO: Make threshold a configurable parameter of this test (ADE-221)

        ASSERT_EQ(out_mat_gapi.size(), sz);

        cv::Mat &outm = out_mat_gapi,
                &outa = out_mat2;
        cv::Mat &refm = out_mat_ocv,
                &refa = out_mat_ocv2;

        // FIXME: Angle result looks inaccurate at OpenCV
        //        (expected relative accuracy like 1e-6)
        EXPECT_LE(cvtest::norm(refm, outm, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(refa, outa, NORM_INF), 1e-3);
    }
}

TEST_P(CmpTest, AccuracyTest)
{
    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1, out;
    if( testWithScalar )
    {
        cv::GScalar in2;
        switch(opType)
        {
        case CMP_EQ: out = cv::gapi::cmpEQ(in1, in2); break;
        case CMP_GT: out = cv::gapi::cmpGT(in1, in2); break;
        case CMP_GE: out = cv::gapi::cmpGE(in1, in2); break;
        case CMP_LT: out = cv::gapi::cmpLT(in1, in2); break;
        case CMP_LE: out = cv::gapi::cmpLE(in1, in2); break;
        case CMP_NE: out = cv::gapi::cmpNE(in1, in2); break;
        default: FAIL() << "no such compare operation type for matrix and scalar!";
        }

        cv::compare(in_mat1, sc, out_mat_ocv, opType);

        cv::GComputation c(GIn(in1, in2), GOut(out));
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());
    }
    else
    {
        cv::GMat in2;
        switch(opType)
        {
        case CMP_EQ: out = cv::gapi::cmpEQ(in1, in2); break;
        case CMP_GT: out = cv::gapi::cmpGT(in1, in2); break;
        case CMP_GE: out = cv::gapi::cmpGE(in1, in2); break;
        case CMP_LT: out = cv::gapi::cmpLT(in1, in2); break;
        case CMP_LE: out = cv::gapi::cmpLE(in1, in2); break;
        case CMP_NE: out = cv::gapi::cmpNE(in1, in2); break;
        default: FAIL() << "no such compare operation type for two matrices!";
        }

        cv::compare(in_mat1, in_mat2, out_mat_ocv, opType);

        cv::GComputation c(GIn(in1, in2), GOut(out));
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(BitwiseTest, AccuracyTest)
{
    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1, in2, out;
    if( testWithScalar )
    {
        cv::GScalar sc1;
        switch(opType)
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
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());
    }
    else
    {
        switch(opType)
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
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());
    }


    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(NotTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::bitwise_not(in);
    cv::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::bitwise_not(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(SelectTest, AccuracyTest)
{
    cv::Mat in_mask(sz, CV_8UC1);
    cv::randu(in_mask, cv::Scalar::all(0), cv::Scalar::all(255));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3;
    auto out = cv::gapi::select(in1, in2, in3);
    cv::GComputation c(GIn(in1, in2, in3), GOut(out));

    c.apply(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat2.copyTo(out_mat_ocv);
        in_mat1.copyTo(out_mat_ocv, in_mask);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(MinTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::min(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::min(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(MaxTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::max(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::max(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(AbsDiffTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::absDiff(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::absdiff(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(AbsDiffCTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar sc1;
    auto out = cv::gapi::absDiffC(in1, sc1);
    cv::GComputation c(cv::GIn(in1, sc1), cv::GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::absdiff(in_mat1, sc, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(SumTest, AccuracyTest)
{
    cv::Scalar out_sum;
    cv::Scalar out_sum_ocv;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sum(in);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_sum), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_sum_ocv = cv::sum(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_sum, out_sum_ocv));
    }
}

#pragma push_macro("countNonZero")
#undef countNonZero
TEST_P(CountNonZeroTest, AccuracyTest)
{
    int out_cnz_gapi = -1;
    int out_cnz_ocv = -2;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::countNonZero(in);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_cnz_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_cnz_ocv = cv::countNonZero(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_cnz_gapi, out_cnz_ocv));
    }
}
#pragma pop_macro("countNonZero")

TEST_P(AddWeightedTest, AccuracyTest)
{
    auto& rng = cv::theRNG();
    double alpha = rng.uniform(0.0, 1.0);
    double beta = rng.uniform(0.0, 1.0);
    double gamma = rng.uniform(0.0, 1.0);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::addWeighted(in1, alpha, in2, beta, gamma, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::addWeighted(in_mat1, alpha, in_mat2, beta, gamma, out_mat_ocv, dtype);
    }
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);
}

TEST_P(NormTest, AccuracyTest)
{
    cv::Scalar out_norm;
    cv::Scalar out_norm_ocv;

    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1;
    cv::GScalar out;
    switch(opType)
    {
        case NORM_L1: out = cv::gapi::normL1(in1); break;
        case NORM_L2: out = cv::gapi::normL2(in1); break;
        case NORM_INF: out = cv::gapi::normInf(in1); break;
        default: FAIL() << "no such norm operation type!";
    }
    out_norm_ocv = cv::norm(in_mat1, opType);
    cv::GComputation c(GIn(in1), GOut(out));
    c.apply(gin(in_mat1), gout(out_norm), getCompileArgs());

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_norm, out_norm_ocv));
    }
}

TEST_P(IntegralTest, AccuracyTest)
{
    int type_out = (type == CV_8U) ? CV_32SC1 : CV_64FC1;
    in_mat1 = cv::Mat(sz, type);

    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Size sz_out = cv::Size(sz.width + 1, sz.height + 1);
    cv::Mat out_mat1(sz_out, type_out);
    cv::Mat out_mat_ocv1(sz_out, type_out);

    cv::Mat out_mat2(sz_out, CV_64FC1);
    cv::Mat out_mat_ocv2(sz_out, CV_64FC1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2;
    std::tie(out1, out2)  = cv::gapi::integral(in1, type_out, CV_64FC1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2));

    c.apply(cv::gin(in_mat1), cv::gout(out_mat1, out_mat2), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::integral(in_mat1, out_mat_ocv1, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv1, out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat2, NORM_INF));
    }
}

TEST_P(ThresholdTest, AccuracyTestBinary)
{
    cv::Scalar thr = initScalarRandU(50);
    cv::Scalar out_scalar;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar th1, mv1;
    out = cv::gapi::threshold(in1, th1, mv1, tt);
    cv::GComputation c(GIn(in1, th1, mv1), GOut(out));

    c.apply(gin(in_mat1, thr, maxval), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::threshold(in_mat1, out_mat_ocv, thr.val[0], maxval.val[0], tt);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_L1));
    }
}

TEST_P(ThresholdOTTest, AccuracyTestOtsu)
{
    cv::Scalar maxval = initScalarRandU(50) + cv::Scalar(50, 50, 50, 50);
    cv::Scalar out_gapi_scalar;
    double ocv_res;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar mv1, scout;
    std::tie<cv::GMat, cv::GScalar>(out, scout) = cv::gapi::threshold(in1, mv1, tt);
    cv::GComputation c(cv::GIn(in1, mv1), cv::GOut(out, scout));

    c.apply(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ocv_res = cv::threshold(in_mat1, out_mat_ocv, maxval.val[0], maxval.val[0], tt);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(ocv_res, out_gapi_scalar.val[0]);
    }
}

TEST_P(InRangeTest, AccuracyTest)
{
    cv::Scalar thrLow = initScalarRandU(100);
    cv::Scalar thrUp = initScalarRandU(100) + cv::Scalar(100, 100, 100, 100);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar th1, mv1;
    auto out = cv::gapi::inRange(in1, th1, mv1);
    cv::GComputation c(GIn(in1, th1, mv1), GOut(out));

    c.apply(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::inRange(in_mat1, thrLow, thrUp, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(Split3Test, AccuracyTest)
{
    cv::Mat out_mat2 = cv::Mat(sz, dtype);
    cv::Mat out_mat3 = cv::Mat(sz, dtype);
    cv::Mat out_mat_ocv2 = cv::Mat(sz, dtype);
    cv::Mat out_mat_ocv3 = cv::Mat(sz, dtype);
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3;
    std::tie(out1, out2, out3)  = cv::gapi::split3(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3));

    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv,  out_mat_gapi, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat2, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv3, out_mat3, NORM_INF));
    }
}

TEST_P(Split4Test, AccuracyTest)
{
    cv::Mat out_mat2 = cv::Mat(sz, dtype);
    cv::Mat out_mat3 = cv::Mat(sz, dtype);
    cv::Mat out_mat4 = cv::Mat(sz, dtype);
    cv::Mat out_mat_ocv2 = cv::Mat(sz, dtype);
    cv::Mat out_mat_ocv3 = cv::Mat(sz, dtype);
    cv::Mat out_mat_ocv4 = cv::Mat(sz, dtype);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4)  = cv::gapi::split4(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3, out4));

    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3, out_mat4), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3, out_mat_ocv4};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv,  out_mat_gapi, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat2, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv3, out_mat3, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv4, out_mat4, NORM_INF));
    }
}

static void ResizeAccuracyTest(const CompareMats& cmpF, int type, int interp, cv::Size sz_in,
    cv::Size sz_out, double fx, double fy, cv::GCompileArgs&& compile_args)
{
    cv::Mat in_mat1 (sz_in, type );
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);

    auto out_mat_sz = sz_out.area() == 0 ? cv::Size(saturate_cast<int>(sz_in.width *fx),
                                                    saturate_cast<int>(sz_in.height*fy))
                                         : sz_out;
    cv::Mat out_mat(out_mat_sz, type);
    cv::Mat out_mat_ocv(out_mat_sz, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::resize(in, sz_out, fx, fy, interp);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::resize(in_mat1, out_mat_ocv, sz_out, fx, fy, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat, out_mat_ocv));
    }
}

TEST_P(ResizeTest, AccuracyTest)
{
    ResizeAccuracyTest(cmpF, type, interp, sz, sz_out, 0.0, 0.0, getCompileArgs());
}

TEST_P(ResizeTestFxFy, AccuracyTest)
{
    ResizeAccuracyTest(cmpF, type, interp, sz, cv::Size{0, 0}, fx, fy, getCompileArgs());
}

TEST_P(ResizePTest, AccuracyTest)
{
    constexpr int planeNum = 3;
    cv::Size sz_in_p {sz.width,  sz.height*planeNum};
    cv::Size sz_out_p{sz_out.width, sz_out.height*planeNum};

    cv::Mat in_mat(sz_in_p, CV_8UC1);
    cv::randn(in_mat, cv::Scalar::all(127.0f), cv::Scalar::all(40.f));

    cv::Mat out_mat    (sz_out_p, CV_8UC1);
    cv::Mat out_mat_ocv_p(sz_out_p, CV_8UC1);

    cv::GMatP in;
    auto out = cv::gapi::resizeP(in, sz_out, interp);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    c.compile(cv::descr_of(in_mat).asPlanar(planeNum), getCompileArgs())
             (cv::gin(in_mat), cv::gout(out_mat));

    for (int i = 0; i < planeNum; i++) {
        const cv::Mat in_mat_roi = in_mat(cv::Rect(0, i*sz.height,  sz.width,  sz.height));
        cv::Mat out_mat_roi = out_mat_ocv_p(cv::Rect(0, i*sz_out.height, sz_out.width, sz_out.height));
        cv::resize(in_mat_roi, out_mat_roi, sz_out, 0, 0, interp);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat, out_mat_ocv_p));
    }
}

TEST_P(Merge3Test, AccuracyTest)
{
    cv::Mat in_mat3(sz, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat3, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3;
    auto out = cv::gapi::merge3(in1, in2, in3);

    cv::GComputation c(cv::GIn(in1, in2, in3), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2, in_mat3), cv::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3};
        cv::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(Merge4Test, AccuracyTest)
{
    cv::Mat in_mat3(sz, type);
    cv::Mat in_mat4(sz, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat3, mean, stddev);
    cv::randn(in_mat4, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3, in4;
    auto out = cv::gapi::merge4(in1, in2, in3, in4);

    cv::GComputation c(cv::GIn(in1, in2, in3, in4), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2, in_mat3, in_mat4), cv::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3, in_mat4};
        cv::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(RemapTest, AccuracyTest)
{
    cv::Mat in_map1(sz, CV_16SC2);
    cv::Mat in_map2 = cv::Mat();
    cv::randu(in_map1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Scalar bv = cv::Scalar();

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    auto out = cv::gapi::remap(in1, in_map1, in_map2, cv::INTER_NEAREST,  cv::BORDER_REPLICATE, bv);
    cv::GComputation c(in1, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::remap(in_mat1, out_mat_ocv, in_map1, in_map2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, bv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(FlipTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::flip(in, flipCode);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::flip(in_mat1, out_mat_ocv, flipCode);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(CropTest, AccuracyTest)
{
    cv::Size sz_out = cv::Size(rect_to.width, rect_to.height);
    if (dtype != -1)
    {
        out_mat_gapi = cv::Mat(sz_out, dtype);
        out_mat_ocv = cv::Mat(sz_out, dtype);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::crop(in, rect_to);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat(in_mat1, rect_to).copyTo(out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz_out);
    }
}

TEST_P(CopyTest, AccuracyTest)
{
    cv::Size sz_out = sz;
    if (dtype != -1)
    {
        out_mat_gapi = cv::Mat(sz_out, dtype);
        out_mat_ocv = cv::Mat(sz_out, dtype);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::copy(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat(in_mat1).copyTo(out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz_out);
    }
}

TEST_P(ConcatHorTest, AccuracyTest)
{
    cv::Size sz_out = sz;

    int wpart = sz_out.width / 4;
    cv::Size sz_in1 = cv::Size(wpart, sz_out.height);
    cv::Size sz_in2 = cv::Size(sz_out.width - wpart, sz_out.height);

    in_mat1 = cv::Mat(sz_in1, type );
    in_mat2 = cv::Mat(sz_in2, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::concatHor(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::hconcat(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(ConcatVertTest, AccuracyTest)
{
    cv::Size sz_out = sz;

    int hpart = sz_out.height * 2/3;
    cv::Size sz_in1 = cv::Size(sz_out.width, hpart);
    cv::Size sz_in2 = cv::Size(sz_out.width, sz_out.height - hpart);

    in_mat1 = cv::Mat(sz_in1, type);
    in_mat2 = cv::Mat(sz_in2, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::concatVert(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::vconcat(in_mat1, in_mat2, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(ConcatVertVecTest, AccuracyTest)
{
    cv::Size sz_out = sz;

    int hpart1 = sz_out.height * 2/5;
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

    cv::Mat out_mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <cv::GMat> mats(3);
    auto out = cv::gapi::concatVert(mats);

    std::vector <cv::Mat> cvmats = {in_mat1, in_mat2, in_mat3};

    cv::GComputation c({mats[0], mats[1], mats[2]}, {out});
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::vconcat(cvmats, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(ConcatHorVecTest, AccuracyTest)
{
    cv::Size sz_out = sz;

    int wpart1 = sz_out.width / 3;
    int wpart2 = sz_out.width / 4;
    cv::Size sz_in1 = cv::Size(wpart1, sz_out.height);
    cv::Size sz_in2 = cv::Size(wpart2, sz_out.height);
    cv::Size sz_in3 = cv::Size(sz_out.width - wpart1 - wpart2, sz_out.height);

    in_mat1 = cv::Mat(sz_in1, type);
    in_mat2 = cv::Mat(sz_in2, type);
    cv::Mat in_mat3 (sz_in3, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);
    cv::randn(in_mat3, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    out_mat_ocv = cv::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <cv::GMat> mats(3);
    auto out = cv::gapi::concatHor(mats);

    std::vector <cv::Mat> cvmats = {in_mat1, in_mat2, in_mat3};

    cv::GComputation c({mats[0], mats[1], mats[2]}, {out});
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::hconcat(cvmats, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(LUTTest, AccuracyTest)
{
    int type_mat = type;
    int type_lut = dtype;
    int type_out = CV_MAKETYPE(CV_MAT_DEPTH(type_lut), CV_MAT_CN(type_mat));

    initMatrixRandU(type_mat, sz, type_out);
    cv::Size sz_lut = cv::Size(1, 256);
    cv::Mat in_lut(sz_lut, type_lut);
    cv::randu(in_lut, cv::Scalar::all(0), cv::Scalar::all(255));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUT(in, in_lut);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::LUT(in_mat1, in_lut, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(ConvertToTest, AccuracyTest)
{
    int type_mat = type;
    int depth_to = dtype;
    int type_out = CV_MAKETYPE(depth_to, CV_MAT_CN(type_mat));
    initMatrixRandU(type_mat, sz, type_out);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::convertTo(in, depth_to, alpha, beta);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat1.convertTo(out_mat_ocv, depth_to, alpha, beta);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(PhaseTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in_x, in_y;
    auto out = cv::gapi::phase(in_x, in_y, angle_in_degrees);

    cv::GComputation c(in_x, in_y, out);
    c.apply(in_mat1, in_mat2, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::phase(in_mat1, in_mat2, out_mat_ocv, angle_in_degrees);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: use a comparison functor instead (after enabling OpenCL)
    {
#if defined(__aarch64__) || defined(__arm__)
        EXPECT_NEAR(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF), 4e-6);
#else
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
#endif
    }
}

TEST_P(SqrtTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sqrt(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::sqrt(in_mat1, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: use a comparison functor instead (after enabling OpenCL)
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(WarpPerspectiveTest, AccuracyTest)
{
    cv::Point center{in_mat1.size() / 2};
    cv::Mat xy = cv::getRotationMatrix2D(center, angle, scale);
    cv::Matx13d z (0, 0, 1);
    cv::Mat transform_mat;
    cv::vconcat(xy, z, transform_mat);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::warpPerspective(in, transform_mat, in_mat1.size(), flags, border_mode, border_value);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::warpPerspective(in_mat1, out_mat_ocv, cv::Mat(transform_mat), in_mat1.size(), flags, border_mode, border_value);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(WarpAffineTest, AccuracyTest)
{
    cv::Point center{in_mat1.size() / 2};
    cv::Mat warp_mat = cv::getRotationMatrix2D(center, angle, scale);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::warpAffine(in, warp_mat, in_mat1.size(), flags, border_mode, border_value);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::warpAffine(in_mat1, out_mat_ocv, warp_mat, in_mat1.size(), flags, border_mode, border_value);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(NormalizeTest, Test)
{
    initMatrixRandN(type, sz, CV_MAKETYPE(ddepth, CV_MAT_CN(type)));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::normalize(in, a, b, norm_type, ddepth);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::normalize(in_mat1, out_mat_ocv, a, b, norm_type, ddepth);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(KMeansNDTest, AccuracyTest)
{
    kmeansTestBody(in_mat1, sz, type, K, flags, getCompileArgs(), cmpF);
}

TEST_P(KMeans2DTest, AccuracyTest)
{
    const int amount = sz.height;
    std::vector<cv::Point2f> in_vector{};
    initPointsVectorRandU(amount, in_vector);
    kmeansTestBody(in_vector, sz, type, K, flags, getCompileArgs());
}

TEST_P(KMeans3DTest, AccuracyTest)
{
    const int amount = sz.height;
    std::vector<cv::Point3f> in_vector{};
    initPointsVectorRandU(amount, in_vector);
    kmeansTestBody(in_vector, sz, type, K, flags, getCompileArgs());
}

TEST_P(TransposeTest, Test)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::transpose(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::transpose(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_ocv, out_mat_gapi));
    }
}
// PLEASE DO NOT PUT NEW ACCURACY TESTS BELOW THIS POINT! //////////////////////

TEST_P(BackendOutputAllocationTest, EmptyOutput)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));

    EXPECT_TRUE(out_mat_gapi.empty());
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());
    EXPECT_FALSE(out_mat_gapi.empty());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: output is allocated to the needed size
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());
}

TEST_P(BackendOutputAllocationTest, CorrectlyPreallocatedOutput)
{
    out_mat_gapi = cv::Mat(sz, type);
    auto out_mat_gapi_ref = out_mat_gapi;  // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::add(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::add(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: output is not reallocated
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());

    EXPECT_EQ(out_mat_gapi_ref.data, out_mat_gapi.data);
}

TEST_P(BackendOutputAllocationTest, IncorrectOutputMeta)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::add(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));

    const auto run_and_compare = [&c, this] ()
    {
        auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

        // G-API code //////////////////////////////////////////////////////////////
        c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());

        // OpenCV code /////////////////////////////////////////////////////////////
        cv::add(in_mat1, in_mat2, out_mat_ocv, cv::noArray());

        // Comparison //////////////////////////////////////////////////////////////
        // Expected: size is changed, type is changed, output is reallocated
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(type, out_mat_gapi.type());

        EXPECT_NE(out_mat_gapi_ref.data, out_mat_gapi.data);
    };

    const auto chan = CV_MAT_CN(type);

    out_mat_gapi = cv::Mat(sz, CV_MAKE_TYPE(CV_64F, chan));
    run_and_compare();

    out_mat_gapi = cv::Mat(sz, CV_MAKE_TYPE(CV_MAT_DEPTH(type), chan + 1));
    run_and_compare();
}

TEST_P(BackendOutputAllocationTest, SmallerPreallocatedSize)
{
    out_mat_gapi = cv::Mat(sz / 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: size is changed, output is reallocated due to original size < curr size
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());

    EXPECT_NE(out_mat_gapi_ref.data, out_mat_gapi.data);
}

TEST_P(BackendOutputAllocationTest, SmallerPreallocatedSizeWithSubmatrix)
{
    out_mat_gapi = cv::Mat(sz / 2, type);

    cv::Mat out_mat_gapi_submat = out_mat_gapi(cv::Rect({10, 0}, sz / 5));
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);

    auto out_mat_gapi_submat_ref = out_mat_gapi_submat; // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi_submat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: submatrix is reallocated and is "detached", original matrix is unchanged
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi_submat, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi_submat.size());
    EXPECT_EQ(sz / 2, out_mat_gapi.size());

    EXPECT_NE(out_mat_gapi_submat_ref.data, out_mat_gapi_submat.data);
    EXPECT_NE(out_mat_gapi.data, out_mat_gapi_submat.datastart);
}

TEST_P(BackendOutputAllocationTest, LargerPreallocatedSize)
{
    out_mat_gapi = cv::Mat(sz * 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: size is changed, output is reallocated
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());

    EXPECT_NE(out_mat_gapi_ref.data, out_mat_gapi.data);
}

TEST_P(BackendOutputAllocationLargeSizeWithCorrectSubmatrixTest,
    LargerPreallocatedSizeWithCorrectSubmatrix)
{
    out_mat_gapi = cv::Mat(sz * 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    cv::Mat out_mat_gapi_submat = out_mat_gapi(cv::Rect({5, 8}, sz));
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);

    auto out_mat_gapi_submat_ref = out_mat_gapi_submat;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi_submat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: submatrix is not reallocated, original matrix is not reallocated
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi_submat, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi_submat.size());
    EXPECT_EQ(sz * 2, out_mat_gapi.size());

    EXPECT_EQ(out_mat_gapi_ref.data, out_mat_gapi.data);
    EXPECT_EQ(out_mat_gapi_submat_ref.data, out_mat_gapi_submat.data);
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);
}

TEST_P(BackendOutputAllocationTest, LargerPreallocatedSizeWithSmallSubmatrix)
{
    out_mat_gapi = cv::Mat(sz * 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    cv::Mat out_mat_gapi_submat = out_mat_gapi(cv::Rect({5, 8}, sz / 2));
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);

    auto out_mat_gapi_submat_ref = out_mat_gapi_submat;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::mul(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi_submat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    cv::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: submatrix is reallocated and is "detached", original matrix is unchanged
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi_submat, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi_submat.size());
    EXPECT_EQ(sz * 2, out_mat_gapi.size());

    EXPECT_EQ(out_mat_gapi_ref.data, out_mat_gapi.data);
    EXPECT_NE(out_mat_gapi_submat_ref.data, out_mat_gapi_submat.data);
    EXPECT_NE(out_mat_gapi.data, out_mat_gapi_submat.datastart);
}

TEST_P(ReInitOutTest, TestWithAdd)
{
    in_mat1 = cv::Mat(sz, type);
    in_mat2 = cv::Mat(sz, type);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(100));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out;
    out = cv::gapi::add(in1, in2, dtype);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));

    const auto run_and_compare = [&c, this] ()
    {
        // G-API code //////////////////////////////////////////////////////////////
        c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), getCompileArgs());

        // OpenCV code /////////////////////////////////////////////////////////////
        cv::add(in_mat1, in_mat2, out_mat_ocv, cv::noArray());

        // Comparison //////////////////////////////////////////////////////////////
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    };

    // run for uninitialized output
    run_and_compare();

    // run for initialized output (can be initialized with a different size)
    initOutMats(out_sz, type);
    run_and_compare();
}

TEST_P(ParseSSDBLTest, ParseTest)
{
    cv::Mat in_mat = generateSSDoutput(sz);
    std::vector<cv::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    cv::GOpaque<cv::Size> op_sz;
    auto out = cv::gapi::parseSSD(in, op_sz, confidence_threshold, filter_label);
    cv::GComputation c(cv::GIn(in, op_sz), cv::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(cv::gin(in_mat, sz), cv::gout(boxes_gapi, labels_gapi), getCompileArgs());

    // Reference code //////////////////////////////////////////////////////////
    parseSSDBLref(in_mat, sz, confidence_threshold, filter_label, boxes_ref, labels_ref);

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(boxes_gapi == boxes_ref);
    EXPECT_TRUE(labels_gapi == labels_ref);
}

TEST_P(ParseSSDTest, ParseTest)
{
    cv::Mat in_mat = generateSSDoutput(sz);
    std::vector<cv::Rect> boxes_gapi, boxes_ref;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    cv::GOpaque<cv::Size> op_sz;
    auto out = cv::gapi::parseSSD(in, op_sz, confidence_threshold,
                                  alignment_to_square, filter_out_of_bounds);
    cv::GComputation c(cv::GIn(in, op_sz), cv::GOut(out));
    c.apply(cv::gin(in_mat, sz), cv::gout(boxes_gapi), getCompileArgs());

    // Reference code //////////////////////////////////////////////////////////
    parseSSDref(in_mat, sz, confidence_threshold, alignment_to_square,
                filter_out_of_bounds, boxes_ref);

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(boxes_gapi == boxes_ref);
}

TEST_P(ParseYoloTest, ParseTest)
{
    cv::Mat in_mat = generateYoloOutput(num_classes, dims_config);
    auto anchors = cv::gapi::nn::parsers::GParseYolo::defaultAnchors();
    std::vector<cv::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    cv::GOpaque<cv::Size> op_sz;
    auto out = cv::gapi::parseYolo(in, op_sz, confidence_threshold, nms_threshold, anchors);
    cv::GComputation c(cv::GIn(in, op_sz), cv::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(cv::gin(in_mat, sz), cv::gout(boxes_gapi, labels_gapi), getCompileArgs());

    // Reference code //////////////////////////////////////////////////////////
    parseYoloRef(in_mat, sz, confidence_threshold, nms_threshold, num_classes, anchors, boxes_ref, labels_ref);

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(boxes_gapi == boxes_ref);
    EXPECT_TRUE(labels_gapi == labels_ref);
}

TEST_P(SizeTest, ParseTest)
{
    cv::GMat in;
    cv::Size out_sz;

    auto out = cv::gapi::streaming::size(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_sz), getCompileArgs());

    EXPECT_EQ(out_sz, sz);
}

TEST_P(SizeRTest, ParseTest)
{
    cv::Rect rect(cv::Point(0,0), sz);
    cv::Size out_sz;

    cv::GOpaque<cv::Rect> op_rect;
    auto out = cv::gapi::streaming::size(op_rect);
    cv::GComputation c(cv::GIn(op_rect), cv::GOut(out));
    c.apply(cv::gin(rect), cv::gout(out_sz), getCompileArgs());

    EXPECT_EQ(out_sz, sz);
}

namespace {
    class TestMediaBGR final : public cv::MediaFrame::IAdapter {
        cv::Mat m_mat;

    public:
        explicit TestMediaBGR(cv::Mat m)
            : m_mat(m) {
        }
        cv::GFrameDesc meta() const override {
            return cv::GFrameDesc{ cv::MediaFormat::BGR, cv::Size(m_mat.cols, m_mat.rows) };
        }
        cv::MediaFrame::View access(cv::MediaFrame::Access) override {
            cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
            cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
            return cv::MediaFrame::View(std::move(pp), std::move(ss));
        }
    };
};

TEST_P(SizeMFTest, ParseTest)
{
    cv::Size out_sz;
    cv::Mat bgr = cv::Mat::eye(sz.height, sz.width, CV_8UC3);
    cv::MediaFrame frame = cv::MediaFrame::Create<TestMediaBGR>(bgr);

    cv::GFrame in;
    auto out = cv::gapi::streaming::size(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(frame), cv::gout(out_sz), getCompileArgs());

    EXPECT_EQ(out_sz, sz);
}

} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_INL_HPP
