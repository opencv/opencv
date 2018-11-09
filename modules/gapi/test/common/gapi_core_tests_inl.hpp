// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_CORE_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_TESTS_INL_HPP

#include "opencv2/gapi/core.hpp"
#include "gapi_core_tests.hpp"

namespace opencv_test
{

TEST_P(MathOpTest, MatricesAccuracyTest )
{
    mathOp opType = ADD;
    int type = 0, dtype = 0;
    cv::Size sz;
    double scale = 1; // mul, div
    bool testWithScalar = false, initOutMatr = false, doReverseOp = false;
    cv::GCompileArgs compile_args;
    std::tie(opType, testWithScalar, type, scale, sz, dtype, initOutMatr, doReverseOp, compile_args) = GetParam();
    initMatsRandU(type, sz, dtype, initOutMatr);

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
                in_mat1.setTo(1, in_mat1 == 0);  // avoid zeros in divide input data
                out = cv::gapi::divRC(sc1, in1, scale, dtype);
                cv::divide(sc, in_mat1, out_mat_ocv, scale, dtype);
                break;
            }
            else
            {
                sc += Scalar(1, 1, 1, 1);  // avoid zeros in divide input data
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
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
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
            in_mat2.setTo(1, in_mat2 == 0);  // avoid zeros in divide input data
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
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
    // TODO: make threshold vs bit-exact criteria be driven by testing parameter
    #if 1
        if (CV_MAT_DEPTH(out_mat_ocv.type()) != CV_32F &&
            CV_MAT_DEPTH(out_mat_ocv.type()) != CV_64F)
        {
            // integral: allow 1% of differences, and no diffs by >1 unit
            EXPECT_LE(countNonZeroPixels(cv::abs(out_mat_gapi - out_mat_ocv) > 0),
                                                           0.01*out_mat_ocv.total());
            EXPECT_LE(countNonZeroPixels(cv::abs(out_mat_gapi - out_mat_ocv) > 1), 0);
        }
        else
        {
            // floating-point: expect 6 decimal digits - best we expect of F32
            EXPECT_EQ(0, cv::countNonZero(cv::abs(out_mat_gapi - out_mat_ocv) >
                                                    1e-6*cv::abs(out_mat_ocv)));
        }
    #else
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    #endif
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(MulDoubleTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    int dtype = std::get<2>(param);
    cv::Size sz_in = std::get<1>(param);
    bool initOut = std::get<3>(param);

    auto& rng = cv::theRNG();
    double d = rng.uniform(0.0, 10.0);
    auto compile_args = std::get<4>(param);
    initMatrixRandU(type, sz_in, dtype, initOut);

    // G-API code ////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    out = cv::gapi::mulC(in1, d, dtype);
    cv::GComputation c(in1, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::multiply(in_mat1, d, out_mat_ocv, 1, dtype);

    // Comparison ////////////////////////////////////////////////////////////
#if 1
    if (CV_MAT_DEPTH(out_mat_ocv.type()) != CV_32F &&
        CV_MAT_DEPTH(out_mat_ocv.type()) != CV_64F)
    {
        // integral: allow 1% of differences, and no diffs by >1 unit
        EXPECT_LE(countNonZeroPixels(cv::abs(out_mat_gapi - out_mat_ocv) > 0),
                                                    0.01*out_mat_ocv.total());
        EXPECT_LE(countNonZeroPixels(cv::abs(out_mat_gapi - out_mat_ocv) > 1), 0);
    }
    else
    {
        // floating-point: expect 6 decimal digits - best we expect of F32
        EXPECT_EQ(0, cv::countNonZero(cv::abs(out_mat_gapi - out_mat_ocv) >
            1e-6*cv::abs(out_mat_ocv)));
    }
#else
    EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
#endif
    EXPECT_EQ(out_mat_gapi.size(), sz_in);
}

TEST_P(DivTest, DISABLED_DivByZeroTest)  // https://github.com/opencv/opencv/pull/12826
{
    int type = 0, dtype = 0;
    cv::Size sz_in;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(type, sz_in, dtype, initOut, compile_args) = GetParam();

    initMatrixRandU(type, sz_in, dtype, initOut);
    in_mat2 = cv::Mat(sz_in, type);
    in_mat2.setTo(cv::Scalar::all(0));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::div(in1, in2, 1.0, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::divide(in_mat1, in_mat2, out_mat_ocv, 1.0, dtype);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(DivCTest, DISABLED_DivByZeroTest)  // https://github.com/opencv/opencv/pull/12826
{
    int type = 0, dtype = 0;
    cv::Size sz_in;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(type, sz_in, dtype, initOut, compile_args) = GetParam();

    initMatrixRandU(type, sz_in, dtype, initOut);
    sc = cv::Scalar::all(0);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar sc1;
    auto out = cv::gapi::divC(in1, sc1, dtype);
    cv::GComputation c(GIn(in1, sc1), GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::divide(in_mat1, sc, out_mat_ocv, dtype);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        cv::Mat zeros = cv::Mat::zeros(sz_in, type);
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != zeros));
    }
}

TEST_P(MeanTest, AccuracyTest)
{
    int type = 0;
    bool initOut = false;
    cv::Size sz_in;
    cv::GCompileArgs compile_args;
    std::tie(type, sz_in, initOut, compile_args) = GetParam();
    initMatrixRandU(type, sz_in, initOut);
    cv::Scalar out_norm;
    cv::Scalar out_norm_ocv;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::mean(in);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_norm), std::move(compile_args));
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
    int type = 0;
    bool initOut = false;
    cv::Size sz_in;
    cv::GCompileArgs compile_args;
    std::tie(type, sz_in, initOut, compile_args) = GetParam();
    initMatrixRandU(type, sz_in, type, initOut);

    in_mat2 = cv::Mat(sz_in, CV_8UC1);
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    in_mat2 = in_mat2 > 128;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in, m;
    auto out = cv::gapi::mask(in, m);

    cv::GComputation c(cv::GIn(in, m), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat_gapi), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = cv::Mat::zeros(in_mat1.size(), in_mat1.type());
        in_mat1.copyTo(out_mat_ocv, in_mat2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    }
}

TEST_P(Polar2CartTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<0>(param);
    auto compile_args = std::get<2>(param);
    initMatsRandU(CV_32FC1, sz_in, CV_32FC1, std::get<1>(param));

    cv::Mat out_mat2;
    cv::Mat out_mat_ocv2;
    if(std::get<1>(param) == true)
    {
        out_mat2 = cv::Mat(sz_in, CV_32FC1);
        out_mat_ocv2 = cv::Mat(sz_in, CV_32FC1);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = cv::gapi::polarToCart(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out1, out2));
    c.apply(gin(in_mat1,in_mat2), gout(out_mat_gapi, out_mat2), std::move(compile_args));
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
        // of this test - which a specific test istantiation could setup.
        //
        // Note that test instantiation for the OpenCV back-end could even let
        // the threshold equal to zero, as CV back-end calls the same kernel.
        //
        // TODO: Make threshold a configurable parameter of this test (ADE-221)

        cv::Mat &outx = out_mat_gapi,
                &outy = out_mat2;
        cv::Mat &refx = out_mat_ocv,
                &refy = out_mat_ocv2;
        cv::Mat difx = cv::abs(refx - outx),
                dify = cv::abs(refy - outy);
        cv::Mat absx = cv::abs(refx),
                absy = cv::abs(refy);

        EXPECT_EQ(0, cv::countNonZero(difx > 1e-6*absx));
        EXPECT_EQ(0, cv::countNonZero(dify > 1e-6*absy));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(Cart2PolarTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<0>(param);
    auto compile_args = std::get<2>(param);
    initMatsRandU(CV_32FC1, sz_in, CV_32FC1, std::get<1>(param));

    cv::Mat out_mat2(sz_in, CV_32FC1);
    cv::Mat out_mat_ocv2(sz_in, CV_32FC1);

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
        // of this test - which a specific test istantiation could setup.
        //
        // Note that test instantiation for the OpenCV back-end could even let
        // the threshold equal to zero, as CV back-end calls the same kernel.
        //
        // TODO: Make threshold a configurable parameter of this test (ADE-221)

        cv::Mat &outm = out_mat_gapi,
                &outa = out_mat2;
        cv::Mat &refm = out_mat_ocv,
                &refa = out_mat_ocv2;
        cv::Mat difm = cv::abs(refm - outm),
                difa = cv::abs(refa - outa);
        cv::Mat absm = cv::abs(refm),
                absa = cv::abs(refa);

        // FIXME: Angle result looks inaccurate at OpenCV
        //        (expected relative accuracy like 1e-6)
        EXPECT_EQ(0, cv::countNonZero(difm > 1e-6*absm));
        EXPECT_EQ(0, cv::countNonZero(difa > 1e-3*absa));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(CmpTest, AccuracyTest)
{
    CmpTypes opType = CMP_EQ;
    int type = 0;
    cv::Size sz;
    bool testWithScalar = false, initOutMatr = false;
    cv::GCompileArgs compile_args;
    std::tie(opType, testWithScalar, type, sz, initOutMatr, compile_args) = GetParam();
    initMatsRandU(type, sz, CV_8U, initOutMatr);

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
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));
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
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BitwiseTest, AccuracyTest)
{
    bitwiseOp opType = AND;
    int type = 0;
    cv::Size sz;
    bool initOutMatr = false;
    cv::GCompileArgs compile_args;
    std::tie(opType, type, sz, initOutMatr, compile_args) = GetParam();
    initMatsRandU(type, sz, type, initOutMatr);

    // G-API code & corresponding OpenCV code ////////////////////////////////
    cv::GMat in1, in2, out;
    switch(opType)
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
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(NotTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatrixRandU(std::get<0>(param), sz_in, std::get<0>(param), std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::bitwise_not(in);
    cv::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::bitwise_not(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(SelectTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatsRandU(type, sz_in, type, std::get<2>(param));
    cv::Mat in_mask(sz_in, CV_8UC1);
    cv::randu(in_mask, cv::Scalar::all(0), cv::Scalar::all(255));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3;
    auto out = cv::gapi::select(in1, in2, in3);
    cv::GComputation c(GIn(in1, in2, in3), GOut(out));

    c.apply(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat2.copyTo(out_mat_ocv);
        in_mat1.copyTo(out_mat_ocv, in_mask);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(MinTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatsRandU(std::get<0>(param), sz_in, std::get<0>(param), std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::min(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::min(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(MaxTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatsRandU(std::get<0>(param), sz_in, std::get<0>(param), std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::max(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::max(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(AbsDiffTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatsRandU(std::get<0>(param), sz_in, std::get<0>(param), std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::absDiff(in1, in2);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::absdiff(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(AbsDiffCTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatsRandU(std::get<0>(param), sz_in, std::get<0>(param), std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar sc1;
    auto out = cv::gapi::absDiffC(in1, sc1);
    cv::GComputation c(cv::GIn(in1, sc1), cv::GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::absdiff(in_mat1, sc, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(SumTest, AccuracyTest)
{
    auto param = GetParam();
    cv::Size sz_in = std::get<1>(param);
    auto tolerance = std::get<3>(param);
    auto compile_args = std::get<4>(param);
    //initMatrixRandU(std::get<0>(param), sz_in, std::get<2>(param));
    initMatsRandN(std::get<0>(param), sz_in, std::get<2>(param)); //TODO: workaround trying to fix SumTest failures


    cv::Scalar out_sum;
    cv::Scalar out_sum_ocv;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sum(in);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(out_sum), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_sum_ocv = cv::sum(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(std::abs(out_sum[0] - out_sum_ocv[0]) / std::max(1.0, std::abs(out_sum_ocv[0])), tolerance)
            << "OCV=" << out_sum_ocv[0] << "   GAPI=" << out_sum[0];
    }
}

TEST_P(AddWeightedTest, AccuracyTest)
{
    int type = 0, dtype = 0;
    cv::Size sz_in;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    double tolerance = 0.0;
    std::tie(type, sz_in, dtype, initOut, tolerance, compile_args) = GetParam();

    auto& rng = cv::theRNG();
    double alpha = rng.uniform(0.0, 1.0);
    double beta = rng.uniform(0.0, 1.0);
    double gamma = rng.uniform(0.0, 1.0);
    initMatsRandU(type, sz_in, dtype, initOut);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::addWeighted(in1, alpha, in2, beta, gamma, dtype);
    cv::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::addWeighted(in_mat1, alpha, in_mat2, beta, gamma, out_mat_ocv, dtype);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        // Note, that we cannot expect bitwise results for add-weighted:
        //
        //    tmp = src1*alpha + src2*beta + gamma;
        //    dst = saturate<DST>( round(tmp) );
        //
        // Because tmp is floating-point, dst depends on compiler optimizations
        //
        // However, we must expect good accuracy of tmp, and rounding correctly

        cv::Mat failures;

        if (out_mat_ocv.type() == CV_32FC1)
        {
            // result: float - may vary in 7th decimal digit
            failures = abs(out_mat_gapi - out_mat_ocv) > abs(out_mat_ocv) * 1e-6;
        }
        else
        {
            // result: integral - rounding may vary if fractional part of tmp
            //                    is nearly 0.5

            cv::Mat inexact, incorrect, diff, tmp;

            inexact = out_mat_gapi != out_mat_ocv;

            // even if rounded differently, check if still rounded correctly
            cv::addWeighted(in_mat1, alpha, in_mat2, beta, gamma, tmp, CV_32F);
            cv::subtract(out_mat_gapi, tmp, diff, cv::noArray(), CV_32F);
            incorrect = abs(diff) >= tolerance;// 0.5000005f; // relative to 6 digits

            failures = inexact & incorrect;
        }

        EXPECT_EQ(0, cv::countNonZero(failures));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(NormTest, AccuracyTest)
{
    NormTypes opType = NORM_INF;
    int type = 0;
    cv::Size sz;
    double tolerance = 0.0;
    cv::GCompileArgs compile_args;
    std::tie(opType, type, sz, tolerance, compile_args) = GetParam();
    initMatrixRandU(type, sz, type, false);

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
    c.apply(gin(in_mat1), gout(out_norm), std::move(compile_args));

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_LE(std::abs(out_norm[0] - out_norm_ocv[0]) / std::max(1.0, std::abs(out_norm_ocv[0])), tolerance)
            << "OCV=" << out_norm_ocv[0] << "   GAPI=" << out_norm[0];
    }
}

TEST_P(IntegralTest, AccuracyTest)
{
    int type = std::get<0>(GetParam());
    cv::Size sz_in = std::get<1>(GetParam());
    auto compile_args = std::get<2>(GetParam());

    int type_out = (type == CV_8U) ? CV_32SC1 : CV_64FC1;
    cv::Mat in_mat1(sz_in, type);

    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Size sz_out = cv::Size(sz_in.width + 1, sz_in.height + 1);
    cv::Mat out_mat1(sz_out, type_out);
    cv::Mat out_mat_ocv1(sz_out, type_out);

    cv::Mat out_mat2(sz_out, CV_64FC1);
    cv::Mat out_mat_ocv2(sz_out, CV_64FC1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2;
    std::tie(out1, out2)  = cv::gapi::integral(in1, type_out, CV_64FC1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2));

    c.apply(cv::gin(in_mat1), cv::gout(out_mat1, out_mat2), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::integral(in_mat1, out_mat_ocv1, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv1 != out_mat1));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));
    }
}

TEST_P(ThresholdTest, AccuracyTestBinary)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_in = std::get<1>(param);
    int tt = std::get<2>(param);

    auto compile_args = std::get<4>(param);
    cv::Scalar thr = initScalarRandU(50);
    cv::Scalar maxval = initScalarRandU(50) + cv::Scalar(50, 50, 50, 50);
    initMatrixRandU(type, sz_in, type, std::get<3>(param));
    cv::Scalar out_scalar;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar th1, mv1;
    out = cv::gapi::threshold(in1, th1, mv1, tt);
    cv::GComputation c(GIn(in1, th1, mv1), GOut(out));

    c.apply(gin(in_mat1, thr, maxval), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::threshold(in_mat1, out_mat_ocv, thr.val[0], maxval.val[0], tt);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(out_mat_gapi.size(), sz_in);
        EXPECT_EQ(0, cv::norm(out_mat_ocv, out_mat_gapi, NORM_L1));
    }
}

TEST_P(ThresholdOTTest, AccuracyTestOtsu)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_in = std::get<1>(param);
    int tt = std::get<2>(param);

    auto compile_args = std::get<4>(param);
    cv::Scalar maxval = initScalarRandU(50) + cv::Scalar(50, 50, 50, 50);
    initMatrixRandU(type, sz_in, type, std::get<3>(param));
    cv::Scalar out_gapi_scalar;
    double ocv_res;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out;
    cv::GScalar mv1, scout;
    std::tie<cv::GMat, cv::GScalar>(out, scout) = cv::gapi::threshold(in1, mv1, tt);
    cv::GComputation c(cv::GIn(in1, mv1), cv::GOut(out, scout));

    c.apply(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ocv_res = cv::threshold(in_mat1, out_mat_ocv, maxval.val[0], maxval.val[0], tt);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
        EXPECT_EQ(ocv_res, out_gapi_scalar.val[0]);
    }
}

TEST_P(InRangeTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_in = std::get<1>(param);

    auto compile_args = std::get<3>(param);
    cv::Scalar thrLow = initScalarRandU(100);
    cv::Scalar thrUp = initScalarRandU(100) + cv::Scalar(100, 100, 100, 100);
    initMatrixRandU(type, sz_in, type, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    cv::GScalar th1, mv1;
    auto out = cv::gapi::inRange(in1, th1, mv1);
    cv::GComputation c(GIn(in1, th1, mv1), GOut(out));

    c.apply(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::inRange(in_mat1, thrLow, thrUp, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(Split3Test, AccuracyTest)
{
    cv::Size sz_in = std::get<0>(GetParam());
    auto compile_args = std::get<1>(GetParam());
    initMatrixRandU(CV_8UC3, sz_in, CV_8UC1);

    cv::Mat out_mat2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat3 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv3 = cv::Mat(sz_in, CV_8UC1);
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3;
    std::tie(out1, out2, out3)  = cv::gapi::split3(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3));

    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv  != out_mat_gapi));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv3 != out_mat3));
    }
}

TEST_P(Split4Test, AccuracyTest)
{
    cv::Size sz_in = std::get<0>(GetParam());
    auto compile_args = std::get<1>(GetParam());
    initMatrixRandU(CV_8UC4, sz_in, CV_8UC1);
    cv::Mat out_mat2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat3 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat4 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv2 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv3 = cv::Mat(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv4 = cv::Mat(sz_in, CV_8UC1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4)  = cv::gapi::split4(in1);
    cv::GComputation c(cv::GIn(in1), cv::GOut(out1, out2, out3, out4));

    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat2, out_mat3, out_mat4), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3, out_mat_ocv4};
        cv::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv  != out_mat_gapi));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv2 != out_mat2));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv3 != out_mat3));
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv4 != out_mat4));
    }
}

static void ResizeAccuracyTest(compare_f cmpF, int type, int interp, cv::Size sz_in, cv::Size sz_out, double fx, double fy, cv::GCompileArgs&& compile_args)
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
    compare_f cmpF;
    int type = 0, interp = 0;
    cv::Size sz_in, sz_out;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, interp, sz_in, sz_out, compile_args) = GetParam();
    ResizeAccuracyTest(cmpF, type, interp, sz_in, sz_out, 0.0, 0.0, std::move(compile_args));
}

TEST_P(ResizeTestFxFy, AccuracyTest)
{
    compare_f cmpF;
    int type = 0, interp = 0;
    cv::Size sz_in;
    double fx = 0.0, fy = 0.0;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, interp, sz_in, fx, fy, compile_args) = GetParam();
    ResizeAccuracyTest(cmpF, type, interp, sz_in, cv::Size{0, 0}, fx, fy, std::move(compile_args));
}

TEST_P(Merge3Test, AccuracyTest)
{
    cv::Size sz_in = std::get<0>(GetParam());
    initMatsRandU(CV_8UC1, sz_in, CV_8UC3);
    auto compile_args = std::get<1>(GetParam());
    cv::Mat in_mat3(sz_in,  CV_8UC1);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat3, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3;
    auto out = cv::gapi::merge3(in1, in2, in3);

    cv::GComputation c(cv::GIn(in1, in2, in3), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2, in_mat3), cv::gout(out_mat_gapi), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3};
        cv::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    }
}

TEST_P(Merge4Test, AccuracyTest)
{
    cv::Size sz_in = std::get<0>(GetParam());
    initMatsRandU(CV_8UC1, sz_in, CV_8UC4);
    auto compile_args = std::get<1>(GetParam());
    cv::Mat in_mat3(sz_in,  CV_8UC1);
    cv::Mat in_mat4(sz_in,  CV_8UC1);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat3, mean, stddev);
    cv::randn(in_mat4, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2, in3, in4;
    auto out = cv::gapi::merge4(in1, in2, in3, in4);

    cv::GComputation c(cv::GIn(in1, in2, in3, in4), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat2, in_mat3, in_mat4), cv::gout(out_mat_gapi), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<cv::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3, in_mat4};
        cv::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    }
}

TEST_P(RemapTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_in = std::get<1>(param);
    auto compile_args = std::get<3>(param);
    initMatrixRandU(type, sz_in, type, std::get<2>(param));
    cv::Mat in_map1(sz_in,  CV_16SC2);
    cv::Mat in_map2 = cv::Mat();
    cv::randu(in_map1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Scalar bv = cv::Scalar();

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1;
    auto out = cv::gapi::remap(in1, in_map1, in_map2, cv::INTER_NEAREST,  cv::BORDER_REPLICATE, bv);
    cv::GComputation c(in1, out);

    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::remap(in_mat1, out_mat_ocv, in_map1, in_map2, cv::INTER_NEAREST, cv::BORDER_REPLICATE, bv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(FlipTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    int flipCode =  std::get<1>(param);
    cv::Size sz_in = std::get<2>(param);
    initMatrixRandU(type, sz_in, type, false);
    auto compile_args = std::get<4>(GetParam());

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::flip(in, flipCode);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::flip(in_mat1, out_mat_ocv, flipCode);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(CropTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Rect rect_to = std::get<1>(param);
    cv::Size sz_in = std::get<2>(param);
    auto compile_args = std::get<4>(param);

    initMatrixRandU(type, sz_in, type, false);
    cv::Size sz_out = cv::Size(rect_to.width, rect_to.height);
    if( std::get<3>(param) == true )
    {
        out_mat_gapi = cv::Mat(sz_out, type);
        out_mat_ocv = cv::Mat(sz_out, type);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::crop(in, rect_to);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat(in_mat1, rect_to).copyTo(out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_out);
    }
}

TEST_P(ConcatHorTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_out = std::get<1>(param);
    auto compile_args = std::get<2>(param);

    int wpart = sz_out.width / 4;
    cv::Size sz_in1 = cv::Size(wpart, sz_out.height);
    cv::Size sz_in2 = cv::Size(sz_out.width - wpart, sz_out.height);

    cv::Mat in_mat1 (sz_in1, type );
    cv::Mat in_mat2 (sz_in2, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::concatHor(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::hconcat(in_mat1, in_mat2, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat));
    }
}

TEST_P(ConcatVertTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_out = std::get<1>(param);
    auto compile_args = std::get<2>(param);

    int hpart = sz_out.height * 2/3;
    cv::Size sz_in1 = cv::Size(sz_out.width, hpart);
    cv::Size sz_in2 = cv::Size(sz_out.width, sz_out.height - hpart);

    cv::Mat in_mat1 (sz_in1, type);
    cv::Mat in_mat2 (sz_in2, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in1, in2;
    auto out = cv::gapi::concatVert(in1, in2);

    cv::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat), std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::vconcat(in_mat1, in_mat2, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat));
    }
}

TEST_P(ConcatVertVecTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_out = std::get<1>(param);
    auto compile_args = std::get<2>(param);

    int hpart1 = sz_out.height * 2/5;
    int hpart2 = sz_out.height / 5;
    cv::Size sz_in1 = cv::Size(sz_out.width, hpart1);
    cv::Size sz_in2 = cv::Size(sz_out.width, hpart2);
    cv::Size sz_in3 = cv::Size(sz_out.width, sz_out.height - hpart1 - hpart2);

    cv::Mat in_mat1 (sz_in1, type);
    cv::Mat in_mat2 (sz_in2, type);
    cv::Mat in_mat3 (sz_in3, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);
    cv::randn(in_mat3, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <cv::GMat> mats(3);
    auto out = cv::gapi::concatVert(mats);

    std::vector <cv::Mat> cvmats = {in_mat1, in_mat2, in_mat3};

    cv::GComputation c({mats[0], mats[1], mats[2]}, {out});
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::vconcat(cvmats, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat));
    }
}

TEST_P(ConcatHorVecTest, AccuracyTest)
{
    auto param = GetParam();
    int type = std::get<0>(param);
    cv::Size sz_out = std::get<1>(param);
    auto compile_args = std::get<2>(param);

    int wpart1 = sz_out.width / 3;
    int wpart2 = sz_out.width / 4;
    cv::Size sz_in1 = cv::Size(wpart1, sz_out.height);
    cv::Size sz_in2 = cv::Size(wpart2, sz_out.height);
    cv::Size sz_in3 = cv::Size(sz_out.width - wpart1 - wpart2, sz_out.height);

    cv::Mat in_mat1 (sz_in1, type);
    cv::Mat in_mat2 (sz_in2, type);
    cv::Mat in_mat3 (sz_in3, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);

    cv::randn(in_mat1, mean, stddev);
    cv::randn(in_mat2, mean, stddev);
    cv::randn(in_mat3, mean, stddev);

    cv::Mat out_mat(sz_out, type);
    cv::Mat out_mat_ocv(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <cv::GMat> mats(3);
    auto out = cv::gapi::concatHor(mats);

    std::vector <cv::Mat> cvmats = {in_mat1, in_mat2, in_mat3};

    cv::GComputation c({mats[0], mats[1], mats[2]}, {out});
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat), std::move(compile_args));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::hconcat(cvmats, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat));
    }
}

TEST_P(LUTTest, AccuracyTest)
{
    auto param = GetParam();
    int type_mat = std::get<0>(param);
    int type_lut = std::get<1>(param);
    int type_out = CV_MAKETYPE(CV_MAT_DEPTH(type_lut), CV_MAT_CN(type_mat));
    cv::Size sz_in = std::get<2>(param);
    auto compile_args = std::get<4>(GetParam());

    initMatrixRandU(type_mat, sz_in, type_out);
    cv::Size sz_lut = cv::Size(1, 256);
    cv::Mat in_lut(sz_lut, type_lut);
    cv::randu(in_lut, cv::Scalar::all(0), cv::Scalar::all(255));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUT(in, in_lut);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::LUT(in_mat1, in_lut, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

TEST_P(ConvertToTest, AccuracyTest)
{
    auto param = GetParam();
    int type_mat = std::get<0>(param);
    int depth_to = std::get<1>(param);
    cv::Size sz_in = std::get<2>(param);
    int type_out = CV_MAKETYPE(depth_to, CV_MAT_CN(type_mat));
    initMatrixRandU(type_mat, sz_in, type_out);
    auto compile_args = std::get<3>(GetParam());

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::convertTo(in, depth_to);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat1.convertTo(out_mat_ocv, depth_to);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
        EXPECT_EQ(out_mat_gapi.size(), sz_in);
    }
}

} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_INL_HPP
