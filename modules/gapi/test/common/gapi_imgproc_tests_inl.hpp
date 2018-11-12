// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_INL_HPP

#include "opencv2/gapi/imgproc.hpp"
#include "gapi_imgproc_tests.hpp"

namespace opencv_test
{
TEST_P(Filter2DTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, borderType = 0, dtype = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, borderType, dtype, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, dtype, initOut);

    cv::Point anchor = {-1, -1};
    double delta = 0;

    cv::Mat kernel = cv::Mat(kernSize, kernSize, CV_32FC1 );
    cv::Scalar kernMean = cv::Scalar(1.0);
    cv::Scalar kernStddev = cv::Scalar(2.0/3);
    randn(kernel, kernMean, kernStddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::filter2D(in, dtype, kernel, anchor, delta, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::filter2D(in_mat1, out_mat_ocv, dtype, kernel, anchor, delta, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BoxFilterTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int filterSize = 0, borderType = 0, dtype = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, filterSize, sz, borderType, dtype, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, dtype, initOut);

    cv::Point anchor = {-1, -1};
    bool normalize = true;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::boxFilter(in, dtype, cv::Size(filterSize, filterSize), anchor, normalize, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::boxFilter(in_mat1, out_mat_ocv, dtype, cv::Size(filterSize, filterSize), anchor, normalize, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(SepFilterTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, initOut, compile_args) = GetParam();

    cv::Mat kernelX(kernSize, 1, CV_32F);
    cv::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);
    initMatsRandN(type, sz, dtype, initOut);

    cv::Point anchor = cv::Point(-1, -1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sepFilter(in, dtype, kernelX, kernelY, anchor, cv::Scalar() );

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::sepFilter2D(in_mat1, out_mat_ocv, dtype, kernelX, kernelY );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BlurTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int filterSize = 0, borderType = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, filterSize, sz, borderType, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    cv::Point anchor = {-1, -1};

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::blur(in, cv::Size(filterSize, filterSize), anchor, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::blur(in_mat1, out_mat_ocv, cv::Size(filterSize, filterSize), anchor, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(GaussianBlurTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF,type, kernSize, sz, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    cv::Size kSize = cv::Size(kernSize, kernSize);
    double sigmaX = rand();

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::gaussianBlur(in, kSize, sigmaX);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::GaussianBlur(in_mat1, out_mat_ocv, kSize, sigmaX);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(MedianBlurTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::medianBlur(in, kernSize);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::medianBlur(in_mat1, out_mat_ocv, kernSize);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(ErodeTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, kernType, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode(in, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::erode(in_mat1, out_mat_ocv, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(Erode3x3Test, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int numIters = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, initOut, numIters, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode3x3(in, numIters);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::erode(in_mat1, out_mat_ocv, kernel, cv::Point(-1, -1), numIters);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(DilateTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, kernType, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate(in, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::dilate(in_mat1, out_mat_ocv, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(Dilate3x3Test, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int numIters = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, initOut, numIters, compile_args) = GetParam();
    initMatsRandN(type, sz, type, initOut);

    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate3x3(in, numIters);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::dilate(in_mat1, out_mat_ocv, kernel, cv::Point(-1,-1), numIters);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}


TEST_P(SobelTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0, dx = 0, dy = 0;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, dx, dy, initOut, compile_args) = GetParam();
    initMatsRandN(type, sz, dtype, initOut);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Sobel(in, dtype, dx, dy, kernSize );

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Sobel(in_mat1, out_mat_ocv, dtype, dx, dy, kernSize);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(EqHistTest, AccuracyTest)
{
    compare_f cmpF;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, initOut, compile_args) = GetParam();
    initMatsRandN(CV_8UC1, sz, CV_8UC1, initOut);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::equalizeHist(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::equalizeHist(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(GetParam()));
    }
}

TEST_P(CannyTest, AccuracyTest)
{
    compare_f cmpF;
    MatType type;
    int apSize = 0;
    double thrLow = 0.0, thrUp = 0.0;
    cv::Size sz;
    bool l2gr = false, initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, thrLow, thrUp, apSize, l2gr, initOut, compile_args) = GetParam();

    initMatsRandN(type, sz, CV_8UC1, initOut);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Canny(in_mat1, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2GrayTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC1, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Gray(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(BGR2GrayTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC1, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2Gray(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(RGB2YUVTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2YUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(YUV2RGBTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));


    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2RGB(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(RGB2LabTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Lab(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2Lab);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(BGR2LUVTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2LUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2Luv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(LUV2BGRTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUV2BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_Luv2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(BGR2YUVTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2YUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(YUV2BGRTest, AccuracyTest)
{
    auto param = GetParam();
    auto compile_args = std::get<3>(param);
    compare_f cmpF = std::get<0>(param);
    initMatsRandN(CV_8UC3, std::get<1>(param), CV_8UC3, std::get<2>(param));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), std::get<1>(param));
    }
}

TEST_P(Symm7x7Test, AccuracyTest)
{
    compare_f cmpF;
    cv::Size sz;
    bool initOut = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, initOut, compile_args) = GetParam();
    initMatsRandN(CV_8UC1, sz, CV_8UC1, initOut);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::symm7x7(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    cv::Point anchor = { -1, -1 };
    double delta = 0;

    int c_int[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
    float c_float[10];
    for (int i = 0; i < 10; i++)
    {
        c_float[i] = c_int[i] / 1024.0f;
    }
    // J & I & H & G & H & I & J
    // I & F & E & D & E & F & I
    // H & E & C & B & C & E & H
    // G & D & B & A & B & D & G
    // H & E & C & B & C & E & H
    // I & F & E & D & E & F & I
    // J & I & H & G & H & I & J

    // A & B & C & D & E & F & G & H & I & J

    // 9 & 8 & 7 & 6 & 7 & 8 & 9
    // 8 & 5 & 4 & 3 & 4 & 5 & 8
    // 7 & 4 & 2 & 1 & 2 & 4 & 7
    // 6 & 3 & 1 & 0 & 1 & 3 & 6
    // 7 & 4 & 2 & 1 & 2 & 4 & 7
    // 8 & 5 & 4 & 3 & 4 & 5 & 8
    // 9 & 8 & 7 & 6 & 7 & 8 & 9

    float coefficients[49] =
    {
        c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9],
        c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
        c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
        c_float[6], c_float[3], c_float[1], c_float[0], c_float[1], c_float[3], c_float[6],
        c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
        c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
        c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9]
    };

    cv::Mat kernel = cv::Mat(7, 7, CV_32FC1);
    float* cf = kernel.ptr<float>();
    for (int i = 0; i < 49; i++)
    {
        cf[i] = coefficients[i];
    }

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::filter2D(in_mat1, out_mat_ocv, CV_8UC1, kernel, anchor, delta, BORDER_REPLICATE);
    }

    //cv::imshow("Output OCV", out_mat_ocv);
    //cv::imshow("Output GAPI", out_mat_gapi);
    //cv::Mat diff = out_mat_ocv - out_mat_gapi;
    //cv::imshow("DIFF", diff);
    //int key = cv::waitKey(0);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
