// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP


#include <iostream>

#include "gapi_imgproc_perf_tests.hpp"

namespace opencv_test
{

  using namespace perf;

  namespace
  {
      void rgb2yuyv(const uchar* rgb_line, uchar* yuv422_line, int width)
      {
          CV_Assert(width % 2 == 0);
          for (int i = 0; i < width; i += 2)
          {
              uchar r = rgb_line[i * 3    ];
              uchar g = rgb_line[i * 3 + 1];
              uchar b = rgb_line[i * 3 + 2];

              yuv422_line[i * 2    ] = cv::saturate_cast<uchar>(-0.14713 * r - 0.28886 * g + 0.436   * b + 128.f);  // U0
              yuv422_line[i * 2 + 1] = cv::saturate_cast<uchar>( 0.299   * r + 0.587   * g + 0.114   * b        );  // Y0
              yuv422_line[i * 2 + 2] = cv::saturate_cast<uchar>(0.615    * r - 0.51499 * g - 0.10001 * b + 128.f);  // V0

              r = rgb_line[i * 3 + 3];
              g = rgb_line[i * 3 + 4];
              b = rgb_line[i * 3 + 5];

              yuv422_line[i * 2 + 3] = cv::saturate_cast<uchar>(0.299 * r + 0.587   * g + 0.114   * b);   // Y1
          }
      }

      void convertRGB2YUV422Ref(const cv::Mat& in, cv::Mat &out)
      {
          out.create(in.size(), CV_8UC2);

          for (int i = 0; i < in.rows; ++i)
          {
              const uchar* in_line_p  = in.ptr<uchar>(i);
              uchar* out_line_p = out.ptr<uchar>(i);
              rgb2yuyv(in_line_p, out_line_p, in.cols);
          }
      }
  }
//------------------------------------------------------------------------------

PERF_TEST_P_(SepFilterPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, compile_args) = GetParam();

    cv::Mat kernelX(kernSize, 1, CV_32F);
    cv::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);
    initMatrixRandN(type, sz, dtype, false);

    cv::Point anchor = cv::Point(-1, -1);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::sepFilter2D(in_mat1, out_mat_ocv, dtype, kernelX, kernelY );
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sepFilter(in, dtype, kernelX, kernelY, anchor, cv::Scalar() );
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
      c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Filter2DPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, borderType = 0, dtype = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, borderType, dtype, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    cv::Point anchor = {-1, -1};
    double delta = 0;

    cv::Mat kernel = cv::Mat(kernSize, kernSize, CV_32FC1 );
    cv::Scalar kernMean = cv::Scalar::all(1.0);
    cv::Scalar kernStddev = cv::Scalar::all(2.0/3);
    randn(kernel, kernMean, kernStddev);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::filter2D(in_mat1, out_mat_ocv, dtype, kernel, anchor, delta, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::filter2D(in, dtype, kernel, anchor, delta, borderType);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }


    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BoxFilterPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int filterSize = 0, borderType = 0, dtype = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, filterSize, sz, borderType, dtype, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    cv::Point anchor = {-1, -1};
    bool normalize = true;

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::boxFilter(in_mat1, out_mat_ocv, dtype, cv::Size(filterSize, filterSize), anchor, normalize, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::boxFilter(in, dtype, cv::Size(filterSize, filterSize), anchor, normalize, borderType);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BlurPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int filterSize = 0, borderType = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, filterSize, sz, borderType, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    cv::Point anchor = {-1, -1};

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::blur(in_mat1, out_mat_ocv, cv::Size(filterSize, filterSize), anchor, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::blur(in, cv::Size(filterSize, filterSize), anchor, borderType);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(GaussianBlurPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, compile_args) = GetParam();

    cv::Size kSize = cv::Size(kernSize, kernSize);
    auto& rng = cv::theRNG();
    double sigmaX = rng();
    initMatrixRandN(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    cv::GaussianBlur(in_mat1, out_mat_ocv, kSize, sigmaX);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::gaussianBlur(in, kSize, sigmaX);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }


    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MedianBlurPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::medianBlur(in_mat1, out_mat_ocv, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::medianBlur(in, kernSize);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(ErodePerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, kernType,  compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::erode(in_mat1, out_mat_ocv, kernel);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode(in, kernel);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(Erode3x3PerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int numIters = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, numIters, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::erode(in_mat1, out_mat_ocv, kernel, cv::Point(-1, -1), numIters);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode3x3(in, numIters);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(DilatePerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, kernType, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::dilate(in_mat1, out_mat_ocv, kernel);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate(in, kernel);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(Dilate3x3PerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int numIters = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, numIters, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::dilate(in_mat1, out_mat_ocv, kernel, cv::Point(-1,-1), numIters);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate3x3(in, numIters);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(SobelPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0, dx = 0, dy = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, dx, dy, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Sobel(in_mat1, out_mat_ocv, dtype, dx, dy, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Sobel(in, dtype, dx, dy, kernSize);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SobelXYPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0, order = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, order, compile_args) = GetParam();

    cv::Mat out_mat_ocv2;
    cv::Mat out_mat_gapi2;

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Sobel(in_mat1, out_mat_ocv, dtype, order, 0, kernSize);
        cv::Sobel(in_mat1, out_mat_ocv2, dtype, 0, order, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::SobelXY(in, dtype, order, kernSize);
    cv::GComputation c(cv::GIn(in), cv::GOut(std::get<0>(out), std::get<1>(out)));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat_gapi2), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat_gapi2));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat_gapi2, out_mat_ocv2));
        EXPECT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(out_mat_gapi2.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CannyPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type;
    int apSize = 0;
    double thrLow = 0.0, thrUp = 0.0;
    cv::Size sz;
    bool l2gr = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, thrLow, thrUp, apSize, l2gr, compile_args) = GetParam();

    initMatrixRandN(type, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Canny(in_mat1, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(EqHistPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC1, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::equalizeHist(in_mat1, out_mat_ocv);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::equalizeHist(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2GrayPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2GRAY);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Gray(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2GrayPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2GRAY);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2Gray(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2YUVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2YUV(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(YUV2RGBPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2RGB(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2LabPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2Lab);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Lab(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2LUVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2Luv);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2LUV(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(LUV2BGRPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_Luv2BGR);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUV2BGR(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2YUVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV);

    cv::GMat in;
    auto out = cv::gapi::BGR2YUV(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(YUV2BGRPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR);

    cv::GMat in;
    auto out = cv::gapi::YUV2BGR(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BayerGR2RGBPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC1, sz, CV_8UC3, false);

    cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BayerGR2RGB);

    cv::GMat in;
    auto out = cv::gapi::BayerGR2RGB(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RGB2HSVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);
    cv::cvtColor(in_mat1, in_mat1, cv::COLOR_BGR2RGB);

    cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2HSV);

    cv::GMat in;
    auto out = cv::gapi::RGB2HSV(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RGB2YUV422PerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    cv::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC2, false);
    cv::cvtColor(in_mat1, in_mat1, cv::COLOR_BGR2RGB);

    convertRGB2YUV422Ref(in_mat1, out_mat_ocv);

    cv::GMat in;
    auto out = cv::gapi::RGB2YUV422(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

}
#endif //OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP
