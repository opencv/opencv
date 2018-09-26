// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../../test/common/gapi_tests_common.hpp"
#include "opencv2/gapi/imgproc.hpp"

namespace opencv_test
{

  using namespace perf;

//------------------------------------------------------------------------------

class SepFilterPerfTest : public TestPerfParams<tuple<MatType, int, cv::Size, int>> {};
PERF_TEST_P_(SepFilterPerfTest, TestPerformance)
{
    MatType type = 0;
    int kernSize = 0, dtype = 0;
    cv::Size sz;
    std::tie(type, kernSize, sz, dtype) = GetParam();

    cv::Mat kernelX(kernSize, 1, CV_32F);
    cv::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);
    initMatsRandN(type, sz, dtype, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
      c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

class Filter2DPerfTest : public TestPerfParams<tuple<MatType,int,cv::Size,int,int>> {};
PERF_TEST_P_(Filter2DPerfTest, TestPerformance)
{
    MatType type = 0;
    int kernSize = 0, borderType = 0, dtype = 0;
    cv::Size sz;
    std::tie(type, kernSize, sz, borderType, dtype) = GetParam();

    initMatsRandN(type, sz, dtype, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class BoxFilterPerfTest : public TestPerfParams<tuple<MatType,int,cv::Size,int,int,double>> {};
PERF_TEST_P_(BoxFilterPerfTest, TestPerformance)
{
    MatType type = 0;
    int filterSize = 0, borderType = 0, dtype = 0;
    cv::Size sz;
    double tolerance = 0.0;
    std::tie(type, filterSize, sz, borderType, dtype, tolerance) = GetParam();

    initMatsRandN(type, sz, dtype, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        cv::Mat absDiff;
        cv::absdiff(out_mat_gapi, out_mat_ocv, absDiff);
        EXPECT_EQ(0, cv::countNonZero(absDiff > tolerance));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class BlurPerfTest : public TestPerfParams<tuple<MatType,int,cv::Size,int,double>> {};
PERF_TEST_P_(BlurPerfTest, TestPerformance)
{
    MatType type = 0;
    int filterSize = 0, borderType = 0;
    cv::Size sz;
    double tolerance = 0.0;
    std::tie(type, filterSize, sz, borderType, tolerance) = GetParam();

    initMatsRandN(type, sz, type, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        cv::Mat absDiff;
        cv::absdiff(out_mat_gapi, out_mat_ocv, absDiff);
        EXPECT_EQ(0, cv::countNonZero(absDiff > tolerance));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class GaussianBlurPerfTest : public TestPerfParams<tuple<MatType, int, cv::Size>> {};
PERF_TEST_P_(GaussianBlurPerfTest, TestPerformance)
{
  MatType type = 0;
  int kernSize = 0;
  cv::Size sz;
  std::tie(type, kernSize, sz) = GetParam();

  cv::Size kSize = cv::Size(kernSize, kernSize);
  auto& rng = cv::theRNG();
  double sigmaX = rng();
  initMatsRandN(type, sz, type, false);

  // OpenCV code ///////////////////////////////////////////////////////////
  cv::GaussianBlur(in_mat1, out_mat_ocv, kSize, sigmaX);

  // G-API code //////////////////////////////////////////////////////////////
  cv::GMat in;
  auto out = cv::gapi::gaussianBlur(in, kSize, sigmaX);
  cv::GComputation c(in, out);

  // Warm-up graph engine:
  c.apply(in_mat1, out_mat_gapi);

  TEST_CYCLE()
  {
    c.apply(in_mat1, out_mat_gapi);
  }

  // Comparison ////////////////////////////////////////////////////////////
  EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
  EXPECT_EQ(out_mat_gapi.size(), sz);

  SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

class MedianBlurPerfTest : public TestPerfParams<tuple<MatType,int,cv::Size>> {};
PERF_TEST_P_(MedianBlurPerfTest, TestPerformance)
{
    MatType type = 0;
    int kernSize = 0;
    cv::Size sz;
    std::tie(type, kernSize, sz) = GetParam();

    initMatsRandN(type, sz, type, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::medianBlur(in_mat1, out_mat_ocv, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::medianBlur(in, kernSize);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class ErodePerfTest : public TestPerfParams<tuple<MatType,int,cv::Size,int>> {};
PERF_TEST_P_(ErodePerfTest, TestPerformance)
{
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    cv::Size sz;
    std::tie(type, kernSize, sz, kernType) = GetParam();

    initMatsRandN(type, sz, type, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class Erode3x3PerfTest : public TestPerfParams<tuple<MatType,cv::Size,int>> {};
PERF_TEST_P_(Erode3x3PerfTest, TestPerformance)
{
    MatType type = 0;
    int numIters = 0;
    cv::Size sz;
    std::tie(type, sz, numIters) = GetParam();

    initMatsRandN(type, sz, type, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class DilatePerfTest : public TestPerfParams<tuple<MatType,int,cv::Size,int>> {};
PERF_TEST_P_(DilatePerfTest, TestPerformance)
{
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    cv::Size sz;
    std::tie(type, kernSize, sz, kernType) = GetParam();

    initMatsRandN(type, sz, type, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class Dilate3x3PerfTest : public TestPerfParams<tuple<MatType,cv::Size,int>> {};
PERF_TEST_P_(Dilate3x3PerfTest, TestPerformance)
{
    MatType type = 0;
    int numIters = 0;
    cv::Size sz;
    std::tie(type, sz, numIters) = GetParam();

    initMatsRandN(type, sz, type, false);

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
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class SobelPerfTest : public TestPerfParams<tuple<MatType,int,cv::Size,int,int,int>> {};
PERF_TEST_P_(SobelPerfTest, TestPerformance)
{
    MatType type = 0;
    int kernSize = 0, dtype = 0, dx = 0, dy = 0;
    cv::Size sz;
    std::tie(type, kernSize, sz, dtype, dx, dy) = GetParam();

    initMatsRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Sobel(in_mat1, out_mat_ocv, dtype, dx, dy, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sobel(in, dtype, dx, dy, kernSize );
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class CannyPerfTest : public TestPerfParams<tuple<MatType,cv::Size,double,double,int,bool>> {};
PERF_TEST_P_(CannyPerfTest, TestPerformance)
{
    MatType type;
    int apSize = 0;
    double thrLow = 0.0, thrUp = 0.0;
    cv::Size sz;
    bool l2gr = false;
    std::tie(type, sz, thrLow, thrUp, apSize, l2gr) = GetParam();

    initMatsRandN(type, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Canny(in_mat1, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class EqHistPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(EqHistPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC1, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::equalizeHist(in_mat1, out_mat_ocv);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::equalizeHist(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class RGB2GrayPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(RGB2GrayPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2GRAY);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Gray(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class BGR2GrayPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(BGR2GrayPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2GRAY);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2Gray(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class RGB2YUVPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(RGB2YUVPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2YUV(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class YUV2RGBPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(YUV2RGBPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2RGB(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class RGB2LabPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(RGB2LabPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2Lab);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Lab(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class BGR2LUVPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(BGR2LUVPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2Luv);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2LUV(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class LUV2BGRPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(LUV2BGRPerfTest, TestPerformance)
{
    cv::Size sz = GetParam();

    initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_Luv2BGR);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUV2BGR(in);
    cv::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi);

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
      EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
      EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

class BGR2YUVPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(BGR2YUVPerfTest, TestPerformance)
{
  cv::Size sz = GetParam();
  initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

  cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV);

  cv::GMat in;
  auto out = cv::gapi::BGR2YUV(in);
  cv::GComputation c(in, out);

  // Warm-up graph engine:
  c.apply(in_mat1, out_mat_gapi);

  TEST_CYCLE()
  {
    c.apply(in_mat1, out_mat_gapi);
  }

  EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
  EXPECT_EQ(out_mat_gapi.size(), sz);

  SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

class YUV2BGRPerfTest : public TestPerfParams<cv::Size> {};
PERF_TEST_P_(YUV2BGRPerfTest, TestPerformance)
{
  cv::Size sz = GetParam();
  initMatsRandN(CV_8UC3, sz, CV_8UC3, false);

  cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR);

  cv::GMat in;
  auto out = cv::gapi::YUV2BGR(in);
  cv::GComputation c(in, out);

  // Warm-up graph engine:
  c.apply(in_mat1, out_mat_gapi);

  TEST_CYCLE()
  {
    c.apply(in_mat1, out_mat_gapi);
  }

  EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
  EXPECT_EQ(out_mat_gapi.size(), sz);

  SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

}
