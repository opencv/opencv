// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP


#include "gapi_imgproc_perf_tests.hpp"

#include "../../test/common/gapi_imgproc_tests_common.hpp"

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

PERF_TEST_P_(MorphologyExPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    cv::MorphTypes op = cv::MORPH_ERODE;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, op, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    cv::MorphShapes defShape = cv::MORPH_RECT;
    int defKernSize = 3;
    cv::Mat kernel = cv::getStructuringElement(defShape, cv::Size(defKernSize, defKernSize));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::morphologyEx(in_mat1, out_mat_ocv, op, kernel);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::morphologyEx(in, op, kernel);
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

PERF_TEST_P_(LaplacianPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Laplacian(in_mat1, out_mat_ocv, dtype, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Laplacian(in, dtype, kernSize);
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

PERF_TEST_P_(BilateralFilterPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int dtype = 0, d = 0, borderType = BORDER_DEFAULT;
    double sigmaColor = 0, sigmaSpace = 0;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, dtype, sz, d, sigmaColor, sigmaSpace,
             compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::bilateralFilter(in_mat1, out_mat_ocv, d, sigmaColor, sigmaSpace, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::bilateralFilter(in, d, sigmaColor, sigmaSpace, borderType);
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

PERF_TEST_P_(GoodFeaturesPerfTest, TestPerformance)
{
    double k = 0.04;

    compare_vector_f<cv::Point2f> cmpF;
    std::string fileName = "";
    int type = -1, maxCorners = -1, blockSize = -1;
    double qualityLevel = 0.0, minDistance = 0.0;
    bool useHarrisDetector = false;
    cv::GCompileArgs compileArgs;
    std::tie(cmpF, fileName, type, maxCorners, qualityLevel,
             minDistance, blockSize, useHarrisDetector, compileArgs) = GetParam();

    initMatFromImage(type, fileName);
    std::vector<cv::Point2f> outVecOCV, outVecGAPI;

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::goodFeaturesToTrack(in_mat1, outVecOCV, maxCorners, qualityLevel, minDistance,
                                cv::noArray(), blockSize, useHarrisDetector, k);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::goodFeaturesToTrack(in, maxCorners, qualityLevel, minDistance, cv::Mat(),
                                             blockSize, useHarrisDetector, k);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Warm-up graph engine:
    c.apply(cv::gin(in_mat1), cv::gout(outVecGAPI), std::move(compileArgs));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(outVecGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(outVecGAPI, outVecOCV));
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(FindContoursPerfTest, TestPerformance)
{
    CompareMats cmpF;
    MatType type;
    cv::Size sz;
    cv::RetrievalModes mode;
    cv::ContourApproximationModes method;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, mode, method, compile_args) = GetParam();

    cv::Mat in;
    initMatForFindingContours(in, sz, type);
    cv::Point offset = cv::Point();
    std::vector<cv::Vec4i> out_hier_gapi = std::vector<cv::Vec4i>();

    std::vector<std::vector<cv::Point>> out_cnts_gapi;
    cv::GComputation c(findContoursTestGAPI(in, mode, method, std::move(compile_args),
                                            out_cnts_gapi, out_hier_gapi, offset));

    TEST_CYCLE()
    {
        c.apply(gin(in, offset), gout(out_cnts_gapi));
    }

    findContoursTestOpenCVCompare(in, mode, method, out_cnts_gapi, out_hier_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FindContoursHPerfTest, TestPerformance)
{
    CompareMats cmpF;
    MatType type;
    cv::Size sz;
    cv::RetrievalModes mode;
    cv::ContourApproximationModes method;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, mode, method, compile_args) = GetParam();

    cv::Mat in;
    initMatForFindingContours(in, sz, type);
    cv::Point offset = cv::Point();

    std::vector<std::vector<cv::Point>> out_cnts_gapi;
    std::vector<cv::Vec4i>              out_hier_gapi;
    cv::GComputation c(findContoursTestGAPI<HIERARCHY>(in, mode, method, std::move(compile_args),
                                                       out_cnts_gapi, out_hier_gapi, offset));

    TEST_CYCLE()
    {
        c.apply(gin(in, offset), gout(out_cnts_gapi, out_hier_gapi));
    }

    findContoursTestOpenCVCompare<HIERARCHY>(in, mode, method, out_cnts_gapi, out_hier_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BoundingRectMatPerfTest, TestPerformance)
{
    CompareRects cmpF;
    cv::Size sz;
    MatType type;
    bool initByVector = false;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, initByVector, compile_args) = GetParam();

    if (initByVector)
    {
        initMatByPointsVectorRandU<cv::Point_>(type, sz, -1);
    }
    else
    {
        initMatrixRandU(type, sz, -1, false);
    }

    cv::Rect out_rect_gapi;
    cv::GComputation c(boundingRectTestGAPI(in_mat1, std::move(compile_args), out_rect_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_rect_gapi));
    }

    boundingRectTestOpenCVCompare(in_mat1, out_rect_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BoundingRectVector32SPerfTest, TestPerformance)
{
    CompareRects cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    std::vector<cv::Point2i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Rect out_rect_gapi;
    cv::GComputation c(boundingRectTestGAPI(in_vector, std::move(compile_args), out_rect_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_rect_gapi));
    }

    boundingRectTestOpenCVCompare(in_vector, out_rect_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BoundingRectVector32FPerfTest, TestPerformance)
{
    CompareRects cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    std::vector<cv::Point2f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Rect out_rect_gapi;
    cv::GComputation c(boundingRectTestGAPI(in_vector, std::move(compile_args), out_rect_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_rect_gapi));
    }

    boundingRectTestOpenCVCompare(in_vector, out_rect_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(FitLine2DMatVectorPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    cv::Size sz;
    MatType type;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, distType, compile_args) = GetParam();

    initMatByPointsVectorRandU<cv::Point_>(type, sz, -1);

    cv::Vec4f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_mat1, distType, std::move(compile_args), out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_mat1, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine2DVector32SPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    cv::Size sz;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<cv::Point2i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Vec4f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine2DVector32FPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    cv::Size sz;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<cv::Point2f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Vec4f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine2DVector64FPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    cv::Size sz;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<cv::Point2d> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Vec4f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DMatVectorPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    cv::Size sz;
    MatType type;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, distType, compile_args) = GetParam();

    initMatByPointsVectorRandU<cv::Point3_>(type, sz, -1);

    cv::Vec6f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_mat1, distType, std::move(compile_args), out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_mat1), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_mat1, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DVector32SPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    cv::Size sz;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<cv::Point3i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Vec6f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DVector32FPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    cv::Size sz;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<cv::Point3f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Vec6f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DVector64FPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    cv::Size sz;
    cv::DistanceTypes distType;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<cv::Point3d> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    cv::Vec6f out_vec_gapi;
    cv::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(cv::gin(in_vector), cv::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
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

PERF_TEST_P_(BGR2RGBPerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2RGB);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2RGB(in);
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

PERF_TEST_P_(BGR2I420PerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2I420(in);
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
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 3 / 2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2I420PerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2I420(in);
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
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 3 / 2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(I4202BGRPerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC1, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::I4202BGR(in);
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
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 2 / 3));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(I4202RGBPerfTest, TestPerformance)
{
    compare_f cmpF;
    cv::Size sz;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC1, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::I4202RGB(in);
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
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 2 / 3));
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

PERF_TEST_P_(ResizePerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = -1;
    int interp = 1;
    cv::Size sz;
    cv::Size sz_out;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, interp, sz, sz_out, compile_args) = GetParam();

    in_mat1 = cv::Mat(sz, type);
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
    compare_f cmpF;
    MatType type = -1;
    int interp = 1;
    cv::Size sz;
    double fx = 1.0;
    double fy = 1.0;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, interp, sz, fx, fy, compile_args) = GetParam();

    in_mat1 = cv::Mat(sz, type);
    cv::Scalar mean = cv::Scalar::all(127);
    cv::Scalar stddev = cv::Scalar::all(40.f);
    cv::randn(in_mat1, mean, stddev);
    cv::Size sz_out = cv:: Size(saturate_cast<int>(sz.width*fx), saturate_cast<int>(sz.height*fy));
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

PERF_TEST_P_(ResizeInSimpleGraphPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = -1;
    cv::Size sz;
    double fx = 0.5;
    double fy = 0.5;
    cv::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, fx, fy, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    cv::Mat add_res_ocv;

    cv::add(in_mat1, in_mat2, add_res_ocv);
    cv::resize(add_res_ocv, out_mat_ocv, cv::Size(), fx, fy);

    cv::GMat in1, in2;
    cv::GMat add_res_gapi = cv::gapi::add(in1, in2);
    cv::GMat out = cv::gapi::resize(add_res_gapi, cv::Size(), fx, fy, INTER_LINEAR);
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

// This test cases were created to control performance result of test scenario mentioned here:
// https://stackoverflow.com/questions/60629331/opencv-gapi-performance-not-good-as-expected

PERF_TEST_P_(BottleneckKernelsConstInputPerfTest, TestPerformance)
{
    compare_f cmpF;
    std::string fileName = "";
    cv::GCompileArgs compile_args;
    double fx = 0.5;
    double fy = 0.5;
    std::tie(cmpF, fileName, compile_args) = GetParam();

    in_mat1 = cv::imread(findDataFile(fileName));

    cv::Mat cvvga;
    cv::Mat cvgray;
    cv::Mat cvblurred;

    cv::resize(in_mat1, cvvga, cv::Size(), fx, fy);
    cv::cvtColor(cvvga, cvgray, cv::COLOR_BGR2GRAY);
    cv::blur(cvgray, cvblurred, cv::Size(3, 3));
    cv::Canny(cvblurred, out_mat_ocv, 32, 128, 3);

    cv::GMat in;
    cv::GMat vga = cv::gapi::resize(in, cv::Size(), fx, fy, INTER_LINEAR);
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

}
#endif //OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP
