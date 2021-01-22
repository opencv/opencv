// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_INL_HPP

#include <opencv2/gapi/imgproc.hpp>
#include "gapi_imgproc_tests.hpp"

#include "gapi_imgproc_tests_common.hpp"

namespace opencv_test
{

// FIXME avoid this code duplicate in perf tests
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
            yuv422_line[i * 2 + 2] = cv::saturate_cast<uchar>( 0.615   * r - 0.51499 * g - 0.10001 * b + 128.f);  // V0

            r = rgb_line[i * 3 + 3];
            g = rgb_line[i * 3 + 4];
            b = rgb_line[i * 3 + 5];

            yuv422_line[i * 2 + 3] = cv::saturate_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);   // Y1
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

TEST_P(Filter2DTest, AccuracyTest)
{
    cv::Point anchor = {-1, -1};
    double delta = 0;

    cv::Mat kernel = cv::Mat(filterSize, CV_32FC1);
    cv::Scalar kernMean, kernStddev;

    const auto kernSize = filterSize.width * filterSize.height;
    const auto bigKernSize = 49;

    if (kernSize < bigKernSize)
    {
        kernMean = cv::Scalar(0.3);
        kernStddev = cv::Scalar(0.5);
    }
    else
    {
        kernMean = cv::Scalar(0.008);
        kernStddev = cv::Scalar(0.008);
    }

    randn(kernel, kernMean, kernStddev);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::filter2D(in, dtype, kernel, anchor, delta, borderType);

    cv::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Point anchor = {-1, -1};
    bool normalize = true;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::boxFilter(in, dtype, cv::Size(filterSize, filterSize), anchor, normalize,
        borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::boxFilter(in_mat1, out_mat_ocv, dtype, cv::Size(filterSize, filterSize), anchor,
            normalize, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(SepFilterTest, AccuracyTest)
{
    cv::Mat kernelX(kernSize, 1, CV_32F);
    cv::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);

    cv::Point anchor = cv::Point(-1, -1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::sepFilter(in, dtype, kernelX, kernelY, anchor, cv::Scalar() );

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Point anchor = {-1, -1};

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::blur(in, cv::Size(filterSize, filterSize), anchor, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Size kSize = cv::Size(kernSize, kernSize);
    double sigmaX = rand();

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::gaussianBlur(in, kSize, sigmaX);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::medianBlur(in, kernSize);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode(in, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::erode3x3(in, numIters);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Mat kernel = cv::getStructuringElement(kernType, cv::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate(in, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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
    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::dilate3x3(in, numIters);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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

TEST_P(MorphologyExTest, AccuracyTest)
{
    cv::MorphShapes defShape = cv::MORPH_RECT;
    int defKernSize = 3;
    cv::Mat kernel = cv::getStructuringElement(defShape, cv::Size(defKernSize, defKernSize));

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::morphologyEx(in, op, kernel);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::morphologyEx(in_mat1, out_mat_ocv, op, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(SobelTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Sobel(in, dtype, dx, dy, kernSize );

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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

TEST_P(SobelXYTest, AccuracyTest)
{
    cv::Mat out_mat_ocv2;
    cv::Mat out_mat_gapi2;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::SobelXY(in, dtype, order, kernSize, 1, 0, border_type, border_val);

    cv::GComputation c(cv::GIn(in), cv::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(cv::gin(in_mat1), cv::gout(out_mat_gapi, out_mat_gapi2), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        // workaround for cv::Sobel
        cv::Mat temp_in;
        if(border_type == cv::BORDER_CONSTANT)
        {
            int n_pixels = (kernSize - 1) / 2;
            cv::copyMakeBorder(in_mat1, temp_in, n_pixels, n_pixels, n_pixels, n_pixels, border_type, border_val);
            in_mat1 = temp_in(cv::Rect(n_pixels, n_pixels, in_mat1.cols, in_mat1.rows));
        }
        cv::Sobel(in_mat1, out_mat_ocv, dtype, order, 0, kernSize, 1, 0, border_type);
        cv::Sobel(in_mat1, out_mat_ocv2, dtype, 0, order, kernSize, 1, 0, border_type);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat_gapi2, out_mat_ocv2));
        EXPECT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(out_mat_gapi2.size(), sz);
    }
}

TEST_P(LaplacianTest, AccuracyTest)
{
    double delta = 10;
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Laplacian(in, dtype, kernSize, scale, delta, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Laplacian(in_mat1, out_mat_ocv, dtype, kernSize, scale, delta, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BilateralFilterTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::bilateralFilter(in, d, sigmaColor, sigmaSpace, borderType);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::bilateralFilter(in_mat1, out_mat_ocv, d, sigmaColor, sigmaSpace, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(EqHistTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::equalizeHist(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::equalizeHist(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(CannyTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
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

TEST_P(GoodFeaturesTest, AccuracyTest)
{
    double k = 0.04;

    initMatFromImage(type, fileName);

    std::vector<cv::Point2f> outVecOCV, outVecGAPI;

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::goodFeaturesToTrack(in, maxCorners, qualityLevel, minDistance, cv::Mat(),
                                             blockSize, useHarrisDetector, k);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(in_mat1), cv::gout(outVecGAPI), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::goodFeaturesToTrack(in_mat1, outVecOCV, maxCorners, qualityLevel, minDistance,
                                cv::noArray(), blockSize, useHarrisDetector, k);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(outVecGAPI, outVecOCV));
    }
}

TEST_P(FindContoursNoOffsetTest, AccuracyTest)
{
    findContoursTestBody(sz, type, mode, method, cmpF, getCompileArgs());
}

TEST_P(FindContoursOffsetTest, AccuracyTest)
{
    const cv::Size sz(1280, 720);
    const MatType2 type = CV_8UC1;
    const cv::RetrievalModes mode = cv::RETR_EXTERNAL;
    const cv::ContourApproximationModes method = cv::CHAIN_APPROX_NONE;
    const CompareMats cmpF = AbsExact().to_compare_obj();
    const cv::Point offset(15, 15);

    findContoursTestBody(sz, type, mode, method, cmpF, getCompileArgs(), offset);
}

TEST_P(FindContoursHNoOffsetTest, AccuracyTest)
{
    findContoursTestBody<HIERARCHY>(sz, type, mode, method, cmpF, getCompileArgs());
}

TEST_P(FindContoursHOffsetTest, AccuracyTest)
{
    const cv::Size sz(1280, 720);
    const MatType2 type = CV_8UC1;
    const cv::RetrievalModes mode = cv::RETR_EXTERNAL;
    const cv::ContourApproximationModes method = cv::CHAIN_APPROX_NONE;
    const CompareMats cmpF = AbsExact().to_compare_obj();
    const cv::Point offset(15, 15);
    std::vector<std::vector<cv::Point>> outCtsOCV,  outCtsGAPI;
    std::vector<cv::Vec4i>              outHierOCV, outHierGAPI;

    findContoursTestBody<HIERARCHY>(sz, type, mode, method, cmpF, getCompileArgs(), offset);
}

TEST_P(BoundingRectMatTest, AccuracyTest)
{
    if (initByVector)
    {
        initMatByPointsVectorRandU<cv::Point_>(type, sz, dtype);
    }
    else
    {
        initMatrixRandU(type, sz, dtype);
    }
    boundingRectTestBody(in_mat1, cmpF, getCompileArgs());
}

TEST_P(BoundingRectVector32STest, AccuracyTest)

{
    std::vector<cv::Point2i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    boundingRectTestBody(in_vector, cmpF, getCompileArgs());
}

TEST_P(BoundingRectVector32FTest, AccuracyTest)
{
    std::vector<cv::Point2f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    boundingRectTestBody(in_vector, cmpF, getCompileArgs());
}

TEST_P(FitLine2DMatVectorTest, AccuracyTest)
{
    fitLineTestBody(in_mat1, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine2DVector32STest, AccuracyTest)
{
    std::vector<cv::Point2i> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine2DVector32FTest, AccuracyTest)
{
    std::vector<cv::Point2f> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine2DVector64FTest, AccuracyTest)
{
    std::vector<cv::Point2d> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DMatVectorTest, AccuracyTest)
{
    fitLineTestBody(in_mat1, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DVector32STest, AccuracyTest)
{
    std::vector<cv::Point3i> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DVector32FTest, AccuracyTest)
{
    std::vector<cv::Point3f> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DVector64FTest, AccuracyTest)
{
    std::vector<cv::Point3d> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(BGR2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2RGB(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2GrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Gray(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2GrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2Gray(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2YUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2YUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(YUV2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2RGB(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2I420Test, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2I420(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 3 / 2));
    }
}

TEST_P(RGB2I420Test, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2I420(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2YUV_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 3 / 2));
    }
}

TEST_P(I4202BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::I4202BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 2 / 3));
    }
}

TEST_P(I4202RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::I4202RGB(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2RGB_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 2 / 3));
    }
}

TEST_P(NV12toRGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in_y;
    cv::GMat in_uv;
    auto out = cv::gapi::NV12toRGB(in_y, in_uv);

    // Additional mat for uv
    cv::Mat in_mat_uv(cv::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::GComputation c(cv::GIn(in_y, in_uv), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat_uv), cv::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, cv::COLOR_YUV2RGB_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(NV12toBGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in_y;
    cv::GMat in_uv;
    auto out = cv::gapi::NV12toBGR(in_y, in_uv);

    // Additional mat for uv
    cv::Mat in_mat_uv(cv::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::GComputation c(cv::GIn(in_y, in_uv), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat_uv), cv::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, cv::COLOR_YUV2BGR_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(NV12toGrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in_y;
    cv::GMat in_uv;
    auto out = cv::gapi::NV12toGray(in_y, in_uv);

    // Additional mat for uv
    cv::Mat in_mat_uv(cv::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::GComputation c(cv::GIn(in_y, in_uv), cv::GOut(out));
    c.apply(cv::gin(in_mat1, in_mat_uv), cv::gout(out_mat_gapi), getCompileArgs());

    cv::Mat out_mat_ocv_planar;
    cv::Mat uv_planar(in_mat1.rows / 2, in_mat1.cols, CV_8UC1, in_mat_uv.data);
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::vconcat(in_mat1, uv_planar, out_mat_ocv_planar);
        cv::cvtColor(out_mat_ocv_planar, out_mat_ocv, cv::COLOR_YUV2GRAY_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

static void toPlanar(const cv::Mat& in, cv::Mat& out)
{
    GAPI_Assert(out.depth() == in.depth());
    GAPI_Assert(out.channels() == 1);
    GAPI_Assert(in.channels() == 3);
    GAPI_Assert(out.cols == in.cols);
    GAPI_Assert(out.rows == 3*in.rows);

    std::vector<cv::Mat> outs(3);
    for (int i = 0; i < 3; i++) {
        outs[i] = out(cv::Rect(0, i*in.rows, in.cols, in.rows));
    }
    cv::split(in, outs);
}

TEST_P(NV12toRGBpTest, AccuracyTest)
{
    cv::Size sz_p = cv::Size(sz.width, sz.height * 3);
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in_y;
    cv::GMat in_uv;
    auto out = cv::gapi::NV12toRGBp(in_y, in_uv);

    // Additional mat for uv
    cv::Mat in_mat_uv(cv::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::GComputation c(cv::GIn(in_y, in_uv), cv::GOut(out));
    cv::Mat out_mat_gapi_planar(cv::Size(sz.width, sz.height * 3), CV_8UC1);
    c.apply(cv::gin(in_mat1, in_mat_uv), cv::gout(out_mat_gapi_planar), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    cv::Mat out_mat_ocv_planar(cv::Size(sz.width, sz.height * 3), CV_8UC1);
    {
        cv::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, cv::COLOR_YUV2RGB_NV12);
        toPlanar(out_mat_ocv, out_mat_ocv_planar);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi_planar, out_mat_ocv_planar));
        EXPECT_EQ(out_mat_gapi_planar.size(), sz_p);
    }
}


TEST_P(NV12toBGRpTest, AccuracyTest)
{
    cv::Size sz_p = cv::Size(sz.width, sz.height * 3);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in_y;
    cv::GMat in_uv;
    auto out = cv::gapi::NV12toBGRp(in_y, in_uv);

    // Additional mat for uv
    cv::Mat in_mat_uv(cv::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::GComputation c(cv::GIn(in_y, in_uv), cv::GOut(out));
    cv::Mat out_mat_gapi_planar(cv::Size(sz.width, sz.height * 3), CV_8UC1);
    c.apply(cv::gin(in_mat1, in_mat_uv), cv::gout(out_mat_gapi_planar), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    cv::Mat out_mat_ocv_planar(cv::Size(sz.width, sz.height * 3), CV_8UC1);
    {
        cv::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, cv::COLOR_YUV2BGR_NV12);
        toPlanar(out_mat_ocv, out_mat_ocv_planar);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi_planar, out_mat_ocv_planar));
        EXPECT_EQ(out_mat_gapi_planar.size(), sz_p);
    }
}

TEST_P(RGB2LabTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2Lab(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2Lab);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2LUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2LUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2Luv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(LUV2BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::LUV2BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_Luv2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BGR2YUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BGR2YUV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BGR2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(YUV2BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::YUV2BGR(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_YUV2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2HSVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2HSV(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_RGB2HSV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(BayerGR2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::BayerGR2RGB(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::cvtColor(in_mat1, out_mat_ocv, cv::COLOR_BayerGR2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}

TEST_P(RGB2YUV422Test, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::RGB2YUV422(in);

    cv::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        convertRGB2YUV422Ref(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
