/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace {

IMPLEMENT_PARAM_CLASS(KSize, cv::Size)

cv::Mat getInnerROI(cv::InputArray m_, cv::Size ksize)
{
    cv::Mat m = getMat(m_);
    cv::Rect roi(ksize.width, ksize.height, m.cols - 2 * ksize.width, m.rows - 2 * ksize.height);
    return m(roi);
}

cv::Mat getInnerROI(cv::InputArray m, int ksize)
{
    return getInnerROI(m, cv::Size(ksize, ksize));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Blur

IMPLEMENT_PARAM_CLASS(Anchor, cv::Point)

PARAM_TEST_CASE(Blur, cv::gpu::DeviceInfo, cv::Size, MatType, KSize, Anchor, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    cv::Point anchor;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        anchor = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Blur, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::blur(loadMat(src, useRoi), dst, ksize, anchor);

    cv::Mat dst_gold;
    cv::blur(src, dst_gold, ksize, anchor);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Blur, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel

IMPLEMENT_PARAM_CLASS(Deriv_X, int)
IMPLEMENT_PARAM_CLASS(Deriv_Y, int)

PARAM_TEST_CASE(Sobel, cv::gpu::DeviceInfo, cv::Size, MatType, KSize, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        dx = GET_PARAM(4);
        dy = GET_PARAM(5);
        borderType = GET_PARAM(6);
        useRoi = GET_PARAM(7);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Sobel, Accuracy)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::Sobel(loadMat(src, useRoi), dst, -1, dx, dy, ksize.width, 1.0, borderType);

    cv::Mat dst_gold;
    cv::Sobel(src, dst_gold, -1, dx, dy, ksize.width, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Sobel, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7))),
    testing::Values(Deriv_X(0), Deriv_X(1), Deriv_X(2)),
    testing::Values(Deriv_Y(0), Deriv_Y(1), Deriv_Y(2)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr

PARAM_TEST_CASE(Scharr, cv::gpu::DeviceInfo, cv::Size, MatType, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        dx = GET_PARAM(3);
        dy = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Scharr, Accuracy)
{
    if (dx + dy != 1)
        return;

    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::Scharr(loadMat(src, useRoi), dst, -1, dx, dy, 1.0, borderType);

    cv::Mat dst_gold;
    cv::Scharr(src, dst_gold, -1, dx, dy, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Scharr, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Deriv_X(0), Deriv_X(1)),
    testing::Values(Deriv_Y(0), Deriv_Y(1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

PARAM_TEST_CASE(GaussianBlur, cv::gpu::DeviceInfo, cv::Size, MatType, KSize, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        borderType = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(GaussianBlur, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    double sigma1 = randomDouble(0.1, 1.0);
    double sigma2 = randomDouble(0.1, 1.0);

    if (ksize.height > 16 && !supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
    {
        try
        {
            cv::gpu::GpuMat dst;
            cv::gpu::GaussianBlur(loadMat(src), dst, ksize, sigma1, sigma2, borderType);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat dst = createMat(size, type, useRoi);
        cv::gpu::GaussianBlur(loadMat(src, useRoi), dst, ksize, sigma1, sigma2, borderType);

        cv::Mat dst_gold;
        cv::GaussianBlur(src, dst_gold, ksize, sigma1, sigma2, borderType);

        EXPECT_MAT_NEAR(dst_gold, dst, 4.0);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, GaussianBlur, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(KSize(cv::Size(3, 3)),
                    KSize(cv::Size(5, 5)),
                    KSize(cv::Size(7, 7)),
                    KSize(cv::Size(9, 9)),
                    KSize(cv::Size(11, 11)),
                    KSize(cv::Size(13, 13)),
                    KSize(cv::Size(15, 15)),
                    KSize(cv::Size(17, 17)),
                    KSize(cv::Size(19, 19)),
                    KSize(cv::Size(21, 21)),
                    KSize(cv::Size(23, 23)),
                    KSize(cv::Size(25, 25)),
                    KSize(cv::Size(27, 27)),
                    KSize(cv::Size(29, 29)),
                    KSize(cv::Size(31, 31))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

PARAM_TEST_CASE(Laplacian, cv::gpu::DeviceInfo, cv::Size, MatType, KSize, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Laplacian, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::Laplacian(loadMat(src, useRoi), dst, -1, ksize.width);

    cv::Mat dst_gold;
    cv::Laplacian(src, dst_gold, -1, ksize.width);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Laplacian, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_32FC1)),
    testing::Values(KSize(cv::Size(1, 1)), KSize(cv::Size(3, 3))),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Erode

IMPLEMENT_PARAM_CLASS(Iterations, int)

PARAM_TEST_CASE(Erode, cv::gpu::DeviceInfo, cv::Size, MatType, Anchor, Iterations, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        anchor = GET_PARAM(3);
        iterations = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Erode, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::erode(loadMat(src, useRoi), dst, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::erode(src, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Erode, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate

PARAM_TEST_CASE(Dilate, cv::gpu::DeviceInfo, cv::Size, MatType, Anchor, Iterations, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        anchor = GET_PARAM(3);
        iterations = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Dilate, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::dilate(loadMat(src, useRoi), dst, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::dilate(src, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Dilate, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// MorphEx

CV_ENUM(MorphOp, cv::MORPH_OPEN, cv::MORPH_CLOSE, cv::MORPH_GRADIENT, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT)
#define ALL_MORPH_OPS testing::Values(MorphOp(cv::MORPH_OPEN), MorphOp(cv::MORPH_CLOSE), MorphOp(cv::MORPH_GRADIENT), MorphOp(cv::MORPH_TOPHAT), MorphOp(cv::MORPH_BLACKHAT))

PARAM_TEST_CASE(MorphEx, cv::gpu::DeviceInfo, cv::Size, MatType, MorphOp, Anchor, Iterations, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int morphOp;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        morphOp = GET_PARAM(3);
        anchor = GET_PARAM(4);
        iterations = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(MorphEx, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::morphologyEx(loadMat(src, useRoi), dst, morphOp, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::morphologyEx(src, dst_gold, morphOp, kernel, anchor, iterations);

    cv::Size border = cv::Size(kernel.cols + (iterations + 1) * kernel.cols + 2, kernel.rows + (iterations + 1) * kernel.rows + 2);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, border), getInnerROI(dst, border), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, MorphEx, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    ALL_MORPH_OPS,
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

PARAM_TEST_CASE(Filter2D, cv::gpu::DeviceInfo, cv::Size, MatType, KSize, Anchor, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    cv::Mat img;
    cv::Mat kernel;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        anchor = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Filter2D, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = randomMat(cv::Size(ksize.width, ksize.height), CV_32FC1, 0.0, 1.0);

    cv::gpu::GpuMat dst = createMat(size, type, useRoi);
    cv::gpu::filter2D(loadMat(src, useRoi), dst, -1, kernel, anchor, borderType);

    cv::Mat dst_gold;
    cv::filter2D(src, dst_gold, -1, kernel, anchor, 0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, CV_MAT_DEPTH(type) == CV_32F ? 1e-1 : 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Filter2D, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC4)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7)), KSize(cv::Size(11, 11)), KSize(cv::Size(13, 13)), KSize(cv::Size(15, 15))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

} // namespace
