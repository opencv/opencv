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

struct KSize : cv::Size
{
    KSize() {}
    KSize(int width, int height) : cv::Size(width, height) {}
};
void PrintTo(KSize ksize, std::ostream* os)
{
    *os << "kernel size " << ksize.width << "x" << ksize.height;
}

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

PARAM_TEST_CASE(Blur, cv::gpu::DeviceInfo, KSize, Anchor, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    cv::Point anchor;
    bool useRoi;

    cv::Mat img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        anchor = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
    }
};

TEST_P(Blur, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::blur(loadMat(img_gray, useRoi), dst, ksize, anchor);

    cv::Mat dst_gold;
    cv::blur(img_gray, dst_gold, ksize, anchor);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

TEST_P(Blur, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::blur(loadMat(img_rgba, useRoi), dst, ksize, anchor);

    cv::Mat dst_gold;
    cv::blur(img_rgba, dst_gold, ksize, anchor);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Blur, testing::Combine(
    ALL_DEVICES,
    testing::Values(KSize(3, 3), KSize(5, 5), KSize(7, 7)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel

IMPLEMENT_PARAM_CLASS(Deriv_X, int)
IMPLEMENT_PARAM_CLASS(Deriv_Y, int)

PARAM_TEST_CASE(Sobel, cv::gpu::DeviceInfo, KSize, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    cv::Mat img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        dx = GET_PARAM(2);
        dy = GET_PARAM(3);
        borderType = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
    }
};

TEST_P(Sobel, Gray)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::Sobel(loadMat(img_gray, useRoi), dst, -1, dx, dy, ksize.width, 1.0, borderType);

    cv::Mat dst_gold;
    cv::Sobel(img_gray, dst_gold, -1, dx, dy, ksize.width, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Sobel, Color)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::Sobel(loadMat(img_rgba, useRoi), dst, -1, dx, dy, ksize.width, 1.0, borderType);

    cv::Mat dst_gold;
    cv::Sobel(img_rgba, dst_gold, -1, dx, dy, ksize.width, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Sobel, testing::Combine(
    ALL_DEVICES,
    testing::Values(KSize(3, 3), KSize(5, 5), KSize(7, 7)),
    testing::Values(Deriv_X(0), Deriv_X(1), Deriv_X(2)),
    testing::Values(Deriv_Y(0), Deriv_Y(1), Deriv_Y(2)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr

PARAM_TEST_CASE(Scharr, cv::gpu::DeviceInfo, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    cv::Mat img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        dx = GET_PARAM(1);
        dy = GET_PARAM(2);
        borderType = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
    }
};

TEST_P(Scharr, Gray)
{
    if (dx + dy != 1)
        return;

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::Scharr(loadMat(img_gray, useRoi), dst, -1, dx, dy, 1.0, borderType);

    cv::Mat dst_gold;
    cv::Scharr(img_gray, dst_gold, -1, dx, dy, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(Scharr, Color)
{
    if (dx + dy != 1)
        return;

    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::Scharr(loadMat(img_rgba, useRoi), dst, -1, dx, dy, 1.0, borderType);

    cv::Mat dst_gold;
    cv::Scharr(img_rgba, dst_gold, -1, dx, dy, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Scharr, testing::Combine(
    ALL_DEVICES,
    testing::Values(Deriv_X(0), Deriv_X(1)),
    testing::Values(Deriv_Y(0), Deriv_Y(1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

PARAM_TEST_CASE(GaussianBlur, cv::gpu::DeviceInfo, KSize, BorderType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    int borderType;
    bool useRoi;

    cv::Mat img;
    double sigma1, sigma2;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        borderType = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());

        sigma1 = randomDouble(0.1, 1.0);
        sigma2 = randomDouble(0.1, 1.0);
    }
};

TEST_P(GaussianBlur, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::GaussianBlur(loadMat(img_gray, useRoi), dst, ksize, sigma1, sigma2, borderType);

    cv::Mat dst_gold;
    cv::GaussianBlur(img_gray, dst_gold, ksize, sigma1, sigma2, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 4.0);
}

TEST_P(GaussianBlur, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::GaussianBlur(loadMat(img_rgba, useRoi), dst, ksize, sigma1, sigma2, borderType);

    cv::Mat dst_gold;
    cv::GaussianBlur(img_rgba, dst_gold, ksize, sigma1, sigma2, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 4.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, GaussianBlur, testing::Combine(
    ALL_DEVICES,
    testing::Values(KSize(3, 3),
                    KSize(5, 5),
                    KSize(7, 7),
                    KSize(9, 9),
                    KSize(11, 11),
                    KSize(13, 13),
                    KSize(15, 15),
                    KSize(17, 17),
                    KSize(19, 19),
                    KSize(21, 21),
                    KSize(23, 23),
                    KSize(25, 25),
                    KSize(27, 27),
                    KSize(29, 29),
                    KSize(31, 31)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

PARAM_TEST_CASE(Laplacian, cv::gpu::DeviceInfo, KSize, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    bool useRoi;

    cv::Mat img;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());
    }
};

TEST_P(Laplacian, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::Laplacian(loadMat(img_gray, useRoi), dst, -1, ksize.width);

    cv::Mat dst_gold;
    cv::Laplacian(img_gray, dst_gold, -1, ksize.width);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, cv::Size(3, 3)), getInnerROI(dst, cv::Size(3, 3)), 0.0);
}

TEST_P(Laplacian, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::Laplacian(loadMat(img_rgba, useRoi), dst, -1, ksize.width);

    cv::Mat dst_gold;
    cv::Laplacian(img_rgba, dst_gold, -1, ksize.width);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, cv::Size(3, 3)), getInnerROI(dst, cv::Size(3, 3)), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Laplacian, testing::Combine(
    ALL_DEVICES,
    testing::Values(KSize(1, 1), KSize(3, 3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Erode

IMPLEMENT_PARAM_CLASS(Iterations, int)

PARAM_TEST_CASE(Erode, cv::gpu::DeviceInfo, Anchor, Iterations, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    cv::Mat img;
    cv::Mat kernel;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        anchor = GET_PARAM(1);
        iterations = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());

        kernel = cv::Mat::ones(3, 3, CV_8U);
    }
};

TEST_P(Erode, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::erode(loadMat(img_gray, useRoi), dst, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::erode(img_gray, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

TEST_P(Erode, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::erode(loadMat(img_rgba, useRoi), dst, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::erode(img_rgba, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Erode, testing::Combine(
    ALL_DEVICES,
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate

PARAM_TEST_CASE(Dilate, cv::gpu::DeviceInfo, Anchor, Iterations, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    cv::Mat img;
    cv::Mat kernel;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        anchor = GET_PARAM(1);
        iterations = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());

        kernel = cv::Mat::ones(3, 3, CV_8U);
    }
};

TEST_P(Dilate, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::dilate(loadMat(img_gray, useRoi), dst, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::dilate(img_gray, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

TEST_P(Dilate, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::dilate(loadMat(img_rgba, useRoi), dst, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::dilate(img_rgba, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Dilate, testing::Combine(
    ALL_DEVICES,
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// MorphEx

CV_ENUM(MorphOp, cv::MORPH_OPEN, cv::MORPH_CLOSE, cv::MORPH_GRADIENT, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT)
#define ALL_MORPH_OPS testing::Values(MorphOp(cv::MORPH_OPEN), MorphOp(cv::MORPH_CLOSE), MorphOp(cv::MORPH_GRADIENT), MorphOp(cv::MORPH_TOPHAT), MorphOp(cv::MORPH_BLACKHAT))

PARAM_TEST_CASE(MorphEx, cv::gpu::DeviceInfo, MorphOp, Anchor, Iterations, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int morphOp;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    cv::Mat img;
    cv::Mat kernel;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        morphOp = GET_PARAM(1);
        anchor = GET_PARAM(2);
        iterations = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());

        kernel = cv::Mat::ones(3, 3, CV_8U);
    }
};

TEST_P(MorphEx, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::morphologyEx(loadMat(img_gray, useRoi), dst, morphOp, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::morphologyEx(img_gray, dst_gold, morphOp, kernel, anchor, iterations);

    cv::Size border = cv::Size(kernel.cols + (iterations + 1) * kernel.cols + 2, kernel.rows + (iterations + 1) * kernel.rows + 2);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, border), getInnerROI(dst, border), 0.0);
}

TEST_P(MorphEx, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::morphologyEx(loadMat(img_rgba, useRoi), dst, morphOp, kernel, anchor, iterations);

    cv::Mat dst_gold;
    cv::morphologyEx(img_rgba, dst_gold, morphOp, kernel, anchor, iterations);

    cv::Size border = cv::Size(kernel.cols + (iterations + 1) * kernel.cols + 2, kernel.rows + (iterations + 1) * kernel.rows + 2);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, border), getInnerROI(dst, border), 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, MorphEx, testing::Combine(
    ALL_DEVICES,
    ALL_MORPH_OPS,
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

PARAM_TEST_CASE(Filter2D, cv::gpu::DeviceInfo, KSize, Anchor, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size ksize;
    cv::Point anchor;
    bool useRoi;

    cv::Mat img;
    cv::Mat kernel;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        anchor = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImage("stereobp/aloe-L.png");
        ASSERT_FALSE(img.empty());

        kernel = cv::Mat::ones(ksize.height, ksize.width, CV_32FC1);
    }
};

TEST_P(Filter2D, Gray)
{
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::gpu::GpuMat dst;
    cv::gpu::filter2D(loadMat(img_gray, useRoi), dst, -1, kernel, anchor);

    cv::Mat dst_gold;
    cv::filter2D(img_gray, dst_gold, -1, kernel, anchor, 0, cv::BORDER_CONSTANT);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

TEST_P(Filter2D, Color)
{
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, CV_BGR2BGRA);

    cv::gpu::GpuMat dst;
    cv::gpu::filter2D(loadMat(img_rgba, useRoi), dst, -1, kernel, anchor);

    cv::Mat dst_gold;
    cv::filter2D(img_rgba, dst_gold, -1, kernel, anchor, 0, cv::BORDER_CONSTANT);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

TEST_P(Filter2D, Gray_32FC1)
{
    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2GRAY);
    src.convertTo(src, CV_32F, 1.0 / 255.0);

    cv::gpu::GpuMat dst;
    cv::gpu::filter2D(loadMat(src, useRoi), dst, -1, kernel, anchor);

    cv::Mat dst_gold;
    cv::filter2D(src, dst_gold, -1, kernel, anchor);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-3);
}

INSTANTIATE_TEST_CASE_P(GPU_Filter, Filter2D, testing::Combine(
    ALL_DEVICES,
    testing::Values(KSize(3, 3), KSize(5, 5), KSize(7, 7), KSize(11, 11), KSize(13, 13), KSize(15, 15)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    WHOLE_SUBMAT));
