/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Zero Lin, Zero.Lin@amd.com
//    Zhang Ying, zhangying913@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "test_precomp.hpp"

#ifdef HAVE_OPENCL

using namespace testing;
using namespace std;
using namespace cv;

PARAM_TEST_CASE(FilterTestBase, MatType,
                int, // kernel size
                Size, // dx, dy
                int, // border type, or iteration
                bool) // roi or not
{
    int type, borderType, ksize;
    bool useRoi;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        borderType = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, type, 5, 16);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);
    }

    void Near(double threshold = 0.0)
    {
        Mat roi, whole;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst_whole, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// blur

typedef FilterTestBase Blur;

OCL_TEST_P(Blur, Mat)
{
    Size kernelSize(ksize, ksize);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        blur(src_roi, dst_roi, kernelSize, Point(-1, -1), borderType);
        ocl::blur(gsrc_roi, gdst_roi, kernelSize, Point(-1, -1), borderType); // TODO anchor

        Near(1.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

typedef FilterTestBase LaplacianTest;

OCL_TEST_P(LaplacianTest, Accuracy)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        // border type is used as a scale factor for the Laplacian kernel
        double scale = static_cast<double>(borderType);

        Laplacian(src_roi, dst_roi, -1, ksize, scale);
        ocl::Laplacian(gsrc_roi, gdst_roi, -1, ksize, scale);

        Near(1e-5);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// erode & dilate

struct ErodeDilate :
        public FilterTestBase
{
    int iterations;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        iterations = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }
};

typedef ErodeDilate Erode;

OCL_TEST_P(Erode, Mat)
{
    // erode or dilate kernel
    Size kernelSize(ksize, ksize);
    Mat kernel;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        kernel = randomMat(kernelSize, CV_8UC1, 0, 3);

        random_roi();

        cv::erode(src_roi, dst_roi, kernel, Point(-1, -1), iterations);
        ocl::erode(gsrc_roi, gdst_roi, kernel, Point(-1, -1), iterations); // TODO iterations, borderType

        Near(1e-5);
    }
}

typedef ErodeDilate Dilate;

OCL_TEST_P(Dilate, Mat)
{
    // erode or dilate kernel
    Mat kernel;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        kernel = randomMat(Size(3, 3), CV_8UC1, 0, 3);

        random_roi();

        cv::dilate(src_roi, dst_roi, kernel, Point(-1, -1), iterations);
        ocl::dilate(gsrc_roi, gdst_roi, kernel, Point(-1, -1), iterations); // TODO iterations, borderType

        Near(1e-5);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel

struct SobelTest :
        public FilterTestBase
{
    int dx, dy;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        borderType = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        Size d = GET_PARAM(2);
        dx = d.width, dy = d.height;
    }
};

OCL_TEST_P(SobelTest, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Sobel(src_roi, dst_roi, -1, dx, dy, ksize, /* scale */ 0.00001, /* delta */0, borderType);
        ocl::Sobel(gsrc_roi, gdst_roi, -1, dx, dy, ksize, /* scale */ 0.00001, /* delta */ 0, borderType);

        Near(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr

typedef SobelTest ScharrTest;

OCL_TEST_P(ScharrTest, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Scharr(src_roi, dst_roi, -1, dx, dy, /* scale */ 1, /* delta */ 0, borderType);
        ocl::Scharr(gsrc_roi, gdst_roi, -1, dx, dy, /* scale */ 1, /* delta */ 0, borderType);

        Near(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

struct GaussianBlurTest :
        public FilterTestBase
{
    double sigma1, sigma2;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        borderType = GET_PARAM(3);

        sigma1 = rng.uniform(0.1, 1.0);
        sigma2 = rng.uniform(0.1, 1.0);
    }
};

OCL_TEST_P(GaussianBlurTest, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        GaussianBlur(src_roi, dst_roi, Size(ksize, ksize), sigma1, sigma2, borderType);
        ocl::GaussianBlur(gsrc_roi, gdst_roi, Size(ksize, ksize), sigma1, sigma2, borderType);

        Near(1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

typedef FilterTestBase Filter2D;

OCL_TEST_P(Filter2D, Mat)
{
    const Size kernelSize(ksize, ksize);
    Mat kernel;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        kernel = randomMat(kernelSize, CV_32FC1, 0.0, 1.0);

        random_roi();

        cv::filter2D(src_roi, dst_roi, -1, kernel, Point(-1, -1), 0.0, borderType); // TODO anchor
        ocl::filter2D(gsrc_roi, gdst_roi, -1, kernel, Point(-1, -1), borderType);

        Near(1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral

typedef FilterTestBase Bilateral;

OCL_TEST_P(Bilateral, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        double sigmacolor = rng.uniform(20, 100);
        double sigmaspace = rng.uniform(10, 40);

        cv::bilateralFilter(src_roi, dst_roi, ksize, sigmacolor, sigmaspace, borderType);
        ocl::bilateralFilter(gsrc_roi, gdst_roi, ksize, sigmacolor, sigmaspace, borderType);

        Near(1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// AdaptiveBilateral

typedef FilterTestBase AdaptiveBilateral;

OCL_TEST_P(AdaptiveBilateral, Mat)
{
    const Size kernelSize(ksize, ksize);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        adaptiveBilateralFilter(src_roi, dst_roi, kernelSize, 5, Point(-1, -1), borderType); // TODO anchor
        ocl::adaptiveBilateralFilter(gsrc_roi, gdst_roi, kernelSize, 5, Point(-1, -1), borderType);

        Near(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// MedianFilter

typedef FilterTestBase MedianFilter;

OCL_TEST_P(MedianFilter, Mat)
{
    for (int i = 0; i < LOOP_TIMES; ++i)
    {
        random_roi();

        medianBlur(src_roi, dst_roi, ksize);
        ocl::medianFilter(gsrc_roi, gdst_roi, ksize);

        Near();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_CASE_P(Filter, Blur, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(3, 5, 7),
                            Values(Size(0, 0)), // not used
                            Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, (int)BORDER_REFLECT_101),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, LaplacianTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(1, 3),
                            Values(Size(0, 0)), // not used
                            Values(1, 2), // value is used as scale factor for kernel
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Erode, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5, 7),
                            Values(Size(0, 0)), // not used
                            testing::Range(1, 2),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Dilate, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5, 7),
                            Values(Size(0, 0)), // not used
                            testing::Range(1, 2),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, SobelTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5),
                            Values(Size(1, 0), Size(1, 1), Size(2, 0), Size(2, 1)),
                            Values((int)BORDER_CONSTANT, (int)BORDER_REFLECT101,
                                   (int)BORDER_REPLICATE, (int)BORDER_REFLECT),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, ScharrTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(0), // not used
                            Values(Size(0, 1), Size(1, 0)),
                            Values((int)BORDER_CONSTANT, (int)BORDER_REFLECT101,
                                   (int)BORDER_REPLICATE, (int)BORDER_REFLECT),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, GaussianBlurTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(3, 5),
                            Values(Size(0, 0)), // not used
                            Values((int)BORDER_CONSTANT, (int)BORDER_REFLECT101,
                                   (int)BORDER_REPLICATE, (int)BORDER_REFLECT),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Filter2D, testing::Combine(
                            Values(CV_8UC1, CV_32FC1, CV_32FC4),
                            Values(3, 15, 25),
                            Values(Size(0, 0)), // not used
                            Values((int)BORDER_CONSTANT, (int)BORDER_REFLECT101,
                                   (int)BORDER_REPLICATE, (int)BORDER_REFLECT),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Bilateral, Combine(
                            Values(CV_8UC1, CV_8UC3),
                            Values(5, 9),
                            Values(Size(0, 0)), // not used
                            Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE,
                                   (int)BORDER_REFLECT, (int)BORDER_WRAP, (int)BORDER_REFLECT_101),
                            Values(false))); // TODO does not work with ROI

INSTANTIATE_TEST_CASE_P(Filter, AdaptiveBilateral, Combine(
                            Values(CV_8UC1, CV_8UC3),
                            Values(5, 9),
                            Values(Size(0, 0)), // not used
                            Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE,
                                   (int)BORDER_REFLECT, (int)BORDER_REFLECT_101),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, MedianFilter, Combine(
                            Values((MatType)CV_8UC1, (MatType)CV_8UC4, (MatType)CV_32FC1, (MatType)CV_32FC4),
                            Values(3, 5),
                            Values(Size(0, 0)), // not used
                            Values(0), // not used
                            Bool()));

#endif // HAVE_OPENCL
