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
//     and/or other materials provided with the distribution.
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
                int, // border type
                double, // optional parameter
                bool) // roi or not
{
    bool isFP;

    int type, borderType, ksize;
    Size size;
    double param;
    bool useRoi;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        size = GET_PARAM(2);
        borderType = GET_PARAM(3);
        param = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        isFP = (CV_MAT_DEPTH(type) == CV_32F || CV_MAT_DEPTH(type) == CV_64F);
    }

    void random_roi(int minSize = 1)
    {
        if (minSize == 0)
            minSize = ksize;
        Size roiSize = randomSize(minSize, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, isFP ? 0 : 5, isFP ? 1 : 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, type, isFP ? 0.20 : 60, isFP ? 0.25 : 70);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);
    }

    void Near()
    {
        if (isFP)
            Near(1e-6, true);
        else
            Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        Mat roi, whole;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        if (relative)
        {
            EXPECT_MAT_NEAR_RELATIVE(dst_whole, whole, threshold);
            EXPECT_MAT_NEAR_RELATIVE(dst_roi, roi, threshold);
        }
        else
        {
            EXPECT_MAT_NEAR(dst_whole, whole, threshold);
            EXPECT_MAT_NEAR(dst_roi, roi, threshold);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// blur

typedef FilterTestBase Blur;

#ifdef ANDROID
OCL_TEST_P(Blur, DISABLED_Mat)
#else
OCL_TEST_P(Blur, Mat)
#endif
{
    Size kernelSize(ksize, ksize);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi(0); // TODO NOTE: min value for size is kernel size (temporary bypass border issues in CPU implementation)

        blur(src_roi, dst_roi, kernelSize, Point(-1, -1), borderType);
        ocl::blur(gsrc_roi, gdst_roi, kernelSize, Point(-1, -1), borderType); // TODO anchor

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

typedef FilterTestBase LaplacianTest;

OCL_TEST_P(LaplacianTest, Accuracy)
{
    double scale = param;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Laplacian(src_roi, dst_roi, -1, ksize, scale, 0, borderType);
        ocl::Laplacian(gsrc_roi, gdst_roi, -1, ksize, scale, 0, borderType);

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// erode & dilate

typedef FilterTestBase Erode;

OCL_TEST_P(Erode, Mat)
{
    // erode or dilate kernel
    Size kernelSize(ksize, ksize);
    Mat kernel;
    int iterations = (int)param;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        kernel = randomMat(kernelSize, CV_8UC1, 0, 3);

        cv::erode(src_roi, dst_roi, kernel, Point(-1, -1), iterations);//, borderType);
        ocl::erode(gsrc_roi, gdst_roi, kernel, Point(-1, -1), iterations);//, borderType);

        Near();
    }
}

typedef FilterTestBase Dilate;

OCL_TEST_P(Dilate, Mat)
{
    // erode or dilate kernel
    Mat kernel;
    int iterations = (int)param;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        kernel = randomMat(Size(3, 3), CV_8UC1, 0, 3);

        random_roi();

        cv::dilate(src_roi, dst_roi, kernel, Point(-1, -1), iterations);
        ocl::dilate(gsrc_roi, gdst_roi, kernel, Point(-1, -1), iterations); // TODO iterations, borderType

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel

typedef FilterTestBase SobelTest;

OCL_TEST_P(SobelTest, Mat)
{
    int dx = size.width, dy = size.height;
    double scale = param;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Sobel(src_roi, dst_roi, -1, dx, dy, ksize, scale, /* delta */0, borderType);
        ocl::Sobel(gsrc_roi, gdst_roi, -1, dx, dy, ksize, scale, /* delta */0, borderType);

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr

typedef FilterTestBase ScharrTest;

OCL_TEST_P(ScharrTest, Mat)
{
    int dx = size.width, dy = size.height;
    double scale = param;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Scharr(src_roi, dst_roi, -1, dx, dy, scale, /* delta */ 0, borderType);
        ocl::Scharr(gsrc_roi, gdst_roi, -1, dx, dy, scale, /* delta */ 0, borderType);

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

typedef FilterTestBase GaussianBlurTest;

OCL_TEST_P(GaussianBlurTest, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        double sigma1 = rng.uniform(0.1, 1.0);
        double sigma2 = rng.uniform(0.1, 1.0);

        GaussianBlur(src_roi, dst_roi, Size(ksize, ksize), sigma1, sigma2, borderType);
        ocl::GaussianBlur(gsrc_roi, gdst_roi, Size(ksize, ksize), sigma1, sigma2, borderType);

        Near(CV_MAT_DEPTH(type) == CV_8U ? 3 : 5e-5, false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

typedef FilterTestBase Filter2D;

OCL_TEST_P(Filter2D, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Point anchor(-1, -1);
        if (size.width >= 0)
            anchor.x = size.width % ksize;
        if (size.height >= 0)
            anchor.y = size.height % ksize;

        const Size kernelSize(ksize, ksize);
        Mat kernel = randomMat(kernelSize, CV_32FC1, 0, 1.0);
        kernel *= 1.0 / (double)(ksize * ksize);

        cv::filter2D(src_roi, dst_roi, -1, kernel, anchor, 0.0, borderType);
        ocl::filter2D(gsrc_roi, gdst_roi, -1, kernel, anchor, 0.0, borderType);

        Near();
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

        Near();
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

        adaptiveBilateralFilter(src_roi, dst_roi, kernelSize, 5, 1, Point(-1, -1), borderType); // TODO anchor
        ocl::adaptiveBilateralFilter(gsrc_roi, gdst_roi, kernelSize, 5, 1, Point(-1, -1), borderType);

        Near();
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

#define FILTER_BORDER_SET_NO_ISOLATED \
    Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, (int)BORDER_WRAP, (int)BORDER_REFLECT_101/*, \
            (int)BORDER_CONSTANT|BORDER_ISOLATED, (int)BORDER_REPLICATE|BORDER_ISOLATED, \
            (int)BORDER_REFLECT|BORDER_ISOLATED, (int)BORDER_WRAP|BORDER_ISOLATED, \
            (int)BORDER_REFLECT_101|BORDER_ISOLATED*/) // WRAP and ISOLATED are not supported by cv:: version

#define FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED \
    Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, /*(int)BORDER_WRAP,*/ (int)BORDER_REFLECT_101/*, \
            (int)BORDER_CONSTANT|BORDER_ISOLATED, (int)BORDER_REPLICATE|BORDER_ISOLATED, \
            (int)BORDER_REFLECT|BORDER_ISOLATED, (int)BORDER_WRAP|BORDER_ISOLATED, \
            (int)BORDER_REFLECT_101|BORDER_ISOLATED*/) // WRAP and ISOLATED are not supported by cv:: version

#define FILTER_DATATYPES Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, \
                                CV_32FC1, CV_32FC3, CV_32FC4, \
                                CV_64FC1, CV_64FC3, CV_64FC4)

INSTANTIATE_TEST_CASE_P(Filter, Blur, Combine(
                            FILTER_DATATYPES,
                            Values(3, 5, 7),
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, LaplacianTest, Combine(
                            FILTER_DATATYPES,
                            Values(1, 3),
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(1.0, 0.2, 3.0), // scalar
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Erode, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5, 7),
                            Values(Size(0, 0)), // not used
                            Values(0), // not used
                            Values(1.0, 2.0, 3.0),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Dilate, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5, 7),
                            Values(Size(0, 0)), // not used
                            Values(0), // not used
                            Values(1.0, 2.0, 3.0),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, SobelTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5),
                            Values(Size(1, 0), Size(1, 1), Size(2, 0), Size(2, 1)), // dx, dy
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, ScharrTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(1),
                            Values(Size(0, 1), Size(1, 0)), // dx, dy
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(1.0, 0.2), // scalar
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, GaussianBlurTest, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(3, 5),
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Filter2D, testing::Combine(
                            FILTER_DATATYPES,
                            Values(3, 15), // TODO 25: CPU implementation has some issues
                            Values(Size(-1, -1), Size(0, 0), Size(2, 1)), // anchor
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, Bilateral, Combine(
                            Values(CV_8UC1, CV_8UC3),
                            Values(5, 9),
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, AdaptiveBilateral, Combine(
                            Values(CV_8UC1, CV_8UC3),
                            Values(5, 9),
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Filter, MedianFilter, Combine(
                            Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(3, 5),
                            Values(Size(0, 0)), // not used
                            Values(0), // not used
                            Values(0.0), // not used
                            Bool()));

#endif // HAVE_OPENCL
