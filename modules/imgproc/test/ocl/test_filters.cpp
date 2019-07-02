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

#include "../test_precomp.hpp"
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

PARAM_TEST_CASE(FilterTestBase, MatType,
                int, // kernel size
                Size, // dx, dy
                BorderType, // border type
                double, // optional parameter
                bool, // roi or not
                int)  // width multiplier
{
    int type, borderType, ksize;
    Size size;
    double param;
    bool useRoi;
    int widthMultiple;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        size = GET_PARAM(2);
        borderType = GET_PARAM(3);
        param = GET_PARAM(4);
        useRoi = GET_PARAM(5);
        widthMultiple = GET_PARAM(6);
    }

    void random_roi(int minSize = 1)
    {
        if (minSize == 0)
            minSize = ksize;

        Size roiSize = randomSize(minSize, MAX_VALUE);
        roiSize.width &= ~((widthMultiple * 2) - 1);
        roiSize.width += widthMultiple;

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -60, 70);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near()
    {
        int depth = CV_MAT_DEPTH(type);
        bool isFP = depth >= CV_32F;

        if (isFP)
            Near(1e-6, true);
        else
            Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral

typedef FilterTestBase Bilateral;

OCL_TEST_P(Bilateral, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        double sigmacolor = rng.uniform(20, 100);
        double sigmaspace = rng.uniform(10, 40);

        OCL_OFF(cv::bilateralFilter(src_roi, dst_roi, ksize, sigmacolor, sigmaspace, borderType));
        OCL_ON(cv::bilateralFilter(usrc_roi, udst_roi, ksize, sigmacolor, sigmaspace, borderType));

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

typedef FilterTestBase LaplacianTest;

OCL_TEST_P(LaplacianTest, Accuracy)
{
    double scale = param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::Laplacian(src_roi, dst_roi, -1, ksize, scale, 10, borderType));
        OCL_ON(cv::Laplacian(usrc_roi, udst_roi, -1, ksize, scale, 10, borderType));

        Near();
    }
}

PARAM_TEST_CASE(Deriv3x3_cols16_rows2_Base, MatType,
                int, // kernel size
                Size, // dx, dy
                BorderType, // border type
                double, // optional parameter
                bool, // roi or not
                int)  // width multiplier
{
    int type, borderType, ksize;
    Size size;
    double param;
    bool useRoi;
    int widthMultiple;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        size = GET_PARAM(2);
        borderType = GET_PARAM(3);
        param = GET_PARAM(4);
        useRoi = GET_PARAM(5);
        widthMultiple = GET_PARAM(6);
    }

    void random_roi()
    {
        size = Size(3, 3);

        Size roiSize = randomSize(size.width, MAX_VALUE, size.height, MAX_VALUE);
        roiSize.width = std::max(size.width + 13, roiSize.width & (~0xf));
        roiSize.height = std::max(size.height + 1, roiSize.height & (~0x1));

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -60, 70);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near()
    {
        Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

typedef Deriv3x3_cols16_rows2_Base Laplacian3_cols16_rows2;

OCL_TEST_P(Laplacian3_cols16_rows2, Accuracy)
{
    double scale = param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::Laplacian(src_roi, dst_roi, -1, ksize, scale, 10, borderType));
        OCL_ON(cv::Laplacian(usrc_roi, udst_roi, -1, ksize, scale, 10, borderType));

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

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::Sobel(src_roi, dst_roi, -1, dx, dy, ksize, scale, /* delta */0, borderType));
        OCL_ON(cv::Sobel(usrc_roi, udst_roi, -1, dx, dy, ksize, scale, /* delta */0, borderType));

        Near();
    }
}

typedef Deriv3x3_cols16_rows2_Base Sobel3x3_cols16_rows2;

OCL_TEST_P(Sobel3x3_cols16_rows2, Mat)
{
    int dx = size.width, dy = size.height;
    double scale = param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::Sobel(src_roi, dst_roi, -1, dx, dy, ksize, scale, /* delta */0, borderType));
        OCL_ON(cv::Sobel(usrc_roi, udst_roi, -1, dx, dy, ksize, scale, /* delta */0, borderType));

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

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::Scharr(src_roi, dst_roi, -1, dx, dy, scale, /* delta */ 0, borderType));
        OCL_ON(cv::Scharr(usrc_roi, udst_roi, -1, dx, dy, scale, /* delta */ 0, borderType));

        Near();
    }
}

typedef Deriv3x3_cols16_rows2_Base Scharr3x3_cols16_rows2;

OCL_TEST_P(Scharr3x3_cols16_rows2, Mat)
{
    int dx = size.width, dy = size.height;
    double scale = param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::Scharr(src_roi, dst_roi, -1, dx, dy, scale, /* delta */ 0, borderType));
        OCL_ON(cv::Scharr(usrc_roi, udst_roi, -1, dx, dy, scale, /* delta */ 0, borderType));

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

typedef FilterTestBase GaussianBlurTest;

OCL_TEST_P(GaussianBlurTest, Mat)
{
    for (int j = 0; j < test_loop_times + 1; j++)
    {
        random_roi();

        double sigma1 = rng.uniform(0.1, 1.0);
        double sigma2 = j % 2 == 0 ? sigma1 : rng.uniform(0.1, 1.0);

        OCL_OFF(cv::GaussianBlur(src_roi, dst_roi, Size(ksize, ksize), sigma1, sigma2, borderType));
        OCL_ON(cv::GaussianBlur(usrc_roi, udst_roi, Size(ksize, ksize), sigma1, sigma2, borderType));

        Near(CV_MAT_DEPTH(type) >= CV_32F ? 1e-3 : 4, CV_MAT_DEPTH(type) >= CV_32F);
    }
}

PARAM_TEST_CASE(GaussianBlur_multicols_Base, MatType,
                int, // kernel size
                Size, // dx, dy
                BorderType, // border type
                double, // optional parameter
                bool, // roi or not
                int)  // width multiplier
{
    int type, borderType, ksize;
    Size size;
    double param;
    bool useRoi;
    int widthMultiple;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        size = GET_PARAM(2);
        borderType = GET_PARAM(3);
        param = GET_PARAM(4);
        useRoi = GET_PARAM(5);
        widthMultiple = GET_PARAM(6);
    }

    void random_roi()
    {
        size = Size(ksize, ksize);

        Size roiSize = randomSize(size.width, MAX_VALUE, size.height, MAX_VALUE);
        if (ksize == 3)
        {
            roiSize.width = std::max((size.width + 15) & 0x10, roiSize.width & (~0xf));
            roiSize.height = std::max(size.height + 1, roiSize.height & (~0x1));
        }
        else if (ksize == 5)
        {
            roiSize.width = std::max((size.width + 3) & 0x4, roiSize.width & (~0x3));
        }

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -60, 70);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near()
    {
        Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

typedef GaussianBlur_multicols_Base GaussianBlur_multicols;

OCL_TEST_P(GaussianBlur_multicols, Mat)
{
    Size kernelSize(ksize, ksize);

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        double sigma1 = rng.uniform(0.1, 1.0);
        double sigma2 = j % 2 == 0 ? sigma1 : rng.uniform(0.1, 1.0);

        OCL_OFF(cv::GaussianBlur(src_roi, dst_roi, Size(ksize, ksize), sigma1, sigma2, borderType));
        OCL_ON(cv::GaussianBlur(usrc_roi, udst_roi, Size(ksize, ksize), sigma1, sigma2, borderType));

        Near(CV_MAT_DEPTH(type) >= CV_32F ? 1e-3 : 4, CV_MAT_DEPTH(type) >= CV_32F);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Erode

typedef FilterTestBase Erode;

OCL_TEST_P(Erode, Mat)
{
    Size kernelSize(ksize, ksize);
    int iterations = (int)param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();
        Mat kernel = ksize==0 ? Mat() : randomMat(kernelSize, CV_8UC1, 0, 2);

        OCL_OFF(cv::erode(src_roi, dst_roi, kernel, Point(-1, -1), iterations) );
        OCL_ON(cv::erode(usrc_roi, udst_roi, kernel, Point(-1, -1), iterations) );

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate

typedef FilterTestBase Dilate;

OCL_TEST_P(Dilate, Mat)
{
    Size kernelSize(ksize, ksize);
    int iterations = (int)param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();
        Mat kernel = ksize==0 ? Mat() : randomMat(kernelSize, CV_8UC1, 0, 2);

        OCL_OFF(cv::dilate(src_roi, dst_roi, kernel, Point(-1, -1), iterations) );
        OCL_ON(cv::dilate(usrc_roi, udst_roi, kernel, Point(-1, -1), iterations) );

        Near();
    }
}

PARAM_TEST_CASE(MorphFilter3x3_cols16_rows2_Base, MatType,
                int, // kernel size
                Size, // dx, dy
                BorderType, // border type
                double, // optional parameter
                bool, // roi or not
                int)  // width multiplier
{
    int type, borderType, ksize;
    Size size;
    double param;
    bool useRoi;
    int widthMultiple;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        size = GET_PARAM(2);
        borderType = GET_PARAM(3);
        param = GET_PARAM(4);
        useRoi = GET_PARAM(5);
        widthMultiple = GET_PARAM(6);
    }

    void random_roi()
    {
        size = Size(3, 3);

        Size roiSize = randomSize(size.width, MAX_VALUE, size.height, MAX_VALUE);
        roiSize.width = std::max(size.width + 13, roiSize.width & (~0xf));
        roiSize.height = std::max(size.height + 1, roiSize.height & (~0x1));

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -60, 70);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near()
    {
        Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

typedef MorphFilter3x3_cols16_rows2_Base MorphFilter3x3_cols16_rows2;

OCL_TEST_P(MorphFilter3x3_cols16_rows2, Mat)
{
    Size kernelSize(ksize, ksize);
    int iterations = (int)param;

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();
        Mat kernel = ksize==0 ? Mat() : randomMat(kernelSize, CV_8UC1, 0, 3);

        OCL_OFF(cv::dilate(src_roi, dst_roi, kernel, Point(-1, -1), iterations) );
        OCL_ON(cv::dilate(usrc_roi, udst_roi, kernel, Point(-1, -1), iterations) );

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// MorphologyEx
IMPLEMENT_PARAM_CLASS(MorphOp, int)
PARAM_TEST_CASE(MorphologyEx, MatType,
                int, // kernel size
                MorphOp, // MORPH_OP
                int, // iterations
                bool)
{
    int type, ksize, op, iterations;
    bool useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        op = GET_PARAM(2);
        iterations = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }

    void random_roi(int minSize = 1)
    {
        if (minSize == 0)
            minSize = ksize;

        Size roiSize = randomSize(minSize, MAX_VALUE);

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -60, 70);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near()
    {
        int depth = CV_MAT_DEPTH(type);
        bool isFP = depth >= CV_32F;

        if (isFP)
            Near(1e-6, true);
        else
            Near(1, false);
    }

    void Near(double threshold, bool relative)
    {
        if (relative)
            OCL_EXPECT_MATS_NEAR_RELATIVE(dst, threshold);
        else
            OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

OCL_TEST_P(MorphologyEx, Mat)
{
    Size kernelSize(ksize, ksize);

    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();
        Mat kernel = randomMat(kernelSize, CV_8UC1, 0, 3);

        OCL_OFF(cv::morphologyEx(src_roi, dst_roi, op, kernel, Point(-1, -1), iterations) );
        OCL_ON(cv::morphologyEx(usrc_roi, udst_roi, op, kernel, Point(-1, -1), iterations) );

        Near();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define FILTER_BORDER_SET_NO_ISOLATED \
    Values((BorderType)BORDER_CONSTANT, (BorderType)BORDER_REPLICATE, (BorderType)BORDER_REFLECT, (BorderType)BORDER_WRAP, (BorderType)BORDER_REFLECT_101/*, \
            (int)BORDER_CONSTANT|BORDER_ISOLATED, (int)BORDER_REPLICATE|BORDER_ISOLATED, \
            (int)BORDER_REFLECT|BORDER_ISOLATED, (int)BORDER_WRAP|BORDER_ISOLATED, \
            (int)BORDER_REFLECT_101|BORDER_ISOLATED*/) // WRAP and ISOLATED are not supported by cv:: version

#define FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED \
    Values((BorderType)BORDER_CONSTANT, (BorderType)BORDER_REPLICATE, (BorderType)BORDER_REFLECT, /*(int)BORDER_WRAP,*/ (BorderType)BORDER_REFLECT_101/*, \
            (int)BORDER_CONSTANT|BORDER_ISOLATED, (int)BORDER_REPLICATE|BORDER_ISOLATED, \
            (int)BORDER_REFLECT|BORDER_ISOLATED, (int)BORDER_WRAP|BORDER_ISOLATED, \
            (int)BORDER_REFLECT_101|BORDER_ISOLATED*/) // WRAP and ISOLATED are not supported by cv:: version

#define FILTER_TYPES Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4)

OCL_INSTANTIATE_TEST_CASE_P(Filter, Bilateral, Combine(
                            Values(CV_8UC1, CV_8UC3),
                            Values(5, 9), // kernel size
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool(),
                            Values(1, 4)));

OCL_INSTANTIATE_TEST_CASE_P(Filter, LaplacianTest, Combine(
                            FILTER_TYPES,
                            Values(1, 3, 5), // kernel size
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(1.0, 0.2, 3.0), // kernel scale
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, Laplacian3_cols16_rows2, Combine(
                            Values((MatType)CV_8UC1),
                            Values(3), // kernel size
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(1.0, 0.2, 3.0), // kernel scale
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, SobelTest, Combine(
                            FILTER_TYPES,
                            Values(3, 5), // kernel size
                            Values(Size(1, 0), Size(1, 1), Size(2, 0), Size(2, 1)), // dx, dy
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, Sobel3x3_cols16_rows2, Combine(
                            Values((MatType)CV_8UC1),
                            Values(3), // kernel size
                            Values(Size(1, 0), Size(1, 1), Size(2, 0), Size(2, 1)), // dx, dy
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, ScharrTest, Combine(
                            FILTER_TYPES,
                            Values(0), // not used
                            Values(Size(0, 1), Size(1, 0)), // dx, dy
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(1.0, 0.2), // kernel scale
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, Scharr3x3_cols16_rows2, Combine(
                            FILTER_TYPES,
                            Values(0), // not used
                            Values(Size(0, 1), Size(1, 0)), // dx, dy
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(1.0, 0.2), // kernel scale
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, GaussianBlurTest, Combine(
                            FILTER_TYPES,
                            Values(3, 5), // kernel size
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, GaussianBlur_multicols, Combine(
                            Values((MatType)CV_8UC1),
                            Values(3, 5), // kernel size
                            Values(Size(0, 0)), // not used
                            FILTER_BORDER_SET_NO_WRAP_NO_ISOLATED,
                            Values(0.0), // not used
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, Erode, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC4),
                            Values(0, 5, 7, 9), // kernel size, 0 means kernel = Mat()
                            Values(Size(0, 0)), //not used
                            Values((BorderType)BORDER_CONSTANT),
                            Values(1.0, 2.0, 3.0, 4.0),
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, Dilate, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC4),
                            Values(0, 3, 5, 7, 9), // kernel size, 0 means kernel = Mat()
                            Values(Size(0, 0)), // not used
                            Values((BorderType)BORDER_CONSTANT),
                            Values(1.0, 2.0, 3.0, 4.0),
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, MorphFilter3x3_cols16_rows2, Combine(
                            Values((MatType)CV_8UC1),
                            Values(0, 3), // kernel size, 0 means kernel = Mat()
                            Values(Size(0, 0)), // not used
                            Values((BorderType)BORDER_CONSTANT),
                            Values(1.0, 2.0, 3.0),
                            Bool(),
                            Values(1))); // not used

OCL_INSTANTIATE_TEST_CASE_P(Filter, MorphologyEx, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(3, 5, 7), // kernel size
                            Values((MorphOp)MORPH_OPEN, (MorphOp)MORPH_CLOSE, (MorphOp)MORPH_GRADIENT, (MorphOp)MORPH_TOPHAT, (MorphOp)MORPH_BLACKHAT), // used as generator of operations
                            Values(1, 2, 3),
                            Bool()));


} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
