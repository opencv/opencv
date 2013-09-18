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

using namespace cvtest;
using namespace testing;
using namespace std;


PARAM_TEST_CASE(FilterTestBase,
                MatType,
                cv::Size, // kernel size
                cv::Size, // dx,dy
                int       // border type, or iteration
                )
{
    //src mat
    cv::Mat mat1;
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int src1x;
    int src1y;
    int dstx;
    int dsty;

    //src mat with roi
    cv::Mat mat1_roi;
    cv::Mat dst_roi;

    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat1;
    cv::ocl::oclMat gdst;

    void random_roi()
    {
#ifdef RANDOMROI
        //randomize ROI
        cv::RNG &rng = TS::ptr()->get_rng();
        roicols = rng.uniform(2, mat1.cols);
        roirows = rng.uniform(2, mat1.rows);
        src1x   = rng.uniform(0, mat1.cols - roicols);
        src1y   = rng.uniform(0, mat1.rows - roirows);
        dstx    = rng.uniform(0, dst.cols  - roicols);
        dsty    = rng.uniform(0, dst.rows  - roirows);
#else
        roicols = mat1.cols;
        roirows = mat1.rows;
        src1x = 0;
        src1y = 0;
        dstx = 0;
        dsty = 0;
#endif

        mat1_roi = mat1(Rect(src1x, src1y, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        gmat1 = mat1_roi;
    }

    void Init(int mat_type)
    {
        cv::Size size(MWIDTH, MHEIGHT);
        mat1 = randomMat(size, mat_type, 5, 16);
        dst  = randomMat(size, mat_type, 5, 16);
    }

    void Near(double threshold)
    {
        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), threshold);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// blur
struct Blur : FilterTestBase
{
    int type;
    cv::Size ksize;
    int bordertype;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        bordertype = GET_PARAM(3);
        Init(type);
    }
};

TEST_P(Blur, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::blur(mat1_roi, dst_roi, ksize, Point(-1, -1), bordertype);
        cv::ocl::blur(gmat1, gdst, ksize, Point(-1, -1), bordertype);
        Near(1.0);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//Laplacian
struct Laplacian : FilterTestBase
{
    int type;
    cv::Size ksize;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        Init(type);
    }
};

TEST_P(Laplacian, Accuracy)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::Laplacian(mat1_roi, dst_roi, -1, ksize.width, 1);
        cv::ocl::Laplacian(gmat1, gdst, -1, ksize.width, 1);
        Near(1e-5);
    }
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// erode & dilate
struct ErodeDilate : FilterTestBase
{
    int type;
    int iterations;

    //erode or dilate kernel
    cv::Mat kernel;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        iterations = GET_PARAM(3);
        Init(type);
        //		rng.fill(kernel, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(3));
        kernel = randomMat(Size(3, 3), CV_8UC1, 0, 3);
    }

};

TEST_P(ErodeDilate, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::erode(mat1_roi, dst_roi, kernel, Point(-1, -1), iterations);
        cv::ocl::erode(gmat1, gdst, kernel, Point(-1, -1), iterations);
        Near(1e-5);
    }
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::dilate(mat1_roi, dst_roi, kernel, Point(-1, -1), iterations);
        cv::ocl::dilate(gmat1, gdst, kernel, Point(-1, -1), iterations);
        Near(1e-5);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel
struct Sobel : FilterTestBase
{
    int type;
    int dx, dy, ksize, bordertype;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        Size s = GET_PARAM(1);
        ksize = s.width;
        s = GET_PARAM(2);
        dx = s.width;
        dy = s.height;
        bordertype = GET_PARAM(3);
        Init(type);
    }
};

TEST_P(Sobel, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::Sobel(mat1_roi, dst_roi, -1, dx, dy, ksize, /*scale*/0.00001,/*delta*/0, bordertype);
        cv::ocl::Sobel(gmat1, gdst, -1, dx, dy, ksize,/*scale*/0.00001,/*delta*/0, bordertype);
        Near(1);
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr
struct Scharr : FilterTestBase
{
    int type;
    int dx, dy, bordertype;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        Size s = GET_PARAM(2);
        dx = s.width;
        dy = s.height;
        bordertype = GET_PARAM(3);
        Init(type);
    }
};

TEST_P(Scharr, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::Scharr(mat1_roi, dst_roi, -1, dx, dy, /*scale*/1,/*delta*/0, bordertype);
        cv::ocl::Scharr(gmat1, gdst, -1, dx, dy,/*scale*/1,/*delta*/0, bordertype);
        Near(1);
    }

}


/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur
struct GaussianBlur : FilterTestBase
{
    int type;
    cv::Size ksize;
    int bordertype;
    double sigma1, sigma2;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        bordertype = GET_PARAM(3);
        Init(type);
        cv::RNG &rng = TS::ptr()->get_rng();
        sigma1 = rng.uniform(0.1, 1.0);
        sigma2 = rng.uniform(0.1, 1.0);
    }
};

TEST_P(GaussianBlur, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::GaussianBlur(mat1_roi, dst_roi, ksize, sigma1, sigma2, bordertype);
        cv::ocl::GaussianBlur(gmat1, gdst, ksize, sigma1, sigma2, bordertype);
        Near(1);
    }

}



////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D
struct Filter2D : FilterTestBase
{
    int type;
    cv::Size ksize;
    int bordertype;
    Point anchor;
    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        bordertype = GET_PARAM(3);
        Init(type);
        anchor = Point(-1,-1);
    }
};

TEST_P(Filter2D, Mat)
{
    cv::Mat kernel = randomMat(cv::Size(ksize.width, ksize.height), CV_32FC1, 0.0, 1.0);
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::filter2D(mat1_roi, dst_roi, -1, kernel, anchor, 0.0, bordertype);
        cv::ocl::filter2D(gmat1, gdst, -1, kernel, anchor, bordertype);
        Near(1);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral
struct Bilateral : FilterTestBase
{
    int type;
    cv::Size ksize;
    int bordertype;
    double sigmacolor, sigmaspace;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        bordertype = GET_PARAM(3);
        Init(type);
        cv::RNG &rng = TS::ptr()->get_rng();
        sigmacolor = rng.uniform(20, 100);
        sigmaspace = rng.uniform(10, 40);
    }
};

TEST_P(Bilateral, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::bilateralFilter(mat1_roi, dst_roi, ksize.width, sigmacolor, sigmaspace, bordertype);
        cv::ocl::bilateralFilter(gmat1, gdst, ksize.width, sigmacolor, sigmaspace, bordertype);
        Near(1);
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// AdaptiveBilateral
struct AdaptiveBilateral : FilterTestBase
{
    int type;
    cv::Size ksize;
    int bordertype;
    Point anchor;
    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);
        bordertype = GET_PARAM(3);
        Init(type);
        anchor = Point(-1,-1);
    }
};

TEST_P(AdaptiveBilateral, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::adaptiveBilateralFilter(mat1_roi, dst_roi, ksize, 5, anchor, bordertype);
        cv::ocl::adaptiveBilateralFilter(gmat1, gdst, ksize, 5, anchor, bordertype);
        Near(1);
    }

}

INSTANTIATE_TEST_CASE_P(Filter, Blur, Combine(
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                        Values(cv::Size(3, 3), cv::Size(5, 5), cv::Size(7, 7)),
                        Values(Size(0, 0)), //not use
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE, (MatType)cv::BORDER_REFLECT, (MatType)cv::BORDER_REFLECT_101)));


INSTANTIATE_TEST_CASE_P(Filter, Laplacian, Combine(
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values(Size(3, 3)),
                        Values(Size(0, 0)), //not use
                        Values(0)));        //not use

INSTANTIATE_TEST_CASE_P(Filter, ErodeDilate, Combine(
                        Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                        Values(Size(0, 0)), //not use
                        Values(Size(0, 0)), //not use
                        Values(1)));


INSTANTIATE_TEST_CASE_P(Filter, Sobel, Combine(
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values(Size(3, 3), Size(5, 5)),
                        Values(Size(1, 0), Size(1, 1), Size(2, 0), Size(2, 1)),
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE)));


INSTANTIATE_TEST_CASE_P(Filter, Scharr, Combine(
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                        Values(Size(0, 0)), //not use
                        Values(Size(0, 1), Size(1, 0)),
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE)));

INSTANTIATE_TEST_CASE_P(Filter, GaussianBlur, Combine(
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC4),
                        Values(Size(3, 3), Size(5, 5)),
                        Values(Size(0, 0)), //not use
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE)));



INSTANTIATE_TEST_CASE_P(Filter, Filter2D, testing::Combine(
                        Values(CV_8UC1, CV_32FC1, CV_32FC4),
                        Values(Size(3, 3), Size(15, 15), Size(25, 25)),
                        Values(Size(0, 0)), //not use
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REFLECT101, (MatType)cv::BORDER_REPLICATE, (MatType)cv::BORDER_REFLECT)));

INSTANTIATE_TEST_CASE_P(Filter, Bilateral, Combine(
                        Values(CV_8UC1, CV_8UC3),
                        Values(Size(5, 5), Size(9, 9)),
                        Values(Size(0, 0)), //not use
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE,
                               (MatType)cv::BORDER_REFLECT, (MatType)cv::BORDER_WRAP, (MatType)cv::BORDER_REFLECT_101)));

INSTANTIATE_TEST_CASE_P(Filter, AdaptiveBilateral, Combine(
                        Values(CV_8UC1, CV_8UC3),
                        Values(Size(5, 5), Size(9, 9)),
                        Values(Size(0, 0)), //not use
                        Values((MatType)cv::BORDER_CONSTANT, (MatType)cv::BORDER_REPLICATE,
                               (MatType)cv::BORDER_REFLECT,  (MatType)cv::BORDER_REFLECT_101)));
#endif // HAVE_OPENCL
