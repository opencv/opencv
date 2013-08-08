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
//    Jia Haipeng, jiahaipeng95@gmail.com
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

////////////////////////////////converto/////////////////////////////////////////////////
PARAM_TEST_CASE(ConvertToTestBase, MatType, MatType)
{
    int type;
    int dst_type;

    //src mat
    cv::Mat mat;
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int dstx;
    int dsty;

    //src mat with roi
    cv::Mat mat_roi;
    cv::Mat dst_roi;

    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type     = GET_PARAM(0);
        dst_type = GET_PARAM(1);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
    }

    void random_roi()
    {
#ifdef RANDOMROI
        //randomize ROI
        cv::RNG &rng = TS::ptr()->get_rng();
        roicols = rng.uniform(1, mat.cols);
        roirows = rng.uniform(1, mat.rows);
        srcx   = rng.uniform(0, mat.cols - roicols);
        srcy   = rng.uniform(0, mat.rows - roirows);
        dstx    = rng.uniform(0, dst.cols  - roicols);
        dsty    = rng.uniform(0, dst.rows  - roirows);
#else
        roicols = mat.cols;
        roirows = mat.rows;
        srcx = 0;
        srcy = 0;
        dstx = 0;
        dsty = 0;
#endif

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        gmat = mat_roi;
    }
};


struct ConvertTo : ConvertToTestBase {};

TEST_P(ConvertTo, Accuracy)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        mat_roi.convertTo(dst_roi, dst_type);
        gmat.convertTo(gdst, dst_type);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), 0.0);
    }
}




///////////////////////////////////////////copyto/////////////////////////////////////////////////////////////

PARAM_TEST_CASE(CopyToTestBase, MatType, bool)
{
    int type;

    cv::Mat mat;
    cv::Mat mask;
    cv::Mat dst;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int dstx;
    int dsty;
    int maskx;
    int masky;

    //src mat with roi
    cv::Mat mat_roi;
    cv::Mat mask_roi;
    cv::Mat dst_roi;

    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gdst;
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        type = GET_PARAM(0);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat = randomMat(rng, size, type, 5, 16, false);
        dst  = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);

    }

    void random_roi()
    {
#ifdef RANDOMROI
        //randomize ROI
        cv::RNG &rng = TS::ptr()->get_rng();
        roicols = rng.uniform(1, mat.cols);
        roirows = rng.uniform(1, mat.rows);
        srcx   = rng.uniform(0, mat.cols - roicols);
        srcy   = rng.uniform(0, mat.rows - roirows);
        dstx    = rng.uniform(0, dst.cols  - roicols);
        dsty    = rng.uniform(0, dst.rows  - roirows);
        maskx   = rng.uniform(0, mask.cols - roicols);
        masky   = rng.uniform(0, mask.rows - roirows);
#else
        roicols = mat.cols;
        roirows = mat.rows;
        srcx = 0;
        srcy = 0;
        dstx = 0;
        dsty = 0;
        maskx = 0;
        masky = 0;
#endif

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));
        dst_roi  = dst(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        gmat = mat_roi;
        gmask = mask_roi;
    }
};

struct CopyTo : CopyToTestBase {};

TEST_P(CopyTo, Without_mask)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        mat_roi.copyTo(dst_roi);
        gmat.copyTo(gdst);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), 0.0);
    }
}

TEST_P(CopyTo, With_mask)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        mat_roi.copyTo(dst_roi, mask_roi);
        gmat.copyTo(gdst, gmask);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), 0.0);
    }
}




///////////////////////////////////////////copyto/////////////////////////////////////////////////////////////

PARAM_TEST_CASE(SetToTestBase, MatType, bool)
{
    int type;
    cv::Scalar val;

    cv::Mat mat;
    cv::Mat mask;

    // set up roi
    int roicols;
    int roirows;
    int srcx;
    int srcy;
    int maskx;
    int masky;

    //src mat with roi
    cv::Mat mat_roi;
    cv::Mat mask_roi;

    //ocl dst mat for testing
    cv::ocl::oclMat gmat_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        type = GET_PARAM(0);

        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Size size(MWIDTH, MHEIGHT);

        mat = randomMat(rng, size, type, 5, 16, false);
        mask = randomMat(rng, size, CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);
        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));

    }

    void random_roi()
    {
#ifdef RANDOMROI
        //randomize ROI
        cv::RNG &rng = TS::ptr()->get_rng();
        roicols = rng.uniform(1, mat.cols);
        roirows = rng.uniform(1, mat.rows);
        srcx   = rng.uniform(0, mat.cols - roicols);
        srcy   = rng.uniform(0, mat.rows - roirows);
        maskx   = rng.uniform(0, mask.cols - roicols);
        masky   = rng.uniform(0, mask.rows - roirows);
#else
        roicols = mat.cols;
        roirows = mat.rows;
        srcx = 0;
        srcy = 0;
        maskx = 0;
        masky = 0;
#endif

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));

        gmat_whole = mat;
        gmat = gmat_whole(Rect(srcx, srcy, roicols, roirows));

        gmask = mask_roi;
    }
};

struct SetTo : SetToTestBase {};

TEST_P(SetTo, Without_mask)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        mat_roi.setTo(val);
        gmat.setTo(val);

        EXPECT_MAT_NEAR(mat, Mat(gmat_whole), 1.);
    }
}

TEST_P(SetTo, With_mask)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        mat_roi.setTo(val, mask_roi);
        gmat.setTo(val, gmask);

        EXPECT_MAT_NEAR(mat, Mat(gmat_whole), 1.);
    }
}

//convertC3C4
PARAM_TEST_CASE(convertC3C4, MatType, cv::Size)
{
    int type;
    cv::Size ksize;

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

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        ksize = GET_PARAM(1);

    }

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

};

TEST_P(convertC3C4, Accuracy)
{
    cv::RNG &rng = TS::ptr()->get_rng();
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        //random_roi();
        int width = rng.uniform(2, MWIDTH);
        int height = rng.uniform(2, MHEIGHT);
        cv::Size size(width, height);

        mat1 = randomMat(rng, size, type, 0, 40, false);
        gmat1 = mat1;

        EXPECT_MAT_NEAR(mat1, Mat(gmat1), 0.0);
    }

}

INSTANTIATE_TEST_CASE_P(MatrixOperation, ConvertTo, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32SC1, CV_32SC4, CV_32FC1, CV_32FC4),
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32SC1, CV_32SC4, CV_32FC1, CV_32FC4)));

INSTANTIATE_TEST_CASE_P(MatrixOperation, CopyTo, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(MatrixOperation, SetTo, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32SC1, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(false))); // Values(false) is the reserved parameter

INSTANTIATE_TEST_CASE_P(MatrixOperation, convertC3C4, Combine(
                            Values(CV_8UC3,  CV_32SC3,  CV_32FC3),
                            Values(cv::Size())));
#endif
