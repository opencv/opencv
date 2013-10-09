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

PARAM_TEST_CASE(ConvertToTestBase, MatType, MatType, int, bool)
{
    int src_depth, dst_depth;
    int cn, dst_type;
    bool use_roi;

    // src mat
    cv::Mat mat;
    cv::Mat dst;

    // set up roi
    int roicols, roirows;
    int srcx, srcy;
    int dstx, dsty;

    // src mat with roi
    cv::Mat mat_roi;
    cv::Mat dst_roi;

    // ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    // ocl mat with roi
    cv::ocl::oclMat gsrc;
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        src_depth = GET_PARAM(0);
        dst_depth = GET_PARAM(1);
        cn = GET_PARAM(2);
        int src_type = CV_MAKE_TYPE(src_depth, cn);
        dst_type = CV_MAKE_TYPE(dst_depth, cn);

        use_roi = GET_PARAM(3);

        cv::RNG &rng = TS::ptr()->get_rng();

        mat = randomMat(rng, randomSize(MIN_VALUE, MAX_VALUE), src_type, 5, 136, false);
        dst = randomMat(rng, use_roi ? randomSize(MIN_VALUE, MAX_VALUE) : mat.size(), dst_type, 5, 136, false);
    }

    void random_roi()
    {
        if (use_roi)
        {
            // randomize ROI
            cv::RNG &rng = TS::ptr()->get_rng();
            roicols = rng.uniform(1, MIN_VALUE);
            roirows = rng.uniform(1, MIN_VALUE);
            srcx = rng.uniform(0, mat.cols - roicols);
            srcy = rng.uniform(0, mat.rows - roirows);
            dstx = rng.uniform(0, dst.cols - roicols);
            dsty = rng.uniform(0, dst.rows - roirows);
        }
        else
        {
            roicols = mat.cols;
            roirows = mat.rows;
            srcx = srcy = 0;
            dstx = dsty = 0;
        }

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));
        dst_roi = dst(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        gsrc = mat_roi;
    }
};

typedef ConvertToTestBase ConvertTo;

TEST_P(ConvertTo, Accuracy)
{
    if((src_depth == CV_64F || dst_depth == CV_64F) &&
            !cv::ocl::Context::getContext()->supportsFeature(cv::ocl::FEATURE_CL_DOUBLE))
    {
        return; // returns silently
    }
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        mat_roi.convertTo(dst_roi, dst_type);
        gsrc.convertTo(gdst, dst_type);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), src_depth == CV_64F ? 1.0 : 0.0);
        EXPECT_MAT_NEAR(dst_roi, Mat(gdst), src_depth == CV_64F ? 1.0 : 0.0);
    }
}

///////////////////////////////////////////copyto/////////////////////////////////////////////////////////////

PARAM_TEST_CASE(CopyToTestBase, MatType, int, bool)
{
    bool use_roi;

    cv::Mat src, mask, dst;

    // set up roi
    int roicols,roirows;
    int srcx, srcy;
    int dstx, dsty;
    int maskx,masky;

    // src mat with roi
    cv::Mat src_roi;
    cv::Mat mask_roi;
    cv::Mat dst_roi;

    // ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    // ocl mat with roi
    cv::ocl::oclMat gsrc, gdst, gmask;

    virtual void SetUp()
    {
        int type = CV_MAKETYPE(GET_PARAM(0), GET_PARAM(1));
        use_roi = GET_PARAM(2);

        cv::RNG &rng = TS::ptr()->get_rng();

        src = randomMat(rng, randomSize(MIN_VALUE, MAX_VALUE), type, 5, 16, false);
        dst = randomMat(rng, use_roi ? randomSize(MIN_VALUE, MAX_VALUE) : src.size(), type, 5, 16, false);
        mask = randomMat(rng, use_roi ? randomSize(MIN_VALUE, MAX_VALUE) : src.size(), CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);
    }

    void random_roi()
    {
        if (use_roi)
        {
            // randomize ROI
            cv::RNG &rng = TS::ptr()->get_rng();
            roicols = rng.uniform(1, MIN_VALUE);
            roirows = rng.uniform(1, MIN_VALUE);
            srcx = rng.uniform(0, src.cols - roicols);
            srcy = rng.uniform(0, src.rows - roirows);
            dstx = rng.uniform(0, dst.cols - roicols);
            dsty = rng.uniform(0, dst.rows - roirows);
            maskx = rng.uniform(0, mask.cols - roicols);
            masky = rng.uniform(0, mask.rows - roirows);
        }
        else
        {
            roicols = src.cols;
            roirows = src.rows;
            srcx = srcy = 0;
            dstx = dsty = 0;
            maskx = masky = 0;
        }

        src_roi = src(Rect(srcx, srcy, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));
        dst_roi = dst(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        gsrc = src_roi;
        gmask = mask_roi;
    }
};

typedef CopyToTestBase CopyTo;

TEST_P(CopyTo, Without_mask)
{
    if((src.depth() == CV_64F) &&
            !cv::ocl::Context::getContext()->supportsFeature(cv::ocl::FEATURE_CL_DOUBLE))
    {
        return; // returns silently
    }
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.copyTo(dst_roi);
        gsrc.copyTo(gdst);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), 0.0);
    }
}

TEST_P(CopyTo, With_mask)
{
    if(src.depth() == CV_64F &&
        !cv::ocl::Context::getContext()->supportsFeature(cv::ocl::FEATURE_CL_DOUBLE))
    {
        return; // returns silently
    }
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.copyTo(dst_roi, mask_roi);
        gsrc.copyTo(gdst, gmask);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), 0.0);
    }
}

/////////////////////////////////////////// setTo /////////////////////////////////////////////////////////////

PARAM_TEST_CASE(SetToTestBase, MatType, int, bool)
{
    int depth, channels;
    bool use_roi;

    cv::Scalar val;

    cv::Mat src;
    cv::Mat mask;

    // set up roi
    int roicols, roirows;
    int srcx, srcy;
    int maskx, masky;

    // src mat with roi
    cv::Mat src_roi;
    cv::Mat mask_roi;

    // ocl dst mat for testing
    cv::ocl::oclMat gsrc_whole;

    // ocl mat with roi
    cv::ocl::oclMat gsrc;
    cv::ocl::oclMat gmask;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        channels = GET_PARAM(1);
        use_roi = GET_PARAM(2);

        cv::RNG &rng = TS::ptr()->get_rng();
        int type = CV_MAKE_TYPE(depth, channels);

        src = randomMat(rng, randomSize(MIN_VALUE, MAX_VALUE), type, 5, 16, false);
        mask = randomMat(rng, use_roi ? randomSize(MIN_VALUE, MAX_VALUE) : src.size(), CV_8UC1, 0, 2,  false);

        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);
        val = cv::Scalar(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0),
                         rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
    }

    void random_roi()
    {
        if (use_roi)
        {
            // randomize ROI
            cv::RNG &rng = TS::ptr()->get_rng();
            roicols = rng.uniform(1, MIN_VALUE);
            roirows = rng.uniform(1, MIN_VALUE);
            srcx = rng.uniform(0, src.cols - roicols);
            srcy = rng.uniform(0, src.rows - roirows);
            maskx = rng.uniform(0, mask.cols - roicols);
            masky = rng.uniform(0, mask.rows - roirows);
        }
        else
        {
            roicols = src.cols;
            roirows = src.rows;
            srcx = srcy = 0;
            maskx = masky = 0;
        }

        src_roi = src(Rect(srcx, srcy, roicols, roirows));
        mask_roi = mask(Rect(maskx, masky, roicols, roirows));

        gsrc_whole = src;
        gsrc = gsrc_whole(Rect(srcx, srcy, roicols, roirows));

        gmask = mask_roi;
    }
};

typedef SetToTestBase SetTo;

TEST_P(SetTo, Without_mask)
{
    if(depth == CV_64F &&
            !cv::ocl::Context::getContext()->supportsFeature(cv::ocl::FEATURE_CL_DOUBLE))
    {
        return; // returns silently
    }
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.setTo(val);
        gsrc.setTo(val);

        EXPECT_MAT_NEAR(src, Mat(gsrc_whole), 1.);
    }
}

TEST_P(SetTo, With_mask)
{
    if(depth == CV_64F &&
            !cv::ocl::Context::getContext()->supportsFeature(cv::ocl::FEATURE_CL_DOUBLE))
    {
        return; // returns silently
    }
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.setTo(val, mask_roi);
        gsrc.setTo(val, gmask);

        EXPECT_MAT_NEAR(src, Mat(gsrc_whole), 1.);
    }
}

// convertC3C4

PARAM_TEST_CASE(convertC3C4, MatType, bool)
{
    int depth;
    bool use_roi;

    //src mat
    cv::Mat src;

    // set up roi
    int roicols, roirows;
    int srcx, srcy;

    //src mat with roi
    cv::Mat src_roi;

    //ocl mat with roi
    cv::ocl::oclMat gsrc_roi;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        use_roi = GET_PARAM(1);
        int type = CV_MAKE_TYPE(depth, 3);

        cv::RNG &rng = TS::ptr()->get_rng();
        src = randomMat(rng, randomSize(1, MAX_VALUE), type, 0, 40, false);
    }

    void random_roi()
    {
        if (use_roi)
        {
            //randomize ROI
            cv::RNG &rng = TS::ptr()->get_rng();
            roicols = rng.uniform(1, src.cols);
            roirows = rng.uniform(1, src.rows);
            srcx = rng.uniform(0, src.cols - roicols);
            srcy = rng.uniform(0, src.rows - roirows);
        }
        else
        {
            roicols = src.cols;
            roirows = src.rows;
            srcx = srcy = 0;
        }

        src_roi = src(Rect(srcx, srcy, roicols, roirows));
    }
};

TEST_P(convertC3C4, Accuracy)
{
    if(depth == CV_64F &&
        !cv::ocl::Context::getContext()->supportsFeature(cv::ocl::FEATURE_CL_DOUBLE))
    {
        return; // returns silently
    }
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        gsrc_roi = src_roi;

        EXPECT_MAT_NEAR(src_roi, Mat(gsrc_roi), 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(MatrixOperation, ConvertTo, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Range(1, 5), Bool()));

INSTANTIATE_TEST_CASE_P(MatrixOperation, CopyTo, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            testing::Range(1, 5), Bool()));

INSTANTIATE_TEST_CASE_P(MatrixOperation, SetTo, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            testing::Range(1, 5), Bool()));

INSTANTIATE_TEST_CASE_P(MatrixOperation, convertC3C4, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Bool()));
#endif
