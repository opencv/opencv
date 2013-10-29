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

using namespace cvtest;
using namespace testing;
using namespace std;

#define MAX_CHANNELS 4

PARAM_TEST_CASE(MergeTestBase, MatDepth, Channels, bool)
{
    int type;
    int channels;
    bool use_roi;

    //src mat
    cv::Mat mat[MAX_CHANNELS];
    //dst mat
    cv::Mat dst;

    // set up roi
    int roicols, roirows;
    int srcx[MAX_CHANNELS];
    int srcy[MAX_CHANNELS];
    int dstx, dsty;

    //src mat with roi
    cv::Mat mat_roi[MAX_CHANNELS];

    //dst mat with roi
    cv::Mat dst_roi;

    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole;

    //ocl mat with roi
    cv::ocl::oclMat gmat[MAX_CHANNELS];
    cv::ocl::oclMat gdst;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        channels = GET_PARAM(1);
        use_roi = GET_PARAM(2);

        cv::Size size(MWIDTH, MHEIGHT);

        for (int i = 0; i < channels; ++i)
            mat[i] = randomMat(size, CV_MAKETYPE(type, 1), 5, 16, false);
        dst = randomMat(size, CV_MAKETYPE(type, channels), 5, 16, false);
    }

    void random_roi()
    {
        if (use_roi)
        {
            //randomize ROI
            roicols = rng.uniform(1, mat[0].cols);
            roirows = rng.uniform(1, mat[0].rows);

            for (int i = 0; i < channels; ++i)
            {
                srcx[i] = rng.uniform(0, mat[i].cols - roicols);
                srcy[i] = rng.uniform(0, mat[i].rows - roirows);
            }

            dstx = rng.uniform(0, dst.cols  - roicols);
            dsty = rng.uniform(0, dst.rows  - roirows);
        }
        else
        {
            roicols = mat[0].cols;
            roirows = mat[0].rows;
            for (int i = 0; i < channels; ++i)
                srcx[i] = srcy[i] = 0;

            dstx = dsty = 0;
        }

        for (int i = 0; i < channels; ++i)
            mat_roi[i] = mat[i](Rect(srcx[i], srcy[i], roicols, roirows));

        dst_roi = dst(Rect(dstx, dsty, roicols, roirows));

        gdst_whole = dst;
        gdst = gdst_whole(Rect(dstx, dsty, roicols, roirows));

        for (int i = 0; i < channels; ++i)
            gmat[i] = mat_roi[i];
    }
};

struct Merge : MergeTestBase {};

OCL_TEST_P(Merge, Accuracy)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::merge(mat_roi, channels, dst_roi);
        cv::ocl::merge(gmat, channels, gdst);

        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), 0.0);
    }
}

PARAM_TEST_CASE(SplitTestBase, MatType, int, bool)
{
    int type;
    int channels;
    bool use_roi;

    //src mat
    cv::Mat mat;

    //dstmat
    cv::Mat dst[MAX_CHANNELS];

    // set up roi
    int roicols, roirows;
    int srcx, srcy;
    int dstx[MAX_CHANNELS];
    int dsty[MAX_CHANNELS];

    //src mat with roi
    cv::Mat mat_roi;

    //dst mat with roi
    cv::Mat dst_roi[MAX_CHANNELS];

    //ocl dst mat for testing
    cv::ocl::oclMat gdst_whole[MAX_CHANNELS];

    //ocl mat with roi
    cv::ocl::oclMat gmat;
    cv::ocl::oclMat gdst[MAX_CHANNELS];

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        channels = GET_PARAM(1);
        use_roi = GET_PARAM(2);

        cv::Size size(MWIDTH, MHEIGHT);

        mat  = randomMat(size, CV_MAKETYPE(type, channels), 5, 16, false);
        for (int i = 0; i < channels; ++i)
            dst[i] = randomMat(size, CV_MAKETYPE(type, 1), 5, 16, false);    }

    void random_roi()
    {
        if (use_roi)
        {
            //randomize ROI
            roicols = rng.uniform(1, mat.cols);
            roirows = rng.uniform(1, mat.rows);
            srcx    = rng.uniform(0, mat.cols - roicols);
            srcy    = rng.uniform(0, mat.rows - roirows);

            for (int i = 0; i < channels; ++i)
            {
                dstx[i] = rng.uniform(0, dst[i].cols  - roicols);
                dsty[i] = rng.uniform(0, dst[i].rows  - roirows);
            }
        }
        else
        {
            roicols = mat.cols;
            roirows = mat.rows;
            srcx = srcy = 0;

            for (int i = 0; i < channels; ++i)
                dstx[i] = dsty[i] = 0;
        }

        mat_roi = mat(Rect(srcx, srcy, roicols, roirows));

        for (int i = 0; i < channels; ++i)
            dst_roi[i] = dst[i](Rect(dstx[i], dsty[i], roicols, roirows));

        for (int i = 0; i < channels; ++i)
        {
            gdst_whole[i] = dst[i];
            gdst[i] = gdst_whole[i](Rect(dstx[i], dsty[i], roicols, roirows));
        }

        gmat = mat_roi;
    }
};

struct Split : SplitTestBase {};

OCL_TEST_P(Split, Accuracy)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::split(mat_roi, dst_roi);
        cv::ocl::split(gmat, gdst);

        for (int i = 0; i < channels; ++i)
            EXPECT_MAT_NEAR(dst[i], Mat(gdst_whole[i]), 0.0);
    }
}


INSTANTIATE_TEST_CASE_P(SplitMerge, Merge, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F), Values(1, 2, 3, 4), Bool()));


INSTANTIATE_TEST_CASE_P(SplitMerge, Split , Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F), Values(1, 2, 3, 4), Bool()));


#endif // HAVE_OPENCL
