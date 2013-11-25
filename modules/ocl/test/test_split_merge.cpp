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

    cv::Mat src, src_roi;
    cv::Mat dst[MAX_CHANNELS], dst_roi[MAX_CHANNELS];

    cv::ocl::oclMat gsrc_whole, gsrc_roi;
    cv::ocl::oclMat gdst_whole[MAX_CHANNELS], gdst_roi[MAX_CHANNELS];

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        channels = GET_PARAM(1);
        use_roi = GET_PARAM(2);
    }

    void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, CV_MAKETYPE(type, channels), 0, 256);
        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);

        for (int i = 0; i < channels; ++i)
        {
            Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
            randomSubMat(dst[i], dst_roi[i], roiSize, dstBorder, CV_MAKETYPE(type, 1), 5, 16);
            generateOclMat(gdst_whole[i], gdst_roi[i], dst[i], roiSize, dstBorder);
        }
    }
};

struct Split : SplitTestBase {};

#ifdef ANDROID
// NOTE: The test fail on Android is the top of the iceberg only
// The real fail reason is memory access vialation somewhere else
OCL_TEST_P(Split, DISABLED_Accuracy)
#else
OCL_TEST_P(Split, Accuracy)
#endif
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::split(src_roi, dst_roi);
        cv::ocl::split(gsrc_roi, gdst_roi);

        for (int i = 0; i < channels; ++i)
        {
            EXPECT_MAT_NEAR(dst[i], gdst_whole[i], 0.0);
            EXPECT_MAT_NEAR(dst_roi[i], gdst_roi[i], 0.0);
        }
    }
}


INSTANTIATE_TEST_CASE_P(SplitMerge, Merge, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F), Values(1, 2, 3, 4), Bool()));


INSTANTIATE_TEST_CASE_P(SplitMerge, Split , Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F), Values(1, 2, 3, 4), Bool()));


#endif // HAVE_OPENCL
