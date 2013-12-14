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

using namespace cv;
using namespace testing;
using namespace std;

////////////////////////////////converto/////////////////////////////////////////////////

PARAM_TEST_CASE(MatrixTestBase, MatDepth, MatDepth, int, bool)
{
    int src_depth, cn, dstType;
    bool use_roi;

    Mat src, dst, src_roi, dst_roi;
    ocl::oclMat gdst, gsrc, gdst_roi, gsrc_roi;

    virtual void SetUp()
    {
        src_depth = GET_PARAM(0);
        cn = GET_PARAM(2);
        dstType = CV_MAKE_TYPE(GET_PARAM(1), cn);

        use_roi = GET_PARAM(3);
    }

    virtual void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, CV_MAKE_TYPE(src_depth, cn), -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        generateOclMat(gsrc, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst, gdst_roi, dst, roiSize, dstBorder);
    }
};

typedef MatrixTestBase ConvertTo;

OCL_TEST_P(ConvertTo, Accuracy)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.convertTo(dst_roi, dstType);
        gsrc_roi.convertTo(gdst_roi, dstType);

        EXPECT_MAT_NEAR(dst, Mat(gdst), src_depth == CV_64F ? 1.0 : 0.0);
        EXPECT_MAT_NEAR(dst_roi, Mat(gdst_roi), src_depth == CV_64F ? 1.0 : 0.0);
    }
}

///////////////////////////////////////////copyto/////////////////////////////////////////////////////////////

struct CopyTo :
        public MatrixTestBase
{
    Mat mask, mask_roi;
    ocl::oclMat gmask, gmask_roi;

    virtual void random_roi()
    {
        int type = CV_MAKE_TYPE(src_depth, cn);
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 5, 16);

        Border maskBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(mask, mask_roi, roiSize, maskBorder, CV_8UC1, 5, 16);

        generateOclMat(gsrc, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst, gdst_roi, dst, roiSize, dstBorder);
        generateOclMat(gmask, gmask_roi, mask, roiSize, maskBorder);
    }
};

OCL_TEST_P(CopyTo, Without_mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.copyTo(dst_roi);
        gsrc_roi.copyTo(gdst_roi);

        EXPECT_MAT_NEAR(dst, Mat(gdst), 0.0);
        EXPECT_MAT_NEAR(dst_roi, Mat(gdst_roi), 0.0);
    }
}

OCL_TEST_P(CopyTo, With_mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        src_roi.copyTo(dst_roi, mask_roi);
        gsrc_roi.copyTo(gdst_roi, gmask_roi);

        EXPECT_MAT_NEAR(dst, Mat(gdst), 0.0);
        EXPECT_MAT_NEAR(dst_roi, Mat(gdst_roi), 0.0);
    }
}

/////////////////////////////////////////// setTo /////////////////////////////////////////////////////////////

typedef CopyTo SetTo;

OCL_TEST_P(SetTo, Without_mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        Scalar scalar = randomScalar(-MAX_VALUE, MAX_VALUE);

        src_roi.setTo(scalar);
        gsrc_roi.setTo(scalar);

        EXPECT_MAT_NEAR(dst, Mat(gdst), 0.0);
        EXPECT_MAT_NEAR(dst_roi, Mat(gdst_roi), 0.0);;
    }
}

OCL_TEST_P(SetTo, With_mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        Scalar scalar = randomScalar(-MAX_VALUE, MAX_VALUE);

        src_roi.setTo(scalar, mask_roi);
        gsrc_roi.setTo(scalar, gmask_roi);

        EXPECT_MAT_NEAR(src, Mat(gsrc), 1.);
        EXPECT_MAT_NEAR(src_roi, Mat(gsrc_roi), 1.);
    }
}

// convertC3C4

PARAM_TEST_CASE(convertC3C4, MatDepth, bool)
{
    int depth;
    bool use_roi;

    Mat src, src_roi;
    ocl::oclMat gsrc, gsrc_roi;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        use_roi = GET_PARAM(1);
    }

    void random_roi()
    {
        int type = CV_MAKE_TYPE(depth, 3);
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);
        generateOclMat(gsrc, gsrc_roi, src, roiSize, srcBorder);
    }
};

OCL_TEST_P(convertC3C4, Accuracy)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        gsrc_roi = src_roi;

        EXPECT_MAT_NEAR(src_roi, Mat(gsrc_roi), 0.0);
        EXPECT_MAT_NEAR(src, Mat(gsrc), 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(MatrixOperation, ConvertTo, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            testing::Range(1, 5), Bool()));

INSTANTIATE_TEST_CASE_P(MatrixOperation, CopyTo, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Values(MatDepth(0)), // not used
                            testing::Range(1, 5), Bool()));

INSTANTIATE_TEST_CASE_P(MatrixOperation, SetTo, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Values((MatDepth)0), // not used
                            testing::Range(1, 5), Bool()));

INSTANTIATE_TEST_CASE_P(MatrixOperation, convertC3C4, Combine(
                            Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F),
                            Bool()));
#endif
