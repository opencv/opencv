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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Nathan, liujun@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(AccumulateBase, std::pair<MatDepth, MatDepth>, Channels, bool)
{
    int sdepth, ddepth, channels;
    bool useRoi;
    double alpha;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_INPUT_PARAMETER(mask);
    TEST_DECLARE_INPUT_PARAMETER(src2);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        const std::pair<MatDepth, MatDepth> depths = GET_PARAM(0);
        sdepth = depths.first, ddepth = depths.second;
        channels = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }

    void random_roi()
    {
        const int stype = CV_MAKE_TYPE(sdepth, channels),
                dtype = CV_MAKE_TYPE(ddepth, channels);

        Size roiSize = randomSize(1, 10);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, stype, -MAX_VALUE, MAX_VALUE);

        Border maskBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(mask, mask_roi, roiSize, maskBorder, CV_8UC1, -MAX_VALUE, MAX_VALUE);
        threshold(mask, mask, 80, 255, THRESH_BINARY);

        Border src2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, stype, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dtype, -MAX_VALUE, MAX_VALUE);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_INPUT_PARAMETER(mask);
        UMAT_UPLOAD_INPUT_PARAMETER(src2);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);

        alpha = randomDouble(-5, 5);
    }
};

/////////////////////////////////// Accumulate ///////////////////////////////////

typedef AccumulateBase Accumulate;

OCL_TEST_P(Accumulate, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulate(src_roi, dst_roi));
        OCL_ON(cv::accumulate(usrc_roi, udst_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-6);
    }
}

OCL_TEST_P(Accumulate, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulate(src_roi, dst_roi, mask_roi));
        OCL_ON(cv::accumulate(usrc_roi, udst_roi, umask_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-6);
    }
}

/////////////////////////////////// AccumulateSquare ///////////////////////////////////

typedef AccumulateBase AccumulateSquare;

OCL_TEST_P(AccumulateSquare, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulateSquare(src_roi, dst_roi));
        OCL_ON(cv::accumulateSquare(usrc_roi, udst_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

OCL_TEST_P(AccumulateSquare, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulateSquare(src_roi, dst_roi, mask_roi));
        OCL_ON(cv::accumulateSquare(usrc_roi, udst_roi, umask_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

/////////////////////////////////// AccumulateProduct ///////////////////////////////////

typedef AccumulateBase AccumulateProduct;

OCL_TEST_P(AccumulateProduct, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulateProduct(src_roi, src2_roi, dst_roi));
        OCL_ON(cv::accumulateProduct(usrc_roi, usrc2_roi, udst_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

OCL_TEST_P(AccumulateProduct, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulateProduct(src_roi, src2_roi, dst_roi, mask_roi));
        OCL_ON(cv::accumulateProduct(usrc_roi, usrc2_roi, udst_roi, umask_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

/////////////////////////////////// AccumulateWeighted ///////////////////////////////////

typedef AccumulateBase AccumulateWeighted;

OCL_TEST_P(AccumulateWeighted, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulateWeighted(src_roi, dst_roi, alpha));
        OCL_ON(cv::accumulateWeighted(usrc_roi, udst_roi, alpha));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

OCL_TEST_P(AccumulateWeighted, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::accumulateWeighted(src_roi, dst_roi, alpha));
        OCL_ON(cv::accumulateWeighted(usrc_roi, udst_roi, alpha));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

/////////////////////////////////// Instantiation ///////////////////////////////////

#define OCL_DEPTH_ALL_COMBINATIONS \
    testing::Values(std::make_pair<MatDepth, MatDepth>(CV_8U, CV_32F), \
    std::make_pair<MatDepth, MatDepth>(CV_16U, CV_32F), \
    std::make_pair<MatDepth, MatDepth>(CV_32F, CV_32F), \
    std::make_pair<MatDepth, MatDepth>(CV_8U, CV_64F), \
    std::make_pair<MatDepth, MatDepth>(CV_16U, CV_64F), \
    std::make_pair<MatDepth, MatDepth>(CV_32F, CV_64F), \
    std::make_pair<MatDepth, MatDepth>(CV_64F, CV_64F))

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, Accumulate, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(ImgProc, AccumulateSquare, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(ImgProc, AccumulateProduct, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(ImgProc, AccumulateWeighted, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));

} } // namespace cvtest::ocl

#endif
