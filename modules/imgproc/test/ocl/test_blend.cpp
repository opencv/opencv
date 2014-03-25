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

#include "test_precomp.hpp"
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(BlendLinear, MatDepth, Channels, bool)
{
    int depth, channels;
    bool useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src1);
    TEST_DECLARE_INPUT_PARAMETER(src2);
    TEST_DECLARE_INPUT_PARAMETER(weights2);
    TEST_DECLARE_INPUT_PARAMETER(weights1);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        channels = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }

    void random_roi()
    {
        const int type = CV_MAKE_TYPE(depth, channels);
        const double upValue = 256;

        Size roiSize = randomSize(1, MAX_VALUE);
        Border src1Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, src1Border, type, -upValue, upValue);

        Border src2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, type, -upValue, upValue);

        Border weights1Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(weights1, weights1_roi, roiSize, weights1Border, CV_32FC1, -upValue, upValue);

        Border weights2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(weights2, weights2_roi, roiSize, weights2Border, CV_32FC1, 1e-2, upValue);

        weights2_roi -= weights1_roi;
        CV_Assert(checkNorm2(weights2_roi, weights2(Rect(weights2Border.lef, weights2Border.top,
                                                        roiSize.width, roiSize.height))) < 1e-6);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src1);
        UMAT_UPLOAD_INPUT_PARAMETER(src2);
        UMAT_UPLOAD_INPUT_PARAMETER(weights1);
        UMAT_UPLOAD_INPUT_PARAMETER(weights2);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double eps = 0.0)
    {
        OCL_EXPECT_MATS_NEAR(dst, eps);
    }
};

OCL_TEST_P(BlendLinear, Accuracy)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::blendLinear(src1_roi, src2_roi, weights1_roi, weights2_roi, dst_roi));
        OCL_ON(cv::blendLinear(usrc1_roi, usrc2_roi, uweights1_roi, uweights2_roi, udst_roi));

        Near(depth <= CV_32S ? 1.0 : 0.2);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImgProc, BlendLinear, Combine(testing::Values(CV_8U, CV_32F), OCL_ALL_CHANNELS, Bool()));

} } // namespace cvtest::ocl

#endif
