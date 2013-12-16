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
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

////////////////////////////////converto/////////////////////////////////////////////////

PARAM_TEST_CASE(MatrixTestBase, MatDepth, MatDepth, Channels, bool)
{
    int src_depth, cn, dstType;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(src)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        src_depth = GET_PARAM(0);
        cn = GET_PARAM(2);
        dstType = CV_MAKE_TYPE(GET_PARAM(1), cn);

        use_roi = GET_PARAM(3);
    }

    virtual void generateTestData()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, CV_MAKE_TYPE(src_depth, cn), -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dstType, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst)
    }
};

typedef MatrixTestBase ConvertTo;

OCL_TEST_P(ConvertTo, Accuracy)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        double alpha = randomDouble(-4, 4), beta = randomDouble(-4, 4);

        OCL_OFF(src_roi.convertTo(dst_roi, dstType, alpha, beta));
        OCL_ON(usrc_roi.convertTo(udst_roi, dstType, alpha, beta));

        double eps = src_depth >= CV_32F || CV_MAT_DEPTH(dstType) >= CV_32F ? 1e-4 : 1;
        OCL_EXPECT_MATS_NEAR(dst, eps);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(MatrixOperation, ConvertTo, Combine(
                            OCL_ALL_DEPTHS, OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));

} } // namespace cvtest::ocl

#endif
