///////////////////////////////////////////////////////////////////////////////////////
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
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Yao Wang yao@multicorewareinc.com
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

PARAM_TEST_CASE(PyrTestBase, MatDepth, Channels, bool)
{
    int depth, channels;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        channels = GET_PARAM(1);
        use_roi = GET_PARAM(2);
    }

    void generateTestData(Size src_roiSize, Size dst_roiSize)
    {
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, src_roiSize, srcBorder, CV_MAKETYPE(depth, channels), -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, dst_roiSize, dstBorder, CV_MAKETYPE(depth, channels), -MAX_VALUE, MAX_VALUE);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0)
    {
        OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

/////////////////////// PyrDown //////////////////////////

typedef PyrTestBase PyrDown;

OCL_TEST_P(PyrDown, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        Size src_roiSize = randomSize(1, MAX_VALUE);
        Size dst_roiSize = Size(randomInt((src_roiSize.width - 1) / 2, (src_roiSize.width + 3) / 2),
                                randomInt((src_roiSize.height - 1) / 2, (src_roiSize.height + 3) / 2));
        dst_roiSize = dst_roiSize.area() == 0 ? Size((src_roiSize.width + 1) / 2, (src_roiSize.height + 1) / 2) : dst_roiSize;
        generateTestData(src_roiSize, dst_roiSize);

        OCL_OFF(pyrDown(src_roi, dst_roi, dst_roiSize));
        OCL_ON(pyrDown(usrc_roi, udst_roi, dst_roiSize));

        Near(depth == CV_32F ? 1e-4f : 1.0f);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImgprocPyr, PyrDown, Combine(
                            Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),
                            Values(1, 2, 3, 4),
                            Bool()
                            ));

/////////////////////// PyrUp //////////////////////////

typedef PyrTestBase PyrUp;

OCL_TEST_P(PyrUp, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        Size src_roiSize = randomSize(1, MAX_VALUE);
        Size dst_roiSize = Size(2 * src_roiSize.width, 2 * src_roiSize.height);
        generateTestData(src_roiSize, dst_roiSize);

        OCL_OFF(pyrUp(src_roi, dst_roi, dst_roiSize));
        OCL_ON(pyrUp(usrc_roi, udst_roi, dst_roiSize));

        Near(depth == CV_32F ? 1e-4f : 1.0f);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImgprocPyr, PyrUp, Combine(
                            Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),
                            Values(1, 2, 3, 4),
                            Bool()
                            ));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
