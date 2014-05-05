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
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

/////////////////////////////////////////////medianFilter//////////////////////////////////////////////////////////

PARAM_TEST_CASE(MedianFilter, MatDepth, Channels, int, bool)
{
    int type;
    int ksize;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        ksize = GET_PARAM(2);
        use_roi = GET_PARAM(3);
    }

    virtual void generateTestData()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0)
    {
        EXPECT_MAT_NEAR(dst, udst, threshold);
        EXPECT_MAT_NEAR(dst_roi, udst_roi, threshold);
    }
};

OCL_TEST_P(MedianFilter, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::medianBlur(src_roi, dst_roi, ksize));
        OCL_ON(cv::medianBlur(usrc_roi, udst_roi, ksize));

        Near(0);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImageProc, MedianFilter, Combine(
                                Values(CV_8U, CV_16U, CV_16S, CV_32F),
                                OCL_ALL_CHANNELS,
                                Values(3, 5),
                                Bool())
                           );
} } // namespace cvtest::ocl

#endif
