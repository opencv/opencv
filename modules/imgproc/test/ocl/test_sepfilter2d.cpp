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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

/////////////////////////////////////////////////////////////////////////////////////////////////
// sepFilter2D
PARAM_TEST_CASE(SepFilter2D, MatDepth, Channels, BorderType, bool, bool)
{
    static const int kernelMinSize = 2;
    static const int kernelMaxSize = 10;

    int type;
    Point anchor;
    int borderType;
    bool useRoi;
    Mat kernelX, kernelY;
    double delta;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        borderType = GET_PARAM(2) | (GET_PARAM(3) ? BORDER_ISOLATED : 0);
        useRoi = GET_PARAM(4);
    }

    void random_roi(bool bitExact)
    {
        Size ksize = randomSize(kernelMinSize, kernelMaxSize);
        if (1 != ksize.width % 2)
            ksize.width++;
        if (1 != ksize.height % 2)
            ksize.height++;

        Mat temp = randomMat(Size(ksize.width, 1), CV_32FC1, -0.5, 1.0);
        cv::normalize(temp, kernelX, 1.0, 0.0, NORM_L1);
        temp = randomMat(Size(1, ksize.height), CV_32FC1, -0.5, 1.0);
        cv::normalize(temp, kernelY, 1.0, 0.0, NORM_L1);

        if (bitExact)
        {
            kernelX.convertTo(temp, CV_32S, 256);
            temp.convertTo(kernelX, CV_32F, 1.0 / 256);
            kernelY.convertTo(temp, CV_32S, 256);
            temp.convertTo(kernelY, CV_32F, 1.0 / 256);
        }

        Size roiSize = randomSize(ksize.width, MAX_VALUE, ksize.height, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        anchor.x = anchor.y = -1;
        delta = randomDouble(-100, 100);

        if (bitExact)
        {
            delta = (int)(delta * 256) / 256.0;
        }

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0)
    {
        OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

OCL_TEST_P(SepFilter2D, Mat)
{
    for (int j = 0; j < test_loop_times + 3; j++)
    {
        random_roi(false);

        OCL_OFF(cv::sepFilter2D(src_roi, dst_roi, -1, kernelX, kernelY, anchor, delta, borderType));
        OCL_ON(cv::sepFilter2D(usrc_roi, udst_roi, -1, kernelX, kernelY, anchor, delta, borderType));

        Near(1.0);
    }
}

OCL_TEST_P(SepFilter2D, Mat_BitExact)
{
    for (int j = 0; j < test_loop_times + 3; j++)
    {
        random_roi(true);

        OCL_OFF(cv::sepFilter2D(src_roi, dst_roi, -1, kernelX, kernelY, anchor, delta, borderType));
        OCL_ON(cv::sepFilter2D(usrc_roi, udst_roi, -1, kernelX, kernelY, anchor, delta, borderType));

        if (src_roi.depth() < CV_32F)
            Near(0.0);
        else
            Near(1e-3);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImageProc, SepFilter2D,
                            Combine(
                                Values(CV_8U, CV_32F),
                                OCL_ALL_CHANNELS,
                                Values(
                                        (BorderType)BORDER_CONSTANT,
                                        (BorderType)BORDER_REPLICATE,
                                        (BorderType)BORDER_REFLECT,
                                        (BorderType)BORDER_REFLECT_101),
                                Bool(), // BORDER_ISOLATED
                                Bool()  // ROI
                                )
                           );


} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
