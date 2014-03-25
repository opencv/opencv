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

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

////////////////////////////////////////// boxFilter ///////////////////////////////////////////////////////

PARAM_TEST_CASE(BoxFilterBase, MatDepth, Channels, BorderType, bool, bool)
{
    static const int kernelMinSize = 2;
    static const int kernelMaxSize = 10;

    int depth, cn, borderType;
    Size ksize, dsize;
    Point anchor;
    bool normalize, useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        borderType = GET_PARAM(2); // only not isolated border tested, because CPU module doesn't support isolated border case.
        normalize = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        int type = CV_MAKE_TYPE(depth, cn);
        ksize = randomSize(kernelMinSize, kernelMaxSize);

        Size roiSize = randomSize(ksize.width, MAX_VALUE, ksize.height, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        anchor.x = randomInt(-1, ksize.width);
        anchor.y = randomInt(-1, ksize.height);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0)
    {
        OCL_EXPECT_MATS_NEAR(dst, threshold);
    }
};

typedef BoxFilterBase BoxFilter;

OCL_TEST_P(BoxFilter, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::boxFilter(src_roi, dst_roi, -1, ksize, anchor, normalize, borderType));
        OCL_ON(cv::boxFilter(usrc_roi, udst_roi, -1, ksize, anchor, normalize, borderType));

        Near(depth <= CV_32S ? 1 : 1e-3);
    }
}

typedef BoxFilterBase SqrBoxFilter;

OCL_TEST_P(SqrBoxFilter, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        int ddepth = depth == CV_8U ? CV_32S : CV_64F;

        OCL_OFF(cv::sqrBoxFilter(src_roi, dst_roi, ddepth, ksize, anchor, normalize, borderType));
        OCL_ON(cv::sqrBoxFilter(usrc_roi, udst_roi, ddepth, ksize, anchor, normalize, borderType));

        Near(depth <= CV_32S ? 1 : 7e-2);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(ImageProc, BoxFilter,
                            Combine(
                                Values(CV_8U, CV_16U, CV_16S, CV_32S, CV_32F),
                                OCL_ALL_CHANNELS,
                                Values((BorderType)BORDER_CONSTANT,
                                       (BorderType)BORDER_REPLICATE,
                                       (BorderType)BORDER_REFLECT,
                                       (BorderType)BORDER_REFLECT_101),
                                Bool(),
                                Bool()  // ROI
                                )
                           );

OCL_INSTANTIATE_TEST_CASE_P(ImageProc, SqrBoxFilter,
                            Combine(
                                Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),
                                OCL_ALL_CHANNELS,
                                Values((BorderType)BORDER_CONSTANT,
                                       (BorderType)BORDER_REPLICATE,
                                       (BorderType)BORDER_REFLECT,
                                       (BorderType)BORDER_REFLECT_101),
                                Bool(),
                                Bool()  // ROI
                                )
                           );


} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
