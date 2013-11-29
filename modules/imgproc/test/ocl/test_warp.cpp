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
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan, lyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Wu Zailong, bullet@yeah.net
//    Xu Pang, pangxu010@163.com
//    Sen Liu, swjtuls1987@126.com
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
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

/////////////////////////////////////////////////////////////////////////////////////////////////
// resize

PARAM_TEST_CASE(Resize, MatType, double, double, Interpolation, bool)
{
    int type, interpolation;
    double fx, fy;
    bool useRoi;

    TEST_DECLARE_INPUT_PARATEMER(src)
    TEST_DECLARE_OUTPUT_PARATEMER(dst)

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        fx = GET_PARAM(1);
        fy = GET_PARAM(2);
        interpolation = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        CV_Assert(fx > 0 && fy > 0);

        Size srcRoiSize = randomSize(1, MAX_VALUE), dstRoiSize;
        dstRoiSize.width = cvRound(srcRoiSize.width * fx);
        dstRoiSize.height = cvRound(srcRoiSize.height * fy);

        if (dstRoiSize.area() == 0)
        {
            random_roi();
            return;
        }

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, srcRoiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, dstRoiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }

    void Near(double threshold = 0.0)
    {
        EXPECT_MAT_NEAR(dst_roi, udst_roi, threshold);
        EXPECT_MAT_NEAR(dst, udst, threshold);
    }
};

OCL_TEST_P(Resize, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        random_roi();

        OCL_OFF(cv::resize(src_roi, dst_roi, Size(), fx, fy, interpolation));
        OCL_ON(cv::resize(usrc_roi, udst_roi, Size(), fx, fy, interpolation));

        Near(1.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(ImgprocWarpResize, Resize, Combine(
                            Values((MatType)CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(0.7, 0.4, 2.0),
                            Values(0.3, 0.6, 2.0),
                            Values((Interpolation)INTER_NEAREST, (Interpolation)INTER_LINEAR),
                            Bool()));

OCL_INSTANTIATE_TEST_CASE_P(ImgprocWarpResizeArea, Resize, Combine(
                            Values((MatType)CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                            Values(0.7, 0.4, 0.5),
                            Values(0.3, 0.6, 0.5),
                            Values((Interpolation)INTER_AREA),
                            Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
