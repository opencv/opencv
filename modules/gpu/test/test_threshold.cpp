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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifdef HAVE_CUDA

using namespace cvtest;

CV_ENUM(ThreshOp, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)

PARAM_TEST_CASE(Threshold, cv::gpu::DeviceInfo, cv::Size, MatType, ThreshOp, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int threshOp;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        threshOp = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Threshold, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    double maxVal = randomDouble(20.0, 127.0);
    double thresh = randomDouble(0.0, maxVal);

    cv::gpu::GpuMat dst = createMat(src.size(), src.type(), useRoi);
    cv::gpu::threshold(loadMat(src, useRoi), dst, thresh, maxVal, threshOp);

    cv::Mat dst_gold;
    cv::threshold(src, dst_gold, thresh, maxVal, threshOp);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

#ifdef OPENCV_TINY_GPU_MODULE
INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Threshold, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    ThreshOp::all(),
    WHOLE_SUBMAT));
#else
INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Threshold, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_16SC1), MatType(CV_32FC1)),
    ThreshOp::all(),
    WHOLE_SUBMAT));
#endif

#endif // HAVE_CUDA
