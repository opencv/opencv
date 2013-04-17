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

////////////////////////////////////////////////////////
// pyrDown

PARAM_TEST_CASE(PyrDown, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(PyrDown, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(cv::Size((size.width + 1) / 2, (size.height + 1) / 2), type, useRoi);
    cv::gpu::pyrDown(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::pyrDown(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-4 : 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, PyrDown, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////
// pyrUp

PARAM_TEST_CASE(PyrUp, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(PyrUp, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::gpu::GpuMat dst = createMat(cv::Size(size.width * 2, size.height * 2), type, useRoi);
    cv::gpu::pyrUp(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::pyrUp(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-4 : 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, PyrUp, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
