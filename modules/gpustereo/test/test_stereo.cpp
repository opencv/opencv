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

//////////////////////////////////////////////////////////////////////////
// StereoBM

struct StereoBM : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(StereoBM, Regression)
{
    cv::Mat left_image  = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_image = readImage("stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);
    cv::Mat disp_gold   = readImage("stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    cv::gpu::StereoBM_GPU bm(0, 128, 19);
    cv::gpu::GpuMat disp;

    bm(loadMat(left_image), loadMat(right_image), disp);

    EXPECT_MAT_NEAR(disp_gold, disp, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Stereo, StereoBM, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////////
// StereoBeliefPropagation

struct StereoBeliefPropagation : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(StereoBeliefPropagation, Regression)
{
    cv::Mat left_image  = readImage("stereobp/aloe-L.png");
    cv::Mat right_image = readImage("stereobp/aloe-R.png");
    cv::Mat disp_gold   = readImage("stereobp/aloe-disp.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    cv::gpu::StereoBeliefPropagation bp(64, 8, 2, 25, 0.1f, 15, 1, CV_16S);
    cv::gpu::GpuMat disp;

    bp(loadMat(left_image), loadMat(right_image), disp);

    cv::Mat h_disp(disp);
    h_disp.convertTo(h_disp, disp_gold.depth());

    EXPECT_MAT_NEAR(disp_gold, h_disp, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Stereo, StereoBeliefPropagation, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////////
// StereoConstantSpaceBP

struct StereoConstantSpaceBP : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(StereoConstantSpaceBP, Regression)
{
    cv::Mat left_image  = readImage("csstereobp/aloe-L.png");
    cv::Mat right_image = readImage("csstereobp/aloe-R.png");

    cv::Mat disp_gold;

    if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
        disp_gold = readImage("csstereobp/aloe-disp.png", cv::IMREAD_GRAYSCALE);
    else
        disp_gold = readImage("csstereobp/aloe-disp_CC1X.png", cv::IMREAD_GRAYSCALE);

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    cv::gpu::StereoConstantSpaceBP csbp(128, 16, 4, 4);
    cv::gpu::GpuMat disp;

    csbp(loadMat(left_image), loadMat(right_image), disp);

    cv::Mat h_disp(disp);
    h_disp.convertTo(h_disp, disp_gold.depth());

    EXPECT_MAT_NEAR(disp_gold, h_disp, 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_Stereo, StereoConstantSpaceBP, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

PARAM_TEST_CASE(ReprojectImageTo3D, cv::gpu::DeviceInfo, cv::Size, MatDepth, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(ReprojectImageTo3D, Accuracy)
{
    cv::Mat disp = randomMat(size, depth, 5.0, 30.0);
    cv::Mat Q = randomMat(cv::Size(4, 4), CV_32FC1, 0.1, 1.0);

    cv::gpu::GpuMat dst;
    cv::gpu::reprojectImageTo3D(loadMat(disp, useRoi), dst, Q, 3);

    cv::Mat dst_gold;
    cv::reprojectImageTo3D(disp, dst_gold, Q, false);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(GPU_Stereo, ReprojectImageTo3D, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16S)),
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
