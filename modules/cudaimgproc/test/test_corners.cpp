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

namespace opencv_test { namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CornerHarris

namespace
{
    IMPLEMENT_PARAM_CLASS(BlockSize, int);
    IMPLEMENT_PARAM_CLASS(ApertureSize, int);
}

PARAM_TEST_CASE(CornerHarris, cv::cuda::DeviceInfo, MatType, BorderType, BlockSize, ApertureSize)
{
    cv::cuda::DeviceInfo devInfo;
    int type;
    int borderType;
    int blockSize;
    int apertureSize;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        blockSize = GET_PARAM(3);
        apertureSize = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CornerHarris, Accuracy)
{
    cv::Mat src = readImageType("stereobm/aloe-L.png", type);
    ASSERT_FALSE(src.empty());

    double k = randomDouble(0.1, 0.9);

    cv::Ptr<cv::cuda::CornernessCriteria> harris = cv::cuda::createHarrisCorner(src.type(), blockSize, apertureSize, k, borderType);

    cv::cuda::GpuMat dst;
    harris->compute(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::cornerHarris(src, dst_gold, blockSize, apertureSize, k, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.02);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CornerHarris, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cornerMinEigen

PARAM_TEST_CASE(CornerMinEigen, cv::cuda::DeviceInfo, MatType, BorderType, BlockSize, ApertureSize)
{
    cv::cuda::DeviceInfo devInfo;
    int type;
    int borderType;
    int blockSize;
    int apertureSize;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        blockSize = GET_PARAM(3);
        apertureSize = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CornerMinEigen, Accuracy)
{
    cv::Mat src = readImageType("stereobm/aloe-L.png", type);
    ASSERT_FALSE(src.empty());

    cv::Ptr<cv::cuda::CornernessCriteria> minEigenVal = cv::cuda::createMinEigenValCorner(src.type(), blockSize, apertureSize, borderType);

    cv::cuda::GpuMat dst;
    minEigenVal->compute(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::cornerMinEigenVal(src, dst_gold, blockSize, apertureSize, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.02);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CornerMinEigen, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));


}} // namespace
#endif // HAVE_CUDA
