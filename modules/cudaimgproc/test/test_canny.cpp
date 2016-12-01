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
// Canny

namespace
{
    IMPLEMENT_PARAM_CLASS(AppertureSize, int)
    IMPLEMENT_PARAM_CLASS(L2gradient, bool)
}

namespace {

PARAM_TEST_CASE(Canny, cv::cuda::DeviceInfo, AppertureSize, L2gradient, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    int apperture_size;
    bool useL2gradient;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        apperture_size = GET_PARAM(1);
        useL2gradient = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Canny, Accuracy)
{
    cv::Mat img = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    double low_thresh = 50.0;
    double high_thresh = 100.0;

    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, useL2gradient);

    cv::cuda::GpuMat edges;
    canny->detect(loadMat(img, useRoi), edges);

    cv::Mat edges_gold;
    cv::Canny(img, edges_gold, low_thresh, high_thresh, apperture_size, useL2gradient);

    EXPECT_MAT_SIMILAR(edges_gold, edges, 2e-2);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Canny, testing::Combine(
    ALL_DEVICES,
    testing::Values(AppertureSize(3), AppertureSize(5)),
    testing::Values(L2gradient(false), L2gradient(true)),
    WHOLE_SUBMAT));

} // namespace

#endif // HAVE_CUDA
