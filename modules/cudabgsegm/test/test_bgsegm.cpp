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

//////////////////////////////////////////////////////
// MOG2

#ifdef HAVE_VIDEO_INPUT

namespace
    {
IMPLEMENT_PARAM_CLASS(UseGray, bool)
    IMPLEMENT_PARAM_CLASS(DetectShadow, bool)
}

PARAM_TEST_CASE(MOG2, cv::cuda::DeviceInfo, std::string, UseGray, DetectShadow, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    std::string inputFile;
    bool useGray;
    bool detectShadow;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());

        inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + GET_PARAM(1);
        useGray = GET_PARAM(2);
        detectShadow = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }
};

CUDA_TEST_P(MOG2, Update)
{
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::cuda::createBackgroundSubtractorMOG2();
    mog2->setDetectShadows(detectShadow);
    cv::cuda::GpuMat foreground = createMat(frame.size(), CV_8UC1, useRoi);

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_gold = cv::createBackgroundSubtractorMOG2();
    mog2_gold->setDetectShadows(detectShadow);
    cv::Mat foreground_gold;

    for (int i = 0; i < 10; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        if (useGray)
        {
            cv::Mat temp;
            cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            cv::swap(temp, frame);
        }

        mog2->apply(loadMat(frame, useRoi), foreground);

        mog2_gold->apply(frame, foreground_gold);

        if (detectShadow)
        {
            ASSERT_MAT_SIMILAR(foreground_gold, foreground, 1e-2);
        }
        else
        {
            ASSERT_MAT_NEAR(foreground_gold, foreground, 0);
        }
    }
}

CUDA_TEST_P(MOG2, getBackgroundImage)
{
    if (useGray)
        return;

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::cuda::createBackgroundSubtractorMOG2();
    mog2->setDetectShadows(detectShadow);
    cv::cuda::GpuMat foreground;

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_gold = cv::createBackgroundSubtractorMOG2();
    mog2_gold->setDetectShadows(detectShadow);
    cv::Mat foreground_gold;

    for (int i = 0; i < 10; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        mog2->apply(loadMat(frame, useRoi), foreground);

        mog2_gold->apply(frame, foreground_gold);
    }

    cv::cuda::GpuMat background = createMat(frame.size(), frame.type(), useRoi);
    mog2->getBackgroundImage(background);

    cv::Mat background_gold;
    mog2_gold->getBackgroundImage(background_gold);

    ASSERT_MAT_NEAR(background_gold, background, 1);
}

INSTANTIATE_TEST_CASE_P(CUDA_BgSegm, MOG2, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("768x576.avi")),
    testing::Values(UseGray(true), UseGray(false)),
    testing::Values(DetectShadow(true), DetectShadow(false)),
    WHOLE_SUBMAT));

#endif

}} // namespace
#endif // HAVE_CUDA
