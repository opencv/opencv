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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma, jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

#if defined(HAVE_XINE)         || \
    defined(HAVE_GSTREAMER)    || \
    defined(HAVE_QUICKTIME)    || \
    defined(HAVE_AVFOUNDATION) || \
    defined(HAVE_FFMPEG)       || \
    defined(WIN32)

#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 1
#else
#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 0
#endif

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

//////////////////////////////////////////////////////
// MOG

namespace
{
    IMPLEMENT_PARAM_CLASS(UseGray, bool)
    IMPLEMENT_PARAM_CLASS(LearningRate, double)
}

PARAM_TEST_CASE(mog, UseGray, LearningRate, bool)
{
    bool useGray;
    double learningRate;
    bool useRoi;

    virtual void SetUp()
    {
        useGray = GET_PARAM(0);
        learningRate = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }
};

OCL_TEST_P(mog, Update)
{
    std::string inputFile = string(cvtest::TS::ptr()->get_data_path()) + "gpu/video/768x576.avi";
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::ocl::MOG mog;
    cv::ocl::oclMat foreground = createMat_ocl(rng, frame.size(), CV_8UC1, useRoi);

    Ptr<cv::BackgroundSubtractorMOG> mog_gold = createBackgroundSubtractorMOG();
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

        mog(loadMat_ocl(rng, frame, useRoi), foreground, (float)learningRate);

        mog_gold->apply(frame, foreground_gold, learningRate);

        EXPECT_MAT_NEAR(foreground_gold, foreground, 0.0);
    }
}
INSTANTIATE_TEST_CASE_P(OCL_Video, mog, testing::Combine(
    testing::Values(UseGray(false), UseGray(true)),
    testing::Values(LearningRate(0.0), LearningRate(0.01)),
    Values(true, false)));

//////////////////////////////////////////////////////
// MOG2

namespace
{
    IMPLEMENT_PARAM_CLASS(DetectShadow, bool)
}

PARAM_TEST_CASE(mog2, UseGray, DetectShadow, bool)
{
    bool useGray;
    bool detectShadow;
    bool useRoi;
    virtual void SetUp()
    {
        useGray = GET_PARAM(0);
        detectShadow = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }
};

OCL_TEST_P(mog2, Update)
{
    std::string inputFile = string(cvtest::TS::ptr()->get_data_path()) + "gpu/video/768x576.avi";
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::ocl::MOG2 mog2;
    mog2.bShadowDetection = detectShadow;
    cv::ocl::oclMat foreground = createMat_ocl(rng, frame.size(), CV_8UC1, useRoi);

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_gold = createBackgroundSubtractorMOG2();
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

        mog2(loadMat_ocl(rng, frame, useRoi), foreground);

        mog2_gold->apply(frame, foreground_gold);

        if (detectShadow)
            EXPECT_MAT_SIMILAR(foreground_gold, foreground, 15e-3)
        else
            EXPECT_MAT_NEAR(foreground_gold, foreground, 0)
    }
}

OCL_TEST_P(mog2, getBackgroundImage)
{
    if (useGray)
        return;

    std::string inputFile = string(cvtest::TS::ptr()->get_data_path()) + "gpu/video/768x576.avi";
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cv::ocl::MOG2 mog2;
    mog2.bShadowDetection = detectShadow;
    cv::ocl::oclMat foreground;

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_gold = createBackgroundSubtractorMOG2();
    mog2_gold->setDetectShadows(detectShadow);
    cv::Mat foreground_gold;

    for (int i = 0; i < 10; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        mog2(loadMat_ocl(rng, frame, useRoi), foreground);

        mog2_gold->apply(frame, foreground_gold);
    }

    cv::ocl::oclMat background = createMat_ocl(rng, frame.size(), frame.type(), useRoi);
    mog2.getBackgroundImage(background);

    cv::Mat background_gold;
    mog2_gold->getBackgroundImage(background_gold);

    EXPECT_MAT_NEAR(background_gold, background, 1.0);
}

INSTANTIATE_TEST_CASE_P(OCL_Video, mog2, testing::Combine(
    testing::Values(UseGray(true), UseGray(false)),
    testing::Values(DetectShadow(true), DetectShadow(false)),
    Values(true, false)));

#endif

#endif
