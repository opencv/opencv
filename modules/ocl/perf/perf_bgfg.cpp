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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "perf_precomp.hpp"

///////////// PyrLKOpticalFlow ////////////////////////

using namespace perf;
using std::tr1::get;
using std::tr1::tuple;
using std::tr1::make_tuple;

#if defined(HAVE_XINE)         || \
    defined(HAVE_GSTREAMER)    || \
    defined(HAVE_QUICKTIME)    || \
    defined(HAVE_AVFOUNDATION) || \
    defined(HAVE_FFMPEG)       || \
    defined(WIN32) /* assume that we have ffmpeg */

#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 1
#else
#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 0
#endif

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

typedef tuple<string, int, double> VideoMOGParamType;
typedef TestBaseWithParam<VideoMOGParamType> VideoMOGFixture;

PERF_TEST_P(VideoMOGFixture, Video_MOG,
            ::testing::Combine(::testing::Values("768x576.avi", "1920x1080.avi"),
            ::testing::Values(1, 3),
            ::testing::Values(0.0, 0.01)))
{
    VideoMOGParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);
    const float learningRate = static_cast<float>(get<2>(params));
    
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::Mat temp;
    if (cn == 1)
        cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
    else
        cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
    cv::swap(temp, frame);

    if(RUN_PLAIN_IMPL)
    {
        cv::BackgroundSubtractorMOG mog;
        cv::Mat foreground;

        mog(frame, foreground, learningRate);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            cv::Mat temp;
            if (cn == 1)
                cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);

            cv::swap(temp, frame);

            TEST_CYCLE()
            mog(frame, foreground, learningRate);

            SANITY_CHECK(foreground);
        }
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::oclMat d_frame(frame);
        cv::ocl::MOG d_mog;
        cv::ocl::oclMat foreground;
        cv::Mat foreground_h;

        d_mog(d_frame, foreground, learningRate);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            cv::Mat temp;
            if (cn == 1)
                cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
            cv::swap(temp, frame);

            d_frame.upload(frame);

            OCL_TEST_CYCLE()
            d_mog(d_frame, foreground, learningRate);

            foreground.download(foreground_h);
            SANITY_CHECK(foreground_h);
        }
    }else
        OCL_PERF_ELSE
}
#endif

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

typedef tuple<string, int> VideoMOG2ParamType;
typedef TestBaseWithParam<VideoMOG2ParamType> VideoMOG2Fixture;

PERF_TEST_P(VideoMOG2Fixture, Video_MOG2,
            ::testing::Combine(::testing::Values("768x576.avi", "1920x1080.avi"),
            ::testing::Values(1, 3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::Mat temp;
    if (cn == 1)
        cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
    else
        cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
    cv::swap(temp, frame);

    if(RUN_PLAIN_IMPL)
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;

        mog2.set("detectShadows", false);
        mog2(frame, foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            cv::Mat temp;
            if (cn == 1)
                cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);

            cv::swap(temp, frame);

            TEST_CYCLE()
                mog2(frame, foreground);

            SANITY_CHECK(foreground);
        }
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::oclMat d_frame(frame);
        cv::ocl::MOG2 d_mog2;
        cv::ocl::oclMat foreground;
        cv::Mat foreground_h;

        d_mog2(d_frame, foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            cv::Mat temp;
            if (cn == 1)
                cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
            cv::swap(temp, frame);

            d_frame.upload(frame);

            OCL_TEST_CYCLE()
                d_mog2(d_frame, foreground);

            foreground.download(foreground_h);
            SANITY_CHECK(foreground_h);
        }
    }else
        OCL_PERF_ELSE
}
#endif

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

typedef TestBaseWithParam<VideoMOG2ParamType> Video_MOG2GetBackgroundImage;

PERF_TEST_P(Video_MOG2GetBackgroundImage, Video_MOG2,
            ::testing::Combine(::testing::Values("768x576.avi", "1920x1080.avi"),
            ::testing::Values(1, 3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::Mat temp;
    if (cn == 1)
        cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
    else
        cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
    cv::swap(temp, frame);

    if(RUN_PLAIN_IMPL)
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;

        mog2.set("detectShadows", false);
        mog2(frame, foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            cv::Mat temp;
            if (cn == 1)
                cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);

            cv::swap(temp, frame);

            TEST_CYCLE()
                mog2(frame, foreground);
        }
        cv::Mat background;
        TEST_CYCLE() 
            mog2.getBackgroundImage(background);

        SANITY_CHECK(background);
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::oclMat d_frame(frame);
        cv::ocl::MOG2 d_mog2;
        cv::ocl::oclMat foreground;
        cv::Mat background_h;

        d_mog2(d_frame, foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            cv::Mat temp;
            if (cn == 1)
                cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
            cv::swap(temp, frame);

            d_frame.upload(frame);
            d_mog2(d_frame, foreground);
        }
        cv::ocl::oclMat background;
        OCL_TEST_CYCLE()
            d_mog2.getBackgroundImage(background);

        background.download(background_h);
        SANITY_CHECK(background_h);
    }else
        OCL_PERF_ELSE
}
#endif

