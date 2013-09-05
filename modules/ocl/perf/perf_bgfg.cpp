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
using namespace perf;
using namespace std;
using namespace cv::ocl;
using namespace cv;
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

static void cvtFrameFmt(vector<Mat>& input, vector<Mat>& output, int output_cn)
{
    for(int i = 0; i< (int)(input.size()); i++)
    {
        if(output_cn == 1)
            cvtColor(input[i], output[i], COLOR_RGB2GRAY);
        else
            cvtColor(input[i], output[i], COLOR_RGB2RGBA);
    }
}

static void prepareData(VideoCapture& cap, int cn, vector<Mat>& frame_buffer, vector<oclMat>& frame_buffer_ocl)
{
    cv::Mat frame;
    std::vector<Mat> frame_buffer_init;
    int nFrame = (int)frame_buffer.size();
    for(int i = 0; i < nFrame; i++)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());
        frame_buffer_init.push_back(frame);
    }

    if(cn == 1)
        cvtFrameFmt(frame_buffer_init, frame_buffer, 1);
    else
        frame_buffer = frame_buffer_init;

    for(int i = 0; i < nFrame; i++)
        frame_buffer_ocl.push_back(cv::ocl::oclMat(frame_buffer[i]));
}

///////////// MOG ////////////////////////
#if BUILD_WITH_VIDEO_INPUT_SUPPORT

typedef tuple<string, int, double> VideoMOGParamType;
typedef TestBaseWithParam<VideoMOGParamType> VideoMOGFixture;

PERF_TEST_P(VideoMOGFixture, MOG,
            ::testing::Combine(::testing::Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
            ::testing::Values(1, 3),
            ::testing::Values(0.0, 0.01)))
{
    VideoMOGParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);
    const float learningRate = static_cast<float>(get<2>(params));

    const int nFrame = 5;

    Mat foreground_cpu;
    std::vector<Mat> frame_buffer(nFrame);
    std::vector<oclMat> frame_buffer_ocl;

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    prepareData(cap, cn, frame_buffer, frame_buffer_ocl);

    if(RUN_PLAIN_IMPL)
    {
        cv::BackgroundSubtractorMOG mog;
        cv::Mat foreground;

        TEST_CYCLE()
        {
            for (int i = 0; i < nFrame; i++)
            {
                mog(frame_buffer[i], foreground, learningRate);
            }
        }
        SANITY_CHECK(foreground);
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::MOG d_mog;
        cv::ocl::oclMat foreground;
        cv::Mat foreground_h;
        OCL_TEST_CYCLE()
        {
            for (int i = 0; i < nFrame; ++i)
            {
                d_mog(frame_buffer_ocl[i], foreground, learningRate);
            }
        }
        foreground.download(foreground_h);
        SANITY_CHECK(foreground_h);
    }else
        OCL_PERF_ELSE
}
#endif

///////////// MOG2 ////////////////////////
#if BUILD_WITH_VIDEO_INPUT_SUPPORT

typedef tuple<string, int> VideoMOG2ParamType;
typedef TestBaseWithParam<VideoMOG2ParamType> VideoMOG2Fixture;

PERF_TEST_P(VideoMOG2Fixture, MOG2,
            ::testing::Combine(::testing::Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
            ::testing::Values(1, 3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);
    int nFrame = 5;

    std::vector<cv::Mat> frame_buffer(nFrame);
    std::vector<cv::ocl::oclMat> frame_buffer_ocl;

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    prepareData(cap, cn, frame_buffer, frame_buffer_ocl);

    if(RUN_PLAIN_IMPL)
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;

        mog2.set("detectShadows", false);

        TEST_CYCLE()
        {
            for (int i = 0; i < nFrame; i++)
            {
                mog2(frame_buffer[i], foreground);
            }
        }
        SANITY_CHECK(foreground);
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::MOG2 d_mog2;
        cv::ocl::oclMat foreground;
        cv::Mat foreground_h;

        OCL_TEST_CYCLE()
        {
            for (int i = 0; i < nFrame; i++)
            {
                d_mog2(frame_buffer_ocl[i], foreground);
            }
        }
        foreground.download(foreground_h);
        SANITY_CHECK(foreground_h);
    }else
        OCL_PERF_ELSE
}
#endif

///////////// MOG2_GetBackgroundImage //////////////////
#if BUILD_WITH_VIDEO_INPUT_SUPPORT

typedef TestBaseWithParam<VideoMOG2ParamType> Video_MOG2GetBackgroundImage;

PERF_TEST_P(Video_MOG2GetBackgroundImage, MOG2,
            ::testing::Combine(::testing::Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
            ::testing::Values(1, 3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);
    int nFrame = 5;

    std::vector<cv::Mat> frame_buffer(nFrame);
    std::vector<cv::ocl::oclMat> frame_buffer_ocl;

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    prepareData(cap, cn, frame_buffer, frame_buffer_ocl);

    if(RUN_PLAIN_IMPL)
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;
        cv::Mat background;
        mog2.set("detectShadows", false);
        TEST_CYCLE()
        {
            for (int i = 0; i < nFrame; i++)
            {
                mog2(frame_buffer[i], foreground);
            }
            mog2.getBackgroundImage(background);
        }
        SANITY_CHECK(background);
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::MOG2 d_mog2;
        cv::ocl::oclMat foreground;
        cv::Mat background_h;
        cv::ocl::oclMat background;

        OCL_TEST_CYCLE()
        {
            for (int i = 0; i < nFrame; i++)
            {
                d_mog2(frame_buffer_ocl[i], foreground);
            }
            d_mog2.getBackgroundImage(background);
        }
        background.download(background_h);
        SANITY_CHECK(background_h);
    }else
        OCL_PERF_ELSE
}
#endif