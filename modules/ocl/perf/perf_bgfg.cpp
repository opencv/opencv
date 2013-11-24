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
//     and/or other materials provided with the distribution.
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
using std::tr1::tuple;
using std::tr1::get;

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

static void cvtFrameFmt(vector<Mat>& input, vector<Mat>& output)
{
    for(int i = 0; i< (int)(input.size()); i++)
    {
        cvtColor(input[i], output[i], COLOR_RGB2GRAY);
    }
}

//prepare data for CPU
static void prepareData(VideoCapture& cap, int cn, vector<Mat>& frame_buffer)
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
        cvtFrameFmt(frame_buffer_init, frame_buffer);
    else
        frame_buffer = frame_buffer_init;
}

//copy CPU data to GPU
static void prepareData(vector<Mat>& frame_buffer, vector<oclMat>& frame_buffer_ocl)
{
    for(int i = 0; i < (int)frame_buffer.size(); i++)
        frame_buffer_ocl.push_back(cv::ocl::oclMat(frame_buffer[i]));
}

///////////// MOG ////////////////////////

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

    prepareData(cap, cn, frame_buffer);

    cv::Mat foreground;
    cv::ocl::oclMat foreground_d;
    if(RUN_PLAIN_IMPL)
    {
        TEST_CYCLE()
        {
            cv::Ptr<cv::BackgroundSubtractorMOG> mog = createBackgroundSubtractorMOG();
            foreground.release();
            for (int i = 0; i < nFrame; i++)
            {
                mog->apply(frame_buffer[i], foreground, learningRate);
            }
        }
        SANITY_CHECK(foreground);
    }
    else if(RUN_OCL_IMPL)
    {
        prepareData(frame_buffer, frame_buffer_ocl);
        CV_Assert((int)(frame_buffer_ocl.size()) == nFrame);
        OCL_TEST_CYCLE()
        {
            cv::ocl::MOG d_mog;
            foreground_d.release();
            for (int i = 0; i < nFrame; ++i)
            {
                d_mog(frame_buffer_ocl[i], foreground_d, learningRate);
            }
        }
        foreground_d.download(foreground);
        SANITY_CHECK(foreground);
    }
    else
        OCL_PERF_ELSE
}

///////////// MOG2 ////////////////////////

typedef tuple<string, int> VideoMOG2ParamType;
typedef TestBaseWithParam<VideoMOG2ParamType> VideoMOG2Fixture;

PERF_TEST_P(VideoMOG2Fixture, DISABLED_MOG2, // TODO Disabled: random hungs on buildslave
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
    prepareData(cap, cn, frame_buffer);
    cv::Mat foreground;
    cv::ocl::oclMat foreground_d;

    if(RUN_PLAIN_IMPL)
    {
        TEST_CYCLE()
        {
            cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
            mog2->setDetectShadows(false);
            foreground.release();

            for (int i = 0; i < nFrame; i++)
            {
                mog2->apply(frame_buffer[i], foreground);
            }
        }
        SANITY_CHECK(foreground);
    }
    else if(RUN_OCL_IMPL)
    {
        prepareData(frame_buffer, frame_buffer_ocl);
        CV_Assert((int)(frame_buffer_ocl.size()) == nFrame);
        OCL_TEST_CYCLE()
        {
            cv::ocl::MOG2 d_mog2;
            foreground_d.release();
            for (int i = 0; i < nFrame; i++)
            {
                d_mog2(frame_buffer_ocl[i], foreground_d);
            }
        }
        foreground_d.download(foreground);
        SANITY_CHECK(foreground);
    }
    else
        OCL_PERF_ELSE
}

///////////// MOG2_GetBackgroundImage //////////////////

typedef TestBaseWithParam<VideoMOG2ParamType> Video_MOG2GetBackgroundImage;

PERF_TEST_P(Video_MOG2GetBackgroundImage, MOG2,
            ::testing::Combine(::testing::Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
            ::testing::Values(3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = perf::TestBase::getDataPath(get<0>(params));
    const int cn = get<1>(params);
    int nFrame = 5;

    std::vector<cv::Mat> frame_buffer(nFrame);
    std::vector<cv::ocl::oclMat> frame_buffer_ocl;

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    prepareData(cap, cn, frame_buffer);

    cv::Mat foreground;
    cv::Mat background;
    cv::ocl::oclMat foreground_d;
    cv::ocl::oclMat background_d;

    if(RUN_PLAIN_IMPL)
    {
        TEST_CYCLE()
        {
            cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
            mog2->setDetectShadows(false);
            foreground.release();
            background.release();
            for (int i = 0; i < nFrame; i++)
            {
                mog2->apply(frame_buffer[i], foreground);
            }
            mog2->getBackgroundImage(background);
        }
        SANITY_CHECK(background);
    }
    else if(RUN_OCL_IMPL)
    {
        prepareData(frame_buffer, frame_buffer_ocl);
        CV_Assert((int)(frame_buffer_ocl.size()) == nFrame);
        OCL_TEST_CYCLE()
        {
            cv::ocl::MOG2 d_mog2;
            foreground_d.release();
            background_d.release();
            for (int i = 0; i < nFrame; i++)
            {
                d_mog2(frame_buffer_ocl[i], foreground_d);
            }
            d_mog2.getBackgroundImage(background_d);
        }
        background_d.download(background);
        SANITY_CHECK(background);
    }
    else
        OCL_PERF_ELSE
}

#endif
