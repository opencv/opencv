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

#include "perf_precomp.hpp"

using namespace std;
using namespace testing;
using namespace perf;

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

#if defined(HAVE_OPENCV_GPU) && defined(HAVE_CUDA)

//////////////////////////////////////////////////////////////////////
// SURF

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, GPU_SURF,
            Values<string>("gpu/perf/aloe.png"))
{
    declare.time(50.0);

    const cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::SURF_GPU d_surf;

        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints, d_descriptors;

        TEST_CYCLE() d_surf(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);

        std::vector<cv::KeyPoint> gpu_keypoints;
        d_surf.downloadKeypoints(d_keypoints, gpu_keypoints);

        cv::Mat gpu_descriptors(d_descriptors);

        sortKeyPoints(gpu_keypoints, gpu_descriptors);

        SANITY_CHECK_KEYPOINTS(gpu_keypoints);
        SANITY_CHECK(gpu_descriptors, 1e-3);
    }
    else
    {
        cv::SURF surf;

        std::vector<cv::KeyPoint> cpu_keypoints;
        cv::Mat cpu_descriptors;

        TEST_CYCLE() surf(img, cv::noArray(), cpu_keypoints, cpu_descriptors);

        SANITY_CHECK_KEYPOINTS(cpu_keypoints);
        SANITY_CHECK(cpu_descriptors);
    }
}

//////////////////////////////////////////////////////
// VIBE

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

DEF_PARAM_TEST(Video_Cn, string, int);

PERF_TEST_P(Video_Cn, GPU_VIBE,
            Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
                    GPU_CHANNELS_1_3_4))
{
    const string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    const int cn = GET_PARAM(1);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (cn != 3)
    {
        cv::Mat temp;
        if (cn == 1)
            cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
        else
            cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
        cv::swap(temp, frame);
    }

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_frame(frame);
        cv::gpu::VIBE_GPU vibe;
        cv::gpu::GpuMat foreground;

        vibe(d_frame, foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            d_frame.upload(frame);

            startTimer(); next();
            vibe(d_frame, foreground);
            stopTimer();
        }

        GPU_SANITY_CHECK(foreground);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

#endif

#endif
