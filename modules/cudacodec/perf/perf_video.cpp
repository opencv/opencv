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
#include "opencv2/highgui/highgui_c.h"

namespace opencv_test { namespace {

DEF_PARAM_TEST_1(FileName, string);

//////////////////////////////////////////////////////
// VideoReader

#if defined(HAVE_NVCUVID) && defined(HAVE_VIDEO_INPUT)

PERF_TEST_P(FileName, VideoReader, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(20);

    const string inputFile = perf::TestBase::getDataPath(GetParam());

    if (PERF_RUN_CUDA())
    {
        cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(inputFile);

        cv::cuda::GpuMat frame;

        TEST_CYCLE_N(10) d_reader->nextFrame(frame);

        CUDA_SANITY_CHECK(frame);
    }
    else
    {
        cv::VideoCapture reader(inputFile);
        ASSERT_TRUE( reader.isOpened() );

        cv::Mat frame;

        TEST_CYCLE_N(10) reader >> frame;

        CPU_SANITY_CHECK(frame);
    }
}

#endif

//////////////////////////////////////////////////////
// VideoWriter

#if defined(HAVE_NVCUVID) && defined(_WIN32)

PERF_TEST_P(FileName, VideoWriter, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(30);

    const string inputFile = perf::TestBase::getDataPath(GetParam());
    const string outputFile = cv::tempfile(".avi");

    const double FPS = 25.0;

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::Mat frame;

    if (PERF_RUN_CUDA())
    {
        cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

        cv::cuda::GpuMat d_frame;

        for (int i = 0; i < 10; ++i)
        {
            reader >> frame;
            ASSERT_FALSE(frame.empty());

            d_frame.upload(frame);

            if (d_writer.empty())
                d_writer = cv::cudacodec::createVideoWriter(outputFile, frame.size(), FPS);

            startTimer(); next();
            d_writer->write(d_frame);
            stopTimer();
        }
    }
    else
    {
        cv::VideoWriter writer;

        for (int i = 0; i < 10; ++i)
        {
            reader >> frame;
            ASSERT_FALSE(frame.empty());

            if (!writer.isOpened())
                writer.open(outputFile, CV_FOURCC('X', 'V', 'I', 'D'), FPS, frame.size());

            startTimer(); next();
            writer.write(frame);
            stopTimer();
        }
    }

    SANITY_CHECK(frame);
}

#endif
}} // namespace
