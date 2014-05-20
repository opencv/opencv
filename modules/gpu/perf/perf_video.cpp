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
    defined(HAVE_QTKIT)        || \
    defined(HAVE_AVFOUNDATION) || \
    defined(HAVE_FFMPEG)       || \
    defined(WIN32) /* assume that we have ffmpeg */

#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 1
#else
#  define BUILD_WITH_VIDEO_INPUT_SUPPORT 0
#endif

namespace cv
{
    template<> void Ptr<CvBGStatModel>::delete_obj()
    {
        cvReleaseBGStatModel(&obj);
    }
}

//////////////////////////////////////////////////////
// InterpolateFrames

typedef pair<string, string> pair_string;

DEF_PARAM_TEST_1(ImagePair, pair_string);

PERF_TEST_P(ImagePair, Video_InterpolateFrames,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_fu, d_fv;
        cv::gpu::GpuMat d_bu, d_bv;

        cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                        10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

        d_flow(d_frame0, d_frame1, d_fu, d_fv);
        d_flow(d_frame1, d_frame0, d_bu, d_bv);

        cv::gpu::GpuMat newFrame;
        cv::gpu::GpuMat d_buf;

        TEST_CYCLE() cv::gpu::interpolateFrames(d_frame0, d_frame1, d_fu, d_fv, d_bu, d_bv, 0.5f, newFrame, d_buf);

        GPU_SANITY_CHECK(newFrame, 1e-4);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////
// CreateOpticalFlowNeedleMap

PERF_TEST_P(ImagePair, Video_CreateOpticalFlowNeedleMap,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u;
        cv::gpu::GpuMat v;

        cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                        10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

        d_flow(d_frame0, d_frame1, u, v);

        cv::gpu::GpuMat vertex, colors;

        TEST_CYCLE() cv::gpu::createOpticalFlowNeedleMap(u, v, vertex, colors);

        GPU_SANITY_CHECK(vertex, 1e-6);
        GPU_SANITY_CHECK(colors);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////
// GoodFeaturesToTrack

DEF_PARAM_TEST(Image_MinDistance, string, double);

PERF_TEST_P(Image_MinDistance, Video_GoodFeaturesToTrack,
            Combine(Values<string>("gpu/perf/aloe.png"),
                    Values(0.0, 3.0)))
{
    const string fileName = GET_PARAM(0);
    const double minDistance = GET_PARAM(1);

    const cv::Mat image = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    const int maxCorners = 8000;
    const double qualityLevel = 0.01;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GoodFeaturesToTrackDetector_GPU d_detector(maxCorners, qualityLevel, minDistance);

        const cv::gpu::GpuMat d_image(image);
        cv::gpu::GpuMat pts;

        TEST_CYCLE() d_detector(d_image, pts);

        GPU_SANITY_CHECK(pts);
    }
    else
    {
        cv::Mat pts;

        TEST_CYCLE() cv::goodFeaturesToTrack(image, pts, maxCorners, qualityLevel, minDistance);

        CPU_SANITY_CHECK(pts);
    }
}

//////////////////////////////////////////////////////
// BroxOpticalFlow

PERF_TEST_P(ImagePair, Video_BroxOpticalFlow,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(300);

    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u;
        cv::gpu::GpuMat v;

        cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                        10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

        TEST_CYCLE() d_flow(d_frame0, d_frame1, u, v);

        GPU_SANITY_CHECK(u, 1e-1);
        GPU_SANITY_CHECK(v, 1e-1);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////
// PyrLKOpticalFlowSparse

DEF_PARAM_TEST(ImagePair_Gray_NPts_WinSz_Levels_Iters, pair_string, bool, int, int, int, int);

PERF_TEST_P(ImagePair_Gray_NPts_WinSz_Levels_Iters, Video_PyrLKOpticalFlowSparse,
            Combine(Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")),
                    Bool(),
                    Values(8000),
                    Values(21),
                    Values(1, 3),
                    Values(1, 30)))
{
    declare.time(20.0);

    const pair_string imagePair = GET_PARAM(0);
    const bool useGray = GET_PARAM(1);
    const int points = GET_PARAM(2);
    const int winSize = GET_PARAM(3);
    const int levels = GET_PARAM(4);
    const int iters = GET_PARAM(5);

    const cv::Mat frame0 = readImage(imagePair.first, useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    const cv::Mat frame1 = readImage(imagePair.second, useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    cv::Mat pts;
    cv::goodFeaturesToTrack(gray_frame, pts, points, 0.01, 0.0);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_pts(pts.reshape(2, 1));

        cv::gpu::PyrLKOpticalFlow d_pyrLK;
        d_pyrLK.winSize = cv::Size(winSize, winSize);
        d_pyrLK.maxLevel = levels - 1;
        d_pyrLK.iters = iters;

        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat nextPts;
        cv::gpu::GpuMat status;

        TEST_CYCLE() d_pyrLK.sparse(d_frame0, d_frame1, d_pts, nextPts, status);

        GPU_SANITY_CHECK(nextPts);
        GPU_SANITY_CHECK(status);
    }
    else
    {
        cv::Mat nextPts;
        cv::Mat status;

        TEST_CYCLE()
        {
            cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, cv::noArray(),
                                     cv::Size(winSize, winSize), levels - 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iters, 0.01));
        }

        CPU_SANITY_CHECK(nextPts);
        CPU_SANITY_CHECK(status);
    }
}

//////////////////////////////////////////////////////
// PyrLKOpticalFlowDense

DEF_PARAM_TEST(ImagePair_WinSz_Levels_Iters, pair_string, int, int, int);

PERF_TEST_P(ImagePair_WinSz_Levels_Iters, Video_PyrLKOpticalFlowDense,
            Combine(Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")),
                    Values(3, 5, 7, 9, 13, 17, 21),
                    Values(1, 3),
                    Values(1, 10)))
{
    declare.time(30);

    const pair_string imagePair = GET_PARAM(0);
    const int winSize = GET_PARAM(1);
    const int levels = GET_PARAM(2);
    const int iters = GET_PARAM(3);

    const cv::Mat frame0 = readImage(imagePair.first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    const cv::Mat frame1 = readImage(imagePair.second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u;
        cv::gpu::GpuMat v;

        cv::gpu::PyrLKOpticalFlow d_pyrLK;
        d_pyrLK.winSize = cv::Size(winSize, winSize);
        d_pyrLK.maxLevel = levels - 1;
        d_pyrLK.iters = iters;

        TEST_CYCLE() d_pyrLK.dense(d_frame0, d_frame1, u, v);

        GPU_SANITY_CHECK(u);
        GPU_SANITY_CHECK(v);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////
// FarnebackOpticalFlow

PERF_TEST_P(ImagePair, Video_FarnebackOpticalFlow,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(10);

    const cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    const cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    const int numLevels = 5;
    const double pyrScale = 0.5;
    const int winSize = 13;
    const int numIters = 10;
    const int polyN = 5;
    const double polySigma = 1.1;
    const int flags = 0;

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u;
        cv::gpu::GpuMat v;

        cv::gpu::FarnebackOpticalFlow d_farneback;
        d_farneback.numLevels = numLevels;
        d_farneback.pyrScale = pyrScale;
        d_farneback.winSize = winSize;
        d_farneback.numIters = numIters;
        d_farneback.polyN = polyN;
        d_farneback.polySigma = polySigma;
        d_farneback.flags = flags;

        TEST_CYCLE() d_farneback(d_frame0, d_frame1, u, v);

        GPU_SANITY_CHECK(u, 1e-4);
        GPU_SANITY_CHECK(v, 1e-4);
    }
    else
    {
        cv::Mat flow;

        TEST_CYCLE() cv::calcOpticalFlowFarneback(frame0, frame1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);

        CPU_SANITY_CHECK(flow);
    }
}

//////////////////////////////////////////////////////
// OpticalFlowDual_TVL1

PERF_TEST_P(ImagePair, Video_OpticalFlowDual_TVL1,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(20);

    const cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    const cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u;
        cv::gpu::GpuMat v;

        cv::gpu::OpticalFlowDual_TVL1_GPU d_alg;

        TEST_CYCLE() d_alg(d_frame0, d_frame1, u, v);

        GPU_SANITY_CHECK(u, 1e-1);
        GPU_SANITY_CHECK(v, 1e-1);
    }
    else
    {
        cv::Mat flow;

        cv::Ptr<cv::DenseOpticalFlow> alg = cv::createOptFlow_DualTVL1();

        TEST_CYCLE() alg->calc(frame0, frame1, flow);

        CPU_SANITY_CHECK(flow);
    }
}

//////////////////////////////////////////////////////
// OpticalFlowBM

void calcOpticalFlowBM(const cv::Mat& prev, const cv::Mat& curr,
                       cv::Size bSize, cv::Size shiftSize, cv::Size maxRange, int usePrevious,
                       cv::Mat& velx, cv::Mat& vely)
{
    cv::Size sz((curr.cols - bSize.width + shiftSize.width)/shiftSize.width, (curr.rows - bSize.height + shiftSize.height)/shiftSize.height);

    velx.create(sz, CV_32FC1);
    vely.create(sz, CV_32FC1);

    CvMat cvprev = prev;
    CvMat cvcurr = curr;

    CvMat cvvelx = velx;
    CvMat cvvely = vely;

    cvCalcOpticalFlowBM(&cvprev, &cvcurr, bSize, shiftSize, maxRange, usePrevious, &cvvelx, &cvvely);
}

PERF_TEST_P(ImagePair, Video_OpticalFlowBM,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(400);

    const cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    const cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    const cv::Size block_size(16, 16);
    const cv::Size shift_size(1, 1);
    const cv::Size max_range(16, 16);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u, v, buf;

        TEST_CYCLE() cv::gpu::calcOpticalFlowBM(d_frame0, d_frame1, block_size, shift_size, max_range, false, u, v, buf);

        GPU_SANITY_CHECK(u);
        GPU_SANITY_CHECK(v);
    }
    else
    {
        cv::Mat u, v;

        TEST_CYCLE() calcOpticalFlowBM(frame0, frame1, block_size, shift_size, max_range, false, u, v);

        CPU_SANITY_CHECK(u);
        CPU_SANITY_CHECK(v);
    }
}

PERF_TEST_P(ImagePair, DISABLED_Video_FastOpticalFlowBM,
            Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(400);

    const cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    const cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    const cv::Size block_size(16, 16);
    const cv::Size shift_size(1, 1);
    const cv::Size max_range(16, 16);

    if (PERF_RUN_GPU())
    {
        const cv::gpu::GpuMat d_frame0(frame0);
        const cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat u, v;

        cv::gpu::FastOpticalFlowBM fastBM;

        TEST_CYCLE() fastBM(d_frame0, d_frame1, u, v, max_range.width, block_size.width);

        GPU_SANITY_CHECK(u, 2);
        GPU_SANITY_CHECK(v, 2);
    }
    else
    {
        FAIL_NO_CPU();
    }
}

//////////////////////////////////////////////////////
// FGDStatModel

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

DEF_PARAM_TEST_1(Video, string);

PERF_TEST_P(Video, Video_FGDStatModel,
            Values(string("gpu/video/768x576.avi")))
{
    const int numIters = 10;

    declare.time(60);

    const string inputFile = perf::TestBase::getDataPath(GetParam());

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_frame(frame);

        cv::gpu::FGDStatModel d_model(4);
        d_model.create(d_frame);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            d_frame.upload(frame);

            startTimer();
            if(!next())
                break;

            d_model.update(d_frame);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            d_frame.upload(frame);

            d_model.update(d_frame);
        }

        const cv::gpu::GpuMat background = d_model.background;
        const cv::gpu::GpuMat foreground = d_model.foreground;

        GPU_SANITY_CHECK(background, 1e-2, ERROR_RELATIVE);
        GPU_SANITY_CHECK(foreground, 1e-2, ERROR_RELATIVE);
    }
    else
    {
        IplImage ipl_frame = frame;
        cv::Ptr<CvBGStatModel> model(cvCreateFGDStatModel(&ipl_frame));

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            ipl_frame = frame;

            startTimer();
            if(!next())
                break;

            cvUpdateBGStatModel(&ipl_frame, model);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            ipl_frame = frame;

            cvUpdateBGStatModel(&ipl_frame, model);
        }

        const cv::Mat background = model->background;
        const cv::Mat foreground = model->foreground;

        CPU_SANITY_CHECK(background);
        CPU_SANITY_CHECK(foreground);
    }
}

#endif

//////////////////////////////////////////////////////
// MOG

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

DEF_PARAM_TEST(Video_Cn_LearningRate, string, MatCn, double);

PERF_TEST_P(Video_Cn_LearningRate, Video_MOG,
            Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
                    GPU_CHANNELS_1_3_4,
                    Values(0.0, 0.01)))
{
    const int numIters = 10;

    const string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    const int cn = GET_PARAM(1);
    const float learningRate = static_cast<float>(GET_PARAM(2));

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
        cv::gpu::MOG_GPU d_mog;
        cv::gpu::GpuMat foreground;

        d_mog(d_frame, foreground, learningRate);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
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

            startTimer();
            if(!next())
                break;

            d_mog(d_frame, foreground, learningRate);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
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

            d_mog(d_frame, foreground, learningRate);
        }

        GPU_SANITY_CHECK(foreground);
    }
    else
    {
        cv::BackgroundSubtractorMOG mog;
        cv::Mat foreground;

        mog(frame, foreground, learningRate);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
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

            startTimer();
            if(!next())
                break;

            mog(frame, foreground, learningRate);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
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

            mog(frame, foreground, learningRate);
        }

        CPU_SANITY_CHECK(foreground);
    }
}

#endif

//////////////////////////////////////////////////////
// MOG2

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

DEF_PARAM_TEST(Video_Cn, string, int);

PERF_TEST_P(Video_Cn, DISABLED_Video_MOG2,
            Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
                    GPU_CHANNELS_1_3_4))
{
    const int numIters = 10;

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
        cv::gpu::MOG2_GPU d_mog2;
        d_mog2.bShadowDetection = false;

        cv::gpu::GpuMat d_frame(frame);
        cv::gpu::GpuMat foreground;

        d_mog2(d_frame, foreground);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
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

            startTimer();
            if(!next())
                break;

            d_mog2(d_frame, foreground);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
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

            d_mog2(d_frame, foreground);
        }

        GPU_SANITY_CHECK(foreground);
    }
    else
    {
        cv::BackgroundSubtractorMOG2 mog2;
        mog2.set("detectShadows", false);

        cv::Mat foreground;

        mog2(frame, foreground);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
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

            startTimer();
            if(!next())
                break;

            mog2(frame, foreground);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
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

            mog2(frame, foreground);
        }

        CPU_SANITY_CHECK(foreground);
    }
}

#endif

//////////////////////////////////////////////////////
// MOG2GetBackgroundImage

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

PERF_TEST_P(Video_Cn, Video_MOG2GetBackgroundImage,
            Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"),
                    GPU_CHANNELS_1_3_4))
{
    const string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    const int cn = GET_PARAM(1);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    if (PERF_RUN_GPU())
    {
        cv::gpu::GpuMat d_frame;
        cv::gpu::MOG2_GPU d_mog2;
        cv::gpu::GpuMat d_foreground;

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

            d_mog2(d_frame, d_foreground);
        }

        cv::gpu::GpuMat background;

        TEST_CYCLE() d_mog2.getBackgroundImage(background);

        GPU_SANITY_CHECK(background, 1);
    }
    else
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;

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

            mog2(frame, foreground);
        }

        cv::Mat background;

        TEST_CYCLE() mog2.getBackgroundImage(background);

        CPU_SANITY_CHECK(background);
    }
}

#endif

//////////////////////////////////////////////////////
// GMG

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

DEF_PARAM_TEST(Video_Cn_MaxFeatures, string, MatCn, int);

PERF_TEST_P(Video_Cn_MaxFeatures, Video_GMG,
            Combine(Values(string("gpu/video/768x576.avi")),
                    GPU_CHANNELS_1_3_4,
                    Values(20, 40, 60)))
{
    const int numIters = 150;

    const std::string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    const int cn = GET_PARAM(1);
    const int maxFeatures = GET_PARAM(2);

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
        cv::gpu::GpuMat foreground;

        cv::gpu::GMG_GPU d_gmg;
        d_gmg.maxFeatures = maxFeatures;

        d_gmg(d_frame, foreground);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
        {
            cap >> frame;
            if (frame.empty())
            {
                cap.release();
                cap.open(inputFile);
                cap >> frame;
            }

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

            startTimer();
            if(!next())
                break;

            d_gmg(d_frame, foreground);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
        {
            cap >> frame;
            if (frame.empty())
            {
                cap.release();
                cap.open(inputFile);
                cap >> frame;
            }

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

            d_gmg(d_frame, foreground);
        }

        GPU_SANITY_CHECK(foreground);
    }
    else
    {
        cv::Mat foreground;
        cv::Mat zeros(frame.size(), CV_8UC1, cv::Scalar::all(0));

        cv::BackgroundSubtractorGMG gmg;
        gmg.set("maxFeatures", maxFeatures);
        gmg.initialize(frame.size(), 0.0, 255.0);

        gmg(frame, foreground);

        int i = 0;

        // collect performance data
        for (; i < numIters; ++i)
        {
            cap >> frame;
            if (frame.empty())
            {
                cap.release();
                cap.open(inputFile);
                cap >> frame;
            }

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            startTimer();
            if(!next())
                break;

            gmg(frame, foreground);

            stopTimer();
        }

        // process last frame in sequence to get data for sanity test
        for (; i < numIters; ++i)
        {
            cap >> frame;
            if (frame.empty())
            {
                cap.release();
                cap.open(inputFile);
                cap >> frame;
            }

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            gmg(frame, foreground);
        }

        CPU_SANITY_CHECK(foreground);
    }
}

#endif

//////////////////////////////////////////////////////
// VideoReader

#if defined(HAVE_NVCUVID) && BUILD_WITH_VIDEO_INPUT_SUPPORT

PERF_TEST_P(Video, DISABLED_Video_VideoReader, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(20);

    const string inputFile = perf::TestBase::getDataPath(GetParam());

    if (PERF_RUN_GPU())
    {
        cv::gpu::VideoReader_GPU d_reader(inputFile);
        ASSERT_TRUE( d_reader.isOpened() );

        cv::gpu::GpuMat frame;

        TEST_CYCLE_N(10) d_reader.read(frame);

        GPU_SANITY_CHECK(frame);
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

#if defined(HAVE_NVCUVID) && defined(WIN32)

PERF_TEST_P(Video, DISABLED_Video_VideoWriter, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(30);

    const string inputFile = perf::TestBase::getDataPath(GetParam());
    const string outputFile = cv::tempfile(".avi");

    const double FPS = 25.0;

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::Mat frame;

    if (PERF_RUN_GPU())
    {
        cv::gpu::VideoWriter_GPU d_writer;

        cv::gpu::GpuMat d_frame;

        for (int i = 0; i < 10; ++i)
        {
            reader >> frame;
            ASSERT_FALSE(frame.empty());

            d_frame.upload(frame);

            if (!d_writer.isOpened())
                d_writer.open(outputFile, frame.size(), FPS);

            startTimer(); next();
            d_writer.write(d_frame);
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
