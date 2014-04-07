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

#ifdef HAVE_OPENCV_LEGACY
#  include "opencv2/legacy.hpp"
#endif

#ifdef HAVE_CUDA

using namespace cvtest;

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

//////////////////////////////////////////////////////
// FGDStatModel

#if BUILD_WITH_VIDEO_INPUT_SUPPORT && defined(HAVE_OPENCV_LEGACY)

namespace cv
{
    template<> void DefaultDeleter<CvBGStatModel>::operator ()(CvBGStatModel* obj) const
    {
        cvReleaseBGStatModel(&obj);
    }
}

PARAM_TEST_CASE(FGDStatModel, cv::cuda::DeviceInfo, std::string)
{
    cv::cuda::DeviceInfo devInfo;
    std::string inputFile;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());

        inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + GET_PARAM(1);
    }
};

CUDA_TEST_P(FGDStatModel, Update)
{
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    IplImage ipl_frame = frame;
    cv::Ptr<CvBGStatModel> model(cvCreateFGDStatModel(&ipl_frame));

    cv::cuda::GpuMat d_frame(frame);
    cv::Ptr<cv::cuda::BackgroundSubtractorFGD> d_fgd = cv::cuda::createBackgroundSubtractorFGD();
    cv::cuda::GpuMat d_foreground, d_background;
    std::vector< std::vector<cv::Point> > foreground_regions;
    d_fgd->apply(d_frame, d_foreground);

    for (int i = 0; i < 5; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        ipl_frame = frame;
        int gold_count = cvUpdateBGStatModel(&ipl_frame, model);

        d_frame.upload(frame);
        d_fgd->apply(d_frame, d_foreground);
        d_fgd->getBackgroundImage(d_background);
        d_fgd->getForegroundRegions(foreground_regions);
        int count = (int) foreground_regions.size();

        cv::Mat gold_background = cv::cvarrToMat(model->background);
        cv::Mat gold_foreground = cv::cvarrToMat(model->foreground);

        ASSERT_MAT_NEAR(gold_background, d_background, 1.0);
        ASSERT_MAT_NEAR(gold_foreground, d_foreground, 0.0);
        ASSERT_EQ(gold_count, count);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_BgSegm, FGDStatModel, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("768x576.avi"))));

#endif

//////////////////////////////////////////////////////
// MOG

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

namespace
{
    IMPLEMENT_PARAM_CLASS(UseGray, bool)
    IMPLEMENT_PARAM_CLASS(LearningRate, double)
}

PARAM_TEST_CASE(MOG, cv::cuda::DeviceInfo, std::string, UseGray, LearningRate, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    std::string inputFile;
    bool useGray;
    double learningRate;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());

        inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + GET_PARAM(1);

        useGray = GET_PARAM(2);

        learningRate = GET_PARAM(3);

        useRoi = GET_PARAM(4);
    }
};

CUDA_TEST_P(MOG, Update)
{
    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    cv::Ptr<cv::BackgroundSubtractorMOG> mog = cv::cuda::createBackgroundSubtractorMOG();
    cv::cuda::GpuMat foreground = createMat(frame.size(), CV_8UC1, useRoi);

    cv::Ptr<cv::BackgroundSubtractorMOG> mog_gold = cv::createBackgroundSubtractorMOG();
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

        mog->apply(loadMat(frame, useRoi), foreground, learningRate);

        mog_gold->apply(frame, foreground_gold, learningRate);

        ASSERT_MAT_NEAR(foreground_gold, foreground, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_BgSegm, MOG, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("768x576.avi")),
    testing::Values(UseGray(true), UseGray(false)),
    testing::Values(LearningRate(0.0), LearningRate(0.01)),
    WHOLE_SUBMAT));

#endif

//////////////////////////////////////////////////////
// MOG2

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

namespace
{
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

//////////////////////////////////////////////////////
// GMG

PARAM_TEST_CASE(GMG, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
{
};

CUDA_TEST_P(GMG, Accuracy)
{
    const cv::cuda::DeviceInfo devInfo = GET_PARAM(0);
    cv::cuda::setDevice(devInfo.deviceID());
    const cv::Size size = GET_PARAM(1);
    const int depth = GET_PARAM(2);
    const int channels = GET_PARAM(3);
    const bool useRoi = GET_PARAM(4);

    const int type = CV_MAKE_TYPE(depth, channels);

    const cv::Mat zeros(size, CV_8UC1, cv::Scalar::all(0));
    const cv::Mat fullfg(size, CV_8UC1, cv::Scalar::all(255));

    cv::Mat frame = randomMat(size, type, 0, 100);
    cv::cuda::GpuMat d_frame = loadMat(frame, useRoi);

    cv::Ptr<cv::BackgroundSubtractorGMG> gmg = cv::cuda::createBackgroundSubtractorGMG();
    gmg->setNumFrames(5);
    gmg->setSmoothingRadius(0);

    cv::cuda::GpuMat d_fgmask = createMat(size, CV_8UC1, useRoi);

    for (int i = 0; i < gmg->getNumFrames(); ++i)
    {
        gmg->apply(d_frame, d_fgmask);

        // fgmask should be entirely background during training
        ASSERT_MAT_NEAR(zeros, d_fgmask, 0);
    }

    frame = randomMat(size, type, 160, 255);
    d_frame = loadMat(frame, useRoi);
    gmg->apply(d_frame, d_fgmask);

    // now fgmask should be entirely foreground
    ASSERT_MAT_NEAR(fullfg, d_fgmask, 0);
}

INSTANTIATE_TEST_CASE_P(CUDA_BgSegm, GMG, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8U), MatType(CV_16U), MatType(CV_32F)),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
