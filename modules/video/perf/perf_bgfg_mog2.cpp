// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

#ifdef HAVE_VIDEO_INPUT
#include "perf_bgfg_utils.hpp"

namespace opencv_test { namespace {

//////////////////////////// Mog2//////////////////////////

typedef tuple<std::string, int> VideoMOG2ParamType;
typedef TestBaseWithParam<VideoMOG2ParamType> MOG2_Apply;
typedef TestBaseWithParam<VideoMOG2ParamType> MOG2_GetBackgroundImage;

PERF_TEST_P(MOG2_Apply, Mog2, Combine(Values("cv/video/768x576.avi", "cv/video/1920x1080.avi"), Values(1,3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = getDataPath(get<0>(params));

    const int cn = get<1>(params);
    int nFrame = 5;

    vector<Mat> frame_buffer(nFrame);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());
    prepareData(cap, cn, frame_buffer);

    Mat foreground;

    TEST_CYCLE()
    {
        Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
        mog2->setDetectShadows(false);
        foreground.release();
        for (int i = 0; i < nFrame; i++)
        {
            mog2->apply(frame_buffer[i], foreground);
        }
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MOG2_GetBackgroundImage, Mog2, Values(
        std::make_pair<string, int>("cv/video/768x576.avi", 5),
        std::make_pair<string, int>("cv/video/1920x1080.avi", 5)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = getDataPath(get<0>(params));

    const int cn = 3;
    const int skipFrames = get<1>(params);
    int nFrame = 10;

    vector<Mat> frame_buffer(nFrame);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());
    prepareData(cap, cn, frame_buffer, skipFrames);

    Mat foreground, background;

    TEST_CYCLE()
    {
        Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
        mog2->setDetectShadows(false);
        foreground.release();
        background.release();
        for (int i = 0; i < nFrame; i++)
        {
            mog2->apply(frame_buffer[i], foreground);
        }
        mog2->getBackgroundImage(background);
    }
#ifdef DEBUG_BGFG
    imwrite(format("fg_%d_%d_mog2.png", frame_buffer[0].rows, cn), foreground);
    imwrite(format("bg_%d_%d_mog2.png", frame_buffer[0].rows, cn), background);
#endif
    SANITY_CHECK_NOTHING();
}

}}// namespace

#endif
