// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL
#include "../perf_bgfg_utils.hpp"

namespace opencv_test {
namespace ocl {

//////////////////////////// Mog2//////////////////////////

typedef tuple<string, int> VideoMOG2ParamType;
typedef TestBaseWithParam<VideoMOG2ParamType> MOG2_Apply;
typedef TestBaseWithParam<VideoMOG2ParamType> MOG2_GetBackgroundImage;

using namespace opencv_test;

OCL_PERF_TEST_P(MOG2_Apply, Mog2, Combine(Values("cv/video/768x576.avi", "cv/video/1920x1080.avi"), Values(1,3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = getDataPath(get<0>(params));

    const int cn = get<1>(params);
    int nFrame = 5;

    vector<Mat> frame_buffer(nFrame);

    cv::VideoCapture cap(inputFile);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");
    prepareData(cap, cn, frame_buffer);

    UMat u_foreground;

    OCL_TEST_CYCLE()
    {
        Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
        mog2->setDetectShadows(false);
        u_foreground.release();
        for (int i = 0; i < nFrame; i++)
        {
            mog2->apply(frame_buffer[i], u_foreground);
        }
    }
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(MOG2_GetBackgroundImage, Mog2, Values(
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
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");
    prepareData(cap, cn, frame_buffer, skipFrames);

    UMat u_foreground, u_background;

    OCL_TEST_CYCLE()
    {
        Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
        mog2->setDetectShadows(false);
        u_foreground.release();
        u_background.release();
        for (int i = 0; i < nFrame; i++)
        {
            mog2->apply(frame_buffer[i], u_foreground);
        }
        mog2->getBackgroundImage(u_background);
    }
#ifdef DEBUG_BGFG
    imwrite(format("fg_%d_%d_mog2_ocl.png", frame_buffer[0].rows, cn), u_foreground.getMat(ACCESS_READ));
    imwrite(format("bg_%d_%d_mog2_ocl.png", frame_buffer[0].rows, cn), u_background.getMat(ACCESS_READ));
#endif
    SANITY_CHECK_NOTHING();
}

}}// namespace opencv_test::ocl

#endif
