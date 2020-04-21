// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL
#ifdef HAVE_VIDEO_INPUT
#include "../perf_bgfg_utils.hpp"

namespace cvtest {
namespace ocl {

//////////////////////////// KNN//////////////////////////

typedef tuple<string, int> VideoKNNParamType;
typedef TestBaseWithParam<VideoKNNParamType> KNN_Apply;
typedef TestBaseWithParam<VideoKNNParamType> KNN_GetBackgroundImage;

using namespace opencv_test;

OCL_PERF_TEST_P(KNN_Apply, KNN, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1,3)))
{
    VideoKNNParamType params = GetParam();

    const string inputFile = getDataPath(get<0>(params));

    const int cn = get<1>(params);
    int nFrame = 5;

    vector<Mat> frame_buffer(nFrame);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());
    prepareData(cap, cn, frame_buffer);

    UMat u_foreground;

    OCL_TEST_CYCLE()
    {
        Ptr<cv::BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN();
        knn->setDetectShadows(false);
        u_foreground.release();
        for (int i = 0; i < nFrame; i++)
        {
            knn->apply(frame_buffer[i], u_foreground);
        }
    }
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(KNN_GetBackgroundImage, KNN, Values(
        std::make_pair<string, int>("gpu/video/768x576.avi", 5),
        std::make_pair<string, int>("gpu/video/1920x1080.avi", 5)))
{
    VideoKNNParamType params = GetParam();

    const string inputFile = getDataPath(get<0>(params));

    const int cn = 3;
    const int skipFrames = get<1>(params);
    int nFrame = 10;

    vector<Mat> frame_buffer(nFrame);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());
    prepareData(cap, cn, frame_buffer, skipFrames);

    UMat u_foreground, u_background;

    OCL_TEST_CYCLE()
    {
        Ptr<cv::BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN();
        knn->setDetectShadows(false);
        u_foreground.release();
        u_background.release();
        for (int i = 0; i < nFrame; i++)
        {
            knn->apply(frame_buffer[i], u_foreground);
        }
        knn->getBackgroundImage(u_background);
    }
#ifdef DEBUG_BGFG
    imwrite(format("fg_%d_%d_knn_ocl.png", frame_buffer[0].rows, cn), u_foreground.getMat(ACCESS_READ));
    imwrite(format("bg_%d_%d_knn_ocl.png", frame_buffer[0].rows, cn), u_background.getMat(ACCESS_READ));
#endif
    SANITY_CHECK_NOTHING();
}

}}// namespace cvtest::ocl

#endif
#endif
