#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL
#ifdef HAVE_VIDEO_INPUT

namespace cvtest {
namespace ocl {

//////////////////////////// Mog2//////////////////////////

typedef tuple<string, int> VideoMOG2ParamType;
typedef TestBaseWithParam<VideoMOG2ParamType> MOG2_Apply;
typedef TestBaseWithParam<VideoMOG2ParamType> MOG2_GetBackgroundImage;

static void cvtFrameFmt(vector<Mat>& input, vector<Mat>& output)
{
    for(int i = 0; i< (int)(input.size()); i++)
    {
        cvtColor(input[i], output[i], COLOR_RGB2GRAY);
    }
}

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

OCL_PERF_TEST_P(MOG2_Apply, Mog2, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1,3)))
{
    VideoMOG2ParamType params = GetParam();

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
        Ptr<cv::BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
        mog2->setDetectShadows(false);
        u_foreground.release();
        for (int i = 0; i < nFrame; i++)
        {
            mog2->apply(frame_buffer[i], u_foreground);
        }
    }
    SANITY_CHECK(u_foreground);
}

OCL_PERF_TEST_P(MOG2_GetBackgroundImage, Mog2, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(3)))
{
    VideoMOG2ParamType params = GetParam();

    const string inputFile = getDataPath(get<0>(params));

    const int cn = get<1>(params);
    int nFrame = 5;

    vector<Mat> frame_buffer(nFrame);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());
    prepareData(cap, cn, frame_buffer);

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
    SANITY_CHECK(u_background);
}

}}// namespace cvtest::ocl

#endif
#endif
