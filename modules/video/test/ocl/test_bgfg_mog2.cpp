#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

#if defined(HAVE_XINE)     || \
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

namespace cvtest {
namespace ocl {

//////////////////////////Mog2_Update///////////////////////////////////

namespace
{
    IMPLEMENT_PARAM_CLASS(UseGray, bool)
    IMPLEMENT_PARAM_CLASS(DetectShadow, bool)
}

PARAM_TEST_CASE(Mog2_Update, UseGray, DetectShadow)
{
    bool useGray;
    bool detectShadow;
    virtual void SetUp()
    {
        useGray = GET_PARAM(0);
        detectShadow = GET_PARAM(1);
    }
};

OCL_TEST_P(Mog2_Update, Accuracy)
{
    string inputFile = string(TS::ptr()->get_data_path()) + "video/768x576.avi";
    VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    Ptr<BackgroundSubtractorMOG2> mog2_cpu = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractorMOG2> mog2_ocl = createBackgroundSubtractorMOG2();

    mog2_cpu->setDetectShadows(detectShadow);
    mog2_ocl->setDetectShadows(detectShadow);

    Mat frame, foreground;
    UMat u_foreground;

    for (int i = 0; i < 10; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        if (useGray)
        {
            Mat temp;
            cvtColor(frame, temp, COLOR_BGR2GRAY);
            swap(temp, frame);
        }

        OCL_OFF(mog2_cpu->apply(frame, foreground));
        OCL_ON (mog2_ocl->apply(frame, u_foreground));

        if (detectShadow)
            EXPECT_MAT_SIMILAR(foreground, u_foreground, 15e-3)
        else
            EXPECT_MAT_NEAR(foreground, u_foreground, 0);
    }
}

//////////////////////////Mog2_getBackgroundImage///////////////////////////////////

PARAM_TEST_CASE(Mog2_getBackgroundImage, DetectShadow)
{
    bool detectShadow;
    virtual void SetUp()
    {
        detectShadow = GET_PARAM(0);
    }
};

OCL_TEST_P(Mog2_getBackgroundImage, Accuracy)
{
    string inputFile = string(TS::ptr()->get_data_path()) + "video/768x576.avi";
    VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    Ptr<BackgroundSubtractorMOG2> mog2_cpu = createBackgroundSubtractorMOG2();
    Ptr<BackgroundSubtractorMOG2> mog2_ocl = createBackgroundSubtractorMOG2();

    mog2_cpu->setDetectShadows(detectShadow);
    mog2_ocl->setDetectShadows(detectShadow);

    Mat frame, foreground;
    UMat u_foreground;

    for (int i = 0; i < 10; ++i)
    {
        cap >> frame;
        ASSERT_FALSE(frame.empty());

        OCL_OFF(mog2_cpu->apply(frame, foreground));
        OCL_ON (mog2_ocl->apply(frame, u_foreground));
    }

    Mat background;
    OCL_OFF(mog2_cpu->getBackgroundImage(background));

    UMat u_background;
    OCL_ON (mog2_ocl->getBackgroundImage(u_background));

    EXPECT_MAT_NEAR(background, u_background, 1.0);
}

///////////////////////////////////////////////////////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(OCL_Video, Mog2_Update, Combine(
                                    Values(UseGray(true), UseGray(false)),
                                    Values(DetectShadow(true), DetectShadow(false)))
                           );

OCL_INSTANTIATE_TEST_CASE_P(OCL_Video, Mog2_getBackgroundImage, (Values(DetectShadow(true), DetectShadow(false)))
                           );

}}// namespace cvtest::ocl

    #endif
#endif