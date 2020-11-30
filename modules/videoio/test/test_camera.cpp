// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Note: all tests here are DISABLED by default due specific requirements.
// Don't use #if 0 - these tests should be tested for compilation at least.
//
// Usage: opencv_test_videoio --gtest_also_run_disabled_tests --gtest_filter=*VideoIO_Camera*<tested case>*

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void test_readFrames(/*const*/ VideoCapture& capture, const int N = 100, Mat* lastFrame = NULL, bool testTimestamps = true)
{
    Mat frame;
    int64 time0 = cv::getTickCount();
    int64 sysTimePrev = time0;
    const double cvTickFreq = cv::getTickFrequency();

    double camTimePrev = 0.0;
    const double fps = capture.get(cv::CAP_PROP_FPS);
    const double framePeriod = fps == 0.0 ? 1. : 1.0 / fps;

    const bool validTickAndFps = cvTickFreq != 0 && fps != 0.;
    testTimestamps &= validTickAndFps;

    for (int i = 0; i < N; i++)
    {
        SCOPED_TRACE(cv::format("frame=%d", i));

        capture >> frame;
        const int64 sysTimeCurr = cv::getTickCount();
        const double camTimeCurr = capture.get(cv::CAP_PROP_POS_MSEC);
        ASSERT_FALSE(frame.empty());

        // Do we have a previous frame?
        if (i > 0 && testTimestamps)
        {
            const double sysTimeElapsedSecs = (sysTimeCurr - sysTimePrev) / cvTickFreq;
            const double camTimeElapsedSecs = (camTimeCurr - camTimePrev) / 1000.;

            // Check that the time between two camera frames and two system time calls
            // are within 1.5 frame periods of one another.
            //
            // 1.5x is chosen to accomodate for a dropped frame, and an additional 50%
            // to account for drift in the scale of the camera and system time domains.
            EXPECT_NEAR(sysTimeElapsedSecs, camTimeElapsedSecs, framePeriod * 1.5);
        }

        EXPECT_GT(cvtest::norm(frame, NORM_INF), 0) << "Complete black image has been received";

        sysTimePrev = sysTimeCurr;
        camTimePrev = camTimeCurr;
    }

    int64 time1 = cv::getTickCount();
    printf("Processed %d frames on %.2f FPS\n", N, (N * cvTickFreq) / (time1 - time0 + 1));
    if (lastFrame) *lastFrame = frame.clone();
}

TEST(DISABLED_VideoIO_Camera, basic)
{
    VideoCapture capture(0);
    ASSERT_TRUE(capture.isOpened());
    std::cout << "Camera 0 via " << capture.getBackendName() << " backend" << std::endl;
    std::cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << std::endl;
    test_readFrames(capture);
    capture.release();
}

TEST(DISABLED_VideoIO_Camera, validate_V4L2_MJPEG)
{
    VideoCapture capture(CAP_V4L2);
    ASSERT_TRUE(capture.isOpened());
    ASSERT_TRUE(capture.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G')));
    std::cout << "Camera 0 via " << capture.getBackendName() << " backend" << std::endl;
    std::cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << std::endl;
    int fourcc = (int)capture.get(CAP_PROP_FOURCC);
    std::cout << "FOURCC code: " << cv::format("0x%8x", fourcc) << std::endl;
    test_readFrames(capture);
    capture.release();
}

TEST(DISABLED_VideoIO_Camera, validate_V4L2_FrameSize)
{
    VideoCapture capture(CAP_V4L2);
    ASSERT_TRUE(capture.isOpened());
    std::cout << "Camera 0 via " << capture.getBackendName() << " backend" << std::endl;
    std::cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << std::endl;
    int fourcc = (int)capture.get(CAP_PROP_FOURCC);
    std::cout << "FOURCC code: " << cv::format("0x%8x", fourcc) << std::endl;
    test_readFrames(capture, 30);

    EXPECT_TRUE(capture.set(CAP_PROP_FRAME_WIDTH, 640));
    EXPECT_TRUE(capture.set(CAP_PROP_FRAME_HEIGHT, 480));
    std::cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << std::endl;
    Mat frame640x480;
    test_readFrames(capture, 30, &frame640x480);
    EXPECT_EQ(640, frame640x480.cols);
    EXPECT_EQ(480, frame640x480.rows);

    EXPECT_TRUE(capture.set(CAP_PROP_FRAME_WIDTH, 1280));
    EXPECT_TRUE(capture.set(CAP_PROP_FRAME_HEIGHT, 720));
    std::cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << std::endl;
    Mat frame1280x720;
    test_readFrames(capture, 30, &frame1280x720);
    EXPECT_EQ(1280, frame1280x720.cols);
    EXPECT_EQ(720, frame1280x720.rows);

    capture.release();
}

}} // namespace
