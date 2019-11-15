// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Note: all tests here are DISABLED by default due specific requirements.
// Don't use #if 0 - these tests should be tested for compilation at least.
//
// Usage: opencv_test_videoio --gtest_also_run_disabled_tests --gtest_filter=*videoio_camera*<tested case>*

#include "test_precomp.hpp"
#include <opencv2/core/utils/configuration.private.hpp>

namespace opencv_test { namespace {

static void test_readFrames(/*const*/ VideoCapture& capture, const int N = 100, Mat* lastFrame = NULL)
{
    Mat frame;
    int64 time0 = cv::getTickCount();
    for (int i = 0; i < N; i++)
    {
        SCOPED_TRACE(cv::format("frame=%d", i));

        capture >> frame;
        ASSERT_FALSE(frame.empty());

        EXPECT_GT(cvtest::norm(frame, NORM_INF), 0) << "Complete black image has been received";
    }
    int64 time1 = cv::getTickCount();
    printf("Processed %d frames on %.2f FPS\n", N, (N * cv::getTickFrequency()) / (time1 - time0 + 1));
    if (lastFrame) *lastFrame = frame.clone();
}

TEST(DISABLED_videoio_camera, basic)
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

TEST(DISABLED_videoio_camera, v4l_read_mjpg)
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

//Following test if for capture device using PhysConn_Video_SerialDigital as crossbar input pin
TEST(DISABLED_videoio_camera, channel6)
{
    VideoCapture capture(0);
    ASSERT_TRUE(capture.isOpened());
    capture.set(CAP_PROP_CHANNEL, 6);
    std::cout << "Camera 0 via " << capture.getBackendName() << " backend" << std::endl;
    std::cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << std::endl;
    test_readFrames(capture);
    capture.release();
}

TEST(DISABLED_videoio_camera, v4l_read_framesize)
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


static
utils::Paths getTestCameras()
{
    static utils::Paths cameras = utils::getConfigurationParameterPaths("OPENCV_TEST_CAMERA_LIST");
    return cameras;
}

TEST(DISABLED_videoio_camera, waitAny_V4L)
{
    auto cameraNames = getTestCameras();
    if (cameraNames.empty())
       throw SkipTestException("No list of tested cameras. Use OPENCV_TEST_CAMERA_LIST parameter");

    const int totalFrames = 50; // number of expected frames (summary for all cameras)
    const int64 timeoutNS = 100 * 1000000;

    const Size frameSize(640, 480);
    const int fpsDefaultEven = 30;
    const int fpsDefaultOdd = 15;

    std::vector<VideoCapture> cameras;
    for (size_t i = 0; i < cameraNames.size(); ++i)
    {
        const auto& name = cameraNames[i];
        int fps = (int)utils::getConfigurationParameterSizeT(cv::format("OPENCV_TEST_CAMERA%d_FPS", (int)i).c_str(), (i & 1) ? fpsDefaultOdd : fpsDefaultEven);
        std::cout << "Camera[" << i << "] = '" << name << "', fps=" << fps << std::endl;
        VideoCapture cap(name, CAP_V4L);
        ASSERT_TRUE(cap.isOpened()) << name;
        EXPECT_TRUE(cap.set(CAP_PROP_FRAME_WIDTH, frameSize.width)) << name;
        EXPECT_TRUE(cap.set(CAP_PROP_FRAME_HEIGHT, frameSize.height)) << name;
        EXPECT_TRUE(cap.set(CAP_PROP_FPS, fps)) << name;
        //launch cameras
        Mat firstFrame;
        EXPECT_TRUE(cap.read(firstFrame));
        EXPECT_EQ(frameSize.width, firstFrame.cols);
        EXPECT_EQ(frameSize.height, firstFrame.rows);
        cameras.push_back(cap);
    }

    std::vector<size_t> frameFromCamera(cameraNames.size(), 0);
    {
        int counter = 0;
        std::vector<int> cameraReady;
        do
        {
            EXPECT_TRUE(VideoCapture::waitAny(cameras, cameraReady, timeoutNS));
            EXPECT_FALSE(cameraReady.empty());
            for (int idx : cameraReady)
            {
                //std::cout << "Reading frame from camera: " << idx << std::endl;
                ASSERT_TRUE(idx >= 0 && (size_t)idx < cameras.size()) << idx;
                VideoCapture& c = cameras[idx];
                Mat frame;
#if 1
                ASSERT_TRUE(c.retrieve(frame)) << idx;
#else
                ASSERT_TRUE(c.read(frame)) << idx;
#endif
                EXPECT_EQ(frameSize.width, frame.cols) << idx;
                EXPECT_EQ(frameSize.height, frame.rows) << idx;

                ++frameFromCamera[idx];
                ++counter;
            }
        }
        while(counter < totalFrames);
    }

    for (size_t i = 0; i < cameraNames.size(); ++i)
    {
        EXPECT_GT(frameFromCamera[i], (size_t)0) << i;
    }
}

}} // namespace
