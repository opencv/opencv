// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Note: all tests here are DISABLED by default due specific requirements.
// Don't use #if 0 - these tests should be tested for compilation at least.
//
// Usage: opencv_test_videoio --gtest_also_run_disabled_tests --gtest_filter=*videoio_camera*<tested case>*

#include "test_precomp.hpp"

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

static char* get_cameras_list()
{
#ifndef WINRT
    return getenv("OPENCV_TEST_CAMERA_LIST");
#else
    return OPENCV_TEST_CAMERA_LIST;
#endif
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

TEST(DISABLED_videoio_camera, v4l_poll_timeout)
{
    //Two identical cameras
    //Test only >= two cameras
    char* datapath_dir = get_cameras_list();
    ASSERT_FALSE(datapath_dir == nullptr || datapath_dir[0] == '\0');
    std::vector<VideoCapture> VCM;
    int step = 0; string path;
    while(true)
    {
        if(datapath_dir[step] == ':' || datapath_dir[step] == '\0')
        {
            VCM.push_back(VideoCapture(path, CAP_V4L));
            path.clear();
            if(datapath_dir[step] != '\0')
                ++step;
        }
        if(datapath_dir[step] == '\0')
            break;
        path += datapath_dir[step];
        ++step;
    }
    Mat frame1;
    EXPECT_TRUE(VCM[0].set(CAP_PROP_FPS, 30));
    EXPECT_TRUE(VCM[1].set(CAP_PROP_FPS, 15));
    for(size_t vc = 0; vc < VCM.size(); ++vc)
    {
        ASSERT_TRUE(VCM[vc].isOpened());
        //launch cameras
        EXPECT_TRUE(VCM[vc].read(frame1));
    }
    std::vector<int> state(VCM.size());

    float EPSILON = 1;
    int ITERATION_COUNT = 10;//Number of steps

    for(int i = 0; i < ITERATION_COUNT; ++i)
    {
        int TIMEOUT = 10;
        TickMeter tm;

        tm.start();
        VideoCapture::waitAny(VCM, state, TIMEOUT);
        tm.stop();

        EXPECT_LE(tm.getTimeMilli(), TIMEOUT + EPSILON);

        TIMEOUT = 23;

        tm.reset();
        tm.start();
        VideoCapture::waitAny(VCM, state, TIMEOUT);
        tm.stop();

        EXPECT_LE(tm.getTimeMilli(), TIMEOUT + EPSILON);

        TIMEOUT = 30;

        tm.reset();
        tm.start();
        VideoCapture::waitAny(VCM, state, TIMEOUT);
        tm.stop();

        EXPECT_LE(tm.getTimeMilli(), TIMEOUT + EPSILON);
    }
}

typedef tuple<int, int> Size_FPS;
typedef testing::TestWithParam< Size_FPS > DISABLED_videoio_fps;

TEST_P(DISABLED_videoio_fps, v4l_poll_fps)
{
    //Two identical cameras
    //Test only >= two cameras
    char* datapath_dir = get_cameras_list();
    ASSERT_FALSE(datapath_dir == nullptr || datapath_dir[0] == '\0');
    std::vector<VideoCapture> VCM;
    int step = 0; string path;
    while(true)
    {
        if(datapath_dir[step] == ':' || datapath_dir[step] == '\0')
        {
            VCM.push_back(VideoCapture(path, CAP_V4L));
            path.clear();
            if(datapath_dir[step] != '\0')
                ++step;
        }
        if(datapath_dir[step] == '\0')
            break;
        path += datapath_dir[step];
        ++step;
    }
    Mat frame1;
    int FPS1 = get<0>(GetParam());
    int FPS2 = get<1>(GetParam());

    EXPECT_TRUE(VCM[0].set(CAP_PROP_FPS, FPS1));
    EXPECT_TRUE(VCM[1].set(CAP_PROP_FPS, FPS2));
    for(size_t vc = 0; vc < VCM.size(); ++vc)
    {
        ASSERT_TRUE(VCM[vc].isOpened());
        //launch cameras
        EXPECT_TRUE(VCM[vc].read(frame1));
    }
    std::cout << FPS1 << std::endl;
    std::cout << FPS2 << std::endl;

    std::vector<int> state(VCM.size());

    std::vector<int> countOfStates1000t0(2, 0);

    int ITERATION_COUNT = 100; //number of expected frames from all cameras
    int TIMEOUT = -1; // milliseconds

    for(int i = 0; i < ITERATION_COUNT; ++i)
    {
        VideoCapture::waitAny(VCM, state, TIMEOUT);

        EXPECT_EQ(VCM.size(), state.size());

        for(unsigned int j = 0; j < VCM.size(); ++j)
        {
            Mat frame;
            if(state[j] == CAP_CAM_READY)
            {
                 EXPECT_TRUE(VCM[j].retrieve(frame1));
                 ++countOfStates1000t0[j];
            }
        }
    }
    EXPECT_TRUE( fabs(((float)FPS1 / (float)FPS2 - (float)countOfStates1000t0[0] / (float)countOfStates1000t0[1])) < 1 );
}

INSTANTIATE_TEST_CASE_P(, DISABLED_videoio_fps, testing::Combine(testing::Values(30, 15), testing::Values(5, 15)));

}} // namespace
