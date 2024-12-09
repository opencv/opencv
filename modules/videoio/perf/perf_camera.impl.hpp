// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Not a standalone header.

#include <opencv2/core/utils/configuration.private.hpp>

namespace opencv_test {
using namespace perf;

static
utils::Paths getTestCameras()
{
    static utils::Paths cameras = utils::getConfigurationParameterPaths("OPENCV_TEST_PERF_CAMERA_LIST");
    return cameras;
}

PERF_TEST(VideoCapture_Camera, waitAny_V4L)
{
    auto cameraNames = getTestCameras();
    if (cameraNames.empty())
       throw SkipTestException("No list of tested cameras. Use OPENCV_TEST_PERF_CAMERA_LIST parameter");

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

    TEST_CYCLE()
    {
        int counter = 0;
        std::vector<int> cameraReady;
        do
        {
            EXPECT_TRUE(VideoCapture::waitAny(cameras, cameraReady, timeoutNS));
            EXPECT_FALSE(cameraReady.empty());
            for (int idx : cameraReady)
            {
                VideoCapture& c = cameras[idx];
                Mat frame;
                ASSERT_TRUE(c.retrieve(frame));
                EXPECT_EQ(frameSize.width, frame.cols);
                EXPECT_EQ(frameSize.height, frame.rows);

                ++counter;
            }
        }
        while(counter < totalFrames);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
