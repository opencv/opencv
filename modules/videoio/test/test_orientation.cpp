// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

typedef TestWithParam<cv::VideoCaptureAPIs> VideoCaptureAPITests;

static void videoOrientationCheck(cv::VideoCapture& cap, double angle, int width, int height)
{
    EXPECT_EQ(angle, cap.get(CAP_PROP_ORIENTATION_META));
    EXPECT_EQ(width, (int)cap.get(CAP_PROP_FRAME_WIDTH));
    EXPECT_EQ(height, (int)cap.get(CAP_PROP_FRAME_HEIGHT));

    Mat frame;
    cap >> frame;

    ASSERT_EQ(width, frame.cols);
    ASSERT_EQ(height, frame.rows);
}

// Related issues:
// - https://github.com/opencv/opencv/issues/26795
// - https://github.com/opencv/opencv/issues/15499
TEST_P(VideoCaptureAPITests, mp4_orientation_default_auto)
{
    cv::VideoCaptureAPIs api = GetParam();
    if (!videoio_registry::hasBackend(api))
        throw SkipTestException("backend " + std::to_string(int(api)) + " was not found");

    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/rotated_metadata.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, api));
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << api << std::endl;
    EXPECT_TRUE(cap.get(CAP_PROP_ORIENTATION_AUTO));

    videoOrientationCheck(cap, 90., 270, 480);
}

TEST_P(VideoCaptureAPITests, mp4_orientation_forced)
{
    cv::VideoCaptureAPIs api = GetParam();
    if (!videoio_registry::hasBackend(api))
        throw SkipTestException("backend " + std::to_string(int(api)) + " was not found");

    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/rotated_metadata.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, api));
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << api << std::endl;
    EXPECT_TRUE(cap.set(CAP_PROP_ORIENTATION_AUTO, false));

    videoOrientationCheck(cap, 90., 480, 270);
}

TEST_P(VideoCaptureAPITests, mp4_orientation_switch)
{
    cv::VideoCaptureAPIs api = GetParam();
    if (!videoio_registry::hasBackend(api))
        throw SkipTestException("backend " + std::to_string(int(api)) + " was not found");

    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/rotated_metadata.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, api));
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << api << std::endl;

    videoOrientationCheck(cap, 90., 270, 480);

    EXPECT_TRUE(cap.set(CAP_PROP_ORIENTATION_AUTO, false));
    EXPECT_FALSE(cap.get(CAP_PROP_ORIENTATION_AUTO));
    videoOrientationCheck(cap, 90., 480, 270);
}


INSTANTIATE_TEST_CASE_P(videoio, VideoCaptureAPITests,
                            testing::Values(
#if defined(__APPLE__)
                                            CAP_AVFOUNDATION,
#endif
                                            CAP_FFMPEG
                                            ));

}} // namespace
