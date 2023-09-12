// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

typedef TestWithParam<cv::VideoCaptureAPIs> VideoCaptureAPITests;

// related issue: https://github.com/opencv/opencv/issues/15499
TEST_P(VideoCaptureAPITests, mp4_orientation_meta_auto)
{
    cv::VideoCaptureAPIs api = GetParam();
    if (!videoio_registry::hasBackend(api))
        throw SkipTestException("backend " + std::to_string(int(api)) + " was not found");

    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/rotated_metadata.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, api));
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << api << std::endl;

    // related issue: https://github.com/opencv/opencv/issues/22088
    EXPECT_EQ(90, cap.get(CAP_PROP_ORIENTATION_META));

    EXPECT_TRUE(cap.set(CAP_PROP_ORIENTATION_AUTO, true));

    Size actual;
    EXPECT_NO_THROW(actual = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                                    (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    EXPECT_EQ(270, actual.width);
    EXPECT_EQ(480, actual.height);

    Mat frame;

    cap >> frame;

    ASSERT_EQ(270, frame.cols);
    ASSERT_EQ(480, frame.rows);
}

// related issue: https://github.com/opencv/opencv/issues/15499
TEST_P(VideoCaptureAPITests, mp4_orientation_no_rotation)
{
    cv::VideoCaptureAPIs api = GetParam();
    if (!videoio_registry::hasBackend(api))
        throw SkipTestException("backend " + std::to_string(int(api)) + " was not found");

    string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/rotated_metadata.mp4";

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(video_file, api));
    cap.set(CAP_PROP_ORIENTATION_AUTO, 0);
    ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << api << std::endl;
    ASSERT_FALSE(cap.get(CAP_PROP_ORIENTATION_AUTO));

    Size actual;
    EXPECT_NO_THROW(actual = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                                    (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    EXPECT_EQ(480, actual.width);
    EXPECT_EQ(270, actual.height);

    Mat frame;

    cap >> frame;

    ASSERT_EQ(480, frame.cols);
    ASSERT_EQ(270, frame.rows);
}

INSTANTIATE_TEST_CASE_P(videoio, VideoCaptureAPITests, testing::Values(CAP_FFMPEG, CAP_AVFOUNDATION));

}} // namespace
