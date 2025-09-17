// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test { namespace {

struct VideoCaptureAPITests: TestWithParam<cv::VideoCaptureAPIs>
{
    void SetUp()
    {
        cv::VideoCaptureAPIs api = GetParam();
        if (!videoio_registry::hasBackend(api))
            throw SkipTestException("backend " + std::to_string(int(api)) + " was not found");

        string video_file = string(cvtest::TS::ptr()->get_data_path()) + "video/rotated_metadata.mp4";

        EXPECT_NO_THROW(cap.open(video_file, api));
        ASSERT_TRUE(cap.isOpened()) << "Can't open the video: " << video_file << " with backend " << api << std::endl;
    }

    void tearDown()
    {
        cap.release();
    }

    void orientationCheck(double angle, int width, int height)
    {
        EXPECT_EQ(angle, cap.get(CAP_PROP_ORIENTATION_META));
        EXPECT_EQ(width, (int)cap.get(CAP_PROP_FRAME_WIDTH));
        EXPECT_EQ(height, (int)cap.get(CAP_PROP_FRAME_HEIGHT));

        Mat frame;
        cap >> frame;

        ASSERT_EQ(width, frame.cols);
        ASSERT_EQ(height, frame.rows);
    }

    VideoCapture cap;
};

// Related issues:
// - https://github.com/opencv/opencv/issues/26795
// - https://github.com/opencv/opencv/issues/15499
TEST_P(VideoCaptureAPITests, mp4_orientation_default_auto)
{
    EXPECT_TRUE(cap.get(CAP_PROP_ORIENTATION_AUTO));
    orientationCheck(90., 270, 480);
}

TEST_P(VideoCaptureAPITests, mp4_orientation_forced)
{
    EXPECT_TRUE(cap.set(CAP_PROP_ORIENTATION_AUTO, false));
    orientationCheck(90., 480, 270);
}

TEST_P(VideoCaptureAPITests, mp4_orientation_switch)
{
    SCOPED_TRACE("Initial orientation with autorotation");
    orientationCheck(90., 270, 480);
    SCOPED_TRACE("Disabled autorotation");
    EXPECT_TRUE(cap.set(CAP_PROP_ORIENTATION_AUTO, false));
    EXPECT_FALSE(cap.get(CAP_PROP_ORIENTATION_AUTO));
    orientationCheck(90., 480, 270);
}


static cv::VideoCaptureAPIs supported_backends[] = {
#ifdef HAVE_AVFOUNDATION
    CAP_AVFOUNDATION,
#endif
    CAP_FFMPEG
};

inline static std::string VideoCaptureAPITests_name_printer(const testing::TestParamInfo<VideoCaptureAPITests::ParamType>& info)
{
    std::ostringstream out;
    out << getBackendNameSafe(info.param);
    return out.str();
}

INSTANTIATE_TEST_CASE_P(videoio, VideoCaptureAPITests, testing::ValuesIn(supported_backends), VideoCaptureAPITests_name_printer);

}} // namespace
