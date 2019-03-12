// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#ifdef HAVE_GSTREAMER

namespace opencv_test
{

typedef tuple< string, Size, Size, int > Param;
typedef testing::TestWithParam< Param > Videoio_Gstreamer_Test;

TEST(Videoio_Gstreamer, 16_bit_convert)
{

    Size frame_size = Size(640,480);
    int count_frames = 10;
    string format = "video/x-raw, format=GRAY16_LE";

    // 16-bit capture pipeline
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << count_frames << " ! " << format;
    pipeline << ", width=" << frame_size.width << ", height=" << frame_size.height << " ! appsink";

    VideoCapture cap;

    ASSERT_NO_THROW(cap.open(pipeline.str(), CAP_GSTREAMER));
    ASSERT_TRUE(cap.isOpened());
    ASSERT_TRUE(cap.get(CAP_PROP_CONVERT_RGB));

    // Check initial result is 8-bit
    Mat frame;
    cap >> frame;
    EXPECT_EQ(frame.size(), frame_size);
    EXPECT_EQ(frame.depth(), CV_8U);
    EXPECT_EQ(frame.channels(), 3);

    // Check result is 16-bit
    cap.set(CAP_PROP_CONVERT_RGB, false);

    Mat frame_16;
    cap >> frame_16;
    EXPECT_EQ(frame_16.size(), frame_size);
    EXPECT_EQ(frame_16.depth(), CV_16U);
    EXPECT_EQ(frame_16.channels(), 1);

    // Test enabling RGB conversion again
    cap.set(CAP_PROP_CONVERT_RGB, true);
    cap >> frame;
    EXPECT_EQ(frame.size(), frame_size);
    EXPECT_EQ(frame.depth(), CV_8U);
    EXPECT_EQ(frame.channels(), 3);

    cap.release();
    ASSERT_FALSE(cap.isOpened());

}

TEST_P(Videoio_Gstreamer_Test, test_object_structure)
{
    string format    = get<0>(GetParam());
    Size frame_size  = get<1>(GetParam());
    Size mat_size    = get<2>(GetParam());
    int convertToRGB = get<3>(GetParam());
    int count_frames = 10;
    std::ostringstream pipeline;
    pipeline << "videotestsrc pattern=ball num-buffers=" << count_frames << " ! " << format;
    pipeline << ", width=" << frame_size.width << ", height=" << frame_size.height << " ! appsink";
    VideoCapture cap;
    ASSERT_NO_THROW(cap.open(pipeline.str(), CAP_GSTREAMER));
    ASSERT_TRUE(cap.isOpened());

    Mat buffer, decode_frame, gray_frame, rgb_frame;
    for (int i = 0; i < count_frames; ++i)
    {
        cap >> buffer;
        decode_frame = (format == "jpegenc ! image/jpeg") ? imdecode(buffer, IMREAD_UNCHANGED) : buffer;
        EXPECT_EQ(mat_size, decode_frame.size());

        cvtColor(decode_frame, rgb_frame, convertToRGB);
        cvtColor(rgb_frame, gray_frame, COLOR_RGB2GRAY);

        vector<Vec3f> circles;
        HoughCircles(gray_frame, circles, HOUGH_GRADIENT, 1, gray_frame.rows/16, 100, 30, 1, 30 );
        if (circles.size() == 1)
        {
            EXPECT_NEAR(18.5, circles[0][2], 1.0);
        }
        else
        {
            ADD_FAILURE() << "Found " << circles.size() << " on frame " << i ;
        }
    }
    {
        Mat frame;
        cap >> frame;
        EXPECT_TRUE(frame.empty());
    }
    cap.release();
    ASSERT_FALSE(cap.isOpened());
}

Param test_data[] = {
    make_tuple("video/x-raw, format=BGR"  , Size(640, 480), Size(640, 480), COLOR_BGR2RGB),
    make_tuple("video/x-raw, format=GRAY8", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=GRAY16_LE", Size(640, 480), Size(640, 480), COLOR_BGR2RGB),
    make_tuple("video/x-raw, format=UYVY" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_UYVY),
    make_tuple("video/x-raw, format=YUY2" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_YUY2),
    make_tuple("video/x-raw, format=YVYU" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_YVYU),
    make_tuple("video/x-raw, format=NV12" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_NV12),
    make_tuple("video/x-raw, format=NV21" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_NV21),
    make_tuple("video/x-raw, format=YV12" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_YV12),
    make_tuple("video/x-raw, format=I420" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_I420),
    make_tuple("video/x-bayer"            , Size(640, 480), Size(640, 480), COLOR_BayerBG2RGB),
    make_tuple("jpegenc ! image/jpeg"     , Size(640, 480), Size(640, 480), COLOR_BGR2RGB)
};

INSTANTIATE_TEST_CASE_P(videoio, Videoio_Gstreamer_Test, testing::ValuesIn(test_data));

} // namespace

#endif
