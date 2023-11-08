// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple< string, Size, Size, int > Param;
typedef testing::TestWithParam< Param > videoio_gstreamer;

TEST_P(videoio_gstreamer, read_check)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

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
        if (gray_frame.depth() == CV_16U)
        {
            gray_frame.convertTo(gray_frame, CV_8U, 255.0/65535);
        }

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

static const Param test_data[] = {
    make_tuple("video/x-raw, format=BGR"  , Size(640, 480), Size(640, 480), COLOR_BGR2RGB),
    make_tuple("video/x-raw, format=BGRA" , Size(640, 480), Size(640, 480), COLOR_BGRA2RGB),
    make_tuple("video/x-raw, format=RGBA" , Size(640, 480), Size(640, 480), COLOR_RGBA2RGB),
    make_tuple("video/x-raw, format=BGRx" , Size(640, 480), Size(640, 480), COLOR_BGRA2RGB),
    make_tuple("video/x-raw, format=RGBx" , Size(640, 480), Size(640, 480), COLOR_RGBA2RGB),
    make_tuple("video/x-raw, format=GRAY8", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=UYVY" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_UYVY),
    make_tuple("video/x-raw, format=YUY2" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_YUY2),
    make_tuple("video/x-raw, format=YVYU" , Size(640, 480), Size(640, 480), COLOR_YUV2RGB_YVYU),
    make_tuple("video/x-raw, format=NV12" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_NV12),
    make_tuple("video/x-raw, format=NV21" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_NV21),
    make_tuple("video/x-raw, format=YV12" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_YV12),
    make_tuple("video/x-raw, format=I420" , Size(640, 480), Size(640, 720), COLOR_YUV2RGB_I420),
    make_tuple("video/x-bayer"            , Size(640, 480), Size(640, 480), COLOR_BayerBG2RGB),
    make_tuple("jpegenc ! image/jpeg"     , Size(640, 480), Size(640, 480), COLOR_BGR2RGB),

    // unaligned cases, strides information must be used
    make_tuple("video/x-raw, format=BGR"  , Size(322, 242), Size(322, 242), COLOR_BGR2RGB),
    make_tuple("video/x-raw, format=GRAY8", Size(322, 242), Size(322, 242), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=NV12" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_NV12),
    make_tuple("video/x-raw, format=NV21" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_NV21),
    make_tuple("video/x-raw, format=YV12" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_YV12),
    make_tuple("video/x-raw, format=I420" , Size(322, 242), Size(322, 363), COLOR_YUV2RGB_I420),

    // 16 bit
    make_tuple("video/x-raw, format=GRAY16_LE", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
    make_tuple("video/x-raw, format=GRAY16_BE", Size(640, 480), Size(640, 480), COLOR_GRAY2RGB),
};

INSTANTIATE_TEST_CASE_P(videoio, videoio_gstreamer, testing::ValuesIn(test_data));

TEST(videoio_gstreamer, unsupported_pipeline)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    // could not link videoconvert0 to matroskamux0, matroskamux0 can't handle caps video/x-raw, format=(string)RGBA
    std::string pipeline = "appsrc ! videoconvert ! video/x-raw, format=(string)RGBA ! matroskamux ! filesink location=test.mkv";
    Size frame_size(640, 480);

    VideoWriter writer;
    EXPECT_NO_THROW(writer.open(pipeline, CAP_GSTREAMER, 0/*fourcc*/, 30/*fps*/, frame_size, true));
    EXPECT_FALSE(writer.isOpened());
    // no frames
    EXPECT_NO_THROW(writer.release());

}

TEST(videoio_gstreamer, gray16_writing)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    Size frame_size(320, 240);

    // generate a noise frame
    Mat frame = Mat(frame_size, CV_16U);
    randu(frame, 0, 65535);

    // generate a temp filename, and fix path separators to how GStreamer expects them
    cv::String temp_file = cv::tempfile(".raw");
    std::replace(temp_file.begin(), temp_file.end(), '\\', '/');

    // write noise frame to file using GStreamer
    std::ostringstream writer_pipeline;
    writer_pipeline << "appsrc ! filesink location=" << temp_file;
    std::vector<int> params {
        VIDEOWRITER_PROP_IS_COLOR, 0/*false*/,
        VIDEOWRITER_PROP_DEPTH, CV_16U
    };
    VideoWriter writer;
    ASSERT_NO_THROW(writer.open(writer_pipeline.str(), CAP_GSTREAMER, 0/*fourcc*/, 30/*fps*/, frame_size, params));
    ASSERT_TRUE(writer.isOpened());
    ASSERT_NO_THROW(writer.write(frame));
    ASSERT_NO_THROW(writer.release());

    // read noise frame back in
    Mat written_frame(frame_size, CV_16U);
    std::ifstream fs(temp_file, std::ios::in | std::ios::binary);
    fs.read((char*)written_frame.ptr(0), frame_size.width * frame_size.height * 2);
    ASSERT_TRUE(fs);
    fs.close();

    // compare to make sure it's identical
    EXPECT_EQ(0, cv::norm(frame, written_frame, NORM_INF));

    // remove temp file
    EXPECT_EQ(0, remove(temp_file.c_str()));
}

TEST(videoio_gstreamer, timeout_property)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    VideoCapture cap;
    cap.open("videotestsrc ! appsink", CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());
    const double default_timeout = 30000; // 30 seconds
    const double open_timeout = 5678; // 3 seconds
    const double read_timeout = 1234; // 1 second
    EXPECT_NEAR(default_timeout, cap.get(CAP_PROP_OPEN_TIMEOUT_MSEC), 1e-3);
    const double current_read_timeout = cap.get(CAP_PROP_READ_TIMEOUT_MSEC);
    const bool read_timeout_supported = current_read_timeout > 0.0;
    if (read_timeout_supported)
    {
        EXPECT_NEAR(default_timeout, current_read_timeout, 1e-3);
    }
    cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout);
    EXPECT_NEAR(open_timeout, cap.get(CAP_PROP_OPEN_TIMEOUT_MSEC), 1e-3);
    if (read_timeout_supported)
    {
        cap.set(CAP_PROP_READ_TIMEOUT_MSEC, read_timeout);
        EXPECT_NEAR(read_timeout, cap.get(CAP_PROP_READ_TIMEOUT_MSEC), 1e-3);
    }
}

//==============================================================================
// Seeking test with manual GStreamer pipeline
typedef testing::TestWithParam<string> gstreamer_bunny;

TEST_P(gstreamer_bunny, manual_seek)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    const string video_file = BunnyParameters::getFilename("." + GetParam());
    const string pipeline = "filesrc location=" + video_file + " ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1";
    const double target_pos = 3000.0;
    const double ms_per_frame = 1000.0 / BunnyParameters::getFps();
    VideoCapture cap;
    cap.open(pipeline, CAP_GSTREAMER);
    ASSERT_TRUE(cap.isOpened());
    Mat img;
    for (int i = 0; i < 10; i++)
    {
        cap >> img;
    }
    EXPECT_FALSE(img.empty());
    cap.set(CAP_PROP_POS_MSEC, target_pos);
    cap >> img;
    EXPECT_FALSE(img.empty());
    double actual_pos = cap.get(CAP_PROP_POS_MSEC);
    EXPECT_NEAR(actual_pos, target_pos, ms_per_frame);
}

static const string bunny_params[] = {
    // string("wmv"),
    string("mov"),
    string("mp4"),
    // string("mpg"),
    string("avi"),
    // string("h264"),
    // string("h265"),
    string("mjpg.avi")
};

INSTANTIATE_TEST_CASE_P(videoio, gstreamer_bunny, testing::ValuesIn(bunny_params));


}} // namespace
