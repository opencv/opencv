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

    EXPECT_EQ(CAP_PROP_UNKNOWN, cap.get(CV__CAP_PROP_LATEST));

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

inline static std::string gstreamer_bunny_name_printer(const testing::TestParamInfo<gstreamer_bunny::ParamType>& info)
{
    std::ostringstream out;
    out << extToStringSafe(info.param);
    return out.str();
}

INSTANTIATE_TEST_CASE_P(videoio, gstreamer_bunny, testing::ValuesIn(bunny_params), gstreamer_bunny_name_printer);

static std::string gstEncoderPipeline(const std::string& encoder, const std::string& file)
{
    return "appsrc ! videoconvert ! " + encoder + " ! matroskamux ! filesink location=" + file;
}

static bool gstEncoderAvailable(const std::string& encoder)
{
    const string file = cv::tempfile(".mkv");
    VideoWriter writer;
    const bool ok = writer.open(gstEncoderPipeline(encoder, file), CAP_GSTREAMER, 0, 25, Size(320, 240));
    writer.release();
    remove(file.c_str());
    return ok;
}

static void writeGstFrames(VideoWriter& writer, Size size, int count, bool staticBackground)
{
    RNG rng(12345);
    Mat background(size, CV_8UC3);
    rng.fill(background, RNG::UNIFORM, 0, 255);
    for (int i = 0; i < count; i++)
    {
        Mat frame(size, CV_8UC3);
        if (staticBackground)
            background.copyTo(frame);
        else
            rng.fill(frame, RNG::UNIFORM, 0, 255);
        circle(frame, Point((i * 13) % size.width, (i * 7) % size.height), 40, Scalar::all(255), -1);
        writer.write(frame);
    }
}

static long fileSize(const std::string& file)
{
    std::ifstream fs(file.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return fs ? (long)fs.tellg() : -1;
}

static long writeWithGstEncoderParams(const std::string& file, const std::vector<int>& params, int count,
                                      bool staticBackground = false, double* readBack = NULL, int prop = -1)
{
    const Size size(320, 240);
    VideoWriter writer;
    if (!writer.open(gstEncoderPipeline("x264enc", file), CAP_GSTREAMER, 0, 25, size, params))
        return -1;
    if (readBack && prop >= 0)
        *readBack = writer.get(prop);
    writeGstFrames(writer, size, count, staticBackground);
    writer.release();
    return fileSize(file);
}

static int maxKeyFrameGap(const std::string& file)
{
    VideoCapture cap(file, CAP_FFMPEG, {CAP_PROP_FORMAT, -1});
    if (!cap.isOpened())
        return -1;
    int gap = 0, maxGap = 0;
    bool seenKeyFrame = false;
    while (cap.grab())
    {
        if (cap.get(CAP_PROP_LRF_HAS_KEY_FRAME) != 0)
        {
            maxGap = std::max(maxGap, gap);
            gap = 0;
            seenKeyFrame = true;
        }
        gap++;
    }
    return seenKeyFrame ? std::max(maxGap, gap) : -1;
}

TEST(videoio_gstreamer_encoder_props, bitrate_changes_size)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    const string lowFile = cv::tempfile(".mkv");
    const string highFile = cv::tempfile(".mkv");
    double readBack = -1;
    const long lowSize = writeWithGstEncoderParams(lowFile, {VIDEOWRITER_PROP_BITRATE, 200000}, 30,
                                                   false, &readBack, VIDEOWRITER_PROP_BITRATE);
    const long highSize = writeWithGstEncoderParams(highFile, {VIDEOWRITER_PROP_BITRATE, 4000000}, 30);
    ASSERT_GT(lowSize, 0);
    ASSERT_GT(highSize, 0);
    EXPECT_EQ(200000, (int)readBack);
    EXPECT_GT(highSize, lowSize);
    remove(lowFile.c_str());
    remove(highFile.c_str());
}

TEST(videoio_gstreamer_encoder_props, crf_changes_size)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    const string lowFile = cv::tempfile(".mkv");
    const string highFile = cv::tempfile(".mkv");
    double readBack = -1;
    const long lowSize = writeWithGstEncoderParams(lowFile, {VIDEOWRITER_PROP_CRF, 18}, 30,
                                                   false, &readBack, VIDEOWRITER_PROP_CRF);
    const long highSize = writeWithGstEncoderParams(highFile, {VIDEOWRITER_PROP_CRF, 40}, 30);
    ASSERT_GT(lowSize, 0);
    ASSERT_GT(highSize, 0);
    EXPECT_EQ(18, (int)readBack);
    EXPECT_GT(lowSize, highSize);
    remove(lowFile.c_str());
    remove(highFile.c_str());
}

TEST(videoio_gstreamer_encoder_props, gop_limits_key_frame_interval)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend is needed to read key frames");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    const string defaultFile = cv::tempfile(".mkv");
    const string gopFile = cv::tempfile(".mkv");
    double readBack = -1;
    ASSERT_GT(writeWithGstEncoderParams(defaultFile, std::vector<int>(), 30, true), 0);
    ASSERT_GT(writeWithGstEncoderParams(gopFile, {VIDEOWRITER_PROP_GOP_SIZE, 5}, 30,
                                        true, &readBack, VIDEOWRITER_PROP_GOP_SIZE), 0);
    EXPECT_EQ(5, (int)readBack);
    EXPECT_GT(maxKeyFrameGap(defaultFile), 5);
    const int gap = maxKeyFrameGap(gopFile);
    EXPECT_GT(gap, 0);
    EXPECT_LE(gap, 5);
    remove(defaultFile.c_str());
    remove(gopFile.c_str());
}

TEST(videoio_gstreamer_encoder_props, preset_round_trip)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("x264enc"))
        throw SkipTestException("x264enc is not available");

    const string file = cv::tempfile(".mkv");
    double readBack = -1;
    ASSERT_GT(writeWithGstEncoderParams(file, {VIDEOWRITER_PROP_PRESET, VIDEOWRITER_PRESET_ULTRAFAST}, 20,
                                        false, &readBack, VIDEOWRITER_PROP_PRESET), 0);
    EXPECT_EQ(VIDEOWRITER_PRESET_ULTRAFAST, (int)readBack);
    remove(file.c_str());
}

TEST(videoio_gstreamer_encoder_props, unsupported_property_fails_open)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");
    if (!gstEncoderAvailable("avenc_mpeg4"))
        throw SkipTestException("avenc_mpeg4 is not available");

    const string file = cv::tempfile(".mkv");
    VideoWriter writer;
    EXPECT_FALSE(writer.open(gstEncoderPipeline("avenc_mpeg4", file), CAP_GSTREAMER, 0, 25, Size(320, 240),
                             {VIDEOWRITER_PROP_PRESET, VIDEOWRITER_PRESET_VERYSLOW}));
    remove(file.c_str());
}

TEST(videoio_gstreamer_encoder_props, fourcc_path_encodes)
{
    if (!videoio_registry::hasBackend(CAP_GSTREAMER))
        throw SkipTestException("GStreamer backend was not found");

    const Size size(320, 240);
    const int count = 20;
    const string file = cv::tempfile(".mkv");
    VideoWriter writer;
    ASSERT_TRUE(writer.open(file, CAP_GSTREAMER, VideoWriter::fourcc('X', '2', '6', '4'), 25, size));
    writeGstFrames(writer, size, count, true);
    writer.release();

    const long rawSize = (long)size.area() * 3 / 2 * count;
    const long encodedSize = fileSize(file);
    ASSERT_GT(encodedSize, 0);
    EXPECT_LT(encodedSize, rawSize / 4);
    remove(file.c_str());

    const string nv12File = cv::tempfile(".mkv");
    VideoWriter nv12Writer;
    ASSERT_TRUE(nv12Writer.open(nv12File, CAP_GSTREAMER, VideoWriter::fourcc('X', '2', '6', '4'), 25, size,
                                {VIDEOWRITER_PROP_COLOR_SPACE, VideoWriter::fourcc('N', 'V', '1', '2')}));
    writeGstFrames(nv12Writer, size, count, true);
    nv12Writer.release();
    const long nv12Size = fileSize(nv12File);
    ASSERT_GT(nv12Size, 0);
    EXPECT_LT(nv12Size, rawSize / 4);
    remove(nv12File.c_str());
}


}} // namespace
