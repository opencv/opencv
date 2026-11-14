// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Reference: https://www.kernel.org/doc/html/v4.8/media/v4l-drivers/vivid.html

// create 1 virtual device of type CAP (0x1) at /dev/video10
//   sudo modprobe vivid ndevs=1 node_types=0x1 vid_cap_nr=10
// make sure user have read/write access (e.g. via group 'video')
//   $ ls -l /dev/video10
//   crw-rw----+ 1 root video ... /dev/video10
// set environment variable:
//   export OPENCV_TEST_V4L2_VIVID_DEVICE=/dev/video10
// run v4l2 tests:
//   opencv_test_videoio --gtest_filter=*videoio_v4l2*


#ifdef HAVE_CAMV4L2

// #define DUMP_CAMERA_FRAME

#include "test_precomp.hpp"
#include <opencv2/core/utils/configuration.private.hpp>
#include <linux/videodev2.h>

// workarounds for older versions
#ifndef v4l2_fourcc_be
#define v4l2_fourcc_be(a, b, c, d) (v4l2_fourcc(a, b, c, d) | (1U << 31))
#endif
#ifndef V4L2_PIX_FMT_Y10
#define V4L2_PIX_FMT_Y10 v4l2_fourcc('Y', '1', '0', ' ')
#endif
#ifndef V4L2_PIX_FMT_Y12
#define V4L2_PIX_FMT_Y12 v4l2_fourcc('Y', '1', '2', ' ')
#endif
#ifndef V4L2_PIX_FMT_Y16
#define V4L2_PIX_FMT_Y16 v4l2_fourcc('Y', '1', '6', ' ')
#endif
#ifndef V4L2_PIX_FMT_Y16_BE
#define V4L2_PIX_FMT_Y16_BE v4l2_fourcc_be('Y', '1', '6', ' ')
#endif
#ifndef V4L2_PIX_FMT_ABGR32
#define V4L2_PIX_FMT_ABGR32 v4l2_fourcc('A', 'R', '2', '4')
#endif
#ifndef V4L2_PIX_FMT_XBGR32
#define V4L2_PIX_FMT_XBGR32 v4l2_fourcc('X', 'R', '2', '4')
#endif

#include "../src/cap_v4l_internal.hpp"


using namespace cv;

namespace opencv_test { namespace {

TEST(videoio_v4l2_payload, minimum_payload_size)
{
    const size_t pixels = 640u * 480u;
    EXPECT_EQ(pixels * 2, v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_YUYV, 640, 480));
    EXPECT_EQ(pixels, v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_GREY, 640, 480));
    EXPECT_EQ(pixels * 3 / 2, v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_YUV420, 640, 480));
    EXPECT_EQ(17u, v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_YUV420, 3, 3));
    EXPECT_EQ(18u, v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_YUV411P, 5, 2));
    EXPECT_EQ(0u, v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_MJPEG, 640, 480));
    EXPECT_EQ(0u, v4l2_internal::minimumPayloadSize(0, 640, 480));
}

TEST(videoio_v4l2_payload, rejects_incomplete_or_error_flagged_frames)
{
    const size_t minimumPayloadSize = v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_YUYV, 640, 480);
    EXPECT_TRUE(v4l2_internal::isFrameValid(false, minimumPayloadSize, minimumPayloadSize));
    EXPECT_FALSE(v4l2_internal::isFrameValid(false, 0, minimumPayloadSize)); // normalize before validating
    EXPECT_FALSE(v4l2_internal::isFrameValid(false, minimumPayloadSize - 1, minimumPayloadSize));
    EXPECT_FALSE(v4l2_internal::isFrameValid(true, minimumPayloadSize, minimumPayloadSize));
    EXPECT_FALSE(v4l2_internal::isFrameValid(false, 0, 0)); // empty variable-size frame
    EXPECT_TRUE(v4l2_internal::isFrameValid(false, 1, 0)); // nonempty variable-size frame
}

TEST(videoio_v4l2_payload, uses_format_size_when_fixed_payload_is_not_reported)
{
    const size_t minimumPayloadSize = v4l2_internal::minimumPayloadSize(V4L2_PIX_FMT_YUYV, 640, 480);
    const size_t mappedLength = minimumPayloadSize + 64;
    size_t payloadSize = 0;
    ASSERT_TRUE(v4l2_internal::getPlanePayloadSize(0, 0, mappedLength, mappedLength,
                                                 payloadSize, minimumPayloadSize));
    EXPECT_EQ(minimumPayloadSize, payloadSize);
    EXPECT_TRUE(v4l2_internal::isFrameValid(false, payloadSize, minimumPayloadSize));
    EXPECT_FALSE(v4l2_internal::isFrameValid(true, payloadSize, minimumPayloadSize));

    ASSERT_TRUE(v4l2_internal::getPlanePayloadSize(minimumPayloadSize - 1, 0, mappedLength, mappedLength,
                                                 payloadSize, minimumPayloadSize));
    EXPECT_FALSE(v4l2_internal::isFrameValid(false, payloadSize, minimumPayloadSize));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(0, 0, minimumPayloadSize - 1, minimumPayloadSize - 1,
                                                  payloadSize, minimumPayloadSize));
    ASSERT_TRUE(v4l2_internal::getPlanePayloadSize(0, 0, mappedLength, mappedLength, payloadSize));
    EXPECT_FALSE(v4l2_internal::isFrameValid(false, payloadSize, 0)); // no fallback for variable-size formats
}

TEST(videoio_v4l2_payload, validates_multiplanar_data_offsets)
{
    size_t payloadSize = 0;
    EXPECT_TRUE(v4l2_internal::getPlanePayloadSize(128, 16, 128, 128, payloadSize));
    EXPECT_EQ(112u, payloadSize);
    EXPECT_TRUE(v4l2_internal::getPlanePayloadSize(256, 32, 256, 256, payloadSize));
    EXPECT_EQ(224u, payloadSize);

    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(8, 16, 64, 64, payloadSize));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(129, 16, 128, 128, payloadSize));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(64, 16, 128, 64, payloadSize));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(80, 16, 128, 64, payloadSize));
    EXPECT_EQ(224u, payloadSize);
}

TEST(videoio_v4l2_payload, copies_unreported_fixed_payload_without_mapping_padding)
{
    const unsigned char source[] = { 0xFF, 0xFF, 1, 2, 3, 4, 0xEE, 0xEE };
    unsigned char destination[4] = {};
    size_t payloadSize = 0;
    ASSERT_TRUE(v4l2_internal::getPlanePayloadSize(0, 2, sizeof(source), sizeof(source), payloadSize, 4));
    ASSERT_EQ(4u, payloadSize);
    size_t destinationOffset = 0;
    ASSERT_TRUE(v4l2_internal::copyPlanePayload(destination, sizeof(destination), destinationOffset,
                                               source, sizeof(source), sizeof(source), payloadSize + 2, 2));
    EXPECT_EQ(4u, destinationOffset);
    for (size_t i = 0; i < sizeof(destination); ++i)
        EXPECT_EQ(i + 1, destination[i]);

    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(0, 2, 8, 8, payloadSize, 7));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(0, 9, 8, 8, payloadSize, 4));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(0, 2, 8, 7, payloadSize, 4));
    EXPECT_FALSE(v4l2_internal::getPlanePayloadSize(0, 2, 8, 8, payloadSize,
                                                  std::numeric_limits<size_t>::max()));
}

TEST(videoio_v4l2_payload, copies_multiplanar_payload_with_checked_offsets)
{
    const unsigned char source[] = { 0xFF, 1, 2, 3 };
    unsigned char destination[4] = {};
    size_t destinationOffset = 0;
    EXPECT_TRUE(v4l2_internal::copyPlanePayload(destination, sizeof(destination), destinationOffset,
                                                source, sizeof(source), sizeof(source), 4, 1));
    EXPECT_EQ(3u, destinationOffset);
    EXPECT_EQ(1, destination[0]);
    EXPECT_EQ(2, destination[1]);
    EXPECT_EQ(3, destination[2]);

    destinationOffset = 0;
    EXPECT_FALSE(v4l2_internal::copyPlanePayload(destination, 2, destinationOffset,
                                                 source, sizeof(source), sizeof(source), 4, 1));
    EXPECT_FALSE(v4l2_internal::copyPlanePayload(destination, sizeof(destination), destinationOffset,
                                                 source, sizeof(source), sizeof(source), 5, 1));
    EXPECT_EQ(0u, destinationOffset);
}

struct Format_Channels_Depth
{
    uint32_t pixel_format;
    uint8_t channels;
    uint8_t depth;
    float mul_width;
    float mul_height;
};

typedef testing::TestWithParam<Format_Channels_Depth> videoio_v4l2;

TEST_P(videoio_v4l2, formats)
{
    utils::Paths devs = utils::getConfigurationParameterPaths("OPENCV_TEST_V4L2_VIVID_DEVICE");
    if (devs.size() != 1)
    {
        throw SkipTestException("OPENCV_TEST_V4L2_VIVID_DEVICE is not set");
    }
    const string device = devs[0];
    const Size sz(640, 480);
    const Format_Channels_Depth params = GetParam();
    const Size esz(sz.width * params.mul_width, sz.height * params.mul_height);

    {
        // Case with RAW output
        VideoCapture cap;
        ASSERT_TRUE(cap.open(device, CAP_V4L2));
        // VideoCapture will set device's format automatically, vivid device will accept it
        ASSERT_TRUE(cap.set(CAP_PROP_FOURCC, params.pixel_format));
        ASSERT_TRUE(cap.set(CAP_PROP_CONVERT_RGB, false));
        for (size_t idx = 0; idx < 3; ++idx)
        {
            Mat img;
            EXPECT_TRUE(cap.grab());
            EXPECT_TRUE(cap.retrieve(img));
            if (params.pixel_format == V4L2_PIX_FMT_SRGGB8 ||
                params.pixel_format == V4L2_PIX_FMT_SBGGR8 ||
                params.pixel_format == V4L2_PIX_FMT_SGBRG8 ||
                params.pixel_format == V4L2_PIX_FMT_SGRBG8)
            {
                EXPECT_EQ((size_t)esz.area(), img.total());
            }
            else
            {
                EXPECT_EQ(esz, img.size());
            }
            EXPECT_EQ(params.channels, img.channels());
            EXPECT_EQ(params.depth, img.depth());
        }
    }
    {
        // case with BGR output
        VideoCapture cap;
        ASSERT_TRUE(cap.open(device, CAP_V4L2));
        // VideoCapture will set device's format automatically, vivid device will accept it
        ASSERT_TRUE(cap.set(CAP_PROP_FOURCC, params.pixel_format));
        for (size_t idx = 0; idx < 3; ++idx)
        {
            Mat img;
            EXPECT_TRUE(cap.grab());
            EXPECT_TRUE(cap.retrieve(img));
            EXPECT_EQ(sz, img.size());
            EXPECT_EQ(3, img.channels());
            EXPECT_EQ(CV_8U, img.depth());
#ifdef DUMP_CAMERA_FRAME
            std::string img_name = "frame_" + fourccToStringSafe(params.pixel_format);
            // V4L2 flag for big-endian formats
            if(params.pixel_format & (1 << 31))
                img_name += "-BE";
            cv::imwrite(img_name + ".png", img);
#endif
        }
    }
}

vector<Format_Channels_Depth> all_params = {
    { V4L2_PIX_FMT_YVU420, 1, CV_8U, 1.f, 1.5f },
    { V4L2_PIX_FMT_YUV420, 1, CV_8U, 1.f, 1.5f },
    { V4L2_PIX_FMT_NV12, 1, CV_8U, 1.f, 1.5f },
    { V4L2_PIX_FMT_NV21, 1, CV_8U, 1.f, 1.5f },
    { V4L2_PIX_FMT_YUV411P, 3, CV_8U, 1.f, 1.f },
//    { V4L2_PIX_FMT_MJPEG, 1, CV_8U, 1.f, 1.f },
//    { V4L2_PIX_FMT_JPEG, 1, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_YUYV, 2, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_UYVY, 2, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_SN9C10X, 3, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_SRGGB8, 1, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_SBGGR8, 1, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_SGBRG8, 1, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_SGRBG8, 1, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_RGB24, 3, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_Y16, 1, CV_16U, 1.f, 1.f },
    { V4L2_PIX_FMT_Y16_BE, 1, CV_16U, 1.f, 1.f },
    { V4L2_PIX_FMT_Y10, 1, CV_16U, 1.f, 1.f },
    { V4L2_PIX_FMT_GREY, 1, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_BGR24, 3, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_XBGR32, 4, CV_8U, 1.f, 1.f },
    { V4L2_PIX_FMT_ABGR32, 4, CV_8U, 1.f, 1.f },
};

inline static std::string param_printer(const testing::TestParamInfo<videoio_v4l2::ParamType>& info)
{
    return fourccToStringSafe(info.param.pixel_format);
}

INSTANTIATE_TEST_CASE_P(/*videoio_v4l2*/, videoio_v4l2, ValuesIn(all_params), param_printer);

TEST(videoio_ffmpeg, camera_index)
{
    utils::Paths devs = utils::getConfigurationParameterPaths("OPENCV_TEST_V4L2_VIVID_DEVICE");
    if (devs.size() != 1)
    {
        throw SkipTestException("OPENCV_TEST_V4L2_VIVID_DEVICE is not set");
    }
    VideoCapture cap;
    ASSERT_TRUE(cap.open(0, CAP_FFMPEG));
    Mat frame;
    ASSERT_TRUE(cap.read(frame));
    ASSERT_FALSE(frame.empty());
}

}} // opencv_test::<anonymous>::

#endif // HAVE_CAMV4L2
