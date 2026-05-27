// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Regression tests for the Sun Raster decoder (grfmt_sunras.cpp).
// These tests guard against:
//   - UBSan "load of invalid enum value" in SunRasterDecoder::readHeader()
//     (see https://github.com/opencv/opencv/issues/29150)
//   - Out-of-range SunRasType / SunRasMapType values causing UB via unchecked C-cast

#include "test_precomp.hpp"

#include <vector>

namespace opencv_test { namespace {

// ---------------------------------------------------------------------------
// Helper: build a minimal Sun Raster byte buffer with caller-supplied fields.
// Sun Raster header layout (all fields big-endian, 32-bit):
//   [0]  magic     = 0x59a66a95
//   [4]  width
//   [8]  height
//   [12] depth     (bits-per-pixel)
//   [16] length    (image data length, may be 0 for RAS_OLD)
//   [20] ras_type  (SunRasType)
//   [24] maptype   (SunRasMapType)
//   [28] maplength
// ---------------------------------------------------------------------------
static std::vector<uint8_t> makeSunRasHeader(
    uint32_t width, uint32_t height, uint32_t depth,
    uint32_t length, uint32_t ras_type, uint32_t maptype, uint32_t maplength)
{
    std::vector<uint8_t> buf(32);
    auto put32 = [&](size_t off, uint32_t v) {
        buf[off+0] = (v >> 24) & 0xff;
        buf[off+1] = (v >> 16) & 0xff;
        buf[off+2] = (v >>  8) & 0xff;
        buf[off+3] = (v      ) & 0xff;
    };
    put32( 0, 0x59a66a95u); // magic
    put32( 4, width);
    put32( 8, height);
    put32(12, depth);
    put32(16, length);
    put32(20, ras_type);
    put32(24, maptype);
    put32(28, maplength);
    return buf;
}

// ---------------------------------------------------------------------------
// Crash / UBSan regression — issue #29150
// Feeding an invalid maptype value (34077 / 0x851d) must NOT trigger UB;
// imdecode must return an empty Mat gracefully.
// ---------------------------------------------------------------------------
TEST(Imgcodecs_SunRaster, invalid_maptype_returns_empty_29150)
{
    // Crafted header from the original bug report:
    //   magic=0x59a66a95, width=0x10101, height=0x10000, depth=1, length=0x1000000
    //   ras_type=0 (RAS_OLD, valid), maptype=0x851d (34077, INVALID)
    const std::vector<uint8_t> image_data = {
        0x59,0xa6,0x6a,0x95, 0x01,0x01,0x00,0x00,
        0x00,0x01,0x00,0x00, 0x00,0x00,0x00,0x01,
        0x01,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,
        0x00,0x00,0x85,0x1d, 0xae,0x5b,0x8d,0xd5,
        0x9c,0x25,0x22,0x41, 0x51,0x92,0x13,0x14,0x33
    };

    cv::Mat result;
    ASSERT_NO_THROW(result = cv::imdecode(image_data, cv::IMREAD_REDUCED_GRAYSCALE_2));
    EXPECT_TRUE(result.empty()) << "imdecode must return empty Mat for invalid maptype";
}

// Invalid ras_type (value well outside [0,3]) must be rejected cleanly.
TEST(Imgcodecs_SunRaster, invalid_rastype_returns_empty)
{
    auto buf = makeSunRasHeader(/*w*/8, /*h*/8, /*depth*/8,
                                /*len*/0, /*ras_type*/0xFFFF, /*maptype*/0, /*mapllen*/0);
    cv::Mat result;
    ASSERT_NO_THROW(result = cv::imdecode(buf, cv::IMREAD_GRAYSCALE));
    EXPECT_TRUE(result.empty()) << "imdecode must return empty Mat for invalid ras_type";
}

// maptype = 2 is outside [0,1] (only RMT_NONE=0, RMT_EQUAL_RGB=1 are defined).
TEST(Imgcodecs_SunRaster, maptype_2_returns_empty)
{
    auto buf = makeSunRasHeader(8, 8, 8, 0, /*ras_type*/0, /*maptype*/2, 0);
    cv::Mat result;
    ASSERT_NO_THROW(result = cv::imdecode(buf, cv::IMREAD_GRAYSCALE));
    EXPECT_TRUE(result.empty()) << "imdecode must return empty Mat for maptype=2";
}

// maptype = UINT32_MAX is also invalid.
TEST(Imgcodecs_SunRaster, maptype_max_returns_empty)
{
    auto buf = makeSunRasHeader(8, 8, 8, 0, 0, 0xFFFFFFFFu, 0);
    cv::Mat result;
    ASSERT_NO_THROW(result = cv::imdecode(buf, cv::IMREAD_GRAYSCALE));
    EXPECT_TRUE(result.empty()) << "imdecode must return empty Mat for maptype=UINT_MAX";
}

// A truncated buffer (shorter than the 32-byte header) must not crash.
TEST(Imgcodecs_SunRaster, truncated_header_returns_empty)
{
    // Only 16 bytes — header read will run out of data.
    const std::vector<uint8_t> buf = {
        0x59,0xa6,0x6a,0x95, 0x00,0x00,0x00,0x08,
        0x00,0x00,0x00,0x08, 0x00,0x00,0x00,0x08
    };
    cv::Mat result;
    ASSERT_NO_THROW(result = cv::imdecode(buf, cv::IMREAD_GRAYSCALE));
    EXPECT_TRUE(result.empty()) << "imdecode must return empty Mat for truncated header";
}

// ---------------------------------------------------------------------------
// Sanity: a well-formed 8-bpp grayscale Sun Raster (RMT_NONE) still decodes.
// We build the full header + pixel data manually so the test has no file I/O.
// ---------------------------------------------------------------------------
TEST(Imgcodecs_SunRaster, valid_8bpp_grayscale_decodes)
{
    const int W = 4, H = 4;
    // RAS_OLD(0), maptype=RMT_NONE(0), maplength=0
    auto buf = makeSunRasHeader(W, H, 8, W*H, 0, 0, 0);

    // Sun Raster rows are padded to 16-bit boundary.
    // W=4: row_bytes = 4 (already even), no padding needed.
    for (int i = 0; i < W * H; ++i)
        buf.push_back(static_cast<uint8_t>(i * 16)); // arbitrary gray values

    cv::Mat result;
    ASSERT_NO_THROW(result = cv::imdecode(buf, cv::IMREAD_GRAYSCALE));
    EXPECT_FALSE(result.empty()) << "imdecode must succeed for a valid 8-bpp Sun Raster";
    EXPECT_EQ(result.cols, W);
    EXPECT_EQ(result.rows, H);
    EXPECT_EQ(result.type(), CV_8UC1);
}

}} // namespace opencv_test
