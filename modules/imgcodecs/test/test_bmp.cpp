// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

#include <vector>

namespace opencv_test { namespace {

// See https://github.com/opencv/opencv/issues/27789
// See https://github.com/opencv/opencv/issues/23233
TEST(Imgcodecs_BMP, encode_decode_over1GB_regression27789)
{
    applyTestTag( CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_LONG );

    // Create large Mat over 1GB
    // 20000 px * 18000 px *  24 bpp(3ch) = 1,080,000,000 bytes
    // 1 GiB                              = 1,073,741,824 bytes
    cv::Mat src(20000, 18000, CV_8UC3, cv::Scalar(0,0,0));

    // Encode large BMP file.
    std::vector<uint8_t> buf;
    bool ret = false;
    ASSERT_NO_THROW(ret = cv::imencode(".bmp", src, buf, {}));
    ASSERT_TRUE(ret);

    src.release(); // To reduce usage memory, it is needed.

    // Decode large BMP file.
    cv::Mat dst;
    ASSERT_NO_THROW(dst = cv::imdecode(buf, cv::IMREAD_COLOR));
    ASSERT_FALSE(dst.empty());
}

TEST(Imgcodecs_BMP, write_read_over1GB_regression27789)
{
    // tag CV_TEST_TAG_VERYLONG applied to skip on CI. The test writes ~1GB file.
    applyTestTag( CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_VERYLONG );
    string bmpFilename = cv::tempfile(".bmp"); // To remove it, test must use EXPECT_* instead of ASSERT_*.

    // Create large Mat over 1GB
    // 20000 px * 18000 px *  24 bpp(3ch) = 1,080,000,000 bytes
    // 1 GiB                              = 1,073,741,824 bytes
    cv::Mat src(20000, 18000, CV_8UC3, cv::Scalar(0,0,0));

    // Write large BMP file.
    bool ret = false;
    EXPECT_NO_THROW(ret = cv::imwrite(bmpFilename, src, {}));
    EXPECT_TRUE(ret);

    // Read large BMP file.
    cv::Mat dst;
    EXPECT_NO_THROW(dst = cv::imread(bmpFilename, cv::IMREAD_COLOR));
    EXPECT_FALSE(dst.empty());

    remove(bmpFilename.c_str());
}

// 2x2 BMP, 16 bpp, BI_BITFIELDS with RGB565 masks. With a BITMAPINFOHEADER (40 bytes) the masks
// follow the header, with a BITMAPV4HEADER (108 bytes) or BITMAPV5HEADER (124 bytes) they are
// header fields.
static std::vector<uchar> makeBitfields565Bmp(uint32_t headerSize)
{
    std::vector<uchar> buf;
    auto put16 = [&buf](uint32_t v) {
        buf.push_back((uchar)(v & 0xff));
        buf.push_back((uchar)((v >> 8) & 0xff));
    };
    auto put32 = [&put16](uint32_t v) {
        put16(v & 0xffff);
        put16(v >> 16);
    };
    const uint32_t pixelsSize = 8;
    const uint32_t offset = 14 + headerSize + (headerSize == 40 ? 12 : 0);

    buf.push_back('B'); buf.push_back('M');
    put32(offset + pixelsSize); put32(0); put32(offset);
    put32(headerSize); put32(2); put32(2); put16(1); put16(16);
    put32(3); // BI_BITFIELDS
    put32(pixelsSize); put32(2835); put32(2835); put32(0); put32(0);
    put32(0xF800); put32(0x07E0); put32(0x001F); // red, green and blue masks
    buf.resize(offset, 0); // rest of a V4/V5 header
    put16(0xF800); put16(0x07E0); // bottom row: red, green
    put16(0xF800); put16(0x07E0); // top row: red, green
    return buf;
}

TEST(Imgcodecs_BMP, read_16bit_bitfields_v4_v5_header)
{
    Mat expected(2, 2, CV_8UC3, Scalar(0, 0, 248)); // red
    expected.col(1).setTo(Scalar(0, 252, 0));      // green
    for (uint32_t headerSize : {40u, 108u, 124u})
    {
        SCOPED_TRACE(cv::format("header size %u", headerSize));
        const std::vector<uchar> buf = makeBitfields565Bmp(headerSize);
        Mat img;
        ASSERT_NO_THROW(img = cv::imdecode(buf, IMREAD_UNCHANGED));
        ASSERT_FALSE(img.empty());
        ASSERT_EQ(CV_8UC3, img.type());
        EXPECT_EQ(0, cvtest::norm(expected, img, NORM_INF));
    }
}


}} // namespace
