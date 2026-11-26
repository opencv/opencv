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

static std::vector<uint8_t> make16BitBitfieldsBmp(uint32_t dib_size, uint32_t rmask, uint32_t gmask, uint32_t bmask, uint16_t pixel)
{
    auto put16 = [](std::vector<uint8_t>& v, size_t off, uint16_t val) {
        v[off] = static_cast<uint8_t>(val & 0xFF);
        v[off + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    };
    auto put32 = [](std::vector<uint8_t>& v, size_t off, uint32_t val) {
        v[off] = static_cast<uint8_t>(val & 0xFF);
        v[off + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        v[off + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
        v[off + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    };

    const uint32_t extra_masks = (dib_size == 40) ? 12u : 0u;
    const uint32_t pixel_offset = 14u + dib_size + extra_masks;
    const uint32_t row_bytes = 4u; // 2 px * 2 bytes = 4 bytes (already 4-byte aligned)
    const uint32_t total_size = pixel_offset + row_bytes * 2u;
    std::vector<uint8_t> buf(total_size, 0);

    // BITMAPFILEHEADER
    buf[0] = 'B'; buf[1] = 'M';
    put32(buf, 2, total_size);
    put32(buf, 10, pixel_offset);

    // DIB header
    put32(buf, 14, dib_size);
    put32(buf, 18, 2);           // width = 2
    put32(buf, 22, 2);           // height = 2 (bottom-up)
    put16(buf, 26, 1);           // planes = 1
    put16(buf, 28, 16);          // bpp = 16
    put32(buf, 30, 3);           // BI_BITFIELDS = 3
    put32(buf, 34, row_bytes * 2u);

    // Masks at offset 14 + 40 = 54 (works for both V1+12 extra bytes and V4/V5 embedded masks)
    put32(buf, 54, rmask);
    put32(buf, 58, gmask);
    put32(buf, 62, bmask);

    // 2x2 pixels
    for (int i = 0; i < 4; ++i)
        put16(buf, pixel_offset + i * 2u, pixel);
    return buf;
}

TEST(Imgcodecs_BMP, read_16bit_bitfields_v4_v5)
{
    // RGB565 pure red: 0xF800 -> BGR (0, 0, 248)
    // RGB555 pure green: 0x03E0 -> BGR (0, 248, 0)
    const uint32_t headers[] = { 40u, 108u, 124u };
    for (uint32_t dib_size : headers)
    {
        auto buf565 = make16BitBitfieldsBmp(dib_size, 0xF800u, 0x07E0u, 0x001Fu, 0xF800u);
        cv::Mat img565 = cv::imdecode(buf565, cv::IMREAD_COLOR);
        ASSERT_FALSE(img565.empty()) << "Failed RGB565 for dib_size=" << dib_size;
        EXPECT_EQ(cv::Size(2, 2), img565.size());
        EXPECT_EQ(CV_8UC3, img565.type());
        EXPECT_EQ(cv::Vec3b(0, 0, 248), img565.at<cv::Vec3b>(0, 0));

        auto buf555 = make16BitBitfieldsBmp(dib_size, 0x7C00u, 0x03E0u, 0x001Fu, 0x03E0u);
        cv::Mat img555 = cv::imdecode(buf555, cv::IMREAD_COLOR);
        ASSERT_FALSE(img555.empty()) << "Failed RGB555 for dib_size=" << dib_size;
        EXPECT_EQ(cv::Size(2, 2), img555.size());
        EXPECT_EQ(CV_8UC3, img555.type());
        EXPECT_EQ(cv::Vec3b(0, 248, 0), img555.at<cv::Vec3b>(0, 0));
    }
}


}} // namespace
