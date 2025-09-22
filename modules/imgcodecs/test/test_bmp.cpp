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


}} // namespace
