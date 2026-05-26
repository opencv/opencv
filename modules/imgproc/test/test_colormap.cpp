// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// All built-in colormap IDs
static const int all_colormaps[] = {
    COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_CIVIDIS, COLORMAP_COOL,
    COLORMAP_DEEPGREEN, COLORMAP_HOT, COLORMAP_HSV, COLORMAP_INFERNO,
    COLORMAP_JET, COLORMAP_MAGMA, COLORMAP_OCEAN, COLORMAP_PARULA,
    COLORMAP_PINK, COLORMAP_PLASMA, COLORMAP_RAINBOW, COLORMAP_SPRING,
    COLORMAP_SUMMER, COLORMAP_TURBO, COLORMAP_TWILIGHT,
    COLORMAP_TWILIGHT_SHIFTED, COLORMAP_VIRIDIS, COLORMAP_WINTER
};

// ---------------------------------------------------------------------------
// Test 1: All built-in colormaps run without error on a normal grayscale image
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, AllColormapsRun)
{
    Mat src(64, 64, CV_8UC1);
    // fill with a ramp 0..255
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            src.at<uchar>(i, j) = (uchar)((i * src.cols + j) * 255 / (src.rows * src.cols));

    for (int cm : all_colormaps)
    {
        Mat dst;
        EXPECT_NO_THROW(applyColorMap(src, dst, cm))
            << "colormap id=" << cm << " threw an exception";
        EXPECT_FALSE(dst.empty())   << "colormap id=" << cm << " produced empty output";
        EXPECT_EQ(dst.type(), CV_8UC3) << "colormap id=" << cm << " output is not CV_8UC3";
        EXPECT_EQ(dst.size(), src.size()) << "colormap id=" << cm << " output size mismatch";
    }
}

// ---------------------------------------------------------------------------
// Test 2: Output is CV_8UC3 regardless of input size
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, OutputType)
{
    Mat src(100, 200, CV_8UC1, Scalar(128));
    Mat dst;
    applyColorMap(src, dst, COLORMAP_JET);
    EXPECT_EQ(dst.type(), CV_8UC3);
    EXPECT_EQ(dst.rows, 100);
    EXPECT_EQ(dst.cols, 200);
}

// ---------------------------------------------------------------------------
// Test 3: 1x1 image (minimal size edge case)
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, SinglePixel)
{
    Mat src(1, 1, CV_8UC1, Scalar(0));
    Mat dst;
    EXPECT_NO_THROW(applyColorMap(src, dst, COLORMAP_JET));
    EXPECT_EQ(dst.rows, 1);
    EXPECT_EQ(dst.cols, 1);
    EXPECT_EQ(dst.type(), CV_8UC3);

    // pixel 0 and pixel 255 should produce different colors (non-trivial map)
    Mat src2(1, 1, CV_8UC1, Scalar(255));
    Mat dst2;
    applyColorMap(src2, dst2, COLORMAP_JET);
    EXPECT_NE(dst.at<Vec3b>(0,0), dst2.at<Vec3b>(0,0))
        << "JET(0) and JET(255) should produce different colors";
}

// ---------------------------------------------------------------------------
// Test 4: Invalid colormap ID throws cv::Exception
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, InvalidColormapThrows)
{
    Mat src(10, 10, CV_8UC1, Scalar(128));
    Mat dst;
    EXPECT_THROW(applyColorMap(src, dst, -1),      cv::Exception);
    EXPECT_THROW(applyColorMap(src, dst, 9999),    cv::Exception);
}

// ---------------------------------------------------------------------------
// Test 5: 3-channel input is accepted and produces 3-channel output
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, ThreeChannelInput)
{
    Mat src(32, 32, CV_8UC3, Scalar(100, 150, 200));
    Mat dst;
    EXPECT_NO_THROW(applyColorMap(src, dst, COLORMAP_HOT));
    EXPECT_EQ(dst.type(), CV_8UC3);
    EXPECT_EQ(dst.size(), src.size());
}

// ---------------------------------------------------------------------------
// Test 6: User-defined colormap (CV_8UC3 lookup table, 256 entries)
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, UserColorMapBGR)
{
    // Build a simple identity-like LUT: gray -> (v, v, v)
    Mat lut(256, 1, CV_8UC3);
    for (int i = 0; i < 256; i++)
        lut.at<Vec3b>(i, 0) = Vec3b((uchar)i, (uchar)i, (uchar)i);

    Mat src(32, 32, CV_8UC1);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            src.at<uchar>(i, j) = (uchar)(i * 8);

    Mat dst;
    EXPECT_NO_THROW(applyColorMap(src, dst, lut));
    EXPECT_EQ(dst.type(), CV_8UC3);

    // For an identity gray LUT, each output pixel should be (v, v, v)
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            uchar v = src.at<uchar>(i, j);
            Vec3b expected(v, v, v);
            EXPECT_EQ(dst.at<Vec3b>(i, j), expected)
                << "mismatch at (" << i << "," << j << ")";
        }
}

// ---------------------------------------------------------------------------
// Test 7: User-defined colormap with wrong size throws
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, UserColorMapWrongSizeThrows)
{
    Mat bad_lut(128, 1, CV_8UC3, Scalar(0, 0, 0));  // 128 entries, not 256
    Mat src(10, 10, CV_8UC1, Scalar(0));
    Mat dst;
    EXPECT_THROW(applyColorMap(src, dst, bad_lut), cv::Exception);
}

// ---------------------------------------------------------------------------
// Test 8: User-defined colormap with wrong type throws
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, UserColorMapWrongTypeThrows)
{
    Mat bad_lut(256, 1, CV_32FC3, Scalar(0, 0, 0));  // float, not uchar
    Mat src(10, 10, CV_8UC1, Scalar(0));
    Mat dst;
    EXPECT_THROW(applyColorMap(src, dst, bad_lut), cv::Exception);
}

// ---------------------------------------------------------------------------
// Test 9: Deterministic output — same input always gives same output
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, Deterministic)
{
    Mat src(32, 32, CV_8UC1);
    randu(src, 0, 256);

    Mat dst1, dst2;
    applyColorMap(src, dst1, COLORMAP_VIRIDIS);
    applyColorMap(src, dst2, COLORMAP_VIRIDIS);

    EXPECT_DOUBLE_EQ(0.0, cv::norm(dst1, dst2, NORM_L1))
        << "applyColorMap is not deterministic";
}

// ---------------------------------------------------------------------------
// Test 10: Known pixel values for COLORMAP_JET (ground truth / regression)
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, PixelValueCorrectness)
{
    // JET is a well-known colormap; these values are fixed by the LUT in colormap.cpp.
    // If someone accidentally edits the LUT table, this test will catch it.
    struct { uchar input; Vec3b expected_bgr; } cases[] = {
        { 0,   Vec3b(128,   0,   0) },  // JET(0)   = dark blue
        { 128, Vec3b(126, 255, 130) },  // JET(128) = green
        { 255, Vec3b(  0,   0, 128) },  // JET(255) = dark red
    };

    for (auto& c : cases)
    {
        Mat src(1, 1, CV_8UC1, Scalar(c.input));
        Mat dst;
        applyColorMap(src, dst, COLORMAP_JET);
        EXPECT_EQ(dst.at<Vec3b>(0, 0), c.expected_bgr)
            << "JET(" << (int)c.input << ") produced wrong color";
    }
}

// ---------------------------------------------------------------------------
// Test 11: Empty input Mat throws cv::Exception
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, EmptyInputThrows)
{
    Mat empty;
    Mat dst;
    EXPECT_THROW(applyColorMap(empty, dst, COLORMAP_JET), cv::Exception);
}

// ---------------------------------------------------------------------------
// Test 12: Zero-size Mat(0,0) should throw, not crash
// Previously caused SIGFPE (division by zero at cols=0 in ColorMap::operator()).
// Fixed by adding CV_Assert(!src.empty()) in colormap.cpp.
// ---------------------------------------------------------------------------
TEST(ApplyColorMap, ZeroSizeInputThrows)
{
    Mat zero(0, 0, CV_8UC1);
    Mat dst;
    EXPECT_THROW(applyColorMap(zero, dst, COLORMAP_JET), cv::Exception);
}

}} // namespace opencv_test
