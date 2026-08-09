// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ImgProc_UnsharpMask, Uint8_BasicEdgeEnhancement)
{
    // Create a 100x100 step-edge image: left half = 50, right half = 200
    Mat src(100, 100, CV_8U, Scalar(50));
    src.colRange(50, 100).setTo(Scalar(200));

    Mat dst;
    unsharpMask(src, dst, 1.0, 1.0, 0.0);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst.type(), src.type());

    // Unsharp mask should enhance edge contrast:
    // Left side of edge (col 48-49) should dip below 50
    // Right side of edge (col 50-51) should rise above 200
    uchar left_val = dst.at<uchar>(50, 48);
    uchar right_val = dst.at<uchar>(50, 51);

    EXPECT_LE(left_val, 50);
    EXPECT_GE(right_val, 200);
}

TEST(ImgProc_UnsharpMask, Float32_Basic)
{
    Mat src(64, 64, CV_32F, Scalar(0.5f));
    src.colRange(32, 64).setTo(Scalar(1.0f));

    Mat dst;
    unsharpMask(src, dst, 1.5, 2.0, 0.0);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst.type(), src.type());

    float left_val = dst.at<float>(32, 30);
    float right_val = dst.at<float>(32, 33);

    EXPECT_LE(left_val, 0.5f);
    EXPECT_GE(right_val, 1.0f);
}

TEST(ImgProc_UnsharpMask, Thresholding)
{
    // Low contrast region (diff < 20) vs High contrast region (diff > 50)
    Mat src(50, 100, CV_8U, Scalar(100));
    // Low contrast bump (100 -> 105)
    src.rowRange(0, 25).colRange(50, 100).setTo(Scalar(105));
    // High contrast step (100 -> 200)
    src.rowRange(25, 50).colRange(50, 100).setTo(Scalar(200));

    Mat dst;
    unsharpMask(src, dst, 1.0, 1.0, 20.0);

    // Low contrast area should remain unaffected because diff < threshold (20)
    EXPECT_EQ(dst.at<uchar>(10, 48), 100);
    EXPECT_EQ(dst.at<uchar>(10, 52), 105);

    // High contrast area should be sharpened
    EXPECT_GE(dst.at<uchar>(35, 52), 200);
}

TEST(ImgProc_UnsharpMask, MultiChannel)
{
    Mat src(64, 64, CV_8UC3, Scalar(100, 150, 200));
    src.colRange(32, 64).setTo(Scalar(50, 100, 150));

    Mat dst;
    unsharpMask(src, dst, 1.0, 1.0, 0.0);

    ASSERT_EQ(dst.size(), src.size());
    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.channels(), 3);
}

}} // namespace opencv_test::anonymous
