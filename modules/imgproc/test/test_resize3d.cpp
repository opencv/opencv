// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ImgProc_Resize3D, Nearest_4D_Uint8)
{
    int sizes_in[4] = { 2, 2, 2, 2 };
    Mat src(4, sizes_in, CV_8U);
    // Fill src with known values
    uchar count = 0;
    for (int d = 0; d < 2; ++d)
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 2; ++w)
                for (int c = 0; c < 2; ++c)
                {
                    int idx[4] = { d, h, w, c };
                    src.at<uchar>(idx) = count++;
                }

    Mat dst;
    Vec3i dsize(4, 4, 4);
    resize3D(src, dst, dsize, 0, 0, 0, INTER_NEAREST);

    ASSERT_EQ(dst.dims, 4);
    EXPECT_EQ(dst.size[0], 4);
    EXPECT_EQ(dst.size[1], 4);
    EXPECT_EQ(dst.size[2], 4);
    EXPECT_EQ(dst.size[3], 2);

    // Check corners
    int idx_000[4] = { 0, 0, 0, 0 };
    int idx_000_c1[4] = { 0, 0, 0, 1 };
    EXPECT_EQ(dst.at<uchar>(idx_000), src.at<uchar>(idx_000));
    EXPECT_EQ(dst.at<uchar>(idx_000_c1), src.at<uchar>(idx_000_c1));

    int idx_last[4] = { 3, 3, 3, 0 };
    int src_last[4] = { 1, 1, 1, 0 };
    EXPECT_EQ(dst.at<uchar>(idx_last), src.at<uchar>(src_last));
}

TEST(ImgProc_Resize3D, Linear_4D_Float32)
{
    int sizes_in[4] = { 2, 2, 2, 1 };
    Mat src(4, sizes_in, CV_32F);

    // Fill with values: d*100 + h*10 + w
    for (int d = 0; d < 2; ++d)
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 2; ++w)
            {
                int idx[4] = { d, h, w, 0 };
                src.at<float>(idx) = (float)(d * 100 + h * 10 + w);
            }

    Mat dst;
    Vec3i dsize(3, 3, 3);
    resize3D(src, dst, dsize, 0, 0, 0, INTER_LINEAR);

    ASSERT_EQ(dst.dims, 4);
    EXPECT_EQ(dst.size[0], 3);
    EXPECT_EQ(dst.size[1], 3);
    EXPECT_EQ(dst.size[2], 3);
    EXPECT_EQ(dst.size[3], 1);

    // Center point (1, 1, 1) in 3x3x3 output should be the average of all 8 corners = (0 + 1 + 10 + 11 + 100 + 101 + 110 + 111)/8 = 55.5
    int center_idx[4] = { 1, 1, 1, 0 };
    EXPECT_NEAR(dst.at<float>(center_idx), 55.5f, 1e-4f);
}

TEST(ImgProc_Resize3D, LargeChannels_576)
{
    int cn = 576;
    int sizes_in[4] = { 2, 4, 4, cn };
    Mat src(4, sizes_in, CV_32F);

    for (int d = 0; d < 2; ++d)
        for (int h = 0; h < 4; ++h)
            for (int w = 0; w < 4; ++w)
                for (int c = 0; c < cn; ++c)
                {
                    int idx[4] = { d, h, w, c };
                    src.at<float>(idx) = (float)(c + 1.0f);
                }

    Mat dst;
    Vec3i dsize(4, 8, 8);
    resize3D(src, dst, dsize, 0, 0, 0, INTER_LINEAR);

    ASSERT_EQ(dst.dims, 4);
    EXPECT_EQ(dst.size[0], 4);
    EXPECT_EQ(dst.size[1], 8);
    EXPECT_EQ(dst.size[2], 8);
    EXPECT_EQ(dst.size[3], cn);

    // Since value is constant across spatial dims for each channel c, output should equal (c + 1)
    int test_idx[4] = { 2, 4, 4, 123 };
    EXPECT_NEAR(dst.at<float>(test_idx), 124.0f, 1e-3f);
}

TEST(ImgProc_Resize3D, UnitLengthAxes)
{
    int sizes_in[4] = { 4, 8, 8, 3 };
    Mat src(4, sizes_in, CV_8U, Scalar(128));

    Mat dst;
    Vec3i dsize(1, 4, 4);
    resize3D(src, dst, dsize, 0, 0, 0, INTER_LINEAR);

    ASSERT_EQ(dst.dims, 4);
    EXPECT_EQ(dst.size[0], 1);
    EXPECT_EQ(dst.size[1], 4);
    EXPECT_EQ(dst.size[2], 4);
    EXPECT_EQ(dst.size[3], 3);

    int test_idx[4] = { 0, 2, 2, 1 };
    EXPECT_EQ(dst.at<uchar>(test_idx), 128);
}

TEST(ImgProc_Resize3D, ScaleFactors)
{
    int sizes_in[4] = { 4, 4, 4, 2 };
    Mat src(4, sizes_in, CV_32F, Scalar(42.0f));

    Mat dst;
    resize3D(src, dst, Vec3i(0, 0, 0), 2.0, 2.0, 2.0, INTER_LINEAR);

    ASSERT_EQ(dst.dims, 4);
    EXPECT_EQ(dst.size[0], 8);
    EXPECT_EQ(dst.size[1], 8);
    EXPECT_EQ(dst.size[2], 8);
    EXPECT_EQ(dst.size[3], 2);
}

}} // namespace opencv_test::anonymous
