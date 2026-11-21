// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<Size> Imgproc_PyrUp_empty;

TEST_P(Imgproc_PyrUp_empty, no_crash)
{
    Mat src(GetParam(), CV_8UC1), dst;

    ASSERT_NO_THROW(pyrUp(src, dst));
    EXPECT_TRUE(dst.empty());
    // a non-empty destination size cannot be filled from an empty source
    EXPECT_THROW(pyrUp(src, dst, Size(1, 10)), cv::Exception);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgproc_PyrUp_empty, testing::Values(
    Size(0, 0), Size(0, 5), Size(5, 0)
));

TEST(Imgproc_PyrUp, pyrUp_regression_22184)
{
    Mat src(100,100,CV_16UC3,Scalar(255,255,255));
    Mat dst(100 * 2 + 1, 100 * 2 + 1, CV_16UC3, Scalar(0,0,0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));
    double min_val = 0;
    minMaxLoc(dst, &min_val);
    ASSERT_GT(cvRound(min_val), 0);
}

TEST(Imgproc_PyrUp, pyrUp_regression_22194)
{
    Mat src(13, 13,CV_16UC3,Scalar(0,0,0));
    {
        int swidth = src.cols;
        int sheight = src.rows;
        int cn = src.channels();
        int count = 0;
        for (int y = 0; y < sheight; y++)
        {
            ushort *src_c = src.ptr<ushort>(y);
            for (int x = 0; x < swidth * cn; x++)
            {
                src_c[x] = (count++) % 10;
            }
        }
    }
    Mat dst(src.cols * 2 - 1, src.rows * 2 - 1, CV_16UC3, Scalar(0,0,0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));

    {
        ushort *dst_c = dst.ptr<ushort>(dst.rows - 1);
        ASSERT_EQ(dst_c[0], 6);
        ASSERT_EQ(dst_c[1], 6);
        ASSERT_EQ(dst_c[2], 1);
    }
}

}
}
