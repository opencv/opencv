// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_PyrDown, pyrDown_regression_custom)
{
    Mat src(16, 16, CV_16UC3, Scalar(0, 0, 0));
    {
        int swidth = src.cols;
        int sheight = src.rows;
        int cn = src.channels();
        int count = 0;
        ASSERT_NE(src.data, nullptr) << "src.data is null. Cannot proceed.";

        for (int y = 0; y < sheight; y++)
        {
            ushort *src_c = src.ptr<ushort>(y);
            ASSERT_NE(src_c, nullptr) << "src.ptr<ushort>(y) is null. Cannot proceed.";
            for (int x = 0; x < swidth * cn; x++)
            {
                src_c[x] = (count++) % 10;
            }
        }
    }

    Size dstSize((src.cols + 1) / 2, (src.rows + 1) / 2);
    Mat dst(dstSize, CV_16UC3, Scalar(0, 0, 0));
    ASSERT_NE(dst.data, nullptr) << "dst.data is null. Cannot proceed.";

    pyrDown(src, dst, dstSize);

    Mat srcp = src(Rect(0, 0, 14, 14));
    Size dstpSize((srcp.cols + 1) / 2, (srcp.rows + 1) / 2);
    Mat dstp(dstpSize, CV_16UC3, Scalar(0, 0, 0));
    ASSERT_NE(srcp.data, nullptr) << "srcp.data is null. Cannot proceed.";
    ASSERT_NE(dstp.data, nullptr) << "dstp.data is null. Cannot proceed.";

    pyrDown(srcp, dstp, dstpSize);

    Mat diff = dst(Rect(Point(), dstp.size())) - dstp;
    EXPECT_TRUE(cv::countNonZero(diff.reshape(1)) == 0) << "The difference between dst and dstp is not zero.";
}

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
