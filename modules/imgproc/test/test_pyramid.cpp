// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_PyrUp, pyrUp_regression_22184)
{
    Mat src(100,100,CV_16UC3,Scalar(255,255,255));
    Mat dst(100 * 2 + 1, 100 * 2 + 1, CV_16UC3, Scalar(0,0,0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));
    double min_val;
    minMaxLoc(dst, &min_val);
    ASSERT_GT(cvRound(min_val), 0);
}

TEST(Imgproc_PyrUp, pyrUp_regression_22195)
{
    Mat src(99, 99,CV_16UC3,Scalar(255,255,255));
    Mat dst(src.cols * 2 - 1, src.rows * 2 - 1, CV_16UC3, Scalar(0,0,0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));
    
    int dwidth = dst.cols;
    int dheight = dst.rows;
    int cn = dst.channels();
    ushort *dst_last = dst.ptr<ushort>(dheight - 1);
    ushort *dst_last2 = dst.ptr<ushort>(dheight - 3);
    for (int x = 0; x < dwidth * cn; x++)
    {
        ASSERT_EQ(dst_last[x], dst_last2[x]);
    }
}

}
}
