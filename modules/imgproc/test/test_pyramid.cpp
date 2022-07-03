// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_PyrUp, pyrUp_regression_22184)
{
    Mat src(100, 100, CV_16UC3, Scalar::all(255));
    Mat dst(100 * 2 + 1, 100 * 2 + 1, CV_16UC3, Scalar::all(0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));
    double min_val = 0;
    minMaxLoc(dst, &min_val);
    ASSERT_GT(cvRound(min_val), 0);
}

}}  // namespace
