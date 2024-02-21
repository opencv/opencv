// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation


#include "test_precomp.hpp"

namespace opencv_test
{

TEST(GAPI_MatWithValue, Simple)
{
    cv::Size sz(2, 2);
    cv::Mat in_mat = cv::Mat::eye(sz, CV_8U);

    cv::GComputationT<cv::GMat(cv::GMat)> addEye([&](cv::GMat in) {
        return in + cv::GMat(cv::Mat::eye(sz, CV_8U));
    });

    cv::Mat out_mat(sz, CV_8U);
    addEye.apply(in_mat, out_mat);

    cv::Mat reference = in_mat*2;
    EXPECT_EQ(0, cvtest::norm(out_mat, reference, NORM_INF));
}

} // namespace opencv_test
