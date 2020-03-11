// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "test_precomp.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace opencv_test
{

G_API_OP(GBlurFrame, <GMat(GFrame)>, "test.blur_frame") {
    static GMatDesc outMeta(GMatDesc in) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVBlurFrame, GBlurFrame)
{
    static void run(const cv::Mat& in, cv::Mat& out) {
        cv::blur(in, out, cv::Size{3,3});
    }
};

struct GFrameTest : public ::testing::Test
{
    cv::Size sz{32,32};
    cv::Mat in_mat;
    cv::Mat out_mat;
    cv::Mat out_mat_ocv;

    GFrameTest()
        : in_mat(cv::Mat(sz, CV_8UC1))
        , out_mat(cv::Mat::zeros(sz, CV_8UC1))
        , out_mat_ocv(cv::Mat::zeros(sz, CV_8UC1))
    {
        cv::randn(in_mat, cv::Scalar::all(127.0f), cv::Scalar::all(40.f));
        cv::blur(in_mat, out_mat_ocv, cv::Size{3,3});
    }

    void check()
    {
        EXPECT_EQ(0, cvtest::norm(out_mat, out_mat_ocv, NORM_INF));
    }
};

TEST_F(GFrameTest, Input)
{
    cv::GFrame in;
    auto out = GBlurFrame::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    auto pkg = cv::gapi::kernels<OCVBlurFrame>();
    c.apply(cv::gin(in_mat), cv::gout(out_mat), cv::compile_args(pkg));

    check();
}

} // namespace opencv_test
