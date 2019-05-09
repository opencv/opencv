// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "test_precomp.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace opencv_test
{

G_TYPED_KERNEL(GResize3c3p, <GMatP(GMat,Size,int)>, "test.resize3c3p") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar == false);
        return in.withSize(sz).asPlanar();
    }
};

G_TYPED_KERNEL(GResize3p3p, <GMatP(GMatP,Size,int)>, "test.resize3p3p") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar);
        return in.withSize(sz);
    }
};

static GMatDesc NV12toRGBoutMeta(GMatDesc inY, GMatDesc inUV)
{
    GAPI_Assert(inY.depth == CV_8U);
    GAPI_Assert(inUV.depth == CV_8U);
    GAPI_Assert(inY.chan == 1);
    GAPI_Assert(inY.planar == false);
    GAPI_Assert(inUV.chan == 2);
    GAPI_Assert(inUV.planar == false);
    GAPI_Assert(inY.size.width  == 2 * inUV.size.width);
    GAPI_Assert(inY.size.height == 2 * inUV.size.height);
    return inY.withType(CV_8U, 3);
}

G_TYPED_KERNEL(GNV12toRGBp, <GMatP(GMat,GMat)>, "test.nv12torgbp") {
    static GMatDesc outMeta(GMatDesc inY, GMatDesc inUV) {
        return NV12toRGBoutMeta(inY, inUV).asPlanar();
    }
};

static void toPlanar(const cv::Mat& in, cv::Mat& out)
{
    GAPI_Assert(out.depth() == in.depth());
    GAPI_Assert(out.channels() == 1);
    GAPI_Assert(in.channels() == 3);
    GAPI_Assert(out.cols == in.cols);
    GAPI_Assert(out.rows == 3*in.rows);

    std::vector<cv::Mat> outs(3);
    for (int i = 0; i < 3; i++) {
        outs[i] = out(cv::Rect(0, i*in.rows, in.cols, in.rows));
    }
    cv::split(in, outs);
}

GAPI_OCV_KERNEL(OCVResize3c3p, GResize3c3p)
{
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat& out)
    {
        cv::Mat resized_mat;
        cv::resize(in, resized_mat, out_sz, 0, 0, interp);

        std::vector<cv::Mat> outs(3);
        for (int i = 0; i < 3; i++) {
            outs[i] = out(cv::Rect(0, i*out_sz.height, out_sz.width, out_sz.height));
        }
        cv::split(resized_mat, outs);
    }
};

GAPI_OCV_KERNEL(OCVResize3p3p, GResize3p3p)
{
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat& out)
    {
        std::vector<cv::Mat> ins(3);
        std::vector<cv::Mat> outs(3);

        int inH = in.rows / 3;
        int inW = in.cols;
        int outH = out.rows / 3;
        int outW = out.cols;
        for (int i = 0; i < 3; i++) {
            ins [i] = in(cv::Rect(0, i*inH, inW, inH));
            outs[i] = out(cv::Rect(0, i*outH, outW, outH));
            cv::resize(ins[i], outs[i], out_sz, 0, 0, interp);
        }
    }
};

GAPI_OCV_KERNEL(OCVNV12toRGBp, GNV12toRGBp)
{
    static void run(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out)
    {
        cv::Mat rgb;
        cv::cvtColorTwoPlane(inY, inUV, rgb, cv::COLOR_YUV2RGB_NV12);
        toPlanar(rgb, out);
    }
};

struct PlanarTest : public TestWithParam <std::pair<cv::Size, cv::Size>> {};
TEST_P(PlanarTest, Resize3c3p)
{
    cv::Size in_sz, out_sz;
    std::tie(in_sz, out_sz) = GetParam();
    int interp = cv::INTER_NEAREST;

    cv::Mat in_mat = cv::Mat(in_sz, CV_8UC3);
    cv::randn(in_mat, cv::Scalar::all(127.0f), cv::Scalar::all(40.f));

    cv::Mat out_mat     = cv::Mat::zeros(out_sz.height*3, out_sz.width, CV_8UC1);
    cv::Mat out_mat_ocv = cv::Mat::zeros(out_sz.height*3, out_sz.width, CV_8UC1);

    cv::GMat in;
    auto out = GResize3c3p::on(in, out_sz, interp);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    auto pkg = cv::gapi::kernels<OCVResize3c3p>();
    c.apply(cv::gin(in_mat), cv::gout(out_mat), cv::compile_args(pkg));

    cv::Mat resized_mat;
    cv::resize(in_mat, resized_mat, out_sz, 0, 0, interp);
    toPlanar(resized_mat, out_mat_ocv);

    EXPECT_EQ(0, cv::countNonZero(out_mat != out_mat_ocv));
}

TEST_P(PlanarTest, Resize3p3p)
{
    cv::Size in_sz, out_sz;
    std::tie(in_sz, out_sz) = GetParam();
    int interp = cv::INTER_NEAREST;

    cv::Mat in_mat = cv::Mat(cv::Size{in_sz.width, in_sz.height*3}, CV_8UC1);
    cv::randn(in_mat, cv::Scalar::all(127.0f), cv::Scalar::all(40.f));

    cv::Mat out_mat     = cv::Mat::zeros(out_sz.height*3, out_sz.width, CV_8UC1);
    cv::Mat out_mat_ocv = cv::Mat::zeros(out_sz.height*3, out_sz.width, CV_8UC1);

    cv::GMatP in;
    auto out = GResize3p3p::on(in, out_sz, interp);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    auto pkg = cv::gapi::kernels<OCVResize3p3p>();

    c.compile(cv::descr_of(in_mat).asPlanar(3), cv::compile_args(pkg))
             (cv::gin(in_mat), cv::gout(out_mat));

    for (int i = 0; i < 3; i++) {
        const cv::Mat in_mat_roi = in_mat(cv::Rect(0,  i*in_sz.height,  in_sz.width,  in_sz.height));
        cv::Mat out_mat_roi = out_mat_ocv(cv::Rect(0, i*out_sz.height, out_sz.width, out_sz.height));
        cv::resize(in_mat_roi, out_mat_roi, out_sz, 0, 0, interp);
    }

    EXPECT_EQ(0, cv::countNonZero(out_mat != out_mat_ocv));
}

TEST_P(PlanarTest, Pipeline)
{
    cv::Size in_sz, out_sz;
    std::tie(in_sz, out_sz) = GetParam();
    int interp = cv::INTER_NEAREST;

    cv::Mat in_mat = cv::Mat(cv::Size{in_sz.width, in_sz.height*3/2}, CV_8UC1);
    cv::randn(in_mat, cv::Scalar::all(127.0f), cv::Scalar::all(40.f));

    cv::Size uv_sz(in_sz.width / 2, in_sz.height / 2);

    cv::Mat y_mat  = cv::Mat(in_sz, CV_8UC1, in_mat.data);
    cv::Mat uv_mat = cv::Mat(uv_sz, CV_8UC2, in_mat.data + in_mat.step1() * in_sz.height);

    cv::Mat out_mat     = cv::Mat::zeros(out_sz.height*3, out_sz.width, CV_8UC1);
    cv::Mat out_mat_ocv = cv::Mat::zeros(out_sz.height*3, out_sz.width, CV_8UC1);

    cv::GMat inY, inUV;
    auto out = GResize3p3p::on(GNV12toRGBp::on(inY, inUV), out_sz, interp);
    cv::GComputation c(cv::GIn(inY, inUV), cv::GOut(out));

    auto pkg = cv::gapi::kernels<OCVNV12toRGBp, OCVResize3p3p>();
    c.apply(cv::gin(y_mat, uv_mat), cv::gout(out_mat), cv::compile_args(pkg));

    cv::Mat rgb, resized_mat;
    cv::cvtColorTwoPlane(y_mat, uv_mat, rgb, cv::COLOR_YUV2RGB_NV12);
    cv::resize(rgb, resized_mat, out_sz, 0, 0, interp);
    toPlanar(resized_mat, out_mat_ocv);

    EXPECT_EQ(0, cv::countNonZero(out_mat != out_mat_ocv));
}

INSTANTIATE_TEST_CASE_P(Sanity, PlanarTest,
                        Values(std::make_pair(cv::Size{8, 8}, cv::Size{4, 4})
                              ,std::make_pair(cv::Size{960, 540}, cv::Size{224, 224})
                              ,std::make_pair(cv::Size{64, 64}, cv::Size{224, 224})
                              ));

} // namespace opencv_test
