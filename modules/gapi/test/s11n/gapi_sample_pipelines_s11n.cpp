// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../test_precomp.hpp"

#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/s11n.hpp>

namespace opencv_test
{

TEST(S11N, Pipeline_Crop_Rect)
{
    cv::Rect rect_to{ 4,10,37,50 };
    cv::Size sz_in = cv::Size(1920, 1080);
    cv::Size sz_out = cv::Size(37, 50);
    cv::Mat in_mat = cv::Mat::eye(sz_in, CV_8UC1);
    cv::Mat out_mat_gapi(sz_out, CV_8UC1);
    cv::Mat out_mat_ocv(sz_out, CV_8UC1);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto out = cv::gapi::crop(in, rect_to);
    auto p = cv::gapi::serialize(cv::GComputation(in, out));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);
    c.apply(in_mat, out_mat_gapi);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = in_mat(rect_to);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}


TEST(S11N, Pipeline_Canny_Bool)
{
    const cv::Size sz_in(1280, 720);
    cv::GMat in;
    double thrLow = 120.0;
    double thrUp = 240.0;
    int apSize = 5;
    bool l2gr = true;
    cv::Mat in_mat = cv::Mat::eye(1280, 720, CV_8UC1);
    cv::Mat out_mat_gapi(sz_in, CV_8UC1);
    cv::Mat out_mat_ocv(sz_in, CV_8UC1);

    // G-API code //////////////////////////////////////////////////////////////
    auto out = cv::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);
    auto p = cv::gapi::serialize(cv::GComputation(in, out));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);
    c.apply(in_mat, out_mat_gapi);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Canny(in_mat, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

TEST(S11N, Pipeline_Not)
{
    cv::GMat in;
    auto p = cv::gapi::serialize(cv::GComputation(in, cv::gapi::bitwise_not(in)));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat ref_mat = ~in_mat;

    cv::Mat out_mat;
    c.apply(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = cv::Mat();
    auto cc = c.compile(cv::descr_of(in_mat));
    cc(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(S11N, Pipeline_Sum_Scalar)
{
    cv::GMat in;
    auto p = cv::gapi::serialize(cv::GComputation(in, cv::gapi::sum(in)));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Scalar ref_scl = cv::sum(in_mat);

    cv::Scalar out_scl;
    c.apply(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = cv::Scalar();
    auto cc = c.compile(cv::descr_of(in_mat));
    cc(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(S11N, Pipeline_BinaryOp)
{
    cv::GMat a, b;
    auto p = cv::gapi::serialize(cv::GComputation(a, b, cv::gapi::add(a, b)));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat ref_mat = (in_mat + in_mat);

    cv::Mat out_mat;
    c.apply(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = cv::Mat();
    auto cc = c.compile(cv::descr_of(in_mat), cv::descr_of(in_mat));
    cc(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(S11N, Pipeline_Binary_Sum_Scalar)
{
    cv::GMat a, b;
    auto p = cv::gapi::serialize(cv::GComputation(a, b, cv::gapi::sum(a + b)));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Scalar ref_scl = cv::sum(in_mat + in_mat);
    cv::Scalar out_scl;
    c.apply(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = cv::Scalar();
    auto cc = c.compile(cv::descr_of(in_mat), cv::descr_of(in_mat));
    cc(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(S11N, Pipeline_Sharpen)
{
    const cv::Size sz_in (1280, 720);
    const cv::Size sz_out( 640, 480);
    cv::Mat in_mat (sz_in,  CV_8UC3);
    in_mat = cv::Scalar(128, 33, 53);

    cv::Mat out_mat(sz_out, CV_8UC3);
    cv::Mat out_mat_y;
    cv::Mat out_mat_ocv(sz_out, CV_8UC3);

    float sharpen_coeffs[] = {
         0.0f, -1.f,  0.0f,
        -1.0f,  5.f, -1.0f,
         0.0f, -1.f,  0.0f
    };
    cv::Mat sharpen_kernel(3, 3, CV_32F, sharpen_coeffs);

    // G-API code //////////////////////////////////////////////////////////////
    cv::GMat in;
    auto vga     = cv::gapi::resize(in, sz_out);
    auto yuv     = cv::gapi::RGB2YUV(vga);
    auto yuv_p   = cv::gapi::split3(yuv);
    auto y_sharp = cv::gapi::filter2D(std::get<0>(yuv_p), -1, sharpen_kernel);
    auto yuv_new = cv::gapi::merge3(y_sharp, std::get<1>(yuv_p), std::get<2>(yuv_p));
    auto out     = cv::gapi::YUV2RGB(yuv_new);

    auto p = cv::gapi::serialize(cv::GComputation(cv::GIn(in), cv::GOut(y_sharp, out)));
    auto c = cv::gapi::deserialize<cv::GComputation>(p);
    c.apply(cv::gin(in_mat), cv::gout(out_mat_y, out_mat));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat smaller;
        cv::resize(in_mat, smaller, sz_out);

        cv::Mat yuv_mat;
        cv::cvtColor(smaller, yuv_mat, cv::COLOR_RGB2YUV);
        std::vector<cv::Mat> yuv_planar(3);
        cv::split(yuv_mat, yuv_planar);
        cv::filter2D(yuv_planar[0], yuv_planar[0], -1, sharpen_kernel);
        cv::merge(yuv_planar, yuv_mat);
        cv::cvtColor(yuv_mat, out_mat_ocv, cv::COLOR_YUV2RGB);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        cv::Mat diff = out_mat_ocv != out_mat;
        std::vector<cv::Mat> diffBGR(3);
        cv::split(diff, diffBGR);
        EXPECT_EQ(0, cvtest::norm(diffBGR[0], NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diffBGR[1], NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diffBGR[2], NORM_INF));
    }

    // Metadata check /////////////////////////////////////////////////////////
    {
        auto cc    = c.compile(cv::descr_of(in_mat));
        auto metas = cc.outMetas();
        ASSERT_EQ(2u, metas.size());

        auto out_y_meta = cv::util::get<cv::GMatDesc>(metas[0]);
        auto out_meta   = cv::util::get<cv::GMatDesc>(metas[1]);

        // Y-output
        EXPECT_EQ(CV_8U,   out_y_meta.depth);
        EXPECT_EQ(1,       out_y_meta.chan);
        EXPECT_EQ(640,     out_y_meta.size.width);
        EXPECT_EQ(480,     out_y_meta.size.height);

        // Final output
        EXPECT_EQ(CV_8U,   out_meta.depth);
        EXPECT_EQ(3,       out_meta.chan);
        EXPECT_EQ(640,     out_meta.size.width);
        EXPECT_EQ(480,     out_meta.size.height);
    }
}

TEST(S11N, Pipeline_CustomRGB2YUV)
{
    const cv::Size sz(1280, 720);
    const int INS = 3;
    std::vector<cv::Mat> in_mats(INS);
    for (auto i : ade::util::iota(INS))
    {
        in_mats[i].create(sz, CV_8U);
        cv::randu(in_mats[i], cv::Scalar::all(0), cv::Scalar::all(255));
    }

    const int OUTS = 3;
    std::vector<cv::Mat> out_mats_cv(OUTS);
    std::vector<cv::Mat> out_mats_gapi(OUTS);
    for (auto i : ade::util::iota(OUTS))
    {
        out_mats_cv[i].create(sz, CV_8U);
        out_mats_gapi[i].create(sz, CV_8U);
    }

    // G-API code //////////////////////////////////////////////////////////////
    {
        cv::GMat r, g, b;
        cv::GMat y = 0.299f*r + 0.587f*g + 0.114f*b;
        cv::GMat u = 0.492f*(b - y);
        cv::GMat v = 0.877f*(r - y);

        auto p = cv::gapi::serialize(cv::GComputation({r, g, b}, {y, u, v}));
        auto c = cv::gapi::deserialize<cv::GComputation>(p);
        c.apply(in_mats, out_mats_gapi);
    }

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat r = in_mats[0], g = in_mats[1], b = in_mats[2];
        cv::Mat y = 0.299f*r + 0.587f*g + 0.114f*b;
        cv::Mat u = 0.492f*(b - y);
        cv::Mat v = 0.877f*(r - y);

        out_mats_cv[0] = y;
        out_mats_cv[1] = u;
        out_mats_cv[2] = v;
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        const auto diff = [](cv::Mat m1, cv::Mat m2, int t) {
            return cv::abs(m1 - m2) > t;
        };

        // FIXME: Not bit-accurate even now!
        cv::Mat
            diff_y = diff(out_mats_cv[0], out_mats_gapi[0], 2),
            diff_u = diff(out_mats_cv[1], out_mats_gapi[1], 2),
            diff_v = diff(out_mats_cv[2], out_mats_gapi[2], 2);

        EXPECT_EQ(0, cvtest::norm(diff_y, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diff_u, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diff_v, NORM_INF));
    }
}

} // namespace opencv_test
