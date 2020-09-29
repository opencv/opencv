// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "test_precomp.hpp"

#include <opencv2/gapi/media.hpp>

////////////////////////////////////////////////////////////////////////////////
// cv::GFrame tests

namespace opencv_test {

G_API_OP(GBlurFrame, <GMat(GFrame)>, "test.blur_frame") {
    static GMatDesc outMeta(GMatDesc in) {
        return in;
    }
};

GAPI_OCV_KERNEL(OCVBlurFrame, GBlurFrame) {
    static void run(const cv::Mat& in, cv::Mat& out) {
        cv::blur(in, out, cv::Size{3,3});
    }
};

struct GFrameTest : public ::testing::Test {
    cv::Size sz{32,32};
    cv::Mat in_mat;
    cv::Mat out_mat;
    cv::Mat out_mat_ocv;

    GFrameTest()
        : in_mat(cv::Mat(sz, CV_8UC1))
        , out_mat(cv::Mat::zeros(sz, CV_8UC1))
        , out_mat_ocv(cv::Mat::zeros(sz, CV_8UC1)) {
        cv::randn(in_mat, cv::Scalar::all(127.0f), cv::Scalar::all(40.f));
        cv::blur(in_mat, out_mat_ocv, cv::Size{3,3});
    }

    void check() {
        EXPECT_EQ(0, cvtest::norm(out_mat, out_mat_ocv, NORM_INF));
    }
};

TEST_F(GFrameTest, Input) {
    cv::GFrame in;
    auto out = GBlurFrame::on(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    auto pkg = cv::gapi::kernels<OCVBlurFrame>();
    c.apply(cv::gin(in_mat), cv::gout(out_mat), cv::compile_args(pkg));

    check();
}

////////////////////////////////////////////////////////////////////////////////
// cv::MediaFrame tests
namespace {
class TestMediaBGR final: public cv::MediaFrame::IAdapter {
    cv::Mat m_mat;
    using Cb = cv::MediaFrame::View::Callback;
    Cb m_cb;

public:
    explicit TestMediaBGR(cv::Mat m, Cb cb = [](){})
        : m_mat(m), m_cb(cb) {
    }
    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{cv::MediaFormat::BGR, cv::Size(m_mat.cols, m_mat.rows)};
    }
    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return cv::MediaFrame::View(std::move(pp), std::move(ss), Cb{m_cb});
    }
};

class TestMediaNV12 final: public cv::MediaFrame::IAdapter {
    cv::Mat m_y;
    cv::Mat m_uv;
public:
    TestMediaNV12(cv::Mat y, cv::Mat uv) : m_y(y), m_uv(uv) {
    }
    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{cv::MediaFormat::NV12, cv::Size(m_y.cols, m_y.rows)};
    }
    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = {
            m_y.ptr(), m_uv.ptr(), nullptr, nullptr
        };
        cv::MediaFrame::View::Strides ss = {
            m_y.step, m_uv.step, 0u, 0u
        };
        return cv::MediaFrame::View(std::move(pp), std::move(ss));
    }
};
} // anonymous namespace

struct MediaFrame_Test: public ::testing::Test {
    using M = cv::Mat;
    using MF = cv::MediaFrame;
    MF frame;
};

struct MediaFrame_BGR: public MediaFrame_Test {
    M bgr;
    MediaFrame_BGR()
        : bgr(M::eye(240, 320, CV_8UC3)) {
        frame = MF::Create<TestMediaBGR>(bgr);
    }
};

TEST_F(MediaFrame_BGR, Meta) {
    auto meta = frame.desc();
    EXPECT_EQ(cv::MediaFormat::BGR, meta.fmt);
    EXPECT_EQ(cv::Size(320,240),    meta.size);
}

TEST_F(MediaFrame_BGR, Access) {
    cv::MediaFrame::View view1 = frame.access(cv::MediaFrame::Access::R);
    EXPECT_EQ(bgr.ptr(), view1.ptr[0]);
    EXPECT_EQ(bgr.step,  view1.stride[0]);

    cv::MediaFrame::View view2 = frame.access(cv::MediaFrame::Access::R);
    EXPECT_EQ(bgr.ptr(), view2.ptr[0]);
    EXPECT_EQ(bgr.step,  view2.stride[0]);
}

struct MediaFrame_NV12: public MediaFrame_Test {
    cv::Size sz;
    cv::Mat buf, y, uv;
    MediaFrame_NV12()
        : sz {320, 240}
        , buf(M::eye(sz.height*3/2, sz.width, CV_8UC1))
        , y  (buf.rowRange(0, sz.height))
        , uv (buf.rowRange(sz.height, sz.height*3/2)) {
        frame = MF::Create<TestMediaNV12>(y, uv);
    }
};

TEST_F(MediaFrame_NV12, Meta) {
    auto meta = frame.desc();
    EXPECT_EQ(cv::MediaFormat::NV12, meta.fmt);
    EXPECT_EQ(cv::Size(320,240),     meta.size);
}

TEST_F(MediaFrame_NV12, Access) {
    cv::MediaFrame::View view1 = frame.access(cv::MediaFrame::Access::R);
    EXPECT_EQ(y. ptr(), view1.ptr   [0]);
    EXPECT_EQ(y. step,  view1.stride[0]);
    EXPECT_EQ(uv.ptr(), view1.ptr   [1]);
    EXPECT_EQ(uv.step,  view1.stride[1]);

    cv::MediaFrame::View view2 = frame.access(cv::MediaFrame::Access::R);
    EXPECT_EQ(y. ptr(), view2.ptr   [0]);
    EXPECT_EQ(y. step,  view2.stride[0]);
    EXPECT_EQ(uv.ptr(), view2.ptr   [1]);
    EXPECT_EQ(uv.step,  view2.stride[1]);
}

TEST(MediaFrame, Callback) {
    int counter = 0;
    cv::Mat bgr = cv::Mat::eye(240, 320, CV_8UC3);
    cv::MediaFrame frame = cv::MediaFrame::Create<TestMediaBGR>(bgr, [&counter](){counter++;});

    // Test that the callback (in this case, incrementing the counter)
    // is called only on View destruction.
    EXPECT_EQ(0, counter);
    {
        cv::MediaFrame::View v1 = frame.access(cv::MediaFrame::Access::R);
        EXPECT_EQ(0, counter);
    }
    EXPECT_EQ(1, counter);
    {
        cv::MediaFrame::View v1 = frame.access(cv::MediaFrame::Access::R);
        EXPECT_EQ(1, counter);
        cv::MediaFrame::View v2 = frame.access(cv::MediaFrame::Access::W);
        EXPECT_EQ(1, counter);
    }
    EXPECT_EQ(3, counter);
}

} // namespace opencv_test
