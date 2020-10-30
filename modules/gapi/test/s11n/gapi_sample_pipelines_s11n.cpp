// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../test_precomp.hpp"

#include <ade/util/iota_range.hpp>
#include <opencv2/gapi/s11n.hpp>
#include "api/render_priv.hpp"
#include "../common/gapi_render_tests.hpp"

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

namespace ThisTest
{
    using GOpBool = GOpaque<bool>;
    using GOpInt = GOpaque<int>;
    using GOpDouble = GOpaque<double>;
    using GOpPoint = GOpaque<cv::Point>;
    using GOpSize = GOpaque<cv::Size>;
    using GOpRect = GOpaque<cv::Rect>;

    using GOpOut = std::tuple<GOpPoint, GOpSize, GOpRect>;

    G_TYPED_KERNEL_M(OpGenerate, <GOpOut(GOpBool, GOpInt, GOpDouble)>, "test.s11n.gopaque")
    {
        static std::tuple<GOpaqueDesc, GOpaqueDesc, GOpaqueDesc> outMeta(const GOpaqueDesc&, const GOpaqueDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_gopaque_desc(), empty_gopaque_desc(), empty_gopaque_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpGenerate, OpGenerate)
    {
        static void run(const bool& b, const int& i, const double& d,
                        cv::Point& p, cv::Size& s, cv::Rect& r)
        {
            p = cv::Point(i, i*2);
            s = b ? cv::Size(42, 42) : cv::Size(7, 7);
            int ii = static_cast<int>(d);
            r = cv::Rect(ii, ii, ii, ii);
        }
    };

    using GArrInt = GArray<int>;
    using GArrDouble = GArray<double>;
    using GArrPoint = GArray<cv::Point>;
    using GArrSize = GArray<cv::Size>;
    using GArrRect = GArray<cv::Rect>;
    using GArrMat = GArray<cv::Mat>;
    using GArrScalar = GArray<cv::Scalar>;

    using GArrOut = std::tuple<GArrPoint, GArrSize, GArrRect, GArrMat>;

    G_TYPED_KERNEL_M(ArrGenerate, <GArrOut(GArrInt, GArrInt, GArrDouble, GArrScalar)>, "test.s11n.garray")
    {
        static std::tuple<GArrayDesc, GArrayDesc, GArrayDesc, GArrayDesc> outMeta(const GArrayDesc&, const GArrayDesc&,
                                                                                  const GArrayDesc&, const GArrayDesc&) {
            return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVArrGenerate, ArrGenerate)
    {
        static void run(const std::vector<int>& b, const std::vector<int>& i,
                        const std::vector<double>& d, const std::vector<cv::Scalar>& sc,
                        std::vector<cv::Point>& p, std::vector<cv::Size>& s,
                        std::vector<cv::Rect>& r, std::vector<cv::Mat>& m)
        {
            p.clear(); p.resize(b.size());
            s.clear(); s.resize(b.size());
            r.clear(); r.resize(b.size());
            m.clear(); m.resize(b.size());

            for (std::size_t idx = 0; idx < b.size(); ++idx)
            {
                p[idx] = cv::Point(i[idx], i[idx]*2);
                s[idx] = b[idx] == 1 ? cv::Size(42, 42) : cv::Size(7, 7);
                int ii = static_cast<int>(d[idx]);
                r[idx] = cv::Rect(ii, ii, ii, ii);
                m[idx] = cv::Mat(3, 3, CV_8UC1, sc[idx]);
            }
        }
    };

    G_TYPED_KERNEL_M(OpArrK1, <std::tuple<GArrInt,GOpSize>(GOpInt, GArrSize)>, "test.s11n.oparrk1")
    {
        static std::tuple<GArrayDesc, GOpaqueDesc> outMeta(const GOpaqueDesc&, const GArrayDesc&) {
            return std::make_tuple(empty_array_desc(), empty_gopaque_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK1, OpArrK1)
    {
        static void run(const int& i, const std::vector<cv::Size>& vs,
                        std::vector<int>& vi, cv::Size& s)
        {
            vi.clear(); vi.resize(vs.size());
            s = cv::Size(i, i);
            for (std::size_t idx = 0; idx < vs.size(); ++ idx)
                vi[idx] = vs[idx].area();
        }
    };

    G_TYPED_KERNEL_M(OpArrK2, <std::tuple<GOpDouble,GArrPoint>(GArrInt, GOpSize)>, "test.s11n.oparrk2")
    {
        static std::tuple<GOpaqueDesc, GArrayDesc> outMeta(const GArrayDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_gopaque_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK2, OpArrK2)
    {
        static void run(const std::vector<int>& vi, const cv::Size& s,
                        double& d, std::vector<cv::Point>& vp)
        {
            vp.clear(); vp.resize(vi.size());
            d = s.area() * 1.5;
            for (std::size_t idx = 0; idx < vi.size(); ++ idx)
                vp[idx] = cv::Point(vi[idx], vi[idx]);
        }
    };

    using GK3Out = std::tuple<cv::GArray<uint64_t>, cv::GArray<int32_t>>;
    G_TYPED_KERNEL_M(OpArrK3, <GK3Out(cv::GArray<bool>, cv::GArray<int32_t>, cv::GOpaque<float>)>, "test.s11n.oparrk3")
    {
        static std::tuple<GArrayDesc, GArrayDesc> outMeta(const GArrayDesc&, const GArrayDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK3, OpArrK3)
    {
        static void run(const std::vector<bool>& vb, const std::vector<int32_t>& vi_in, const float& f,
                        std::vector<uint64_t>& vui, std::vector<int32_t>& vi)
        {
            vui.clear(); vui.resize(vi_in.size());
            vi.clear();  vi.resize(vi_in.size());

            for (std::size_t idx = 0; idx < vi_in.size(); ++ idx)
            {
                vi[idx] = vb[idx] ? vi_in[idx] : -vi_in[idx];
                vui[idx] = vb[idx] ? static_cast<uint64_t>(vi_in[idx] * f) :
                                     static_cast<uint64_t>(vi_in[idx] / f);
            }
        }
    };

    using GK4Out = std::tuple<cv::GOpaque<int>, cv::GArray<std::string>>;
    G_TYPED_KERNEL_M(OpArrK4, <GK4Out(cv::GOpaque<bool>, cv::GOpaque<std::string>)>, "test.s11n.oparrk4")
    {
        static std::tuple<GOpaqueDesc, GArrayDesc> outMeta(const GOpaqueDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_gopaque_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK4, OpArrK4)
    {
        static void run(const bool& b, const std::string& s,
                        int& i, std::vector<std::string>& vs)
        {
            vs.clear();
            vs.resize(2);
            i = b ? 42 : 24;
            auto s_copy = s + " world";
            vs = std::vector<std::string>{s_copy, s_copy};
        }
    };
} // namespace ThisTest

TEST(S11N, Pipeline_GOpaque)
{
    using namespace ThisTest;
    GOpBool in1;
    GOpInt in2;
    GOpDouble in3;

    auto out = OpGenerate::on(in1, in2, in3);
    cv::GComputation c(cv::GIn(in1, in2, in3), cv::GOut(std::get<0>(out), std::get<1>(out), std::get<2>(out)));

    auto p = cv::gapi::serialize(c);
    auto dc = cv::gapi::deserialize<cv::GComputation>(p);

    bool b = true;
    int i = 33;
    double d = 128.7;
    cv::Point pp;
    cv::Size s;
    cv::Rect r;
    dc.apply(cv::gin(b, i, d), cv::gout(pp, s, r), cv::compile_args(cv::gapi::kernels<OCVOpGenerate>()));

    EXPECT_EQ(pp, cv::Point(i, i*2));
    EXPECT_EQ(s, cv::Size(42, 42));
    int ii = static_cast<int>(d);
    EXPECT_EQ(r, cv::Rect(ii, ii, ii, ii));
}

TEST(S11N, Pipeline_GArray)
{
    using namespace ThisTest;
    GArrInt in1, in2;
    GArrDouble in3;
    GArrScalar in4;

    auto out = ArrGenerate::on(in1, in2, in3, in4);
    cv::GComputation c(cv::GIn(in1, in2, in3, in4),
                       cv::GOut(std::get<0>(out), std::get<1>(out),
                                std::get<2>(out), std::get<3>(out)));

    auto p = cv::gapi::serialize(c);
    auto dc = cv::gapi::deserialize<cv::GComputation>(p);

    std::vector<int> b {1, 0, -1};
    std::vector<int> i {3, 0 , 59};
    std::vector<double> d {0.7, 120.5, 44.14};
    std::vector<cv::Scalar> sc {cv::Scalar::all(10), cv::Scalar::all(15), cv::Scalar::all(99)};
    std::vector<cv::Point> pp;
    std::vector<cv::Size> s;
    std::vector<cv::Rect> r;
    std::vector<cv::Mat> m;
    dc.apply(cv::gin(b, i, d, sc), cv::gout(pp, s, r, m), cv::compile_args(cv::gapi::kernels<OCVArrGenerate>()));

    for (std::size_t idx = 0; idx < b.size(); ++idx)
    {
        EXPECT_EQ(pp[idx], cv::Point(i[idx], i[idx]*2));
        EXPECT_EQ(s[idx], b[idx] == 1 ? cv::Size(42, 42) : cv::Size(7, 7));
        int ii = static_cast<int>(d[idx]);
        EXPECT_EQ(r[idx], cv::Rect(ii, ii, ii, ii));
    }
}

TEST(S11N, Pipeline_GArray_GOpaque_Multinode)
{
    using namespace ThisTest;
    GOpInt in1;
    GArrSize in2;

    auto tmp = OpArrK1::on(in1, in2);
    auto out = OpArrK2::on(std::get<0>(tmp), std::get<1>(tmp));

    cv::GComputation c(cv::GIn(in1, in2),
                       cv::GOut(std::get<0>(out), std::get<1>(out)));

    auto p = cv::gapi::serialize(c);
    auto dc = cv::gapi::deserialize<cv::GComputation>(p);

    int i = 42;
    std::vector<cv::Size> s{cv::Size(11, 22), cv::Size(13, 18)};
    double d;
    std::vector<cv::Point> pp;

    dc.apply(cv::gin(i, s), cv::gout(d, pp), cv::compile_args(cv::gapi::kernels<OCVOpArrK1, OCVOpArrK2>()));

    auto st = cv::Size(i ,i);
    EXPECT_EQ(d, st.area() * 1.5);

    for (std::size_t idx = 0; idx < s.size(); ++idx)
    {
        EXPECT_EQ(pp[idx], cv::Point(s[idx].area(), s[idx].area()));
    }
}

TEST(S11N, Pipeline_GArray_GOpaque_2)
{
    using namespace ThisTest;

    cv::GArray<bool> in1;
    cv::GArray<int32_t> in2;
    cv::GOpaque<float> in3;
    auto out = OpArrK3::on(in1, in2, in3);
    cv::GComputation c(cv::GIn(in1, in2, in3),
                       cv::GOut(std::get<0>(out), std::get<1>(out)));

    auto p = cv::gapi::serialize(c);
    auto dc = cv::gapi::deserialize<cv::GComputation>(p);

    std::vector<bool> b {true, false, false};
    std::vector<int32_t> i {234324, -234252, 999};
    float f = 0.85f;
    std::vector<int32_t> out_i;
    std::vector<uint64_t> out_ui;
    dc.apply(cv::gin(b, i, f), cv::gout(out_ui, out_i), cv::compile_args(cv::gapi::kernels<OCVOpArrK3>()));

    for (std::size_t idx = 0; idx < b.size(); ++idx)
    {
        EXPECT_EQ(out_i[idx], b[idx] ? i[idx] : -i[idx]);
        EXPECT_EQ(out_ui[idx], b[idx] ? static_cast<uint64_t>(i[idx] * f) :
                                        static_cast<uint64_t>(i[idx] / f));
    }
}

TEST(S11N, Pipeline_GArray_GOpaque_3)
{
    using namespace ThisTest;

    cv::GOpaque<bool> in1;
    cv::GOpaque<std::string> in2;
    auto out = OpArrK4::on(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2),
                       cv::GOut(std::get<0>(out), std::get<1>(out)));

    auto p = cv::gapi::serialize(c);
    auto dc = cv::gapi::deserialize<cv::GComputation>(p);

    bool b = false;
    std::string s("hello");
    int i = 0;
    std::vector<std::string> vs{};
    dc.apply(cv::gin(b, s), cv::gout(i, vs), cv::compile_args(cv::gapi::kernels<OCVOpArrK4>()));

    EXPECT_EQ(24, i);
    std::vector<std::string> vs_ref{"hello world", "hello world"};
    EXPECT_EQ(vs_ref, vs);
}

TEST(S11N, Pipeline_Render_NV12)
{
    cv::Size sz (100, 200);
    int rects_num = 10;
    int text_num  = 10;
    int image_num = 10;

    int thick = 2;
    int lt = LINE_8;
    cv::Scalar color(111, 222, 77);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;

    // Rects
    int shift = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect rect(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Rect(rect, color, thick, lt, shift));
    }

    // Mosaic
    int cellsz = 50;
    int decim = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect mos(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Mosaic(mos, cellsz, decim));
    }

    // Text
    std::string text = "Some text";
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;
    for (int i = 0; i < text_num; ++i) {
        cv::Point org(200 + i, 200 + i);
        prims.emplace_back(cv::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));
    }

    // Image
    double transparency = 1.0;
    cv::Rect rect_img(0 ,0 , 50, 50);
    cv::Mat img(rect_img.size(), CV_8UC3, color);
    cv::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    for (int i = 0; i < image_num; ++i) {
        cv::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};

        prims.emplace_back(cv::gapi::wip::draw::Image({org_img, img, alpha}));
    }

    // Circle
    cv::Point center(300, 400);
    int rad = 25;
    prims.emplace_back(cv::gapi::wip::draw::Circle({center, rad, color, thick, lt, shift}));

    // Line
    cv::Point point_next(300, 425);
    prims.emplace_back(cv::gapi::wip::draw::Line({center, point_next, color, thick, lt, shift}));

    // Poly
    std::vector<cv::Point> points = {{300, 400}, {290, 450}, {348, 410}, {300, 400}};
    prims.emplace_back(cv::gapi::wip::draw::Poly({points, color, thick, lt, shift}));

    cv::GMat y_in, uv_in, y_out, uv_out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = cv::gapi::wip::draw::renderNV12(y_in, uv_in, arr);
    cv::GComputation comp(cv::GIn(y_in, uv_in, arr), cv::GOut(y_out, uv_out));

    auto serialized = cv::gapi::serialize(comp);
    auto dc = cv::gapi::deserialize<cv::GComputation>(serialized);

    cv::Mat y(1920, 1080, CV_8UC1);
    cv::Mat uv(960, 540, CV_8UC2);
    cv::randu(y, cv::Scalar(0), cv::Scalar(255));
    cv::randu(uv, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat y_ref_mat = y.clone(), uv_ref_mat = uv.clone();
    dc.apply(cv::gin(y, uv, prims), cv::gout(y, uv));

    // OpenCV code //////////////////////////////////////////////////////////////
    cv::Mat yuv;
    cv::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

    for (int i = 0; i < rects_num; ++i) {
        cv::Rect rect(200 + i, 200 + i, 200, 200);
        cv::rectangle(yuv, rect, cvtBGRToYUVC(color), thick, lt, shift);
    }

    for (int i = 0; i < rects_num; ++i) {
        cv::Rect mos(200 + i, 200 + i, 200, 200);
         drawMosaicRef(yuv, mos, cellsz);
    }

    for (int i = 0; i < text_num; ++i) {
        cv::Point org(200 + i, 200 + i);
        cv::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color), thick, lt, blo);
    }

    for (int i = 0; i < image_num; ++i) {
        cv::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};
        cv::Mat yuv_img;
        cv::cvtColor(img, yuv_img, cv::COLOR_BGR2YUV);
        blendImageRef(yuv, org_img, yuv_img, alpha);
    }

    cv::circle(yuv, center, rad, cvtBGRToYUVC(color), thick, lt, shift);
    cv::line(yuv, center, point_next, cvtBGRToYUVC(color), thick, lt, shift);
    std::vector<std::vector<cv::Point>> pp{points};
    cv::fillPoly(yuv, pp, cvtBGRToYUVC(color), lt, shift);

    // YUV -> NV12
    cv::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);

    EXPECT_EQ(cv::norm( y,  y_ref_mat), 0);
    EXPECT_EQ(cv::norm(uv, uv_ref_mat), 0);
}

TEST(S11N, Pipeline_Render_RGB)
{
    cv::Size sz (100, 200);
    int rects_num = 10;
    int text_num  = 10;
    int image_num = 10;

    int thick = 2;
    int lt = LINE_8;
    cv::Scalar color(111, 222, 77);

    // G-API code //////////////////////////////////////////////////////////////
    cv::gapi::wip::draw::Prims prims;

    // Rects
    int shift = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect rect(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Rect(rect, color, thick, lt, shift));
    }

    // Mosaic
    int cellsz = 50;
    int decim = 0;
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect mos(200 + i, 200 + i, 200, 200);
        prims.emplace_back(cv::gapi::wip::draw::Mosaic(mos, cellsz, decim));
    }

    // Text
    std::string text = "Some text";
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;
    for (int i = 0; i < text_num; ++i) {
        cv::Point org(200 + i, 200 + i);
        prims.emplace_back(cv::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));
    }

    // Image
    double transparency = 1.0;
    cv::Rect rect_img(0 ,0 , 50, 50);
    cv::Mat img(rect_img.size(), CV_8UC3, color);
    cv::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    for (int i = 0; i < image_num; ++i) {
        cv::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};

        prims.emplace_back(cv::gapi::wip::draw::Image({org_img, img, alpha}));
    }

    // Circle
    cv::Point center(300, 400);
    int rad = 25;
    prims.emplace_back(cv::gapi::wip::draw::Circle({center, rad, color, thick, lt, shift}));

    // Line
    cv::Point point_next(300, 425);
    prims.emplace_back(cv::gapi::wip::draw::Line({center, point_next, color, thick, lt, shift}));

    // Poly
    std::vector<cv::Point> points = {{300, 400}, {290, 450}, {348, 410}, {300, 400}};
    prims.emplace_back(cv::gapi::wip::draw::Poly({points, color, thick, lt, shift}));

    cv::GMat in, out;
    cv::GArray<cv::gapi::wip::draw::Prim> arr;
    out = cv::gapi::wip::draw::render3ch(in, arr);
    cv::GComputation comp(cv::GIn(in, arr), cv::GOut(out));

    auto serialized = cv::gapi::serialize(comp);
    auto dc = cv::gapi::deserialize<cv::GComputation>(serialized);

    cv::Mat input(1920, 1080, CV_8UC3);
    cv::randu(input, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat ref_mat = input.clone();
    dc.apply(cv::gin(input, prims), cv::gout(input));

    // OpenCV code //////////////////////////////////////////////////////////////
    for (int i = 0; i < rects_num; ++i) {
        cv::Rect rect(200 + i, 200 + i, 200, 200);
        cv::rectangle(ref_mat, rect, color, thick, lt, shift);
    }

    for (int i = 0; i < rects_num; ++i) {
        cv::Rect mos(200 + i, 200 + i, 200, 200);
         drawMosaicRef(ref_mat, mos, cellsz);
    }

    for (int i = 0; i < text_num; ++i) {
        cv::Point org(200 + i, 200 + i);
        cv::putText(ref_mat, text, org, ff, fs, color, thick, lt, blo);
    }

    for (int i = 0; i < image_num; ++i) {
        cv::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};
        blendImageRef(ref_mat, org_img, img, alpha);
    }

    cv::circle(ref_mat, center, rad, color, thick, lt, shift);
    cv::line(ref_mat, center, point_next, color, thick, lt, shift);
    std::vector<std::vector<cv::Point>> pp{points};
    cv::fillPoly(ref_mat, pp, color, lt, shift);

    EXPECT_EQ(cv::norm(input,  ref_mat), 0);
}
} // namespace opencv_test
