// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <chrono>
#include <stdexcept>
#include <ade/util/iota_range.hpp>
#include "logger.hpp"

#include <opencv2/gapi/core.hpp>

#include "executor/thread_pool.hpp"

namespace opencv_test
{

namespace
{
    G_TYPED_KERNEL(GInvalidResize, <GMat(GMat,Size,double,double,int)>, "org.opencv.test.invalid_resize")
    {
         static GMatDesc outMeta(GMatDesc in, Size, double, double, int) { return in; }
    };

    GAPI_OCV_KERNEL(GOCVInvalidResize, GInvalidResize)
    {
        static void run(const cv::Mat& in, cv::Size sz, double fx, double fy, int interp, cv::Mat &out)
        {
            cv::resize(in, out, sz, fx, fy, interp);
        }
    };

    G_TYPED_KERNEL(GReallocatingCopy, <GMat(GMat)>, "org.opencv.test.reallocating_copy")
    {
         static GMatDesc outMeta(GMatDesc in) { return in; }
    };

    GAPI_OCV_KERNEL(GOCVReallocatingCopy, GReallocatingCopy)
    {
        static void run(const cv::Mat& in, cv::Mat &out)
        {
            out = in.clone();
        }
    };

    G_TYPED_KERNEL(GCustom, <GMat(GMat)>, "org.opencv.test.custom")
    {
         static GMatDesc outMeta(GMatDesc in) { return in; }
    };

    G_TYPED_KERNEL(GZeros, <GMat(GMat, GMatDesc)>, "org.opencv.test.zeros")
    {
        static GMatDesc outMeta(GMatDesc /*in*/, GMatDesc user_desc)
        {
            return user_desc;
        }
    };

    GAPI_OCV_KERNEL(GOCVZeros, GZeros)
    {
        static void run(const cv::Mat&      /*in*/,
                        const cv::GMatDesc& /*desc*/,
                        cv::Mat&            out)
        {
            out.setTo(0);
        }
    };

    G_TYPED_KERNEL(GBusyWait, <GMat(GMat, uint32_t)>, "org.busy_wait") {
        static GMatDesc outMeta(GMatDesc in, uint32_t)
        {
            return in;
        }
    };

    GAPI_OCV_KERNEL(GOCVBusyWait, GBusyWait)
    {
        static void run(const cv::Mat& in,
                        const uint32_t time_in_ms,
                        cv::Mat&       out)
        {
            using namespace std::chrono;
            auto s = high_resolution_clock::now();
            in.copyTo(out);
            auto e = high_resolution_clock::now();

            const auto elapsed_in_ms =
                static_cast<int32_t>(duration_cast<milliseconds>(e-s).count());

            int32_t diff = time_in_ms - elapsed_in_ms;
            const auto need_to_wait_in_ms = static_cast<uint32_t>(std::max(0, diff));

            s = high_resolution_clock::now();
            e = s;
            while (duration_cast<milliseconds>(e-s).count() < need_to_wait_in_ms) {
                e = high_resolution_clock::now();
            }
        }
    };

    // These definitions test the correct macro work if the kernel has multiple output values
    G_TYPED_KERNEL(GRetGArrayTupleOfGMat2Kernel,  <GArray<std::tuple<GMat, GMat>>(GMat, Scalar)>,                                         "org.opencv.test.retarrayoftupleofgmat2kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat3Kernel,  <GArray<std::tuple<GMat, GMat, GMat>>(GMat)>,                                           "org.opencv.test.retarrayoftupleofgmat3kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat4Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat>>(GMat)>,                                     "org.opencv.test.retarrayoftupleofgmat4kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat5Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat>>(GMat)>,                               "org.opencv.test.retarrayoftupleofgmat5kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat6Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,                         "org.opencv.test.retarrayoftupleofgmat6kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat7Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,                   "org.opencv.test.retarrayoftupleofgmat7kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat8Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,             "org.opencv.test.retarrayoftupleofgmat8kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat9Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,       "org.opencv.test.retarrayoftupleofgmat9kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat10Kernel, <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>, "org.opencv.test.retarrayoftupleofgmat10kernel") {};

    G_TYPED_KERNEL_M(GRetGMat2Kernel,     <std::tuple<GMat, GMat>(GMat, GMat, GMat)>,                                     "org.opencv.test.retgmat2kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat3Kernel,     <std::tuple<GMat, GMat, GMat>(GMat, GScalar)>,                                  "org.opencv.test.retgmat3kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat4Kernel,     <std::tuple<GMat, GMat, GMat, GMat>(GMat, GArray<int>, GScalar)>,               "org.opencv.test.retgmat4kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat5Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat>(GMat)>,                               "org.opencv.test.retgmat5kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat6Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,                         "org.opencv.test.retgmat6kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat7Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,                   "org.opencv.test.retgmat7kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat8Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,             "org.opencv.test.retgmat8kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat9Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,       "org.opencv.test.retgmat9kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat10Kernel,    <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>, "org.opencv.test.retgmat10kernel")     {};
}

TEST(GAPI_Pipeline, OverloadUnary_MatMat)
{
    cv::GMat in;
    cv::GComputation comp(in, cv::gapi::bitwise_not(in));

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat ref_mat = ~in_mat;

    cv::Mat out_mat;
    comp.apply(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = cv::Mat();
    auto cc = comp.compile(cv::descr_of(in_mat));
    cc(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_Pipeline, OverloadUnary_MatScalar)
{
    cv::GMat in;
    cv::GComputation comp(in, cv::gapi::sum(in));

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Scalar ref_scl = cv::sum(in_mat);

    cv::Scalar out_scl;
    comp.apply(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = cv::Scalar();
    auto cc = comp.compile(cv::descr_of(in_mat));
    cc(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(GAPI_Pipeline, OverloadBinary_Mat)
{
    cv::GMat a, b;
    cv::GComputation comp(a, b, cv::gapi::add(a, b));

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat ref_mat = (in_mat+in_mat);

    cv::Mat out_mat;
    comp.apply(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = cv::Mat();
    auto cc = comp.compile(cv::descr_of(in_mat), cv::descr_of(in_mat));
    cc(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_Pipeline, OverloadBinary_Scalar)
{
    cv::GMat a, b;
    cv::GComputation comp(a, b, cv::gapi::sum(a + b));

    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Scalar ref_scl = cv::sum(in_mat+in_mat);

    cv::Scalar out_scl;
    comp.apply(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = cv::Scalar();
    auto cc = comp.compile(cv::descr_of(in_mat), cv::descr_of(in_mat));
    cc(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(GAPI_Pipeline, Sharpen)
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

    cv::GComputation c(cv::GIn(in), cv::GOut(y_sharp, out));
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

TEST(GAPI_Pipeline, CustomRGB2YUV)
{
    const cv::Size sz(1280, 720);

    // BEWARE:
    //
    //    std::vector<cv::Mat> out_mats_cv(3, cv::Mat(sz, CV_8U))
    //
    // creates a vector of 3 elements pointing to the same Mat!
    // FIXME: Make a G-API check for that
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
        out_mats_cv  [i].create(sz, CV_8U);
        out_mats_gapi[i].create(sz, CV_8U);
    }

    // G-API code //////////////////////////////////////////////////////////////
    {
        cv::GMat r, g, b;
        cv::GMat y = 0.299f*r + 0.587f*g + 0.114f*b;
        cv::GMat u = 0.492f*(b - y);
        cv::GMat v = 0.877f*(r - y);

        cv::GComputation customCvt({r, g, b}, {y, u, v});
        customCvt.apply(in_mats, out_mats_gapi);
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
            return cv::abs(m1-m2) > t;
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

TEST(GAPI_Pipeline, PipelineWithInvalidKernel)
{
    cv::GMat in, out;
    cv::Mat in_mat(500, 500, CV_8UC1), out_mat;
    out = GInvalidResize::on(in, cv::Size(300, 300), 0.0, 0.0, cv::INTER_LINEAR);

    const auto pkg = cv::gapi::kernels<GOCVInvalidResize>();
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    EXPECT_THROW(comp.apply(in_mat, out_mat, cv::compile_args(pkg)), std::logic_error);
}

TEST(GAPI_Pipeline, InvalidOutputComputation)
{
    cv::GMat in1, out1, out2, out3;

    std::tie(out1, out2, out2) = cv::gapi::split3(in1);
    cv::GComputation c({in1}, {out1, out2, out3});
    cv::Mat in_mat;
    cv::Mat out_mat1, out_mat2, out_mat3, out_mat4;
    std::vector<cv::Mat> u_outs = {out_mat1, out_mat2, out_mat3, out_mat4};
    std::vector<cv::Mat> u_ins = {in_mat};

    EXPECT_THROW(c.apply(u_ins, u_outs), std::logic_error);
}

TEST(GAPI_Pipeline, PipelineAllocatingKernel)
{
    cv::GMat in, out;
    cv::Mat in_mat(500, 500, CV_8UC1), out_mat;
    out = GReallocatingCopy::on(in);

    const auto pkg = cv::gapi::kernels<GOCVReallocatingCopy>();
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    EXPECT_THROW(comp.apply(in_mat, out_mat, cv::compile_args(pkg)), std::logic_error);
}

TEST(GAPI_Pipeline, CreateKernelImplFromLambda)
{
    cv::Size size(300, 300);
    int type = CV_8UC3;
    cv::Mat in_mat(size, type);
    cv::randu(in_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    int value = 5;

    cv::GMat in;
    cv::GMat out = GCustom::on(in);
    cv::GComputation comp(in, out);

    // OpenCV //////////////////////////////////////////////////////////////////////////
    auto ref_mat = in_mat + value;

    // G-API //////////////////////////////////////////////////////////////////////////
    auto impl = cv::gapi::cpu::ocv_kernel<GCustom>([&value](const cv::Mat& src, cv::Mat& dst)
                {
                    dst = src + value;
                });

    cv::Mat out_mat;
    auto pkg = cv::gapi::kernels(impl);
    comp.apply(in_mat, out_mat, cv::compile_args(pkg));

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
}

TEST(GAPI_Pipeline, ReplaceDefaultByLambda)
{
    cv::Size size(300, 300);
    int type = CV_8UC3;
    cv::Mat in_mat1(size, type);
    cv::Mat in_mat2(size, type);
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::GMat in1, in2;
    cv::GMat out = cv::gapi::add(in1, in2);
    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(out));

    // OpenCV //////////////////////////////////////////////////////////////////////////
    cv::Mat ref_mat = in_mat1 + in_mat2;


    // G-API //////////////////////////////////////////////////////////////////////////
    bool is_called = false;
    auto impl = cv::gapi::cpu::ocv_kernel<cv::gapi::core::GAdd>([&is_called]
                (const cv::Mat& src1, const cv::Mat& src2, int, cv::Mat& dst)
                {
                    is_called = true;
                    dst = src1 + src2;
                });

    cv::Mat out_mat;
    auto pkg = cv::gapi::kernels(impl);
    comp.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
    EXPECT_TRUE(is_called);
}

struct AddImpl
{
    void operator()(const cv::Mat& in1, const cv::Mat& in2, int, cv::Mat& out)
    {
        out = in1 + in2;
        is_called = true;
    }

    bool is_called = false;
};

TEST(GAPI_Pipeline, ReplaceDefaultByFunctor)
{
    cv::Size size(300, 300);
    int type = CV_8UC3;
    cv::Mat in_mat1(size, type);
    cv::Mat in_mat2(size, type);
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::GMat in1, in2;
    cv::GMat out = cv::gapi::add(in1, in2);
    cv::GComputation comp(cv::GIn(in1, in2), cv::GOut(out));

    // OpenCV //////////////////////////////////////////////////////////////////////////
    cv::Mat ref_mat = in_mat1 + in_mat2;


    // G-API ///////////////////////////////////////////////////////////////////////////
    AddImpl f;
    EXPECT_FALSE(f.is_called);
    auto impl = cv::gapi::cpu::ocv_kernel<cv::gapi::core::GAdd>(f);

    cv::Mat out_mat;
    auto pkg = cv::gapi::kernels(impl);
    comp.apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
    EXPECT_TRUE(f.is_called);
}

TEST(GAPI_Pipeline, GraphOutputIs1DMat)
{
    int dim = 100;
    cv::Mat in_mat(1, 1, CV_8UC3);
    cv::Mat out_mat;

    cv::GMat in;
    auto cc = cv::GComputation(in, GZeros::on(in, cv::GMatDesc(CV_8U, {dim})))
        .compile(cv::descr_of(in_mat), cv::compile_args(cv::gapi::kernels<GOCVZeros>()));

    // NB: Computation is able to write 1D output cv::Mat to empty out_mat.
    ASSERT_NO_THROW(cc(cv::gin(in_mat), cv::gout(out_mat)));
    ASSERT_EQ(1, out_mat.size.dims());
    ASSERT_EQ(dim, out_mat.size[0]);

    // NB: Computation is able to write 1D output cv::Mat
    // to pre-allocated with the same meta out_mat.
    ASSERT_NO_THROW(cc(cv::gin(in_mat), cv::gout(out_mat)));
    ASSERT_EQ(1, out_mat.size.dims());
    ASSERT_EQ(dim, out_mat.size[0]);
}

TEST(GAPI_Pipeline, 1DMatBetweenIslands)
{
    int dim = 100;
    cv::Mat in_mat(1, 1, CV_8UC3);
    cv::Mat out_mat;

    cv::Mat ref_mat({dim}, CV_8U);
    ref_mat.dims = 1;
    ref_mat.setTo(0);

    cv::GMat in;
    auto out = cv::gapi::copy(GZeros::on(cv::gapi::copy(in), cv::GMatDesc(CV_8U, {dim})));
    auto cc = cv::GComputation(in, out)
        .compile(cv::descr_of(in_mat), cv::compile_args(cv::gapi::kernels<GOCVZeros>()));

    cc(cv::gin(in_mat), cv::gout(out_mat));

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
}

TEST(GAPI_Pipeline, 1DMatWithinSingleIsland)
{
    int dim = 100;
    cv::Size blur_sz(3, 3);
    cv::Mat in_mat(10, 10, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat out_mat;

    cv::Mat ref_mat({dim}, CV_8U);
    ref_mat.dims = 1;
    ref_mat.setTo(0);

    cv::GMat in;
    auto out = cv::gapi::blur(
            GZeros::on(cv::gapi::blur(in, blur_sz), cv::GMatDesc(CV_8U, {dim})), blur_sz);
    auto cc = cv::GComputation(in, out)
        .compile(cv::descr_of(in_mat), cv::compile_args(cv::gapi::kernels<GOCVZeros>()));

    cc(cv::gin(in_mat), cv::gout(out_mat));

    EXPECT_EQ(0, cv::norm(out_mat, ref_mat));
}

TEST(GAPI_Pipeline, BranchesExecutedInParallel)
{
    cv::GMat in;
    // NB: cv::gapi::copy used to prevent fusing OCV backend operations
    // into the single island where they will be executed in turn
    auto out0 = GBusyWait::on(cv::gapi::copy(in), 1000u /*1sec*/);
    auto out1 = GBusyWait::on(cv::gapi::copy(in), 1000u /*1sec*/);
    auto out2 = GBusyWait::on(cv::gapi::copy(in), 1000u /*1sec*/);
    auto out3 = GBusyWait::on(cv::gapi::copy(in), 1000u /*1sec*/);

    cv::GComputation comp(cv::GIn(in), cv::GOut(out0,out1,out2,out3));
    cv::Mat in_mat = cv::Mat::eye(32, 32, CV_8UC1);
    cv::Mat out_mat0, out_mat1, out_mat2, out_mat3;

    using namespace std::chrono;
    auto s = high_resolution_clock::now();
    comp.apply(cv::gin(in_mat), cv::gout(out_mat0, out_mat1, out_mat2, out_mat3),
               cv::compile_args(cv::use_threaded_executor(4u),
                                cv::gapi::kernels<GOCVBusyWait>()));
    auto e = high_resolution_clock::now();
    const auto elapsed_in_ms = duration_cast<milliseconds>(e-s).count();;

    EXPECT_GE(1200u, elapsed_in_ms);
}

} // namespace opencv_test
