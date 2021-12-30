// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"

#include <stdexcept>

#include <opencv2/gapi/gtransform.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include "compiler/gmodel.hpp"
#include "compiler/gmodel_priv.hpp"

#include "api/gcomputation_priv.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "compiler/passes/passes.hpp"

#include "compiler/passes/pattern_matching.hpp"

#include "../common/gapi_tests_common.hpp"

#include "logger.hpp"

namespace opencv_test
{
// --------------------------------------------------------------------------------------
// Accuracy integration tests (GComputation-level)

namespace {
// FIXME: replace listener with something better (e.g. check graph via GModel?)
// custom "listener" to check what kernels are called within the test
struct KernelListener { std::map<std::string, size_t> counts; };
KernelListener& getListener() {
    static KernelListener l;
    return l;
}

using CompCreator = std::function<cv::GComputation()>;
using CompileArgsCreator = std::function<cv::GCompileArgs()>;
using Verifier = std::function<void(KernelListener)>;
}  // anonymous namespace

// Custom kernels && transformations below:

G_TYPED_KERNEL(MyNV12toBGR, <GMat(GMat, GMat)>, "test.my_nv12_to_bgr") {
    static GMatDesc outMeta(GMatDesc in_y, GMatDesc in_uv) {
        return cv::gapi::imgproc::GNV12toBGR::outMeta(in_y, in_uv);
    }
};
GAPI_OCV_KERNEL(MyNV12toBGRImpl, MyNV12toBGR)
{
    static void run(const cv::Mat& in_y, const cv::Mat& in_uv, cv::Mat &out)
    {
        getListener().counts[MyNV12toBGR::id()]++;
        cv::cvtColorTwoPlane(in_y, in_uv, out, cv::COLOR_YUV2BGR_NV12);
    }
};
G_TYPED_KERNEL(MyPlanarResize, <GMatP(GMatP, Size, int)>, "test.my_planar_resize") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int interp) {
        return cv::gapi::imgproc::GResizeP::outMeta(in, sz, interp);
    }
};
GAPI_OCV_KERNEL(MyPlanarResizeImpl, MyPlanarResize) {
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat &out)
    {
        getListener().counts[MyPlanarResize::id()]++;
        int inH = in.rows / 3;
        int inW = in.cols;
        int outH = out.rows / 3;
        int outW = out.cols;
        for (int i = 0; i < 3; i++) {
            auto in_plane = in(cv::Rect(0, i*inH, inW, inH));
            auto out_plane = out(cv::Rect(0, i*outH, outW, outH));
            cv::resize(in_plane, out_plane, out_sz, 0, 0, interp);
        }
    }
};
G_TYPED_KERNEL(MyInterleavedResize, <GMat(GMat, Size, int)>, "test.my_interleaved_resize") {
    static GMatDesc outMeta(GMatDesc in, Size sz, int interp) {
        return cv::gapi::imgproc::GResize::outMeta(in, sz, 0.0, 0.0, interp);
    }
};
GAPI_OCV_KERNEL(MyInterleavedResizeImpl, MyInterleavedResize) {
    static void run(const cv::Mat& in, cv::Size out_sz, int interp, cv::Mat &out)
    {
        getListener().counts[MyInterleavedResize::id()]++;
        cv::resize(in, out, out_sz, 0.0, 0.0, interp);
    }
};
G_TYPED_KERNEL(MyToNCHW, <GMatP(GMat)>, "test.my_to_nchw") {
    static GMatDesc outMeta(GMatDesc in) {
        GAPI_Assert(in.depth == CV_8U);
        GAPI_Assert(in.chan == 3);
        GAPI_Assert(in.planar == false);
        return in.asPlanar();
    }
};
GAPI_OCV_KERNEL(MyToNCHWImpl, MyToNCHW) {
    static void run(const cv::Mat& in, cv::Mat& out)
    {
        getListener().counts[MyToNCHW::id()]++;
        auto sz = in.size();
        auto w = sz.width;
        auto h = sz.height;
        cv::Mat ins[3] = {};
        cv::split(in, ins);
        for (int i = 0; i < 3; i++) {
            auto in_plane = ins[i];
            auto out_plane = out(cv::Rect(0, i*h, w, h));
            in_plane.copyTo(out_plane);
        }
    }
};
using GMat4 = std::tuple<GMat, GMat, GMat, GMat>;
G_TYPED_KERNEL_M(MySplit4, <GMat4(GMat)>, "test.my_split4") {
    static std::tuple<GMatDesc, GMatDesc, GMatDesc, GMatDesc> outMeta(GMatDesc in) {
        const auto out_depth = in.depth;
        const auto out_desc = in.withType(out_depth, 1);
        return std::make_tuple(out_desc, out_desc, out_desc, out_desc);
    }
};
GAPI_OCV_KERNEL(MySplit4Impl, MySplit4) {
    static void run(const cv::Mat& in, cv::Mat& out1, cv::Mat& out2, cv::Mat& out3, cv::Mat& out4)
    {
        getListener().counts[MySplit4::id()]++;
        cv::Mat outs[] = { out1, out2, out3, out4 };
        cv::split(in, outs);
    }
};

GAPI_TRANSFORM(NV12Transform, <cv::GMat(cv::GMat, cv::GMat)>, "test.nv12_transform")
{
    static cv::GMat pattern(const cv::GMat& y, const cv::GMat& uv)
    {
        GMat out = cv::gapi::NV12toBGR(y, uv);
        return out;
    }

    static cv::GMat substitute(const cv::GMat& y, const cv::GMat& uv)
    {
        GMat out = MyNV12toBGR::on(y, uv);
        return out;
    }
};
GAPI_TRANSFORM(ResizeTransform, <cv::GMat(cv::GMat)>, "3 x Resize -> Interleaved Resize")
{
    static cv::GMat pattern(const cv::GMat& in)
    {
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        const auto resize = std::bind(&cv::gapi::resize, std::placeholders::_1,
            cv::Size(100, 100), 0, 0, cv::INTER_AREA);
        return cv::gapi::merge3(resize(b), resize(g), resize(r));
    }

    static cv::GMat substitute(const cv::GMat& in)
    {
        return MyInterleavedResize::on(in, cv::Size(100, 100), cv::INTER_AREA);
    }
};
GAPI_TRANSFORM(ResizeTransformToCustom, <cv::GMat(cv::GMat)>, "Resize -> Custom Resize")
{
    static cv::GMat pattern(const cv::GMat& in)
    {
        return cv::gapi::resize(in, cv::Size(100, 100), 0, 0, cv::INTER_AREA);
    }

    static cv::GMat substitute(const cv::GMat& in)
    {
        return MyInterleavedResize::on(in, cv::Size(100, 100), cv::INTER_AREA);
    }
};
GAPI_TRANSFORM(ChainTransform1, <GMatP(GMat)>, "Resize + toNCHW -> toNCHW + PlanarResize")
{
    static GMatP pattern(const cv::GMat& in)
    {
        return MyToNCHW::on(cv::gapi::resize(in, cv::Size(60, 60)));
    }

    static GMatP substitute(const cv::GMat& in)
    {
        return MyPlanarResize::on(MyToNCHW::on(in), cv::Size(60, 60), cv::INTER_LINEAR);
    }
};
GAPI_TRANSFORM(ChainTransform2, <GMatP(GMat, GMat)>, "NV12toBGR + toNCHW -> NV12toBGRp")
{
    static GMatP pattern(const GMat& y, const GMat& uv)
    {
        return MyToNCHW::on(MyNV12toBGR::on(y, uv));
    }

    static GMatP substitute(const GMat& y, const GMat& uv)
    {
        return cv::gapi::NV12toBGRp(y, uv);
    }
};
GAPI_TRANSFORM(Split4Transform, <GMat4(GMat)>, "Split4 -> Custom Split4")
{
    static GMat4 pattern(const GMat& in)
    {
        return cv::gapi::split4(in);
    }

    static GMat4 substitute(const GMat& in)
    {
        return MySplit4::on(in);
    }
};
GAPI_TRANSFORM(Split4Merge3Transform, <GMat(GMat)>, "Split4 + Merge3 -> Custom Split4 + Merge3")
{
    static GMat pattern(const GMat& in)
    {
        GMat tmp1, tmp2, tmp3, unused;
        std::tie(tmp1, tmp2, tmp3, unused) = cv::gapi::split4(in);
        return cv::gapi::merge3(tmp1, tmp2, tmp3);
    }

    static GMat substitute(const GMat& in)
    {
        GMat tmp1, tmp2, tmp3, unused;
        std::tie(tmp1, tmp2, tmp3, unused) = MySplit4::on(in);
        return cv::gapi::merge3(tmp1, tmp2, tmp3);
    }
};
GAPI_TRANSFORM(Merge4Split4Transform, <GMat4(GMat, GMat, GMat, GMat)>,
    "Merge4 + Split4 -> Merge4 + Custom Split4")
{
    static GMat4 pattern(const GMat& in1, const GMat& in2, const GMat& in3,
        const GMat& in4)
    {
        return cv::gapi::split4(cv::gapi::merge4(in1, in2, in3, in4));
    }

    static GMat4 substitute(const GMat& in1, const GMat& in2, const GMat& in3,
        const GMat& in4)
    {
        return MySplit4::on(cv::gapi::merge4(in1, in2, in3, in4));
    }
};

// --------------------------------------------------------------------------------------
// Integration tests

TEST(PatternMatchingIntegrationBasic, OneTransformationApplied)
{
    cv::Size in_sz(640, 480);
    cv::Mat input(in_sz, CV_8UC3);
    cv::randu(input, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::Mat orig_graph_output, transformed_graph_output;

    auto orig_args = cv::compile_args();
    auto transform_args = cv::compile_args(
        cv::gapi::kernels<MyInterleavedResizeImpl, ResizeTransform>());

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    const auto make_computation = [] () {
        GMat in;
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        const auto resize = std::bind(&cv::gapi::resize, std::placeholders::_1,
            cv::Size(100, 100), 0, 0, cv::INTER_AREA);
        GMat out = cv::gapi::merge3(resize(b), resize(g), resize(r));
        return cv::GComputation(cv::GIn(in), cv::GOut(out));
    };

    {
        // Run original graph
        auto mainC = make_computation();
        mainC.apply(cv::gin(input), cv::gout(orig_graph_output), std::move(orig_args));
    }

    // Generate transformed graph (passing transformations via compile args)
    auto mainC = make_computation();  // get new copy with new Priv
    mainC.apply(cv::gin(input), cv::gout(transformed_graph_output), std::move(transform_args));

    // Compare
    ASSERT_TRUE(AbsExact()(orig_graph_output, transformed_graph_output));

    // Custom verification via listener
    ASSERT_EQ(1u, listener.counts.size());
    // called in transformed graph:
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MyInterleavedResize::id()));
    ASSERT_EQ(1u, listener.counts.at(MyInterleavedResize::id()));
}

TEST(PatternMatchingIntegrationBasic, SameTransformationAppliedSeveralTimes)
{
    cv::Size in_sz(640, 480);
    cv::Mat input(in_sz, CV_8UC3);
    cv::randu(input, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::Mat orig_graph_output, transformed_graph_output;

    auto orig_args = cv::compile_args();
    auto transform_args = cv::compile_args(
        cv::gapi::kernels<MyInterleavedResizeImpl, ResizeTransformToCustom>());

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    const auto make_computation = [] () {
        GMat in;
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        const auto resize = std::bind(&cv::gapi::resize, std::placeholders::_1,
            cv::Size(100, 100), 0, 0, cv::INTER_AREA);
        GMat out = cv::gapi::merge3(resize(b), resize(g), resize(r));
        return cv::GComputation(cv::GIn(in), cv::GOut(out));
    };

    {
        // Run original graph
        auto mainC = make_computation();
        mainC.apply(cv::gin(input), cv::gout(orig_graph_output), std::move(orig_args));
    }

    // Generate transformed graph (passing transformations via compile args)
    auto mainC = make_computation();  // get new copy with new Priv
    mainC.apply(cv::gin(input), cv::gout(transformed_graph_output), std::move(transform_args));

    // Compare
    ASSERT_TRUE(AbsExact()(orig_graph_output, transformed_graph_output));

    // Custom verification via listener
    ASSERT_EQ(1u, listener.counts.size());
    // called in transformed graph:
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MyInterleavedResize::id()));
    ASSERT_EQ(3u, listener.counts.at(MyInterleavedResize::id()));
}

TEST(PatternMatchingIntegrationBasic, OneNV12toBGRTransformationApplied)
{
    cv::Size in_sz(640, 480);
    cv::Mat y(in_sz, CV_8UC1), uv(cv::Size(in_sz.width / 2, in_sz.height / 2), CV_8UC2);
    cv::randu(y, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::randu(uv, cv::Scalar::all(100), cv::Scalar::all(200));
    cv::Mat orig_graph_output, transformed_graph_output;

    auto orig_args = cv::compile_args();
    auto transform_args = cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl, NV12Transform>());

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    const auto make_computation = [] () {
        GMat in1, in2;
        GMat bgr = cv::gapi::NV12toBGR(in1, in2);
        GMat out = cv::gapi::resize(bgr, cv::Size(100, 100));
        return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
    };

    {
        // Run original graph
        auto mainC = make_computation();
        mainC.apply(cv::gin(y, uv), cv::gout(orig_graph_output), std::move(orig_args));
    }

    // Generate transformed graph (passing transformations via compile args)
    auto mainC = make_computation();  // get new copy with new Priv
    mainC.apply(cv::gin(y, uv), cv::gout(transformed_graph_output), std::move(transform_args));

    // Compare
    ASSERT_TRUE(AbsExact()(orig_graph_output, transformed_graph_output));

    // Custom verification via listener
    ASSERT_EQ(1u, listener.counts.size());
    // called in transformed graph:
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MyNV12toBGR::id()));
    ASSERT_EQ(1u, listener.counts.at(MyNV12toBGR::id()));
}

TEST(PatternMatchingIntegrationBasic, TwoTransformationsApplied)
{
    cv::Size in_sz(640, 480);
    cv::Mat y(in_sz, CV_8UC1), uv(cv::Size(in_sz.width / 2, in_sz.height / 2), CV_8UC2);
    cv::randu(y, cv::Scalar::all(0), cv::Scalar::all(100));
    cv::randu(uv, cv::Scalar::all(100), cv::Scalar::all(200));
    cv::Mat orig_graph_output, transformed_graph_output;

    auto orig_args = cv::compile_args();
    auto transform_args = cv::compile_args(
        cv::gapi::kernels<MyNV12toBGRImpl, MyInterleavedResizeImpl, ResizeTransform,
            NV12Transform>());  // compile args with transformations

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    const auto make_computation = [] () {
        GMat in1, in2;
        GMat bgr = cv::gapi::NV12toBGR(in1, in2);
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(bgr);
        const auto resize = std::bind(&cv::gapi::resize, std::placeholders::_1,
            cv::Size(100, 100), 0, 0, cv::INTER_AREA);
        GMat tmp1 = cv::gapi::resize(bgr, cv::Size(90, 90));
        GMat tmp2 = cv::gapi::bitwise_not(cv::gapi::merge3(resize(b), resize(g), resize(r)));
        GMat out = cv::gapi::resize(tmp1 + GScalar(10.0), cv::Size(100, 100)) + tmp2;
        return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
    };

    {
        // Run original graph
        auto mainC = make_computation();
        mainC.apply(cv::gin(y, uv), cv::gout(orig_graph_output), std::move(orig_args));
    }

    // Generate transformed graph (passing transformations via compile args)
    auto mainC = make_computation();  // get new copy with new Priv
    mainC.apply(cv::gin(y, uv), cv::gout(transformed_graph_output), std::move(transform_args));

    // Compare
    ASSERT_TRUE(AbsExact()(orig_graph_output, transformed_graph_output));

    // Custom verification via listener
    ASSERT_EQ(2u, listener.counts.size());
    // called in transformed graph:
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MyNV12toBGR::id()));
    ASSERT_EQ(1u, listener.counts.at(MyNV12toBGR::id()));
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MyInterleavedResize::id()));
    ASSERT_EQ(1u, listener.counts.at(MyInterleavedResize::id()));
}

struct PatternMatchingIntegrationE2E : testing::Test
{
    cv::GComputation makeComputation() {
        GMat in1, in2;
        GMat bgr = MyNV12toBGR::on(in1, in2);
        GMat resized = cv::gapi::resize(bgr, cv::Size(60, 60));
        GMatP out = MyToNCHW::on(resized);
        return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
    }

    void runTest(cv::GCompileArgs&& transform_args) {
        cv::Size in_sz(640, 480);
        cv::Mat y(in_sz, CV_8UC1), uv(cv::Size(in_sz.width / 2, in_sz.height / 2), CV_8UC2);
        cv::randu(y, cv::Scalar::all(0), cv::Scalar::all(100));
        cv::randu(uv, cv::Scalar::all(100), cv::Scalar::all(200));
        cv::Mat orig_graph_output, transformed_graph_output;

        auto& listener = getListener();
        listener.counts.clear();  // clear counters before testing
        {
            // Run original graph
            auto mainC = makeComputation();
            mainC.apply(cv::gin(y, uv), cv::gout(orig_graph_output),
                cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl, MyToNCHWImpl>()));
        }

        // Generate transformed graph (passing transformations via compile args)
        auto mainC = makeComputation();  // get new copy with new Priv
        mainC.apply(cv::gin(y, uv), cv::gout(transformed_graph_output), std::move(transform_args));

        // Compare
        ASSERT_TRUE(AbsExact()(orig_graph_output, transformed_graph_output));

        // Custom verification via listener
        ASSERT_EQ(3u, listener.counts.size());
        // called in original graph:
        ASSERT_NE(listener.counts.cend(), listener.counts.find(MyNV12toBGR::id()));
        ASSERT_NE(listener.counts.cend(), listener.counts.find(MyToNCHW::id()));
        ASSERT_EQ(1u, listener.counts.at(MyNV12toBGR::id()));
        ASSERT_EQ(1u, listener.counts.at(MyToNCHW::id()));
        // called in transformed graph:
        ASSERT_NE(listener.counts.cend(), listener.counts.find(MyPlanarResize::id()));
        ASSERT_EQ(1u, listener.counts.at(MyPlanarResize::id()));
    }
};

TEST_F(PatternMatchingIntegrationE2E, ChainTransformationsApplied)
{
    runTest(cv::compile_args(
        cv::gapi::kernels<MyPlanarResizeImpl, ChainTransform1, ChainTransform2>()));
}

TEST_F(PatternMatchingIntegrationE2E, ReversedChainTransformationsApplied)
{
    runTest(cv::compile_args(
        cv::gapi::kernels<ChainTransform2, MyPlanarResizeImpl, ChainTransform1>()));
}

struct PatternMatchingIntegrationUnusedNodes : testing::Test
{
    cv::GComputation makeComputation() {
        GMat in1, in2;
        GMat bgr = cv::gapi::NV12toBGR(in1, in2);
        GMat b1, g1, r1;
        std::tie(b1, g1, r1) = cv::gapi::split3(bgr);
        // FIXME: easier way to call split4??
        GMat merged4 = cv::gapi::merge4(b1, g1, r1, b1);
        GMat b2, g2, r2, unused;
        std::tie(b2, g2, r2, unused) = cv::gapi::split4(merged4);
        GMat out = cv::gapi::merge3(b2, g2, r2);
        return cv::GComputation(cv::GIn(in1, in2), cv::GOut(out));
    }

    void runTest(cv::GCompileArgs&& transform_args) {
        cv::Size in_sz(640, 480);
        cv::Mat y(in_sz, CV_8UC1), uv(cv::Size(in_sz.width / 2, in_sz.height / 2), CV_8UC2);
        cv::randu(y, cv::Scalar::all(0), cv::Scalar::all(100));
        cv::randu(uv, cv::Scalar::all(100), cv::Scalar::all(200));

        cv::Mat orig_graph_output, transformed_graph_output;

        auto& listener = getListener();
        listener.counts.clear();  // clear counters before testing
        {
            // Run original graph
            auto mainC = makeComputation();
            mainC.apply(cv::gin(y, uv), cv::gout(orig_graph_output),
                cv::compile_args(cv::gapi::kernels<MyNV12toBGRImpl, MyToNCHWImpl>()));
        }

        // Generate transformed graph (passing transformations via compile args)
        auto mainC = makeComputation();  // get new copy with new Priv
        mainC.apply(cv::gin(y, uv), cv::gout(transformed_graph_output), std::move(transform_args));

        // Compare
        ASSERT_TRUE(AbsExact()(orig_graph_output, transformed_graph_output));

        // Custom verification via listener
        ASSERT_EQ(1u, listener.counts.size());
        // called in transformed graph:
        ASSERT_NE(listener.counts.cend(), listener.counts.find(MySplit4::id()));
        ASSERT_EQ(1u, listener.counts.at(MySplit4::id()));
    }
};

TEST_F(PatternMatchingIntegrationUnusedNodes, SingleOpTransformApplied)
{
    runTest(cv::compile_args(cv::gapi::kernels<MySplit4Impl, Split4Transform>()));
}

// FIXME: enable once unused nodes are properly handled by Transformation API
TEST_F(PatternMatchingIntegrationUnusedNodes, DISABLED_TransformWithInternalUnusedNodeApplied)
{
    runTest(cv::compile_args(cv::gapi::kernels<MySplit4Impl, Split4Merge3Transform>()));
}

TEST_F(PatternMatchingIntegrationUnusedNodes, TransformWithOutputUnusedNodeApplied)
{
    runTest(cv::compile_args(cv::gapi::kernels<MySplit4Impl, Merge4Split4Transform>()));
}

// --------------------------------------------------------------------------------------
// Bad arg integration tests (GCompiler-level) - General

struct PatternMatchingIntegrationBadArgTests : testing::Test
{
    cv::GComputation makeComputation() {
        GMat in;
        GMat a, b, c, d;
        std::tie(a, b, c, d) = MySplit4::on(in);  // using custom Split4 to check if it's called
        GMat out = cv::gapi::merge3(a + b, cv::gapi::bitwise_not(c), d * cv::GScalar(2.0));
        return cv::GComputation(cv::GIn(in), cv::GOut(out));
    }

    void runTest(cv::GCompileArgs&& transform_args) {
        cv::Size in_sz(640, 480);
        cv::Mat input(in_sz, CV_8UC4);
        cv::randu(input, cv::Scalar::all(70), cv::Scalar::all(140));

        cv::Mat output;

        // Generate transformed graph (passing transformations via compile args)
        auto mainC = makeComputation();  // get new copy with new Priv
        ASSERT_NO_THROW(mainC.apply(cv::gin(input), cv::gout(output), std::move(transform_args)));
    }
};

TEST_F(PatternMatchingIntegrationBadArgTests, NoTransformations)
{
    auto transform_args = cv::compile_args(cv::gapi::kernels<MySplit4Impl>());

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    runTest(std::move(transform_args));

    // Custom verification via listener
    ASSERT_EQ(1u, listener.counts.size());
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MySplit4::id()));
    ASSERT_EQ(1u, listener.counts.at(MySplit4::id()));
}

TEST_F(PatternMatchingIntegrationBadArgTests, WrongTransformation)
{
    // Here Split4Transform::pattern is "looking for" cv::gapi::split4 but it's not used
    auto transform_args = cv::compile_args(cv::gapi::kernels<MySplit4Impl, Split4Transform>());

    auto& listener = getListener();
    listener.counts.clear();  // clear counters before testing

    runTest(std::move(transform_args));

    // Custom verification via listener
    ASSERT_EQ(1u, listener.counts.size());
    ASSERT_NE(listener.counts.cend(), listener.counts.find(MySplit4::id()));
    ASSERT_EQ(1u, listener.counts.at(MySplit4::id()));
}

// --------------------------------------------------------------------------------------
// Bad arg integration tests (GCompiler-level) - Endless Loops

GAPI_TRANSFORM(EndlessLoopTransform, <cv::GMat(cv::GMat)>, "pattern in substitute")
{
    static cv::GMat pattern(const cv::GMat& in)
    {
        return cv::gapi::resize(in, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
    }

    static cv::GMat substitute(const cv::GMat& in)
    {
        cv::GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(in);
        auto resize = std::bind(&cv::gapi::resize,
            std::placeholders::_1, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
        cv::GMat out = cv::gapi::merge3(resize(b), resize(g), resize(r));
        return out;
    }
};

TEST(PatternMatchingIntegrationEndlessLoops, PatternInSubstituteInOneTransform)
{
    cv::Size in_sz(640, 480);
    cv::Mat input(in_sz, CV_8UC3);
    cv::randu(input, cv::Scalar::all(0), cv::Scalar::all(100));

    auto c = [] () {
        GMat in;
        GMat tmp = cv::gapi::resize(in, cv::Size(100, 100), 0, 0, cv::INTER_LINEAR);
        GMat out = cv::gapi::bitwise_not(tmp);
        return cv::GComputation(cv::GIn(in), cv::GOut(out));
    }();

    EXPECT_THROW(
        cv::gimpl::GCompiler(c, cv::descr_of(cv::gin(input)),
            cv::compile_args(cv::gapi::kernels<EndlessLoopTransform>())),
        std::exception);
}


} // namespace opencv_test
