// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"
#include <opencv2/gapi/rmat.hpp>
#include "rmat_test_common.hpp"

#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

namespace opencv_test
{

// This test set takes RMat type as a template parameter and launces simple
// blur(isl1) -> blur(isl2) computation passing RMat as input, as output
// and both input and output
template<typename RMatAdapterT>
struct RMatIntTestBase {
    cv::Mat in_mat;
    cv::Mat out_mat;
    cv::Mat out_mat_ref;
    cv::GComputation comp;
    bool inCallbackCalled;
    bool outCallbackCalled;

    static constexpr int w = 8;
    static constexpr int h = 8;

    RMatIntTestBase()
        : in_mat(h, w, CV_8UC1)
        , out_mat(h, w, CV_8UC1)
        , out_mat_ref(h, w, CV_8UC1)
        , comp([](){
              cv::GMat in;
              auto tmp = cv::gapi::blur(in, {3,3});
              auto out = cv::gapi::blur(tmp, {3,3});
              cv::gapi::island("test", cv::GIn(in), cv::GOut(tmp));
              return cv::GComputation(in, out);
          })
        , inCallbackCalled(false)
        , outCallbackCalled(false) {
        cv::randu(in_mat, cv::Scalar::all(127), cv::Scalar::all(40));
    }

    void check() {
        comp.apply(in_mat, out_mat_ref);
        EXPECT_EQ(0, cvtest::norm(out_mat_ref, out_mat, NORM_INF));
    }

    RMat createRMat(cv::Mat& mat, bool& callbackCalled) {
        return {cv::make_rmat<RMatAdapterT>(mat, callbackCalled)};
    }
};

template<typename RMatAdapterT>
struct RMatIntTest : public RMatIntTestBase<RMatAdapterT>
{
    template<typename In, typename Out>
    void run(const In& in, Out& out, cv::GCompileArgs&& compile_args) {
        for (int i = 0; i < 2; i++) {
            EXPECT_FALSE(this->inCallbackCalled);
            EXPECT_FALSE(this->outCallbackCalled);
            auto compile_args_copy = compile_args;
            this->comp.apply(cv::gin(in), cv::gout(out), std::move(compile_args_copy));
            EXPECT_FALSE(this->inCallbackCalled);
            if (std::is_same<RMat,Out>::value) {
                EXPECT_TRUE(this->outCallbackCalled);
            } else {
                EXPECT_FALSE(this->outCallbackCalled);
            }
            this->outCallbackCalled = false;
        }
        this->check();
    }
};

template<typename RMatAdapterT>
struct RMatIntTestStreaming : public RMatIntTestBase<RMatAdapterT>
{
    template <typename M>
    cv::GMatDesc getDesc(const M& m) { return cv::descr_of(m); }

    void checkOutput(const cv::Mat&) { this->check(); }

    void checkOutput(const RMat& rm) {
        auto view = rm.access(RMat::Access::R);
        this->out_mat = cv::Mat(view.size(), view.type(), view.ptr());
        this->check();
    }

    template<typename In, typename Out>
    void run(const In& in, Out& out, cv::GCompileArgs&& compile_args) {
        auto sc = this->comp.compileStreaming(getDesc(in), std::move(compile_args));

        sc.setSource(cv::gin(in));
        sc.start();

        std::size_t frame = 0u;
        constexpr std::size_t num_frames = 10u;
        EXPECT_FALSE(this->inCallbackCalled);
        EXPECT_FALSE(this->outCallbackCalled);
        while (sc.pull(cv::gout(out)) && frame < num_frames) {
            frame++;
            this->checkOutput(out);
            EXPECT_FALSE(this->inCallbackCalled);
            EXPECT_FALSE(this->outCallbackCalled);
        }
        EXPECT_EQ(num_frames, frame);
    }
};

struct OcvKernels {
    cv::gapi::GKernelPackage kernels() { return cv::gapi::imgproc::cpu::kernels(); }
};
struct FluidKernels {
    cv::gapi::GKernelPackage kernels() { return cv::gapi::imgproc::fluid::kernels(); }
};

struct RMatIntTestCpuRef : public
    RMatIntTest<RMatAdapterRef>, OcvKernels {};
struct RMatIntTestCpuCopy : public
    RMatIntTest<RMatAdapterCopy>, OcvKernels {};
struct RMatIntTestCpuRefStreaming : public
    RMatIntTestStreaming<RMatAdapterRef>, OcvKernels  {};
struct RMatIntTestCpuCopyStreaming : public
    RMatIntTestStreaming<RMatAdapterCopy>, OcvKernels {};
struct RMatIntTestCpuRefFluid : public
    RMatIntTest<RMatAdapterRef>, FluidKernels {};
struct RMatIntTestCpuCopyFluid : public
    RMatIntTest<RMatAdapterCopy>, FluidKernels {};
struct RMatIntTestCpuRefStreamingFluid : public
    RMatIntTestStreaming<RMatAdapterRef>, FluidKernels {};
struct RMatIntTestCpuCopyStreamingFluid : public
    RMatIntTestStreaming<RMatAdapterCopy>, FluidKernels {};

template<typename T>
struct RMatIntTypedTest : public ::testing::Test, public T {};

using RMatIntTestTypes = ::testing::Types< RMatIntTestCpuRef
                                         , RMatIntTestCpuCopy
                                         , RMatIntTestCpuRefStreaming
                                         , RMatIntTestCpuCopyStreaming
                                         , RMatIntTestCpuRefFluid
                                         , RMatIntTestCpuCopyFluid
                                         , RMatIntTestCpuRefStreamingFluid
                                         , RMatIntTestCpuCopyStreamingFluid
                                         >;

TYPED_TEST_CASE(RMatIntTypedTest, RMatIntTestTypes);

TYPED_TEST(RMatIntTypedTest, In) {
    auto in_rmat = this->createRMat(this->in_mat, this->inCallbackCalled);
    this->run(in_rmat, this->out_mat, cv::compile_args(this->kernels()));
}

TYPED_TEST(RMatIntTypedTest, Out) {
    auto out_rmat = this->createRMat(this->out_mat, this->outCallbackCalled);
    this->run(this->in_mat, out_rmat, cv::compile_args(this->kernels()));
}

TYPED_TEST(RMatIntTypedTest, InOut) {
    auto  in_rmat = this->createRMat(this->in_mat, this->inCallbackCalled);
    auto out_rmat = this->createRMat(this->out_mat, this->outCallbackCalled);
    this->run(in_rmat, out_rmat, cv::compile_args(this->kernels()));
}

} // namespace opencv_test
