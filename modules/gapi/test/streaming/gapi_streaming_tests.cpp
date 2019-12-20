// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

#include <opencv2/gapi/streaming/cap.hpp>

namespace opencv_test
{
namespace
{
void initTestDataPath()
{
#ifndef WINRT
    static bool initialized = false;
    if (!initialized)
    {
        // Since G-API has no own test data (yet), it is taken from the common space
        const char* testDataPath = getenv("OPENCV_TEST_DATA_PATH");
        GAPI_Assert(testDataPath != nullptr);

        cvtest::addDataSearchPath(testDataPath);
        initialized = true;
    }
#endif // WINRT
}

enum class KernelPackage: int
{
    OCV,
    OCV_FLUID,
    OCL,
    OCL_FLUID,
};
std::ostream& operator<< (std::ostream &os, const KernelPackage &e)
{
    switch (e)
    {
#define _C(X) case KernelPackage::X: os << #X; break
        _C(OCV);
        _C(OCV_FLUID);
        _C(OCL);
        _C(OCL_FLUID);
#undef _C
    default: GAPI_Assert(false);
    }
    return os;
}

struct GAPI_Streaming: public ::testing::TestWithParam<KernelPackage> {
    GAPI_Streaming() { initTestDataPath(); }

    cv::gapi::GKernelPackage getKernelPackage()
    {
        using namespace cv::gapi;
        switch (GetParam())
        {
        case KernelPackage::OCV:
            return cv::gapi::combine(core::cpu::kernels(),
                                     imgproc::cpu::kernels());
            break;

        case KernelPackage::OCV_FLUID:
            return cv::gapi::combine(core::cpu::kernels(),
                                     imgproc::cpu::kernels(),
                                     core::fluid::kernels());
            break;

        // FIXME: OpenCL backend seem to work fine with Streaming
        // however the results are not very bit exact with CPU
        // It may be a problem but may be just implementation innacuracy.
        // Need to customize the comparison function in tests where OpenCL
        // is involved.
        case KernelPackage::OCL:
            return cv::gapi::combine(core::ocl::kernels(),
                                     imgproc::ocl::kernels());
            break;

        case KernelPackage::OCL_FLUID:
            return cv::gapi::combine(core::ocl::kernels(),
                                     imgproc::ocl::kernels(),
                                     core::fluid::kernels());
            break;
        }
        throw std::logic_error("Unknown package");
    }
};

} // anonymous namespace

TEST_P(GAPI_Streaming, SmokeTest_ConstInput_GMat)
{
    // This graph models the following use-case:
    // Canny here is used as some "feature detector"
    //
    // Island/device layout may be different given the contents
    // of the passed kernel package.
    //
    // The expectation is that we get as much islands in the
    // graph as backends the GKernelPackage contains.
    //
    // [Capture] --> Crop --> Resize --> Canny --> [out]

    const auto crop_rc = cv::Rect(13, 75, 377, 269);
    const auto resample_sz = cv::Size(224, 224);
    const auto thr_lo = 64.;
    const auto thr_hi = 192.;

    cv::GMat in;
    auto roi = cv::gapi::crop(in, crop_rc);
    auto res = cv::gapi::resize(roi, resample_sz);
    auto out = cv::gapi::Canny(res, thr_lo, thr_hi);
    cv::GComputation c(in, out);

    // Input data
    cv::Mat in_mat = cv::imread(findDataFile("cv/edgefilter/kodim23.png"));
    cv::Mat out_mat_gapi;

    // OpenCV reference image
    cv::Mat out_mat_ocv;
    {
        cv::Mat tmp;
        cv::resize(in_mat(crop_rc), tmp, resample_sz);
        cv::Canny(tmp, out_mat_ocv, thr_lo, thr_hi);
    }

    // Compilation & testing
    auto ccomp = c.compileStreaming(cv::descr_of(in_mat),
                                    cv::compile_args(cv::gapi::use_only{getKernelPackage()}));
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(cv::gin(in_mat));

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Fetch the result 15 times
    for (int i = 0; i < 15; i++) {
        // With constant inputs, the stream is endless so
        // the blocking pull() should never return `false`.
        EXPECT_TRUE(ccomp.pull(cv::gout(out_mat_gapi)));
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    }

    EXPECT_TRUE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());
}

TEST_P(GAPI_Streaming, SmokeTest_VideoInput_GMat)
{
    const auto crop_rc = cv::Rect(13, 75, 377, 269);
    const auto resample_sz = cv::Size(224, 224);
    const auto thr_lo = 64.;
    const auto thr_hi = 192.;

    cv::GMat in;
    auto roi = cv::gapi::crop(in, crop_rc);
    auto res = cv::gapi::resize(roi, resample_sz);
    auto out = cv::gapi::Canny(res, thr_lo, thr_hi);
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in), out));

    // OpenCV reference image code
    auto opencv_ref = [&](const cv::Mat &in_mat, cv::Mat &out_mat) {
        cv::Mat tmp;
        cv::resize(in_mat(crop_rc), tmp, resample_sz);
        cv::Canny(tmp, out_mat, thr_lo, thr_hi);
    };

    // Compilation & testing
    auto ccomp = c.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                                    cv::compile_args(cv::gapi::use_only{getKernelPackage()}));
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi")));

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Process the full video
    cv::Mat in_mat_gapi, out_mat_gapi;

    std::size_t frames = 0u;
    while (ccomp.pull(cv::gout(in_mat_gapi, out_mat_gapi))) {
        frames++;
        cv::Mat out_mat_ocv;
        opencv_ref(in_mat_gapi, out_mat_ocv);
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    }
    EXPECT_LT(0u, frames);
    EXPECT_FALSE(ccomp.running());

    // Stop can be called at any time (even if the pipeline is not running)
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());
}

TEST_P(GAPI_Streaming, Regression_CompileTimeScalar)
{
    // There was a bug with compile-time GScalars.  Compile-time
    // GScalars generate their own DATA nodes at GModel/GIslandModel
    // level, resulting in an extra link at the GIslandModel level, so
    // GStreamingExecutor automatically assigned an input queue to
    // such edges. Since there were no in-graph producer for that
    // data, no data were pushed to such queue what lead to a
    // deadlock.

    cv::GMat in;
    cv::GMat tmp = cv::gapi::copy(in);
    for (int i = 0; i < 3; i++) {
        tmp = tmp & cv::gapi::blur(in, cv::Size(3,3));
    }
    cv::GComputation c(cv::GIn(in), cv::GOut(tmp, tmp + 1));

    auto ccomp = c.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,512}},
                                    cv::compile_args(cv::gapi::use_only{getKernelPackage()}));

    cv::Mat in_mat = cv::imread(findDataFile("cv/edgefilter/kodim23.png"));
    cv::Mat out_mat1, out_mat2;

    // Fetch the result 15 times
    ccomp.setSource(cv::gin(in_mat));
    ccomp.start();
    for (int i = 0; i < 15; i++) {
        EXPECT_TRUE(ccomp.pull(cv::gout(out_mat1, out_mat2)));
    }

    ccomp.stop();
}

TEST_P(GAPI_Streaming, SmokeTest_StartRestart)
{
    cv::GMat in;
    auto res = cv::gapi::resize(in, cv::Size{300,200});
    auto out = cv::gapi::Canny(res, 95, 220);
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in), out));

    auto ccomp = c.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                                    cv::compile_args(cv::gapi::use_only{getKernelPackage()}));
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // Run 1
    std::size_t num_frames1 = 0u;
    ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi")));
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    cv::Mat out1, out2;
    while (ccomp.pull(cv::gout(out1, out2))) num_frames1++;

    EXPECT_FALSE(ccomp.running());

    // Run 2
    std::size_t num_frames2 = 0u;
    ccomp.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi")));
    ccomp.start();
    EXPECT_TRUE(ccomp.running());
    while (ccomp.pull(cv::gout(out1, out2))) num_frames2++;

    EXPECT_FALSE(ccomp.running());

    EXPECT_LT(0u, num_frames1);
    EXPECT_LT(0u, num_frames2);
    EXPECT_EQ(num_frames1, num_frames2);
}

TEST_P(GAPI_Streaming, SmokeTest_VideoConstSource_NoHang)
{
    // A video source is a finite one, while const source is not.
    // Check that pipeline completes when a video source completes.
    auto refc = cv::GComputation([](){
        cv::GMat in;
        return cv::GComputation(in, cv::gapi::copy(in));
    }).compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                        cv::compile_args(cv::gapi::use_only{getKernelPackage()}));

    refc.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi")));
    refc.start();
    std::size_t ref_frames = 0u;
    cv::Mat tmp;
    while (refc.pull(cv::gout(tmp))) ref_frames++;
    EXPECT_EQ(100u, ref_frames);

    cv::GMat in;
    cv::GMat in2;
    cv::GMat roi = cv::gapi::crop(in2, cv::Rect{1,1,256,256});
    cv::GMat blr = cv::gapi::blur(roi, cv::Size(3,3));
    cv::GMat out = blr - in;
    auto testc = cv::GComputation(cv::GIn(in, in2), cv::GOut(out))
        .compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{256,256}},
                          cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                          cv::compile_args(cv::gapi::use_only{getKernelPackage()}));

    cv::Mat in_const = cv::Mat::eye(cv::Size(256,256), CV_8UC3);
    testc.setSource(cv::gin(in_const,
                            gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"))));
    testc.start();
    std::size_t test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;

    EXPECT_EQ(ref_frames, test_frames);
}

TEST_P(GAPI_Streaming, SmokeTest_AutoMeta)
{
    cv::GMat in;
    cv::GMat in2;
    cv::GMat roi = cv::gapi::crop(in2, cv::Rect{1,1,256,256});
    cv::GMat blr = cv::gapi::blur(roi, cv::Size(3,3));
    cv::GMat out = blr - in;

    auto testc = cv::GComputation(cv::GIn(in, in2), cv::GOut(out))
        .compileStreaming(cv::compile_args(cv::gapi::use_only{getKernelPackage()}));

    cv::Mat in_const = cv::Mat::eye(cv::Size(256,256), CV_8UC3);
    cv::Mat tmp;

    // Test with one video source
    auto in_src = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"));
    testc.setSource(cv::gin(in_const, in_src));
    testc.start();

    std::size_t test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;
    EXPECT_EQ(100u, test_frames);

    // Now test with another one
    in_src = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/1920x1080.avi"));
    testc.setSource(cv::gin(in_const, in_src));
    testc.start();

    test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;
    EXPECT_EQ(165u, test_frames);
}


TEST_P(GAPI_Streaming, SmokeTest_AutoMeta_2xConstMat)
{
    cv::GMat in;
    cv::GMat in2;
    cv::GMat roi = cv::gapi::crop(in2, cv::Rect{1,1,256,256});
    cv::GMat blr = cv::gapi::blur(roi, cv::Size(3,3));
    cv::GMat out = blr - in;

    auto testc = cv::GComputation(cv::GIn(in, in2), cv::GOut(out))
        .compileStreaming(cv::compile_args(cv::gapi::use_only{getKernelPackage()}));

    cv::Mat in_const = cv::Mat::eye(cv::Size(256,256), CV_8UC3);
    cv::Mat tmp;

    // Test with first image
    auto in_src = cv::imread(findDataFile("cv/edgefilter/statue.png"));
    testc.setSource(cv::gin(in_const, in_src));
    testc.start();

    ASSERT_TRUE(testc.pull(cv::gout(tmp)));

    testc.stop();

    // Now test with second image
    in_src = cv::imread(findDataFile("cv/edgefilter/kodim23.png"));
    testc.setSource(cv::gin(in_const, in_src));
    testc.start();

    ASSERT_TRUE(testc.pull(cv::gout(tmp)));

    testc.stop();
}

TEST_P(GAPI_Streaming, SmokeTest_AutoMeta_VideoScalar)
{
    cv::GMat in_m;
    cv::GScalar in_s;
    cv::GMat out_m = in_m * in_s;

    auto testc = cv::GComputation(cv::GIn(in_m, in_s), cv::GOut(out_m))
        .compileStreaming(cv::compile_args(cv::gapi::use_only{getKernelPackage()}));

    cv::Mat tmp;
    // Test with one video source and scalar
    auto in_src = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"));
    testc.setSource(cv::gin(in_src, cv::Scalar{1.25}));
    testc.start();

    std::size_t test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;
    EXPECT_EQ(100u, test_frames);

    // Now test with another one video source and scalar
    in_src = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/1920x1080.avi"));
    testc.setSource(cv::gin(in_src, cv::Scalar{0.75}));
    testc.start();

    test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;
    EXPECT_EQ(165u, test_frames);
}

INSTANTIATE_TEST_CASE_P(TestStreaming, GAPI_Streaming,
                        Values(  KernelPackage::OCV
                             //, KernelPackage::OCL // FIXME: Fails bit-exactness check, maybe relax it?
                               , KernelPackage::OCV_FLUID
                             //, KernelPackage::OCL // FIXME: Fails bit-exactness check, maybe relax it?
                               ));

namespace TypesTest
{
    G_API_OP(SumV, <cv::GArray<int>(cv::GMat)>, "test.gapi.sumv") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };
    G_API_OP(AddV, <cv::GMat(cv::GMat,cv::GArray<int>)>, "test.gapi.addv") {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GArrayDesc &) {
            return in;
        }
    };

    GAPI_OCV_KERNEL(OCVSumV, SumV) {
        static void run(const cv::Mat &in, std::vector<int> &out) {
            CV_Assert(in.depth() == CV_8U);
            const auto length = in.cols * in.channels();
            out.resize(length);

            const uchar *ptr = in.ptr(0);
            for (int c = 0; c < length; c++) {
                out[c] = ptr[c];
            }
            for (int r = 1; r < in.rows; r++) {
                ptr = in.ptr(r);
                for (int c = 0; c < length; c++) {
                    out[c] += ptr[c];
                }
            }
        }
    };

    GAPI_OCV_KERNEL(OCVAddV, AddV) {
        static void run(const cv::Mat &in, const std::vector<int> &inv, cv::Mat &out) {
            CV_Assert(in.depth() == CV_8U);
            const auto length = in.cols * in.channels();
            CV_Assert(length == static_cast<int>(inv.size()));

            for (int r = 0; r < in.rows; r++) {
                const uchar *in_ptr = in.ptr(r);
                uchar *out_ptr = out.ptr(r);

                for (int c = 0; c < length; c++) {
                    out_ptr[c] = cv::saturate_cast<uchar>(in_ptr[c] + inv[c]);
                }
            }
        }
    };

    GAPI_FLUID_KERNEL(FluidAddV, AddV, false) {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View &in,
                        const std::vector<int> &inv,
                        cv::gapi::fluid::Buffer &out) {
            const uchar *in_ptr = in.InLineB(0);
            uchar *out_ptr = out.OutLineB(0);

            const auto length = in.meta().size.width * in.meta().chan;
            CV_Assert(length == static_cast<int>(inv.size()));

            for (int c = 0; c < length; c++) {
                out_ptr[c] = cv::saturate_cast<uchar>(in_ptr[c] + inv[c]);
            }
        }
    };
} // namespace TypesTest

TEST_P(GAPI_Streaming, SmokeTest_AutoMeta_VideoArray)
{
    cv::GMat in_m;
    cv::GArray<int> in_v;
    cv::GMat out_m = TypesTest::AddV::on(in_m, in_v) - in_m;

    // Run pipeline
    auto testc = cv::GComputation(cv::GIn(in_m, in_v), cv::GOut(out_m))
                    .compileStreaming(cv::compile_args(cv::gapi::kernels<TypesTest::OCVAddV>()));

    cv::Mat tmp;
    // Test with one video source and vector
    auto in_src = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"));
    std::vector<int> first_in_vec(768*3, 1);
    testc.setSource(cv::gin(in_src, first_in_vec));
    testc.start();

    std::size_t test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;
    EXPECT_EQ(100u, test_frames);

    // Now test with another one
    in_src = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/1920x1080.avi"));
    std::vector<int> second_in_vec(1920*3, 1);
    testc.setSource(cv::gin(in_src, second_in_vec));
    testc.start();

    test_frames = 0u;
    while (testc.pull(cv::gout(tmp))) test_frames++;
    EXPECT_EQ(165u, test_frames);
}

TEST(GAPI_Streaming_Types, InputScalar)
{
    // This test verifies if Streaming works with Scalar data @ input.

    cv::GMat in_m;
    cv::GScalar in_s;
    cv::GMat out_m = in_m * in_s;
    cv::GComputation c(cv::GIn(in_m, in_s), cv::GOut(out_m));

    // Input data
    cv::Mat in_mat = cv::Mat::eye(256, 256, CV_8UC1);
    cv::Scalar in_scl = 32;

    // Run pipeline
    auto sc = c.compileStreaming(cv::descr_of(in_mat), cv::descr_of(in_scl));
    sc.setSource(cv::gin(in_mat, in_scl));
    sc.start();

    for (int i = 0; i < 10; i++)
    {
        cv::Mat out;
        EXPECT_TRUE(sc.pull(cv::gout(out)));
        EXPECT_EQ(0., cv::norm(out, in_mat.mul(in_scl), cv::NORM_INF));
    }
}

TEST(GAPI_Streaming_Types, InputVector)
{
    // This test verifies if Streaming works with Vector data @ input.

    cv::GMat in_m;
    cv::GArray<int> in_v;
    cv::GMat out_m = TypesTest::AddV::on(in_m, in_v) - in_m;
    cv::GComputation c(cv::GIn(in_m, in_v), cv::GOut(out_m));

    // Input data
    cv::Mat in_mat = cv::Mat::eye(256, 256, CV_8UC1);
    std::vector<int> in_vec;
    TypesTest::OCVSumV::run(in_mat, in_vec);
    EXPECT_EQ(std::vector<int>(256,1), in_vec); // self-sanity-check

    auto opencv_ref = [&](const cv::Mat &in, const std::vector<int> &inv, cv::Mat &out) {
        cv::Mat tmp = in_mat.clone(); // allocate the same amount of memory as graph does
        TypesTest::OCVAddV::run(in, inv, tmp);
        out = tmp - in;
    };

    // Run pipeline
    auto sc = c.compileStreaming(cv::descr_of(in_mat),
                                 cv::descr_of(in_vec),
                                 cv::compile_args(cv::gapi::kernels<TypesTest::OCVAddV>()));
    sc.setSource(cv::gin(in_mat, in_vec));
    sc.start();

    for (int i = 0; i < 10; i++)
    {
        cv::Mat out_mat;
        EXPECT_TRUE(sc.pull(cv::gout(out_mat)));

        cv::Mat ref_mat;
        opencv_ref(in_mat, in_vec, ref_mat);
        EXPECT_EQ(0., cv::norm(ref_mat, out_mat, cv::NORM_INF));
    }
}

TEST(GAPI_Streaming_Types, XChangeScalar)
{
    // This test verifies if Streaming works when pipeline steps
    // (islands) exchange Scalar data.

    initTestDataPath();

    cv::GMat in;
    cv::GScalar m = cv::gapi::mean(in);
    cv::GMat tmp = cv::gapi::convertTo(in, CV_32F) - m;
    cv::GMat out = cv::gapi::blur(tmp, cv::Size(3,3));
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in),
                                             cv::gapi::convertTo(out, CV_8U)));

    auto ocv_ref = [](const cv::Mat &in_mat, cv::Mat &out_mat) {
        cv::Scalar ocv_m = cv::mean(in_mat);
        cv::Mat ocv_tmp;
        in_mat.convertTo(ocv_tmp, CV_32F);
        ocv_tmp -= ocv_m;
        cv::blur(ocv_tmp, ocv_tmp, cv::Size(3,3));
        ocv_tmp.convertTo(out_mat, CV_8U);
    };

    // Here we want mean & convertTo run on OCV
    // and subC & blur3x3 on Fluid.
    // FIXME: With the current API it looks quite awful:
    auto ocv_kernels = cv::gapi::core::cpu::kernels(); // convertTo
    ocv_kernels.remove<cv::gapi::core::GSubC>();

    auto fluid_kernels = cv::gapi::combine(cv::gapi::core::fluid::kernels(),     // subC
                                           cv::gapi::imgproc::fluid::kernels()); // box3x3
    fluid_kernels.remove<cv::gapi::core::GConvertTo>();
    fluid_kernels.remove<cv::gapi::core::GMean>();

    // FIXME: Now
    // - fluid kernels take over ocv kernels (including Copy, SubC, & Box3x3)
    // - selected kernels (which were removed from the fluid package) remain in OCV
    //   (ConvertTo + some others)
    // FIXME: This is completely awful. User should easily pick up specific kernels
    // to an empty kernel package to craft his own but not do it via exclusion.
    // Need to expose kernel declarations to public headers to enable kernels<..>()
    // on user side.
    auto kernels = cv::gapi::combine(ocv_kernels, fluid_kernels);

    // Compile streaming pipeline
    auto sc = c.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                                 cv::compile_args(cv::gapi::use_only{kernels}));
    sc.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi")));
    sc.start();

    cv::Mat in_frame;
    cv::Mat out_mat_gapi;
    cv::Mat out_mat_ref;

    std::size_t num_frames = 0u;
    while (sc.pull(cv::gout(in_frame, out_mat_gapi))) {
        num_frames++;
        ocv_ref(in_frame, out_mat_ref);
        EXPECT_EQ(0., cv::norm(out_mat_gapi, out_mat_ref, cv::NORM_INF));
    }
    EXPECT_LT(0u, num_frames);
}

TEST(GAPI_Streaming_Types, XChangeVector)
{
    // This test verifies if Streaming works when pipeline steps
    // (islands) exchange Vector data.

    initTestDataPath();

    cv::GMat in1, in2;
    cv::GMat in = cv::gapi::crop(in1, cv::Rect{0,0,576,576});
    cv::GScalar m = cv::gapi::mean(in);
    cv::GArray<int> s = TypesTest::SumV::on(in2); // (in2 = eye, so s = [1,0,0,1,..])
    cv::GMat out = TypesTest::AddV::on(in - m, s);

    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(cv::gapi::copy(in), out));

    auto ocv_ref = [](const cv::Mat &in_mat1, const cv::Mat &in_mat2, cv::Mat &out_mat) {
        cv::Mat in_roi = in_mat1(cv::Rect{0,0,576,576});
        cv::Scalar ocv_m = cv::mean(in_roi);
        std::vector<int> ocv_v;
        TypesTest::OCVSumV::run(in_mat2, ocv_v);

        out_mat.create(cv::Size(576,576), CV_8UC3);
        cv::Mat in_tmp = in_roi - ocv_m;
        TypesTest::OCVAddV::run(in_tmp, ocv_v, out_mat);
    };

    // Let crop/mean/sumV be calculated via OCV,
    // and AddV/subC be calculated via Fluid
    auto ocv_kernels = cv::gapi::core::cpu::kernels();
    ocv_kernels.remove<cv::gapi::core::GSubC>();
    ocv_kernels.include<TypesTest::OCVSumV>();

    auto fluid_kernels = cv::gapi::core::fluid::kernels();
    fluid_kernels.include<TypesTest::FluidAddV>();

    // Here OCV takes precedense over Fluid, with SubC & SumV remaining
    // in Fluid.
    auto kernels = cv::gapi::combine(fluid_kernels, ocv_kernels);

    // Compile streaming pipeline
    cv::Mat in_eye = cv::Mat::eye(cv::Size(576, 576), CV_8UC3);
    auto sc = c.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                                 cv::GMatDesc{CV_8U,3,cv::Size{576,576}},
                                 cv::compile_args(cv::gapi::use_only{kernels}));
    sc.setSource(cv::gin(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi")),
                         in_eye));
    sc.start();

    cv::Mat in_frame;
    cv::Mat out_mat_gapi;
    cv::Mat out_mat_ref;

    std::size_t num_frames = 0u;
    while (sc.pull(cv::gout(in_frame, out_mat_gapi))) {
        num_frames++;
        ocv_ref(in_frame, in_eye, out_mat_ref);
        EXPECT_EQ(0., cv::norm(out_mat_gapi, out_mat_ref, cv::NORM_INF));
    }
    EXPECT_LT(0u, num_frames);
}

TEST(GAPI_Streaming_Types, OutputScalar)
{
    // This test verifies if Streaming works when pipeline
    // produces scalar data only

    initTestDataPath();

    cv::GMat in;
    cv::GScalar out = cv::gapi::mean(in);
    auto sc = cv::GComputation(cv::GIn(in), cv::GOut(out))
        .compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}});

    const auto video_path = findDataFile("cv/video/768x576.avi");
    sc.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(video_path));
    sc.start();

    cv::VideoCapture cap;
    cap.open(video_path);

    cv::Mat tmp;
    cv::Scalar out_scl;
    std::size_t num_frames = 0u;
    while (sc.pull(cv::gout(out_scl)))
    {
        num_frames++;
        cap >> tmp;
        cv::Scalar out_ref = cv::mean(tmp);
        EXPECT_EQ(out_ref, out_scl);
    }
    EXPECT_LT(0u, num_frames);
}

TEST(GAPI_Streaming_Types, OutputVector)
{
    // This test verifies if Streaming works when pipeline
    // produces vector data only

    initTestDataPath();
    auto pkg = cv::gapi::kernels<TypesTest::OCVSumV>();

    cv::GMat in1, in2;
    cv::GMat roi = cv::gapi::crop(in2, cv::Rect(3,3,256,256));
    cv::GArray<int> out = TypesTest::SumV::on(cv::gapi::mul(roi, in1));
    auto sc = cv::GComputation(cv::GIn(in1, in2), cv::GOut(out))
        .compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{256,256}},
                          cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                          cv::compile_args(pkg));

    auto ocv_ref = [](const cv::Mat &ocv_in1,
                      const cv::Mat &ocv_in2,
                      std::vector<int> &ocv_out) {
        auto ocv_roi = ocv_in2(cv::Rect{3,3,256,256});
        TypesTest::OCVSumV::run(ocv_roi.mul(ocv_in1), ocv_out);
    };

    cv::Mat in_eye = cv::Mat::eye(cv::Size(256, 256), CV_8UC3);
    const auto video_path = findDataFile("cv/video/768x576.avi");
    sc.setSource(cv::gin(in_eye, gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(video_path)));
    sc.start();

    cv::VideoCapture cap;
    cap.open(video_path);

    cv::Mat tmp;
    std::vector<int> ref_vec;
    std::vector<int> out_vec;
    std::size_t num_frames = 0u;
    while (sc.pull(cv::gout(out_vec)))
    {
        num_frames++;
        cap >> tmp;
        ref_vec.clear();
        ocv_ref(in_eye, tmp, ref_vec);
        EXPECT_EQ(ref_vec, out_vec);
    }
    EXPECT_LT(0u, num_frames);
}

struct GAPI_Streaming_Unit: public ::testing::Test {
    cv::Mat m;

    cv::GComputation cc;
    cv::GStreamingCompiled sc;

    cv::GCompiled ref;

    GAPI_Streaming_Unit()
        : m(cv::Mat::ones(224,224,CV_8UC3))
        , cc([]{
                cv::GMat a, b;
                cv::GMat c = a + b*2;
                return cv::GComputation(cv::GIn(a, b), cv::GOut(c));
            })
    {
        initTestDataPath();

        const auto a_desc = cv::descr_of(m);
        const auto b_desc = cv::descr_of(m);
        sc  = cc.compileStreaming(a_desc, b_desc);
        ref = cc.compile(a_desc, b_desc);
    }
};

TEST_F(GAPI_Streaming_Unit, TestTwoVideoSourcesFail)
{
    const auto c_ptr = gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"));
    auto c_desc = cv::GMatDesc{CV_8U,3,{768,576}};
    auto m_desc = cv::descr_of(m);

    sc = cc.compileStreaming(c_desc, m_desc);
    EXPECT_NO_THROW(sc.setSource(cv::gin(c_ptr, m)));

    sc = cc.compileStreaming(m_desc, c_desc);
    EXPECT_NO_THROW(sc.setSource(cv::gin(m, c_ptr)));

    sc = cc.compileStreaming(c_desc, c_desc);
    EXPECT_ANY_THROW(sc.setSource(cv::gin(c_ptr, c_ptr)));
}

TEST_F(GAPI_Streaming_Unit, TestStartWithoutnSetSource)
{
    EXPECT_ANY_THROW(sc.start());
}

TEST_F(GAPI_Streaming_Unit, TestStopWithoutStart1)
{
    // It is ok!
    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, TestStopWithoutStart2)
{
    // It should be ok as well
    sc.setSource(cv::gin(m, m));
    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, StopStartStop)
{
    cv::Mat out;
    EXPECT_NO_THROW(sc.stop());
    EXPECT_NO_THROW(sc.setSource(cv::gin(m, m)));
    EXPECT_NO_THROW(sc.start());

    std::size_t i = 0u;
    while (i++ < 10u) {EXPECT_TRUE(sc.pull(cv::gout(out)));};

    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, ImplicitStop)
{
    EXPECT_NO_THROW(sc.setSource(cv::gin(m, m)));
    EXPECT_NO_THROW(sc.start());
    // No explicit stop here - pipeline stops successfully at the test exit
}

TEST_F(GAPI_Streaming_Unit, StartStopStart_NoSetSource)
{
    EXPECT_NO_THROW(sc.setSource(cv::gin(m, m)));
    EXPECT_NO_THROW(sc.start());
    EXPECT_NO_THROW(sc.stop());
    EXPECT_ANY_THROW(sc.start()); // Should fails since setSource was not called
}

TEST_F(GAPI_Streaming_Unit, StartStopStress_Const)
{
    // Runs 100 times with no deadlock - assumed stable (robust) enough
    for (int i = 0; i < 100; i++)
    {
        sc.stop();
        sc.setSource(cv::gin(m, m));
        sc.start();
        cv::Mat out;
        for (int j = 0; j < 5; j++) EXPECT_TRUE(sc.pull(cv::gout(out)));
    }
}

TEST_F(GAPI_Streaming_Unit, StartStopStress_Video)
{
    // Runs 100 times with no deadlock - assumed stable (robust) enough
    sc = cc.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                             cv::GMatDesc{CV_8U,3,cv::Size{768,576}});
    m = cv::Mat::eye(cv::Size{768,576}, CV_8UC3);
    for (int i = 0; i < 100; i++)
    {
        auto src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"));
        sc.stop();
        sc.setSource(cv::gin(src, m));
        sc.start();
        cv::Mat out;
        for (int j = 0; j < 5; j++) EXPECT_TRUE(sc.pull(cv::gout(out)));
    }
}

TEST_F(GAPI_Streaming_Unit, PullNoStart)
{
    sc.setSource(cv::gin(m, m));

    cv::Mat out;
    EXPECT_ANY_THROW(sc.pull(cv::gout(out)));
}


TEST_F(GAPI_Streaming_Unit, SetSource_Multi_BeforeStart)
{
    cv::Mat eye = cv::Mat::eye  (224, 224, CV_8UC3);
    cv::Mat zrs = cv::Mat::zeros(224, 224, CV_8UC3);

    // Call setSource two times, data specified last time
    // should be actually processed.
    sc.setSource(cv::gin(zrs, zrs));
    sc.setSource(cv::gin(eye, eye));

    // Run the pipeline, acquire result once
    sc.start();
    cv::Mat out, out_ref;
    EXPECT_TRUE(sc.pull(cv::gout(out)));
    sc.stop();

    // Pipeline should process `eye` mat, not `zrs`
    ref(cv::gin(eye, eye), cv::gout(out_ref));
    EXPECT_EQ(0., cv::norm(out, out_ref, cv::NORM_INF));
}

TEST_F(GAPI_Streaming_Unit, SetSource_During_Execution)
{
    cv::Mat zrs = cv::Mat::zeros(224, 224, CV_8UC3);

    sc.setSource(cv::gin(m, m));
    sc.start();
    EXPECT_ANY_THROW(sc.setSource(cv::gin(zrs, zrs)));
    EXPECT_ANY_THROW(sc.setSource(cv::gin(zrs, zrs)));
    EXPECT_ANY_THROW(sc.setSource(cv::gin(zrs, zrs)));
    sc.stop();
}

TEST_F(GAPI_Streaming_Unit, SetSource_After_Completion)
{
    sc.setSource(cv::gin(m, m));

    // Test pipeline with `m` input
    sc.start();
    cv::Mat out, out_ref;
    EXPECT_TRUE(sc.pull(cv::gout(out)));
    sc.stop();

    // Test against ref
    ref(cv::gin(m, m), cv::gout(out_ref));
    EXPECT_EQ(0., cv::norm(out, out_ref, cv::NORM_INF));

    // Now set another source
    cv::Mat eye = cv::Mat::eye(224, 224, CV_8UC3);
    sc.setSource(cv::gin(eye, m));
    sc.start();
    EXPECT_TRUE(sc.pull(cv::gout(out)));
    sc.stop();

    // Test against new ref
    ref(cv::gin(eye, m), cv::gout(out_ref));
    EXPECT_EQ(0., cv::norm(out, out_ref, cv::NORM_INF));
}

} // namespace opencv_test
