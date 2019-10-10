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

#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

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

cv::gapi::GKernelPackage OCV_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(cv::gapi::core::cpu::kernels(),
                          cv::gapi::imgproc::cpu::kernels());
    return pkg;
}

cv::gapi::GKernelPackage OCV_FLUID_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(OCV_KERNELS(),
                          cv::gapi::core::fluid::kernels());
    return pkg;
}

#if 0
// FIXME: OpenCL backend seem to work fine with Streaming
// however the results are not very bit exact with CPU
// It may be a problem but may be just implementation innacuracy.
// Need to customize the comparison function in tests where OpenCL
// is involved.
cv::gapi::GKernelPackage OCL_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(cv::gapi::core::ocl::kernels(),
                          cv::gapi::imgproc::ocl::kernels());
    return pkg;
}

cv::gapi::GKernelPackage OCL_FLUID_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(OCL_KERNELS(),
                          cv::gapi::core::fluid::kernels());
    return pkg;
}
#endif // 0

struct GAPI_Streaming: public ::testing::TestWithParam<cv::gapi::GKernelPackage> {
    GAPI_Streaming() { initTestDataPath(); }
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
                                    cv::compile_args(cv::gapi::use_only{GetParam()}));
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
                                    cv::compile_args(cv::gapi::use_only{GetParam()}));
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(cv::gapi::GVideoCapture{findDataFile("cv/video/768x576.avi")});

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Process the full video
    cv::Mat in_mat_gapi, out_mat_gapi;

    while (ccomp.pull(cv::gout(in_mat_gapi, out_mat_gapi))) {
        cv::Mat out_mat_ocv;
        opencv_ref(in_mat_gapi, out_mat_ocv);
        EXPECT_EQ(0, cv::countNonZero(out_mat_gapi != out_mat_ocv));
    }
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
                                    cv::compile_args(cv::gapi::use_only{GetParam()}));

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

TEST_P(GAPI_Streaming, TestStartRestart)
{
    cv::GMat in;
    auto res = cv::gapi::resize(in, cv::Size{300,200});
    auto out = cv::gapi::Canny(res, 95, 220);
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in), out));

    auto ccomp = c.compileStreaming(cv::GMatDesc{CV_8U,3,cv::Size{768,576}},
                                    cv::compile_args(cv::gapi::use_only{GetParam()}));
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // Run 1
    std::size_t num_frames1 = 0u;
    ccomp.setSource(cv::gapi::GVideoCapture{findDataFile("cv/video/768x576.avi")});
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    cv::Mat out1, out2;
    while (ccomp.pull(cv::gout(out1, out2))) num_frames1++;

    EXPECT_FALSE(ccomp.running());

    // Run 2
    std::size_t num_frames2 = 0u;
    ccomp.start();
    EXPECT_TRUE(ccomp.running());
    while (ccomp.pull(cv::gout(out1, out2))) num_frames2++;

    EXPECT_FALSE(ccomp.running());

    EXPECT_LT(0u, num_frames1);
    EXPECT_LT(0u, num_frames2);
    EXPECT_EQ(num_frames1, num_frames2);
}

INSTANTIATE_TEST_CASE_P(TestStreaming, GAPI_Streaming,
                        Values(  OCV_KERNELS()
                                 //, OCL_KERNELS() // FIXME: Fails bit-exactness check, maybe relax it?
                               , OCV_FLUID_KERNELS()
                                 //, OCL_FLUID_KERNELS() // FIXME: Fails bit-exactness check, maybe relax it?
                               ));

struct GAPI_Streaming_Unit: public ::testing::Test {
    cv::GStreamingCompiled sc;
    cv::Mat m;

    cv::GCompiled ref;

    GAPI_Streaming_Unit()
        : m(cv::Mat::ones(224,224,CV_8UC3))
    {
        initTestDataPath();

        cv::GMat a, b;
        cv::GMat c = a + b*2;
        auto cc = cv::GComputation(cv::GIn(a, b), cv::GOut(c));

        const auto a_desc = cv::descr_of(m);
        const auto b_desc = cv::descr_of(m);
        sc  = cc.compileStreaming(a_desc, b_desc);
        ref = cc.compile(a_desc, b_desc);
    }
};

TEST_F(GAPI_Streaming_Unit, TestTwoVideoSourcesFail)
{
    EXPECT_NO_THROW(sc.setSource(cv::gin(cv::gapi::GVideoCapture{findDataFile("cv/video/768x576.avi")}, m)));
    EXPECT_NO_THROW(sc.setSource(cv::gin(m, cv::gapi::GVideoCapture{findDataFile("cv/video/768x576.avi")})));
    EXPECT_ANY_THROW(sc.setSource(cv::gin(cv::gapi::GVideoCapture{findDataFile("cv/video/768x576.avi")},
                                          cv::gapi::GVideoCapture{findDataFile("cv/video/768x576.avi")})));
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
    while (i++ < 10u) sc.pull(cv::gout(out));

    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, ImplicitStop)
{
    EXPECT_NO_THROW(sc.setSource(cv::gin(m, m)));
    EXPECT_NO_THROW(sc.start());
    // No explicit stop here - pipeline stops successfully at the test exit
}

TEST_F(GAPI_Streaming_Unit, StartStopStress)
{
    // Runs 100 times with no deadlock - assumed stable (robust) enough
    sc.setSource(cv::gin(m, m));
    for (int i = 0; i < 100; i++)
    {
        sc.stop();
        sc.start();
        cv::Mat out;
        for (int j = 0; j < 5; j++) EXPECT_TRUE(sc.pull(cv::gout(out)));
    }
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
    sc.pull(cv::gout(out));
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
    sc.pull(cv::gout(out));
    sc.stop();

    // Test against ref
    ref(cv::gin(m, m), cv::gout(out_ref));
    EXPECT_EQ(0., cv::norm(out, out_ref, cv::NORM_INF));

    // Now set another source
    cv::Mat eye = cv::Mat::eye(224, 224, CV_8UC3);
    sc.setSource(cv::gin(eye, m));
    sc.start();
    sc.pull(cv::gout(out));
    sc.stop();

    // Test against new ref
    ref(cv::gin(eye, m), cv::gout(out_ref));
    EXPECT_EQ(0., cv::norm(out, out_ref, cv::NORM_INF));
}


} // namespace opencv_test
