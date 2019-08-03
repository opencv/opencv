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

cv::gapi::GKernelPackage OCL_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(cv::gapi::core::ocl::kernels(),
                          cv::gapi::imgproc::ocl::kernels());
    return pkg;
}

cv::gapi::GKernelPackage OCV_FLUID_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(OCV_KERNELS(),
                          cv::gapi::core::fluid::kernels());
    return pkg;
}

cv::gapi::GKernelPackage OCL_FLUID_KERNELS()
{
    static cv::gapi::GKernelPackage pkg =
        cv::gapi::combine(OCL_KERNELS(),
                          cv::gapi::core::fluid::kernels());
    return pkg;
}

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
        if (cv::countNonZero(out_mat_gapi != out_mat_ocv) > 0) {
            cv::imshow("G-API", out_mat_gapi);
            cv::imshow("OpenCV", out_mat_ocv);
            cv::imshow("Delta", out_mat_gapi - out_mat_ocv);
            cv::waitKey(0);
        }
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

INSTANTIATE_TEST_CASE_P(TestStreaming, GAPI_Streaming,
                        Values(  OCV_KERNELS()
                             //, OCL_KERNELS()       -- known issues with OpenCL backend, commented out
                               , OCV_FLUID_KERNELS()
                             //, OCL_FLUID_KERNELS() -- known issues with OpenCL backend, commented out
                                 ));
} // namespace opencv_test
