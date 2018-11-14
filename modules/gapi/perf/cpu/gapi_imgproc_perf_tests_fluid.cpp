// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_imgproc_perf_tests.hpp"
#include "../../src/backends/fluid/gfluidimgproc.hpp"


#define IMGPROC_FLUID cv::gapi::imgproc::fluid::kernels()

namespace opencv_test
{

    INSTANTIATE_TEST_CASE_P(SobelPerfTestFluid, SobelPerfTest,
        Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),  // add CV_32FC1 when ready
            Values(3),                                     // add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(-1, CV_16S, CV_32F),
            Values(0, 1),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

    INSTANTIATE_TEST_CASE_P(SobelPerfTestFluid32F, SobelPerfTest,
        Combine(Values(ToleranceFilter(1e-3f, 0.0).to_compare_f()),
            Values(CV_32FC1),
            Values(3),                                     // add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(CV_32F),
            Values(0, 1),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

}
