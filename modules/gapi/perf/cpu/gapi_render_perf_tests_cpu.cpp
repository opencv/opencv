// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_render_perf_tests.hpp"
#include <opencv2/gapi/render.hpp>

#define RENDER_CPU cv::gapi::render::ocv::kernels()

namespace opencv_test
{


INSTANTIATE_TEST_CASE_P(RenderTestTextsCPU_8U, RenderTestTexts,
    Combine(Values(AbsExact().to_compare_obj()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p, cv::Size(12, 12)),
        Values(cv::compile_args(RENDER_CPU))));

INSTANTIATE_TEST_CASE_P(RenderTestRectsCPU_8U, RenderTestRects,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(RENDER_CPU))));

INSTANTIATE_TEST_CASE_P(RenderTestCirclesCPU_8U, RenderTestCircles,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(RENDER_CPU))));

INSTANTIATE_TEST_CASE_P(RenderTestLinesCPU_8U, RenderTestCircles,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(RENDER_CPU))));

INSTANTIATE_TEST_CASE_P(RenderTestMosaicsCPU_8U, RenderTestCircles,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(RENDER_CPU))));

INSTANTIATE_TEST_CASE_P(RenderTestImagesCPU_8U, RenderTestCircles,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(RENDER_CPU))));

INSTANTIATE_TEST_CASE_P(RenderTestPolylinesCPU_8U, RenderTestPolylines,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_8UC3),
        Values(szVGA, sz720p, sz1080p),
        Values(cv::compile_args(RENDER_CPU))));
}
