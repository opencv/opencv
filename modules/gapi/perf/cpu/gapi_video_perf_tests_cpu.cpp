// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../perf_precomp.hpp"

#include "../common/gapi_video_perf_tests.hpp"
#include <opencv2/gapi/cpu/video.hpp>

namespace
{
#define VIDEO_CPU cv::gapi::video::cpu::kernels()

#ifdef HAVE_OPENCV_VIDEO
#define WITH_VIDEO(X) X
#else
#define WITH_VIDEO(X) DISABLED_##X
#endif // HAVE_OPENCV_VIDEO

#define INSTANTIATE_TEST_CASE_MACRO_P(prefix, test_case_name, generator, ...) \
    INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator, __VA_ARGS__)
} // namespace


namespace opencv_test
{
INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(OptFlowLKPerfTestCPU), OptFlowLKPerfTest,
                              Combine(Values("cv/optflow/rock_%01d.bmp",
                                             "cv/optflow/frames/1080p_%02d.png"),
                                      Values(1, 3, 4),
                                      Values(std::make_tuple(9, 9), std::make_tuple(15, 15)),
                                      Values(7, 11),
                                      Values(cv::TermCriteria(cv::TermCriteria::COUNT |
                                                              cv::TermCriteria::EPS,
                                                              30, 0.01)),
                                      Values(cv::compile_args(VIDEO_CPU))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(OptFlowLKForPyrPerfTestCPU), OptFlowLKForPyrPerfTest,
                              Combine(Values("cv/optflow/rock_%01d.bmp",
                                             "cv/optflow/frames/1080p_%02d.png"),
                                      Values(1, 3, 4),
                                      Values(std::make_tuple(9, 9), std::make_tuple(15, 15)),
                                      Values(7, 11),
                                      Values(cv::TermCriteria(cv::TermCriteria::COUNT |
                                                              cv::TermCriteria::EPS,
                                                              30, 0.01)),
                                      Values(true, false),
                                      Values(cv::compile_args(VIDEO_CPU))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(OptFlowLKInternalPerfTestCPU),
                              OptFlowLKForPyrPerfTest,
                              Combine(Values("cv/optflow/rock_%01d.bmp"),
                                      Values(1),
                                      Values(std::make_tuple(10, 10)),
                                      Values(15),
                                      Values(cv::TermCriteria(cv::TermCriteria::COUNT |
                                                              cv::TermCriteria::EPS,
                                                              21, 0.05)),
                                      Values(true),
                                      Values(cv::compile_args(VIDEO_CPU))));
} // opencv_test
