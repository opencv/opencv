// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_video_tests.hpp"
#include <opencv2/gapi/cpu/video.hpp>

namespace
{
#define VIDEO_CPU [] () { return cv::compile_args(cv::gapi::video::cpu::kernels()); }

#ifdef HAVE_OPENCV_VIDEO
#define WITH_VIDEO(X) X
#else
#define WITH_VIDEO(X) DISABLED_##X
#endif // HAVE_OPENCV_VIDEO

#define INSTANTIATE_TEST_CASE_MACRO_P(prefix, test_case_name, generator, ...) \
    INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator, __VA_ARGS__)
}  // anonymous namespace

namespace opencv_test
{
INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildOptFlowPyramidTestCPU), BuildOptFlowPyramidTest,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/rock_1.bmp",
                                             "cv/optflow/frames/1080p_01.png"),
                                      Values(7, 11),
                                      Values(1000),
                                      testing::Bool(),
                                      Values(BORDER_DEFAULT, BORDER_TRANSPARENT),
                                      Values(BORDER_DEFAULT, BORDER_TRANSPARENT),
                                      testing::Bool()));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildOptFlowPyramidInternalTestCPU),
                              BuildOptFlowPyramidTest,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/rock_1.bmp"),
                                      Values(15),
                                      Values(3),
                                      Values(true),
                                      Values(BORDER_REFLECT_101),
                                      Values(BORDER_CONSTANT),
                                      Values(true)));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(OptFlowLKTestCPU), OptFlowLKTest,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/rock_%01d.bmp",
                                             "cv/optflow/frames/1080p_%02d.png"),
                                      Values(1, 3, 4),
                                      Values(std::make_tuple(9, 9), std::make_tuple(15, 15)),
                                      Values(7, 11),
                                      Values(cv::TermCriteria(cv::TermCriteria::COUNT |
                                                              cv::TermCriteria::EPS,
                                                              30, 0.01))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(OptFlowLKTestForPyrCPU), OptFlowLKTestForPyr,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/rock_%01d.bmp",
                                             "cv/optflow/frames/1080p_%02d.png"),
                                      Values(1, 3, 4),
                                      Values(std::make_tuple(9, 9), std::make_tuple(15, 15)),
                                      Values(7, 11),
                                      Values(cv::TermCriteria(cv::TermCriteria::COUNT |
                                                              cv::TermCriteria::EPS,
                                                              30, 0.01)),
                                      testing::Bool()));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(OptFlowLKInternalTestCPU), OptFlowLKTestForPyr,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/rock_%01d.bmp"),
                                      Values(1),
                                      Values(std::make_tuple(10, 10)),
                                      Values(15),
                                      Values(cv::TermCriteria(cv::TermCriteria::COUNT |
                                                              cv::TermCriteria::EPS,
                                                              21, 0.05)),
                                      Values(true)));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildPyr_CalcOptFlow_PipelineTestCPU),
                              BuildPyr_CalcOptFlow_PipelineTest,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/frames/1080p_%02d.png"),
                                      Values(7, 11),
                                      Values(1000),
                                      testing::Bool()));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildPyr_CalcOptFlow_PipelineInternalTestCPU),
                              BuildPyr_CalcOptFlow_PipelineTest,
                              Combine(Values(VIDEO_CPU),
                                      Values("cv/optflow/rock_%01d.bmp"),
                                      Values(15),
                                      Values(3),
                                      Values(true)));
} // opencv_test
