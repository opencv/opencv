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
INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildOptFlowPyramidPerfTestCPU),
                              BuildOptFlowPyramidPerfTest,
                              Combine(Values("cv/optflow/rock_1.bmp",
                                             "cv/optflow/frames/1080p_01.png"),
                                      Values(7, 11),
                                      Values(1000),
                                      testing::Bool(),
                                      Values(BORDER_DEFAULT, BORDER_TRANSPARENT),
                                      Values(BORDER_DEFAULT, BORDER_TRANSPARENT),
                                      testing::Bool(),
                                      Values(cv::compile_args(VIDEO_CPU))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildOptFlowPyramidInternalPerfTestCPU),
                              BuildOptFlowPyramidPerfTest,
                              Combine(Values("cv/optflow/rock_1.bmp"),
                                      Values(15),
                                      Values(3),
                                      Values(true),
                                      Values(BORDER_REFLECT_101),
                                      Values(BORDER_CONSTANT),
                                      Values(true),
                                      Values(cv::compile_args(VIDEO_CPU))));

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
                                      testing::Bool(),
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

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildPyr_CalcOptFlow_PipelinePerfTestCPU),
                              BuildPyr_CalcOptFlow_PipelinePerfTest,
                              Combine(Values("cv/optflow/frames/1080p_%02d.png"),
                                      Values(7, 11),
                                      Values(1000),
                                      testing::Bool(),
                                      Values(cv::compile_args(VIDEO_CPU))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BuildPyr_CalcOptFlow_PipelineInternalTestPerfCPU),
                              BuildPyr_CalcOptFlow_PipelinePerfTest,
                              Combine(Values("cv/optflow/rock_%01d.bmp"),
                                      Values(15),
                                      Values(3),
                                      Values(true),
                                      Values(cv::compile_args(VIDEO_CPU))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(BackgroundSubtractorPerfTestCPU),
                              BackgroundSubtractorPerfTest,
                              Combine(Values(cv::gapi::video::TYPE_BS_MOG2,
                                             cv::gapi::video::TYPE_BS_KNN),
                                      Values("cv/video/768x576.avi", "cv/video/1920x1080.avi"),
                                      testing::Bool(),
                                      Values(0., 0.5, 1.),
                                      Values(5),
                                      Values(cv::compile_args(VIDEO_CPU)),
                                      Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(KalmanFilterControlPerfTestCPU),
                              KalmanFilterControlPerfTest,
                              Combine(Values(CV_32FC1, CV_64FC1),
                                      Values(2, 5),
                                      Values(2, 5),
                                      Values(5),
                                      testing::Bool(),
                                      Values(cv::compile_args(VIDEO_CPU))));

INSTANTIATE_TEST_CASE_MACRO_P(WITH_VIDEO(KalmanFilterNoControlPerfTestCPU),
                              KalmanFilterNoControlPerfTest,
                              Combine(Values(CV_32FC1, CV_64FC1),
                                      Values(2, 5),
                                      Values(2, 5),
                                      Values(5),
                                      testing::Bool(),
                                      Values(cv::compile_args(VIDEO_CPU))));
} // opencv_test
