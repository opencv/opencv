// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_stereo_tests.hpp"

#include <opencv2/gapi/stereo.hpp> // For ::gapi::stereo::disparity/depth
#include <opencv2/gapi/cpu/stereo.hpp>

namespace
{
#define STEREO_CPU [] () { return cv::compile_args(cv::gapi::use_only{cv::gapi::calib3d::cpu::kernels()}); }
}  // anonymous namespace

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(CPU_Tests, TestGAPIStereo,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_32FC1),
                                Values(STEREO_CPU),
                                Values(cv::gapi::StereoOutputFormat::DEPTH_FLOAT16,
                                       cv::gapi::StereoOutputFormat::DEPTH_FLOAT32,
                                       cv::gapi::StereoOutputFormat::DISPARITY_FIXED16_12_4,
                                       cv::gapi::StereoOutputFormat::DEPTH_16F,
                                       cv::gapi::StereoOutputFormat::DEPTH_32F,
                                       cv::gapi::StereoOutputFormat::DISPARITY_16Q_11_4),
                                Values(16),
                                Values(43),
                                Values(63.5),
                                Values(3.6),
                                Values(AbsExact().to_compare_obj())));

} // opencv_test
