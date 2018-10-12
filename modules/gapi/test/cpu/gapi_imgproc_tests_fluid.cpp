// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_imgproc_tests.hpp"
#include "backends/fluid/gfluidimgproc.hpp"

#define IMGPROC_FLUID cv::gapi::imgproc::fluid::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(RGB2GrayTestFluid, RGB2GrayTest,
                        Combine(Values(cv::Size(1280, 720),
                                cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestFluid, BGR2GrayTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestFluid, RGB2YUVTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestFluid, YUV2RGBTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2LabTestFluid, RGB2LabTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

// FIXME: Not supported by Fluid yet (no kernel implemented)
INSTANTIATE_TEST_CASE_P(BGR2LUVTestFluid, BGR2LUVTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(blurTestFluid, BlurTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
                                Values(0.0),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(gaussBlurTestFluid, GaussianBlurTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(medianBlurTestFluid, MedianBlurTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(erodeTestFluid, ErodeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(dilateTestFluid, DilateTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SobelTestFluid, SobelTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(0, 1),
                                Values(1, 2),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(boxFilterTestFluid, BoxFilterTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(0.0),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

// FIXME: Tests are failing on Fluid backend (accuracy issue?)
INSTANTIATE_TEST_CASE_P(sepFilterTestFluid, SepFilterTest,
                        Combine(Values(CV_32FC1),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(filter2DTestFluid, Filter2DTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(3), // add kernel size=4,5,7 when implementation ready
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_FLUID))));

} // opencv_test
