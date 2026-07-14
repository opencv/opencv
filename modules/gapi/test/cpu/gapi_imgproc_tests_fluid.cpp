// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_imgproc_tests.hpp"

namespace
{
#define IMGPROC_FLUID [] () { return cv::compile_args(cv::gapi::use_only{cv::gapi::imgproc::fluid::kernels()}); }
}  // anonymous namespace

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(ResizeTestFluid, ResizeTest,
                        Combine(Values(CV_8UC3/*CV_8UC1, CV_16UC1, CV_16SC1*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(30, 30)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_obj()),
                                Values(/*cv::INTER_NEAREST,*/ cv::INTER_LINEAR/*, cv::INTER_AREA*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(30, 30))));

INSTANTIATE_TEST_CASE_P(ResizeTestFxFyFluid, ResizeTestFxFy,
                        Combine(Values(CV_8UC3/*CV_8UC1, CV_16UC1, CV_16SC1*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(30, 30)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_obj()),
                                Values(/*cv::INTER_NEAREST,*/ cv::INTER_LINEAR/*, cv::INTER_AREA*/),
                                Values(0.5, 1, 2),
                                Values(0.5, 1, 2)));

INSTANTIATE_TEST_CASE_P(RGB2GrayTestFluid, RGB2GrayTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(1e-3).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestFluid, BGR2GrayTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC1),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(1e-3).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestFluid, RGB2YUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(1e-3).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestFluid, YUV2RGBTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(1e-3).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2LabTestFluid, RGB2LabTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_FLUID),
                                Values(AbsSimilarPoints(1, 0.05).to_compare_obj())));

// FIXME: Not supported by Fluid yet (no kernel implemented)
INSTANTIATE_TEST_CASE_P(BGR2LUVTestFluid, BGR2LUVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(5e-3, 6).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2HSVTestFluid, RGB2HSVTest,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(1e-3).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BayerGR2RGBTestFluid, BayerGR2RGBTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC3),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceColor(1e-3).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(RGB2YUV422TestFluid, RGB2YUV422Test,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720)),
                                Values(CV_8UC2),
                                Values(IMGPROC_FLUID),
                                Values(AbsTolerance(1).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(blurTestFluid, BlurTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-4f, 0.01).to_compare_obj()),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(gaussBlurTestFluid, GaussianBlurTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-3f, 0.01).to_compare_obj()),
                                Values(3, 5)));

INSTANTIATE_TEST_CASE_P(medianBlurTestFluid, MedianBlurTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(3))); // add kernel size=5 when implementation is ready

INSTANTIATE_TEST_CASE_P(erodeTestFluid, ErodeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE)));

INSTANTIATE_TEST_CASE_P(dilateTestFluid, DilateTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1),
                                Values(IMGPROC_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE)));

INSTANTIATE_TEST_CASE_P(SobelTestFluid, SobelTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(0, 1),
                                Values(1, 2)));

INSTANTIATE_TEST_CASE_P(SobelTestFluid32F, SobelTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-4f, 0.01).to_compare_obj()),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(0, 1),
                                Values(1, 2)));

INSTANTIATE_TEST_CASE_P(SobelXYTestFluid, SobelXYTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_16S, CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(3),
                                Values(1, 2),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101),
                                Values(0, 1, 255)));

INSTANTIATE_TEST_CASE_P(SobelXYTestFluid32F, SobelXYTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-4f, 0.01).to_compare_obj()),
                                Values(3),
                                Values(1, 2),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101),
                                Values(0, 1, 255)));

INSTANTIATE_TEST_CASE_P(boxFilterTestFluid32, BoxFilterTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-4f, 0.01).to_compare_obj()),
                                Values(3), // add kernel size=5 when implementation is ready
                                Values(cv::BORDER_DEFAULT)));

INSTANTIATE_TEST_CASE_P(sepFilterTestFluid, SepFilterTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-4f, 0.01).to_compare_obj()),
                                Values(3))); // add kernel size=5 when implementation is ready

INSTANTIATE_TEST_CASE_P(filter2DTestFluid, Filter2DTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(128, 128)),
                                Values(-1, CV_32F),
                                Values(IMGPROC_FLUID),
                                Values(ToleranceFilter(1e-4f, 0.01).to_compare_obj()),
                                Values(cv::Size(3, 3)), // add kernel size=4x4,5x5,7x7 when implementation ready
                                Values(cv::BORDER_DEFAULT)));

} // opencv_test
