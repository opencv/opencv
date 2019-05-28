// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_imgproc_tests.hpp"

#define IMGPROC_GPU cv::gapi::imgproc::gpu::kernels()

namespace opencv_test
{


INSTANTIATE_TEST_CASE_P(Filter2DTestGPU, Filter2DTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 4, 5, 7),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BoxFilterTestGPU, BoxFilterTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_f()),
                                Values(/*CV_8UC1,*/ CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3,5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));  //TODO: 8UC1 doesn't work

INSTANTIATE_TEST_CASE_P(SepFilterTestGPU_8U, SepFilterTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_16S, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SepFilterTestGPU_other, SepFilterTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BlurTestGPU, BlurTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-4, 2).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3,5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::BORDER_DEFAULT),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(gaussBlurTestGPU, GaussianBlurTest,
                        Combine(Values(ToleranceFilter(1e-5f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3),  // FIXIT 5
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(MedianBlurTestGPU, MedianBlurTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(ErodeTestGPU, ErodeTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Erode3x3TestGPU, Erode3x3Test,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(DilateTestGPU, DilateTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Dilate3x3TestGPU, Dilate3x3Test,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SobelTestGPU, SobelTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-4, 2).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(-1, CV_16S, CV_32F),
                                Values(0, 1),
                                Values(1, 2),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SobelTestGPU32F, SobelTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-4, 2).to_compare_f()),
                                Values(CV_32FC1),
                                Values(3, 5),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(CV_32F),
                                Values(0, 1),
                                Values(1, 2),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(EqHistTestGPU, EqHistTest,
                        Combine(Values(AbsExact().to_compare_f()),  // FIXIT Non reliable check
                                Values(cv::Size(1280, 720),
                                cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(CannyTestGPU, CannyTest,
                        Combine(Values(AbsSimilarPoints(0, 0.05).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
                                Values(3.0, 120.0),
                                Values(125.0, 240.0),
                                Values(3, 5),
                                testing::Bool(),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2GrayTestGPU, RGB2GrayTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2GrayTestGPU, BGR2GrayTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2YUVTestGPU, RGB2YUVTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2RGBTestGPU, YUV2RGBTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2LabTestGPU, RGB2LabTest,
                        Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2LUVTestGPU, BGR2LUVTest,
                        Combine(Values(ToleranceColor(5e-3, 6).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(LUV2BGRTestGPU, LUV2BGRTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2YUVTestGPU, BGR2YUVTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2BGRTestGPU, YUV2BGRTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(IMGPROC_GPU))));


} // opencv_test
