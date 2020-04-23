// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_imgproc_perf_tests.hpp"

#define IMGPROC_GPU cv::gapi::imgproc::gpu::kernels()

namespace opencv_test
{


INSTANTIATE_TEST_CASE_P(SepFilterPerfTestGPU_8U, SepFilterPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1, CV_16S, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SepFilterPerfTestGPU_other, SepFilterPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU))));



INSTANTIATE_TEST_CASE_P(Filter2DPerfTestGPU, Filter2DPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 4, 5, 7),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BoxFilterPerfTestGPU, BoxFilterPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(/*CV_8UC1,*/ CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3,5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::BORDER_DEFAULT),
                                Values(-1, CV_32F),
                                Values(cv::compile_args(IMGPROC_GPU)))); //TODO: 8UC1 doesn't work

INSTANTIATE_TEST_CASE_P(BlurPerfTestGPU, BlurPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::BORDER_DEFAULT),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(GaussianBlurPerfTestGPU, GaussianBlurPerfTest,
                        Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()), //TODO: too relaxed?
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(MedianBlurPerfTestGPU, MedianBlurPerfTest,
                         Combine(Values(AbsExact().to_compare_f()),
                                 Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                 Values(3, 5),
                                 Values(szVGA, sz720p, sz1080p),
                                 Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(ErodePerfTestGPU, ErodePerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Erode3x3PerfTestGPU, Erode3x3PerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(DilatePerfTestGPU, DilatePerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(cv::MorphShapes::MORPH_RECT,
                                       cv::MorphShapes::MORPH_CROSS,
                                       cv::MorphShapes::MORPH_ELLIPSE),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(Dilate3x3PerfTestGPU, Dilate3x3PerfTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(1,2,4),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestGPU, SobelPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1, CV_16S, CV_32F),
                                Values(0, 1),
                                Values(1, 2),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestGPU32F, SobelPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_32FC1),
                                Values(3, 5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(CV_32F),
                                Values(0, 1),
                                Values(1, 2),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(LaplacianPerfTestGPU, LaplacianPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(5),
                                Values(szVGA, sz720p, sz1080p),
                                Values(-1),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BilateralFilterPerfTestGPU, BilateralFilterPerfTest,
                        Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
                                Values(CV_32FC1, CV_32FC3),
                                Values(-1),
                                Values(szVGA, sz720p, sz1080p),
                                Values(5),
                                Values(100),
                                Values(40),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(CannyPerfTestGPU, CannyPerfTest,
                        Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()),
                                Values(CV_8UC1, CV_8UC3),
                                Values(szVGA, sz720p, sz1080p),
                                Values(3.0, 120.0),
                                Values(125.0, 240.0),
                                Values(3, 5),
                                Values(true, false),
                                Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(EqHistPerfTestGPU, EqHistPerfTest,
                        Combine(Values(AbsExact().to_compare_f()),  // FIXIT unrealiable check
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2GrayPerfTestGPU, RGB2GrayPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2GrayPerfTestGPU, BGR2GrayPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2YUVPerfTestGPU, RGB2YUVPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2RGBPerfTestGPU, YUV2RGBPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(RGB2LabPerfTestGPU, RGB2LabPerfTest,
                        Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2LUVPerfTestGPU, BGR2LUVPerfTest,
                        Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(LUV2BGRPerfTestGPU, LUV2BGRPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(BGR2YUVPerfTestGPU, BGR2YUVPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

INSTANTIATE_TEST_CASE_P(YUV2BGRPerfTestGPU, YUV2BGRPerfTest,
                        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
                        Values(szVGA, sz720p, sz1080p),
                        Values(cv::compile_args(IMGPROC_GPU))));

}
