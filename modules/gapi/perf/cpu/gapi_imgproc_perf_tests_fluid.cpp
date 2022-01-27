// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_imgproc_perf_tests.hpp"

#define IMGPROC_FLUID cv::gapi::imgproc::fluid::kernels()
#define CORE_FLUID cv::gapi::core::fluid::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(SepFilterPerfTestFluid_8U, SepFilterPerfTest,
    Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
            Values(CV_8UC1, CV_8UC3),
            Values(3),
            Values(szVGA, sz720p, sz1080p),
            Values(-1, CV_16S, CV_32F),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SepFilterPerfTestFluid_other, SepFilterPerfTest,
    Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
            Values(CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),
            Values(szVGA, sz720p, sz1080p),
            Values(-1, CV_32F),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(Filter2DPerfTestFluid, Filter2DPerfTest,
    Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),                                     // TODO: add 4, 5, 7 when kernel is ready
            Values(szVGA, sz720p, sz1080p),
            Values(cv::BORDER_DEFAULT),
            Values(-1, CV_32F),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BoxFilterPerfTestFluid, BoxFilterPerfTest,
    Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),                                     // TODO: add size=5, when kernel is ready
            Values(szVGA, sz720p, sz1080p),
            Values(cv::BORDER_DEFAULT),
            Values(-1, CV_32F),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BlurPerfTestFluid, BlurPerfTest,
    Combine(Values(ToleranceFilter(1e-4f, 0.01).to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),                                     // TODO: add size=5, when kernel is ready
            Values(szVGA, sz720p, sz1080p),
            Values(cv::BORDER_DEFAULT),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(GaussianBlurPerfTestFluid, GaussianBlurPerfTest,
    Combine(Values(ToleranceFilter(1e-3f, 0.01).to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3, 5),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(MedianBlurPerfTestFluid, MedianBlurPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),                                     // TODO: add size=5, when kernel is ready
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(ErodePerfTestFluid, ErodePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),                                     // TODO: add size=5, when kernel is ready
            Values(szVGA, sz720p, sz1080p),
            Values(cv::MorphShapes::MORPH_RECT,
                   cv::MorphShapes::MORPH_CROSS,
                   cv::MorphShapes::MORPH_ELLIPSE),
            Values(cv::compile_args(IMGPROC_FLUID))));

// GAPI/fluid does not support iterations parameter for the Erode kernel
INSTANTIATE_TEST_CASE_P(DISABLED_Erode3x3PerfTestFluid, Erode3x3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(szVGA, sz720p, sz1080p),
            Values(1, 2, 4),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(DilatePerfTestFluid, DilatePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(3),                                     // TODO: add size=5, when kernel is ready
            Values(szVGA, sz720p, sz1080p),
            Values(cv::MorphShapes::MORPH_RECT,
                   cv::MorphShapes::MORPH_CROSS,
                   cv::MorphShapes::MORPH_ELLIPSE),
            Values(cv::compile_args(IMGPROC_FLUID))));

// GAPI/fluid does not support iterations parameter for the Dilate kernel
INSTANTIATE_TEST_CASE_P(DISABLED_Dilate3x3PerfTestFluid, Dilate3x3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(szVGA, sz720p, sz1080p),
            Values(1, 2, 4),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestFluid, SobelPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(3),                                     // TODO: add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(-1, CV_16S, CV_32F),
            Values(0, 1),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SobelPerfTestFluid32F, SobelPerfTest,
    Combine(Values(ToleranceFilter(1e-3f, 0.0).to_compare_f()),
            Values(CV_32FC1),
            Values(3),                                     // TODO: add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(CV_32F),
            Values(0, 1),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SobelXYPerfTestFluid, SobelXYPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(3),                                     // TODO: add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(-1, CV_16S, CV_32F),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(SobelXYPerfTestFluid32F, SobelXYPerfTest,
    Combine(Values(ToleranceFilter(1e-3f, 0.0).to_compare_f()),
            Values(CV_32FC1),
            Values(3),                                     // TODO: add 5x5 once supported
            Values(szVGA, sz720p, sz1080p),
            Values(CV_32F),
            Values(1, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2GrayPerfTestFluid, RGB2GrayPerfTest,
    Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BGR2GrayPerfTestFluid, BGR2GrayPerfTest,
    Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2YUVPerfTestFluid, RGB2YUVPerfTest,
    Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(YUV2RGBPerfTestFluid, YUV2RGBPerfTest,
    Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BGR2YUVPerfTestFluid, BGR2YUVPerfTest,
    Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(YUV2BGRPerfTestFluid, YUV2BGRPerfTest,
    Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BayerGR2RGBPerfTestFluid, BayerGR2RGBPerfTest,
        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2YUV422PerfTestFluid, RGB2YUV422PerfTest,
        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2HSVPerfTestFluid, RGB2HSVPerfTest,
        Combine(Values(ToleranceColor(1e-3).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(BGR2LUVPerfTestFluid, BGR2LUVPerfTest,
    Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(RGB2LabPerfTestFluid, RGB2LabPerfTest,
    Combine(Values(AbsSimilarPoints(1, 0.05).to_compare_f()),
            Values(szVGA, sz720p, sz1080p),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(ResizePerfTestFluid, ResizePerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()),
            Values(CV_8UC3),
            Values(cv::INTER_LINEAR),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::Size(64, 64),
                   cv::Size(30, 30)),
            Values(cv::compile_args(IMGPROC_FLUID))));

#define IMGPROC_FLUID cv::gapi::imgproc::fluid::kernels()
INSTANTIATE_TEST_CASE_P(BottleneckKernelsPerfTestFluid, BottleneckKernelsConstInputPerfTest,
    Combine(Values(AbsSimilarPoints(0, 1).to_compare_f()),
            Values("cv/optflow/frames/1080p_00.png", "cv/optflow/frames/720p_00.png",
                   "cv/optflow/frames/VGA_00.png", "cv/dnn_face/recognition/Aaron_Tippin_0001.jpg"),
            Values(cv::compile_args(IMGPROC_FLUID))));

INSTANTIATE_TEST_CASE_P(ResizeInSimpleGraphPerfTestFluid, ResizeInSimpleGraphPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()),
            Values(CV_8UC3),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(0.5),
            Values(0.5),
            Values(cv::compile_args(cv::gapi::combine(IMGPROC_FLUID, CORE_FLUID)))));

INSTANTIATE_TEST_CASE_P(ResizeFxFyPerfTestFluid, ResizeFxFyPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()),
            Values(CV_8UC3),
            Values(cv::INTER_LINEAR),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(0.5, 0.25, 2),
            Values(0.5, 0.25, 2),
            Values(cv::compile_args(IMGPROC_FLUID))));
}
