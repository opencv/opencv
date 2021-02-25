// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_core_perf_tests.hpp"
#include <opencv2/gapi/cpu/core.hpp>

#define CORE_CPU cv::gapi::core::cpu::kernels()

namespace opencv_test
{


INSTANTIATE_TEST_CASE_P(AddPerfTestCPU, AddPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AddCPerfTestCPU, AddCPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SubPerfTestCPU, SubPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SubCPerfTestCPU, SubCPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SubRCPerfTestCPU, SubRCPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MulPerfTestCPU, MulPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MulDoublePerfTestCPU, MulDoublePerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MulCPerfTestCPU, MulCPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivPerfTestCPU, DivPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivCPerfTestCPU, DivCPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivRCPerfTestCPU, DivRCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MaskPerfTestCPU, MaskPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_16UC1, CV_16SC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MeanPerfTestCPU, MeanPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Polar2CartPerfTestCPU, Polar2CartPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Cart2PolarPerfTestCPU, Cart2PolarPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CmpPerfTestCPU, CmpPerfTest,
    Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CmpWithScalarPerfTestCPU, CmpWithScalarPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(BitwisePerfTestCPU, BitwisePerfTest,
    Combine(Values(AND, OR, XOR),
            testing::Bool(),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(BitwiseNotPerfTestCPU, BitwiseNotPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SelectPerfTestCPU, SelectPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MinPerfTestCPU, MinPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MaxPerfTestCPU, MaxPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffPerfTestCPU, AbsDiffPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffCPerfTestCPU, AbsDiffCPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SumPerfTestCPU, SumPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        //Values(0.0),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CountNonZeroPerfTestCPU, CountNonZeroPerfTest,
                        Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
                                Values(szSmall128, szVGA, sz720p, sz1080p),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AddWeightedPerfTestCPU, AddWeightedPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(-1, CV_8U, CV_16U, CV_32F),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(NormPerfTestCPU, NormPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
        Values(NORM_INF, NORM_L1, NORM_L2),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(IntegralPerfTestCPU, IntegralPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestCPU, ThresholdPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestCPU, ThresholdOTPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1),
        Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(InRangePerfTestCPU, InRangePerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Split3PerfTestCPU, Split3PerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Split4PerfTestCPU, Split4PerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Merge3PerfTestCPU, Merge3PerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Merge4PerfTestCPU, Merge4PerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(RemapPerfTestCPU, RemapPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(FlipPerfTestCPU, FlipPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(0, 1, -1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CropPerfTestCPU, CropPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50)),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CopyPerfTestCPU, CopyPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorPerfTestCPU, ConcatHorPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorVecPerfTestCPU, ConcatHorVecPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertPerfTestCPU, ConcatVertPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertVecPerfTestCPU, ConcatVertVecPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestCPU, LUTPerfTest,
    Combine(Values(CV_8UC1, CV_8UC3),
        Values(CV_8UC1),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestCustomCPU, LUTPerfTest,
    Combine(Values(CV_8UC3),
        Values(CV_8UC3),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::compile_args(CORE_CPU))));


INSTANTIATE_TEST_CASE_P(ConvertToPerfTestCPU, ConvertToPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(CV_8U, CV_16U, CV_16S, CV_32F),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(2.5, 1.0),
            Values(0.0),
            Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(KMeansNDPerfTestCPU, KMeansNDPerfTest,
                        Combine(Values(cv::Size(1, 20),
                                       cv::Size(16, 4096)),
                                Values(AbsTolerance(0.01).to_compare_obj()),
                                Values(5, 15),
                                Values(cv::KMEANS_RANDOM_CENTERS,
                                       cv::KMEANS_PP_CENTERS,
                                       cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_USE_INITIAL_LABELS,
                                       cv::KMEANS_PP_CENTERS     | cv::KMEANS_USE_INITIAL_LABELS),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(KMeans2DPerfTestCPU, KMeans2DPerfTest,
                        Combine(Values(20, 4096),
                                Values(5, 15),
                                Values(cv::KMEANS_RANDOM_CENTERS,
                                       cv::KMEANS_PP_CENTERS,
                                       cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_USE_INITIAL_LABELS,
                                       cv::KMEANS_PP_CENTERS     | cv::KMEANS_USE_INITIAL_LABELS),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(KMeans3DPerfTestCPU, KMeans3DPerfTest,
                        Combine(Values(20, 4096),
                                Values(5, 15),
                                Values(cv::KMEANS_RANDOM_CENTERS,
                                       cv::KMEANS_PP_CENTERS,
                                       cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_USE_INITIAL_LABELS,
                                       cv::KMEANS_PP_CENTERS     | cv::KMEANS_USE_INITIAL_LABELS),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ResizePerfTestCPU, ResizePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_16UC1, CV_16SC1),
        Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(cv::Size(64, 64),
            cv::Size(30, 30)),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ResizeFxFyPerfTestCPU, ResizeFxFyPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
        Values(CV_8UC1, CV_16UC1, CV_16SC1),
        Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
        Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(0.5, 0.1),
        Values(0.5, 0.1),
        Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ParseSSDBLPerfTestCPU, ParseSSDBLPerfTest,
                        Combine(Values(sz720p, sz1080p),
                                Values(0.3f, 0.7f),
                                Values(0, 1),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ParseSSDPerfTestCPU, ParseSSDPerfTest,
                        Combine(Values(sz720p, sz1080p),
                                Values(0.3f, 0.7f),
                                testing::Bool(),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ParseYoloPerfTestCPU, ParseYoloPerfTest,
                        Combine(Values(sz720p, sz1080p),
                                Values(0.3f, 0.7f),
                                Values(0.5),
                                Values(7, 80),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SizePerfTestCPU, SizePerfTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_32FC1),
                                Values(szSmall128, szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SizeRPerfTestCPU, SizeRPerfTest,
                        Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
                                Values(cv::compile_args(CORE_CPU))));
} // opencv_test
