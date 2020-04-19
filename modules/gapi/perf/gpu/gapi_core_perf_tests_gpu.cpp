// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_core_perf_tests.hpp"

#define CORE_GPU cv::gapi::core::gpu::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(AddPerfTestGPU, AddPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(AddCPerfTestGPU, AddCPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(SubPerfTestGPU, SubPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(SubCPerfTestGPU, SubCPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(SubRCPerfTestGPU, SubRCPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MulPerfTestGPU, MulPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MulDoublePerfTestGPU, MulDoublePerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MulCPerfTestGPU, MulCPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(DivPerfTestGPU, DivPerfTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_f()),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(DivCPerfTestGPU, DivCPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(DivRCPerfTestGPU, DivRCPerfTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_f()),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));
//TODO: mask test doesn't work
INSTANTIATE_TEST_CASE_P(DISABLED_MaskPerfTestGPU, MaskPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MeanPerfTestGPU, MeanPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Polar2CartPerfTestGPU, Polar2CartPerfTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_f()),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Cart2PolarPerfTestGPU, Cart2PolarPerfTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-2, 2).to_compare_f()),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(CmpPerfTestGPU, CmpPerfTest,
                        Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(CmpWithScalarPerfTestGPU, CmpWithScalarPerfTest,
                        Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(BitwisePerfTestGPU, BitwisePerfTest,
                        Combine(Values(AND, OR, XOR),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(BitwiseNotPerfTestGPU, BitwiseNotPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(SelectPerfTestGPU, SelectPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MinPerfTestGPU, MinPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MaxPerfTestGPU, MaxPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffPerfTestGPU, AbsDiffPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffCPerfTestGPU, AbsDiffCPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(SumPerfTestGPU, SumPerfTest,
                        Combine(Values(AbsToleranceScalar(1e-5).to_compare_f()),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(AddWeightedPerfTestGPU, AddWeightedPerfTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(NormPerfTestGPU, NormPerfTest,
                        Combine(Values(AbsToleranceScalar(1e-5).to_compare_f()),
                                Values(NORM_INF, NORM_L1, NORM_L2),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(IntegralPerfTestGPU, IntegralPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestGPU, ThresholdPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestGPU, ThresholdOTPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1 ),
                                Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(InRangePerfTestGPU, InRangePerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Split3PerfTestGPU, Split3PerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Split4PerfTestGPU, Split4PerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Merge3PerfTestGPU, Merge3PerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Merge4PerfTestGPU, Merge4PerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(RemapPerfTestGPU, RemapPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(FlipPerfTestGPU, FlipPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(0,1,-1),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(CropPerfTestGPU, CropPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(CopyPerfTestGPU, CopyPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorPerfTestGPU, ConcatHorPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertPerfTestGPU, ConcatVertPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

//TODO: fix this backend to allow ConcatVertVec ConcatHorVec
INSTANTIATE_TEST_CASE_P(DISABLED_ConcatHorVecPerfTestGPU, ConcatHorVecPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
        Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
        Values(cv::compile_args(CORE_GPU))));


INSTANTIATE_TEST_CASE_P(DISABLED_ConcatVertVecPerfTestGPU, ConcatVertVecPerfTest,
                        Combine(Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestGPU, LUTPerfTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(CV_8UC1),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestCustomGPU, LUTPerfTest,
                        Combine(Values(CV_8UC3),
                                Values(CV_8UC3),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));


INSTANTIATE_TEST_CASE_P(ConvertToPerfTestGPU, ConvertToPerfTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_32FC1),
                                Values(CV_8U, CV_16U, CV_16S, CV_32F),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ResizePerfTestGPU, ResizePerfTest,
                        Combine(Values(AbsSimilarPoints(2, 0.05).to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(cv::Size(64,64),
                                       cv::Size(30,30)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ResizeFxFyPerfTestGPU, ResizeFxFyPerfTest,
                        Combine(Values(AbsSimilarPoints(2, 0.05).to_compare_f()),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values( szSmall128, szVGA, sz720p, sz1080p ),
                                Values(0.5, 0.1),
                                Values(0.5, 0.1),
                                Values(cv::compile_args(CORE_GPU))));
}
