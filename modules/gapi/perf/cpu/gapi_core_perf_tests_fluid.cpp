// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_core_perf_tests.hpp"

#define CORE_FLUID cv::gapi::core::fluid::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(PhasePerfTestFluid, PhasePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_32FC1, CV_64FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SqrtPerfTestFluid, SqrtPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_32FC1, CV_64FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AddPerfTestFluid, AddPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AddCPerfTestFluid, AddCPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SubPerfTestFluid, SubPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 0).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SubCPerfTestFluid, SubCPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SubRCPerfTestFluid, SubRCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MulPerfTestFluid, MulPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(2.0),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MulDoublePerfTestFluid, MulDoublePerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MulCPerfTestFluid, MulCPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(DivPerfTestFluid, DivPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(2.3),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(DivCPerfTestFluid, DivCPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(1.0),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(DivRCPerfTestFluid, DivRCPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(1.0),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MaskPerfTestFluid, MaskPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_16UC1, CV_16SC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MeanPerfTestFluid, MeanPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Polar2CartPerfTestFluid, Polar2CartPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Cart2PolarPerfTestFluid, Cart2PolarPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-04, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(CmpPerfTestFluid, CmpPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(CmpWithScalarPerfTestFluid, CmpWithScalarPerfTest,
    Combine(Values(AbsSimilarPoints(1, 0.01).to_compare_f()),
            Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(BitwisePerfTestFluid, BitwisePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(AND, OR, XOR),
            testing::Bool(),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(BitwiseNotPerfTestFluid, BitwiseNotPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SelectPerfTestFluid, SelectPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MinPerfTestFluid, MinPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MaxPerfTestFluid, MaxPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AbsDiffPerfTestFluid, AbsDiffPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AbsDiffCPerfTestFluid, AbsDiffCPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SumPerfTestFluid, SumPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AddWeightedPerfTestFluid, AddWeightedPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(-1, CV_8U, CV_32F),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AddWeightedPerfTestFluid_short, AddWeightedPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_16UC1, CV_16SC1),
            Values(-1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(NormPerfTestFluid, NormPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(NORM_INF, NORM_L1, NORM_L2),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(IntegralPerfTestFluid, IntegralPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestFluid, ThresholdPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC,
                   cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestFluid, ThresholdOTPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1),
            Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(InRangePerfTestFluid, InRangePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Split3PerfTestFluid, Split3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Split4PerfTestFluid, Split4PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Merge3PerfTestFluid, Merge3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Merge4PerfTestFluid, Merge4PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(RemapPerfTestFluid, RemapPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(FlipPerfTestFluid, FlipPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(0, 1, -1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(CropPerfTestFluid, CropPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50)),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ConcatHorPerfTestFluid, ConcatHorPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ConcatHorVecPerfTestFluid, ConcatHorVecPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ConcatVertPerfTestFluid, ConcatVertPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ConcatVertVecPerfTestFluid, ConcatVertVecPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestFluid, LUTPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3),
            Values(CV_8UC1),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(cv::compile_args(CORE_FLUID))));

// FIXIT: This test case doesn't work [3030]
// INSTANTIATE_TEST_CASE_P(LUTPerfTestCustomFluid, LUTPerfTest,
//     Combine(Values(AbsExact().to_compare_f()),
//             Values(CV_8UC3),
//             Values(CV_8UC3),
//             Values(szSmall128, szVGA, sz720p, sz1080p),
//             Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ConvertToPerfTestFluid, ConvertToPerfTest,
    Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()),
            Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(CV_8U, CV_16U, CV_16S, CV_32F),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(1.0, 2.5),
            Values(0.0),
            Values(cv::compile_args(CORE_FLUID))));

} // opencv_test
