// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_core_tests.hpp"

namespace
{
#define CORE_FLUID [] () { return cv::compile_args(cv::gapi::use_only{cv::gapi::core::fluid::kernels()}); }
}  // anonymous namespace

namespace opencv_test
{
// FIXME: Windows accuracy problems after recent update!
INSTANTIATE_TEST_CASE_P(MathOpTestFluid, MathOpTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
                                Values(CORE_FLUID),
                                Values(ADD, SUB, DIV, MUL),
                                testing::Bool(),
                                Values(1.0),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(SubTestFluid, MathOpTest,
                        Combine(Values(CV_8UC1, CV_16SC1 , CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
                                Values(CORE_FLUID),
                                Values(SUB),
                                testing::Bool(),
                                Values (1.0),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(MulSTestFluid, MulDoubleTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1), // FIXME: extend with more types
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(DivCTestFluid, DivCTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8U, CV_32F),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(DISABLED_MeanTestFluid, MeanTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(AbsDiffTestFluid, AbsDiffTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(AbsDiffCTestFluid, AbsDiffCTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(BitwiseTestFluid, BitwiseTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(AND, OR, XOR)));

INSTANTIATE_TEST_CASE_P(BitwiseNotTestFluid, NotTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(MinTestFluid, MinTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(MaxTestFluid, MaxTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(DISABLED_SumTestFluid, SumTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(AbsToleranceScalar(1e-5).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(CompareTestFluid, CmpTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8U),
                                Values(CORE_FLUID),
                                Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(AddWeightedTestFluid, AddWeightedTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
                                Values(CORE_FLUID),
                                Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(DISABLED_NormTestFluid, NormTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(AbsToleranceScalar(1e-5).to_compare_obj()),
                                Values(NORM_INF, NORM_L1, NORM_L2)));

INSTANTIATE_TEST_CASE_P(DISABLED_IntegralTestFluid, IntegralTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(LUTTestFluid, LUTTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8UC1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(ConvertToFluid, ConvertToTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8U, CV_16U, CV_32F),
                                Values(CORE_FLUID),
                                Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_obj()),
                                Values(2.5, 1.0, -1.0),
                                Values(250.0, 0.0, -128.0)));

INSTANTIATE_TEST_CASE_P(DISABLED_ConcatHorTestFluid, ConcatHorTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(DISABLED_ConcatVertTestFluid, ConcatVertTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(Split3TestFluid, Split3Test,
                        Combine(Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8UC1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(Split4TestFluid, Split4Test,
                        Combine(Values(CV_8UC4),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8UC1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(Merge3TestFluid, Merge3Test,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8UC3),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(Merge4TestFluid, Merge4Test,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8UC4),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(DISABLED_RemapTestFluid, RemapTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(DISABLED_FlipTestFluid, FlipTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(0,1,-1)));

INSTANTIATE_TEST_CASE_P(DISABLED_CropTestFluid, CropTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50))));

INSTANTIATE_TEST_CASE_P(SelectTestFluid, SelectTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(Polar2CartFluid, Polar2CartTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_32FC1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(Cart2PolarFluid, Cart2PolarTest,
                        Combine(Values(CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_32FC1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(PhaseFluid, PhaseTest,
                        Combine(Values(CV_32F, CV_32FC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
         /* angle_in_degrees */ testing::Bool()));

INSTANTIATE_TEST_CASE_P(SqrtFluid, SqrtTest,
                        Combine(Values(CV_32F, CV_32FC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(ThresholdTestFluid, ThresholdTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV,
                                       cv::THRESH_TRUNC,
                                       cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
                                Values(cv::Scalar(0, 0, 0, 0),
                                       cv::Scalar(100, 100, 100, 100),
                                       cv::Scalar(255, 255, 255, 255))));

INSTANTIATE_TEST_CASE_P(DISABLED_ThresholdTestFluid, ThresholdOTTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE)));

INSTANTIATE_TEST_CASE_P(InRangeTestFluid, InRangeTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(ResizeTestFluid, ResizeTest,
                        Combine(Values(CV_8UC3/*CV_8UC1, CV_16UC1, CV_16SC1*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128),
                                       cv::Size(64, 64),
                                       cv::Size(30, 30)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(/*cv::INTER_NEAREST,*/ cv::INTER_LINEAR/*, cv::INTER_AREA*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128),
                                       cv::Size(64, 64),
                                       cv::Size(30, 30))));

INSTANTIATE_TEST_CASE_P(ResizeTestFxFyFluid, ResizeTestFxFy,
                        Combine(Values(CV_8UC3/*CV_8UC1, CV_16UC1, CV_16SC1*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128),
                                       cv::Size(64, 64),
                                       cv::Size(30, 30)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(AbsExact().to_compare_obj()),
                                Values(/*cv::INTER_NEAREST,*/ cv::INTER_LINEAR/*, cv::INTER_AREA*/),
                                Values(0.5, 1, 2),
                                Values(0.5, 1, 2)));

INSTANTIATE_TEST_CASE_P(BackendOutputAllocationTestFluid, BackendOutputAllocationTest,
                        Combine(Values(CV_8UC3, CV_16SC2, CV_32FC1),
                                Values(cv::Size(50, 50)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(BackendOutputAllocationLargeSizeWithCorrectSubmatrixTestFluid,
                        BackendOutputAllocationLargeSizeWithCorrectSubmatrixTest,
                        Combine(Values(CV_8UC3, CV_16SC2, CV_32FC1),
                                Values(cv::Size(50, 50)),
                                Values(-1),
                                Values(CORE_FLUID)));

INSTANTIATE_TEST_CASE_P(ReInitOutTestFluid, ReInitOutTest,
                        Combine(Values(CV_8UC3, CV_16SC4, CV_32FC1),
                                Values(cv::Size(640, 480)),
                                Values(-1),
                                Values(CORE_FLUID),
                                Values(cv::Size(640, 400),
                                       cv::Size(10, 480))));

INSTANTIATE_TEST_CASE_P(CopyTestFluid, CopyTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_FLUID)));
}
