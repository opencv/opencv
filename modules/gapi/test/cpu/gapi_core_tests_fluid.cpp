// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_core_tests.hpp"
#include "backends/fluid/gfluidcore.hpp"

namespace opencv_test
{

#define CORE_FLUID cv::gapi::core::fluid::kernels()


// FIXME: Windows accuracy problems after recent update!
INSTANTIATE_TEST_CASE_P(MathOpTestFluid, MathOpTest,
                        Combine(Values(ADD, SUB, DIV, MUL),
                                testing::Bool(),
                                Values(CV_8UC3, CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(1.0),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
                                testing::Bool(),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(MulSTestFluid, MulDoubleTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1), // FIXME: extend with more types
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(DivCTestFluid, DivCTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8U, CV_32F),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AbsDiffTestFluid, AbsDiffTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(AbsDiffCTestFluid, AbsDiffCTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(BitwiseTestFluid, BitwiseTest,
                        Combine(Values(AND, OR, XOR),
                                Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))),
                        opencv_test::PrintBWCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseNotTestFluid, NotTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MinTestFluid, MinTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(MaxTestFluid, MaxTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(CompareTestFluid, CmpTest,
                        Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                testing::Bool(),
                                Values(CV_8UC3, CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))),
                        opencv_test::PrintCmpCoreParams());

INSTANTIATE_TEST_CASE_P(AddWeightedTestFluid, AddWeightedTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
                                testing::Bool(),
                                Values(0.5000005),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(LUTTestFluid, LUTTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(CV_8UC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ConvertToFluid, ConvertToTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_32FC1),
                                Values(CV_8U, CV_16U, CV_32F),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Split3TestFluid, Split3Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Split4TestFluid, Split4Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Merge3TestFluid, Merge3Test,
                        Combine(Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Merge4TestFluid, Merge4Test,
                        Combine(Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(SelectTestFluid, SelectTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Polar2CartFluid, Polar2CartTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(Cart2PolarFluid, Cart2PolarTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(ThresholdTestFluid, ThresholdTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV,
                                       cv::THRESH_TRUNC,
                                       cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(InRangeTestFluid, InRangeTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1920, 1080),
                                       cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(
                        ResizeTestFluid, ResizeTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values(CV_8UC3/*CV_8UC1, CV_16UC1, CV_16SC1*/),
                                Values(/*cv::INTER_NEAREST,*/ cv::INTER_LINEAR/*, cv::INTER_AREA*/),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128),
                                       cv::Size(64, 64),
                                       cv::Size(30, 30)),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128),
                                       cv::Size(64, 64),
                                       cv::Size(30, 30)),
                                Values(cv::compile_args(CORE_FLUID))));

//----------------------------------------------------------------------
// FIXME: Clean-up test configurations which are enabled already
#if 0
INSTANTIATE_TEST_CASE_P(MathOpTestCPU, MathOpTest,
                        Combine(Values(ADD, DIV, MUL),
                                testing::Bool(),
                                Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(false)),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(SubTestCPU, MathOpTest,
                        Combine(Values(SUB),
                                testing::Bool(),
                                Values(CV_8UC1, CV_16SC1 , CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                testing::Bool()),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(MulSTestCPU, MulSTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(DivCTestCPU, DivCTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(MeanTestCPU, MeanTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(SelectTestCPU, SelectTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(Polar2CartCPU, Polar2CartTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(Cart2PolarCPU, Cart2PolarTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(CompareTestCPU, CmpTest,
                        Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                testing::Bool(),
                                Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()),
                        opencv_test::PrintCmpCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseTestCPU, BitwiseTest,
                        Combine(Values(AND, OR, XOR),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()),
                        opencv_test::PrintBWCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseNotTestCPU, NotTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
 /*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(MinTestCPU, MinTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(MaxTestCPU, MaxTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(SumTestCPU, SumTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool())
                                Values(0.0),
                       );

INSTANTIATE_TEST_CASE_P(AbsDiffTestCPU, AbsDiffTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(AbsDiffCTestCPU, AbsDiffCTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(AddWeightedTestCPU, AddWeightedTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(NormTestCPU, NormTest,
                        Combine(Values(NORM_INF, NORM_L1, NORM_L2),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))),
                                Values(0.0),
                        opencv_test::PrintNormCoreParams());

INSTANTIATE_TEST_CASE_P(IntegralTestCPU, IntegralTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))));

INSTANTIATE_TEST_CASE_P(ThresholdTestCPU, ThresholdTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(ThresholdTestCPU, ThresholdOTTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE),
/*init output matrices or not*/ testing::Bool()));


INSTANTIATE_TEST_CASE_P(InRangeTestCPU, InRangeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(Split3TestCPU, Split3Test,
                        (Values(cv::Size(1280, 720),
                                cv::Size(640, 480),
                                cv::Size(128, 128))));

INSTANTIATE_TEST_CASE_P(Split4TestCPU, Split4Test,
                        (Values(cv::Size(1280, 720),
                                cv::Size(640, 480),
                                cv::Size(128, 128))));

INSTANTIATE_TEST_CASE_P(Merge3TestCPU, Merge3Test,
                        (Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))));

INSTANTIATE_TEST_CASE_P(Merge4TestCPU, Merge4Test,
                        (Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))));

INSTANTIATE_TEST_CASE_P(RemapTestCPU, RemapTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(FlipTestCPU, FlipTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(0,1,-1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(CropTestCPU, CropTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50)),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(LUTTestCPU, LUTTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(LUTTestCustomCPU, LUTTest,
                        Combine(Values(CV_8UC3),
                                Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool()));

INSTANTIATE_TEST_CASE_P(ConvertToCPU, ConvertToTest,
                        Combine(Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_32FC1),
                                Values(CV_8U, CV_16U, CV_32F),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))));

INSTANTIATE_TEST_CASE_P(ConcatHorTestCPU, ConcatHorTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))));
INSTANTIATE_TEST_CASE_P(ConcatVertTestCPU, ConcatVertTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128))));

//----------------------------------------------------------------------
#endif // 0

}
