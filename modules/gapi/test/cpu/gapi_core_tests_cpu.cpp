// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_core_tests.hpp"
#include "opencv2/gapi/cpu/core.hpp"

#define CORE_CPU cv::gapi::core::cpu::kernels()

namespace opencv_test
{


// FIXME: Wut? See MulTestCPU/MathOpTest below (duplicate?)
INSTANTIATE_TEST_CASE_P(AddTestCPU, MathOpTest,
                        Combine(Values(ADD, MUL),
                                testing::Bool(),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(1.0),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(false),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(MulTestCPU, MathOpTest,
                        Combine(Values(MUL),
                                testing::Bool(),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(1.0, 0.5, 2.0),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(false),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(SubTestCPU, MathOpTest,
                        Combine(Values(SUB),
                                testing::Bool(),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values (1.0),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(DivTestCPU, MathOpTest,
                        Combine(Values(DIV),
                                testing::Bool(),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values (1.0, 0.5, 2.0),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(MulTestCPU, MulDoubleTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivTestCPU, DivTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivCTestCPU, DivCTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MeanTestCPU, MeanTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MaskTestCPU, MaskTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SelectTestCPU, SelectTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Polar2CartCPU, Polar2CartTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Cart2PolarCPU, Cart2PolarTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CompareTestCPU, CmpTest,
                        Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                testing::Bool(),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintCmpCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseTestCPU, BitwiseTest,
                        Combine(Values(AND, OR, XOR),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintBWCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseNotTestCPU, NotTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
 /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MinTestCPU, MinTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MaxTestCPU, MaxTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SumTestCPU, SumTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(1e-5),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffTestCPU, AbsDiffTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffCTestCPU, AbsDiffCTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

// FIXME: Comparison introduced by YL doesn't work with C3
INSTANTIATE_TEST_CASE_P(AddWeightedTestCPU, AddWeightedTest,
                        Combine(Values( CV_8UC1/*, CV_8UC3*/, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
/*init output matrices or not*/ testing::Bool(),
                                Values(0.5000005),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(NormTestCPU, NormTest,
                        Combine(Values(NORM_INF, NORM_L1, NORM_L2),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(1e-5),
                                Values(cv::compile_args(CORE_CPU))),
                        opencv_test::PrintNormCoreParams());

INSTANTIATE_TEST_CASE_P(IntegralTestCPU, IntegralTest,
                        Combine(Values( CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ThresholdTestCPU, ThresholdTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ThresholdTestCPU, ThresholdOTTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));


INSTANTIATE_TEST_CASE_P(InRangeTestCPU, InRangeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Split3TestCPU, Split3Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Split4TestCPU, Split4Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, ResizeTest,
                        Combine(Values(AbsSimilarPoints(2, 0.05).to_compare_f()),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::Size(64,64),
                                       cv::Size(30,30)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ResizeTestCPU, ResizeTestFxFy,
                        Combine(Values(AbsSimilarPoints(2, 0.05).to_compare_f()),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(0.5, 0.1),
                                Values(0.5, 0.1),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Merge3TestCPU, Merge3Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Merge4TestCPU, Merge4Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(RemapTestCPU, RemapTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(FlipTestCPU, FlipTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(0,1,-1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CropTestCPU, CropTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50)),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(LUTTestCPU, LUTTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(LUTTestCustomCPU, LUTTest,
                        Combine(Values(CV_8UC3),
                                Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConvertToCPU, ConvertToTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(CV_8U, CV_16U, CV_16S, CV_32F),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorTestCPU, ConcatHorTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertTestCPU, ConcatVertTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertVecTestCPU, ConcatVertVecTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorVecTestCPU, ConcatHorVecTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_CPU))));
}
