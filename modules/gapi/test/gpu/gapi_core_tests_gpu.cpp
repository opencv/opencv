// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_core_tests.hpp"
#include "opencv2/gapi/gpu/core.hpp"

#define CORE_GPU cv::gapi::core::gpu::kernels()

namespace opencv_test
{

// FIXME: Wut? See MulTestGPU/MathOpTest below (duplicate?)
INSTANTIATE_TEST_CASE_P(AddTestGPU, MathOpTest,
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
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(MulTestGPU, MathOpTest,
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
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(SubTestGPU, MathOpTest,
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
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(DivTestGPU, MathOpTest,
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
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintMathOpCoreParams());

INSTANTIATE_TEST_CASE_P(MulTestGPU, MulDoubleTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(DivTestGPU, DivTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(DivCTestGPU, DivCTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MeanTestGPU, MeanTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
    /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

//TODO: mask test doesn't work
#if 0
INSTANTIATE_TEST_CASE_P(MaskTestGPU, MaskTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));
#endif

INSTANTIATE_TEST_CASE_P(SelectTestGPU, SelectTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Polar2CartGPU, Polar2CartTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Cart2PolarGPU, Cart2PolarTest,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(CompareTestGPU, CmpTest,
                        Combine(Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                testing::Bool(),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintCmpCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseTestGPU, BitwiseTest,
                        Combine(Values(AND, OR, XOR),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintBWCoreParams());

INSTANTIATE_TEST_CASE_P(BitwiseNotTestGPU, NotTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
 /*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MinTestGPU, MinTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MaxTestGPU, MaxTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(SumTestGPU, SumTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(1e-3), //TODO: too relaxed?
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffTestGPU, AbsDiffTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffCTestGPU, AbsDiffCTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

// FIXME: Comparison introduced by YL doesn't work with C3
INSTANTIATE_TEST_CASE_P(AddWeightedTestGPU, AddWeightedTest,
                        Combine(Values( CV_8UC1/*, CV_8UC3*/, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
/*init output matrices or not*/ testing::Bool(),
                                Values(0.50005),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(NormTestGPU, NormTest,
                        Combine(Values(NORM_INF, NORM_L1, NORM_L2),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(1e-3), //TODO: too relaxed?
                                Values(cv::compile_args(CORE_GPU))),
                        opencv_test::PrintNormCoreParams());

INSTANTIATE_TEST_CASE_P(IntegralTestGPU, IntegralTest,
                        Combine(Values( CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ThresholdTestGPU, ThresholdTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ThresholdTestGPU, ThresholdOTTest,
                        Combine(Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::THRESH_OTSU, cv::THRESH_TRIANGLE),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));


INSTANTIATE_TEST_CASE_P(InRangeTestGPU, InRangeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Split3TestGPU, Split3Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Split4TestGPU, Split4Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ResizeTestGPU, ResizeTest,
                        Combine(Values(AbsSimilarPoints(2, 0.05).to_compare_f()),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::Size(64,64),
                                       cv::Size(30,30)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ResizeTestGPU, ResizeTestFxFy,
                        Combine(Values(AbsSimilarPoints(2, 0.05).to_compare_f()),
                                Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(0.5, 0.1),
                                Values(0.5, 0.1),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Merge3TestGPU, Merge3Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(Merge4TestGPU, Merge4Test,
                        Combine(Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(RemapTestGPU, RemapTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(FlipTestGPU, FlipTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(0,1,-1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(CropTestGPU, CropTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Rect(10, 8, 20, 35), cv::Rect(4, 10, 37, 50)),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(LUTTestGPU, LUTTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                Values(CV_8UC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(LUTTestCustomGPU, LUTTest,
                        Combine(Values(CV_8UC3),
                                Values(CV_8UC3),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ConvertToGPU, ConvertToTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(CV_8U, CV_16U, CV_16S, CV_32F),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorTestGPU, ConcatHorTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertTestGPU, ConcatVertTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

//TODO: fix this backend to allow ConcatVertVec ConcatHorVec
#if 0
INSTANTIATE_TEST_CASE_P(ConcatVertVecTestGPU, ConcatVertVecTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorVecTestGPU, ConcatHorVecTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(cv::compile_args(CORE_GPU))));
#endif
}
