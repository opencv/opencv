// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "../common/gapi_operators_tests.hpp"
#include "opencv2/gapi/cpu/core.hpp"

#define CORE_FLUID cv::gapi::core::cpu::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(MathOperatorTestFluid, MathOperatorMatMatTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values( opPlusM, opMinusM, opDivM,
                                        opGreater, opLess, opGreaterEq, opLessEq, opEq, opNotEq),
                                Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                   cv::Size(640, 480),
                                   cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

//FIXME: Some Mat/Scalar Fluid kernels are not there yet!
INSTANTIATE_TEST_CASE_P(DISABLED_MathOperatorTestFluid, MathOperatorMatScalarTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values( opPlus, opPlusR, opMinus, opMinusR, opMul, opMulR,  // FIXIT avoid division by values near zero: opDiv, opDivR,
                                        opGT, opLT, opGE, opLE, opEQ, opNE,
                                        opGTR, opLTR, opGER, opLER, opEQR, opNER),
                                Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestFluid, MathOperatorMatMatTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values( opAnd, opOr, opXor ),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                   cv::Size(640, 480),
                                   cv::Size(128, 128)),
                                Values(-1),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

//FIXME: Some Mat/Scalar Fluid kernels are not there yet!
INSTANTIATE_TEST_CASE_P(DISABLED_BitwiseOperatorTestFluid, MathOperatorMatScalarTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values( opAND, opOR, opXOR, opANDR, opORR, opXORR ),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));

INSTANTIATE_TEST_CASE_P(BitwiseNotOperatorTestFluid, NotOperatorTest,
                    Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                            Values(cv::Size(1280, 720),
                                   cv::Size(640, 480),
                                   cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_FLUID))));
}
