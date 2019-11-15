// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_operators_tests.hpp"

namespace
{
#define CORE_GPU [] () { return cv::compile_args(cv::gapi::core::gpu::kernels()); }
}  // anonymous namespace

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(MathOperatorTestGPU, MathOperatorMatMatTest,
                    Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                            Values(cv::Size(1280, 720),
                                   cv::Size(640, 480),
                                   cv::Size(128, 128)),
                            Values(-1, CV_8U, CV_32F),
                            Values(CORE_GPU),
                            Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_obj()),
                            Values( opPlusM, opMinusM, opDivM,
                                    opGreater, opLess, opGreaterEq, opLessEq, opEq, opNotEq)));

INSTANTIATE_TEST_CASE_P(MathOperatorTestGPU, MathOperatorMatScalarTest,
                        Combine(Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
                                Values(CORE_GPU),
                                Values(Tolerance_FloatRel_IntAbs(1e-4, 2).to_compare_obj()),
                                Values( opPlus, opPlusR, opMinus, opMinusR, opMul, opMulR,  // FIXIT avoid division by values near zero: opDiv, opDivR,
                                        opGT, opLT, opGE, opLE, opEQ, opNE,
                                        opGTR, opLTR, opGER, opLER, opEQR, opNER)));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestGPU, MathOperatorMatMatTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AbsExact().to_compare_obj()),
                                Values( opAnd, opOr, opXor )));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestGPU, MathOperatorMatScalarTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AbsExact().to_compare_obj()),
                                Values( opAND, opOR, opXOR, opANDR, opORR, opXORR )));

INSTANTIATE_TEST_CASE_P(BitwiseNotOperatorTestGPU, NotOperatorTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
                                Values(CORE_GPU)));
}
