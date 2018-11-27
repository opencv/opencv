// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_operators_tests.hpp"
#include "opencv2/gapi/gpu/core.hpp"

#define CORE_GPU cv::gapi::core::gpu::kernels()

namespace opencv_test
{


INSTANTIATE_TEST_CASE_P(MathOperatorTestGPU, MathOperatorMatMatTest,
                    Combine(Values(Tolerance_FloatRel_IntAbs(1e-5, 2).to_compare_f()),
                            Values( opPlusM, opMinusM, opDivM,
                                    opGreater, opLess, opGreaterEq, opLessEq, opEq, opNotEq),
                            Values(CV_8UC1, CV_16SC1, CV_32FC1),
                            Values(cv::Size(1280, 720),
                               cv::Size(640, 480),
                               cv::Size(128, 128)),
                            Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                            Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(MathOperatorTestGPU, MathOperatorMatScalarTest,
                        Combine(Values(Tolerance_FloatRel_IntAbs(1e-4, 2).to_compare_f()),
                                Values( opPlus, opPlusR, opMinus, opMinusR, opMul, opMulR,  // FIXIT avoid division by values near zero: opDiv, opDivR,
                                        opGT, opLT, opGE, opLE, opEQ, opNE,
                                        opGTR, opLTR, opGER, opLER, opEQR, opNER),
                                Values(CV_8UC1, CV_16SC1, CV_32FC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1, CV_8U, CV_32F),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestGPU, MathOperatorMatMatTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values( opAnd, opOr, opXor ),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                   cv::Size(640, 480),
                                   cv::Size(128, 128)),
                                Values(-1),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestGPU, MathOperatorMatScalarTest,
                        Combine(Values(AbsExact().to_compare_f()),
                                Values( opAND, opOR, opXOR, opANDR, opORR, opXORR ),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
                                Values(-1),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(BitwiseNotOperatorTestGPU, NotOperatorTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                       cv::Size(640, 480),
                                       cv::Size(128, 128)),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));
}
