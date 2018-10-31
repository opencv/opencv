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

class AbsExactGPU : public Wrappable<AbsExactGPU>
{
public:
    AbsExactGPU() {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const { return cv::countNonZero(in1 != in2) == 0; }
private:
};

class AbsToleranceGPU : public Wrappable<AbsToleranceGPU>
{
public:
    AbsToleranceGPU(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        cv::Mat absDiff; cv::absdiff(in1, in2, absDiff);
        return cv::countNonZero(absDiff > _tol) == 0;
    }
private:
    double _tol;
};

class AbsTolerance32FGPU : public Wrappable<AbsTolerance32FGPU>
{
public:
    AbsTolerance32FGPU(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (CV_MAT_DEPTH(in1.type()) == CV_32F)
            return ((cv::countNonZero(cv::abs(in1 - in2) > (_tol)*cv::abs(in2))) ? false : true);
        else
            return ((cv::countNonZero(in1 != in2) <= (_tol8u)* in2.total()) ? true : false);
    }
private:
    double _tol;
    double _tol8u;
};


INSTANTIATE_TEST_CASE_P(MathOperatorTestGPU, MathOperatorMatMatTest,
                    Combine(Values(AbsTolerance32FGPU(1e-5, 1e-3).to_compare_f()),
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
                        Combine(Values(AbsTolerance32FGPU(1e-4, 1e-2).to_compare_f()),
                                Values( opPlus, opPlusR, opMinus, opMinusR, opMul, opMulR, opDiv, opDivR,
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
                        Combine(Values(AbsExactGPU().to_compare_f()),
                                Values( opAnd, opOr, opXor ),
                                Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                Values(cv::Size(1280, 720),
                                   cv::Size(640, 480),
                                   cv::Size(128, 128)),
                                Values(-1),
/*init output matrices or not*/ testing::Bool(),
                                Values(cv::compile_args(CORE_GPU))));

INSTANTIATE_TEST_CASE_P(BitwiseOperatorTestGPU, MathOperatorMatScalarTest,
                        Combine(Values(AbsExactGPU().to_compare_f()),
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
