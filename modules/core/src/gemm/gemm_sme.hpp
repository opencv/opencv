// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_GEMM_SME_HPP
#define OPENCV_CORE_SRC_GEMM_SME_HPP

#include <stddef.h>

namespace cv { namespace gemm_engine { namespace sme {

// Floats in one streaming vector (VL). Call only when the CPU has SME.
int vectorLength();

// C(m x n) = alpha*A*B + beta*C for one k block, run in streaming mode.
// A: panels of R rows (R = VL or 2*VL), each k x R, aPanelStride floats apart.
// B: panels of 2*VL columns, each k x 2*VL, bPanelStride floats apart.
void gemmPanels(int m, int n, int k, int R, const float* A, size_t aPanelStride,
                const float* B, size_t bPanelStride, float alpha, float beta, float* C, size_t ldc);

}}} // cv::gemm_engine::sme

#endif // OPENCV_CORE_SRC_GEMM_SME_HPP
