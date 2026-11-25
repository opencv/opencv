// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "fast_gemm.hpp"

#include "fast_gemm_thin.simd.hpp"
#include "layers/cpu_kernels/fast_gemm_thin.simd_declarations.hpp"

namespace cv { namespace dnn {

void fastGemmThinTB(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                    int M, int N, int K, float alpha, const float *A, int lda,
                    const float *B, int ldb, float beta, float *C, int ldc, bool multi_thread) {
    CV_CPU_DISPATCH(fastGemmThinTBKernel, (batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda,
                                           B, ldb, beta, C, ldc, multi_thread),
                    CV_CPU_DISPATCH_MODES_ALL);
}

}}  // namespace cv::dnn
