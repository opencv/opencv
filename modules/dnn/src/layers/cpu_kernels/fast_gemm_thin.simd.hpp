// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utility.hpp" // parallel_for_
#include <algorithm>

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void fastGemmThinTBKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                          int M, int N, int K, float alpha, const float *A, int lda,
                          const float *B, int ldb, float beta, float *C, int ldc, bool multi_thread);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // namespace cv::dnn

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

static constexpr int FAST_GEMM_THIN_TB_MIN_TASK_MACS = 1 << 15;

static inline void fast_gemm_thin_tb_store(float* c, float d, float alpha, float beta) {
    *c = beta == 0.f ? alpha * d : alpha * d + beta * *c;
}

static void fast_gemm_thin_tb(float alpha, float beta, int M, int n0, int n1, int K,
                              const float* A, int lda, const float* B, int ldb,
                              float* C, int ldc) {
    for (int m = 0; m < M; m++) {
        const float* a = A + (size_t)m * lda;
        float* c = C + (size_t)m * ldc;
        int n = n0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VL = VTraits<v_float32>::vlanes();
        for (; n + 4 <= n1; n += 4) {
            const float* b0 = B + (size_t)n * ldb;
            const float* b1 = b0 + ldb;
            const float* b2 = b1 + ldb;
            const float* b3 = b2 + ldb;
            v_float32 s0 = vx_setzero_f32(), s1 = vx_setzero_f32();
            v_float32 s2 = vx_setzero_f32(), s3 = vx_setzero_f32();
            int k = 0;
            for (; k + VL <= K; k += VL) {
                v_float32 av = vx_load(a + k);
                s0 = v_fma(av, vx_load(b0 + k), s0);
                s1 = v_fma(av, vx_load(b1 + k), s1);
                s2 = v_fma(av, vx_load(b2 + k), s2);
                s3 = v_fma(av, vx_load(b3 + k), s3);
            }
            float d0 = v_reduce_sum(s0), d1 = v_reduce_sum(s1);
            float d2 = v_reduce_sum(s2), d3 = v_reduce_sum(s3);
            for (; k < K; k++) {
                d0 += a[k] * b0[k];
                d1 += a[k] * b1[k];
                d2 += a[k] * b2[k];
                d3 += a[k] * b3[k];
            }
            fast_gemm_thin_tb_store(c + n, d0, alpha, beta);
            fast_gemm_thin_tb_store(c + n + 1, d1, alpha, beta);
            fast_gemm_thin_tb_store(c + n + 2, d2, alpha, beta);
            fast_gemm_thin_tb_store(c + n + 3, d3, alpha, beta);
        }
#endif
        for (; n < n1; n++) {
            const float* b = B + (size_t)n * ldb;
            float d = 0.f;
            for (int k = 0; k < K; k++)
                d += a[k] * b[k];
            fast_gemm_thin_tb_store(c + n, d, alpha, beta);
        }
    }
}

void fastGemmThinTBKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                          int M, int N, int K, float alpha, const float *A, int lda,
                          const float *B, int ldb, float beta, float *C, int ldc, bool multi_thread) {
    const int nblk = std::max(4, (FAST_GEMM_THIN_TB_MIN_TASK_MACS / std::max(M * K, 1) + 3) / 4 * 4);
    const int n_tasks = (N + nblk - 1) / nblk;
    const int total = (int)batch * n_tasks;
    auto fn = [&](const Range& r) {
        for (int t = r.start; t < r.end; t++) {
            const size_t i = (size_t)(t / n_tasks);
            const int n0 = (t % n_tasks) * nblk;
            fast_gemm_thin_tb(alpha, beta, M, n0, std::min(N, n0 + nblk), K,
                              A + A_offsets[i], lda, B + B_offsets[i], ldb,
                              C + C_offsets[i], ldc);
        }
    };
    if (multi_thread && total > 1)
        parallel_for_(Range(0, total), fn);
    else
        fn(Range(0, total));
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // namespace cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
