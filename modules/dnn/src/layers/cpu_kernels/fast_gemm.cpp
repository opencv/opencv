// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "fast_gemm.hpp"

#include "fast_gemm_kernels.simd.hpp"
#include "layers/cpu_kernels/fast_gemm_kernels.simd_declarations.hpp"

namespace cv { namespace dnn {

void fastGemmPackB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckEQ(B.dims, 2, "fastGemmPackB: input mat should be two-dimensional");
    CV_CheckTypeEQ(B.type(), CV_32F, "fastGemmPackB: only float32 is supported for now");

    auto B_shape = shape(B);
    int K = B_shape[0], N = B_shape[1], ldb0 = N, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

#if CV_TRY_NEON && CV_NEON_AARCH64
    if (opt.use_neon_aarch64) {
        int size_packed_B = opt_NEON_AARCH64::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B);
        opt_NEON_AARCH64::fastGemmPackBKernel(B.ptr<const char>(), (char *)packed_B.data(), N, K, ldb0, ldb1, B.elemSize());
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        int size_packed_B = opt_AVX2::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B);
        opt_AVX2::fastGemmPackBKernel(B.ptr<const char>(), (char *)packed_B.data(), N, K, ldb0, ldb1, B.elemSize());
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        int size_packed_B = opt_AVX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B);
        opt_AVX::fastGemmPackBKernel(B.ptr<const char>(), (char *)packed_B.data(), N, K, ldb0, ldb1, B.elemSize());
    } else
#endif
    {
        int size_packed_B = cpu_baseline::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B);
        cpu_baseline::fastGemmPackBKernel(B.ptr<const char>(), (char *)packed_B.data(), N, K, ldb0, ldb1, B.elemSize());
    }
}

static void fast_gemm_thin(float alpha, float beta, int M, int N, int K,
                           const char *a_, int lda0, int lda1,
                           const char *b_, int ldb,
                           char *c_, int ldc) {
    const float* a = (const float*)a_;

    auto fn = [&](const Range &r) {
        for(int start = r.start ; start < r.end; start++ ) {
            float* c_i = (float*)c_ + start * ldc;
            if (beta == 0.f)
                for(int j = 0; j < N; j++ ) c_i[j] = 0.f;
            else if (beta != 1.f)
                for(int j = 0; j < N; j++ ) c_i[j] *= beta;
            for(int k = 0; k < K; k++ ) {
                const float* b_k = (const float*)b_ + k * ldb;
                float aval = alpha * a[start * lda0 + k * lda1];
                for(int j = 0; j < N; j++ )
                    c_i[j] += aval * b_k[j];
            }
        }
    };

    int total = M; // outer loops
    int cost_per_thread = static_cast<int>(K * N); // inner loops
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fastGemm(bool trans_a, int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *packed_B, float beta,
              float *C, int ldc, FastGemmOpt &opt) {
    int lda0 = lda, lda1 = 1;
    if (trans_a) {
        std::swap(lda0, lda1);
    }

#if CV_TRY_NEON && CV_NEON_AARCH64
    if (opt.use_neon_aarch64) {
        opt_NEON_AARCH64::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1, (const char *)packed_B, beta, (char *)C, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1, (const char *)packed_B, beta, (char *)C, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1, (const char *)packed_B, beta, (char *)C, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1, (const char *)packed_B, beta, (char *)C, ldc, sizeof(float));
    }
}

void fastGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
              float alpha, const float *A, int lda0, int lda1, const float *B, int ldb0, int ldb1,
              float beta, float *C, int ldc, FastGemmOpt &opt) {

    const char *a = (const char *)A;
    const char *b = (const char *)B;
    char *c = (char *)C;

    int M = trans_a ? na : ma;
    int N = trans_b ? mb : nb;
    int K = trans_a ? ma : na;

    if (trans_a) {
        std::swap(lda0, lda1);
    }
    if (trans_b) {
        std::swap(ldb0, ldb1);
    }

    if (!trans_b && ldb1 == 1 && (M <= 4 || (uint64_t)M * N * K <= 10000)) {
        return fast_gemm_thin(alpha, beta, M, N, K, a, lda0, lda1, b, ldb0, c, ldc);
    }

#if CV_TRY_NEON && CV_NEON_AARCH64
    if (opt.use_neon_aarch64) {
        opt_NEON_AARCH64::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1,
                                         (const char *)B, ldb0, ldb1, beta, (char *)C, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1,
                                         (const char *)B, ldb0, ldb1, beta, (char *)C, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1,
                                         (const char *)B, ldb0, ldb1, beta, (char *)C, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmKernel(M, N, K, alpha, (const char *)A, lda0, lda1,
                                         (const char *)B, ldb0, ldb1, beta, (char *)C, ldc, sizeof(float));
    }
}

void fastGemm(bool trans_a, bool trans_b,
               float alpha, const Mat &A, const Mat &B,
               float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/gemm: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/gemm: B and C should have the same type");
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/gemm: only support float32 for now");

    const auto shape_a = shape(A);
    const auto shape_b = shape(B);
    const auto shape_c = shape(C);

    int ma = shape_a[0], na = shape_a[1];
    int mb = shape_b[0], nb = shape_b[1];

    int lda0 = na, lda1 = 1, ldb0 = nb, ldb1 = 1, ldc = shape_c[1];

    // printf("trans_a=%d, trans_b=%d, ma=%d, na=%d, mb=%d, nb=%d, alpha=%f, beta=%f, lda0=%d, lda1=%d, ldb0=%d, ldb1=%d\n", static_cast<int>(trans_a), static_cast<int>(trans_b), ma, na, mb, nb, alpha, beta, lda0, lda1, ldb0, ldb1);

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    fastGemm(trans_a, trans_b, ma, na, mb, nb,
             alpha, a, lda0, lda1, b, ldb0, ldb1,
             beta, c, ldc, opt);
}

}} // cv::dnn
