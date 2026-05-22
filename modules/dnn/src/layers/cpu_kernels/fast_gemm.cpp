// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "fast_gemm.hpp"
#include "mlas_gemm.hpp"

#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "fast_gemm_kernels.simd.hpp"
#include "layers/cpu_kernels/fast_gemm_kernels.simd_declarations.hpp"
#undef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "fast_gemm_kernels.default.hpp"

namespace cv { namespace dnn {

int fastGemmMC(const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmMC();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmMC();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmMC();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmMC();
    } else
#endif
    {
        return cpu_baseline::fastGemmMC();
    }
}

int fastGemmNC(const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmNC();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmNC();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmNC();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmNC();
    } else
#endif
    {
        return cpu_baseline::fastGemmNC();
    }
}

int fastGemmKC(const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmKC();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmKC();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmKC();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmKC();
    } else
#endif
    {
        return cpu_baseline::fastGemmKC();
    }
}

size_t fastGemmPackBSize(size_t N, size_t K, const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return static_cast<size_t>(opt_NEON::fastGemmPackBSize(N, K));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return static_cast<size_t>(opt_AVX2::fastGemmPackBSize(N, K));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return static_cast<size_t>(opt_AVX::fastGemmPackBSize(N, K));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return static_cast<size_t>(opt_LASX::fastGemmPackBSize(N, K));
    } else
#endif
    {
        return static_cast<size_t>(cpu_baseline::fastGemmPackBSize(N, K));
    }
}

void fastGemmPackB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckTypeEQ(B.type(), CV_32F, "fastGemmPackB: only float32 is supported for now");

    auto B_shape = shape(B);
    int batch = total(B_shape, 0, B_shape.size() - 2),
        K = B_shape[B_shape.size() - 2], N = B_shape.back(), ldb0 = N, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

    const auto *b = B.ptr<const char>();
    int esz = B.elemSize();

#if CV_TRY_NEON
    if (opt.use_neon) {
        size_t size_packed_B = opt_NEON::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_NEON::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        size_t size_packed_B = opt_AVX2::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_AVX2::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        size_t size_packed_B = opt_AVX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_AVX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        size_t size_packed_B = opt_LASX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_LASX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
    {
        size_t size_packed_B = cpu_baseline::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            cpu_baseline::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    }
}

void fastGemmPackB(bool trans, size_t N, size_t K, const float *B, size_t ldb, float *packed_B, const FastGemmOpt &opt) {
    size_t ldb0 = ldb, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

    const auto &b = (const char *)B;
    auto *packed_b = (char *)packed_B;

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    }
}

static constexpr int FAST_GEMM_THIN_MAX_M = 12;
#if !CV_SIMD
// Scalar-fallback strip width (unroll factor, not SIMD lanes); 4 matches the narrowest universal-intrinsic float width so layout stays consistent with SIMD builds.
static constexpr int FAST_GEMM_THIN_SCALAR_NR = 4;
#endif

static inline int fast_gemm_thin_lanes() {
#if (CV_SIMD || CV_SIMD_SCALABLE)
    return VTraits<v_float32>::vlanes();
#else
    return FAST_GEMM_THIN_SCALAR_NR;
#endif
}

static inline void fast_gemm_thin_strip(int M, int K, float alpha,
                                        const float* A, int lda0, int lda1,
                                        const float* b_strip,
                                        float beta, float* c_strip, int ldc) {
#if CV_SIMD
    const int NR = VTraits<v_float32>::vlanes();
    v_float32 acc[FAST_GEMM_THIN_MAX_M];
    for (int m = 0; m < M; m++) acc[m] = vx_setzero_f32();

    for (int k = 0; k < K; k++) {
        v_float32 bv = vx_load(b_strip + k * NR);
        for (int m = 0; m < M; m++) {
            v_float32 am = vx_setall_f32(A[m * lda0 + k * lda1]);
            acc[m] = v_fma(bv, am, acc[m]);
        }
    }

    const v_float32 v_alpha = vx_setall_f32(alpha);
    if (beta == 0.f) {
        for (int m = 0; m < M; m++)
            vx_store(c_strip + m * ldc, v_mul(acc[m], v_alpha));
    } else if (beta == 1.f) {
        for (int m = 0; m < M; m++) {
            v_float32 cv = vx_load(c_strip + m * ldc);
            vx_store(c_strip + m * ldc, v_fma(acc[m], v_alpha, cv));
        }
    } else {
        const v_float32 v_beta = vx_setall_f32(beta);
        for (int m = 0; m < M; m++) {
            v_float32 cv = vx_load(c_strip + m * ldc);
            cv = v_mul(cv, v_beta);
            vx_store(c_strip + m * ldc, v_fma(acc[m], v_alpha, cv));
        }
    }
#elif CV_SIMD_SCALABLE
    // Scalable vector types (e.g. RVV) are sizeless and cannot form arrays;
    // back the per-row accumulators with a scalar scratch buffer.
    const int NR = VTraits<v_float32>::vlanes();
    float acc_buf[FAST_GEMM_THIN_MAX_M * VTraits<v_float32>::max_nlanes];
    for (int m = 0; m < M; m++) vx_store(acc_buf + m * NR, vx_setzero_f32());

    for (int k = 0; k < K; k++) {
        v_float32 bv = vx_load(b_strip + k * NR);
        for (int m = 0; m < M; m++) {
            v_float32 am = vx_setall_f32(A[m * lda0 + k * lda1]);
            v_float32 acc_m = vx_load(acc_buf + m * NR);
            vx_store(acc_buf + m * NR, v_fma(bv, am, acc_m));
        }
    }

    const v_float32 v_alpha = vx_setall_f32(alpha);
    if (beta == 0.f) {
        for (int m = 0; m < M; m++)
            vx_store(c_strip + m * ldc, v_mul(vx_load(acc_buf + m * NR), v_alpha));
    } else if (beta == 1.f) {
        for (int m = 0; m < M; m++) {
            v_float32 cv = vx_load(c_strip + m * ldc);
            vx_store(c_strip + m * ldc, v_fma(vx_load(acc_buf + m * NR), v_alpha, cv));
        }
    } else {
        const v_float32 v_beta = vx_setall_f32(beta);
        for (int m = 0; m < M; m++) {
            v_float32 cv = vx_load(c_strip + m * ldc);
            cv = v_mul(cv, v_beta);
            vx_store(c_strip + m * ldc, v_fma(vx_load(acc_buf + m * NR), v_alpha, cv));
        }
    }
#else
    const int NR = FAST_GEMM_THIN_SCALAR_NR;
    float acc[FAST_GEMM_THIN_MAX_M * FAST_GEMM_THIN_SCALAR_NR] = {0};
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            float a_mk = A[m * lda0 + k * lda1];
            for (int c = 0; c < NR; c++)
                acc[m * NR + c] += a_mk * b_strip[k * NR + c];
        }
    }
    if (beta == 0.f) {
        for (int m = 0; m < M; m++)
            for (int c = 0; c < NR; c++)
                c_strip[m * ldc + c] = alpha * acc[m * NR + c];
    } else {
        for (int m = 0; m < M; m++)
            for (int c = 0; c < NR; c++)
                c_strip[m * ldc + c] = beta * c_strip[m * ldc + c] + alpha * acc[m * NR + c];
    }
#endif
}

bool fastGemmThinEligible(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (M > FAST_GEMM_THIN_MAX_M) return false;
    const int NR = fast_gemm_thin_lanes();
    if (N < 2 * NR) return false;
    return true;
}

size_t fastGemmThinPackBSize(int N, int K) {
    const int NR = fast_gemm_thin_lanes();
    const int n_strips = (N + NR - 1) / NR;
    return (size_t)n_strips * (size_t)NR * (size_t)K;
}

void fastGemmThinPackB(int N, int K, const float* B, size_t ldb_K, size_t ldb_N, float* packed_B) {
    const int NR = fast_gemm_thin_lanes();
    const int n_strips = (N + NR - 1) / NR;
    for (int s = 0; s < n_strips; s++) {
        const int j0 = s * NR;
        const int nr = std::min(NR, N - j0);
        float* strip = packed_B + (size_t)s * NR * K;
        for (int k = 0; k < K; k++) {
            float* out = strip + k * NR;
            int c = 0;
            for (; c < nr; c++) out[c] = B[k * ldb_K + (j0 + c) * ldb_N];
            for (; c < NR; c++) out[c] = 0.f;
        }
    }
}

void fastGemmThin(int M, int N, int K, float alpha,
                  const float* A, int lda0, int lda1,
                  const float* packed_B, float beta,
                  float* C, int ldc, bool multi_thread) {
    const int NR = fast_gemm_thin_lanes();
    const int n_full_strips = N / NR;
    const int n_tail = N - n_full_strips * NR;

    auto fn = [&](const Range& r) {
        for (int s = r.start; s < r.end; s++) {
            const float* b_strip = packed_B + (size_t)s * NR * K;
            float* c_strip = C + s * NR;
            fast_gemm_thin_strip(M, K, alpha, A, lda0, lda1, b_strip, beta, c_strip, ldc);
        }
    };
    if (multi_thread && n_full_strips > 1) {
        parallel_for_(Range(0, n_full_strips), fn, (double)n_full_strips * M * K * NR * (1.0 / 1024.0));
    } else {
        fn(Range(0, n_full_strips));
    }

    if (n_tail > 0) {
        const int j0 = n_full_strips * NR;
        const float* b_strip = packed_B + (size_t)n_full_strips * NR * K;
        for (int m = 0; m < M; m++) {
            for (int c = 0; c < n_tail; c++) {
                float acc_s = 0.f;
                for (int k = 0; k < K; k++)
                    acc_s += A[m * lda0 + k * lda1] * b_strip[k * NR + c];
                const float mul = alpha * acc_s;
                float* p = C + m * ldc + j0 + c;
                *p = (beta == 0.f) ? mul : beta * (*p) + mul;
            }
        }
    }
}

static void fast_gemm_thin(float alpha, float beta, int M, int N, int K,
                           const char *a_, int lda0, int lda1,
                           const char *b_, int ldb,
                           char *c_, int ldc, bool multi_thread) {
    const float* A = (const float*)a_;
    const float* B = (const float*)b_;
    float* C = (float*)c_;

    if (fastGemmThinEligible(M, N, K)) {
        AutoBuffer<float> packed(fastGemmThinPackBSize(N, K));
        fastGemmThinPackB(N, K, B, (size_t)ldb, 1, packed.data());
        fastGemmThin(M, N, K, alpha, A, lda0, lda1, packed.data(), beta, C, ldc, multi_thread);
        return;
    }

    auto fn = [&](const Range &r) {
        for (int i = r.start; i < r.end; i++) {
            float* c_i = C + i * ldc;
            if (beta == 0.f)
                for (int j = 0; j < N; j++) c_i[j] = 0.f;
            else if (beta != 1.f)
                for (int j = 0; j < N; j++) c_i[j] *= beta;
            for (int k = 0; k < K; k++) {
                const float* b_k = B + k * ldb;
                float aval = alpha * A[i * lda0 + k * lda1];
                for (int j = 0; j < N; j++)
                    c_i[j] += aval * b_k[j];
            }
        }
    };
    if (multi_thread) {
        parallel_for_(Range(0, M), fn, (double)M * K * N * (1.0 / 1024.0));
    } else {
        fn(Range(0, M));
    }
}

void fastGemm(bool trans_a, int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *packed_B, float beta,
              float *C, int ldc, FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *packed_b = (const char *)packed_B;
    char *c = (char *)C;

    int lda0 = lda, lda1 = 1;
    if (trans_a) {
        std::swap(lda0, lda1);
    }

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
    {
        cpu_baseline::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
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

    if (!trans_b && ldb1 == 1 && (fastGemmThinEligible(M, N, K) || (uint64_t)M * N * K <= 10000)) {
        return fast_gemm_thin(alpha, beta, M, N, K, a, lda0, lda1, b, ldb0, c, ldc, opt.multi_thread);
    }

#ifdef HAVE_MLAS
    const bool a_row_major = (lda0 == 1 || lda1 == 1);
    const bool b_row_major = (ldb0 == 1 || ldb1 == 1);
    if (a_row_major && b_row_major) {
        const int phys_lda = std::max(lda0, lda1);
        const int phys_ldb = std::max(ldb0, ldb1);
        if (mlasSgemm(trans_a, trans_b, M, N, K,
                      alpha, A, phys_lda, B, phys_ldb,
                      beta, C, ldc)) {
            return;
        }
    }
#endif

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
    {
        cpu_baseline::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                     b, ldb0, ldb1, beta,
                                     c, ldc, sizeof(float), opt.multi_thread);
    }
}

void fastGemm(bool trans_a, bool trans_b,
              float alpha, const Mat &A, const Mat &B,
              float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/fastGemm: only support float32 for now");
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/fastGemm: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/fastGemm: B and C should have the same type");

    const auto shape_a = shape(A);
    CV_CheckEQ(shape_a.size(), static_cast<size_t>(2), "DNN/fastGemm: A must be 2-dimensional");
    const auto shape_b = shape(B);
    CV_CheckEQ(shape_b.size(), static_cast<size_t>(2), "DNN/fastGemm: B must be 2-dimensional");
    const auto shape_c = shape(C);
    CV_CheckEQ(shape_c.size(), static_cast<size_t>(2), "DNN/fastGemm: C must be 2-dimensional");

    int ma = shape_a[0], na = shape_a[1];
    int mb = shape_b[0], nb = shape_b[1];

    int lda0 = na, lda1 = 1, ldb0 = nb, ldb1 = 1, ldc = shape_c[1];

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    fastGemm(trans_a, trans_b, ma, na, mb, nb,
             alpha, a, lda0, lda1, b, ldb0, ldb1,
             beta, c, ldc, opt);
}

void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                   int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                   const float *B, int ldb0, int ldb1, float beta, float *C, int ldc, FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *b = (const char *)B;
    char *c = (char *)C;

    // Below ~10K MACs (M*N*K) the blocked SIMD kernel's pack/tile overhead outweighs its speedup, so route tiny problems through the unblocked thin path.
    if (ldb1 == 1 && (fastGemmThinEligible(M, N, K) || (uint64_t)M * N * K <= 10000)) {
        const size_t esz = sizeof(float);
        // Parallelise over batch with single-thread inner thin gemm
        if (opt.multi_thread && batch > 1) {
            parallel_for_(Range(0, (int)batch), [&](const Range& r) {
                for (int i = r.start; i < r.end; i++) {
                    fast_gemm_thin(alpha, beta, M, N, K,
                                   a + A_offsets[i] * esz, lda0, lda1,
                                   b + B_offsets[i] * esz, ldb0,
                                   c + C_offsets[i] * esz, ldc, false);
                }
            }, (double)batch * M * N * K * (1.0 / 1024.0));
        } else {
            for (size_t i = 0; i < batch; i++) {
                fast_gemm_thin(alpha, beta, M, N, K,
                               a + A_offsets[i] * esz, lda0, lda1,
                               b + B_offsets[i] * esz, ldb0,
                               c + C_offsets[i] * esz, ldc, opt.multi_thread);
            }
        }
        return;
    }

#ifdef HAVE_MLAS
    bool a_ok = false, b_ok = false;
    bool mlas_trans_a = false, mlas_trans_b = false;
    int  mlas_lda = 0, mlas_ldb = 0;
    if (lda1 == 1)      { a_ok = true; mlas_trans_a = false; mlas_lda = lda0; }
    else if (lda0 == 1) { a_ok = true; mlas_trans_a = true;  mlas_lda = lda1; }
    if (ldb1 == 1)      { b_ok = true; mlas_trans_b = false; mlas_ldb = ldb0; }
    else if (ldb0 == 1) { b_ok = true; mlas_trans_b = true;  mlas_ldb = ldb1; }
    if (a_ok && b_ok) {
        if (mlasSgemmBatch(batch, A_offsets, B_offsets, C_offsets,
                            mlas_trans_a, mlas_trans_b, M, N, K,
                            alpha, A, mlas_lda, B, mlas_ldb,
                            beta, C, ldc)) {
            return;
        }
    }
#endif

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    }
}

void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *packed_B_offsets, const size_t *C_offsets,
                   int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                   const float *packed_B, float beta, float *C, int ldc, FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *b = (const char *)packed_B;
    char *c = (char *)C;

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    }
}

void fastGemmBatch(bool trans_a, bool trans_b,
                   float alpha, const Mat &A, const Mat &B,
                   float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/fastGemmBatch: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/fastGemmBatch: B and C should have the same type");
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/fastGemmBatch: only support float32 for now");

    const auto shape_a = shape(A);
    const auto shape_b = shape(B);
    const auto shape_c = shape(C);
    CV_CheckGE(shape_a.size(), static_cast<size_t>(2), "DNN/fastGemmBatch: A must be n-dimensional (n >= 2)");
    CV_CheckGE(shape_b.size(), static_cast<size_t>(2), "DNN/fastGemmBatch: B must be n-dimensional (n >= 2)");

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    MatMulHelper helper;
    helper.compute(trans_a, trans_b, shape_a, shape_b, shape_c);

    fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                  helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1, b, helper.ldb0,
                  helper.ldb1, beta, c, helper.ldc, opt);
}


void fastGemmBatch(size_t batch,
                   const std::vector<size_t> &A_offsets, const std::vector<size_t> &B_offsets, const std::vector<size_t> &C_offsets,
                   int M, int N, int K, float alpha, const Mat&A, int lda0, int lda1,
                   const Mat&B, int ldb0, int ldb1, float beta, Mat&C, int ldc, FastGemmOpt &opt){
    const char *a = (const char *)A.ptr<const float>();
    const char *b = (const char *)B.ptr<const float>();
    char *c = (char *)C.ptr<float>();

    if (ldb1 == 1 && (fastGemmThinEligible(M, N, K) || (uint64_t)M * N * K <= 10000)) {
        const size_t esz = sizeof(float);
        if (opt.multi_thread && batch > 1) {
            parallel_for_(Range(0, (int)batch), [&](const Range& r) {
                for (int i = r.start; i < r.end; i++) {
                    fast_gemm_thin(alpha, beta, M, N, K,
                                   a + A_offsets[i] * esz, lda0, lda1,
                                   b + B_offsets[i] * esz, ldb0,
                                   c + C_offsets[i] * esz, ldc, false);
                }
            }, (double)batch * M * N * K * (1.0 / 1024.0));
        } else {
            for (size_t i = 0; i < batch; i++) {
                fast_gemm_thin(alpha, beta, M, N, K,
                               a + A_offsets[i] * esz, lda0, lda1,
                               b + B_offsets[i] * esz, ldb0,
                               c + C_offsets[i] * esz, ldc, opt.multi_thread);
            }
        }
        return;
    }

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    }
}


void pagedAttnQKGemm(
    const Mat& Q,const std::vector<Mat> &K, Mat& A,
    int T_q, int Nq, int N_k, int T_s, int D,
    const FastGemmOpt &opt
) {
    size_t esz = Q.elemSize();

    for (size_t s = 0; s < K.size(); s++){
        CV_CheckTypeEQ(Q.type(), K[s].type(), "pagedAttnQKGemmKernel: Q and K should have the same type");
        CV_CheckTrue(esz == K[s].elemSize(), "pagedAttnQKGemmKernel: Q and K should have the same element size");
    }
    CV_CheckTypeEQ(Q.type(), A.type(), "pagedAttnQKGemmKernel: Q and A should have the same type");

    CV_CheckTrue(
        T_s % fastGemmNC(opt) == 0,
        "pagedAttnQKGemmKernel: T_s should be divisible by the macro tile size"
    );

    const auto shape_q = shape(Q);
    CV_CheckTrue(
        ((shape_q.size() == 3) || (shape_q.size() == 4)),
        "pagedAttnQKGemmKernel: Q must be 3D or 4D (T_q x Nq x D) or (kq_groups x T_q x Nq x D)"
    );
    const auto shape_a = shape(A);
    CV_CheckTrue( shape_a.size() == 4, "pagedAttnQKGemmKernel: A must be 4D (B x N_q x T_q x D)" );

    const int B = shape_a[0];
    for (size_t s = 0; s < K.size(); s++) {
        const auto shape_k = shape(K[s]);
        CV_CheckTrue(
            shape_k.size() == 3,
            "pagedAttnQKGemmKernel: each K must be 4D (B x N_k x T_s x D)"
        );
        CV_CheckEQ(shape_k[0], B, "pagedAttnQKGemmKernel: the batch size of K should be the same as A");
        CV_CheckEQ(shape_k[1], N_k, "pagedAttnQKGemmKernel: the number of heads in K should match that of Q");
        CV_CheckEQ(shape_k[2], D * T_s, "pagedAttnQKGemmKernel: the head dimension of K should match that of Q and A");
    }

    std::vector<const char*> packed_K;
    for (size_t s = 0; s < K.size(); s++){
        packed_K.push_back(K[s].ptr<const char>());
    }

    char*a = A.ptr<char>();
    bool isQ3D = shape_q.size() == 3;
#if CV_TRY_NEON
    if (opt.use_neon)
        opt_NEON::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2)
        opt_AVX2::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
#if CV_TRY_AVX
    if (opt.use_avx)
        opt_AVX::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx)
        opt_LASX::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
    cpu_baseline::pagedAttnQKGemmKernel(
        Q.ptr<const char>(), packed_K, a,
        B, T_q, Nq, N_k, T_s, D,
        esz, isQ3D
    );


}


void pagedAttnAVGemm(
    const Mat& A,const std::vector<Mat> &V, Mat& Out,
    int T_q, int Nq, int N_k, int T_s, int D,
    const FastGemmOpt &opt
) {
    size_t esz = A.elemSize();

    for (size_t s = 0; s < V.size(); s++) {
        CV_CheckTypeEQ(A.type(), V[s].type(), "pagedAttnAVGemmKernel: A and V should have the same type");
        CV_CheckTrue(esz == V[s].elemSize(), "pagedAttnAVGemmKernel: A and V should have the same element size");
    }
    CV_CheckTypeEQ(A.type(), Out.type(), "pagedAttnAVGemmKernel: A and Out should have the same type");

    CV_CheckTrue(
        T_s % fastGemmKC(opt) == 0,
        "pagedAttnAVGemmKernel: T_s should be divisible by the macro tile size"
    );

    const auto shape_a = shape(A);
    CV_CheckTrue(
        shape_a.size() == 4,
        "pagedAttnAVGemmKernel: A must be 4D (B x n_q x T_q x D)"
    );

    const int B = shape_a[0];
    for (size_t s = 0; s < V.size(); s++) {
        const auto shape_v = shape(V[s]);
        CV_CheckTrue(
            shape_v.size() == 3,
            "pagedAttnAVGemmKernel: each V must be 3D (B x N_k x (T_s * D))"
        );
        CV_CheckEQ(shape_v[0], B, "pagedAttnAVGemmKernel: the batch size of V should be the same as A");
        CV_CheckEQ(shape_v[1], N_k, "pagedAttnAVGemmKernel: the number of heads in V should match that of A");
        CV_CheckEQ(shape_v[2], (int)fastGemmPackBSize(D, T_s, opt),
                    "pagedAttnAVGemmKernel: the last dimension of V-pages should be be equal to packed D x T_s Mat size");
    }

    std::vector<const char*> packed_V;
    for (size_t s = 0; s < V.size(); s++){
        packed_V.push_back(V[s].ptr<const char>());
    }

    bool canonical_layout = shape(Out).size() == 3;
#if CV_TRY_NEON
    if (opt.use_neon)
        opt_NEON::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    }
    else
#endif
#if CV_TRY_AVX
    if (opt.use_avx){
        opt_AVX::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    }
    else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx){
        opt_LASX::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    }
    else
#endif
    {
        cpu_baseline::pagedAttnAVGemmKernel(
                A.ptr<const char>(), packed_V, Out.ptr<char>(),
                B, T_q, Nq, N_k, T_s, D,
                esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
            );

    }


}

}} // cv::dnn
