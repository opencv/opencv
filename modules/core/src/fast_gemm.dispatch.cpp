// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "precomp.hpp"
#include "opencv2/core/private/fast_gemm.hpp"

#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "fast_gemm.simd.hpp"
#include "fast_gemm.simd_declarations.hpp"
#undef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "fast_gemm.default.hpp"

namespace cv { namespace hal {

int fastGemmMC(const FastGemmOpt &opt) {
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        return opt_RVV::fastGemmMC();
    } else
#endif
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
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        return opt_RVV::fastGemmNC();
    } else
#endif
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
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        return opt_RVV::fastGemmKC();
    } else
#endif
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

int fastGemmNR(const FastGemmOpt &opt) {
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        return opt_RVV::fastGemmNR();
    } else
#endif
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmNR();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmNR();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmNR();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmNR();
    } else
#endif
    {
        return cpu_baseline::fastGemmNR();
    }
}



size_t fastGemmPackBSize(size_t N, size_t K, const FastGemmOpt &opt) {
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        return static_cast<size_t>(opt_RVV::fastGemmPackBSize((int)N, (int)K));
    } else
#endif
#if CV_TRY_NEON
    if (opt.use_neon) {
        return static_cast<size_t>(opt_NEON::fastGemmPackBSize((int)N, (int)K));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return static_cast<size_t>(opt_AVX2::fastGemmPackBSize((int)N, (int)K));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return static_cast<size_t>(opt_AVX::fastGemmPackBSize((int)N, (int)K));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return static_cast<size_t>(opt_LASX::fastGemmPackBSize((int)N, (int)K));
    } else
#endif
    {
        return static_cast<size_t>(cpu_baseline::fastGemmPackBSize((int)N, (int)K));
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

#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        opt_RVV::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
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

#if CV_SIMD_SCALABLE
// Apply the alpha/beta epilogue to one accumulator row.
static inline void fast_gemm_thin_store_row(float* c, v_float32 acc,
                                            v_float32 v_alpha, float beta) {
    if (beta == 0.f)
        vx_store(c, v_mul(acc, v_alpha));
    else if (beta == 1.f)
        vx_store(c, v_fma(acc, v_alpha, vx_load(c)));
    else
        vx_store(c, v_fma(acc, v_alpha, v_mul(vx_load(c), vx_setall_f32(beta))));
}

// Multiply MR (1..4) rows of A by one packed-B strip. Sizeless vector types
// cannot form the acc[] array the CV_SIMD path uses, so the accumulators are
// named variables selected by the compile-time MR (the `if (MR > n)` branches
// fold away) and stay in registers for the whole K loop. Accumulation remains
// in increasing-k order, bit-exact with the previous scratch-buffer code.
template<int MR>
static inline void fast_gemm_thin_block(int K, const float* A, int lda0, int lda1,
                                        const float* b_strip, int NR,
                                        v_float32 v_alpha, float beta,
                                        float* c_strip, int ldc) {
    v_float32 c0 = vx_setzero_f32(), c1 = vx_setzero_f32();
    v_float32 c2 = vx_setzero_f32(), c3 = vx_setzero_f32();
    for (int k = 0; k < K; k++) {
        v_float32 bv = vx_load(b_strip + k * NR);
        const float* a = A + k * lda1;
                     c0 = v_fma(bv, vx_setall_f32(a[0]),         c0);
        if (MR > 1) c1 = v_fma(bv, vx_setall_f32(a[lda0]),       c1);
        if (MR > 2) c2 = v_fma(bv, vx_setall_f32(a[2 * lda0]),   c2);
        if (MR > 3) c3 = v_fma(bv, vx_setall_f32(a[3 * lda0]),   c3);
    }
                 fast_gemm_thin_store_row(c_strip,             c0, v_alpha, beta);
    if (MR > 1)  fast_gemm_thin_store_row(c_strip + ldc,       c1, v_alpha, beta);
    if (MR > 2)  fast_gemm_thin_store_row(c_strip + 2 * ldc,   c2, v_alpha, beta);
    if (MR > 3)  fast_gemm_thin_store_row(c_strip + 3 * ldc,   c3, v_alpha, beta);
}
#endif // CV_SIMD_SCALABLE

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
    // Process the strip in register-resident row blocks; the common thin case
    // M <= 4 (transformer token generation) is a single pass.
    const int NR = VTraits<v_float32>::vlanes();
    const v_float32 v_alpha = vx_setall_f32(alpha);
    int m = 0;
    for (; m + 4 <= M; m += 4)
        fast_gemm_thin_block<4>(K, A + m * lda0, lda0, lda1, b_strip, NR, v_alpha, beta, c_strip + m * ldc, ldc);
    switch (M - m) {
        case 1: fast_gemm_thin_block<1>(K, A + m * lda0, lda0, lda1, b_strip, NR, v_alpha, beta, c_strip + m * ldc, ldc); break;
        case 2: fast_gemm_thin_block<2>(K, A + m * lda0, lda0, lda1, b_strip, NR, v_alpha, beta, c_strip + m * ldc, ldc); break;
        case 3: fast_gemm_thin_block<3>(K, A + m * lda0, lda0, lda1, b_strip, NR, v_alpha, beta, c_strip + m * ldc, ldc); break;
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

static constexpr int FAST_GEMM_THIN_TB_MAX_M = 4;

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
                std::fill(c_i, c_i + N, 0.f);
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
              float *C, int ldc, const FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *packed_b = (const char *)packed_B;
    char *c = (char *)C;

    int lda0 = lda, lda1 = 1;
    if (trans_a) {
        std::swap(lda0, lda1);
    }

#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        opt_RVV::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
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

static void fast_gemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
                      float alpha, const float *A, int lda0, int lda1, const float *B, int ldb0, int ldb1,
                      float beta, float *C, int ldc, const FastGemmOpt &opt, bool allow_thin) {
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

    if (allow_thin && !trans_b && ldb1 == 1 && (fastGemmThinEligible(M, N, K) || (uint64_t)M * N * K <= 10000)) {
        return fast_gemm_thin(alpha, beta, M, N, K, a, lda0, lda1, b, ldb0, c, ldc, opt.multi_thread);
    }

#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        opt_RVV::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                b, ldb0, ldb1, beta,
                                c, ldc, sizeof(float), opt.multi_thread);
    } else
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

void fastGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
              float alpha, const float *A, int lda0, int lda1, const float *B, int ldb0, int ldb1,
              float beta, float *C, int ldc, const FastGemmOpt &opt) {
    fast_gemm(trans_a, trans_b, ma, na, mb, nb, alpha, A, lda0, lda1, B, ldb0, ldb1, beta, C, ldc, opt, true);
}


void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                   int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                   const float *B, int ldb0, int ldb1, float beta, float *C, int ldc, const FastGemmOpt &opt) {
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

    if (M <= FAST_GEMM_THIN_TB_MAX_M && lda1 == 1 && ldb0 == 1 && ldb1 != 1) {
#if CV_TRY_RVV && CV_RVV
        if (opt.use_rvv) {
            opt_RVV::fastGemmThinTBKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda0, B, ldb1, beta, C, ldc, opt.multi_thread);
        } else
#endif
#if CV_TRY_NEON
        if (opt.use_neon) {
            opt_NEON::fastGemmThinTBKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda0, B, ldb1, beta, C, ldc, opt.multi_thread);
        } else
#endif
#if CV_TRY_AVX2
        if (opt.use_avx2) {
            opt_AVX2::fastGemmThinTBKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda0, B, ldb1, beta, C, ldc, opt.multi_thread);
        } else
#endif
        // No AVX branch: below AVX2 the intrinsics are 128-bit and the baseline copy runs faster.
#if CV_TRY_LASX
        if (opt.use_lasx) {
            opt_LASX::fastGemmThinTBKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda0, B, ldb1, beta, C, ldc, opt.multi_thread);
        } else
#endif
        {
            cpu_baseline::fastGemmThinTBKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda0, B, ldb1, beta, C, ldc, opt.multi_thread);
        }
        return;
    }

#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        opt_RVV::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
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
                   const float *packed_B, float beta, float *C, int ldc, const FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *b = (const char *)packed_B;
    char *c = (char *)C;

#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv) {
        opt_RVV::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
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

void fastGemmPagedQK(const char *Q, const std::vector<const char *> &K, char *A,
                     size_t B, size_t T_q, size_t Nq, size_t N_k, size_t T_s, size_t D, size_t T_k,
                     float sm_scale, size_t esz, bool isQ3d, const FastGemmOpt &opt) {
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv)
        opt_RVV::pagedAttnQKGemmKernel(
            Q, K, A,
            B, T_q, Nq, N_k, T_s, D, T_k,
            sm_scale, esz, isQ3d
        );
    else
#endif
#if CV_TRY_NEON
    if (opt.use_neon)
        opt_NEON::pagedAttnQKGemmKernel(
            Q, K, A,
            B, T_q, Nq, N_k, T_s, D, T_k,
            sm_scale, esz, isQ3d
        );
    else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2)
        opt_AVX2::pagedAttnQKGemmKernel(
            Q, K, A,
            B, T_q, Nq, N_k, T_s, D, T_k,
            sm_scale, esz, isQ3d
        );
    else
#endif
#if CV_TRY_AVX
    if (opt.use_avx)
        opt_AVX::pagedAttnQKGemmKernel(
            Q, K, A,
            B, T_q, Nq, N_k, T_s, D, T_k,
            sm_scale, esz, isQ3d
        );
    else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx)
        opt_LASX::pagedAttnQKGemmKernel(
            Q, K, A,
            B, T_q, Nq, N_k, T_s, D, T_k,
            sm_scale, esz, isQ3d
        );
    else
#endif
    cpu_baseline::pagedAttnQKGemmKernel(
        Q, K, A,
        B, T_q, Nq, N_k, T_s, D, T_k,
        sm_scale, esz, isQ3d
    );
}


void fastGemmPagedAV(const char *A, const std::vector<const char *> &V, char *Out,
                     size_t B, size_t T_a, size_t Nq, size_t N_k, size_t T_s, size_t D, size_t T_v,
                     size_t esz, bool canonical_layout, size_t packed_stride, const FastGemmOpt &opt) {
#if CV_TRY_RVV && CV_RVV
    if (opt.use_rvv)
        opt_RVV::pagedAttnAVGemmKernel(
            A, V, Out,
            B, T_a, Nq, N_k, T_s, D, T_v,
            esz, canonical_layout, packed_stride
        );
    else
#endif
#if CV_TRY_NEON
    if (opt.use_neon)
        opt_NEON::pagedAttnAVGemmKernel(
            A, V, Out,
            B, T_a, Nq, N_k, T_s, D, T_v,
            esz, canonical_layout, packed_stride
        );
    else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::pagedAttnAVGemmKernel(
            A, V, Out,
            B, T_a, Nq, N_k, T_s, D, T_v,
            esz, canonical_layout, packed_stride
        );
    }
    else
#endif
#if CV_TRY_AVX
    if (opt.use_avx){
        opt_AVX::pagedAttnAVGemmKernel(
            A, V, Out,
            B, T_a, Nq, N_k, T_s, D, T_v,
            esz, canonical_layout, packed_stride
        );
    }
    else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx){
        opt_LASX::pagedAttnAVGemmKernel(
            A, V, Out,
            B, T_a, Nq, N_k, T_s, D, T_v,
            esz, canonical_layout, packed_stride
        );
    }
    else
#endif
    {
        cpu_baseline::pagedAttnAVGemmKernel(
                A, V, Out,
                B, T_a, Nq, N_k, T_s, D, T_v,
                esz, canonical_layout, packed_stride
            );

    }
}

static const int64_t FAST_GEMM_HAL_MIN_MACS = 1 << 15;
static const int64_t FAST_GEMM_HAL_MIN_THIN_MACS = 1 << 19;

static bool fast_gemm_overlap(const void *p0, size_t size0, const void *p1, size_t size1) {
    const uchar *a = (const uchar *)p0, *b = (const uchar *)p1;
    return a < b + size1 && b < a + size0;
}

bool fastGemm32f(const float* src1, size_t src1_step, const float* src2, size_t src2_step,
                 float alpha, const float* src3, size_t src3_step, float beta,
                 float* dst, size_t dst_step, int m_a, int n_a, int n_d, int flags) {
    const size_t esz = sizeof(float);
    const bool trans_a = (flags & CV_HAL_GEMM_1_T) != 0, trans_b = (flags & CV_HAL_GEMM_2_T) != 0;
    const bool trans_c = (flags & CV_HAL_GEMM_3_T) != 0;
    const int M = trans_a ? n_a : m_a, K = trans_a ? m_a : n_a, N = n_d;
    const int mb = trans_b ? N : K, nb = trans_b ? K : N;
    if (M <= 0 || N <= 0 || K <= 0)
        return false;
    const int64_t macs = (int64_t)M * N * K;
    if (macs < FAST_GEMM_HAL_MIN_MACS || ((M < 4 || K < 8) && macs < FAST_GEMM_HAL_MIN_THIN_MACS))
        return false;
    if (src1_step % esz || src2_step % esz || src3_step % esz || dst_step % esz ||
        std::max(std::max(src1_step, src2_step), dst_step) / esz > (size_t)INT_MAX)
        return false;

    const size_t d_size = (M - 1) * dst_step + N * esz;
    if (fast_gemm_overlap(dst, d_size, src1, (m_a - 1) * src1_step + n_a * esz) ||
        fast_gemm_overlap(dst, d_size, src2, (mb - 1) * src2_step + nb * esz))
        return false;

    const bool use_c = beta != 0.f && src3 && src3_step != 0;
    if (use_c && src3 != dst) {
        const int mc = trans_c ? N : M, nc = trans_c ? M : N;
        if (fast_gemm_overlap(dst, d_size, src3, (mc - 1) * src3_step + nc * esz))
            return false;
        for (int i = 0; i < M; i++) {
            float *d = (float *)((uchar *)dst + i * dst_step);
            if (trans_c) {
                const uchar *c = (const uchar *)src3 + i * esz;
                for (int j = 0; j < N; j++)
                    d[j] = *(const float *)(c + j * src3_step);
            } else {
                memcpy(d, (const uchar *)src3 + i * src3_step, N * esz);
            }
        }
    } else if (use_c && (trans_c || src3_step != dst_step)) {
        return false;
    }

    FastGemmOpt opt;
    opt.init();
    fast_gemm(trans_a, trans_b, m_a, n_a, mb, nb, alpha,
              src1, (int)(src1_step / esz), 1, src2, (int)(src2_step / esz), 1,
              use_c ? beta : 0.f, dst, (int)(dst_step / esz), opt, false);
    return true;
}

}} // cv::hal
