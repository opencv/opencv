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

namespace cv { namespace dnn {

#define FAST_GEMM_IMPLEMENT_PACK(N, suffix, styp, dtyp) \
static void fast_gemm_pack##N##suffix( int m, int k, const void* A_, \
                                      int lda0, int lda1, void* packA_ ) \
{ \
    const styp* A = (const styp*)A_; \
    dtyp* packA = (dtyp*)packA_; \
    for( int i = 0; i < m; i += N ) { \
        if (i + N-1 < m) { \
            const styp* a_ptr = A + lda0*i; \
            for( int j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                FAST_GEMM_LOAD_TO_BUF_##N(styp); \
                FAST_GEMM_PACK##suffix##_##N(buf, packA); \
            } \
        } else { \
            const styp* a_ptr[N]; \
            for (int k = 0; k < N; k++) a_ptr[k] = A + lda0*(i+k < m ? i+k : i); \
            for( int j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                FAST_GEMM_LOAD_TO_BUF_BORDERS_##N(styp); \
                FAST_GEMM_PACK##suffix##_##N(buf, packA); \
            } \
        } \
    } \
}

#define FAST_GEMM_LOAD_TO_BUF_6(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_6(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j] }

#define FAST_GEMM_LOAD_TO_BUF_8(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_8(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j] }

#define FAST_GEMM_LOAD_TO_BUF_12(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7], \
        a_ptr[j+lda0*8], a_ptr[j+lda0*9], a_ptr[j+lda0*10], a_ptr[j+lda0*11] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_12(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j], \
        a_ptr[8][j], a_ptr[9][j], a_ptr[10][j], a_ptr[11][j] }

#define FAST_GEMM_PACK_COPY(src, dst, N) \
    memcpy((dst), (src), N*sizeof(src[0]))
#define FAST_GEMM_PACK_f32_8(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 8)
#define FAST_GEMM_PACK_f32_12(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 12)

FAST_GEMM_IMPLEMENT_PACK(8, _f32, float, float)
FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float)

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// AVX and AVX2 (16 x 256-bit registers)
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround for FMA
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

static void fast_gemm12x8_f32(int k, const char *a_, const char *b_,
                              char *c_, int ldc, const void* palpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;
    float alpha = *(const float*)palpha;

    __m256 s00 = _mm256_setzero_ps(),
           s10 = _mm256_setzero_ps(),
           s20 = _mm256_setzero_ps(),
           s30 = _mm256_setzero_ps(),
           s40 = _mm256_setzero_ps(),
           s50 = _mm256_setzero_ps(),
           s60 = _mm256_setzero_ps(),
           s70 = _mm256_setzero_ps(),
           s80 = _mm256_setzero_ps(),
           s90 = _mm256_setzero_ps(),
           sa0 = _mm256_setzero_ps(),
           sb0 = _mm256_setzero_ps();
        for (int p = 0; p < k; p++, a += FAST_GEMM_F32_MR, b += FAST_GEMM_F32_NR) {
            __m256 b0 = _mm256_loadu_ps(b);

            __m256 a0 = _mm256_set1_ps(*a);
            s00 = _mm256_fmadd_ps(b0, a0, s00);
            __m256 a1 = _mm256_set1_ps(*(a + 1));
            s10 = _mm256_fmadd_ps(b0, a1, s10);
            __m256 a2 = _mm256_set1_ps(*(a + 2));
            s20 = _mm256_fmadd_ps(b0, a2, s20);

            a0 = _mm256_set1_ps(*(a + 3));
            s30 = _mm256_fmadd_ps(b0, a0, s30);
            a1 = _mm256_set1_ps(*(a + 4));
            s40 = _mm256_fmadd_ps(b0, a1, s40);
            a2 = _mm256_set1_ps(*(a + 5));
            s50 = _mm256_fmadd_ps(b0, a2, s50);

            a0 = _mm256_set1_ps(*(a + 6));
            s60 = _mm256_fmadd_ps(b0, a0, s60);
            a1 = _mm256_set1_ps(*(a + 7));
            s70 = _mm256_fmadd_ps(b0, a1, s70);
            a2 = _mm256_set1_ps(*(a + 8));
            s80 = _mm256_fmadd_ps(b0, a2, s80);

            a0 = _mm256_set1_ps(*(a + 9));
            s90 = _mm256_fmadd_ps(b0, a0, s90);
            a1 = _mm256_set1_ps(*(a + 10));
            sa0 = _mm256_fmadd_ps(b0, a1, sa0);
            a2 = _mm256_set1_ps(*(a + 11));
            sb0 = _mm256_fmadd_ps(b0, a2, sb0);
        }

        __m256 c0, c1, c2, c3, v_alpha = _mm256_set1_ps(alpha);
    #define FAST_GEMM_FINALE(row0, row1, row2, row3)    \
        c0 = _mm256_loadu_ps(c + row0 * ldc);   \
        c1 = _mm256_loadu_ps(c + row1 * ldc);   \
        c2 = _mm256_loadu_ps(c + row2 * ldc);   \
        c3 = _mm256_loadu_ps(c + row3 * ldc);   \
        c0 = _mm256_fmadd_ps(s##row0##0, v_alpha, c0);  \
        c1 = _mm256_fmadd_ps(s##row1##0, v_alpha, c1);  \
        c1 = _mm256_fmadd_ps(s##row1##0, v_alpha, c2);  \
        c1 = _mm256_fmadd_ps(s##row1##0, v_alpha, c3);  \
        _mm256_storeu_ps(c + row0 * ldc, c0);   \
        _mm256_storeu_ps(c + row1 * ldc, c1);   \
        _mm256_storeu_ps(c + row2 * ldc, c2);   \
        _mm256_storeu_ps(c + row3 * ldc, c3);   \

        FAST_GEMM_FINALE(0, 1,  2,  3);
        FAST_GEMM_FINALE(4, 5,  6,  7);
        FAST_GEMM_FINALE(8, 9, 10, 11);
    #undef FAST_GEMM_FINALE
}

#endif // CV_AVX

// NEON AARCH64 (32 128-bit registers)
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_NEON && CV_NEON_AARCH64

static void fast_gemm8x12_f32(int k, const char *a_, const char *b_,
                              char *c_, int ldc, const void* palpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;
    float alpha = *(const float*)palpha;

    float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00;
    float32x4_t s10 = s00, s11 = s00, s12 = s00;
    float32x4_t s20 = s00, s21 = s00, s22 = s00;
    float32x4_t s30 = s00, s31 = s00, s32 = s00;
    float32x4_t s40 = s00, s41 = s00, s42 = s00;
    float32x4_t s50 = s00, s51 = s00, s52 = s00;
    float32x4_t s60 = s00, s61 = s00, s62 = s00;
    float32x4_t s70 = s00, s71 = s00, s72 = s00;

    for(int p = 0; p < k; p++, a += FAST_GEMM_F32_MR, b += FAST_GEMM_F32_NR)
    {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);

        s00 = vfmaq_laneq_f32(s00, b0, a0, 0);
        s01 = vfmaq_laneq_f32(s01, b1, a0, 0);
        s02 = vfmaq_laneq_f32(s02, b2, a0, 0);
        s10 = vfmaq_laneq_f32(s10, b0, a0, 1);
        s11 = vfmaq_laneq_f32(s11, b1, a0, 1);
        s12 = vfmaq_laneq_f32(s12, b2, a0, 1);

        s20 = vfmaq_laneq_f32(s20, b0, a0, 2);
        s21 = vfmaq_laneq_f32(s21, b1, a0, 2);
        s22 = vfmaq_laneq_f32(s22, b2, a0, 2);
        s30 = vfmaq_laneq_f32(s30, b0, a0, 3);
        s31 = vfmaq_laneq_f32(s31, b1, a0, 3);
        s32 = vfmaq_laneq_f32(s32, b2, a0, 3);

        a0 = vld1q_f32(a + 4);

        s40 = vfmaq_laneq_f32(s40, b0, a0, 0);
        s41 = vfmaq_laneq_f32(s41, b1, a0, 0);
        s42 = vfmaq_laneq_f32(s42, b2, a0, 0);
        s50 = vfmaq_laneq_f32(s50, b0, a0, 1);
        s51 = vfmaq_laneq_f32(s51, b1, a0, 1);
        s52 = vfmaq_laneq_f32(s52, b2, a0, 1);

        s60 = vfmaq_laneq_f32(s60, b0, a0, 2);
        s61 = vfmaq_laneq_f32(s61, b1, a0, 2);
        s62 = vfmaq_laneq_f32(s62, b2, a0, 2);
        s70 = vfmaq_laneq_f32(s70, b0, a0, 3);
        s71 = vfmaq_laneq_f32(s71, b1, a0, 3);
        s72 = vfmaq_laneq_f32(s72, b2, a0, 3);
    }

    float32x4_t c0, c1, c2, c3, c4, c5, v_alpha = vdupq_n_f32(alpha);
#define FAST_GEMM_FINALE(row0, row1)         \
    c0 = vld1q_f32(c + row0 * ldc);          \
    c1 = vld1q_f32(c + row0 * ldc + 4);      \
    c2 = vld1q_f32(c + row0 * ldc + 8);      \
    c3 = vld1q_f32(c + row1 * ldc);          \
    c4 = vld1q_f32(c + row1 * ldc + 4);      \
    c5 = vld1q_f32(c + row1 * ldc + 8);      \
    c0 = vfmaq_f32(c0, s##row0##0, v_alpha); \
    c1 = vfmaq_f32(c1, s##row0##1, v_alpha); \
    c2 = vfmaq_f32(c2, s##row0##2, v_alpha); \
    c3 = vfmaq_f32(c3, s##row1##0, v_alpha); \
    c4 = vfmaq_f32(c4, s##row1##1, v_alpha); \
    c5 = vfmaq_f32(c5, s##row1##2, v_alpha); \
    vst1q_f32(c + row0 * ldc, c0);           \
    vst1q_f32(c + row0 * ldc + 4, c1);       \
    vst1q_f32(c + row0 * ldc + 8, c2);       \
    vst1q_f32(c + row1 * ldc, c3);           \
    vst1q_f32(c + row1 * ldc + 4, c4);       \
    vst1q_f32(c + row1 * ldc + 8, c5);

    FAST_GEMM_FINALE(0, 1);
    FAST_GEMM_FINALE(2, 3);
    FAST_GEMM_FINALE(4, 5);
    FAST_GEMM_FINALE(6, 7);
#undef FAST_GEMM_FINALE
}

#endif // CV_NEON && CV_NEON_AARCH64

CV_CPU_OPTIMIZATION_NAMESPACE_END

void fast_gemm_packB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckEQ(B.dims, 2, "fast_gemm_packB: input mat should be two-dimensional");
    CV_CheckTypeEQ(B.type(), CV_32F, "fast_gemm_packB: only float32 is supported for now");

    auto B_shape = shape(B);
    int K = B_shape[0], N = B_shape[1], ldb0 = N, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

    int esz = B.elemSize(),
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_NR = FAST_GEMM_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    if (!packed_B.empty()) {
        packed_B.clear();
    }
    int total_size = static_cast<int>((N + NC - 1) / NC) * NC * K;
    packed_B.resize(total_size);

    const auto *ptr_B = B.ptr<char>();
    auto ptr_packed = packed_B.data();
    int n_tiles = (N + NC - 1) / NC;
    for (int r = 0; r < n_tiles; ++r) {
        int j0 = r * NC;
        int nc = N - j0 < NC ? N - j0 : NC;
        int _nc = static_cast<int>((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        for (int k = 0; k < K; k += KC) {
            int kc = K - k < KC ? K - k : KC;
#if CV_TRY_AVX || CV_TRY_AVX2
            if (opt.use_avx || opt.use_avx2) {
                fast_gemm_pack8_f32(nc, kc, ptr_B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, ptr_packed);
            } else
#endif
#if CV_TRY_NEON
            if (opt.use_neon_aarch64) {
                fast_gemm_pack12_f32(nc, kc, ptr_B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, ptr_packed);
            } else
#endif
            {
                fast_gemm_pack8_f32(nc, kc, ptr_B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, ptr_packed);
            }
            ptr_packed += _nc * kc;
        }
    }
}

static void fast_gemm_f32(int k, const char *a_, const char *b_,
                          char *c_, int ldc, const void* palpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;
    float alpha = *(const float*)palpha;

    float sbuf[FAST_GEMM_F32_MR * FAST_GEMM_F32_NR];
    memset(sbuf, 0, sizeof(sbuf));
    for(int p = 0; p < k; p++) {
        for( int i = 0; i < FAST_GEMM_F32_MR; i++ ) {
            float ai = a[FAST_GEMM_F32_MR * p + i];
            for( int j = 0; j < FAST_GEMM_F32_NR; j++ )
                sbuf[i * FAST_GEMM_F32_NR + j] += b[FAST_GEMM_F32_NR * p + j] * ai;
        }
    }
    for (int i = 0; i < FAST_GEMM_F32_MR; i++) {
        for (int j = 0; j < FAST_GEMM_F32_NR; j++)
            c[i * ldc + j] += alpha * sbuf[i * FAST_GEMM_F32_NR + j];
    }
}

static void fast_gemm_macro_kernel(int m, int n, int k,
                                   const char *packA, const char *packB,
                                   const void* palpha, char *c, int ldc0,
                                   int MR, int NR, FastGemmOpt &opt) {
    int esz = sizeof(float);
    int ldc0_esz = ldc0 * esz;

    double tempC[FAST_GEMM_F32_MR * FAST_GEMM_F32_NR]; // make sure the buffer is big enough
    for(int i = 0; i < m; i += MR) {
        for(int j = 0; j < n; j += NR) {
            char* cptr0 = &c[i * ldc0_esz + j * esz];
            char* cptr = cptr0;
            int ldc = ldc0;
            int mr = m - i < MR ? m - i : MR;
            int nr = n - j < NR ? n - j : NR;
            int nr_esz = nr * esz;
            bool partial = (bool)((mr < MR) | (nr < NR));
            if (partial) {
                memset(tempC, 0, sizeof(tempC));
                cptr = (char*)tempC;
                ldc = NR;
                for(int p = 0; p < mr; p++)
                    memcpy(cptr + p*(ldc*esz), cptr0 + p*ldc0_esz, nr_esz);
            }

#if CV_TRY_AVX
            if (opt.use_avx) {
                opt_AVX::fast_gemm12x8_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha);
            } else
#endif
#if CV_TRY_AVX2
            if (opt.use_avx2) {
                opt_AVX2::fast_gemm_12x8_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha);
            } else
#endif
#if CV_TRY_NEON
            if (opt.use_neon_aarch64) {
                cpu_baseline::fast_gemm8x12_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha);
            } else
#endif
            {
                fast_gemm_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha);
            }

            if (partial) {
                for(int p = 0; p < mr; p++)
                    memcpy(cptr0 + p*ldc0_esz, cptr + p*(ldc*esz), nr_esz);
            }
        }
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

void fast_gemm(bool trans_a, int M, int N, int K,
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

    const void* palpha = (const void*)&alpha;

    int esz = sizeof(float);
    int GEMM_MC = FAST_GEMM_F32_MC,
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_MR = FAST_GEMM_F32_MR,
        GEMM_NR = FAST_GEMM_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    size_t buff_size = KC * MC * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_MAX_STACKBUF;
    int m_tiles = (M + MC - 1) / MC;
    int n_tiles = (N + NC - 1) / NC;
    int total_tiles = m_tiles * n_tiles;

    auto fn = [&](const Range &r) {
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size)); // TODO: use AutoBuffer
        const char *packed_b_ = packed_b;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = c + (i0 * ldc + j0) * esz;
            packed_b_ = packed_b + j0 * K * esz;

            if (beta == 0.f) {
                for(int i = 0; i < mc; i++)
                    memset(c_block + i * ldc_block * esz, 0, nc * esz);
            } else if (beta != 1.f) {
                for(int i = 0; i < mc; i++) {
                    float* c_i = (float*)c_block + i * ldc_block;
                    for(int j = 0; j < nc; j++)
                        c_i[j] *= beta;
                }
            }

            int _nc = static_cast<int>((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR * esz;
            for(int k0 = 0; k0 < K; k0 += KC)
            {
                int kc = K - k0 < KC ? K - k0 : KC;
#if CV_TRY_AVX || CV_TRY_AVX2
                if (opt.use_avx || opt.use_avx2) {
                    fast_gemm_pack12_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                } else
#endif
#if CV_TRY_NEON
                if (opt.use_neon_aarch64) {
                    fast_gemm_pack8_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                } else
#endif
                {
                    fast_gemm_pack12_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                }
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b_, palpha,
                                           c_block, ldc_block, GEMM_MR, GEMM_NR, opt);
                packed_b_ += _nc * kc;
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    int total = total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fast_gemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
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

    const void* palpha = (const void*)&alpha;

    if (!trans_b && ldb1 == 1 && (M <= 4 || (uint64_t)M * N * K <= 10000)) {
        return fast_gemm_thin(alpha, beta, M, N, K, a, lda0, lda1, b, ldb0, c, ldc);
    }

    int esz = sizeof(float);

    int GEMM_MC = FAST_GEMM_F32_MC,
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_MR = FAST_GEMM_F32_MR,
        GEMM_NR = FAST_GEMM_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = FAST_GEMM_STORAGE / ((MC + NC) * esz);
    KC = KC > 8 ? KC : 8;
    KC = KC < K ? KC : K;

    size_t buff_size = KC * (MC + NC) * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_MAX_STACKBUF;
    int m_tiles = (M + MC - 1) / MC;
    int n_tiles = (N + NC - 1) / NC;
    int total_tiles = m_tiles * n_tiles;

    std::function<void(int, int, const void*, int, int, void*)> a_packer, b_packer;
    a_packer = fast_gemm_pack8_f32;
    b_packer = fast_gemm_pack12_f32;

    auto fn = [&](const Range &r) {
        char* pack_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        char* pack_b = pack_a + KC * MC * esz;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = c + (i0 * ldc + j0) * esz;

            if (beta == 0.f) {
                for(int i = 0; i < mc; i++)
                    memset(c_block + i * ldc_block * esz, 0, nc * esz);
            } else if (beta != 1.f) {
                for(int i = 0; i < mc; i++) {
                    float* c_i = (float*)c_block + i * ldc_block;
                    for(int j = 0; j < nc; j++)
                        c_i[j] *= beta;
                }
            }

            for(int k0 = 0; k0 < K; k0 += KC)
            {
                int kc = K - k0 < KC ? K - k0 : KC;
#if CV_TRY_AVX || CV_TRY_AVX2
                if (opt.use_avx || opt.use_avx2) {
                    fast_gemm_pack12_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, pack_a);
                    fast_gemm_pack8_f32(nc, kc, b + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, pack_b);
                } else
#endif
#if CV_TRY_NEON
                if (opt.use_neon_aarch64) {
                    fast_gemm_pack8_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, pack_a);
                    fast_gemm_pack12_f32(nc, kc, b + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, pack_b);
                } else
#endif
                {
                    fast_gemm_pack12_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, pack_a);
                    fast_gemm_pack8_f32(nc, kc, b + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, pack_b);
                }
                fast_gemm_macro_kernel(mc, nc, kc, pack_a, pack_b, palpha,
                                       c_block, ldc_block, GEMM_MR, GEMM_NR, opt);
            }
        }

        if (!use_stackbuff) {
            free(pack_a);
        }
    };

    int total = total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fast_gemm(bool trans_a, bool trans_b,
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

    fast_gemm(trans_a, trans_b, ma, na, mb, nb,
             alpha, a, lda0, lda1, b, ldb0, ldb1,
             beta, c, ldc, opt);
}

}} // cv::dnn
