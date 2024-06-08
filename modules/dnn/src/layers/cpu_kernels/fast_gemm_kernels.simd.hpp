// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/utility.hpp> // parallel_for_

#define FAST_GEMM_STORAGE (1<<20) // 2^20
#define FAST_GEMM_MAX_STACKBUF (1 << 14)

#if CV_AVX
#define FAST_GEMM_F32_MC 60
#define FAST_GEMM_F32_NC 320
#elif CV_LASX
#define FAST_GEMM_F32_MC 48
#define FAST_GEMM_F32_NC 128
#else // CV_NEON_AARCH64, SIMD128
#define FAST_GEMM_F32_MC 144
#define FAST_GEMM_F32_NC 72
#endif

#if CV_AVX
#define FAST_GEMM_F32_MR 12
#define FAST_GEMM_F32_NR 8
#elif CV_LASX
#define FAST_GEMM_F32_MR 12
#define FAST_GEMM_F32_NR 16
#else // CV_NEON_AARCH64, CV_SIMD128
#define FAST_GEMM_F32_MR 8
#define FAST_GEMM_F32_NR 12
#endif

#if CV_AVX
#define FAST_GEMM_F32_PACKED_STRIDE_K 128
#else // CV_LASX, CV_NEON_AARCH64, CV_SIMD128
#define FAST_GEMM_F32_PACKED_STRIDE_K 64
#endif

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

#define FAST_GEMM_LOAD_TO_BUF_16(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7], \
        a_ptr[j+lda0*8], a_ptr[j+lda0*9], a_ptr[j+lda0*10], a_ptr[j+lda0*11], \
        a_ptr[j+lda0*12], a_ptr[j+lda0*13], a_ptr[j+lda0*14], a_ptr[j+lda0*15] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_16(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j], \
        a_ptr[8][j], a_ptr[9][j], a_ptr[10][j], a_ptr[11][j], \
        a_ptr[12][j], a_ptr[13][j], a_ptr[14][j], a_ptr[15][j] }

#define FAST_GEMM_PACK_COPY(src, dst, N) \
    memcpy((dst), (src), N*sizeof(src[0]))
#define FAST_GEMM_PACK_f32_8(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 8)
#define FAST_GEMM_PACK_f32_12(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 12)
#define FAST_GEMM_PACK_f32_16(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 16)

namespace cv { namespace dnn {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

int fastGemmPackBSize(int N, int K);

void fastGemmPackBKernel(const char *B, char *packed_B, int N, int K, int ldb0, int ldb1, int esz);

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *B, int ldb0, int ldb1,
                    float beta, char *C, int ldc, int esz, bool multi_thread);
void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *packed_B, float beta, char *C, int ldc, int esz, bool multi_thread);

void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *B, int ldb0, int ldb1, float beta, char *C, int ldc, int esz);
void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *packed_B, float beta, char *C, int ldc, int esz);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

/*
    Compute kernels that optimized for different platforms
*/
#if CV_NEON && CV_NEON_AARCH64 // AARCH64: 32 x 128-bit registers

FAST_GEMM_IMPLEMENT_PACK(8, _f32, float, float) // a packer
FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float) // b packer

static inline void fast_gemm8x12_f32(int k, const char *a_, const char *b_,
                                     char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

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

#elif CV_AVX // AVX and AVX2 (16 x 256-bit registers)

FAST_GEMM_IMPLEMENT_PACK(8, _f32, float, float) // a packer
FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float) // b packer

#if !CV_FMA3 // AVX workaround for FMA
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

static inline void fast_gemm12x8_f32(int k, const char *a_, const char *b_, char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

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
           s100 = _mm256_setzero_ps(),
           s110 = _mm256_setzero_ps();
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
        s100 = _mm256_fmadd_ps(b0, a1, s100);
        a2 = _mm256_set1_ps(*(a + 11));
        s110 = _mm256_fmadd_ps(b0, a2, s110);
    }

    __m256 c0, c1, c2, c3, v_alpha = _mm256_set1_ps(alpha);
#define FAST_GEMM_FINALE(row0, row1, row2, row3)    \
    c0 = _mm256_loadu_ps(c + row0 * ldc);   \
    c1 = _mm256_loadu_ps(c + row1 * ldc);   \
    c2 = _mm256_loadu_ps(c + row2 * ldc);   \
    c3 = _mm256_loadu_ps(c + row3 * ldc);   \
    c0 = _mm256_fmadd_ps(s##row0##0, v_alpha, c0);  \
    c1 = _mm256_fmadd_ps(s##row1##0, v_alpha, c1);  \
    c2 = _mm256_fmadd_ps(s##row2##0, v_alpha, c2);  \
    c3 = _mm256_fmadd_ps(s##row3##0, v_alpha, c3);  \
    _mm256_storeu_ps(c + row0 * ldc, c0);   \
    _mm256_storeu_ps(c + row1 * ldc, c1);   \
    _mm256_storeu_ps(c + row2 * ldc, c2);   \
    _mm256_storeu_ps(c + row3 * ldc, c3);   \

    FAST_GEMM_FINALE(0, 1,  2,  3);
    FAST_GEMM_FINALE(4, 5,  6,  7);
    FAST_GEMM_FINALE(8, 9, 10, 11);
#undef FAST_GEMM_FINALE
}

#elif CV_LASX // LASX (32 x 256-bit registers)

FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float) // a packer
FAST_GEMM_IMPLEMENT_PACK(16, _f32, float, float) // b packer

static inline void fast_gemm12x16_f32(int k, const char *a_, const char *b_, char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

    __m256 s00  = _v256_setall_ps(0), s01  = s00,
           s10  = s00, s11  = s00,
           s20  = s00, s21  = s00,
           s30  = s00, s31  = s00,
           s40  = s00, s41  = s00,
           s50  = s00, s51  = s00,
           s60  = s00, s61  = s00,
           s70  = s00, s71  = s00,
           s80  = s00, s81  = s00,
           s90  = s00, s91  = s00,
           s100 = s00, s101 = s00,
           s110 = s00, s111 = s00;
    for (int p = 0; p < k; p++, a += FAST_GEMM_F32_MR, b += FAST_GEMM_F32_NR) {
        __m256 b0 = (__m256)__lasx_xvld(b, 0), b1 = (__m256)__lasx_xvld(b + 8, 0);

        __m256 a0 = _v256_setall_ps(*a);
        s00 = __lasx_xvfmadd_s(b0, a0, s00);
        s01 = __lasx_xvfmadd_s(b1, a0, s01);
        __m256 a1 = _v256_setall_ps(*(a + 1));
        s10 = __lasx_xvfmadd_s(b0, a1, s10);
        s11 = __lasx_xvfmadd_s(b1, a1, s11);
        __m256 a2 = _v256_setall_ps(*(a + 2));
        s20 = __lasx_xvfmadd_s(b0, a2, s20);
        s21 = __lasx_xvfmadd_s(b1, a2, s21);
        __m256 a3 = _v256_setall_ps(*(a + 3));
        s30 = __lasx_xvfmadd_s(b0, a3, s30);
        s31 = __lasx_xvfmadd_s(b1, a3, s31);

        a0 = _v256_setall_ps(*(a + 4));
        s40 = __lasx_xvfmadd_s(b0, a0, s40);
        s41 = __lasx_xvfmadd_s(b1, a0, s41);
        a1 = _v256_setall_ps(*(a + 5));
        s50 = __lasx_xvfmadd_s(b0, a1, s50);
        s51 = __lasx_xvfmadd_s(b1, a1, s51);
        a2 = _v256_setall_ps(*(a + 6));
        s60 = __lasx_xvfmadd_s(b0, a2, s60);
        s61 = __lasx_xvfmadd_s(b1, a2, s61);
        a3 = _v256_setall_ps(*(a + 7));
        s70 = __lasx_xvfmadd_s(b0, a3, s70);
        s71 = __lasx_xvfmadd_s(b1, a3, s71);

        a0 = _v256_setall_ps(*(a + 8));
        s80 = __lasx_xvfmadd_s(b0, a0, s80);
        s81 = __lasx_xvfmadd_s(b1, a0, s81);
        a1 = _v256_setall_ps(*(a + 9));
        s90 = __lasx_xvfmadd_s(b0, a1, s90);
        s91 = __lasx_xvfmadd_s(b1, a1, s91);
        a2 = _v256_setall_ps(*(a + 10));
        s100 = __lasx_xvfmadd_s(b0, a2, s100);
        s101 = __lasx_xvfmadd_s(b1, a2, s101);
        a3 = _v256_setall_ps(*(a + 11));
        s110 = __lasx_xvfmadd_s(b0, a3, s110);
        s111 = __lasx_xvfmadd_s(b1, a3, s111);
    }

    __m256 c0, c1, c2, c3, c4, c5, c6, c7, v_alpha = _v256_setall_ps(alpha);
#define FAST_GEMM_FINALE(row0, row1, row2, row3)       \
    c0 = (__m256)__lasx_xvld(c + row0 * ldc, 0);       \
    c1 = (__m256)__lasx_xvld(c + row0 * ldc, 8 * 4);   \
    c2 = (__m256)__lasx_xvld(c + row1 * ldc, 0);       \
    c3 = (__m256)__lasx_xvld(c + row1 * ldc, 8 * 4);   \
    c4 = (__m256)__lasx_xvld(c + row2 * ldc, 0);       \
    c5 = (__m256)__lasx_xvld(c + row2 * ldc, 8 * 4);   \
    c6 = (__m256)__lasx_xvld(c + row3 * ldc, 0);       \
    c7 = (__m256)__lasx_xvld(c + row3 * ldc, 8 * 4);   \
    c0 = __lasx_xvfmadd_s(s##row0##0, v_alpha, c0);    \
    c1 = __lasx_xvfmadd_s(s##row0##1, v_alpha, c1);    \
    c2 = __lasx_xvfmadd_s(s##row1##0, v_alpha, c2);    \
    c3 = __lasx_xvfmadd_s(s##row1##1, v_alpha, c3);    \
    c4 = __lasx_xvfmadd_s(s##row2##0, v_alpha, c4);    \
    c5 = __lasx_xvfmadd_s(s##row2##1, v_alpha, c5);    \
    c6 = __lasx_xvfmadd_s(s##row3##0, v_alpha, c6);    \
    c7 = __lasx_xvfmadd_s(s##row3##1, v_alpha, c7);    \
    __lasx_xvst(c0, c + row0 * ldc,     0);            \
    __lasx_xvst(c1, c + row0 * ldc, 8 * 4);            \
    __lasx_xvst(c2, c + row1 * ldc,     0);            \
    __lasx_xvst(c3, c + row1 * ldc, 8 * 4);            \
    __lasx_xvst(c4, c + row2 * ldc,     0);            \
    __lasx_xvst(c5, c + row2 * ldc, 8 * 4);            \
    __lasx_xvst(c6, c + row3 * ldc,     0);            \
    __lasx_xvst(c7, c + row3 * ldc, 8 * 4);

    FAST_GEMM_FINALE(0, 1,  2,  3);
    FAST_GEMM_FINALE(4, 5,  6,  7);
    FAST_GEMM_FINALE(8, 9, 10, 11);
#undef FAST_GEMM_FINALE
}

#elif CV_SIMD128 // armv7: 16 x 128-bit registers

FAST_GEMM_IMPLEMENT_PACK(8, _f32, float, float) // a packer
FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float) // b packer

static inline void fast_gemm8x12_f32(int k, const char *a_, const char *b_,
                                     char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

    v_float32x4 s00 = v_setzero_f32(), s01 = s00, s02 = s00;
    v_float32x4 s10 = s00, s11 = s00, s12 = s00;
    v_float32x4 s20 = s00, s21 = s00, s22 = s00;
    v_float32x4 s30 = s00, s31 = s00, s32 = s00;
    v_float32x4 s40 = s00, s41 = s00, s42 = s00;
    v_float32x4 s50 = s00, s51 = s00, s52 = s00;
    v_float32x4 s60 = s00, s61 = s00, s62 = s00;
    v_float32x4 s70 = s00, s71 = s00, s72 = s00;

    for(int p = 0; p < k; p++, a += FAST_GEMM_F32_MR, b += FAST_GEMM_F32_NR) {
        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);

        v_float32x4 a0 = v_setall_f32(*a);
        s00 = v_fma(b0, a0, s00);
        s01 = v_fma(b1, a0, s01);
        s02 = v_fma(b2, a0, s02);
        v_float32x4 a1 = v_setall_f32(*(a + 1));
        s10 = v_fma(b0, a1, s10);
        s11 = v_fma(b1, a1, s11);
        s12 = v_fma(b2, a1, s12);

        v_float32x4 a2 = v_setall_f32(*(a + 2));
        s20 = v_fma(b0, a2, s20);
        s21 = v_fma(b1, a2, s21);
        s22 = v_fma(b2, a2, s22);
        v_float32x4 a3 = v_setall_f32(*(a + 3));
        s30 = v_fma(b0, a3, s30);
        s31 = v_fma(b1, a3, s31);
        s32 = v_fma(b2, a3, s32);

        a0 = v_setall_f32(*(a + 4));
        s40 = v_fma(b0, a0, s40);
        s41 = v_fma(b1, a0, s41);
        s42 = v_fma(b2, a0, s42);
        a1 = v_setall_f32(*(a + 5));
        s50 = v_fma(b0, a1, s50);
        s51 = v_fma(b1, a1, s51);
        s52 = v_fma(b2, a1, s52);

        a2 = v_setall_f32(*(a + 6));
        s60 = v_fma(b0, a2, s60);
        s61 = v_fma(b1, a2, s61);
        s62 = v_fma(b2, a2, s62);
        a3 = v_setall_f32(*(a + 7));
        s70 = v_fma(b0, a3, s70);
        s71 = v_fma(b1, a3, s71);
        s72 = v_fma(b2, a3, s72);
    }

    v_float32x4 c0, c1, c2, c3, c4, c5, v_alpha = v_setall_f32(alpha);
#define FAST_GEMM_FINALE(row0, row1)       \
    c0 = v_load(c + row0 * ldc);         \
    c1 = v_load(c + row0 * ldc + 4);     \
    c2 = v_load(c + row0 * ldc + 8);     \
    c3 = v_load(c + row1 * ldc);         \
    c4 = v_load(c + row1 * ldc + 4);     \
    c5 = v_load(c + row1 * ldc + 8);     \
    c0 = v_fma(s##row0##0, v_alpha, c0); \
    c1 = v_fma(s##row0##1, v_alpha, c1); \
    c2 = v_fma(s##row0##2, v_alpha, c2); \
    c3 = v_fma(s##row1##0, v_alpha, c3); \
    c4 = v_fma(s##row1##1, v_alpha, c4); \
    c5 = v_fma(s##row1##2, v_alpha, c5); \
    v_store(c + row0 * ldc, c0);         \
    v_store(c + row0 * ldc + 4, c1);     \
    v_store(c + row0 * ldc + 8, c2);     \
    v_store(c + row1 * ldc, c3);         \
    v_store(c + row1 * ldc + 4, c4);     \
    v_store(c + row1 * ldc + 8, c5);

    FAST_GEMM_FINALE(0, 1);
    FAST_GEMM_FINALE(2, 3);
    FAST_GEMM_FINALE(4, 5);
    FAST_GEMM_FINALE(6, 7);
#undef FAST_GEMM_FINALE
}

#endif

static inline void fast_gemm_macro_kernel(int m, int n, int k,
                                          const char *packed_A, const char *packed_B,
                                          float alpha, char *c, int ldc0, int esz) {
    int ldc0_esz = ldc0 * esz;

    double tempC[FAST_GEMM_F32_MR * FAST_GEMM_F32_NR]; // make sure the buffer is big enough
    for(int i = 0; i < m; i += FAST_GEMM_F32_MR) {
        for(int j = 0; j < n; j += FAST_GEMM_F32_NR) {
            char* cptr0 = &c[i * ldc0_esz + j * esz];
            char* cptr = cptr0;
            int ldc = ldc0;
            int mr = m - i < FAST_GEMM_F32_MR ? m - i : FAST_GEMM_F32_MR;
            int nr = n - j < FAST_GEMM_F32_NR ? n - j : FAST_GEMM_F32_NR;
            int nr_esz = nr * esz;
            bool partial = (bool)((mr < FAST_GEMM_F32_MR) | (nr < FAST_GEMM_F32_NR));
            if (partial) {
                memset(tempC, 0, sizeof(tempC));
                cptr = (char *)tempC;
                ldc = FAST_GEMM_F32_NR;
                for(int p = 0; p < mr; p++)
                    memcpy(cptr + p * (ldc * esz), cptr0 + p * ldc0_esz, nr_esz);
            }
#if CV_NEON && CV_NEON_AARCH64
            fast_gemm8x12_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);
#elif CV_AVX
            fast_gemm12x8_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);
#elif CV_LASX
            fast_gemm12x16_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);
#elif CV_SIMD128
            fast_gemm8x12_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);
#endif

            if (partial) {
                for(int p = 0; p < mr; p++)
                    memcpy(cptr0 + p * ldc0_esz, cptr + p * (ldc * esz), nr_esz);
            }
        }
    }
}

int fastGemmPackBSize(int N, int K) {
    int GEMM_NC = FAST_GEMM_F32_NC, GEMM_NR = FAST_GEMM_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

    return static_cast<int>((N + NC - 1) / NC) * NC * K;
}

void fastGemmPackBKernel(const char *B, char *packed_B, int N, int K, int ldb0, int ldb1, int esz) {
    int GEMM_NC = FAST_GEMM_F32_NC, GEMM_NR = FAST_GEMM_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    int n_tiles = (N + NC - 1) / NC;
    for (int r = 0; r < n_tiles; ++r) {
        int j0 = r * NC;
        int nc = N - j0 < NC ? N - j0 : NC;
        int _nc = static_cast<int>((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR * esz;
        for (int k = 0; k < K; k += KC) {
            int kc = K - k < KC ? K - k : KC;
#if CV_NEON && CV_NEON_AARCH64
            fast_gemm_pack12_f32(nc, kc, B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_B);
#elif CV_AVX
            fast_gemm_pack8_f32(nc, kc, B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_B);
#elif CV_LASX
            fast_gemm_pack16_f32(nc, kc, B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_B);
#elif CV_SIMD128
            fast_gemm_pack12_f32(nc, kc, B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_B);
#endif
            packed_B += _nc * kc;
        }
    }
}

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *B, int ldb0, int ldb1,
                    float beta, char *C, int ldc, int esz, bool multi_thread) {
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

    auto fn = [&](const Range &r) {
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        char* packed_b = packed_a + KC * MC * esz;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = C + (i0 * ldc + j0) * esz;

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
                // pack a
#if CV_NEON && CV_NEON_AARCH64
                fast_gemm_pack8_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_AVX
                fast_gemm_pack12_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_LASX
                fast_gemm_pack12_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_SIMD128
                fast_gemm_pack8_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#endif

                // pack b
#if CV_NEON && CV_NEON_AARCH64
                fast_gemm_pack12_f32(nc, kc, B + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#elif CV_AVX
                fast_gemm_pack8_f32(nc, kc, B + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#elif CV_LASX
                fast_gemm_pack16_f32(nc, kc, B + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#elif CV_SIMD128
                fast_gemm_pack12_f32(nc, kc, B + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#endif

                // run kernel
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b, alpha, c_block, ldc_block, esz);
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    if (multi_thread) {
        int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
        double nstripes = (size_t)total_tiles * cost_per_thread * (1 / 1024.0);
        parallel_for_(Range(0, total_tiles), fn, nstripes);
    } else {
        fn(Range(0, total_tiles));
    }

}

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *packed_B, float beta, char *C, int ldc, int esz, bool multi_thread) {
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
        const char *packed_b_ = packed_B;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = C + (i0 * ldc + j0) * esz;
            packed_b_ = packed_B + j0 * K * esz;

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
                // pack a
#if CV_NEON && CV_NEON_AARCH64
                fast_gemm_pack8_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_AVX
                fast_gemm_pack12_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_LASX
                fast_gemm_pack12_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_SIMD128
                fast_gemm_pack8_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#endif

                // run kernel
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b_, alpha, c_block, ldc_block, esz);
                packed_b_ += _nc * kc;
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    if (multi_thread) {
        int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
        double nstripes = (size_t)total_tiles * cost_per_thread * (1 / 1024.0);
        parallel_for_(Range(0, total_tiles), fn, nstripes);
    } else {
        fn(Range(0, total_tiles));
    }
}

void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *B, int ldb0, int ldb1, float beta, char *C, int ldc, int esz) {
    int GEMM_MC = FAST_GEMM_F32_MC,
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_MR = FAST_GEMM_F32_MR,
        GEMM_NR = FAST_GEMM_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    size_t buff_size = KC * (MC + NC) * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_MAX_STACKBUF;
    int m_tiles = (M + MC - 1) / MC;
    int n_tiles = (N + NC - 1) / NC;
    int total_tiles = m_tiles * n_tiles;

    auto fn = [&](const Range &r) {
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        char* packed_b = packed_a + KC * MC * esz;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            const int batch_index = static_cast<int>(tile_idx / total_tiles);
            const int m_tiles_index = static_cast<int>((tile_idx - batch_index * total_tiles) / n_tiles);
            const int n_tiles_index = static_cast<int>(tile_idx % n_tiles);

            int i0 = m_tiles_index * MC;
            int j0 = n_tiles_index * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            const char *a_block = A + A_offsets[batch_index] * esz;
            const char *b_block = B + B_offsets[batch_index] * esz;
            char* c_block = C + C_offsets[batch_index] * esz + (i0 * ldc + j0) * esz;

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
                // pack a
#if CV_NEON && CV_NEON_AARCH64
                fast_gemm_pack8_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_AVX
                fast_gemm_pack12_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_LASX
                fast_gemm_pack12_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_SIMD128
                fast_gemm_pack8_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#endif

                // pack b
#if CV_NEON && CV_NEON_AARCH64
                fast_gemm_pack12_f32(nc, kc, b_block + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#elif CV_AVX
                fast_gemm_pack8_f32(nc, kc, b_block + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#elif CV_LASX
                fast_gemm_pack16_f32(nc, kc, b_block + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#elif CV_SIMD128
                fast_gemm_pack12_f32(nc, kc, b_block + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
#endif

                // run kernel
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b, alpha, c_block, ldc_block, esz);
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    int total = batch * total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *packed_B, float beta, char *C, int ldc, int esz) {
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
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        const char *packed_b = packed_B;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            const int batch_index = static_cast<int>(tile_idx / total_tiles);
            const int m_tiles_index = static_cast<int>((tile_idx - batch_index * total_tiles) / n_tiles);
            const int n_tiles_index = static_cast<int>(tile_idx % n_tiles);

            int i0 = m_tiles_index * MC;
            int j0 = n_tiles_index * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            const char *a_block = A + A_offsets[batch_index] * esz;
            packed_b = packed_B + B_offsets[batch_index] * esz + j0 * K * esz;
            char* c_block = C + C_offsets[batch_index] * esz + (i0 * ldc + j0) * esz;

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
                // pack a
#if CV_NEON && CV_NEON_AARCH64
                fast_gemm_pack8_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_AVX
                fast_gemm_pack12_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_LASX
                fast_gemm_pack12_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#elif CV_SIMD128
                fast_gemm_pack8_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
#endif

                // run kernel
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b, alpha, c_block, ldc_block, esz);
                packed_b += _nc * kc;
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    int total = batch * total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

}} // cv::dnn

#undef FAST_GEMM_STORAGE
#undef FAST_GEMM_MAX_STACKBUF
#ifdef FAST_GEMM_F32_MC
#undef FAST_GEMM_F32_MC
#endif
#ifdef FAST_GEMM_F32_NC
#undef FAST_GEMM_F32_NC
#endif
#ifdef FAST_GEMM_F32_MR
#undef FAST_GEMM_F32_MR
#endif
#ifdef FAST_GEMM_F32_NR
#undef FAST_GEMM_F32_NR
#endif
#ifdef FAST_GEMM_F32_PACKED_STRIDE_K
#undef FAST_GEMM_F32_PACKED_STRIDE_K
#endif
#undef FAST_GEMM_IMPLEMENT_PACK
#undef FAST_GEMM_LOAD_TO_BUF_8
#undef FAST_GEMM_LOAD_TO_BUF_BORDERS_8
#undef FAST_GEMM_LOAD_TO_BUF_12
#undef FAST_GEMM_LOAD_TO_BUF_BORDERS_12
#undef FAST_GEMM_LOAD_TO_BUF_16
#undef FAST_GEMM_LOAD_TO_BUF_BORDERS_16
#undef FAST_GEMM_PACK_COPY
#undef FAST_GEMM_PACK_f32_8
#undef FAST_GEMM_PACK_f32_12
#undef FAST_GEMM_PACK_f32_16
