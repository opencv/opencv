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

namespace cv { namespace dnn {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void fast_gemm_micro_kernel_f32(int, const char *, const char *, char *, int, const void *, int, int);

// AVX and AVX2 (16 x 256-bit registers)
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround for FMA
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

void fast_gemm12x8_f32(int k, const char *a_, const char *b_,
                       char *c_, int ldc, const void *palpha, int mr, int nr) {
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
           s100 = _mm256_setzero_ps(),
           s110 = _mm256_setzero_ps();
        for (int p = 0; p < k; p++, a += mr, b += nr) {
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

void fast_gemm_micro_kernel_f32(int k, const char *a_, const char *b_,
                                char *c_, int ldc, const void *palpha, int mr, int nr) {
    fast_gemm12x8_f32(k, a_, b_, c_, ldc, palpha, mr, nr);
}

#endif // CV_AVX

CV_CPU_OPTIMIZATION_NAMESPACE_END

namespace opt_NEON_AARCH64 {

// NEON AARCH64 (32 128-bit registers)
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_NEON && CV_NEON_AARCH64

void fast_gemm8x12_f32(int k, const char *a_, const char *b_,
                       char *c_, int ldc, const void* palpha, int mr, int nr) {
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

    for(int p = 0; p < k; p++, a += mr, b += nr)
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

}

}} // cv::dnn
