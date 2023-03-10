// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
    CV_Assert(convMR == 4 && convNR == 24);
    __m256 c00 = _mm256_set1_ps(0.f), c01 = c00, c02 = c00;
    __m256 c10 = c00, c11 = c00, c12 = c00;
    __m256 c20 = c00, c21 = c00, c22 = c00;
    __m256 c30 = c00, c31 = c00, c32 = c00;

    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();

    for (int p = 0; p < np; p++, a += convMR, b += convNR)
    {
        a0 = _mm256_set1_ps(a[0]), a1 = _mm256_set1_ps(a[1]);
        b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8), b2 = _mm256_load_ps(b + 16);

        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);
        c02 = _mm256_fmadd_ps(b2, a0, c02);

        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);
        c12 = _mm256_fmadd_ps(b2, a1, c12);

        a0 = _mm256_set1_ps(a[2]), a1 = _mm256_set1_ps(a[3]);

        c20 = _mm256_fmadd_ps(b0, a0, c20);
        c21 = _mm256_fmadd_ps(b1, a0, c21);
        c22 = _mm256_fmadd_ps(b2, a0, c22);

        c30 = _mm256_fmadd_ps(b0, a1, c30);
        c31 = _mm256_fmadd_ps(b1, a1, c31);
        c32 = _mm256_fmadd_ps(b2, a1, c32);
    }

    if (!init_c)
    {
        c00 = _mm256_add_ps(c00, _mm256_load_ps(c));
        c01 = _mm256_add_ps(c01, _mm256_load_ps(c + 8));
        c02 = _mm256_add_ps(c02, _mm256_load_ps(c + 16));

        c10 = _mm256_add_ps(c10, _mm256_load_ps(c + ldc));
        c11 = _mm256_add_ps(c11, _mm256_load_ps(c + ldc + 8));
        c12 = _mm256_add_ps(c12, _mm256_load_ps(c + ldc + 16));

        c20 = _mm256_add_ps(c20, _mm256_load_ps(c + ldc*2));
        c21 = _mm256_add_ps(c21, _mm256_load_ps(c + ldc*2 + 8));
        c22 = _mm256_add_ps(c22, _mm256_load_ps(c + ldc*2 + 16));

        c30 = _mm256_add_ps(c30, _mm256_load_ps(c + ldc*3));
        c31 = _mm256_add_ps(c31, _mm256_load_ps(c + ldc*3 + 8));
        c32 = _mm256_add_ps(c32, _mm256_load_ps(c + ldc*3 + 16));
    }

    _mm256_storeu_ps(c, c00), _mm256_storeu_ps(c+8, c01), _mm256_storeu_ps(c+16, c02);
    _mm256_storeu_ps(c + ldc, c10), _mm256_storeu_ps(c + ldc + 8, c11), _mm256_storeu_ps(c + ldc + 16, c12);
    _mm256_storeu_ps(c + ldc*2, c20), _mm256_storeu_ps(c + ldc*2 + 8, c21), _mm256_storeu_ps(c + ldc*2 + 16, c22);
    _mm256_storeu_ps(c + ldc*3, c30), _mm256_storeu_ps(c + ldc*3 + 8, c31), _mm256_storeu_ps(c + ldc*3 + 16, c32);
    _mm256_zeroupper();
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

// NEON code work around.
namespace opt_NEON
{
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_NEON

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
#if CV_NEON_AARCH64
    if (convMR == 4 && convNR == 28) // AARCH64
    {
        float32x4_t c00 = vdupq_n_f32(0.f), c01 = c00, c02 = c00, c03 = c00, c04 = c00, c05 = c00, c06 = c00;
        float32x4_t c10 = vdupq_n_f32(0.f), c11 = c10, c12 = c10, c13 = c10, c14 = c10, c15 = c10, c16 = c10;
        float32x4_t c20 = vdupq_n_f32(0.f), c21 = c20, c22 = c20, c23 = c20, c24 = c20, c25 = c20, c26 = c20;
        float32x4_t c30 = vdupq_n_f32(0.f), c31 = c30, c32 = c30, c33 = c30, c34 = c30, c35 = c30, c36 = c30;

        for( int p = 0; p < np; p++, a += convMR, b += convNR )
        {
            float32x4_t a0 = vld1q_f32(a), b0, b1, b2;
            b0 = vld1q_f32(b); b1 = vld1q_f32(b + 4); b2 = vld1q_f32(b + 8);

            c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
            c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
            c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
            c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
            c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
            c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
            c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
            c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
            c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
            c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
            c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
            c32 = vfmaq_laneq_f32(c32, b2, a0, 3);

            b0 = vld1q_f32(b + 12); b1 = vld1q_f32(b + 16); b2 = vld1q_f32(b + 20);

            c03 = vfmaq_laneq_f32(c03, b0, a0, 0);
            c04 = vfmaq_laneq_f32(c04, b1, a0, 0);
            c05 = vfmaq_laneq_f32(c05, b2, a0, 0);
            c13 = vfmaq_laneq_f32(c13, b0, a0, 1);
            c14 = vfmaq_laneq_f32(c14, b1, a0, 1);
            c15 = vfmaq_laneq_f32(c15, b2, a0, 1);
            c23 = vfmaq_laneq_f32(c23, b0, a0, 2);
            c24 = vfmaq_laneq_f32(c24, b1, a0, 2);
            c25 = vfmaq_laneq_f32(c25, b2, a0, 2);
            c33 = vfmaq_laneq_f32(c33, b0, a0, 3);
            c34 = vfmaq_laneq_f32(c34, b1, a0, 3);
            c35 = vfmaq_laneq_f32(c35, b2, a0, 3);

            b0 = vld1q_f32(b + 24);
            c06 = vfmaq_laneq_f32(c06, b0, a0, 0);
            c16 = vfmaq_laneq_f32(c16, b0, a0, 1);
            c26 = vfmaq_laneq_f32(c26, b0, a0, 2);
            c36 = vfmaq_laneq_f32(c36, b0, a0, 3);
        }

        if (!init_c)
        {
            c00 = vaddq_f32(c00, vld1q_f32(c));
            c01 = vaddq_f32(c01, vld1q_f32(c + 4));
            c02 = vaddq_f32(c02, vld1q_f32(c + 8));
            c03 = vaddq_f32(c03, vld1q_f32(c + 12));
            c04 = vaddq_f32(c04, vld1q_f32(c + 16));
            c05 = vaddq_f32(c05, vld1q_f32(c + 20));
            c06 = vaddq_f32(c06, vld1q_f32(c + 24));

            c10 = vaddq_f32(c10, vld1q_f32(c + ldc));
            c11 = vaddq_f32(c11, vld1q_f32(c + ldc + 4));
            c12 = vaddq_f32(c12, vld1q_f32(c + ldc + 8));
            c13 = vaddq_f32(c13, vld1q_f32(c + ldc + 12));
            c14 = vaddq_f32(c14, vld1q_f32(c + ldc + 16));
            c15 = vaddq_f32(c15, vld1q_f32(c + ldc + 20));
            c16 = vaddq_f32(c16, vld1q_f32(c + ldc + 24));

            c20 = vaddq_f32(c20, vld1q_f32(c + ldc*2));
            c21 = vaddq_f32(c21, vld1q_f32(c + ldc*2 + 4));
            c22 = vaddq_f32(c22, vld1q_f32(c + ldc*2 + 8));
            c23 = vaddq_f32(c23, vld1q_f32(c + ldc*2 + 12));
            c24 = vaddq_f32(c24, vld1q_f32(c + ldc*2 + 16));
            c25 = vaddq_f32(c25, vld1q_f32(c + ldc*2 + 20));
            c26 = vaddq_f32(c26, vld1q_f32(c + ldc*2 + 24));

            c30 = vaddq_f32(c30, vld1q_f32(c + ldc*3));
            c31 = vaddq_f32(c31, vld1q_f32(c + ldc*3 + 4));
            c32 = vaddq_f32(c32, vld1q_f32(c + ldc*3 + 8));
            c33 = vaddq_f32(c33, vld1q_f32(c + ldc*3 + 12));
            c34 = vaddq_f32(c34, vld1q_f32(c + ldc*3 + 16));
            c35 = vaddq_f32(c35, vld1q_f32(c + ldc*3 + 20));
            c36 = vaddq_f32(c36, vld1q_f32(c + ldc*3 + 24));
        }

        vst1q_f32(c, c00); vst1q_f32(c+4, c01);
        vst1q_f32(c+8, c02); vst1q_f32(c+12, c03);
        vst1q_f32(c+16, c04); vst1q_f32(c+20, c05);
        vst1q_f32(c+24, c06);

        vst1q_f32(c+ldc, c10); vst1q_f32(c+ldc+4, c11);
        vst1q_f32(c+ldc+8, c12); vst1q_f32(c+ldc+12, c13);
        vst1q_f32(c+ldc+16, c14); vst1q_f32(c+ldc+20, c15);
        vst1q_f32(c+ldc+24, c16);

        vst1q_f32(c+ldc*2, c20); vst1q_f32(c+ldc*2+4, c21);
        vst1q_f32(c+ldc*2+8, c22); vst1q_f32(c+ldc*2+12, c23);
        vst1q_f32(c+ldc*2+16, c24); vst1q_f32(c+ldc*2+20, c25);
        vst1q_f32(c+ldc*2+24, c26);

        vst1q_f32(c+ldc*3, c30); vst1q_f32(c+ldc*3+4, c31);
        vst1q_f32(c+ldc*3+8, c32); vst1q_f32(c+ldc*3+12, c33);
        vst1q_f32(c+ldc*3+16, c34); vst1q_f32(c+ldc*3+20, c35);
        vst1q_f32(c+ldc*3+24, c36);
    }
    else
#endif
    if (convMR == 4 && convNR == 12) // ARMv7
    {
        float32x4_t c0 = vdupq_n_f32(0.f), c1 = c0, c2 = c0;
        float32x4_t c3 = vdupq_n_f32(0.f), c4 = c3, c5 = c3;
        float32x4_t c6 = vdupq_n_f32(0.f), c7 = c6, c8 = c6;
        float32x4_t c9 = vdupq_n_f32(0.f), c10 = c9, c11 = c9;

        float32x2_t a0 = vdup_n_f32(0.0f), a1 = a0;
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);

        for (int p = 0; p < np; p++, a += convMR, b += convNR)
        {
            a0 = vld1_f32(a), a1 = vld1_f32(a+2);
            b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);

            c0 = vmlaq_lane_f32(c0, b0, a0, 0);
            c1 = vmlaq_lane_f32(c1, b1, a0, 0);
            c2 = vmlaq_lane_f32(c2, b2, a0, 0);

            c3 = vmlaq_lane_f32(c3, b0, a0, 1);
            c4 = vmlaq_lane_f32(c4, b1, a0, 1);
            c5 = vmlaq_lane_f32(c5, b2, a0, 1);

            c6 = vmlaq_lane_f32(c6, b0, a1, 0);
            c7 = vmlaq_lane_f32(c7, b1, a1, 0);
            c8 = vmlaq_lane_f32(c8, b2, a1, 0);

            c9  = vmlaq_lane_f32(c9 , b0, a1, 1);
            c10 = vmlaq_lane_f32(c10, b1, a1, 1);
            c11 = vmlaq_lane_f32(c11, b2, a1, 1);
        }

        if (!init_c)
        {
            c0 = vaddq_f32(c0, vld1q_f32(c));
            c1 = vaddq_f32(c1, vld1q_f32(c + 4));
            c2 = vaddq_f32(c2, vld1q_f32(c + 8));

            c3 = vaddq_f32(c3, vld1q_f32(c + ldc));
            c4 = vaddq_f32(c4, vld1q_f32(c + ldc + 4));
            c5 = vaddq_f32(c5, vld1q_f32(c + ldc + 8));

            c6 = vaddq_f32(c6, vld1q_f32(c + ldc * 2));
            c7 = vaddq_f32(c7, vld1q_f32(c + ldc * 2 + 4));
            c8 = vaddq_f32(c8, vld1q_f32(c + ldc * 2 + 8));

            c9  = vaddq_f32(c9 , vld1q_f32(c + ldc * 3));
            c10 = vaddq_f32(c10, vld1q_f32(c + ldc * 3 + 4));
            c11 = vaddq_f32(c11, vld1q_f32(c + ldc * 3 + 8));
        }

        vst1q_f32(c, c0), vst1q_f32(c+4, c1), vst1q_f32(c+8, c2);
        vst1q_f32(c + ldc, c3), vst1q_f32(c + ldc + 4, c4), vst1q_f32(c + ldc + 8, c5);
        vst1q_f32(c + ldc*2, c6), vst1q_f32(c + ldc*2 + 4, c7), vst1q_f32(c + ldc*2 + 8, c8);
        vst1q_f32(c + ldc*3, c9), vst1q_f32(c + ldc*3 + 4, c10), vst1q_f32(c + ldc*3 + 8, c11);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported convMR and/or convNR in opt_NEON::convBlock");
}

#endif
}
}} // namespace cv::dnn
