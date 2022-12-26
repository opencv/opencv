// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FAST_CONVOLUTION_SIMD_HPP
#define OPENCV_FAST_CONVOLUTION_SIMD_HPP

#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace dnn {

static void convBlockMR1NoSIMD(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                               const float minval, const float maxval, bool ifMinMaxAct, const int outLen)
{
    std::vector<float> cbuffer(outLen, 0);
    float* cbuf = cbuffer.data();
    for( int p = 0; p < np; p++ )
    {
        float ai = a[p];
        for( int j = 0; j < outLen; j++ )
            cbuf[j] += b[CONV_NR*p + j] * ai;
    }

    if (init_c)
    {
        for(int j = 0; j < outLen; j++)
        {
            c[j] += cbuf[j] + bias;
            if (ifMinMaxAct)
                c[j] = std::min(std::max(c[j], minval), maxval);
        }
    }
    else
    {
        for(int j = 0; j < outLen; j++)
        {
            c[j] = cbuf[j] + bias;
            if (ifMinMaxAct)
                c[j] = std::min(std::max(c[j], minval), maxval);
        }
    }
}

void convBlockMR1(int np, const float* a, const float* b, float *c, const float bias, bool init_c,
                  const float minval, const float maxval, bool ifMinMaxAct, const int outLen)
{
#if CV_SIMD128
    // The outLen represents the valid output value in CONV_NR length.
    // When outLen is very small, we use the no-SIMD branch.
    const int CONV_NRby3 = CONV_NR/3;
    if (outLen > CONV_NRby3)
    {
        v_float32x4 c0  = v_setall_f32(bias), c1 = c0, c2 = c0; // CONV_NR == 12
#if CONV_NR == 28 || CONV_NR == 24
        v_float32x4 c3 = c0, c4 = c0, c5 = c0;
#endif
#if CONV_NR == 28
        v_float32x4 c6 = c0;
#endif
        for (int p = 0; p < np; p++, a++, b += CONV_NR)
        {
            v_float32x4 a0 = v_setall_f32(a[0]);
            v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);
#if CONV_NR == 28 || CONV_NR == 24
            v_float32x4 b3 = v_load(b + 12), b4 = v_load(b + 16), b5 = v_load(b + 20);
#endif
#if CONV_NR == 28
            v_float32x4 b6 = v_load(b + 24);
#endif

            c0 = v_fma(b0, a0, c0);
            c1 = v_fma(b1, a0, c1);
            c2 = v_fma(b2, a0, c2);
#if CONV_NR == 28 || CONV_NR == 24
            c3 = v_fma(b3, a0, c3);
            c4 = v_fma(b4, a0, c4);
            c5 = v_fma(b5, a0, c5);
#endif
#if CONV_NR == 28
            c6 = v_fma(b6, a0, c6);
#endif
        }

        if (init_c)
        {
            c0 += v_load(c);
            c1 += v_load(c + 4);
            c2 += v_load(c + 8);
#if CONV_NR == 28 || CONV_NR == 24
            c3 += v_load(c + 12);
            c4 += v_load(c + 16);
            c5 += v_load(c + 20);
#endif
#if CONV_NR == 28
            c6  += v_load(c + 24);
#endif
        }

        if (ifMinMaxAct)
        {
           v_float32x4 vmax = v_setall_f32(maxval), vmin = v_setall_f32(minval);
           c0 = v_min(v_max(c0, vmin), vmax);
           c1 = v_min(v_max(c1, vmin), vmax);
           c2 = v_min(v_max(c2, vmin), vmax);
#if CONV_NR == 28 || CONV_NR == 24
           c3 = v_min(v_max(c3, vmin), vmax);
           c4 = v_min(v_max(c4, vmin), vmax);
           c5 = v_min(v_max(c5, vmin), vmax);
#endif
#if CONV_NR == 28
            c6 = v_min(v_max(c6, vmin), vmax);
#endif
        }

        v_store(c, c0);
        v_store(c + 4, c1);
        v_store(c + 8, c2);
#if CONV_NR == 28 || CONV_NR == 24
        v_store(c + 12, c3);
        v_store(c + 16, c4);
        v_store(c + 20, c5);
#endif
#if CONV_NR == 28
        v_store(c + 24, c6);
#endif
     }
     else
         convBlockMR1NoSIMD(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, outLen);
#else
    convBlockMR1NoSIMD(np, a, b, c, bias, init_c, minval, maxval, ifMinMaxAct, outLen);
#endif
}

#if CV_SIMD128
#if CONV_MR == 4 && CONV_NR == 24
static void convBlock4x24(int np, const float* a, const float* b, float* c, int ldc, bool init_c)
{
    v_float32x4 c0  = v_setzero_f32(), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0;
    v_float32x4 c6  = v_setzero_f32(), c7 = c6, c8 = c6, c9 = c6, c10 = c6, c11 = c6;
    v_float32x4 c12 = v_setzero_f32(), c13 = c12, c14 = c12, c15 = c12, c16 = c12, c17 = c12;
    v_float32x4 c18 = v_setzero_f32(), c19 = c18, c20 = c18, c21 = c18, c22 = c18, c23 = c18;

    for (int p = 0; p < np; p++, a += CONV_MR, b += CONV_NR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);
        v_float32x4 b3 = v_load(b + 12), b4 = v_load(b + 16), b5 = v_load(b + 20);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b1, a0, c1);
        c2 = v_fma(b2, a0, c2);
        c3 = v_fma(b3, a0, c3);
        c4 = v_fma(b4, a0, c4);
        c5 = v_fma(b5, a0, c5);

        a0  = v_setall_f32(a[1]);
        c6  = v_fma(b0, a0, c6);
        c7  = v_fma(b1, a0, c7);
        c8  = v_fma(b2, a0, c8);
        c9  = v_fma(b3, a0, c9);
        c10 = v_fma(b4, a0, c10);
        c11 = v_fma(b5, a0, c11);

        a0 = v_setall_f32(a[2]);
        c12 = v_fma(b0, a0, c12);
        c13 = v_fma(b1, a0, c13);
        c14 = v_fma(b2, a0, c14);
        c15 = v_fma(b3, a0, c15);
        c16 = v_fma(b4, a0, c16);
        c17 = v_fma(b5, a0, c17);

        a0 = v_setall_f32(a[3]);
        c18 = v_fma(b0, a0, c18);
        c19 = v_fma(b1, a0, c19);
        c20 = v_fma(b2, a0, c20);
        c21 = v_fma(b3, a0, c21);
        c22 = v_fma(b4, a0, c22);
        c23 = v_fma(b5, a0, c23);
    }

    if (!init_c)
    {
        c0 += v_load(c);
        c1 += v_load(c + 4);
        c2 += v_load(c + 8);
        c3 += v_load(c + 12);
        c4 += v_load(c + 16);
        c5 += v_load(c + 20);

        c6  += v_load(c + ldc);
        c7  += v_load(c + ldc + 4);
        c8  += v_load(c + ldc + 8);
        c9  += v_load(c + ldc + 12);
        c10 += v_load(c + ldc + 16);
        c11 += v_load(c + ldc + 20);

        c12 += v_load(c + ldc*2);
        c13 += v_load(c + ldc*2 + 4);
        c14 += v_load(c + ldc*2 + 8);
        c15 += v_load(c + ldc*2 + 12);
        c16 += v_load(c + ldc*2 + 16);
        c17 += v_load(c + ldc*2 + 20);

        c18 += v_load(c + ldc*3);
        c19 += v_load(c + ldc*3 + 4);
        c20 += v_load(c + ldc*3 + 8);
        c21 += v_load(c + ldc*3 + 12);
        c22 += v_load(c + ldc*3 + 16);
        c23 += v_load(c + ldc*3 + 20);
    }

    v_store(c, c0);
    v_store(c + 4, c1);
    v_store(c + 8, c2);
    v_store(c + 12, c3);
    v_store(c + 16, c4);
    v_store(c + 20, c5);

    v_store(c + ldc, c6);
    v_store(c + ldc + 4, c7);
    v_store(c + ldc + 8, c8);
    v_store(c + ldc + 12, c9);
    v_store(c + ldc + 16, c10);
    v_store(c + ldc + 20, c11);

    v_store(c + ldc * 2, c12);
    v_store(c + ldc * 2 + 4, c13);
    v_store(c + ldc * 2 + 8, c14);
    v_store(c + ldc * 2 + 12, c15);
    v_store(c + ldc * 2 + 16, c16);
    v_store(c + ldc * 2 + 20, c17);

    v_store(c + ldc * 3, c18);
    v_store(c + ldc * 3 + 4, c19);
    v_store(c + ldc * 3 + 8, c20);
    v_store(c + ldc * 3 + 12, c21);
    v_store(c + ldc * 3 + 16, c22);
    v_store(c + ldc * 3 + 20, c23);
}
#endif

static void convBlock4x8(int np, const float* a, const float* b, float* c, int ldc, bool init_c)
{
    CV_Assert(CONV_NR >= 4);
    v_float32x4 c0  = v_setzero_f32(), c1 = c0, c2 = c0, c3 = c0;
    v_float32x4 c4 = c0, c5 = c0, c6 = c0, c7 = c0;

    for (int p = 0; p < np; p++, a += CONV_MR, b += CONV_NR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 a1 = v_setall_f32(a[1]);
        v_float32x4 a2 = v_setall_f32(a[2]);
        v_float32x4 a3 = v_setall_f32(a[3]);

        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b1, a0, c1);

        c2 = v_fma(b0, a1, c2);
        c3 = v_fma(b1, a1, c3);

        c4 = v_fma(b0, a2, c4);
        c5 = v_fma(b1, a2, c5);

        c6  = v_fma(b0, a3, c6);
        c7  = v_fma(b1, a3, c7);
    }

    if (!init_c)
    {
        c0 += v_load(c);
        c1 += v_load(c + 4);

        c2  += v_load(c + ldc);
        c3  += v_load(c + ldc + 4);

        c4 += v_load(c + ldc*2);
        c5 += v_load(c + ldc*2 + 4);

        c6 += v_load(c + ldc*3);
        c7 += v_load(c + ldc*3 + 4);
    }

    v_store(c, c0);
    v_store(c + 4, c1);
    v_store(c + ldc, c2);
    v_store(c + ldc + 4, c3);
    v_store(c + ldc * 2, c4);
    v_store(c + ldc * 2 + 4, c5);
    v_store(c + ldc * 3, c6);
    v_store(c + ldc * 3 + 4, c7);
}

static void convBlock4x4(int np, const float* a, const float* b, float* c, int ldc, bool init_c)
{
    CV_Assert(CONV_NR >= 4);
    v_float32x4 c0  = v_setzero_f32(), c1 = c0, c2 = c0, c3 = c0;

    for (int p = 0; p < np; p++, a += CONV_MR, b += CONV_NR)
    {
        v_float32x4 a0 = v_setall_f32(a[0]);
        v_float32x4 a1 = v_setall_f32(a[1]);
        v_float32x4 a2 = v_setall_f32(a[2]);
        v_float32x4 a3 = v_setall_f32(a[3]);

        v_float32x4 b0 = v_load(b);

        c0 = v_fma(b0, a0, c0);
        c1 = v_fma(b0, a1, c1);
        c2 = v_fma(b0, a2, c2);
        c3 = v_fma(b0, a3, c3);
    }

    if (!init_c)
    {
        c0 += v_load(c);
        c1 += v_load(c + ldc);
        c2 += v_load(c + ldc*2);
        c3 += v_load(c + ldc*3);
    }

    v_store(c, c0);
    v_store(c + ldc, c1);
    v_store(c + ldc * 2, c2);
    v_store(c + ldc * 3, c3);
}
#endif

static void convBlockNoSIMD(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int outLen)
{
    std::vector<float> cbuffer(CONV_MR * outLen, 0);
    float* cbuf = cbuffer.data();
    for( int p = 0; p < np; p++ )
    {
        for( int i = 0; i < CONV_MR; i++ )
        {
            float ai = a[CONV_MR*p + i];
            for( int j = 0; j < outLen; j++ )
                cbuf[i * outLen+j] += b[CONV_NR*p + j] * ai;
        }
    }

    if (!init_c)
    {
        for(int i = 0; i < CONV_MR; i++)
        {
            for(int j = 0; j < outLen; j++)
                c[i*ldc + j] += cbuf[i*outLen + j];
        }
    }
    else
    {
        for(int i = 0; i < CONV_MR; i++)
        {
            for(int j = 0; j < outLen; j++)
                c[i*ldc + j] = cbuf[i*outLen + j];
        }
    }
}

void convBlock(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int outLen)
{
    // The possible outLen range is [24, 8~1].
#if CV_SIMD128
#if CONV_MR == 4 && CONV_NR == 24
    const int CONV_NRby3 = CONV_NR/3;
    if (outLen > CONV_NRby3)
    {
        convBlock4x24(np, a, b, c, ldc, init_c);
        return;
    }
#endif

    if (outLen <= 8 && outLen > 4)
    {
        convBlock4x8(np, a, b, c, ldc, init_c);
        return;
    }

    if (outLen <= 4 && outLen > 1)
    {
        convBlock4x4(np, a, b, c, ldc, init_c);
        return;
    }
    convBlockNoSIMD(np, a, b, c, ldc, init_c, outLen);
#else
    convBlockNoSIMD(np, a, b, c, ldc, init_c, outLen);
#endif
}
} // namespace dnn

namespace opt_NEON
{
#if CV_TRY_NEON
void convBlock_NEON(int np, const float* a, const float* b, float* c, int ldc, bool init_c)
{
#if CONV_MR == 4 && CONV_NR == 28  // AARCH64
    {
        float32x4_t c00 = vdupq_n_f32(0.f), c01 = c00, c02 = c00, c03 = c00, c04 = c00, c05 = c00, c06 = c00;
        float32x4_t c10 = vdupq_n_f32(0.f), c11 = c10, c12 = c10, c13 = c10, c14 = c10, c15 = c10, c16 = c10;
        float32x4_t c20 = vdupq_n_f32(0.f), c21 = c20, c22 = c20, c23 = c20, c24 = c20, c25 = c20, c26 = c20;
        float32x4_t c30 = vdupq_n_f32(0.f), c31 = c30, c32 = c30, c33 = c30, c34 = c30, c35 = c30, c36 = c30;

        for( int p = 0; p < np; p++, a += CONV_MR, b += CONV_NR )
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
#elif CONV_MR == 4 && CONV_NR == 12 // ARMv7
    {
        float32x4_t c0 = vdupq_n_f32(0.f), c1 = c0, c2 = c0;
        float32x4_t c3 = vdupq_n_f32(0.f), c4 = c3, c5 = c3;
        float32x4_t c6 = vdupq_n_f32(0.f), c7 = c6, c8 = c6;
        float32x4_t c9 = vdupq_n_f32(0.f), c10 = c9, c11 = c9;


        float32x2_t a0 = vdup_n_f32(0.0f), a1 = a0;
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);

        for (int p = 0; p < np; p++, a += CONV_MR, b += CONV_NR)
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
//#else
//#error "unsupported CONV_MR and/or CONV_NR in convBlock_NEON."
#endif
}
#endif
} // namespace opt_NEON

} // namespace cv
#endif //OPENCV_FAST_CONVOLUTION_SIMD_HPP
