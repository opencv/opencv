// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FAST_CONVOLUTION_SIMD_HPP
#define OPENCV_FAST_CONVOLUTION_SIMD_HPP

#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace dnn {

void convBlock(int k, const float *a, const float *b,
        float *c, int ldc, const float *bias,
        float minval, float maxval, bool ifActiv)
{
#if CV_SIMD128
#if FAST_CONV_MR == 4 && FAST_CONV_NR == 24
    {
        v_float32x4 c0 = v_setall_f32(bias[0]), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0;
        v_float32x4 c6 = v_setall_f32(bias[1]), c7 = c6, c8 = c6, c9 = c6, c10 = c6, c11 = c6;
        v_float32x4 c12 = v_setall_f32(bias[2]), c13 = c12, c14 = c12, c15 = c12, c16 = c12, c17 = c12;
        v_float32x4 c18 = v_setall_f32(bias[3]), c19 = c18, c20 = c18, c21 = c18, c22 = c18, c23 = c18;

        for (int p = 0; p < k; p++, a += FAST_CONV_MR, b += FAST_CONV_NR)
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

        if (ifActiv) {
            v_float32x4 vmin = v_setall_f32(minval), vmax = v_setall_f32(maxval);
            c0 = v_min(v_max(c0, vmin), vmax);
            c1 = v_min(v_max(c1, vmin), vmax);
            c2 = v_min(v_max(c2, vmin), vmax);
            c3 = v_min(v_max(c3, vmin), vmax);
            c4 = v_min(v_max(c4, vmin), vmax);
            c5 = v_min(v_max(c5, vmin), vmax);
            c6 = v_min(v_max(c6, vmin), vmax);
            c7 = v_min(v_max(c7, vmin), vmax);
            c8 = v_min(v_max(c8, vmin), vmax);
            c9 = v_min(v_max(c9, vmin), vmax);
            c10 = v_min(v_max(c10, vmin), vmax);
            c11 = v_min(v_max(c11, vmin), vmax);
            c12 = v_min(v_max(c12, vmin), vmax);
            c13 = v_min(v_max(c13, vmin), vmax);
            c14 = v_min(v_max(c14, vmin), vmax);
            c15 = v_min(v_max(c15, vmin), vmax);
            c16 = v_min(v_max(c16, vmin), vmax);
            c17 = v_min(v_max(c17, vmin), vmax);
            c18 = v_min(v_max(c18, vmin), vmax);
            c19 = v_min(v_max(c19, vmin), vmax);
            c20 = v_min(v_max(c20, vmin), vmax);
            c21 = v_min(v_max(c21, vmin), vmax);
            c22 = v_min(v_max(c22, vmin), vmax);
            c23 = v_min(v_max(c23, vmin), vmax);
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
#else
    for (int i = 0; i < FAST_CONV_MR; i++)
    {
        float beta = bias[i];
        for (int j = 0; j < FAST_CONV_NR; j++)
            c[i*ldc + j] = beta;
    }
    for (int p = 0; p < k; p++)
    {
        for (int i = 0; i < FAST_CONV_MR; i++)
        {
            float alpha = a[FAST_CONV_MR*p + i];
            for (int j = 0; j < FAST_CONV_NR; j++)
            {
                c[i*ldc+j] += b[FAST_CONV_NR*p + j]*alpha;
            }
        }
    }
    if (ifActiv)
    {
        for (int i = 0; i < FAST_CONV_MR; i++)
        {
            for (int j = 0; j < FAST_CONV_NR; j++)
            {
                float v = c[i*ldc + j];
                v = std::min(std::max(v, minval), maxval);
                c[i*ldc + j] = v;
            }
        }
    }
#endif
}
} // namespace dnn

namespace opt_NEON
{
#if CV_TRY_NEON
void convBlock_NEON(int k, const float *a, const float *b,
                float *c, int ldc, const float *bias,
                float minval, float maxval, bool ifActiv)
{
#if CV_NEON_AARCH64 && FAST_CONV_MR == 4 && FAST_CONV_NR == 28  // AARCH64
    {
        float32x4_t c0 = vdupq_n_f32(bias[0]), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c24 = c0;
        float32x4_t c6 = vdupq_n_f32(bias[1]), c7 = c6, c8 = c6, c9 = c6, c10 = c6, c11 = c6, c25 = c6;
        float32x4_t c12 = vdupq_n_f32(bias[2]), c13 = c12, c14 = c12, c15 = c12, c16 = c12, c17 = c12, c26 = c12;
        float32x4_t c18 = vdupq_n_f32(bias[3]), c19 = c18, c20 = c18, c21 = c18, c22 = c18, c23 = c18, c27 = c18;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);

        for (int p = 0; p < k; p++, a += FAST_CONV_MR)
        {
            a0 = vld1q_f32(a);
            b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
            b += 12;

            c0 = vfmaq_laneq_f32(c0, b0, a0, 0);
            c1 = vfmaq_laneq_f32(c1, b1, a0, 0);
            c2 = vfmaq_laneq_f32(c2, b2, a0, 0);
            c6 = vfmaq_laneq_f32(c6, b0, a0, 1);
            c7 = vfmaq_laneq_f32(c7, b1, a0, 1);
            c8 = vfmaq_laneq_f32(c8, b2, a0, 1);
            c12 = vfmaq_laneq_f32(c12, b0, a0, 2);
            c13 = vfmaq_laneq_f32(c13, b1, a0, 2);
            c14 = vfmaq_laneq_f32(c14, b2, a0, 2);
            c18 = vfmaq_laneq_f32(c18, b0, a0, 3);
            c19 = vfmaq_laneq_f32(c19, b1, a0, 3);
            c20 = vfmaq_laneq_f32(c20, b2, a0, 3);

            b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
            b += 12;

            c3 = vfmaq_laneq_f32(c3, b0, a0, 0);
            c4 = vfmaq_laneq_f32(c4, b1, a0, 0);
            c5 = vfmaq_laneq_f32(c5, b2, a0, 0);

            c9 = vfmaq_laneq_f32(c9, b0, a0, 1);
            c10 = vfmaq_laneq_f32(c10, b1, a0, 1);
            c11 = vfmaq_laneq_f32(c11, b2, a0, 1);

            c15 = vfmaq_laneq_f32(c15, b0, a0, 2);
            c16 = vfmaq_laneq_f32(c16, b1, a0, 2);
            c17 = vfmaq_laneq_f32(c17, b2, a0, 2);

            c21 = vfmaq_laneq_f32(c21, b0, a0, 3);

            b0 = vld1q_f32(b);
            b += 4;

            c22 = vfmaq_laneq_f32(c22, b1, a0, 3);
            c23 = vfmaq_laneq_f32(c23, b2, a0, 3);

            c24 = vfmaq_laneq_f32(c24, b0, a0, 0);
            c25 = vfmaq_laneq_f32(c25, b0, a0, 1);
            c26 = vfmaq_laneq_f32(c26, b0, a0, 2);
            c27 = vfmaq_laneq_f32(c27, b0, a0, 3);
        }

        if (ifActiv) {
            b0 = vdupq_n_f32(minval), b1 = vdupq_n_f32(maxval);
            c0 = vminq_f32(vmaxq_f32(c0, b0), b1);
            c1 = vminq_f32(vmaxq_f32(c1, b0), b1);
            c2 = vminq_f32(vmaxq_f32(c2, b0), b1);
            c3 = vminq_f32(vmaxq_f32(c3, b0), b1);
            c4 = vminq_f32(vmaxq_f32(c4, b0), b1);
            c5 = vminq_f32(vmaxq_f32(c5, b0), b1);
            c6 = vminq_f32(vmaxq_f32(c6, b0), b1);
            c7 = vminq_f32(vmaxq_f32(c7, b0), b1);
            c8 = vminq_f32(vmaxq_f32(c8, b0), b1);
            c9 = vminq_f32(vmaxq_f32(c9, b0), b1);
            c10 = vminq_f32(vmaxq_f32(c10, b0), b1);
            c11 = vminq_f32(vmaxq_f32(c11, b0), b1);
            c12 = vminq_f32(vmaxq_f32(c12, b0), b1);
            c13 = vminq_f32(vmaxq_f32(c13, b0), b1);
            c14 = vminq_f32(vmaxq_f32(c14, b0), b1);
            c15 = vminq_f32(vmaxq_f32(c15, b0), b1);
            c16 = vminq_f32(vmaxq_f32(c16, b0), b1);
            c17 = vminq_f32(vmaxq_f32(c17, b0), b1);
            c18 = vminq_f32(vmaxq_f32(c18, b0), b1);
            c19 = vminq_f32(vmaxq_f32(c19, b0), b1);
            c20 = vminq_f32(vmaxq_f32(c20, b0), b1);
            c21 = vminq_f32(vmaxq_f32(c21, b0), b1);
            c22 = vminq_f32(vmaxq_f32(c22, b0), b1);
            c23 = vminq_f32(vmaxq_f32(c23, b0), b1);
            c24 = vminq_f32(vmaxq_f32(c24, b0), b1);
            c25 = vminq_f32(vmaxq_f32(c25, b0), b1);
            c26 = vminq_f32(vmaxq_f32(c26, b0), b1);
            c27 = vminq_f32(vmaxq_f32(c27, b0), b1);
        }
        vst1q_f32(c, c0);
        vst1q_f32(c + 4, c1);
        vst1q_f32(c + 8, c2);
        vst1q_f32(c + 12, c3);
        vst1q_f32(c + 16, c4);
        vst1q_f32(c + 20, c5);
        vst1q_f32(c + 24, c24);

        vst1q_f32(c + ldc, c6);
        vst1q_f32(c + ldc + 4, c7);
        vst1q_f32(c + ldc + 8, c8);
        vst1q_f32(c + ldc + 12, c9);
        vst1q_f32(c + ldc + 16, c10);
        vst1q_f32(c + ldc + 20, c11);
        vst1q_f32(c + ldc + 24, c25);

        vst1q_f32(c + ldc * 2, c12);
        vst1q_f32(c + ldc * 2 + 4, c13);
        vst1q_f32(c + ldc * 2 + 8, c14);
        vst1q_f32(c + ldc * 2 + 12, c15);
        vst1q_f32(c + ldc * 2 + 16, c16);
        vst1q_f32(c + ldc * 2 + 20, c17);
        vst1q_f32(c + ldc * 2 + 24, c26);

        vst1q_f32(c + ldc * 3, c18);
        vst1q_f32(c + ldc * 3 + 4, c19);
        vst1q_f32(c + ldc * 3 + 8, c20);
        vst1q_f32(c + ldc * 3 + 12, c21);
        vst1q_f32(c + ldc * 3 + 16, c22);
        vst1q_f32(c + ldc * 3 + 20, c23);
        vst1q_f32(c + ldc * 3 + 24, c27);
    }
#elif (!defined(CV_NEON_AARCH64) || !CV_NEON_AARCH64) && FAST_CONV_MR == 4 && FAST_CONV_NR == 12 // ARMv7
    {
        float32x4_t c0 = vdupq_n_f32(bias[0]), c1 = c0, c2 = c0;
        float32x4_t c3 = vdupq_n_f32(bias[1]), c4 = c3, c5 = c3;
        float32x4_t c6 = vdupq_n_f32(bias[2]), c7 = c6, c8 = c6;
        float32x4_t c9 = vdupq_n_f32(bias[3]), c10 = c9, c11 = c9;

        float32x2_t a0 = vdup_n_f32(0.0f), a1 = a0;
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f), b2 = vdupq_n_f32(0.0f);

        for (int p = 0; p < k; p++, a += FAST_CONV_MR, b += FAST_CONV_NR)
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

        if (ifActiv)
        {
            b0 = vdupq_n_f32(minval), b1 = vdupq_n_f32(maxval);
            c0 = vminq_f32(vmaxq_f32(c0, b0), b1);
            c1 = vminq_f32(vmaxq_f32(c1, b0), b1);
            c2 = vminq_f32(vmaxq_f32(c2, b0), b1);
            c3 = vminq_f32(vmaxq_f32(c3, b0), b1);
            c4 = vminq_f32(vmaxq_f32(c4, b0), b1);
            c5 = vminq_f32(vmaxq_f32(c5, b0), b1);
            c6 = vminq_f32(vmaxq_f32(c6, b0), b1);
            c7 = vminq_f32(vmaxq_f32(c7, b0), b1);
            c8 = vminq_f32(vmaxq_f32(c8, b0), b1);
            c9 = vminq_f32(vmaxq_f32(c9, b0), b1);
            c10 = vminq_f32(vmaxq_f32(c10, b0), b1);
            c11 = vminq_f32(vmaxq_f32(c11, b0), b1);
        }
        vst1q_f32(c, c0); vst1q_f32(c+4, c1); vst1q_f32(c+8, c2);
        vst1q_f32(c + ldc, c3); vst1q_f32(c + ldc + 4, c4); vst1q_f32(c + ldc + 8, c5);
        vst1q_f32(c + ldc*2, c6); vst1q_f32(c + ldc*2 + 4, c7); vst1q_f32(c + ldc*2 + 8, c8);
        vst1q_f32(c + ldc*3, c9); vst1q_f32(c + ldc*3 + 4, c10); vst1q_f32(c + ldc*3 + 8, c11);
    }
#else
#error "unsupported FAST_CONV_MR and/or FAST_CONV_NR in convBlock_NEON."
#endif
}
#endif
} // namespace opt_NEON

} // namespace cv
#endif //OPENCV_FAST_CONVOLUTION_SIMD_HPP
