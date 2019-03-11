/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2012-2015, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#include <cfloat>
#include <cmath>

#include "common.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

#define FASTATAN2CONST(scale) \
        f32 P1((f32)( 0.9997878412794807  * (180.0 / M_PI) * scale)), \
        P3((f32)(-0.3258083974640975  * (180.0 / M_PI) * scale)), \
        P5((f32)( 0.1555786518463281  * (180.0 / M_PI) * scale)), \
        P7((f32)(-0.04432655554792128 * (180.0 / M_PI) * scale)), \
         A_90((f32)(90.f * scale)), \
        A_180((f32)(180.f * scale)), \
        A_360((f32)(360.f * scale)); \
        float32x4_t eps(vdupq_n_f32((float)DBL_EPSILON)), \
         _90(vdupq_n_f32(A_90)), \
        _180(vdupq_n_f32(A_180)), \
        _360(vdupq_n_f32(A_360)), \
           z(vdupq_n_f32(0.0f)), \
        p1(vdupq_n_f32(P1)), \
        p3(vdupq_n_f32(P3)), \
        p5(vdupq_n_f32(P5)), \
        p7(vdupq_n_f32(P7));

#define FASTATAN2SCALAR(y, x, a) \
    { \
        f32 ax = std::abs(x), ay = std::abs(y); \
        f32 c, c2; \
        if (ax >= ay) \
        { \
            c = ay / (ax + (float)DBL_EPSILON); \
            c2 = c * c; \
            a = (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c; \
        } \
        else \
        { \
            c = ax / (ay + (float)DBL_EPSILON); \
            c2 = c * c; \
            a = A_90 - (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c; \
        } \
        if (x < 0) \
            a = A_180 - a; \
        if (y < 0) \
            a = A_360 - a; \
    }

#define FASTATAN2VECTOR(v_y, v_x, a) \
    { \
        float32x4_t ax = vabsq_f32(v_x), ay = vabsq_f32(v_y); \
        float32x4_t tmin = vminq_f32(ax, ay), tmax = vmaxq_f32(ax, ay); \
        float32x4_t c = vmulq_f32(tmin, internal::vrecpq_f32(vaddq_f32(tmax, eps))); \
        float32x4_t c2 = vmulq_f32(c, c); \
        a = vmulq_f32(c2, p7); \
 \
        a = vmulq_f32(vaddq_f32(a, p5), c2); \
        a = vmulq_f32(vaddq_f32(a, p3), c2); \
        a = vmulq_f32(vaddq_f32(a, p1), c); \
 \
        a = vbslq_f32(vcgeq_f32(ax, ay), a, vsubq_f32(_90, a)); \
        a = vbslq_f32(vcltq_f32(v_x, z), vsubq_f32(_180, a), a); \
        a = vbslq_f32(vcltq_f32(v_y, z), vsubq_f32(_360, a), a); \
 \
    }

} // namespace

#endif

void phase(const Size2D &size,
           const s16 * src0Base, ptrdiff_t src0Stride,
           const s16 * src1Base, ptrdiff_t src1Stride,
           u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    FASTATAN2CONST(256.0f / 360.0f)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    float32x4_t v_05 = vdupq_n_f32(0.5f);

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const s16 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src0 + j);
            internal::prefetch(src1 + j);

            int16x8_t v_src00 = vld1q_s16(src0 + j), v_src01 = vld1q_s16(src0 + j + 8);
            int16x8_t v_src10 = vld1q_s16(src1 + j), v_src11 = vld1q_s16(src1 + j + 8);

            // 0
            float32x4_t v_src0_p = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src00)));
            float32x4_t v_src1_p = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src10)));
            float32x4_t v_dst32f0;
            FASTATAN2VECTOR(v_src1_p, v_src0_p, v_dst32f0)

            v_src0_p = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src00)));
            v_src1_p = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src10)));
            float32x4_t v_dst32f1;
            FASTATAN2VECTOR(v_src1_p, v_src0_p, v_dst32f1)

            uint16x8_t v_dst16s0 = vcombine_u16(vmovn_u32(vcvtq_u32_f32(vaddq_f32(v_dst32f0, v_05))),
                                                vmovn_u32(vcvtq_u32_f32(vaddq_f32(v_dst32f1, v_05))));

            // 1
            v_src0_p = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src01)));
            v_src1_p = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src11)));
            FASTATAN2VECTOR(v_src1_p, v_src0_p, v_dst32f0)

            v_src0_p = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src01)));
            v_src1_p = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src11)));
            FASTATAN2VECTOR(v_src1_p, v_src0_p, v_dst32f1)

            uint16x8_t v_dst16s1 = vcombine_u16(vmovn_u32(vcvtq_u32_f32(vaddq_f32(v_dst32f0, v_05))),
                                                vmovn_u32(vcvtq_u32_f32(vaddq_f32(v_dst32f1, v_05))));

            vst1q_u8(dst + j, vcombine_u8(vmovn_u16(v_dst16s0),
                                          vmovn_u16(v_dst16s1)));
        }
        for (; j < roiw8; j += 8)
        {
            int16x8_t v_src0 = vld1q_s16(src0 + j);
            int16x8_t v_src1 = vld1q_s16(src1 + j);

            float32x4_t v_src0_p = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src0)));
            float32x4_t v_src1_p = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1)));
            float32x4_t v_dst32f0;
            FASTATAN2VECTOR(v_src1_p, v_src0_p, v_dst32f0)

            v_src0_p = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src0)));
            v_src1_p = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1)));
            float32x4_t v_dst32f1;
            FASTATAN2VECTOR(v_src1_p, v_src0_p, v_dst32f1)

            uint16x8_t v_dst = vcombine_u16(vmovn_u32(vcvtq_u32_f32(vaddq_f32(v_dst32f0, v_05))),
                                            vmovn_u32(vcvtq_u32_f32(vaddq_f32(v_dst32f1, v_05))));

            vst1_u8(dst + j, vmovn_u16(v_dst));
        }

        for (; j < size.width; j++)
        {
            f32 x = src0[j], y = src1[j];
            f32 a;
            FASTATAN2SCALAR(y, x, a)
            dst[j] = (u8)(s32)floor(a + 0.5f);
        }
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void phase(const Size2D &size,
           const f32 * src0Base, ptrdiff_t src0Stride,
           const f32 * src1Base, ptrdiff_t src1Stride,
           f32 * dstBase, ptrdiff_t dstStride,
           f32 scale)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    FASTATAN2CONST(scale)
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const f32 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const f32 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src0 + j);
            internal::prefetch(src1 + j);

            float32x4_t v_src00 = vld1q_f32(src0 + j), v_src01 = vld1q_f32(src0 + j + 4);
            float32x4_t v_src10 = vld1q_f32(src1 + j), v_src11 = vld1q_f32(src1 + j + 4);

            float32x4_t v_dst32f;
            // 0
            FASTATAN2VECTOR(v_src10, v_src00, v_dst32f)
            vst1q_f32(dst + j,     v_dst32f);
            // 1
            FASTATAN2VECTOR(v_src11, v_src01, v_dst32f)
            vst1q_f32(dst + j + 4, v_dst32f);
        }
        if(j + 4 <= size.width)
        {
            float32x4_t v_src0 = vld1q_f32(src0 + j);
            float32x4_t v_src1 = vld1q_f32(src1 + j);

            float32x4_t v_dst32f;
            FASTATAN2VECTOR(v_src1, v_src0, v_dst32f)
            vst1q_f32(dst + j, v_dst32f);
            j += 4;
        }

        for (; j < size.width; j++)
        {
            f32 a;
            FASTATAN2SCALAR(src1[j], src0[j], a)
            dst[j] = a;
        }
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)scale;
#endif
}

} // namespace CAROTENE_NS
