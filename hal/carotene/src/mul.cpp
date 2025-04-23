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
 * Copyright (C) 2014-2016, NVIDIA Corporation, all rights reserved.
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

#include "common.hpp"
#include "vtransform.hpp"

#include <cstring>
#include <cfloat>
#include <cmath>
#include <limits>

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

bool isIntegerScale(f32 scale)
{
    return std::fabs(scale - static_cast<s32>(scale)) < FLT_EPSILON;
}

template <s32 shift>
void mulu8(const Size2D &size,
           const u8 * src0Base, ptrdiff_t src0Stride,
           const u8 * src1Base, ptrdiff_t src1Stride,
           u8 * dstBase, ptrdiff_t dstStride,
           CONVERT_POLICY cpolicy)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                v_dst0 = vshrq_n_u16(v_dst0, shift);
                v_dst1 = vshrq_n_u16(v_dst1, shift);

                vst1q_u8(dst + j, vcombine_u8(vqmovn_u16(v_dst0), vqmovn_u16(v_dst1)));
            }
            for (; j < roiw8; j += 8)
            {
                uint16x8_t v_dst = vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j));
                vst1_u8(dst + j, vqmovn_u16(vshrq_n_u16(v_dst, shift)));
            }

            for (; j < size.width; j++)
            {
                u16 val = (u16)src0[j] * (u16)src1[j];
                dst[j] = internal::saturate_cast<u8>(val >> shift);
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                v_dst0 = vshrq_n_u16(v_dst0, shift);
                v_dst1 = vshrq_n_u16(v_dst1, shift);

                vst1q_u8(dst + j, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
            }
            for (; j < roiw8; j += 8)
            {
                uint16x8_t v_dst = vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j));
                vst1_u8(dst + j, vmovn_u16(vshrq_n_u16(v_dst, shift)));
            }

            for (; j < size.width; j++)
            {
                u16 val = (u16)src0[j] * (u16)src1[j];
                dst[j] = (u8)(val >> shift);
            }
        }
    }
}

template <s32 shift>
void muls16(const Size2D &size,
            const u8 * src0Base, ptrdiff_t src0Stride,
            const u8 * src1Base, ptrdiff_t src1Stride,
            s16 * dstBase, ptrdiff_t dstStride,
            CONVERT_POLICY cpolicy)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    uint16x8_t v_32767 = vdupq_n_u16(0x7FFF);

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                v_dst0 = vshrq_n_u16(v_dst0, shift);
                v_dst1 = vshrq_n_u16(v_dst1, shift);

                vst1q_s16(dst + j, vreinterpretq_s16_u16(vminq_u16(v_32767, v_dst0)));
                vst1q_s16(dst + j + 8, vreinterpretq_s16_u16(vminq_u16(v_32767, v_dst1)));
            }
            for (; j < roiw8; j += 8)
            {
                uint16x8_t v_dst = vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j));
                v_dst = vshrq_n_u16(v_dst, shift);
                vst1q_s16(dst + j, vreinterpretq_s16_u16(vminq_u16(v_32767, v_dst)));
            }

            for (; j < size.width; j++)
            {
                u16 val = (u16)src0[j] * (u16)src1[j];
                dst[j] = internal::saturate_cast<s16>(val >> shift);
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                v_dst0 = vshrq_n_u16(v_dst0, shift);
                v_dst1 = vshrq_n_u16(v_dst1, shift);

                vst1q_s16(dst + j, vreinterpretq_s16_u16(v_dst0));
                vst1q_s16(dst + j + 8, vreinterpretq_s16_u16(v_dst1));
            }
            for (; j < roiw8; j += 8)
            {
                uint16x8_t v_dst = vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j));
                v_dst = vshrq_n_u16(v_dst, shift);
                vst1q_s16(dst + j, vreinterpretq_s16_u16(v_dst));
            }

            for (; j < size.width; j++)
            {
                u16 val = (u16)src0[j] * (u16)src1[j];
                dst[j] = (s16)(val >> shift);
            }
        }
    }
}

typedef void (* mulFuncu8)(const Size2D &size,
                           const u8 * src0Base, ptrdiff_t src0Stride,
                           const u8 * src1Base, ptrdiff_t src1Stride,
                           u8 * dstBase, ptrdiff_t dstStride,
                           CONVERT_POLICY cpolicy);

typedef void (* mulFuncs16)(const Size2D &size,
                            const u8 * src0Base, ptrdiff_t src0Stride,
                            const u8 * src1Base, ptrdiff_t src1Stride,
                            s16 * dstBase, ptrdiff_t dstStride,
                            CONVERT_POLICY cpolicy);

} // namespace

#endif

void mul(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         u8 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    if ((scale * UCHAR_MAX * UCHAR_MAX) < 1.0f)
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            u8 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(u8) * size.width);
        }
        return;
    }

    s32 iscale = static_cast<s32>(scale), exp = 0;
    f32 significand = frexp(scale, &exp);
    bool is_integer_scale = isIntegerScale(scale),
         is_power_of_2 = (significand == 0.5f) && (exp <= 0);
    exp = -exp + 1;

    if (is_power_of_2)
    {
        static const mulFuncu8 funcs[16] =
        {
            NULL,
            mulu8<1>,
            mulu8<2>,
            mulu8<3>,
            mulu8<4>,
            mulu8<5>,
            mulu8<6>,
            mulu8<7>,
            mulu8<8>,
            mulu8<9>,
            mulu8<10>,
            mulu8<11>,
            mulu8<12>,
            mulu8<13>,
            mulu8<14>,
            mulu8<15>
        };

        mulFuncu8 func = funcs[exp];

        func(size,
             src0Base, src0Stride,
             src1Base, src1Stride,
             dstBase, dstStride,
             cpolicy);

        return;
    }

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                    uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                    vst1q_u8(dst + j, vcombine_u8(vqmovn_u16(v_dst0), vqmovn_u16(v_dst1)));
                }
                for (; j < roiw8; j += 8)
                {
                    vst1_u8(dst + j, vqmovn_u16(vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j))));
                }

                for (; j < size.width; j++)
                {
                    u16 val = (u16)src0[j] * (u16)src1[j];
                    dst[j] = internal::saturate_cast<u8>(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);

                    uint8x16_t v_src0 = vld1q_u8(src0 + j);
                    uint8x16_t v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
                    uint16x8_t v_src1_p = vmovl_u8(vget_low_u8(v_src1));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    v_src0_p = vmovl_u8(vget_high_u8(v_src0));
                    v_src1_p = vmovl_u8(vget_high_u8(v_src1));
                    float32x4_t v_dst2f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    float32x4_t v_dst3f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    uint16x8_t v_dst0u = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                                      vqmovn_u32(vcvtq_u32_f32(v_dst1f)));
                    uint16x8_t v_dst1u = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(v_dst2f)),
                                                      vqmovn_u32(vcvtq_u32_f32(v_dst3f)));
                    vst1q_u8(dst + j, vcombine_u8(vqmovn_u16(v_dst0u), vqmovn_u16(v_dst1u)));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_src0 = vmovl_u8(vld1_u8(src0 + j));
                    uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + j));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1)))), scale);
                    uint16x8_t v_dstu = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                                     vqmovn_u32(vcvtq_u32_f32(v_dst1f)));
                    vst1_u8(dst + j, vqmovn_u16(v_dstu));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = internal::saturate_cast<u8>((s32)trunc(fval));
                }
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                    uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                    vst1q_u8(dst + j, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
                }
                for (; j < roiw8; j += 8)
                {
                    vst1_u8(dst + j, vmovn_u16(vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j))));
                }

                for (; j < size.width; j++)
                {
                    u16 val = (u16)src0[j] * (u16)src1[j];
                    dst[j] = (u8)(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);
                    uint8x16_t v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
                    uint16x8_t v_src1_p = vmovl_u8(vget_low_u8(v_src1));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    v_src0_p = vmovl_u8(vget_high_u8(v_src0));
                    v_src1_p = vmovl_u8(vget_high_u8(v_src1));
                    float32x4_t v_dst2f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    float32x4_t v_dst3f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    uint16x8_t v_dst0u = vcombine_u16(vmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                                      vmovn_u32(vcvtq_u32_f32(v_dst1f)));
                    uint16x8_t v_dst1u = vcombine_u16(vmovn_u32(vcvtq_u32_f32(v_dst2f)),
                                                      vmovn_u32(vcvtq_u32_f32(v_dst3f)));
                    vst1q_u8(dst + j, vcombine_u8(vmovn_u16(v_dst0u), vmovn_u16(v_dst1u)));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_src0 = vmovl_u8(vld1_u8(src0 + j));
                    uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + j));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1)))), scale);
                    uint16x8_t v_dstu = vcombine_u16(vmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                                     vmovn_u32(vcvtq_u32_f32(v_dst1f)));
                    vst1_u8(dst + j, vmovn_u16(v_dstu));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = (u8)(s32)trunc(fval);
                }
            }
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
    (void)cpolicy;
    (void)scale;
#endif
}

void mul(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         s16 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (((scale * UCHAR_MAX * UCHAR_MAX) < 1.0f) && (scale >= 0))
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            s16 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(s16) * size.width);
        }
        return;
    }

    s32 iscale = static_cast<s32>(scale), exp = 0;
    f32 significand = frexp(scale, &exp);
    bool is_integer_scale = isIntegerScale(scale),
         is_power_of_2 = (significand == 0.5f) && (exp <= 0);
    exp = -exp + 1;

    if (is_power_of_2)
    {
        static const mulFuncs16 funcs[16] =
        {
            NULL,
            muls16<1>,
            muls16<2>,
            muls16<3>,
            muls16<4>,
            muls16<5>,
            muls16<6>,
            muls16<7>,
            muls16<8>,
            muls16<9>,
            muls16<10>,
            muls16<11>,
            muls16<12>,
            muls16<13>,
            muls16<14>,
            muls16<15>
        };

        mulFuncs16 func = funcs[exp];

        func(size,
             src0Base, src0Stride,
             src1Base, src1Stride,
             dstBase, dstStride,
             cpolicy);

        return;
    }

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    uint16x8_t v_32767 = vdupq_n_u16(0x7FFF);

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                    uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                    vst1q_s16(dst + j, vreinterpretq_s16_u16(vminq_u16(v_32767, v_dst0)));
                    vst1q_s16(dst + j +8, vreinterpretq_s16_u16(vminq_u16(v_32767, v_dst1)));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_dst = vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j));
                    vst1q_s16(dst + j, vreinterpretq_s16_u16(vminq_u16(v_32767, v_dst)));
                }

                for (; j < size.width; j++)
                {
                    u16 val = (u16)src0[j] * (u16)src1[j];
                    dst[j] = internal::saturate_cast<s16>(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);
                    uint8x16_t v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
                    uint16x8_t v_src1_p = vmovl_u8(vget_low_u8(v_src1));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vqmovn_s32(vcvtq_s32_f32(v_dst1f))));

                    v_src0_p = vmovl_u8(vget_high_u8(v_src0));
                    v_src1_p = vmovl_u8(vget_high_u8(v_src1));
                    v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    vst1q_s16(dst + j + 8, vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                        vqmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_src0 = vmovl_u8(vld1_u8(src0 + j));
                    uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + j));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vqmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = internal::saturate_cast<s16>((s32)trunc(fval));
                }
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j), v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_dst0 = vmull_u8(vget_low_u8(v_src0), vget_low_u8(v_src1));
                    uint16x8_t v_dst1 = vmull_u8(vget_high_u8(v_src0), vget_high_u8(v_src1));

                    vst1q_s16(dst + j, vreinterpretq_s16_u16(v_dst0));
                    vst1q_s16(dst + j + 8, vreinterpretq_s16_u16(v_dst1));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_dst = vmull_u8(vld1_u8(src0 + j), vld1_u8(src1 + j));
                    vst1q_s16(dst + j, vreinterpretq_s16_u16(v_dst));
                }

                for (; j < size.width; j++)
                {
                    u16 val = (u16)src0[j] * (u16)src1[j];
                    dst[j] = (s16)(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);
                    uint8x16_t v_src1 = vld1q_u8(src1 + j);

                    uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
                    uint16x8_t v_src1_p = vmovl_u8(vget_low_u8(v_src1));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vmovn_s32(vcvtq_s32_f32(v_dst1f))));

                    v_src0_p = vmovl_u8(vget_high_u8(v_src0));
                    v_src1_p = vmovl_u8(vget_high_u8(v_src1));
                    v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p)))), scale);
                    v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p)))), scale);
                    vst1q_s16(dst + j + 8, vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                        vmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_src0 = vmovl_u8(vld1_u8(src0 + j));
                    uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + j));
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))),
                                                                vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = (s16)(s32)trunc(fval);
                }
            }
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
    (void)cpolicy;
    (void)scale;
#endif
}

void mul(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const s16 * src1Base, ptrdiff_t src1Stride,
         s16 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    if (scale == 0.0f)
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            s16 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(s16) * size.width);
        }
        return;
    }

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    bool is_integer_scale = isIntegerScale(scale);
    s32 iscale = static_cast<s32>(scale);

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const s16 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);

                    int16x8_t v_src0_p = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src0)));
                    int16x8_t v_src1_p = vld1q_s16(src1 + j);
                    int16x8_t v_dst = vcombine_s16(vqmovn_s32(vmull_s16(vget_low_s16(v_src0_p), vget_low_s16(v_src1_p))),
                                                   vqmovn_s32(vmull_s16(vget_high_s16(v_src0_p), vget_high_s16(v_src1_p))));
                    vst1q_s16(dst + j, v_dst);

                    v_src0_p = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src0)));
                    v_src1_p = vld1q_s16(src1 + j + 8);
                    v_dst = vcombine_s16(vqmovn_s32(vmull_s16(vget_low_s16(v_src0_p), vget_low_s16(v_src1_p))),
                                                   vqmovn_s32(vmull_s16(vget_high_s16(v_src0_p), vget_high_s16(v_src1_p))));
                    vst1q_s16(dst + j + 8, v_dst);
                }
                for (; j < roiw8; j += 8)
                {
                    int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vld1q_u8(src0 + j))));
                    int16x8_t v_src1 = vld1q_s16(src1 + j);
                    int16x8_t v_dst = vcombine_s16(vqmovn_s32(vmull_s16(vget_low_s16(v_src0), vget_low_s16(v_src1))),
                                                   vqmovn_s32(vmull_s16(vget_high_s16(v_src0), vget_high_s16(v_src1))));
                    vst1q_s16(dst + j, v_dst);
                }

                for (; j < size.width; j++)
                {
                    s32 val = (s32)src0[j] * (s32)src1[j];
                    dst[j] = internal::saturate_cast<s16>(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);

                    uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
                    int16x8_t v_src1_p = vld1q_s16(src1 + j);
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1_p)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1_p)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vqmovn_s32(vcvtq_s32_f32(v_dst1f))));

                    v_src0_p = vmovl_u8(vget_high_u8(v_src0));
                    v_src1_p = vld1q_s16(src1 + j + 8);
                    v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1_p)))), scale);
                    v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1_p)))), scale);
                    vst1q_s16(dst + j + 8, vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                        vqmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_src0 = vmovl_u8(vld1_u8(src0 + j));
                    int16x8_t v_src1 = vld1q_s16(src1 + j);
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vqmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vqmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = internal::saturate_cast<s16>((s32)trunc(fval));
                }
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);

                    int16x8_t v_src0_p = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src0)));
                    int16x8_t v_src1_p = vld1q_s16(src1 + j);
                    int16x8_t v_dst = vcombine_s16(vmovn_s32(vmull_s16(vget_low_s16(v_src0_p), vget_low_s16(v_src1_p))),
                                                   vmovn_s32(vmull_s16(vget_high_s16(v_src0_p), vget_high_s16(v_src1_p))));
                    vst1q_s16(dst + j, v_dst);

                    v_src0_p = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src0)));
                    v_src1_p = vld1q_s16(src1 + j + 8);
                    v_dst = vcombine_s16(vmovn_s32(vmull_s16(vget_low_s16(v_src0_p), vget_low_s16(v_src1_p))),
                                                   vmovn_s32(vmull_s16(vget_high_s16(v_src0_p), vget_high_s16(v_src1_p))));
                    vst1q_s16(dst + j + 8, v_dst);
                }
                for (; j < roiw8; j += 8)
                {
                    int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vld1q_u8(src0 + j))));
                    int16x8_t v_src1 = vld1q_s16(src1 + j);
                    int16x8_t v_dst = vcombine_s16(vmovn_s32(vmull_s16(vget_low_s16(v_src0), vget_low_s16(v_src1))),
                                                   vmovn_s32(vmull_s16(vget_high_s16(v_src0), vget_high_s16(v_src1))));
                    vst1q_s16(dst + j, v_dst);
                }

                for (; j < size.width; j++)
                {
                    s32 val = (s32)src0[j] * (s32)src1[j];
                    dst[j] = (s16)(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    uint8x16_t v_src0 = vld1q_u8(src0 + j);

                    uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
                    int16x8_t v_src1_p = vld1q_s16(src1 + j);
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1_p)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1_p)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vmovn_s32(vcvtq_s32_f32(v_dst1f))));

                    v_src0_p = vmovl_u8(vget_high_u8(v_src0));
                    v_src1_p = vld1q_s16(src1 + j + 8);
                    v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))),
                                                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1_p)))), scale);
                    v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))),
                                                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1_p)))), scale);
                    vst1q_s16(dst + j + 8, vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                        vmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }
                for (; j < roiw8; j += 8)
                {
                    uint16x8_t v_src0 = vmovl_u8(vld1_u8(src0 + j));
                    int16x8_t v_src1 = vld1q_s16(src1 + j);
                    float32x4_t v_dst0f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1)))), scale);
                    float32x4_t v_dst1f = vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))),
                                                                vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1)))), scale);
                    vst1q_s16(dst + j, vcombine_s16(vmovn_s32(vcvtq_s32_f32(v_dst0f)),
                                                    vmovn_s32(vcvtq_s32_f32(v_dst1f))));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = (s16)(s32)trunc(fval);
                }
            }
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
    (void)cpolicy;
    (void)scale;
#endif
}

namespace {

#ifdef CAROTENE_NEON

template <typename T>
inline T mulSaturateQ(const T &v1, const T &v2, const float scale)
{
    return internal::vcombine(internal::vqmovn(mulSaturateQ(internal::vmovl(internal::vget_low(v1)),
                                                            internal::vmovl(internal::vget_low(v2)), scale)),
                              internal::vqmovn(mulSaturateQ(internal::vmovl(internal::vget_high(v1)),
                                                            internal::vmovl(internal::vget_high(v2)), scale))
                             );
}
template <>
inline int32x4_t mulSaturateQ<int32x4_t>(const int32x4_t &v1, const int32x4_t &v2, const float scale)
{ return vcvtq_s32_f32(vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(v1), vcvtq_f32_s32(v2)), scale)); }
template <>
inline uint32x4_t mulSaturateQ<uint32x4_t>(const uint32x4_t &v1, const uint32x4_t &v2, const float scale)
{ return vcvtq_u32_f32(vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(v1), vcvtq_f32_u32(v2)), scale)); }

template <typename T>
inline T mulSaturate(const T &v1, const T &v2, const float scale)
{
    return internal::vqmovn(mulSaturateQ(internal::vmovl(v1), internal::vmovl(v2), scale));
}
template <>
inline int32x2_t mulSaturate<int32x2_t>(const int32x2_t &v1, const int32x2_t &v2, const float scale)
{ return vcvt_s32_f32(vmul_n_f32(vmul_f32(vcvt_f32_s32(v1), vcvt_f32_s32(v2)), scale)); }
template <>
inline uint32x2_t mulSaturate<uint32x2_t>(const uint32x2_t &v1, const uint32x2_t &v2, const float scale)
{ return vcvt_u32_f32(vmul_n_f32(vmul_f32(vcvt_f32_u32(v1), vcvt_f32_u32(v2)), scale)); }


template <typename T>
inline T mulWrapQ(const T &v1, const T &v2, const float scale)
{
    return internal::vcombine(internal::vmovn(mulWrapQ(internal::vmovl(internal::vget_low(v1)),
                                                       internal::vmovl(internal::vget_low(v2)), scale)),
                              internal::vmovn(mulWrapQ(internal::vmovl(internal::vget_high(v1)),
                                                       internal::vmovl(internal::vget_high(v2)), scale))
                             );
}
template <>
inline int32x4_t mulWrapQ<int32x4_t>(const int32x4_t &v1, const int32x4_t &v2, const float scale)
{ return vcvtq_s32_f32(vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(v1), vcvtq_f32_s32(v2)), scale)); }
template <>
inline uint32x4_t mulWrapQ<uint32x4_t>(const uint32x4_t &v1, const uint32x4_t &v2, const float scale)
{ return vcvtq_u32_f32(vmulq_n_f32(vmulq_f32(vcvtq_f32_u32(v1), vcvtq_f32_u32(v2)), scale)); }

template <typename T>
inline T mulWrap(const T &v1, const T &v2, const float scale)
{
    return internal::vmovn(mulWrapQ(internal::vmovl(v1), internal::vmovl(v2), scale));
}
template <>
inline int32x2_t mulWrap<int32x2_t>(const int32x2_t &v1, const int32x2_t &v2, const float scale)
{ return vcvt_s32_f32(vmul_n_f32(vmul_f32(vcvt_f32_s32(v1), vcvt_f32_s32(v2)), scale)); }
template <>
inline uint32x2_t mulWrap<uint32x2_t>(const uint32x2_t &v1, const uint32x2_t &v2, const float scale)
{ return vcvt_u32_f32(vmul_n_f32(vmul_f32(vcvt_f32_u32(v1), vcvt_f32_u32(v2)), scale)); }


template <int n> inline  uint8x16_t vshrq_n(const uint8x16_t  & v0) { return vshrq_n_u8 (v0, n); }
template <int n> inline   int8x16_t vshrq_n(const int8x16_t   & v0) { return vshrq_n_s8 (v0, n); }
template <int n> inline  uint16x8_t vshrq_n(const uint16x8_t  & v0) { return vshrq_n_u16(v0, n); }
template <int n> inline   int16x8_t vshrq_n(const int16x8_t   & v0) { return vshrq_n_s16(v0, n); }
template <int n> inline  uint32x4_t vshrq_n(const uint32x4_t  & v0) { return vshrq_n_u32(v0, n); }
template <int n> inline   int32x4_t vshrq_n(const int32x4_t   & v0) { return vshrq_n_s32(v0, n); }
template <int n> inline  uint64x2_t vshrq_n(const uint64x2_t  & v0) { return vshrq_n_u64(v0, n); }
template <int n> inline   int64x2_t vshrq_n(const int64x2_t   & v0) { return vshrq_n_s64(v0, n); }

template <int n> inline   uint8x8_t vshr_n(const uint8x8_t   & v0) { return vshr_n_u8 (v0, n); }
template <int n> inline    int8x8_t vshr_n(const int8x8_t    & v0) { return vshr_n_s8 (v0, n); }
template <int n> inline  uint16x4_t vshr_n(const uint16x4_t  & v0) { return vshr_n_u16(v0, n); }
template <int n> inline   int16x4_t vshr_n(const int16x4_t   & v0) { return vshr_n_s16(v0, n); }
template <int n> inline  uint32x2_t vshr_n(const uint32x2_t  & v0) { return vshr_n_u32(v0, n); }
template <int n> inline   int32x2_t vshr_n(const int32x2_t   & v0) { return vshr_n_s32(v0, n); }
template <int n> inline  uint64x1_t vshr_n(const uint64x1_t  & v0) { return vshr_n_u64(v0, n); }
template <int n> inline   int64x1_t vshr_n(const int64x1_t   & v0) { return vshr_n_s64(v0, n); }

template <int n> inline  uint8x16_t vrshrq_n(const uint8x16_t  & v0) { return vrshrq_n_u8 (v0, n); }
template <int n> inline   int8x16_t vrshrq_n(const int8x16_t   & v0) { return vrshrq_n_s8 (v0, n); }
template <int n> inline  uint16x8_t vrshrq_n(const uint16x8_t  & v0) { return vrshrq_n_u16(v0, n); }
template <int n> inline   int16x8_t vrshrq_n(const int16x8_t   & v0) { return vrshrq_n_s16(v0, n); }
template <int n> inline  uint32x4_t vrshrq_n(const uint32x4_t  & v0) { return vrshrq_n_u32(v0, n); }
template <int n> inline   int32x4_t vrshrq_n(const int32x4_t   & v0) { return vrshrq_n_s32(v0, n); }
template <int n> inline  uint64x2_t vrshrq_n(const uint64x2_t  & v0) { return vrshrq_n_u64(v0, n); }
template <int n> inline   int64x2_t vrshrq_n(const int64x2_t   & v0) { return vrshrq_n_s64(v0, n); }

template <int n> inline   uint8x8_t vrshr_n(const uint8x8_t   & v0) { return vrshr_n_u8 (v0, n); }
template <int n> inline    int8x8_t vrshr_n(const int8x8_t    & v0) { return vrshr_n_s8 (v0, n); }
template <int n> inline  uint16x4_t vrshr_n(const uint16x4_t  & v0) { return vrshr_n_u16(v0, n); }
template <int n> inline   int16x4_t vrshr_n(const int16x4_t   & v0) { return vrshr_n_s16(v0, n); }
template <int n> inline  uint32x2_t vrshr_n(const uint32x2_t  & v0) { return vrshr_n_u32(v0, n); }
template <int n> inline   int32x2_t vrshr_n(const int32x2_t   & v0) { return vrshr_n_s32(v0, n); }
template <int n> inline  uint64x1_t vrshr_n(const uint64x1_t  & v0) { return vrshr_n_u64(v0, n); }
template <int n> inline   int64x1_t vrshr_n(const int64x1_t   & v0) { return vrshr_n_s64(v0, n); }

template <typename T, typename WT, s32 shift>
void mulShift(const Size2D &size,
              const T * src0Base, ptrdiff_t src0Stride,
              const T * src1Base, ptrdiff_t src1Stride,
              T * dstBase, ptrdiff_t dstStride,
              CONVERT_POLICY cpolicy)
{
    typedef typename internal::VecTraits<T>::vec128 vec128;
    typedef typename internal::VecTraits<WT>::vec128 wvec128;
    typedef typename internal::VecTraits<T>::vec64 vec64;
    const size_t step128 = 16 / sizeof(T);
    size_t roiw128 = size.width >= (step128 - 1) ? size.width - step128 + 1 : 0;
    const size_t step64 = 8 / sizeof(T);
    size_t roiw64 = size.width >= (step64 - 1) ? size.width - step64 + 1 : 0;

    wvec128 v_mask = internal::vdupq_n((WT)(1<<shift));

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const T * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        T * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                vec128 v_src0 = internal::vld1q(src0 + j), v_src1 = internal::vld1q(src1 + j);
                wvec128 v_mul0 = internal::vmull( internal::vget_low(v_src0),  internal::vget_low(v_src1));
                wvec128 v_mul1 = internal::vmull(internal::vget_high(v_src0), internal::vget_high(v_src1));

                vec64 v_res0 = internal::vqmovn(vrshrq_n<shift>(internal::vqsubq(v_mul0, vshrq_n<shift>(internal::vbicq(v_mask, v_mul0)) )));
                vec64 v_res1 = internal::vqmovn(vrshrq_n<shift>(internal::vqsubq(v_mul1, vshrq_n<shift>(internal::vbicq(v_mask, v_mul1)) )));

                internal::vst1q(dst + j, internal::vcombine(v_res0, v_res1));
            }
            for (; j < roiw64; j += step64)
            {
                wvec128 v_mul = internal::vmull(internal::vld1(src0 + j), internal::vld1(src1 + j));
                vec64 v_res = internal::vqmovn(vrshrq_n<shift>(internal::vqsubq(v_mul, vshrq_n<shift>(internal::vbicq(v_mask, v_mul)) )));
                internal::vst1(dst + j, v_res);
            }

            for (; j < size.width; j++)
            {
                WT val = (WT)src0[j] * (WT)src1[j];
                dst[j] = internal::saturate_cast<T>((val - (((1<<shift) & ~val) >> shift) + (1<<(shift-1))) >> shift);
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                vec128 v_src0 = internal::vld1q(src0 + j), v_src1 = internal::vld1q(src1 + j);
                wvec128 v_mul0 = internal::vmull( internal::vget_low(v_src0),  internal::vget_low(v_src1));
                wvec128 v_mul1 = internal::vmull(internal::vget_high(v_src0), internal::vget_high(v_src1));

                vec64 v_res0 = internal::vmovn(vrshrq_n<shift>(internal::vqsubq(v_mul0, vshrq_n<shift>(internal::vbicq(v_mask, v_mul0)) )));
                vec64 v_res1 = internal::vmovn(vrshrq_n<shift>(internal::vqsubq(v_mul1, vshrq_n<shift>(internal::vbicq(v_mask, v_mul1)) )));

                internal::vst1q(dst + j, internal::vcombine(v_res0, v_res1));
            }
            for (; j < roiw64; j += step64)
            {
                wvec128 v_mul = internal::vmull(internal::vld1(src0 + j), internal::vld1(src1 + j));
                vec64 v_res = internal::vmovn(vrshrq_n<shift>(internal::vqsubq(v_mul, vshrq_n<shift>(internal::vbicq(v_mask, v_mul)) )));
                internal::vst1(dst + j, v_res);
            }

            for (; j < size.width; j++)
            {
                WT val = (WT)src0[j] * (WT)src1[j];
                dst[j] = (T)((val - (((1<<shift) & ~val) >> shift) + (1<<(shift-1))) >> shift);
            }
        }
    }
}
#endif

template <typename T, typename WT>
void mul(const Size2D &size,
         const T * src0Base, ptrdiff_t src0Stride,
         const T * src1Base, ptrdiff_t src1Stride,
         T * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    typedef typename internal::VecTraits<T>::vec128 vec128;

    typedef void (* mulFunc)(const Size2D &size,
                             const T * src0Base, ptrdiff_t src0Stride,
                             const T * src1Base, ptrdiff_t src1Stride,
                             T * dstBase, ptrdiff_t dstStride,
                             CONVERT_POLICY cpolicy);

    if (scale == 0.0f ||
        (std::numeric_limits<T>::is_integer &&
         (scale * std::numeric_limits<T>::max() * std::numeric_limits<T>::max()) <  1.0f &&
         (scale * std::numeric_limits<T>::max() * std::numeric_limits<T>::max()) > -1.0f))
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            T * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(T) * size.width);
        }
        return;
    }

    s32 iscale = static_cast<s32>(scale), exp = 0;
    f32 significand = frexp(scale, &exp);
    bool is_integer_scale = isIntegerScale(scale),
         is_power_of_2 = (significand == 0.5f) && (exp <= 0);
    exp = -exp + 1;

    if (is_power_of_2)
    {
        static const mulFunc funcs[16] =
        {
            NULL,
            mulShift<T,WT,1>,
            mulShift<T,WT,2>,
            mulShift<T,WT,3>,
            mulShift<T,WT,4>,
            mulShift<T,WT,5>,
            mulShift<T,WT,6>,
            mulShift<T,WT,7>,
            mulShift<T,WT,8>,
            mulShift<T,WT,9>,
            mulShift<T,WT,10>,
            mulShift<T,WT,11>,
            mulShift<T,WT,12>,
            mulShift<T,WT,13>,
            mulShift<T,WT,14>,
            mulShift<T,WT,15>
        };

        mulFunc func = funcs[exp];

        func(size,
             src0Base, src0Stride,
             src1Base, src1Stride,
             dstBase, dstStride,
             cpolicy);

        return;
    }

    const size_t step128 = 16 / sizeof(T);
    size_t roiw128 = size.width >= (step128 - 1) ? size.width - step128 + 1 : 0;
    const size_t step64 = 8 / sizeof(T);
    size_t roiw64 = size.width >= (step64 - 1) ? size.width - step64 + 1 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const T * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        T * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw128; j += step128)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    vec128 v_src0 = internal::vld1q(src0 + j), v_src1 = internal::vld1q(src1 + j);
                    internal::vst1q(dst + j, internal::vcombine(
                                                internal::vqmovn(internal::vmull(internal::vget_low(v_src0),
                                                                                 internal::vget_low(v_src1))),
                                                internal::vqmovn(internal::vmull(internal::vget_high(v_src0),
                                                                                 internal::vget_high(v_src1)))
                                                               )
                                   );
                }
                for (; j < roiw64; j += step64)
                {
                    internal::vst1(dst + j, internal::vqmovn(internal::vmull(internal::vld1(src0 + j),
                                                                             internal::vld1(src1 + j))));
                }

                for (; j < size.width; j++)
                {
                    WT val = (WT)src0[j] * (WT)src1[j];
                    dst[j] = internal::saturate_cast<T>(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw128; j += step128)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    internal::vst1q(dst + j, mulSaturateQ(internal::vld1q(src0 + j),
                                                          internal::vld1q(src1 + j), scale));
                }
                for (; j < roiw64; j += step64)
                {
                    internal::vst1(dst + j, mulSaturate(internal::vld1(src0 + j),
                                                        internal::vld1(src1 + j), scale));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = internal::saturate_cast<T>(fval);
                }
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw128; j += step128)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    vec128 v_src0 = internal::vld1q(src0 + j), v_src1 = internal::vld1q(src1 + j);
                    internal::vst1q(dst + j, internal::vcombine(
                                                 internal::vmovn(internal::vmull(internal::vget_low(v_src0),
                                                                                 internal::vget_low(v_src1))),
                                                 internal::vmovn(internal::vmull(internal::vget_high(v_src0),
                                                                                 internal::vget_high(v_src1)))
                                                               )
                                   );
                }
                for (; j < roiw64; j += step64)
                {
                    internal::vst1(dst + j, internal::vmovn(internal::vmull(internal::vld1(src0 + j),
                                                                            internal::vld1(src1 + j))));
                }

                for (; j < size.width; j++)
                {
                    WT val = (WT)src0[j] * (WT)src1[j];
                    dst[j] = (T)(val);
                }
            }
            else // generic case using floats
            {
                for (; j < roiw128; j += step128)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    internal::vst1q(dst + j, mulWrapQ(internal::vld1q(src0 + j),
                                                      internal::vld1q(src1 + j), scale));
                }
                for (; j < roiw64; j += step64)
                {
                    internal::vst1(dst + j, mulWrap(internal::vld1(src0 + j),
                                                    internal::vld1(src1 + j), scale));
                }

                for (; j < size.width; j++)
                {
                    f32 fval = (f32)src0[j] * (f32)src1[j] * scale;
                    dst[j] = (T)((s32)trunc(fval));
                }
            }
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
    (void)cpolicy;
    (void)scale;
#endif
}

}

void mul(const Size2D &size,
         const s8 * src0Base, ptrdiff_t src0Stride,
         const s8 * src1Base, ptrdiff_t src1Stride,
         s8 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    mul<s8,s16>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void mul(const Size2D &size,
         const u16 * src0Base, ptrdiff_t src0Stride,
         const u16 * src1Base, ptrdiff_t src1Stride,
         u16 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    mul<u16,u32>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void mul(const Size2D &size,
         const s16 * src0Base, ptrdiff_t src0Stride,
         const s16 * src1Base, ptrdiff_t src1Stride,
         s16 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    mul<s16,s32>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void mul(const Size2D &size,
         const s32 * src0Base, ptrdiff_t src0Stride,
         const s32 * src1Base, ptrdiff_t src1Stride,
         s32 * dstBase, ptrdiff_t dstStride,
         f64 scale,
         CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    typedef void (* mulFunc)(const Size2D &size,
                             const s32 * src0Base, ptrdiff_t src0Stride,
                             const s32 * src1Base, ptrdiff_t src1Stride,
                             s32 * dstBase, ptrdiff_t dstStride,
                             CONVERT_POLICY cpolicy);

    if (!std::isnormal(scale) ||
        ((scale * std::numeric_limits<s32>::max() * std::numeric_limits<s32>::max()) <  1.0f &&
         (scale * std::numeric_limits<s32>::max() * std::numeric_limits<s32>::max()) > -1.0f))
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            s32 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(s32) * size.width);
        }
        return;
    }

    s32 iscale = static_cast<s32>(scale), exp = 0;
    f64 significand = frexp(scale, &exp);
    bool is_integer_scale = isIntegerScale(scale),
         is_power_of_2 = (significand == 0.5) && (exp <= 0);
    exp = -exp + 1;

    if (is_power_of_2)
    {
        static const mulFunc funcs[16] =
        {
            NULL,
            mulShift<s32,s64,1>,
            mulShift<s32,s64,2>,
            mulShift<s32,s64,3>,
            mulShift<s32,s64,4>,
            mulShift<s32,s64,5>,
            mulShift<s32,s64,6>,
            mulShift<s32,s64,7>,
            mulShift<s32,s64,8>,
            mulShift<s32,s64,9>,
            mulShift<s32,s64,10>,
            mulShift<s32,s64,11>,
            mulShift<s32,s64,12>,
            mulShift<s32,s64,13>,
            mulShift<s32,s64,14>,
            mulShift<s32,s64,15>
        };

        mulFunc func = funcs[exp];

        func(size,
             src0Base, src0Stride,
             src1Base, src1Stride,
             dstBase, dstStride,
             cpolicy);

        return;
    }

    size_t roiw128 = size.width >= 3 ? size.width - 3 : 0;
    size_t roiw64 = size.width >= 1 ? size.width - 1 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const s32 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        s32 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw128; j += 4)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    int32x4_t v_src0 = internal::vld1q(src0 + j), v_src1 = internal::vld1q(src1 + j);
                    internal::vst1q(dst + j, internal::vcombine(
                                                internal::vqmovn(internal::vmull(internal::vget_low(v_src0),
                                                                                 internal::vget_low(v_src1))),
                                                internal::vqmovn(internal::vmull(internal::vget_high(v_src0),
                                                                                 internal::vget_high(v_src1)))
                                                               )
                                   );
                }
                for (; j < roiw64; j += 2)
                {
                    internal::vst1(dst + j, internal::vqmovn(internal::vmull(internal::vld1(src0 + j),
                                                                             internal::vld1(src1 + j))));
                }

                for (; j < size.width; j++)
                {
                    s64 val = (s64)src0[j] * (s64)src1[j];
                    dst[j] = internal::saturate_cast<s32>(val);
                }
            }
            else // generic case using floats
            {
                for (; j < size.width; j++)
                {
                    f64 fval = src0[j] * src1[j] * scale;
                    dst[j] = internal::saturate_cast<s32>(fval);
                }
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            if (is_integer_scale && iscale == 1)
            {
                for (; j < roiw128; j += 4)
                {
                    internal::prefetch(src0 + j);
                    internal::prefetch(src1 + j);
                    int32x4_t v_src0 = internal::vld1q(src0 + j), v_src1 = internal::vld1q(src1 + j);
                    internal::vst1q(dst + j, internal::vcombine(
                                                 internal::vmovn(internal::vmull(internal::vget_low(v_src0),
                                                                                 internal::vget_low(v_src1))),
                                                 internal::vmovn(internal::vmull(internal::vget_high(v_src0),
                                                                                 internal::vget_high(v_src1)))
                                                               )
                                   );
                }
                for (; j < roiw64; j += 2)
                {
                    internal::vst1(dst + j, internal::vmovn(internal::vmull(internal::vld1(src0 + j),
                                                                            internal::vld1(src1 + j))));
                }

                for (; j < size.width; j++)
                {
                    s64 val = (s64)src0[j] * (s64)src1[j];
                    dst[j] = (s32)(val);
                }
            }
            else // generic case using floats
            {
                for (; j < size.width; j++)
                {
                    f64 fval = src0[j] * src1[j] * scale;
                    dst[j] = (s32)trunc(fval);
                }
            }
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
    (void)cpolicy;
    (void)scale;
#endif
}

void mul(const Size2D &size,
         const f32 * src0Base, ptrdiff_t src0Stride,
         const f32 * src1Base, ptrdiff_t src1Stride,
         f32 * dstBase, ptrdiff_t dstStride,
         f32 scale)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (scale == 0.0f)
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            f32 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(f32) * size.width);
        }
        return;
    }

    size_t roiw128 = size.width >= 3 ? size.width - 3 : 0;
    size_t roiw64 = size.width >= 1 ? size.width - 1 : 0;

    if (std::fabs(scale - 1.0f) < FLT_EPSILON)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const f32 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
            const f32 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
            f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw128; j += 4)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                vst1q_f32(dst + j, vmulq_f32(vld1q_f32(src0 + j), vld1q_f32(src1 + j)));
            }

            for (; j < roiw64; j += 2)
            {
                vst1_f32(dst + j, vmul_f32(vld1_f32(src0 + j), vld1_f32(src1 + j)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = src0[j] * src1[j];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const f32 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
            const f32 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
            f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw128; j += 4)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                vst1q_f32(dst + j, vmulq_n_f32(vmulq_f32(vld1q_f32(src0 + j), vld1q_f32(src1 + j)), scale));
            }

            for (; j < roiw64; j += 2)
            {
                vst1_f32(dst + j, vmul_n_f32(vmul_f32(vld1_f32(src0 + j), vld1_f32(src1 + j)), scale));
            }

            for (; j < size.width; j++)
            {
                dst[j] = src0[j] * src1[j] * scale;
            }
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
