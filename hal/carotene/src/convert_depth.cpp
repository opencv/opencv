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
 * Copyright (C) 2014, NVIDIA Corporation, all rights reserved.
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

#include <cstring>

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

template <int shift>
void lshiftConst(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint8x16_t v_src = vld1q_u8(src + j);
            int16x8_t v_dst0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src)));
            int16x8_t v_dst1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src)));

            vst1q_s16(dst + j, vshlq_n_s16(v_dst0, shift));
            vst1q_s16(dst + j + 8, vshlq_n_s16(v_dst1, shift));
        }
        for (; j < roiw8; j += 8)
        {
            int16x8_t v_dst = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + j)));
            vst1q_s16(dst + j, vshlq_n_s16(v_dst, shift));
        }

        for (; j < size.width; j++)
        {
            dst[j] = ((s16)src[j] << shift);
        }
    }
}

template <>
void lshiftConst<0>(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    s16 * dstBase, ptrdiff_t dstStride)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint8x16_t v_src = vld1q_u8(src + j);
            int16x8_t v_dst0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src)));
            int16x8_t v_dst1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src)));

            vst1q_s16(dst + j, v_dst0);
            vst1q_s16(dst + j + 8, v_dst1);
        }
        for (; j < roiw8; j += 8)
        {
            int16x8_t v_dst = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + j)));
            vst1q_s16(dst + j, v_dst);
        }

        for (; j < size.width; j++)
        {
            dst[j] = (s16)src[j];
        }
    }
}

template <int shift>
void rshiftConst(const Size2D &size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 CONVERT_POLICY cpolicy)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src + j);
                int16x8_t v_src0 = vshrq_n_s16(vld1q_s16(src + j), shift),
                          v_src1 = vshrq_n_s16(vld1q_s16(src + j + 8), shift);
                uint8x16_t v_dst = vcombine_u8(vqmovun_s16(v_src0),
                                               vqmovun_s16(v_src1));
                vst1q_u8(dst + j, v_dst);
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src = vshrq_n_s16(vld1q_s16(src + j), shift);
                vst1_u8(dst + j, vqmovun_s16(v_src));
            }

            for (; j < size.width; j++)
            {
                dst[j] = internal::saturate_cast<u8>((src[j] >> shift));
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src + j);
                int16x8_t v_src0 = vshrq_n_s16(vld1q_s16(src + j), shift),
                          v_src1 = vshrq_n_s16(vld1q_s16(src + j + 8), shift);
                int8x16_t v_dst = vcombine_s8(vmovn_s16(v_src0),
                                              vmovn_s16(v_src1));
                vst1q_u8(dst + j, vreinterpretq_u8_s8(v_dst));
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src = vshrq_n_s16(vld1q_s16(src + j), shift);
                vst1_u8(dst + j, vreinterpret_u8_s8(vmovn_s16(v_src)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = (u8)((src[j] >> shift));
            }
        }
    }
}

template <>
void rshiftConst<0>(const Size2D &size,
                    const s16 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    CONVERT_POLICY cpolicy)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src + j);
                int16x8_t v_src0 = vld1q_s16(src + j), v_src1 = vld1q_s16(src + j + 8);
                uint8x16_t v_dst = vcombine_u8(vqmovun_s16(v_src0), vqmovun_s16(v_src1));
                vst1q_u8(dst + j, v_dst);
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src = vld1q_s16(src + j);
                vst1_u8(dst + j, vqmovun_s16(v_src));
            }

            for (; j < size.width; j++)
            {
                dst[j] = internal::saturate_cast<u8>(src[j]);
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src + j);
                int16x8_t v_src0 = vld1q_s16(src + j), v_src1 = vld1q_s16(src + j + 8);
                int8x16_t v_dst = vcombine_s8(vmovn_s16(v_src0), vmovn_s16(v_src1));
                vst1q_u8(dst + j, vreinterpretq_u8_s8(v_dst));
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src = vld1q_s16(src + j);
                vst1_u8(dst + j, vreinterpret_u8_s8(vmovn_s16(v_src)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = (u8)src[j];
            }
        }
    }
}

typedef void (* lshiftConstFunc)(const Size2D &size,
                                const u8 * srcBase, ptrdiff_t srcStride,
                                s16 * dstBase, ptrdiff_t dstStride);

typedef void (* rshiftConstFunc)(const Size2D &size,
                                const s16 * srcBase, ptrdiff_t srcStride,
                                u8 * dstBase, ptrdiff_t dstStride,
                                CONVERT_POLICY cpolicy);

} // namespace

#endif

void lshift(const Size2D &size,
            const u8 * srcBase, ptrdiff_t srcStride,
            s16 * dstBase, ptrdiff_t dstStride,
            u32 shift)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    if (shift >= 16u)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
            std::memset(dst, 0, sizeof(s16) * size.width);
        }
        return;
    }

    // this ugly contruction is needed to avoid:
    // /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/arm_neon.h:3581:59: error: argument must be a constant
    // return (int16x8_t)__builtin_neon_vshl_nv8hi (__a, __b, 1);

    lshiftConstFunc funcs[16] =
    {
        lshiftConst<0>,
        lshiftConst<1>,
        lshiftConst<2>,
        lshiftConst<3>,
        lshiftConst<4>,
        lshiftConst<5>,
        lshiftConst<6>,
        lshiftConst<7>,
        lshiftConst<8>,
        lshiftConst<9>,
        lshiftConst<10>,
        lshiftConst<11>,
        lshiftConst<12>,
        lshiftConst<13>,
        lshiftConst<14>,
        lshiftConst<15>
    }, func = funcs[shift];

    func(size, srcBase, srcStride, dstBase, dstStride);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)shift;
#endif
}

void rshift(const Size2D &size,
            const s16 * srcBase, ptrdiff_t srcStride,
            u8 * dstBase, ptrdiff_t dstStride,
            u32 shift, CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    if (shift >= 16)
    {
        if (cpolicy == CONVERT_POLICY_WRAP)
        {
            size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
            size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
            int16x8_t v_zero = vdupq_n_s16(0);

            for (size_t i = 0; i < size.height; ++i)
            {
                const s16 * src = internal::getRowPtr(srcBase, srcStride, i);
                u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
                size_t j = 0;

                for (; j < roiw16; j += 16)
                {
                    internal::prefetch(src + j);
                    int16x8_t v_src0 = vld1q_s16(src + j), v_src1 = vld1q_s16(src + j + 8);
                    uint8x16_t v_dst = vcombine_u8(vmovn_u16(vcltq_s16(v_src0, v_zero)),
                                                   vmovn_u16(vcltq_s16(v_src1, v_zero)));
                    vst1q_u8(dst + j, v_dst);
                }
                for (; j < roiw8; j += 8)
                {
                    int16x8_t v_src = vld1q_s16(src + j);
                    vst1_u8(dst + j, vmovn_u16(vcltq_s16(v_src, v_zero)));
                }

                for (; j < size.width; j++)
                {
                    dst[j] = src[j] >= 0 ? 0 : 255;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < size.height; ++i)
            {
                u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
                std::memset(dst, 0, sizeof(u8) * size.width);
            }
        }
        return;
    }

    // this ugly contruction is needed to avoid:
    // /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/arm_neon.h:3581:59: error: argument must be a constant
    // return (int16x8_t)__builtin_neon_vshr_nv8hi (__a, __b, 1);

    rshiftConstFunc funcs[16] =
    {
        rshiftConst<0>,
        rshiftConst<1>,
        rshiftConst<2>,
        rshiftConst<3>,
        rshiftConst<4>,
        rshiftConst<5>,
        rshiftConst<6>,
        rshiftConst<7>,
        rshiftConst<8>,
        rshiftConst<9>,
        rshiftConst<10>,
        rshiftConst<11>,
        rshiftConst<12>,
        rshiftConst<13>,
        rshiftConst<14>,
        rshiftConst<15>
    }, func = funcs[shift];

    func(size, srcBase, srcStride, dstBase, dstStride, cpolicy);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)shift;
    (void)cpolicy;
#endif
}

} // namespace CAROTENE_NS
