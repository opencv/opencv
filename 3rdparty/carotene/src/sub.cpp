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
#include "vtransform.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

template <typename T, typename WT>
struct SubWrap
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0,
                     const typename internal::VecTraits<T>::vec128 & v_src1,
                     typename internal::VecTraits<T>::vec128 & v_dst) const
    {
        v_dst = internal::vsubq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0,
                     const typename internal::VecTraits<T>::vec64 & v_src1,
                     typename internal::VecTraits<T>::vec64 & v_dst) const
    {
        v_dst = internal::vsub(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = (T)((WT)src0[0] - (WT)src1[0]);
    }
};

template <typename T, typename WT>
struct SubSaturate
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0,
                     const typename internal::VecTraits<T>::vec128 & v_src1,
                     typename internal::VecTraits<T>::vec128 & v_dst) const
    {
        v_dst = internal::vqsubq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0,
                     const typename internal::VecTraits<T>::vec64 & v_src1,
                     typename internal::VecTraits<T>::vec64 & v_dst) const
    {
        v_dst = internal::vqsub(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = internal::saturate_cast<T>((WT)src0[0] - (WT)src1[0]);
    }
};

} // namespace

#endif

void sub(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         u8 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (policy == CONVERT_POLICY_SATURATE)
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubSaturate<u8, s16>());
    }
    else
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubWrap<u8, s16>());
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         s16 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        u16 * dstu16 = internal::getRowPtr((u16 *)dstBase, dstStride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src0 + j);
            internal::prefetch(src1 + j);
            uint8x16_t v_src00 = vld1q_u8(src0 + j), v_src01 = vld1q_u8(src0 + j + 16);
            uint8x16_t v_src10 = vld1q_u8(src1 + j), v_src11 = vld1q_u8(src1 + j + 16);
            vst1q_u16(dstu16 + j, vsubl_u8(vget_low_u8(v_src00), vget_low_u8(v_src10)));
            vst1q_u16(dstu16 + j + 8, vsubl_u8(vget_high_u8(v_src00), vget_high_u8(v_src10)));
            vst1q_u16(dstu16 + j + 16, vsubl_u8(vget_low_u8(v_src01), vget_low_u8(v_src11)));
            vst1q_u16(dstu16 + j + 24, vsubl_u8(vget_high_u8(v_src01), vget_high_u8(v_src11)));
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v_src0 = vld1_u8(src0 + j);
            uint8x8_t v_src1 = vld1_u8(src1 + j);
            vst1q_u16(dstu16 + j, vsubl_u8(v_src0, v_src1));
        }

        for (; j < size.width; j++)
            dst[j] = (s16)src0[j] - (s16)src1[j];
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

void sub(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         f32 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src0 + j);
            internal::prefetch(src1 + j);
            uint8x16_t v_src00 = vld1q_u8(src0 + j), v_src01 = vld1q_u8(src0 + j + 16);
            uint8x16_t v_src10 = vld1q_u8(src1 + j), v_src11 = vld1q_u8(src1 + j + 16);
            int16x8_t vsl = vreinterpretq_s16_u16(vsubl_u8( vget_low_u8(v_src00),  vget_low_u8(v_src10)));
            int16x8_t vsh = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(v_src00), vget_high_u8(v_src10)));

            vst1q_f32(dst + j +  0, vcvtq_f32_s32(vmovl_s16(  vget_low_s16(vsl) )));
            vst1q_f32(dst + j +  4, vcvtq_f32_s32(vmovl_s16( vget_high_s16(vsl) )));
            vst1q_f32(dst + j +  8, vcvtq_f32_s32(vmovl_s16(  vget_low_s16(vsh) )));
            vst1q_f32(dst + j + 12, vcvtq_f32_s32(vmovl_s16( vget_high_s16(vsh) )));

            vsl = vreinterpretq_s16_u16(vsubl_u8( vget_low_u8(v_src01),  vget_low_u8(v_src11)));
            vsh = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(v_src01), vget_high_u8(v_src11)));

            vst1q_f32(dst + j + 16, vcvtq_f32_s32(vmovl_s16(  vget_low_s16(vsl) )));
            vst1q_f32(dst + j + 20, vcvtq_f32_s32(vmovl_s16( vget_high_s16(vsl) )));
            vst1q_f32(dst + j + 24, vcvtq_f32_s32(vmovl_s16(  vget_low_s16(vsh) )));
            vst1q_f32(dst + j + 28, vcvtq_f32_s32(vmovl_s16( vget_high_s16(vsh) )));
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v_src0 = vld1_u8(src0 + j);
            uint8x8_t v_src1 = vld1_u8(src1 + j);

            int16x8_t vs = vreinterpretq_s16_u16(vsubl_u8(v_src0, v_src1));
            vst1q_f32(dst + j + 0, vcvtq_f32_s32(vmovl_s16(  vget_low_s16(vs) )));
            vst1q_f32(dst + j + 4, vcvtq_f32_s32(vmovl_s16( vget_high_s16(vs) )));
        }
        for(; j < size.width; j++)
            dst[j] = (f32)src0[j] - (f32)src1[j];
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

void sub(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const s16 * src1Base, ptrdiff_t src1Stride,
         s16 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const s16 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (policy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                uint8x16_t v_src0 = vld1q_u8(src0 + j);
                int16x8_t v_src00 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src0)));
                int16x8_t v_src01 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src0)));
                int16x8_t v_src10 = vld1q_s16(src1 + j), v_src11 = vld1q_s16(src1 + j + 8);
                int16x8_t v_dst0 = vqsubq_s16(v_src00, v_src10);
                int16x8_t v_dst1 = vqsubq_s16(v_src01, v_src11);
                vst1q_s16(dst + j, v_dst0);
                vst1q_s16(dst + j + 8, v_dst1);
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src0 + j)));
                int16x8_t v_src1 = vld1q_s16(src1 + j);
                int16x8_t v_dst = vqsubq_s16(v_src0, v_src1);
                vst1q_s16(dst + j, v_dst);
            }

            for (; j < size.width; j++)
                dst[j] = internal::saturate_cast<s16>((s32)src0[j] - (s32)src1[j]);
        }
        else
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                uint8x16_t v_src0 = vld1q_u8(src0 + j);
                int16x8_t v_src00 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src0)));
                int16x8_t v_src01 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src0)));
                int16x8_t v_src10 = vld1q_s16(src1 + j), v_src11 = vld1q_s16(src1 + j + 8);
                int16x8_t v_dst0 = vsubq_s16(v_src00, v_src10);
                int16x8_t v_dst1 = vsubq_s16(v_src01, v_src11);
                vst1q_s16(dst + j, v_dst0);
                vst1q_s16(dst + j + 8, v_dst1);
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src0 + j)));
                int16x8_t v_src1 = vld1q_s16(src1 + j);
                int16x8_t v_dst = vsubq_s16(v_src0, v_src1);
                vst1q_s16(dst + j, v_dst);
            }

            for (; j < size.width; j++)
                dst[j] = (s16)((s32)src0[j] - (s32)src1[j]);
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
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const s16 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         s16 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (policy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                int16x8_t v_src00 = vld1q_s16(src0 + j), v_src01 = vld1q_s16(src0 + j + 8);
                uint8x16_t v_src1 = vld1q_u8(src1 + j);
                int16x8_t v_src10 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src1)));
                int16x8_t v_src11 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src1)));
                int16x8_t v_dst0 = vqsubq_s16(v_src00, v_src10);
                int16x8_t v_dst1 = vqsubq_s16(v_src01, v_src11);
                vst1q_s16(dst + j, v_dst0);
                vst1q_s16(dst + j + 8, v_dst1);
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src0 = vld1q_s16(src0 + j);
                int16x8_t v_src1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src1 + j)));
                int16x8_t v_dst = vqsubq_s16(v_src0, v_src1);
                vst1q_s16(dst + j, v_dst);
            }

            for (; j < size.width; j++)
                dst[j] = internal::saturate_cast<s16>((s32)src0[j] - (s32)src1[j]);
        }
        else
        {
            for (; j < roiw16; j += 16)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);
                int16x8_t v_src00 = vld1q_s16(src0 + j), v_src01 = vld1q_s16(src0 + j + 8);
                uint8x16_t v_src1 = vld1q_u8(src1 + j);
                int16x8_t v_src10 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src1)));
                int16x8_t v_src11 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src1)));
                int16x8_t v_dst0 = vsubq_s16(v_src00, v_src10);
                int16x8_t v_dst1 = vsubq_s16(v_src01, v_src11);
                vst1q_s16(dst + j, v_dst0);
                vst1q_s16(dst + j + 8, v_dst1);
            }
            for (; j < roiw8; j += 8)
            {
                int16x8_t v_src0 = vld1q_s16(src0 + j);
                int16x8_t v_src1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src1 + j)));
                int16x8_t v_dst = vsubq_s16(v_src0, v_src1);
                vst1q_s16(dst + j, v_dst);
            }

            for (; j < size.width; j++)
                dst[j] = (s16)((s32)src0[j] - (s32)src1[j]);
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
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const s8 * src0Base, ptrdiff_t src0Stride,
         const s8 * src1Base, ptrdiff_t src1Stride,
         s8 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (policy == CONVERT_POLICY_SATURATE)
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubSaturate<s8, s16>());
    }
    else
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubWrap<s8, s16>());
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const s16 * src0Base, ptrdiff_t src0Stride,
         const s16 * src1Base, ptrdiff_t src1Stride,
         s16 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (policy == CONVERT_POLICY_SATURATE)
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubSaturate<s16, s32>());
    }
    else
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubWrap<s16, s32>());
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const u16 * src0Base, ptrdiff_t src0Stride,
         const u16 * src1Base, ptrdiff_t src1Stride,
         u16 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (policy == CONVERT_POLICY_SATURATE)
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubSaturate<u16, s32>());
    }
    else
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubWrap<u16, s32>());
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const s32 * src0Base, ptrdiff_t src0Stride,
         const s32 * src1Base, ptrdiff_t src1Stride,
         s32 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (policy == CONVERT_POLICY_SATURATE)
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubSaturate<s32, s64>());
    }
    else
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubWrap<s32, s64>());
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const u32 * src0Base, ptrdiff_t src0Stride,
         const u32 * src1Base, ptrdiff_t src1Stride,
         u32 *dstBase, ptrdiff_t dstStride,
         CONVERT_POLICY policy)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (policy == CONVERT_POLICY_SATURATE)
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubSaturate<u32, s64>());
    }
    else
    {
        internal::vtransform(size,
                             src0Base, src0Stride,
                             src1Base, src1Stride,
                             dstBase, dstStride,
                             SubWrap<u32, s64>());
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)policy;
#endif
}

void sub(const Size2D &size,
         const f32 * src0Base, ptrdiff_t src0Stride,
         const f32 * src1Base, ptrdiff_t src1Stride,
         f32 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride,
                         SubWrap<f32, f32>());
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

} // namespace CAROTENE_NS
