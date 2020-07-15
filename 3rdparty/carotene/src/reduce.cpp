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

#include "common.hpp"

#include <cstring>

namespace CAROTENE_NS {

void reduceColSum(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  s32 * dstBase)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    memset(dstBase, 0, size.width*sizeof(s32));
    size_t i = 0;
    for (; i + 16 <= size.width; i += 16)
    {
        const u8* src_address = srcBase + i;

        int32x4_t sll = vmovq_n_s32(0);
        int32x4_t slh = vmovq_n_s32(0);
        int32x4_t shl = vmovq_n_s32(0);
        int32x4_t shh = vmovq_n_s32(0);

        for (size_t h = 0; h < size.height; h += 256)
        {
            size_t lim = std::min(h + 256, size.height);

            uint16x8_t sl = vmovq_n_u16(0);
            uint16x8_t sh = vmovq_n_u16(0);

            for (size_t k = h; k < lim; ++k, src_address += srcStride)
            {
                internal::prefetch(src_address + srcStride, 0);

                uint8x16_t v = vld1q_u8(src_address);

                sl = vaddw_u8(sl, vget_low_u8(v));
                sh = vaddw_u8(sh, vget_high_u8(v));
            }

            int32x4_t vsll = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sl)));
            int32x4_t vslh = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sl)));
            int32x4_t vshl = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sh)));
            int32x4_t vshh = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sh)));

            sll = vqaddq_s32(sll, vsll);
            slh = vqaddq_s32(slh, vslh);
            shl = vqaddq_s32(shl, vshl);
            shh = vqaddq_s32(shh, vshh);
        }

        vst1q_s32(dstBase + i + 0, sll);
        vst1q_s32(dstBase + i + 4, slh);
        vst1q_s32(dstBase + i + 8, shl);
        vst1q_s32(dstBase + i + 12, shh);
    }

    for(size_t h = 0; h < size.height; ++h)
    {
        for(size_t j = i ; j < size.width; j++ )
        {
            if (((u32)(dstBase[j] += srcBase[j + srcStride * h])) > 0x7fFFffFFu)
                dstBase[j] = 0x7fFFffFF;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
#endif
}

void reduceColMax(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    memcpy(dstBase, srcBase, size.width);
    size_t i = 0;
    for (; i + 16*4 <= size.width; i += 16*4)
    {
        const u8* src_address = srcBase + i;

        uint8x16_t s1 = vld1q_u8(src_address + 0);
        uint8x16_t s2 = vld1q_u8(src_address + 16);
        uint8x16_t s3 = vld1q_u8(src_address + 32);
        uint8x16_t s4 = vld1q_u8(src_address + 48);

        src_address += srcStride;

        for(size_t h = 1; h < size.height; ++h, src_address += srcStride)
        {
            internal::prefetch(src_address + srcStride, 0);
            internal::prefetch(src_address + srcStride, 32);

            uint8x16_t v1 = vld1q_u8(src_address + 0);
            uint8x16_t v2 = vld1q_u8(src_address + 16);
            uint8x16_t v3 = vld1q_u8(src_address + 32);
            uint8x16_t v4 = vld1q_u8(src_address + 48);

            s1 = vmaxq_u8(s1, v1);
            s2 = vmaxq_u8(s2, v2);
            s3 = vmaxq_u8(s3, v3);
            s4 = vmaxq_u8(s4, v4);
        }

        vst1q_u8(dstBase + i + 0, s1);
        vst1q_u8(dstBase + i + 16, s2);
        vst1q_u8(dstBase + i + 32, s3);
        vst1q_u8(dstBase + i + 48, s4);
    }

    for (; i + 16 <= size.width; i += 16)
    {
        const u8* src_address = srcBase + i;
        uint8x16_t s1 = vld1q_u8(src_address);
        src_address += srcStride;
        for(size_t h = 1; h < size.height; ++h, src_address += srcStride)
        {
            internal::prefetch(src_address + srcStride, 0);

            uint8x16_t v1 = vld1q_u8(src_address);
            s1 = vmaxq_u8(s1, v1);
        }
        vst1q_u8(dstBase + i, s1);
    }

    if (i < size.width)
        for(size_t h = 1; h < size.height; ++h)
            for(size_t j = i ; j < size.width; j++ )
                dstBase[j] = std::max(dstBase[j], srcBase[j + srcStride * h]);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
#endif
}

void reduceColMin(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    memcpy(dstBase, srcBase, size.width);
    size_t i = 0;
    for (; i + 16*4 <= size.width; i += 16*4)
    {
        const u8* src_address = srcBase + i;

        uint8x16_t s1 = vld1q_u8(src_address + 0);
        uint8x16_t s2 = vld1q_u8(src_address + 16);
        uint8x16_t s3 = vld1q_u8(src_address + 32);
        uint8x16_t s4 = vld1q_u8(src_address + 48);

        src_address += srcStride;

        for(size_t h = 1; h < size.height; ++h, src_address += srcStride)
        {
            internal::prefetch(src_address + srcStride, 0);
            internal::prefetch(src_address + srcStride, 32);

            uint8x16_t v1 = vld1q_u8(src_address + 0);
            uint8x16_t v2 = vld1q_u8(src_address + 16);
            uint8x16_t v3 = vld1q_u8(src_address + 32);
            uint8x16_t v4 = vld1q_u8(src_address + 48);

            s1 = vminq_u8(s1, v1);
            s2 = vminq_u8(s2, v2);
            s3 = vminq_u8(s3, v3);
            s4 = vminq_u8(s4, v4);
        }

        vst1q_u8(dstBase + i + 0, s1);
        vst1q_u8(dstBase + i + 16, s2);
        vst1q_u8(dstBase + i + 32, s3);
        vst1q_u8(dstBase + i + 48, s4);
    }

    for (; i + 16 <= size.width; i += 16)
    {
        const u8* src_address = srcBase + i;
        uint8x16_t s1 = vld1q_u8(src_address);
        src_address += srcStride;
        for(size_t h = 1; h < size.height; ++h, src_address += srcStride)
        {
            internal::prefetch(src_address + srcStride, 0);

            uint8x16_t v1 = vld1q_u8(src_address);
            s1 = vminq_u8(s1, v1);
        }
        vst1q_u8(dstBase + i, s1);
    }

    if (i < size.width)
        for(size_t h = 1; h < size.height; ++h)
            for(size_t j = i ; j < size.width; j++ )
                dstBase[j] = std::min(dstBase[j], srcBase[j + srcStride * h]);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
#endif
}

void reduceColSum(const Size2D &size,
                  const f32 * srcBase, ptrdiff_t srcStride,
                  f32 * dstBase)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    memcpy(dstBase, srcBase, size.width*sizeof(f32));
    size_t srcstep = srcStride/sizeof(f32);
    size_t i = 0;
    for (; i + 16 <= size.width; i += 16)
    {
        const f32* src_address = srcBase + i;

        float32x4_t s1 = vld1q_f32(src_address + 0);
        float32x4_t s2 = vld1q_f32(src_address + 4);
        float32x4_t s3 = vld1q_f32(src_address + 8);
        float32x4_t s4 = vld1q_f32(src_address + 12);

        src_address += srcstep;

        for(size_t h = 1; h < size.height; ++h, src_address += srcstep)
        {
            internal::prefetch(src_address + srcstep, 0);
            internal::prefetch(src_address + srcstep, 32);

            float32x4_t v1 = vld1q_f32(src_address + 0);
            float32x4_t v2 = vld1q_f32(src_address + 4);
            float32x4_t v3 = vld1q_f32(src_address + 8);
            float32x4_t v4 = vld1q_f32(src_address + 12);

            s1 = vaddq_f32(s1, v1);
            s2 = vaddq_f32(s2, v2);
            s3 = vaddq_f32(s3, v3);
            s4 = vaddq_f32(s4, v4);
        }

        vst1q_f32(dstBase + i + 0, s1);
        vst1q_f32(dstBase + i + 4, s2);
        vst1q_f32(dstBase + i + 8, s3);
        vst1q_f32(dstBase + i + 12, s4);
    }

    for (; i + 4 <= size.width; i += 4)
    {
        const f32* src_address = srcBase + i;
        float32x4_t s1 = vld1q_f32(src_address);
        src_address += srcstep;
        for(size_t h = 1; h < size.height; ++h, src_address += srcstep)
        {
            internal::prefetch(src_address + srcstep, 0);

            float32x4_t v1 = vld1q_f32(src_address);
            s1 = vaddq_f32(s1, v1);
        }
        vst1q_f32(dstBase + i, s1);
    }

    if (i < size.width)
        for(size_t h = 1; h < size.height; ++h)
        {
            for(size_t j = i ; j < size.width; j++ )
            {
                dstBase[j] += srcBase[j + srcstep * h];
            }
        }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
#endif
}

void reduceColMax(const Size2D &size,
                  const f32 * srcBase, ptrdiff_t srcStride,
                  f32 * dstBase)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    memcpy(dstBase, srcBase, size.width*sizeof(f32));
    size_t srcstep = srcStride/sizeof(f32);
    size_t i = 0;
    for (; i + 16 <= size.width; i += 16)
    {
        const f32* src_address = srcBase + i;

        float32x4_t s1 = vld1q_f32(src_address + 0);
        float32x4_t s2 = vld1q_f32(src_address + 4);
        float32x4_t s3 = vld1q_f32(src_address + 8);
        float32x4_t s4 = vld1q_f32(src_address + 12);

        src_address += srcstep;

        for(size_t h = 1; h < size.height; ++h, src_address += srcstep)
        {
            internal::prefetch(src_address + srcstep, 0);
            internal::prefetch(src_address + srcstep, 32);

            float32x4_t v1 = vld1q_f32(src_address + 0);
            float32x4_t v2 = vld1q_f32(src_address + 4);
            float32x4_t v3 = vld1q_f32(src_address + 8);
            float32x4_t v4 = vld1q_f32(src_address + 12);

            s1 = vmaxq_f32(s1, v1);
            s2 = vmaxq_f32(s2, v2);
            s3 = vmaxq_f32(s3, v3);
            s4 = vmaxq_f32(s4, v4);
        }

        vst1q_f32(dstBase + i + 0, s1);
        vst1q_f32(dstBase + i + 4, s2);
        vst1q_f32(dstBase + i + 8, s3);
        vst1q_f32(dstBase + i + 12, s4);
    }

    for (; i + 4 <= size.width; i += 4)
    {
        const f32* src_address = srcBase + i;
        float32x4_t s1 = vld1q_f32(src_address);
        src_address += srcstep;
        for(size_t h = 1; h < size.height; ++h, src_address += srcstep)
        {
            internal::prefetch(src_address + srcstep, 0);

            float32x4_t v1 = vld1q_f32(src_address);
            s1 = vmaxq_f32(s1, v1);
        }
        vst1q_f32(dstBase + i, s1);
    }

    if (i < size.width)
        for(size_t h = 1; h < size.height; ++h)
            for(size_t j = i ; j < size.width; j++ )
                dstBase[j] = std::max(dstBase[j], srcBase[j + srcstep * h]);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
#endif
}

void reduceColMin(const Size2D &size,
                  const f32 * srcBase, ptrdiff_t srcStride,
                  f32 * dstBase)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    memcpy(dstBase, srcBase, size.width*sizeof(f32));
    size_t srcstep = srcStride/sizeof(f32);
    size_t i = 0;
    for (; i + 16 <= size.width; i += 16)
    {
        const f32* src_address = srcBase + i;

        float32x4_t s1 = vld1q_f32(src_address + 0);
        float32x4_t s2 = vld1q_f32(src_address + 4);
        float32x4_t s3 = vld1q_f32(src_address + 8);
        float32x4_t s4 = vld1q_f32(src_address + 12);

        src_address += srcstep;

        for(size_t h = 1; h < size.height; ++h, src_address += srcstep)
        {
            internal::prefetch(src_address + srcstep, 0);
            internal::prefetch(src_address + srcstep, 32);

            float32x4_t v1 = vld1q_f32(src_address + 0);
            float32x4_t v2 = vld1q_f32(src_address + 4);
            float32x4_t v3 = vld1q_f32(src_address + 8);
            float32x4_t v4 = vld1q_f32(src_address + 12);

            s1 = vminq_f32(s1, v1);
            s2 = vminq_f32(s2, v2);
            s3 = vminq_f32(s3, v3);
            s4 = vminq_f32(s4, v4);
        }

        vst1q_f32(dstBase + i + 0, s1);
        vst1q_f32(dstBase + i + 4, s2);
        vst1q_f32(dstBase + i + 8, s3);
        vst1q_f32(dstBase + i + 12, s4);
    }

    for (; i + 4 <= size.width; i += 4)
    {
        const f32* src_address = srcBase + i;
        float32x4_t s1 = vld1q_f32(src_address);
        src_address += srcstep;
        for(size_t h = 1; h < size.height; ++h, src_address += srcstep)
        {
            internal::prefetch(src_address + srcstep, 0);

            float32x4_t v1 = vld1q_f32(src_address);
            s1 = vminq_f32(s1, v1);
        }
        vst1q_f32(dstBase + i, s1);
    }

    if (i < size.width)
        for(size_t h = 1; h < size.height; ++h)
            for(size_t j = i ; j < size.width; j++ )
                dstBase[j] = std::min(dstBase[j], srcBase[j + srcstep * h]);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
#endif
}

} // namespace CAROTENE_NS
