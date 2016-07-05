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

namespace CAROTENE_NS {

//magic number; must be multiple of 4
#define NORM32F_BLOCK_SIZE 2048

s32 normInf(const Size2D &_size,
            const u8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        if (size.width >= 16)
        {
            uint8x16_t s = vld1q_u8(src);
            for (i = 16; i <= size.width - 16; i += 16)
            {
                internal::prefetch(src + i);
                uint8x16_t s1 = vld1q_u8(src + i);
                s = vmaxq_u8(s1, s);
            }
            u8 s2[8];
            uint8x8_t s3 = vmax_u8(vget_low_u8(s), vget_high_u8(s));
            vst1_u8(s2, s3);
            for (u32 j = 0; j < 8; j++)
                result = std::max((s32)(s2[j]), result);
        }
        for ( ; i < size.width; i++)
            result = std::max((s32)(src[i]), result);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normInf(const Size2D &_size,
            const s8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        if (size.width >= 16)
        {
            uint8x16_t s = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(src)));
            for (i = 16; i <= size.width - 16; i += 16)
            {
                internal::prefetch(src + i);
                uint8x16_t s1 = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(src + i)));
                s = vmaxq_u8(s1, s);
            }
            u8 s2[8];
            uint8x8_t s3 = vmax_u8(vget_low_u8(s), vget_high_u8(s));
            vst1_u8(s2, s3);
            for (u32 j = 0; j < 8; j++)
                result = std::max((s32)(s2[j]), result);
        }
        for ( ; i < size.width; i++)
            result = std::max((s32)(std::abs(src[i])), result);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normInf(const Size2D &_size,
            const u16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        if (size.width >= 8)
        {
            uint16x8_t s = vld1q_u16(src);
            for (i = 8; i <= size.width - 8; i += 8)
            {
                internal::prefetch(src + i);
                uint16x8_t s1 = vld1q_u16(src + i);
                s = vmaxq_u16(s1, s);
            }
            u16 s2[4];
            uint16x4_t s3 = vmax_u16(vget_low_u16(s), vget_high_u16(s));
            vst1_u16(s2, s3);
            for (u32 j = 0; j < 4; j++)
                result = std::max((s32)(s2[j]), result);
        }
        for ( ; i < size.width; i++)
            result = std::max((s32)(src[i]), result);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normInf(const Size2D &_size,
            const s16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        if (size.width >= 8)
        {
            uint16x8_t s = vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(src)));
            for (i = 8; i <= size.width - 8; i += 8)
            {
                internal::prefetch(src + i);
                uint16x8_t s1 = vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(src + i)));
                s = vmaxq_u16(s1, s);
            }
            u16 s2[4];
            uint16x4_t s3 = vmax_u16(vget_low_u16(s), vget_high_u16(s));
            vst1_u16(s2, s3);
            for (u32 j = 0; j < 4; j++)
                result = std::max((s32)(s2[j]), result);
        }
        for ( ; i < size.width; i++)
            result = std::max(std::abs((s32)(src[i])), result);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normInf(const Size2D &_size,
            const s32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        if (size.width >= 4)
        {
            uint32x4_t s = vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(src)));
            for (i = 4; i <= size.width - 4; i += 4)
            {
                internal::prefetch(src + i);
                uint32x4_t s1 = vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(src + i)));
                s = vmaxq_u32(s1, s);
            }
            u32 s2[2];
            uint32x2_t s3 = vmax_u32(vget_low_u32(s), vget_high_u32(s));
            vst1_u32(s2, s3);
            for (u32 j = 0; j < 2; j++)
                result = std::max((s32)(s2[j]), result);
        }
        for ( ; i < size.width; i++)
            result = std::max((s32)(std::abs(src[i])), result);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

f32 normInf(const Size2D &_size,
            const f32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    f32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        if (size.width >= 4)
        {
            float32x4_t s = vabsq_f32(vld1q_f32(src));
            for (i = 4; i <= size.width - 4; i += 4 )
            {
                internal::prefetch(src + i);
                float32x4_t s1 = vld1q_f32(src + i);
                float32x4_t sa = vabsq_f32(s1);
                s = vmaxq_f32(sa, s);
            }
            f32 s2[2];
            float32x2_t s3 = vmax_f32(vget_low_f32(s), vget_high_f32(s));
            vst1_f32(s2, s3);
            for (u32 j = 0; j < 2; j++)
                result = std::max(s2[j], result);
        }
        for (; i < size.width; i++)
            result = std::max(std::abs(src[i]), result);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

s32 normL1(const Size2D &_size,
           const u8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        uint32x4_t vs = vmovq_n_u32(0);
        for (; i < roiw8;)
        {
            size_t limit = std::min(size.width, i + 256) - 8;
            uint8x8_t s0 = vld1_u8(src + i);
            uint16x8_t s = vmovl_u8(s0);

            for (i += 8; i <= limit; i += 8)
            {
                internal::prefetch(src + i);
                uint8x8_t s1 = vld1_u8(src + i);
                s = vaddw_u8(s, s1);
            }

            uint16x4_t s4 = vadd_u16(vget_low_u16(s), vget_high_u16(s));
            vs = vaddw_u16(vs, s4);
        }

        u32 s2[2];
        uint32x2_t vs2 = vadd_u32(vget_low_u32(vs), vget_high_u32(vs));
        vst1_u32(s2, vs2);

        result += (s32)(s2[0] + s2[1]);

        for ( ; i < size.width; i++)
            result += (s32)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normL1(const Size2D &_size,
           const s8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        uint32x4_t vs = vmovq_n_u32(0);

        for (; i < roiw8;)
        {
            size_t limit = std::min(size.width, i + 256) - 8;
            uint8x8_t s0 = vreinterpret_u8_s8(vabs_s8(vld1_s8(src + i)));
            uint16x8_t s = vmovl_u8(s0);

            for (i += 8; i <= limit; i += 8)
            {
                internal::prefetch(src + i);
                uint8x8_t s1 = vreinterpret_u8_s8(vabs_s8(vld1_s8(src + i)));
                s = vaddw_u8(s, s1);
            }

            uint16x4_t s4 = vadd_u16(vget_low_u16(s), vget_high_u16(s));
            vs = vaddw_u16(vs, s4);
        }

        u32 s2[2];
        uint32x2_t vs2 = vadd_u32(vget_low_u32(vs), vget_high_u32(vs));
        vst1_u32(s2, vs2);

        result += (s32)(s2[0] + s2[1]);

        for ( ; i < size.width; i++)
            result += (s32)(std::abs(src[i]));
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normL1(const Size2D &_size,
           const u16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        uint32x4_t vs = vmovq_n_u32(0);
        for (; i < roiw4; i += 4)
        {
            internal::prefetch(src + i);
            uint16x4_t s = vld1_u16(src + i);
            vs = vaddw_u16(vs, s);
        }
        u32 s2[4];
        vst1q_u32(s2, vs);
        for (u32 j = 0; j < 4; j++)
            result += s2[j];
        for ( ; i < size.width; i++)
            result += (s32)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normL1(const Size2D &_size,
           const s16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        uint32x4_t vs = vmovq_n_u32(0);
        for (; i < roiw4; i += 4)
        {
            internal::prefetch(src + i);
            uint16x4_t s = vreinterpret_u16_s16(vabs_s16(vld1_s16(src + i)));
            vs = vaddw_u16(vs, s);
        }
        u32 s2[4];
        vst1q_u32(s2, vs);
        for (u32 j = 0; j < 4; j++)
            result += s2[j];
        for ( ; i < size.width; i++)
            result += (s32)(std::abs(src[i]));
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

f64 normL1(const Size2D &_size,
           const s32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        for (; i < roiw4;)
        {
            size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            float32x4_t s = vcvtq_f32_s32(vabsq_s32(vld1q_s32(src + i)));
            for (i += 4; i <= limit; i += 4 )
            {
                internal::prefetch(src + i);
                float32x4_t s1 = vcvtq_f32_s32(vabsq_s32(vld1q_s32(src + i)));
                s = vaddq_f32(s, s1);
            }

            f32 s2[4];
            vst1q_f32(s2, s);

            for (u32 j = 0; j < 4; j++)
                result += (f64)(s2[j]);
        }
        for ( ; i < size.width; i++)
            result += (f64)(std::abs(src[i]));
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

f64 normL1(const Size2D &_size,
           const f32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        for (; i < roiw4;)
        {
            size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            float32x4_t s = vabsq_f32(vld1q_f32(src + i));
            for (i += 4; i <= limit; i += 4)
            {
                internal::prefetch(src + i);
                float32x4_t s1 = vld1q_f32(src + i);
                float32x4_t sa = vabsq_f32(s1);
                s = vaddq_f32(sa, s);
            }

            f32 s2[4];
            vst1q_f32(s2, s);

            for (u32 j = 0; j < 4; j++)
                result += (f64)(s2[j]);
        }
        for (; i < size.width; i++)
            result += std::abs((f64)(src[i]));
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

s32 normL2(const Size2D &_size,
           const u8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        uint32x4_t sl = vmovq_n_u32(0);
        uint32x4_t sh = vmovq_n_u32(0);

        for (; i < roiw8; i += 8)
        {
            internal::prefetch(src + i);
            uint8x8_t s1 = vld1_u8(src + i);
            uint16x8_t sq = vmull_u8(s1, s1);

            sl = vaddw_u16(sl, vget_low_u16(sq));
            sh = vaddw_u16(sh, vget_high_u16(sq));
        }

        uint32x4_t s = vaddq_u32(sl, sh);
        uint32x2_t ss = vadd_u32(vget_low_u32(s), vget_high_u32(s));

        u32 s2[2];
        vst1_u32(s2, ss);

        result += (s32)(s2[0] + s2[1]);

        for (; i < size.width; i++)
            result += (s32)(src[i]) * (s32)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 normL2(const Size2D &_size,
           const s8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        int32x4_t sl = vmovq_n_s32(0);
        int32x4_t sh = vmovq_n_s32(0);

        for (; i < roiw8; i += 8)
        {
            internal::prefetch(src + i);
            int8x8_t s1 = vld1_s8(src + i);
            int16x8_t sq = vmull_s8(s1, s1);

            sl = vaddw_s16(sl, vget_low_s16(sq));
            sh = vaddw_s16(sh, vget_high_s16(sq));
        }

        int32x4_t s = vaddq_s32(sl, sh);
        int32x2_t ss = vadd_s32(vget_low_s32(s), vget_high_s32(s));

        s32 s2[2];
        vst1_s32(s2, ss);

        result += s2[0] + s2[1];

        for (; i < size.width; i++)
            result += (s32)(src[i]) * (s32)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

f64 normL2(const Size2D &_size,
           const u16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        for (; i < roiw4;)
        {
            size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            uint16x4_t s0 = vld1_u16(src+i);
            float32x4_t s = vcvtq_f32_u32(vmull_u16(s0,s0));
            for (i += 4; i <= limit; i += 4 )
            {
                internal::prefetch(src + i);
                uint16x4_t s1 = vld1_u16(src+i);
                float32x4_t sq = vcvtq_f32_u32(vmull_u16(s1, s1));
                s = vaddq_f32(s, sq);
            }
            f32 s2[4];
            vst1q_f32(s2, s);
            for (u32 j = 0; j < 4; j++)
                result += (f64)(s2[j]);
        }

        for ( ; i < size.width; i++)
            result += (f64)(src[i]) * (f64)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

f64 normL2(const Size2D &_size,
           const s16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        for (; i < roiw4;)
        {
            size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            int16x4_t s0 = vld1_s16(src+i);
            float32x4_t s = vcvtq_f32_s32(vmull_s16(s0,s0));
            for (i += 4; i <= limit; i += 4 )
            {
                internal::prefetch(src + i);
                int16x4_t s1 = vld1_s16(src+i);
                float32x4_t sq = vcvtq_f32_s32(vmull_s16(s1, s1));
                s = vaddq_f32(s, sq);
            }
            f32 s2[4];
            vst1q_f32(s2, s);
            for (u32 j = 0; j < 4; j++)
                result += (f64)(s2[j]);
        }

        for ( ; i < size.width; i++)
            result += (f64)(src[i]) * (f64)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

f64 normL2(const Size2D &_size,
           const s32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        for (; i < roiw4;)
        {
            size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            float32x4_t s = vcvtq_f32_s32(vld1q_s32(src + i));
            s = vmulq_f32(s, s);
            for (i += 4; i <= limit; i += 4 )
            {
                internal::prefetch(src + i);
                float32x4_t s1 = vcvtq_f32_s32(vld1q_s32(src + i));
                s = vmlaq_f32(s, s1, s1);
            }

            f32 s2[4];
            vst1q_f32(s2, s);

            for (u32 j = 0; j < 4; j++)
                result += (f64)(s2[j]);
        }
        for ( ; i < size.width; i++)
            result += (f64)(src[i]) * (f64)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

f64 normL2(const Size2D &_size,
           const f32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width >= 3 ? size.width - 3 : 0;
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;
        for (; i < roiw4;)
        {
            size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            float32x4_t s = vld1q_f32(src + i);
            s = vmulq_f32(s, s);
            for (i += 4; i <= limit; i += 4 )
            {
                internal::prefetch(src + i);
                float32x4_t s1 = vld1q_f32(src + i);
                s = vmlaq_f32(s, s1, s1);
            }

            f32 s2[4];
            vst1q_f32(s2, s);

            for (u32 j = 0; j < 4; j++)
                result += (f64)(s2[j]);
        }
        for ( ; i < size.width; i++)
            result += (f64)(src[i]) * (f64)(src[i]);
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0.;
#endif
}

s32 diffNormInf(const Size2D &_size,
                const u8 * src0Base, ptrdiff_t src0Stride,
                const u8 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src1 = internal::getRowPtr( src0Base,  src0Stride, k);
        const u8* src2 = internal::getRowPtr( src1Base,  src1Stride, k);
        size_t i = 0;

        if (size.width >= 16)
        {
            uint8x16_t vs3 = vdupq_n_u8(0);
            for (; i < size.width - 16; i += 16)
            {
                internal::prefetch(src1 + i);
                internal::prefetch(src2 + i);

                uint8x16_t vs1 = vld1q_u8(src1 + i);
                uint8x16_t vs2 = vld1q_u8(src2 + i);

                vs3 = vmaxq_u8(vs3, vabdq_u8(vs1, vs2));
            }

            u8 s2[8];
            vst1_u8(s2, vpmax_u8(vget_low_u8(vs3), vget_high_u8(vs3)));

            for (u32 j = 0; j < 8; j++)
                result = std::max((s32)(s2[j]), result);
        }

        for (; i < size.width; i++)
        {
            result = std::max(std::abs((s32)(src1[i]) - (s32)(src2[i])), result);
        }
    }
    return result;
#else
    (void)_size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;

    return 0;
#endif
}

f32 diffNormInf(const Size2D &_size,
                const f32 * src0Base, ptrdiff_t src0Stride,
                const f32 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    f32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src1 = internal::getRowPtr( src0Base,  src0Stride, k);
        const f32* src2 = internal::getRowPtr( src1Base,  src1Stride, k);
        size_t i = 0;

        if (size.width >= 4)
        {
            float32x4_t s = vabdq_f32(vld1q_f32(src1), vld1q_f32(src2));

            for (i += 4; i <= size.width - 4; i += 4 )
            {
                internal::prefetch(src1 + i);
                internal::prefetch(src2 + i);

                float32x4_t vs1 = vld1q_f32(src1 + i);
                float32x4_t vs2 = vld1q_f32(src2 + i);

                float32x4_t vd = vabdq_f32(vs2, vs1);
                s = vmaxq_f32(s, vd);
            }

            f32 s2[4];
            vst1q_f32(s2, s);

            for (u32 j = 0; j < 4; j++)
                if (s2[j] > result)
                    result = s2[j];
        }

        for (; i < size.width; i++)
        {
            f32 v = std::abs(src1[i] - src2[i]);
            if (v > result)
                result = v;
        }
    }
    return result;
#else
    (void)_size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;

    return 0.;
#endif
}

s32 diffNormL1(const Size2D &_size,
               const u8 * src0Base, ptrdiff_t src0Stride,
               const u8 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src1 = internal::getRowPtr( src0Base,  src0Stride, k);
        const u8* src2 = internal::getRowPtr( src1Base,  src1Stride, k);
        size_t i = 0;

        if (size.width >= 16)
        {
            for(; i <= size.width - 16;)
            {
                size_t limit = std::min(size.width, i + 2*256) - 16;
                uint16x8_t si1 = vmovq_n_u16(0);
                uint16x8_t si2 = vmovq_n_u16(0);

                for (; i <= limit; i += 16)
                {
                    internal::prefetch(src1 + i);
                    internal::prefetch(src2 + i);

                    uint8x16_t vs1 = vld1q_u8(src1 + i);
                    uint8x16_t vs2 = vld1q_u8(src2 + i);

                    si1 = vabal_u8(si1, vget_low_u8(vs1), vget_low_u8(vs2));
                    si2 = vabal_u8(si2, vget_high_u8(vs1), vget_high_u8(vs2));
                }

                u32 s2[4];
                vst1q_u32(s2, vaddq_u32(vpaddlq_u16(si1), vpaddlq_u16(si2)));

                for (u32 j = 0; j < 4; j++)
                {
                    if ((s32)(0x7fFFffFFu - s2[j]) <= result)
                    {
                        return 0x7fFFffFF; //result already saturated
                    }
                    result = (s32)((u32)(result) + s2[j]);
                }
            }

        }

        for (; i < size.width; i++)
        {
            u32 v = std::abs((s32)(src1[i]) - (s32)(src2[i]));

            if ((s32)(0x7fFFffFFu - v) <= result)
            {
                return 0x7fFFffFF; //result already saturated
            }
            result = (s32)((u32)(result) + v);
        }
    }
    return result;
#else
    (void)_size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;

    return 0;
#endif
}

f64 diffNormL1(const Size2D &_size,
               const f32 * src0Base, ptrdiff_t src0Stride,
               const f32 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src1 = internal::getRowPtr( src0Base,  src0Stride, k);
        const f32* src2 = internal::getRowPtr( src1Base,  src1Stride, k);
        size_t i = 0;

        if (size.width >= 4)
        {
            for(; i <= size.width - 4;)
            {
                size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
                float32x4_t s = vmovq_n_f32(0.0f);

                for (; i <= limit; i += 4 )
                {
                    internal::prefetch(src1 + i);
                    internal::prefetch(src2 + i);

                    float32x4_t vs1 = vld1q_f32(src1 + i);
                    float32x4_t vs2 = vld1q_f32(src2 + i);

                    float32x4_t vd = vabdq_f32(vs2, vs1);
                    s = vaddq_f32(s, vd);
                }

                f32 s2[4];
                vst1q_f32(s2, s);

                for (u32 j = 0; j < 4; j++)
                    result += (f64)(s2[j]);
            }
        }

        for (; i < size.width; i++)
        {
            f32 v = std::abs(src1[i] - src2[i]);
            result += (f64)(v);
        }
    }
    return result;
#else
    (void)_size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;

    return 0.;
#endif
}

s32 diffNormL2(const Size2D &_size,
               const u8 * src0Base, ptrdiff_t src0Stride,
               const u8 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src1 = internal::getRowPtr( src0Base,  src0Stride, k);
        const u8* src2 = internal::getRowPtr( src1Base,  src1Stride, k);
        size_t i = 0;

#define NORML28U_BLOCK_SIZE (33024*2) //bigger block size can result in integer overflow
        if (size.width >= 16)
        {
            for(; i <= size.width - 16;)
            {
                size_t limit = std::min(size.width, i + NORML28U_BLOCK_SIZE) - 16;
                uint32x4_t si1 = vmovq_n_u32(0);
                uint32x4_t si2 = vmovq_n_u32(0);

                for (; i <= limit; i += 16)
                {
                    internal::prefetch(src1 + i);
                    internal::prefetch(src2 + i);

                    uint8x16_t vs1 = vld1q_u8(src1 + i);
                    uint8x16_t vs2 = vld1q_u8(src2 + i);

                    uint16x8_t vdlo = vabdl_u8(vget_low_u8(vs1), vget_low_u8(vs2));
                    uint16x8_t vdhi = vabdl_u8(vget_high_u8(vs1), vget_high_u8(vs2));

                    si1 = vmlal_u16(si1, vget_low_u16(vdlo), vget_low_u16(vdlo));
                    si2 = vmlal_u16(si2, vget_high_u16(vdlo), vget_high_u16(vdlo));

                    si1 = vmlal_u16(si1, vget_low_u16(vdhi), vget_low_u16(vdhi));
                    si2 = vmlal_u16(si2, vget_high_u16(vdhi), vget_high_u16(vdhi));
                }

                u32 s2[4];
                vst1q_u32(s2, vqaddq_u32(si1, si2));

                for (u32 j = 0; j < 4; j++)
                {
                    if ((s32)(0x7fFFffFFu - s2[j]) <= result)
                    {
                        return 0x7fFFffFF; //result already saturated
                    }
                    result += (s32)s2[j];
                }
            }

        }

        for (; i < size.width; i++)
        {
            s32 v = (s32)(src1[i]) - (s32)(src2[i]);
            v *= v;

            if ((s32)(0x7fFFffFFu - (u32)(v)) <= result)
            {
                return 0x7fFFffFF; //result already saturated
            }
            result += v;
        }
    }
    return result;
#else
    (void)_size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;

    return 0;
#endif
}

f64 diffNormL2(const Size2D &_size,
               const f32 * src0Base, ptrdiff_t src0Stride,
               const f32 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src1 = internal::getRowPtr( src0Base,  src0Stride, k);
        const f32* src2 = internal::getRowPtr( src1Base,  src1Stride, k);
        size_t i = 0;

        if (size.width >= 4)
        {
            for(; i <= size.width - 4;)
            {
                size_t limit = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
                float32x4_t s = vmovq_n_f32(0.0f);

                for (; i <= limit; i += 4 )
                {
                    internal::prefetch(src1 + i);
                    internal::prefetch(src2 + i);

                    float32x4_t vs1 = vld1q_f32(src1 + i);
                    float32x4_t vs2 = vld1q_f32(src2 + i);

                    float32x4_t vd = vsubq_f32(vs2,vs1);
                    s = vmlaq_f32(s, vd, vd);
                }

                f32 s2[4];
                vst1q_f32(s2, s);

                for (u32 j = 0; j < 4; j++)
                    result += (f64)(s2[j]);
            }
        }

        for (; i < size.width; i++)
        {
            f32 v = src1[i] - src2[i];
            result += v * v;
        }
    }
    return result;
#else
    (void)_size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;

    return 0.;
#endif
}

} // namespace CAROTENE_NS
