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

f64 dotProduct(const Size2D &_size,
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

// It is possible to accumulate up to 66051 uchar multiplication results in uint32 without overflow
// We process 16 elements and accumulate two new elements per step. So we could handle 66051/2*16 elements
#define DOT_UINT_BLOCKSIZE 66050*8
    f64 result = 0.0;
    for (size_t row = 0; row < size.height; ++row)
    {
        const u8 * src0 = internal::getRowPtr(src0Base, src0Stride, row);
        const u8 * src1 = internal::getRowPtr(src1Base, src1Stride, row);

        size_t i = 0;
        uint64x2_t ws = vmovq_n_u64(0);

        while(i + 16 <= size.width)
        {
            size_t lim = std::min(i + DOT_UINT_BLOCKSIZE, size.width) - 16;

            uint32x4_t s1 = vmovq_n_u32(0);
            uint32x4_t s2 = vmovq_n_u32(0);

            for (; i <= lim; i += 16)
            {
                internal::prefetch(src0 + i);
                internal::prefetch(src1 + i);

                uint8x16_t vs1 = vld1q_u8(src0 + i);
                uint8x16_t vs2 = vld1q_u8(src1 + i);

                uint16x8_t vdot1 = vmull_u8(vget_low_u8(vs1), vget_low_u8(vs2));
                uint16x8_t vdot2 = vmull_u8(vget_high_u8(vs1), vget_high_u8(vs2));

                s1 = vpadalq_u16(s1, vdot1);
                s2 = vpadalq_u16(s2, vdot2);
            }

            ws = vpadalq_u32(ws, s1);
            ws = vpadalq_u32(ws, s2);
        }

        if(i + 8 <= size.width)
        {
            uint8x8_t vs1 = vld1_u8(src0 + i);
            uint8x8_t vs2 = vld1_u8(src1 + i);

            ws = vpadalq_u32(ws, vpaddlq_u16(vmull_u8(vs1, vs2)));
            i += 8;
        }

        result += (double)vget_lane_u64(vadd_u64(vget_low_u64(ws), vget_high_u64(ws)), 0);

        for (; i < size.width; ++i)
            result += s32(src0[i]) * s32(src1[i]);
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

f64 dotProduct(const Size2D &_size,
               const s8 * src0Base, ptrdiff_t src0Stride,
               const s8 * src1Base, ptrdiff_t src1Stride)
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

// It is possible to accumulate up to 131071 schar multiplication results in sint32 without overflow
// We process 16 elements and accumulate two new elements per step. So we could handle 131071/2*16 elements
#define DOT_INT_BLOCKSIZE 131070*8
    f64 result = 0.0;
    for (size_t row = 0; row < size.height; ++row)
    {
        const s8 * src0 = internal::getRowPtr(src0Base, src0Stride, row);
        const s8 * src1 = internal::getRowPtr(src1Base, src1Stride, row);

        size_t i = 0;
        int64x2_t ws = vmovq_n_s64(0);

        while(i + 16 <= size.width)
        {
            size_t lim = std::min(i + DOT_UINT_BLOCKSIZE, size.width) - 16;

            int32x4_t s1 = vmovq_n_s32(0);
            int32x4_t s2 = vmovq_n_s32(0);

            for (; i <= lim; i += 16)
            {
                internal::prefetch(src0 + i);
                internal::prefetch(src1 + i);

                int8x16_t vs1 = vld1q_s8(src0 + i);
                int8x16_t vs2 = vld1q_s8(src1 + i);

                int16x8_t vdot1 = vmull_s8(vget_low_s8(vs1), vget_low_s8(vs2));
                int16x8_t vdot2 = vmull_s8(vget_high_s8(vs1), vget_high_s8(vs2));

                s1 = vpadalq_s16(s1, vdot1);
                s2 = vpadalq_s16(s2, vdot2);
            }

            ws = vpadalq_s32(ws, s1);
            ws = vpadalq_s32(ws, s2);
        }

        if(i + 8 <= size.width)
        {
            int8x8_t vs1 = vld1_s8(src0 + i);
            int8x8_t vs2 = vld1_s8(src1 + i);

            ws = vpadalq_s32(ws, vpaddlq_s16(vmull_s8(vs1, vs2)));
            i += 8;
        }

        result += (double)vget_lane_s64(vadd_s64(vget_low_s64(ws), vget_high_s64(ws)), 0);

        for (; i < size.width; ++i)
            result += s32(src0[i]) * s32(src1[i]);
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

f64 dotProduct(const Size2D &_size,
               const f32 * src0Base, ptrdiff_t src0Stride,
               const f32 * src1Base, ptrdiff_t src1Stride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (src0Stride == src1Stride &&
        src0Stride == (ptrdiff_t)(size.width * sizeof(f32)))
    {
        size.width *= size.height;
        size.height = 1;
    }

#define DOT_FLOAT_BLOCKSIZE (1 << 13)
    f64 result = 0.0;
    for (size_t row = 0; row < size.height; ++row)
    {
        const f32 * src0 = internal::getRowPtr(src0Base, src0Stride, row);
        const f32 * src1 = internal::getRowPtr(src1Base, src1Stride, row);

        size_t i = 0;
        while(i + 4 <= size.width)
        {
            size_t lim = std::min(i + DOT_FLOAT_BLOCKSIZE, size.width) - 4;
            float32x4_t v_sum = vdupq_n_f32(0.0f);

            for( ; i <= lim; i += 4 )
            {
                internal::prefetch(src0 + i);
                internal::prefetch(src1 + i);
                v_sum = vmlaq_f32(v_sum, vld1q_f32(src0 + i), vld1q_f32(src1 + i));
            }

            float32x2_t vres = vpadd_f32(vget_low_f32(v_sum),vget_high_f32(v_sum));
            result += vget_lane_f32(vres, 0) + vget_lane_f32(vres, 1);
        }

        if(i + 2 <= size.width)
        {
            float32x2_t vres = vmul_f32(vld1_f32(src0 + i), vld1_f32(src1 + i));
            result += vget_lane_f32(vres, 0) + vget_lane_f32(vres, 1);
            i += 2;
        }

        for (; i < size.width; ++i)
            result += src0[i] * src1[i];
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

} // namespace CAROTENE_NS
