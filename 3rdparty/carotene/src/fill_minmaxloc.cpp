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

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

template <typename T>
void process(const T * src, size_t j0, size_t j1, size_t i,
             T minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
             T maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity)
{
    for (size_t j = j0; j < j1; ++j)
    {
        T val = src[j];

        if (val == maxVal)
        {
            if (maxLocCount < maxLocCapacity)
            {
                maxLocPtr[maxLocCount] = j;
                maxLocPtr[maxLocCount + 1] = i;
            }
            maxLocCount += 2;
        }

        if (val == minVal)
        {
            if (minLocCount < minLocCapacity)
            {
                minLocPtr[minLocCount] = j;
                minLocPtr[minLocCount + 1] = i;
            }
            minLocCount += 2;
        }
    }
}

} // namespace

#endif

void fillMinMaxLocs(const Size2D & size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                    u8 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    uint8x16_t v_maxval16 = vdupq_n_u8(maxVal), v_minval16 = vdupq_n_u8(minVal);
    uint8x8_t v_maxval8 = vdup_n_u8(maxVal), v_minval8 = vdup_n_u8(minVal);

    u64 mask[2] = { 0ul };

    minLocCapacity <<= 1;
    maxLocCapacity <<= 1;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0;

        for ( ; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint8x16_t v_src = vld1q_u8(src + j);

            uint8x16_t v_maxmask = vceqq_u8(v_src, v_maxval16);
            uint8x16_t v_minmask = vceqq_u8(v_src, v_minval16);
            uint8x16_t v_mask = vorrq_u8(v_maxmask, v_minmask);

            vst1q_u8((u8 *)&mask[0], v_mask);

            if (mask[0])
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
            if (mask[1])
                process(src, j + 8, j + 16, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }
        for ( ; j < roiw8; j += 8)
        {
            uint8x8_t v_src = vld1_u8(src + j);

            uint8x8_t v_maxmask = vceq_u8(v_src, v_maxval8);
            uint8x8_t v_minmask = vceq_u8(v_src, v_minval8);
            uint8x8_t v_mask = vorr_u8(v_maxmask, v_minmask);

            vst1_u8((u8 *)&mask[0], v_mask);

            if (mask[0])
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }

        process(src, j, size.width, i,
                minVal, minLocPtr, minLocCount, minLocCapacity,
                maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
    }

    minLocCount >>= 1;
    maxLocCount >>= 1;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minLocPtr;
    (void)minLocCount;
    (void)minLocCapacity;
    (void)maxVal;
    (void)maxLocPtr;
    (void)maxLocCount;
    (void)maxLocCapacity;
#endif
}

void fillMinMaxLocs(const Size2D & size,
                    const u16 * srcBase, ptrdiff_t srcStride,
                    u16 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                    u16 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    uint16x8_t v_maxval8 = vdupq_n_u16(maxVal),
               v_minval8 = vdupq_n_u16(minVal);
    u64 mask[2] = { 0ul };

    minLocCapacity <<= 1;
    maxLocCapacity <<= 1;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0;

        for ( ; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint16x8_t v_src0 = vld1q_u16(src + j), v_src1 = vld1q_u16(src + j + 8);

            uint16x8_t v_mask0 = vorrq_u16(vceqq_u16(v_src0, v_maxval8), vceqq_u16(v_src0, v_minval8));
            uint16x8_t v_mask1 = vorrq_u16(vceqq_u16(v_src1, v_maxval8), vceqq_u16(v_src1, v_minval8));

            vst1q_u8((u8 *)&mask[0], vcombine_u8(vmovn_u16(v_mask0), vmovn_u16(v_mask1)));

            if (mask[0])
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
            if (mask[1])
                process(src, j + 8, j + 16, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }
        for ( ; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            uint16x8_t v_src = vld1q_u16(src + j);

            uint16x8_t v_maxmask = vceqq_u16(v_src, v_maxval8);
            uint16x8_t v_minmask = vceqq_u16(v_src, v_minval8);
            uint16x8_t v_mask = vorrq_u16(v_maxmask, v_minmask);

            vst1_u8((u8 *)&mask[0], vmovn_u16(v_mask));

            if (mask[0])
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }

        process(src, j, size.width, i,
                minVal, minLocPtr, minLocCount, minLocCapacity,
                maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
    }

    minLocCount >>= 1;
    maxLocCount >>= 1;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minLocPtr;
    (void)minLocCount;
    (void)minLocCapacity;
    (void)maxVal;
    (void)maxLocPtr;
    (void)maxLocCount;
    (void)maxLocCapacity;
#endif
}

void fillMinMaxLocs(const Size2D & size,
                    const s16 * srcBase, ptrdiff_t srcStride,
                    s16 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                    s16 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    int16x8_t v_maxval8 = vdupq_n_s16(maxVal),
              v_minval8 = vdupq_n_s16(minVal);
    u64 mask[2] = { 0ul };

    minLocCapacity <<= 1;
    maxLocCapacity <<= 1;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0;

        for ( ; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            int16x8_t v_src0 = vld1q_s16(src + j), v_src1 = vld1q_s16(src + j + 8);

            uint16x8_t v_mask0 = vorrq_u16(vceqq_s16(v_src0, v_maxval8), vceqq_s16(v_src0, v_minval8));
            uint16x8_t v_mask1 = vorrq_u16(vceqq_s16(v_src1, v_maxval8), vceqq_s16(v_src1, v_minval8));

            vst1q_u8((u8 *)&mask[0], vcombine_u8(vmovn_u16(v_mask0), vmovn_u16(v_mask1)));

            if (mask[0])
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
            if (mask[1])
                process(src, j + 8, j + 16, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }
        for ( ; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int16x8_t v_src = vld1q_s16(src + j);

            uint16x8_t v_maxmask = vceqq_s16(v_src, v_maxval8);
            uint16x8_t v_minmask = vceqq_s16(v_src, v_minval8);
            uint16x8_t v_mask = vorrq_u16(v_maxmask, v_minmask);

            vst1_u8((u8 *)&mask[0], vmovn_u16(v_mask));

            if (mask[0])
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }

        process(src, j, size.width, i,
                minVal, minLocPtr, minLocCount, minLocCapacity,
                maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
    }

    minLocCount >>= 1;
    maxLocCount >>= 1;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minLocPtr;
    (void)minLocCount;
    (void)minLocCapacity;
    (void)maxVal;
    (void)maxLocPtr;
    (void)maxLocCount;
    (void)maxLocCapacity;
#endif
}

void fillMinMaxLocs(const Size2D & size,
                    const s32 * srcBase, ptrdiff_t srcStride,
                    s32 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                    s32 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    int32x4_t v_maxval4 = vdupq_n_s32(maxVal),
              v_minval4 = vdupq_n_s32(minVal);
    u64 mask = 0ul;

    minLocCapacity <<= 1;
    maxLocCapacity <<= 1;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0;

        for ( ; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int32x4_t v_src0 = vld1q_s32(src + j), v_src1 = vld1q_s32(src + j + 4);

            uint32x4_t v_mask0 = vorrq_u32(vceqq_s32(v_src0, v_maxval4), vceqq_s32(v_src0, v_minval4));
            uint32x4_t v_mask1 = vorrq_u32(vceqq_s32(v_src1, v_maxval4), vceqq_s32(v_src1, v_minval4));

            vst1_u8((u8 *)&mask, vmovn_u16(vcombine_u16(vmovn_u32(v_mask0), vmovn_u32(v_mask1))));

            if (mask)
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }

        process(src, j, size.width, i,
                minVal, minLocPtr, minLocCount, minLocCapacity,
                maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
    }

    minLocCount >>= 1;
    maxLocCount >>= 1;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minLocPtr;
    (void)minLocCount;
    (void)minLocCapacity;
    (void)maxVal;
    (void)maxLocPtr;
    (void)maxLocCount;
    (void)maxLocCapacity;
#endif
}

void fillMinMaxLocs(const Size2D & size,
                    const u32 * srcBase, ptrdiff_t srcStride,
                    u32 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                    u32 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    uint32x4_t v_maxval4 = vdupq_n_u32(maxVal),
               v_minval4 = vdupq_n_u32(minVal);
    u64 mask = 0ul;

    minLocCapacity <<= 1;
    maxLocCapacity <<= 1;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u32 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0;

        for ( ; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            uint32x4_t v_src0 = vld1q_u32(src + j), v_src1 = vld1q_u32(src + j + 4);

            uint32x4_t v_mask0 = vorrq_u32(vceqq_u32(v_src0, v_maxval4), vceqq_u32(v_src0, v_minval4));
            uint32x4_t v_mask1 = vorrq_u32(vceqq_u32(v_src1, v_maxval4), vceqq_u32(v_src1, v_minval4));

            vst1_u8((u8 *)&mask, vmovn_u16(vcombine_u16(vmovn_u32(v_mask0), vmovn_u32(v_mask1))));

            if (mask)
                process(src, j, j + 8, i,
                        minVal, minLocPtr, minLocCount, minLocCapacity,
                        maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
        }

        process(src, j, size.width, i,
                minVal, minLocPtr, minLocCount, minLocCapacity,
                maxVal, maxLocPtr, maxLocCount, maxLocCapacity);
    }

    minLocCount >>= 1;
    maxLocCount >>= 1;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minLocPtr;
    (void)minLocCount;
    (void)minLocCapacity;
    (void)maxVal;
    (void)maxLocPtr;
    (void)maxLocCount;
    (void)maxLocCapacity;
#endif
}

} // namespace CAROTENE_NS
