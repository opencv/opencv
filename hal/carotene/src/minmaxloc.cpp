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
#include "vtransform.hpp"

#include <limits>

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

template <typename T>
void minMaxVals(const Size2D &size,
                const T * srcBase, ptrdiff_t srcStride,
                T * pMinVal, T * pMaxVal)
{
    using namespace internal;

    typedef typename VecTraits<T>::vec128 vec128;
    typedef typename VecTraits<T>::vec64 vec64;

    u32 step_base = 32 / sizeof(T), step_tail = 8 / sizeof(T);
    size_t roiw_base = size.width >= (step_base - 1) ? size.width - step_base + 1 : 0;
    size_t roiw_tail = size.width >= (step_tail - 1) ? size.width - step_tail + 1 : 0;

    T maxVal = std::numeric_limits<T>::min();
    T minVal = std::numeric_limits<T>::max();
    vec128 v_min_base = vdupq_n(minVal), v_max_base = vdupq_n(maxVal);
    vec64 v_min_tail = vdup_n(minVal), v_max_tail = vdup_n(maxVal);

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src = getRowPtr(srcBase, srcStride, i);
        size_t j = 0;

        for (; j < roiw_base; j += step_base)
        {
            prefetch(src + j);
            vec128 v_src0 = vld1q(src + j), v_src1 = vld1q(src + j + 16 / sizeof(T));
            v_min_base = vminq(v_min_base, v_src0);
            v_max_base = vmaxq(v_max_base, v_src0);
            v_min_base = vminq(v_min_base, v_src1);
            v_max_base = vmaxq(v_max_base, v_src1);
        }
        for (; j < roiw_tail; j += step_tail)
        {
            vec64 v_src0 = vld1(src + j);
            v_min_tail = vmin(v_min_tail, v_src0);
            v_max_tail = vmax(v_max_tail, v_src0);
        }

        for (; j < size.width; j++)
        {
            T srcval = src[j];
            minVal = std::min(srcval, minVal);
            maxVal = std::max(srcval, maxVal);
        }
    }

    // collect min & max values
    T ar[16 / sizeof(T)];
    vst1q(ar, vcombine(vmin(v_min_tail, vmin(vget_low(v_min_base), vget_high(v_min_base))),
                       vmax(v_max_tail, vmax(vget_low(v_max_base), vget_high(v_max_base)))));

    for (size_t x = 0; x < 8u / sizeof(T); ++x)
    {
        minVal = std::min(minVal, ar[x]);
        maxVal = std::max(maxVal, ar[x + 8 / sizeof(T)]);
    }

    if (pMaxVal)
        *pMaxVal = maxVal;
    if (pMinVal)
        *pMinVal = minVal;
}

} // namespace

#endif

void minMaxVals(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * pMinVal, u8 * pMaxVal)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minMaxVals<u8>(size,
                   srcBase, srcStride,
                   pMinVal, pMaxVal);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMinVal;
    (void)pMaxVal;
#endif
}

void minMaxVals(const Size2D &size,
                const s16 * srcBase, ptrdiff_t srcStride,
                s16 * pMinVal, s16 * pMaxVal)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minMaxVals<s16>(size,
                    srcBase, srcStride,
                    pMinVal, pMaxVal);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMinVal;
    (void)pMaxVal;
#endif
}

void minMaxVals(const Size2D &size,
                const u16 * srcBase, ptrdiff_t srcStride,
                u16 * pMinVal, u16 * pMaxVal)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minMaxVals<u16>(size,
                    srcBase, srcStride,
                    pMinVal, pMaxVal);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMinVal;
    (void)pMaxVal;
#endif
}

void minMaxVals(const Size2D &size,
                const s32 * srcBase, ptrdiff_t srcStride,
                s32 * pMinVal, s32 * pMaxVal)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minMaxVals<s32>(size,
                    srcBase, srcStride,
                    pMinVal, pMaxVal);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMinVal;
    (void)pMaxVal;
#endif
}

void minMaxVals(const Size2D &size,
                const u32 * srcBase, ptrdiff_t srcStride,
                u32 * pMinVal, u32 * pMaxVal)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minMaxVals<u32>(size,
                    srcBase, srcStride,
                    pMinVal, pMaxVal);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMinVal;
    (void)pMaxVal;
#endif
}

void minMaxLoc(const Size2D &size,
               const f32 * srcBase, ptrdiff_t srcStride,
               f32 &minVal, size_t &minCol, size_t &minRow,
               f32 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = srcBase[0];
    minCol = 0;
    minRow = 0;
    maxVal = srcBase[0];
    maxCol = 0;
    maxRow = 0;
    for(size_t l = 0, i = 0; l < size.height; ++l, i = 0)
    {
        const f32 * src = internal::getRowPtr( srcBase, srcStride, l);
        if (size.width >= 16)
        {
            u32 tmp0123[4] = { 0, 1, 2, 3 };
            uint32x4_t   c4       = vdupq_n_u32(4);

#if SIZE_MAX > UINT32_MAX
            size_t boundAll = size.width - (4 - 1);
            for(size_t b = 0; i < boundAll; b = i)
            {
                size_t bound = std::min<size_t>(boundAll, b + 0xffffFFFC);
#else
            {
                size_t bound = size.width - (4 - 1);
#endif
                uint32x4_t  lineIdxOffset = vld1q_u32(tmp0123);
                float32x4_t  n_min    = vdupq_n_f32(minVal);
                uint32x4_t   n_minIdx = vdupq_n_u32(0xffffFFFC);
                float32x4_t  n_max    = vdupq_n_f32(maxVal);
                uint32x4_t   n_maxIdx = vdupq_n_u32(0xffffFFFC);

                for(; i < bound; i+=4)
                {
                    internal::prefetch(src + i);
                    float32x4_t line = vld1q_f32(src + i);

                    uint32x4_t minmask = vcltq_f32(line, n_min);
                    uint32x4_t maxmask = vcgtq_f32(line, n_max);

                    n_min    = vbslq_f32(minmask, line, n_min);
                    n_minIdx = vbslq_u32(minmask, lineIdxOffset, n_minIdx);
                    n_max    = vbslq_f32(maxmask, line, n_max);
                    n_maxIdx = vbslq_u32(maxmask, lineIdxOffset, n_maxIdx);

                    // idx[] +=4
                    lineIdxOffset = vaddq_u32(lineIdxOffset, c4);
                }

                f32 fmin[4], fmax[4];
                u32 fminIdx[4], fmaxIdx[4];

                vst1q_f32(fmin, n_min);
                vst1q_f32(fmax, n_max);

                vst1q_u32(fminIdx, n_minIdx);
                vst1q_u32(fmaxIdx, n_maxIdx);

                size_t minIdx = fminIdx[0];
                size_t maxIdx = fmaxIdx[0];
                minVal = fmin[0];
                maxVal = fmax[0];

                for (s32 j = 1; j < 4; ++j)
                {
                    f32 minval = fmin[j];
                    f32 maxval = fmax[j];
                    if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
                    {
                        minIdx = fminIdx[j];
                        minVal = minval;
                    }
                    if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
                    {
                        maxIdx = fmaxIdx[j];
                        maxVal = maxval;
                    }
                }
                if(minIdx < 0xffffFFFC)
                {
#if SIZE_MAX > UINT32_MAX
                    minCol = b + minIdx;
#else
                    minCol = minIdx;
#endif
                    minRow = l;
                }
                if(maxIdx < 0xffffFFFC)
                {
#if SIZE_MAX > UINT32_MAX
                    maxCol = b + maxIdx;
#else
                    maxCol = maxIdx;
#endif
                    maxRow = l;
                }
            }
        }
        for(; i < size.width; ++i )
        {
            float val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minCol = i;
                minRow = l;
            }
            else if( val > maxVal )
            {
                maxVal = val;
                maxCol = i;
                maxRow = l;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

void minMaxLoc(const Size2D &size,
               const f32 * srcBase, ptrdiff_t srcStride,
               const u8 * maskBase, ptrdiff_t maskStride,
               f32 &minVal, size_t &minCol, size_t &minRow,
               f32 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = std::numeric_limits<f32>::max();
    minCol = size.width;
    minRow = size.height;
    maxVal = -std::numeric_limits<f32>::max();
    maxCol = size.width;
    maxRow = size.height;
    for(size_t l = 0, i = 0; l < size.height; ++l, i = 0)
    {
        const f32 * src = internal::getRowPtr( srcBase, srcStride, l);
        const u8 * mask = internal::getRowPtr( maskBase, maskStride, l);
        if (size.width >= 16)
        {
            u32 tmp0123[4] = { 0, 1, 2, 3 };
            uint32x4_t  uOne      = vdupq_n_u32(1);
            uint32x4_t   c4       = vdupq_n_u32(4);

#if SIZE_MAX > UINT32_MAX
            size_t boundAll = size.width - (4 - 1);
            for(size_t b = 0; i < boundAll; b = i)
            {
                size_t bound = std::min<size_t>(boundAll, b + 0xffffFFFC);
#else
            {
                size_t bound = size.width - (4 - 1);
#endif
                uint32x4_t  lineIdxOffset = vld1q_u32(tmp0123);
                float32x4_t  n_min    = vdupq_n_f32(minVal);
                uint32x4_t   n_minIdx = vdupq_n_u32(0xffffFFFC);
                float32x4_t  n_max    = vdupq_n_f32(maxVal);
                uint32x4_t   n_maxIdx = vdupq_n_u32(0xffffFFFC);

                for(; i < bound; i+=4)
                {
                    internal::prefetch(src + i);
                    internal::prefetch(mask + i);
                    float32x4_t line = vld1q_f32(src + i);
                    uint8x8_t maskLine = vld1_u8(mask + i);

                    uint32x4_t maskLine4 = vmovl_u16(vget_low_u16(vmovl_u8(maskLine)));
                    maskLine4 = vcgeq_u32(maskLine4, uOne);

                    uint32x4_t minmask = vcltq_f32(line, n_min);
                    uint32x4_t maxmask = vcgtq_f32(line, n_max);

                    minmask = vandq_u32(minmask, maskLine4);
                    maxmask = vandq_u32(maxmask, maskLine4);

                    n_min    = vbslq_f32(minmask, line, n_min);
                    n_minIdx = vbslq_u32(minmask, lineIdxOffset, n_minIdx);
                    n_max    = vbslq_f32(maxmask, line, n_max);
                    n_maxIdx = vbslq_u32(maxmask, lineIdxOffset, n_maxIdx);

                    // idx[] +=4
                    lineIdxOffset = vaddq_u32(lineIdxOffset, c4);
                }

                f32 fmin[4], fmax[4];
                u32 fminIdx[4], fmaxIdx[4];

                vst1q_f32(fmin, n_min);
                vst1q_f32(fmax, n_max);

                vst1q_u32(fminIdx, n_minIdx);
                vst1q_u32(fmaxIdx, n_maxIdx);

                size_t minIdx = fminIdx[0];
                size_t maxIdx = fmaxIdx[0];
                minVal = fmin[0];
                maxVal = fmax[0];

                for (s32 j = 1; j < 4; ++j)
                {
                    f32 minval = fmin[j];
                    f32 maxval = fmax[j];
                    if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
                    {
                        minIdx = fminIdx[j];
                        minVal = minval;
                    }
                    if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
                    {
                        maxIdx = fmaxIdx[j];
                        maxVal = maxval;
                    }
                }
                if(minIdx < 0xffffFFFC)
                {
#if SIZE_MAX > UINT32_MAX
                    minCol = b + minIdx;
#else
                    minCol = minIdx;
#endif
                    minRow = l;
                }
                if(maxIdx < 0xffffFFFC)
                {
#if SIZE_MAX > UINT32_MAX
                    maxCol = b + maxIdx;
#else
                    maxCol = maxIdx;
#endif
                    maxRow = l;
                }
            }
        }
        for(; i < size.width; i++ )
        {
            if (!mask[i])
                continue;
            f32 val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minCol = i;
                minRow = l;
            }
            if( val > maxVal )
            {
                maxVal = val;
                maxCol = i;
                maxRow = l;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)maskBase;
    (void)maskStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

void minMaxLoc(const Size2D &size,
               const s32 * srcBase, ptrdiff_t srcStride,
               s32 &minVal, size_t &minCol, size_t &minRow,
               s32 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = srcBase[0];
    minCol = 0;
    minRow = 0;
    maxVal = srcBase[0];
    maxCol = 0;
    maxRow = 0;
    for(size_t l = 0, i = 0; l < size.height; ++l, i = 0)
    {
        const s32 * src = internal::getRowPtr( srcBase, srcStride, l);
        if (size.width >= 16)
        {
            u32 tmp0123[4] = { 0, 1, 2, 3 };
            uint32x4_t c4       = vdupq_n_u32(4);

#if SIZE_MAX > UINT32_MAX
            size_t boundAll = size.width - (4 - 1);
            for(size_t b = 0; i < boundAll; b = i)
            {
                size_t bound = std::min<size_t>(boundAll, b + 0xffffFFFC);
#else
            {
                size_t bound = size.width - (4 - 1);
#endif
                uint32x4_t  lineIdxOffset = vld1q_u32(tmp0123);
                int32x4_t  n_min    = vdupq_n_s32(minVal);
                uint32x4_t   n_minIdx = vdupq_n_u32(0xffffFFFC);
                int32x4_t  n_max    = vdupq_n_s32(maxVal);
                uint32x4_t   n_maxIdx = vdupq_n_u32(0xffffFFFC);

                for(; i < bound; i+=4 )
                {
                    internal::prefetch(src + i);
                    int32x4_t line = vld1q_s32(src + i);

                    uint32x4_t minmask = vcltq_s32(line, n_min);
                    uint32x4_t maxmask = vcgtq_s32(line, n_max);

                    n_min    = vbslq_s32(minmask, line, n_min);
                    n_minIdx = vbslq_u32(minmask, lineIdxOffset, n_minIdx);
                    n_max    = vbslq_s32(maxmask, line, n_max);
                    n_maxIdx = vbslq_u32(maxmask, lineIdxOffset, n_maxIdx);

                    // idx[] +=4
                    lineIdxOffset = vaddq_u32(lineIdxOffset, c4);
                }

                s32 fmin[4], fmax[4];
                u32 fminIdx[4], fmaxIdx[4];

                vst1q_s32(fmin, n_min);
                vst1q_s32(fmax, n_max);

                vst1q_u32(fminIdx, n_minIdx);
                vst1q_u32(fmaxIdx, n_maxIdx);

                size_t minIdx = fminIdx[0];
                size_t maxIdx = fmaxIdx[0];
                minVal = fmin[0];
                maxVal = fmax[0];

                for (s32 j = 1; j < 4; ++j)
                {
                    s32 minval = fmin[j];
                    s32 maxval = fmax[j];
                    if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
                    {
                        minIdx = fminIdx[j];
                        minVal = minval;
                    }
                    if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
                    {
                        maxIdx = fmaxIdx[j];
                        maxVal = maxval;
                    }
                }
                if(minIdx < 0xffffFFFC)
                {
#if SIZE_MAX > UINT32_MAX
                    minCol = b + minIdx;
#else
                    minCol = minIdx;
#endif
                    minRow = l;
                }
                if(maxIdx < 0xffffFFFC)
                {
#if SIZE_MAX > UINT32_MAX
                    maxCol = b + maxIdx;
#else
                    maxCol = maxIdx;
#endif
                    maxRow = l;
                }
            }
        }
        for(; i < size.width; ++i )
        {
            s32 val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minCol = i;
                minRow = l;
            }
            else if( val > maxVal )
            {
                maxVal = val;
                maxCol = i;
                maxRow = l;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

void minMaxLoc(const Size2D &size,
               const s16 * srcBase, ptrdiff_t srcStride,
               s16 &minVal, size_t &minCol, size_t &minRow,
               s16 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = srcBase[0];
    minCol = 0;
    minRow = 0;
    maxVal = srcBase[0];
    maxCol = 0;
    maxRow = 0;
    for(size_t l = 0, i = 0; l < size.height; ++l, i = 0)
    {
        const s16 * src = internal::getRowPtr( srcBase,  srcStride, l);
        if (size.width >= 32)
        {
            u32 tmp0123[4] = { 0, 1, 2, 3 };
            uint32x4_t c8        = vdupq_n_u32(8);

#if SIZE_MAX > UINT32_MAX
            size_t boundAll = size.width - (8 - 1);
            for(size_t b = 0; i < boundAll; b = i)
            {
                size_t bound = std::min<size_t>(boundAll, b + 0xffffFFF8);
#else
            {
                size_t bound = size.width - (8 - 1);
#endif
                uint32x4_t  lineIdxOffset = vld1q_u32(tmp0123);
                int16x8_t  n_min    = vdupq_n_s16(minVal);
                uint32x4_t n_minIdxl = vdupq_n_u32(0xffffFFF8);
                uint32x4_t n_minIdxh = vdupq_n_u32(0xffffFFF8);
                int16x8_t  n_max    = vdupq_n_s16(maxVal);
                uint32x4_t n_maxIdxl = vdupq_n_u32(0xffffFFF8);
                uint32x4_t n_maxIdxh = vdupq_n_u32(0xffffFFF8);

                for(; i < bound; i+=8 )
                {
                    internal::prefetch(src + i);
                    int16x8_t line = vld1q_s16(src + i);

                    uint16x8_t minmask = vcltq_s16(line, n_min);
                    uint16x8_t maxmask = vcgtq_s16(line, n_max);

                    n_min    = vbslq_s16(minmask, line, n_min);
                    uint16x4_t minml = vget_low_u16(minmask);
                    uint16x4_t minmh = vget_high_u16(minmask);
                    uint32x4_t minml2 = vmovl_u16(minml);
                    uint32x4_t minmh2 = vmovl_u16(minmh);
                    minml2 = vqshlq_n_u32(minml2, 31);
                    minmh2 = vqshlq_n_u32(minmh2, 31);
                    n_minIdxl = vbslq_u32(minml2, lineIdxOffset, n_minIdxl);
                    n_minIdxh = vbslq_u32(minmh2, lineIdxOffset, n_minIdxh);

                    n_max    = vbslq_s16(maxmask, line, n_max);
                    uint16x4_t maxml = vget_low_u16(maxmask);
                    uint16x4_t maxmh = vget_high_u16(maxmask);
                    uint32x4_t maxml2 = vmovl_u16(maxml);
                    uint32x4_t maxmh2 = vmovl_u16(maxmh);
                    maxml2 = vqshlq_n_u32(maxml2, 31);
                    maxmh2 = vqshlq_n_u32(maxmh2, 31);
                    n_maxIdxl = vbslq_u32(maxml2, lineIdxOffset, n_maxIdxl);
                    n_maxIdxh = vbslq_u32(maxmh2, lineIdxOffset, n_maxIdxh);

                    // idx[] +=8
                    lineIdxOffset = vaddq_u32(lineIdxOffset, c8);
                }

                // fix high part of indexes
                uint32x4_t c4 = vdupq_n_u32((int32_t) 4);
                n_minIdxh = vaddq_u32(n_minIdxh, c4);
                n_maxIdxh = vaddq_u32(n_maxIdxh, c4);

                s16 fmin[8], fmax[8];
                u32 fminIdx[8], fmaxIdx[8];

                vst1q_s16(fmin, n_min);
                vst1q_s16(fmax, n_max);
                vst1q_u32(fminIdx+0, n_minIdxl);
                vst1q_u32(fmaxIdx+0, n_maxIdxl);
                vst1q_u32(fminIdx+4, n_minIdxh);
                vst1q_u32(fmaxIdx+4, n_maxIdxh);

                size_t minIdx = fminIdx[0];
                size_t maxIdx = fmaxIdx[0];
                minVal = fmin[0];
                maxVal = fmax[0];

                for (s32 j = 1; j < 8; ++j)
                {
                    s16 minval = fmin[j];
                    s16 maxval = fmax[j];
                    if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
                    {
                        minIdx = fminIdx[j];
                        minVal = minval;
                    }
                    if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
                    {
                        maxIdx = fmaxIdx[j];
                        maxVal = maxval;
                    }
                }
                if(minIdx < 0xffffFFF8)
                {
#if SIZE_MAX > UINT32_MAX
                    minCol = b + minIdx;
#else
                    minCol = minIdx;
#endif
                    minRow = l;
                }
                if(maxIdx < 0xffffFFF8)
                {
#if SIZE_MAX > UINT32_MAX
                    maxCol = b + maxIdx;
#else
                    maxCol = maxIdx;
#endif
                    maxRow = l;
                }
            }
        }
        for(; i < size.width; ++i )
        {
            short val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minCol = i;
                minRow = l;
            }
            else if( val > maxVal )
            {
                maxVal = val;
                maxCol = i;
                maxRow = l;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

void minMaxLoc(const Size2D &size,
               const u16 * srcBase, ptrdiff_t srcStride,
               u16 &minVal, size_t &minCol, size_t &minRow,
               u16 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = srcBase[0];
    minCol = 0;
    minRow = 0;
    maxVal = srcBase[0];
    maxCol = 0;
    maxRow = 0;
    for(size_t l = 0, i = 0; l < size.height; ++l, i = 0)
    {
        const u16 * src = internal::getRowPtr( srcBase,  srcStride, l);
        if (size.width >= 32)
        {
            u32 tmp0123[4] = { 0, 1, 2, 3 };
            uint32x4_t c8        = vdupq_n_u32(8);

#if SIZE_MAX > UINT32_MAX
            size_t boundAll = size.width - (8 - 1);
            for(size_t b = 0; i < boundAll; b = i)
            {
                size_t bound = std::min<size_t>(boundAll, b + 0xffffFFF8);
#else
            {
                size_t bound = size.width - (8 - 1);
#endif
                uint32x4_t  lineIdxOffset = vld1q_u32(tmp0123);
                uint16x8_t  n_min    = vdupq_n_u16(minVal);
                uint32x4_t n_minIdxl = vdupq_n_u32(0xffffFFF8);
                uint32x4_t n_minIdxh = vdupq_n_u32(0xffffFFF8);
                uint16x8_t  n_max    = vdupq_n_u16(maxVal);
                uint32x4_t n_maxIdxl = vdupq_n_u32(0xffffFFF8);
                uint32x4_t n_maxIdxh = vdupq_n_u32(0xffffFFF8);

                for(; i < bound; i+=8 )
                {
                    internal::prefetch(src + i);
                    uint16x8_t line = vld1q_u16(src + i);

                    uint16x8_t minmask = vcltq_u16(line, n_min);
                    uint16x8_t maxmask = vcgtq_u16(line, n_max);

                    n_min    = vbslq_u16(minmask, line, n_min);
                    uint16x4_t minml = vget_low_u16(minmask);
                    uint16x4_t minmh = vget_high_u16(minmask);
                    uint32x4_t minml2 = vmovl_u16(minml);
                    uint32x4_t minmh2 = vmovl_u16(minmh);
                    minml2 = vqshlq_n_u32(minml2, 31);
                    minmh2 = vqshlq_n_u32(minmh2, 31);
                    n_minIdxl = vbslq_u32(minml2, lineIdxOffset, n_minIdxl);
                    n_minIdxh = vbslq_u32(minmh2, lineIdxOffset, n_minIdxh);

                    n_max    = vbslq_u16(maxmask, line, n_max);
                    uint16x4_t maxml = vget_low_u16(maxmask);
                    uint16x4_t maxmh = vget_high_u16(maxmask);
                    uint32x4_t maxml2 = vmovl_u16(maxml);
                    uint32x4_t maxmh2 = vmovl_u16(maxmh);
                    maxml2 = vqshlq_n_u32(maxml2, 31);
                    maxmh2 = vqshlq_n_u32(maxmh2, 31);
                    n_maxIdxl = vbslq_u32(maxml2, lineIdxOffset, n_maxIdxl);
                    n_maxIdxh = vbslq_u32(maxmh2, lineIdxOffset, n_maxIdxh);

                    // idx[] +=8
                    lineIdxOffset = vaddq_u32(lineIdxOffset, c8);
                }

                // fix high part of indexes
                uint32x4_t c4 = vdupq_n_u32(4);
                n_minIdxh = vaddq_u32(n_minIdxh, c4);
                n_maxIdxh = vaddq_u32(n_maxIdxh, c4);

                u16 fmin[8], fmax[8];
                u32 fminIdx[8], fmaxIdx[8];

                vst1q_u16(fmin, n_min);
                vst1q_u16(fmax, n_max);
                vst1q_u32(fminIdx+0, n_minIdxl);
                vst1q_u32(fmaxIdx+0, n_maxIdxl);
                vst1q_u32(fminIdx+4, n_minIdxh);
                vst1q_u32(fmaxIdx+4, n_maxIdxh);

                size_t minIdx = fminIdx[0];
                size_t maxIdx = fmaxIdx[0];
                minVal = fmin[0];
                maxVal = fmax[0];

                for (s32 j = 1; j < 8; ++j)
                {
                    u16 minval = fmin[j];
                    u16 maxval = fmax[j];
                    if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
                    {
                        minIdx = fminIdx[j];
                        minVal = minval;
                    }
                    if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
                    {
                        maxIdx = fmaxIdx[j];
                        maxVal = maxval;
                    }
                }
                if(minIdx < 0xffffFFF8)
                {
#if SIZE_MAX > UINT32_MAX
                    minCol = b + minIdx;
#else
                    minCol = minIdx;
#endif
                    minRow = l;
                }
                if(maxIdx < 0xffffFFF8)
                {
#if SIZE_MAX > UINT32_MAX
                    maxCol = b + maxIdx;
#else
                    maxCol = maxIdx;
#endif
                    maxRow = l;
                }
            }
        }
        for(; i < size.width; ++i )
        {
            u16 val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minCol = i;
                minRow = l;
            }
            else if( val > maxVal )
            {
                maxVal = val;
                maxCol = i;
                maxRow = l;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

#ifdef CAROTENE_NEON
namespace {

void minMaxLocBlock(const u8 * src, u32 len,
                    u8 &minVal, u16 &minIdx,
                    u8 &maxVal, u16 &maxIdx)
{
    u16 tmp0123[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint8x16_t n_min     = vdupq_n_u8(src[0]);
    uint16x8_t n_minIdxl = vdupq_n_u16(0);
    uint16x8_t n_minIdxh = vdupq_n_u16(0);
    uint8x16_t n_max     = vdupq_n_u8(src[0]);
    uint16x8_t n_maxIdxl = vdupq_n_u16(0);
    uint16x8_t n_maxIdxh = vdupq_n_u16(0);
    uint16x8_t c16       = vdupq_n_u16(16);
    uint16x8_t lineIdxOffset = vld1q_u16(tmp0123);

    s32 i = 0;
    s32 bound = len - (16 - 1);
    for(; i < bound; i+=16 )
    {
        internal::prefetch(src + i);
        uint8x16_t line = vld1q_u8(src + i);

        uint8x16_t minmask = vcltq_u8(line, n_min);
        uint8x16_t maxmask = vcgtq_u8(line, n_max);

        n_min    = vbslq_u8(minmask, line, n_min);
        uint8x8_t minml = vget_low_u8(minmask);
        uint8x8_t minmh = vget_high_u8(minmask);
        uint16x8_t minml2 = vmovl_u8(minml);
        uint16x8_t minmh2 = vmovl_u8(minmh);
        minml2 = vqshlq_n_u16(minml2, 15);
        minmh2 = vqshlq_n_u16(minmh2, 15);
        n_minIdxl = vbslq_u16(minml2, lineIdxOffset, n_minIdxl);
        n_minIdxh = vbslq_u16(minmh2, lineIdxOffset, n_minIdxh);

        n_max    = vbslq_u8(maxmask, line, n_max);
        uint8x8_t maxml = vget_low_u8(maxmask);
        uint8x8_t maxmh = vget_high_u8(maxmask);
        uint16x8_t maxml2 = vmovl_u8(maxml);
        uint16x8_t maxmh2 = vmovl_u8(maxmh);
        maxml2 = vqshlq_n_u16(maxml2, 15);
        maxmh2 = vqshlq_n_u16(maxmh2, 15);
        n_maxIdxl = vbslq_u16(maxml2, lineIdxOffset, n_maxIdxl);
        n_maxIdxh = vbslq_u16(maxmh2, lineIdxOffset, n_maxIdxh);

        // idx[] +=16
        lineIdxOffset = vaddq_u16(lineIdxOffset, c16);
    }

    // fix high part of indexes
    uint16x8_t c8 = vdupq_n_u16(8);
    n_minIdxh = vaddq_u16(n_minIdxh, c8);
    n_maxIdxh = vaddq_u16(n_maxIdxh, c8);

    u8 fmin[16], fmax[16];
    u16 fminIdx[16], fmaxIdx[16];
    /*{
        uint8x8_t min_low  = vget_low_u8(n_min);
        uint8x8_t min_high = vget_high_u8(n_min);
        uint8x8_t max_low  = vget_low_u8(n_max);
        uint8x8_t max_high = vget_high_u8(n_max);

        uint8x8_t minmask  = vclt_u8(min_low, min_high);
        uint8x8_t maxmask  = vcgt_u8(max_low, max_high);

        uint8x8_t min2     = vbsl_u8(minmask, min_low, min_high);
        uint8x8_t max2     = vbsl_u8(maxmask, max_low, max_high);

        uint16x8_t minidxmask = vmovl_u8(minmask);
        uint16x8_t maxidxmask = vmovl_u8(maxmask);
        minidxmask = vqshlq_n_u16(minidxmask, 15);
        maxidxmask = vqshlq_n_u16(maxidxmask, 15);

        uint16x8_t n_minIdx = vbslq_u16(minidxmask, n_minIdxl, n_minIdxh);
        uint16x8_t n_maxIdx = vbslq_u16(maxidxmask, n_maxIdxl, n_maxIdxh);

        vst1_u8((uint8_t*)fmin, min2);
        vst1_u8((uint8_t*)fmax, max2);

        vst1q_u16((uint16_t*)(fminIdx), n_minIdx);
        vst1q_u16((uint16_t*)(fmaxIdx), n_maxIdx);
    }*/

    vst1q_u8(fmin, n_min);
    vst1q_u8(fmax, n_max);
    vst1q_u16(fminIdx+0, n_minIdxl);
    vst1q_u16(fmaxIdx+0, n_maxIdxl);
    vst1q_u16(fminIdx+8, n_minIdxh);
    vst1q_u16(fmaxIdx+8, n_maxIdxh);

    minIdx = fminIdx[0];
    maxIdx = fmaxIdx[0];
    minVal = fmin[0];
    maxVal = fmax[0];

    for (s32 j = 1; j < 16; ++j)
    {
        u8 minval = fmin[j];
        u8 maxval = fmax[j];
        if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
        {
            minIdx = fminIdx[j];
            minVal = minval;
        }
        if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
        {
            maxIdx = fmaxIdx[j];
            maxVal = maxval;
        }
    }

    for(; i < (s32)len; ++i )
    {
        u8 val = src[i];
        if( val < minVal )
        {
            minVal = val;
            minIdx = (u16)i;
        }
        else if( val > maxVal )
        {
            maxVal = val;
            maxIdx = (u16)i;
        }
    }
}

void minMaxLocBlock(const s8 * src, u32 len,
                    s8 &minVal, u16 &minIdx,
                    s8 &maxVal, u16 &maxIdx)
{
    u16 tmp0123[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    int8x16_t n_min      = vdupq_n_s8(src[0]);
    uint16x8_t n_minIdxl = vdupq_n_u16(0);
    uint16x8_t n_minIdxh = vdupq_n_u16(0);
    int8x16_t n_max      = vdupq_n_s8(src[0]);
    uint16x8_t n_maxIdxl = vdupq_n_u16(0);
    uint16x8_t n_maxIdxh = vdupq_n_u16(0);
    uint16x8_t c16       = vdupq_n_u16(16);
    uint16x8_t lineIdxOffset = vld1q_u16(tmp0123);

    s32 i = 0;
    s32 bound = len - (16 - 1);
    for(; i < bound; i+=16 )
    {
        internal::prefetch(src + i);
        int8x16_t line = vld1q_s8(src + i);

        uint8x16_t minmask = vcltq_s8(line, n_min);
        uint8x16_t maxmask = vcgtq_s8(line, n_max);

        n_min    = vbslq_s8(minmask, line, n_min);
        uint8x8_t minml = vget_low_u8(minmask);
        uint8x8_t minmh = vget_high_u8(minmask);
        uint16x8_t minml2 = vmovl_u8(minml);
        uint16x8_t minmh2 = vmovl_u8(minmh);
        minml2 = vqshlq_n_u16(minml2, 15);
        minmh2 = vqshlq_n_u16(minmh2, 15);
        n_minIdxl = vbslq_u16(minml2, lineIdxOffset, n_minIdxl);
        n_minIdxh = vbslq_u16(minmh2, lineIdxOffset, n_minIdxh);

        n_max    = vbslq_s8(maxmask, line, n_max);
        uint8x8_t maxml = vget_low_u8(maxmask);
        uint8x8_t maxmh = vget_high_u8(maxmask);
        uint16x8_t maxml2 = vmovl_u8(maxml);
        uint16x8_t maxmh2 = vmovl_u8(maxmh);
        maxml2 = vqshlq_n_u16(maxml2, 15);
        maxmh2 = vqshlq_n_u16(maxmh2, 15);
        n_maxIdxl = vbslq_u16(maxml2, lineIdxOffset, n_maxIdxl);
        n_maxIdxh = vbslq_u16(maxmh2, lineIdxOffset, n_maxIdxh);

        // idx[] +=16
        lineIdxOffset = vaddq_u16(lineIdxOffset, c16);
    }

    // fix high part of indexes
    uint16x8_t c8 = vdupq_n_u16(8);
    n_minIdxh = vaddq_u16(n_minIdxh, c8);
    n_maxIdxh = vaddq_u16(n_maxIdxh, c8);

    s8 fmin[16], fmax[16];
    u16 fminIdx[16], fmaxIdx[16];

    vst1q_s8(fmin, n_min);
    vst1q_s8(fmax, n_max);
    vst1q_u16(fminIdx+0, n_minIdxl);
    vst1q_u16(fmaxIdx+0, n_maxIdxl);
    vst1q_u16(fminIdx+8, n_minIdxh);
    vst1q_u16(fmaxIdx+8, n_maxIdxh);

    minIdx = fminIdx[0];
    maxIdx = fmaxIdx[0];
    minVal = fmin[0];
    maxVal = fmax[0];

    for (s32 j = 1; j < 16; ++j)
    {
        s8 minval = fmin[j];
        s8 maxval = fmax[j];
        if (minval < minVal || (minval == minVal && fminIdx[j] < minIdx))
        {
            minIdx = fminIdx[j];
            minVal = minval;
        }
        if (maxval > maxVal || (maxval == maxVal && fmaxIdx[j] < maxIdx))
        {
            maxIdx = fmaxIdx[j];
            maxVal = maxval;
        }
    }

    for(; i < (s32)len; ++i )
    {
        s8 val = src[i];
        if( val < minVal )
        {
            minVal = val;
            minIdx = (u16)i;
        }
        else if( val > maxVal )
        {
            maxVal = val;
            maxIdx = (u16)i;
        }
    }
}

} // namespace
#endif // CAROTENE_NEON

#define USHORT_BLOCK_MAX_SIZE (1 << 16)

void minMaxLoc(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 &minVal, size_t &minCol, size_t &minRow,
               u8 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = srcBase[0];
    minCol = 0;
    minRow = 0;
    maxVal = srcBase[0];
    maxCol = 0;
    maxRow = 0;
    for(size_t l = 0; l < size.height; ++l)
    {
        const u8 * src = internal::getRowPtr( srcBase,  srcStride, l);
        if (size.width > 128)
        {
            for(size_t blockStart = 0; blockStart < size.width; blockStart += USHORT_BLOCK_MAX_SIZE)
            {
                u8 locMinVal, locMaxVal;
                u16 locMinIdx, locMaxIdx;
                size_t tail = size.width - blockStart;
                minMaxLocBlock(src + blockStart, tail < USHORT_BLOCK_MAX_SIZE ? tail : USHORT_BLOCK_MAX_SIZE,
                               locMinVal, locMinIdx, locMaxVal, locMaxIdx);

                if (locMinVal == 0 && locMaxVal == 255)
                {
                    minCol = blockStart + locMinIdx;
                    maxCol = blockStart + locMaxIdx;
                    minRow = l;
                    maxRow = l;
                    minVal = 0;
                    maxVal = 255;
                    return;
                }
                else
                {
                    if (locMinVal < minVal)
                    {
                        minCol = blockStart + locMinIdx;
                        minRow = l;
                        minVal = locMinVal;
                    }
                    if (locMaxVal > maxVal)
                    {
                        maxCol = blockStart + locMaxIdx;
                        maxRow = l;
                        maxVal = locMaxVal;
                    }
                }
            }
        }
        else
        {
            for(size_t i = 0; i < size.width; ++i )
            {
                u8 val = src[i];
                if( val < minVal )
                {
                    minVal = val;
                    minCol = i;
                    minRow = l;
                }
                else if( val > maxVal )
                {
                    maxVal = val;
                    maxCol = i;
                    maxRow = l;
                }
            }
        }

    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

void minMaxLoc(const Size2D &size,
               const s8 * srcBase, ptrdiff_t srcStride,
               s8 &minVal, size_t &minCol, size_t &minRow,
               s8 &maxVal, size_t &maxCol, size_t &maxRow)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    minVal = srcBase[0];
    minCol = 0;
    minRow = 0;
    maxVal = srcBase[0];
    maxCol = 0;
    maxRow = 0;
    for(size_t l = 0; l < size.height; ++l)
    {
        const s8 * src = internal::getRowPtr( srcBase,  srcStride, l);
        if (size.width > 128)
        {
            for(size_t blockStart = 0; blockStart < size.width; blockStart += USHORT_BLOCK_MAX_SIZE)
            {
                s8 locMinVal, locMaxVal;
                u16 locMinIdx, locMaxIdx;
                size_t tail = size.width - blockStart;
                minMaxLocBlock(src + blockStart, tail < USHORT_BLOCK_MAX_SIZE ? tail : USHORT_BLOCK_MAX_SIZE,
                               locMinVal, locMinIdx, locMaxVal, locMaxIdx);

                if (locMinVal == -128 && locMaxVal == 127)
                {
                    minCol = blockStart + locMinIdx;
                    maxCol = blockStart + locMaxIdx;
                    minRow = l;
                    maxRow = l;
                    minVal = -128;
                    maxVal = 127;
                    return;
                }
                else
                {
                    if (locMinVal < minVal)
                    {
                        minCol = blockStart + locMinIdx;
                        minRow = l;
                        minVal = locMinVal;
                    }
                    if (locMaxVal > maxVal)
                    {
                        maxCol = blockStart + locMaxIdx;
                        maxRow = l;
                        maxVal = locMaxVal;
                    }
                }
            }
        }
        else
        {
            for(size_t i = 0; i < size.width; ++i )
            {
                s8 val = src[i];
                if( val < minVal )
                {
                    minVal = val;
                    minRow = l;
                    minCol = i;
                }
                else if( val > maxVal )
                {
                    maxVal = val;
                    maxRow = l;
                    maxCol = i;
                }
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)minVal;
    (void)minCol;
    (void)minRow;
    (void)maxVal;
    (void)maxCol;
    (void)maxRow;
#endif
}

} // namespace CAROTENE_NS
