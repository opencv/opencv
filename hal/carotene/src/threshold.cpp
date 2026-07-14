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

void thresholdBinary(const Size2D &size,
                     const u8 *srcBase, ptrdiff_t srcStride,
                     u8 *dstBase, ptrdiff_t dstStride,
                     u8 threshold, u8 trueValue, u8 falseValue)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint8x16_t vthreshold = vdupq_n_u8(threshold);
    uint8x8_t  vthreshold8 = vdup_n_u8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    if(trueValue == 255 && falseValue == 0)
    {
        for (size_t i = 0; i < size.height; ++i) {
            const u8* src = internal::getRowPtr(srcBase, srcStride, i);
            u8* dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw32; j += 32) {
                internal::prefetch(src + j);
                uint8x16_t v0 = vld1q_u8(src + j);
                uint8x16_t v1 = vld1q_u8(src + j + 16);
                uint8x16_t r0 = vcgtq_u8(v0, vthreshold);
                uint8x16_t r1 = vcgtq_u8(v1, vthreshold);
                vst1q_u8(dst + j, r0);
                vst1q_u8(dst + j + 16, r1);
            }
            for (; j < roiw8; j += 8) {
                uint8x8_t v0 = vld1_u8(src + j);
                uint8x8_t r0 = vcgt_u8(v0, vthreshold8);
                vst1_u8(dst + j, r0);
            }

            for (; j < size.width; j++) {
                *(dst + j) = *(src + j) > threshold ? 255 : 0;
            }
        }
    }
    else
    {
        uint8x16_t vtrue_value = vdupq_n_u8(trueValue);
        uint8x8_t  vtrue_value8 = vdup_n_u8(trueValue);
        uint8x16_t vfalse_value = vdupq_n_u8(falseValue);
        uint8x8_t  vfalse_value8 = vdup_n_u8(falseValue);

        for (size_t i = 0; i < size.height; ++i) {
            const u8* src = internal::getRowPtr(srcBase, srcStride, i);
            u8* dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw32; j += 32) {
                internal::prefetch(src + j);
                uint8x16_t v0 = vld1q_u8(src + j);
                uint8x16_t v1 = vld1q_u8(src + j + 16);
                uint8x16_t r0 = vcgtq_u8(v0, vthreshold);
                uint8x16_t r1 = vcgtq_u8(v1, vthreshold);
                uint8x16_t r0a = vbslq_u8(r0, vtrue_value, vfalse_value);
                uint8x16_t r1a = vbslq_u8(r1, vtrue_value, vfalse_value);
                vst1q_u8(dst + j, r0a);
                vst1q_u8(dst + j + 16, r1a);
            }
            for (; j < roiw8; j += 8) {
                uint8x8_t v0 = vld1_u8(src + j);
                uint8x8_t r0 = vcgt_u8(v0, vthreshold8);
                uint8x8_t r0a = vbsl_u8(r0, vtrue_value8, vfalse_value8);
                vst1_u8(dst + j, r0a);
            }

            for (; j < size.width; j++) {
                *(dst + j) = *(src + j) > threshold ? trueValue : falseValue;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)trueValue;
    (void)falseValue;
#endif
}

void thresholdRange(const Size2D &size,
                    const u8 *srcBase, ptrdiff_t srcStride,
                    u8 *dstBase, ptrdiff_t dstStride,
                    u8 lowerThreshold, u8 upperThreshold,
                    u8 trueValue, u8 falseValue)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    uint8x16_t v_lower = vdupq_n_u8(lowerThreshold), v_upper = vdupq_n_u8(upperThreshold);
    uint8x8_t  v_lower8 = vdup_n_u8(lowerThreshold), v_upper8 = vdup_n_u8(upperThreshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    if(trueValue == 255 && falseValue == 0)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
            u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw32; j += 32)
            {
                internal::prefetch(src + j);
                uint8x16_t v_src0 = vld1q_u8(src + j), v_src1 = vld1q_u8(src + j + 16);
                uint8x16_t v_dst0 = vandq_u8(vcgeq_u8(v_src0, v_lower), vcleq_u8(v_src0, v_upper));
                uint8x16_t v_dst1 = vandq_u8(vcgeq_u8(v_src1, v_lower), vcleq_u8(v_src1, v_upper));
                vst1q_u8(dst + j, v_dst0);
                vst1q_u8(dst + j + 16, v_dst1);
            }
            for (; j < roiw8; j += 8)
            {
                uint8x8_t v_src = vld1_u8(src + j);
                uint8x8_t v_dst = vand_u8(vcge_u8(v_src, v_lower8), vcle_u8(v_src, v_upper8));
                vst1_u8(dst + j, v_dst);
            }

            for (; j < size.width; j++)
            {
                u8 srcVal = src[j];
                dst[j] = lowerThreshold <= srcVal && srcVal <= upperThreshold ? 255 : 0;
            }
        }
    }
    else
    {
        uint8x16_t vtrue_value = vdupq_n_u8(trueValue);
        uint8x8_t  vtrue_value8 = vdup_n_u8(trueValue);
        uint8x16_t vfalse_value = vdupq_n_u8(falseValue);
        uint8x8_t  vfalse_value8 = vdup_n_u8(falseValue);

        for (size_t i = 0; i < size.height; ++i)
        {
            const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
            u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw32; j += 32)
            {
                internal::prefetch(src + j);
                uint8x16_t v_src0 = vld1q_u8(src + j), v_src1 = vld1q_u8(src + j + 16);
                uint8x16_t v_dst0 = vandq_u8(vcgeq_u8(v_src0, v_lower), vcleq_u8(v_src0, v_upper));
                uint8x16_t v_dst1 = vandq_u8(vcgeq_u8(v_src1, v_lower), vcleq_u8(v_src1, v_upper));
                v_dst0 = vbslq_u8(v_dst0, vtrue_value, vfalse_value);
                v_dst1 = vbslq_u8(v_dst1, vtrue_value, vfalse_value);
                vst1q_u8(dst + j, v_dst0);
                vst1q_u8(dst + j + 16, v_dst1);
            }
            for (; j < roiw8; j += 8)
            {
                uint8x8_t v_src = vld1_u8(src + j);
                uint8x8_t v_dst = vand_u8(vcge_u8(v_src, v_lower8), vcle_u8(v_src, v_upper8));
                v_dst = vbsl_u8(v_dst, vtrue_value8, vfalse_value8);
                vst1_u8(dst + j, v_dst);
            }

            for (; j < size.width; j++)
            {
                u8 srcVal = src[j];
                dst[j] = lowerThreshold <= srcVal && srcVal <= upperThreshold ? trueValue : falseValue;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)lowerThreshold;
    (void)upperThreshold;
    (void)trueValue;
    (void)falseValue;
#endif
}

void thresholdBinary(const Size2D &size,
                     const u8 *srcBase, ptrdiff_t srcStride,
                     u8 *dstBase, ptrdiff_t dstStride,
                     u8 threshold, u8 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint8x16_t vthreshold = vdupq_n_u8(threshold);
    uint8x16_t vvalue = vdupq_n_u8(value);
    uint8x8_t  vthreshold8 = vdup_n_u8(threshold);
    uint8x8_t  vvalue8 = vdup_n_u8(value);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            uint8x16_t v0  = vld1q_u8(src + j);
            uint8x16_t v1  = vld1q_u8(src + j + 16);
            uint8x16_t r0 = vcgtq_u8(v0, vthreshold);
            uint8x16_t r1 = vcgtq_u8(v1, vthreshold);
            uint8x16_t r0a = vandq_u8(r0, vvalue);
            uint8x16_t r1a = vandq_u8(r1, vvalue);
            vst1q_u8(dst + j, r0a);
            vst1q_u8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v0  = vld1_u8(src + j);
            uint8x8_t r0 = vcgt_u8(v0, vthreshold8);
            uint8x8_t r0a = vand_u8(r0, vvalue8);
            vst1_u8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? value : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdBinaryInv(const Size2D &size,
                        const u8 *srcBase, ptrdiff_t srcStride,
                        u8 *dstBase, ptrdiff_t dstStride,
                        u8 threshold, u8 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint8x16_t vthreshold = vdupq_n_u8(threshold);
    uint8x16_t vvalue = vdupq_n_u8(value);
    uint8x8_t  vthreshold8 = vdup_n_u8(threshold);
    uint8x8_t  vvalue8 = vdup_n_u8(value);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            uint8x16_t v0  = vld1q_u8(src + j);
            uint8x16_t v1  = vld1q_u8(src + j + 16);
            uint8x16_t r0 = vcleq_u8(v0, vthreshold);
            uint8x16_t r1 = vcleq_u8(v1, vthreshold);
            uint8x16_t r0a = vandq_u8(r0, vvalue);
            uint8x16_t r1a = vandq_u8(r1, vvalue);
            vst1q_u8(dst + j, r0a);
            vst1q_u8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v0  = vld1_u8(src + j);
            uint8x8_t r0 = vcle_u8(v0, vthreshold8);
            uint8x8_t r0a = vand_u8(r0, vvalue8);
            vst1_u8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : value;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdTruncate(const Size2D &size,
                       const u8 *srcBase, ptrdiff_t srcStride,
                       u8 *dstBase, ptrdiff_t dstStride,
                       u8 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint8x16_t vthreshold = vdupq_n_u8(threshold);
    uint8x8_t  vthreshold8 = vdup_n_u8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            uint8x16_t v0  = vld1q_u8(src + j);
            uint8x16_t v1  = vld1q_u8(src + j + 16);
            uint8x16_t r0 = vqsubq_u8(v0, vthreshold);
            uint8x16_t r1 = vqsubq_u8(v1, vthreshold);
            uint8x16_t r0a = vqsubq_u8(v0, r0);
            uint8x16_t r1a = vqsubq_u8(v1, r1);
            vst1q_u8(dst + j, r0a);
            vst1q_u8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v0  = vld1_u8(src + j);
            uint8x8_t r0 = vqsub_u8(v0, vthreshold8);
            uint8x8_t r0a = vqsub_u8(v0, r0);
            vst1_u8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? threshold : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZero(const Size2D &size,
                     const u8 *srcBase, ptrdiff_t srcStride,
                     u8 *dstBase, ptrdiff_t dstStride,
                     u8 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint8x16_t vthreshold = vdupq_n_u8(threshold);
    uint8x8_t  vthreshold8 = vdup_n_u8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            uint8x16_t v0  = vld1q_u8(src + j);
            uint8x16_t v1  = vld1q_u8(src + j + 16);
            uint8x16_t r0 = vcgtq_u8(v0, vthreshold);
            uint8x16_t r1 = vcgtq_u8(v1, vthreshold);
            uint8x16_t r0a = vandq_u8(v0, r0);
            uint8x16_t r1a = vandq_u8(v1, r1);
            vst1q_u8(dst + j, r0a);
            vst1q_u8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v0  = vld1_u8(src + j);
            uint8x8_t r0 = vcgt_u8(v0, vthreshold8);
            uint8x8_t r0a = vand_u8(v0, r0);
            vst1_u8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? *(src + j) : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZeroInv(const Size2D &size,
                        const u8 *srcBase, ptrdiff_t srcStride,
                        u8 *dstBase, ptrdiff_t dstStride,
                        u8 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint8x16_t vthreshold = vdupq_n_u8(threshold);
    uint8x8_t  vthreshold8 = vdup_n_u8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            uint8x16_t v0  = vld1q_u8(src + j);
            uint8x16_t v1  = vld1q_u8(src + j + 16);
            uint8x16_t r0 = vcgtq_u8(v0, vthreshold);
            uint8x16_t r1 = vcgtq_u8(v1, vthreshold);
            uint8x16_t r0a = vbicq_u8(v0, r0);
            uint8x16_t r1a = vbicq_u8(v1, r1);
            vst1q_u8(dst + j, r0a);
            vst1q_u8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v0  = vld1_u8(src + j);
            uint8x8_t r0 = vcgt_u8(v0, vthreshold8);
            uint8x8_t r0a = vbic_u8(v0, r0);
            vst1_u8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdBinary(const Size2D &size,
                     const s8 *srcBase, ptrdiff_t srcStride,
                     s8 *dstBase, ptrdiff_t dstStride,
                     s8 threshold, s8 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int8x16_t vthreshold = vdupq_n_s8(threshold);
    int8x16_t vvalue = vdupq_n_s8(value);
    int8x8_t  vthreshold8 = vdup_n_s8(threshold);
    int8x8_t  vvalue8 = vdup_n_s8(value);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s8* src = internal::getRowPtr(srcBase, srcStride, i);
        s8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            int8x16_t v0  = vld1q_s8(src + j);
            int8x16_t v1  = vld1q_s8(src + j + 16);
            int8x16_t r0 = vreinterpretq_s8_u8(vcgtq_s8(v0, vthreshold));
            int8x16_t r1 = vreinterpretq_s8_u8(vcgtq_s8(v1, vthreshold));
            int8x16_t r0a = vandq_s8(r0, vvalue);
            int8x16_t r1a = vandq_s8(r1, vvalue);
            vst1q_s8(dst + j, r0a);
            vst1q_s8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            int8x8_t v0  = vld1_s8(src + j);
            int8x8_t r0 = vreinterpret_s8_u8(vcgt_s8(v0, vthreshold8));
            int8x8_t r0a = vand_s8(r0, vvalue8);
            vst1_s8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? value : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdBinaryInv(const Size2D &size,
                        const s8 *srcBase, ptrdiff_t srcStride,
                        s8 *dstBase, ptrdiff_t dstStride,
                        s8 threshold, s8 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int8x16_t vthreshold = vdupq_n_s8(threshold);
    int8x16_t vvalue = vdupq_n_s8(value);
    int8x8_t  vthreshold8 = vdup_n_s8(threshold);
    int8x8_t  vvalue8 = vdup_n_s8(value);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s8* src = internal::getRowPtr(srcBase, srcStride, i);
        s8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            int8x16_t v0  = vld1q_s8(src + j);
            int8x16_t v1  = vld1q_s8(src + j + 16);
            int8x16_t r0 = vreinterpretq_s8_u8(vcleq_s8(v0, vthreshold));
            int8x16_t r1 = vreinterpretq_s8_u8(vcleq_s8(v1, vthreshold));
            int8x16_t r0a = vandq_s8(r0, vvalue);
            int8x16_t r1a = vandq_s8(r1, vvalue);
            vst1q_s8(dst + j, r0a);
            vst1q_s8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            int8x8_t v0  = vld1_s8(src + j);
            int8x8_t r0 = vreinterpret_s8_u8(vcle_s8(v0, vthreshold8));
            int8x8_t r0a = vand_s8(r0, vvalue8);
            vst1_s8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : value;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdTruncate(const Size2D &size,
                       const s8 *srcBase, ptrdiff_t srcStride,
                       s8 *dstBase, ptrdiff_t dstStride,
                       s8 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int8x16_t vthreshold = vdupq_n_s8(threshold);
    int8x8_t  vthreshold8 = vdup_n_s8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s8* src = internal::getRowPtr(srcBase, srcStride, i);
        s8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            int8x16_t v0  = vld1q_s8(src + j);
            int8x16_t v1  = vld1q_s8(src + j + 16);
            int8x16_t r0 = vqsubq_s8(v0, vthreshold);
            int8x16_t r1 = vqsubq_s8(v1, vthreshold);
            int8x16_t r0a = vqsubq_s8(v0, r0);
            int8x16_t r1a = vqsubq_s8(v1, r1);
            vst1q_s8(dst + j, r0a);
            vst1q_s8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            int8x8_t v0  = vld1_s8(src + j);
            int8x8_t r0 = vqsub_s8(v0, vthreshold8);
            int8x8_t r0a = vqsub_s8(v0, r0);
            vst1_s8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? threshold : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZero(const Size2D &size,
                     const s8 *srcBase, ptrdiff_t srcStride,
                     s8 *dstBase, ptrdiff_t dstStride,
                     s8 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int8x16_t vthreshold = vdupq_n_s8(threshold);
    int8x8_t  vthreshold8 = vdup_n_s8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s8* src = internal::getRowPtr(srcBase, srcStride, i);
        s8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            int8x16_t v0  = vld1q_s8(src + j);
            int8x16_t v1  = vld1q_s8(src + j + 16);
            int8x16_t r0 = vreinterpretq_s8_u8(vcgtq_s8(v0, vthreshold));
            int8x16_t r1 = vreinterpretq_s8_u8(vcgtq_s8(v1, vthreshold));
            int8x16_t r0a = vandq_s8(v0, r0);
            int8x16_t r1a = vandq_s8(v1, r1);
            vst1q_s8(dst + j, r0a);
            vst1q_s8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            int8x8_t v0  = vld1_s8(src + j);
            int8x8_t r0 = vreinterpret_s8_u8(vcgt_s8(v0, vthreshold8));
            int8x8_t r0a = vand_s8(v0, r0);
            vst1_s8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? *(src + j) : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZeroInv(const Size2D &size,
                        const s8 *srcBase, ptrdiff_t srcStride,
                        s8 *dstBase, ptrdiff_t dstStride,
                        s8 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int8x16_t vthreshold = vdupq_n_s8(threshold);
    int8x8_t  vthreshold8 = vdup_n_s8(threshold);
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s8* src = internal::getRowPtr(srcBase, srcStride, i);
        s8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            int8x16_t v0  = vld1q_s8(src + j);
            int8x16_t v1  = vld1q_s8(src + j + 16);
            int8x16_t r0 = vreinterpretq_s8_u8(vcgtq_s8(v0, vthreshold));
            int8x16_t r1 = vreinterpretq_s8_u8(vcgtq_s8(v1, vthreshold));
            int8x16_t r0a = vbicq_s8(v0, r0);
            int8x16_t r1a = vbicq_s8(v1, r1);
            vst1q_s8(dst + j, r0a);
            vst1q_s8(dst + j + 16, r1a);
        }
        for (; j < roiw8; j += 8)
        {
            int8x8_t v0  = vld1_s8(src + j);
            int8x8_t r0 = vreinterpret_s8_u8(vcgt_s8(v0, vthreshold8));
            int8x8_t r0a = vbic_s8(v0, r0);
            vst1_s8(dst + j, r0a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdBinary(const Size2D &size,
                     const s16 *srcBase, ptrdiff_t srcStride,
                     s16 *dstBase, ptrdiff_t dstStride,
                     s16 threshold, s16 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int16x8_t vthreshold16 = vdupq_n_s16(threshold);
    int16x8_t vvalue16 = vdupq_n_s16(value);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            int16x8_t v0  = vld1q_s16(src + j);
            int16x8_t v1  = vld1q_s16(src + j + 8);
            uint16x8_t r0 = vcgtq_s16(v0, vthreshold16);
            uint16x8_t r1 = vcgtq_s16(v1, vthreshold16);
            uint16x8_t r0a = vandq_u16(r0, vreinterpretq_u16_s16(vvalue16));
            uint16x8_t r1a = vandq_u16(r1, vreinterpretq_u16_s16(vvalue16));
            vst1q_u16((u16*)dst + j, r0a);
            vst1q_u16((u16*)dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? value : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdBinaryInv(const Size2D &size,
                        const s16 *srcBase, ptrdiff_t srcStride,
                        s16 *dstBase, ptrdiff_t dstStride,
                        s16 threshold, s16 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int16x8_t vthreshold16 = vdupq_n_s16(threshold);
    int16x8_t vvalue16 = vdupq_n_s16(value);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            int16x8_t v0  = vld1q_s16(src + j);
            int16x8_t v1  = vld1q_s16(src + j + 8);
            uint16x8_t r0 = vcleq_s16(v0, vthreshold16);
            uint16x8_t r1 = vcleq_s16(v1, vthreshold16);
            uint16x8_t r0a = vandq_u16(r0, vreinterpretq_u16_s16(vvalue16));
            uint16x8_t r1a = vandq_u16(r1, vreinterpretq_u16_s16(vvalue16));
            vst1q_s16(dst + j, vreinterpretq_s16_u16(r0a));
            vst1q_s16(dst + j + 8, vreinterpretq_s16_u16(r1a));
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : value;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdTruncate(const Size2D &size,
                       const s16 *srcBase, ptrdiff_t srcStride,
                       s16 *dstBase, ptrdiff_t dstStride,
                       s16 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int16x8_t vthreshold16 = vdupq_n_s16(threshold);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            int16x8_t v0  = vld1q_s16(src + j);
            int16x8_t v1  = vld1q_s16(src + j + 8);
            int16x8_t r0 = vminq_s16(v0, vthreshold16);
            int16x8_t r1 = vminq_s16(v1, vthreshold16);
            vst1q_s16(dst + j, r0);
            vst1q_s16(dst + j + 8, r1);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? threshold : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZero(const Size2D &size,
                     const s16 *srcBase, ptrdiff_t srcStride,
                     s16 *dstBase, ptrdiff_t dstStride,
                     s16 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int16x8_t vthreshold16 = vdupq_n_s16(threshold);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            int16x8_t v0  = vld1q_s16(src + j);
            int16x8_t v1  = vld1q_s16(src + j + 8);
            uint16x8_t r0 = vcgtq_s16(v0, vthreshold16);
            uint16x8_t r1 = vcgtq_s16(v1, vthreshold16);
            uint16x8_t r0a = vandq_u16(vreinterpretq_u16_s16(v0), r0);
            uint16x8_t r1a = vandq_u16(vreinterpretq_u16_s16(v1), r1);
            vst1q_u16((u16*)dst + j, r0a);
            vst1q_u16((u16*)dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? *(src + j) : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZeroInv(const Size2D &size,
                        const s16 *srcBase, ptrdiff_t srcStride,
                        s16 *dstBase, ptrdiff_t dstStride,
                        s16 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int16x8_t vthreshold16 = vdupq_n_s16(threshold);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s16* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            int16x8_t v0  = vld1q_s16(src + j);
            int16x8_t v1  = vld1q_s16(src + j + 8);
            uint16x8_t r0 = vcgtq_s16(v0, vthreshold16);
            uint16x8_t r1 = vcgtq_s16(v1, vthreshold16);
            uint16x8_t r0a = vbicq_u16(vreinterpretq_u16_s16(v0), r0);
            uint16x8_t r1a = vbicq_u16(vreinterpretq_u16_s16(v1), r1);
            vst1q_u16((u16*)dst + j, r0a);
            vst1q_u16((u16*)dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdBinary(const Size2D &size,
                     const u16 *srcBase, ptrdiff_t srcStride,
                     u16 *dstBase, ptrdiff_t dstStride,
                     u16 threshold, u16 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint16x8_t vthreshold16 = vdupq_n_u16(threshold);
    uint16x8_t vvalue16 = vdupq_n_u16(value);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16* src = internal::getRowPtr(srcBase, srcStride, i);
        u16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint16x8_t v0  = vld1q_u16(src + j);
            uint16x8_t v1  = vld1q_u16(src + j + 8);
            uint16x8_t r0 = vcgtq_u16(v0, vthreshold16);
            uint16x8_t r1 = vcgtq_u16(v1, vthreshold16);
            uint16x8_t r0a = vandq_u16(r0, vvalue16);
            uint16x8_t r1a = vandq_u16(r1, vvalue16);
            vst1q_u16(dst + j, r0a);
            vst1q_u16(dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? value : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdBinaryInv(const Size2D &size,
                        const u16 *srcBase, ptrdiff_t srcStride,
                        u16 *dstBase, ptrdiff_t dstStride,
                        u16 threshold, u16 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint16x8_t vthreshold16 = vdupq_n_u16(threshold);
    uint16x8_t vvalue16 = vdupq_n_u16(value);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16* src = internal::getRowPtr(srcBase, srcStride, i);
        u16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint16x8_t v0  = vld1q_u16(src + j);
            uint16x8_t v1  = vld1q_u16(src + j + 8);
            uint16x8_t r0 = vcleq_u16(v0, vthreshold16);
            uint16x8_t r1 = vcleq_u16(v1, vthreshold16);
            uint16x8_t r0a = vandq_u16(r0, vvalue16);
            uint16x8_t r1a = vandq_u16(r1, vvalue16);
            vst1q_u16(dst + j, r0a);
            vst1q_u16(dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : value;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdTruncate(const Size2D &size,
                       const u16 *srcBase, ptrdiff_t srcStride,
                       u16 *dstBase, ptrdiff_t dstStride,
                       u16 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint16x8_t vthreshold16 = vdupq_n_u16(threshold);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16* src = internal::getRowPtr(srcBase, srcStride, i);
        u16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint16x8_t v0  = vld1q_u16(src + j);
            uint16x8_t v1  = vld1q_u16(src + j + 8);
            uint16x8_t r0 = vminq_u16(v0, vthreshold16);
            uint16x8_t r1 = vminq_u16(v1, vthreshold16);
            vst1q_u16(dst + j, r0);
            vst1q_u16(dst + j + 8, r1);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? threshold : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZero(const Size2D &size,
                     const u16 *srcBase, ptrdiff_t srcStride,
                     u16 *dstBase, ptrdiff_t dstStride,
                     u16 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint16x8_t vthreshold16 = vdupq_n_u16(threshold);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16* src = internal::getRowPtr(srcBase, srcStride, i);
        u16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint16x8_t v0  = vld1q_u16(src + j);
            uint16x8_t v1  = vld1q_u16(src + j + 8);
            uint16x8_t r0 = vcgtq_u16(v0, vthreshold16);
            uint16x8_t r1 = vcgtq_u16(v1, vthreshold16);
            uint16x8_t r0a = vandq_u16(v0, r0);
            uint16x8_t r1a = vandq_u16(v1, r1);
            vst1q_u16(dst + j, r0a);
            vst1q_u16(dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? *(src + j) : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZeroInv(const Size2D &size,
                        const u16 *srcBase, ptrdiff_t srcStride,
                        u16 *dstBase, ptrdiff_t dstStride,
                        u16 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint16x8_t vthreshold16 = vdupq_n_u16(threshold);
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16* src = internal::getRowPtr(srcBase, srcStride, i);
        u16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            uint16x8_t v0  = vld1q_u16(src + j);
            uint16x8_t v1  = vld1q_u16(src + j + 8);
            uint16x8_t r0 = vcgtq_u16(v0, vthreshold16);
            uint16x8_t r1 = vcgtq_u16(v1, vthreshold16);
            uint16x8_t r0a = vbicq_u16(v0, r0);
            uint16x8_t r1a = vbicq_u16(v1, r1);
            vst1q_u16(dst + j, r0a);
            vst1q_u16(dst + j + 8, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdBinary(const Size2D &size,
                     const s32 *srcBase, ptrdiff_t srcStride,
                     s32 *dstBase, ptrdiff_t dstStride,
                     s32 threshold, s32 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int32x4_t  vthreshold8 = vdupq_n_s32(threshold);
    int32x4_t  vvalue8 = vdupq_n_s32(value);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32* src = internal::getRowPtr(srcBase, srcStride, i);
        s32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int32x4_t v0  = vld1q_s32(src + j);
            int32x4_t v1  = vld1q_s32(src + j + 4);
            uint32x4_t r0 = vcgtq_s32(v0, vthreshold8);
            uint32x4_t r1 = vcgtq_s32(v1, vthreshold8);
            uint32x4_t r0a = vandq_u32(r0, vreinterpretq_u32_s32(vvalue8));
            uint32x4_t r1a = vandq_u32(r1, vreinterpretq_u32_s32(vvalue8));
            vst1q_u32((u32*)dst + j, r0a);
            vst1q_u32((u32*)dst + j + 4, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? value : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdBinaryInv(const Size2D &size,
                        const s32 *srcBase, ptrdiff_t srcStride,
                        s32 *dstBase, ptrdiff_t dstStride,
                        s32 threshold, s32 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int32x4_t  vthreshold8 = vdupq_n_s32(threshold);
    int32x4_t  vvalue8 = vdupq_n_s32(value);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32* src = internal::getRowPtr(srcBase, srcStride, i);
        s32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int32x4_t v0  = vld1q_s32(src + j);
            int32x4_t v1  = vld1q_s32(src + j + 4);
            uint32x4_t r0 = vcleq_s32(v0, vthreshold8);
            uint32x4_t r1 = vcleq_s32(v1, vthreshold8);
            uint32x4_t r0a = vandq_u32(r0, vreinterpretq_u32_s32(vvalue8));
            uint32x4_t r1a = vandq_u32(r1, vreinterpretq_u32_s32(vvalue8));
            vst1q_s32(dst + j, vreinterpretq_s32_u32(r0a));
            vst1q_s32(dst + j + 4, vreinterpretq_s32_u32(r1a));
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : value;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdTruncate(const Size2D &size,
                       const s32 *srcBase, ptrdiff_t srcStride,
                       s32 *dstBase, ptrdiff_t dstStride,
                       s32 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int32x4_t  vthreshold8 = vdupq_n_s32(threshold);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32* src = internal::getRowPtr(srcBase, srcStride, i);
        s32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int32x4_t v0  = vld1q_s32(src + j);
            int32x4_t v1  = vld1q_s32(src + j + 4);
            int32x4_t r0 = vminq_s32(v0, vthreshold8);
            int32x4_t r1 = vminq_s32(v1, vthreshold8);
            vst1q_s32(dst + j, r0);
            vst1q_s32(dst + j + 4, r1);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? threshold : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZero(const Size2D &size,
                     const s32 *srcBase, ptrdiff_t srcStride,
                     s32 *dstBase, ptrdiff_t dstStride,
                     s32 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int32x4_t  vthreshold8 = vdupq_n_s32(threshold);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32* src = internal::getRowPtr(srcBase, srcStride, i);
        s32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int32x4_t v0  = vld1q_s32(src + j);
            int32x4_t v1  = vld1q_s32(src + j + 4);
            uint32x4_t r0 = vcgtq_s32(v0, vthreshold8);
            uint32x4_t r1 = vcgtq_s32(v1, vthreshold8);
            uint32x4_t r0a = vandq_u32(vreinterpretq_u32_s32(v0), r0);
            uint32x4_t r1a = vandq_u32(vreinterpretq_u32_s32(v1), r1);
            vst1q_u32((u32*)dst + j, r0a);
            vst1q_u32((u32*)dst + j + 4, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? *(src + j) : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZeroInv(const Size2D &size,
                        const s32 *srcBase, ptrdiff_t srcStride,
                        s32 *dstBase, ptrdiff_t dstStride,
                        s32 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    int32x4_t  vthreshold8 = vdupq_n_s32(threshold);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const s32* src = internal::getRowPtr(srcBase, srcStride, i);
        s32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            int32x4_t v0  = vld1q_s32(src + j);
            int32x4_t v1  = vld1q_s32(src + j + 4);
            uint32x4_t r0 = vcgtq_s32(v0, vthreshold8);
            uint32x4_t r1 = vcgtq_s32(v1, vthreshold8);
            uint32x4_t r0a = vbicq_u32(vreinterpretq_u32_s32(v0), r0);
            uint32x4_t r1a = vbicq_u32(vreinterpretq_u32_s32(v1), r1);
            vst1q_u32((u32*)dst + j, r0a);
            vst1q_u32((u32*)dst + j + 4, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdBinary(const Size2D &size,
                     const f32 *srcBase, ptrdiff_t srcStride,
                     f32 *dstBase, ptrdiff_t dstStride,
                     f32 threshold, f32 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    float32x4_t  vthreshold8 = vdupq_n_f32(threshold);
    float32x4_t  vvalue8 = vdupq_n_f32(value);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const f32* src = internal::getRowPtr(srcBase, srcStride, i);
        f32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            float32x4_t v0  = vld1q_f32(src + j);
            float32x4_t v1  = vld1q_f32(src + j + 4);
            uint32x4_t r0 = vcgtq_f32(v0, vthreshold8);
            uint32x4_t r1 = vcgtq_f32(v1, vthreshold8);
            uint32x4_t r0a = vandq_u32(r0, vreinterpretq_u32_f32(vvalue8));
            uint32x4_t r1a = vandq_u32(r1, vreinterpretq_u32_f32(vvalue8));
            vst1q_u32((u32*)dst + j, r0a);
            vst1q_u32((u32*)dst + j + 4, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? value : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdBinaryInv(const Size2D &size,
                        const f32 *srcBase, ptrdiff_t srcStride,
                        f32 *dstBase, ptrdiff_t dstStride,
                        f32 threshold, f32 value)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    float32x4_t  vthreshold8 = vdupq_n_f32(threshold);
    float32x4_t  vvalue8 = vdupq_n_f32(value);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const f32* src = internal::getRowPtr(srcBase, srcStride, i);
        f32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            float32x4_t v0  = vld1q_f32(src + j);
            float32x4_t v1  = vld1q_f32(src + j + 4);
            uint32x4_t r0 = vcleq_f32(v0, vthreshold8);
            uint32x4_t r1 = vcleq_f32(v1, vthreshold8);
            uint32x4_t r0a = vandq_u32(r0, vreinterpretq_u32_f32(vvalue8));
            uint32x4_t r1a = vandq_u32(r1, vreinterpretq_u32_f32(vvalue8));
            vst1q_f32(dst + j, vreinterpretq_f32_u32(r0a));
            vst1q_f32(dst + j + 4, vreinterpretq_f32_u32(r1a));
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : value;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
    (void)value;
#endif
}

void thresholdTruncate(const Size2D &size,
                       const f32 *srcBase, ptrdiff_t srcStride,
                       f32 *dstBase, ptrdiff_t dstStride,
                       f32 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    float32x4_t  vthreshold8 = vdupq_n_f32(threshold);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const f32* src = internal::getRowPtr(srcBase, srcStride, i);
        f32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            float32x4_t v0  = vld1q_f32(src + j);
            float32x4_t v1  = vld1q_f32(src + j + 4);
            float32x4_t r0 = vminq_f32(v0, vthreshold8);
            float32x4_t r1 = vminq_f32(v1, vthreshold8);
            vst1q_f32(dst + j, r0);
            vst1q_f32(dst + j + 4, r1);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? threshold : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZero(const Size2D &size,
                     const f32 *srcBase, ptrdiff_t srcStride,
                     f32 *dstBase, ptrdiff_t dstStride,
                     f32 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    float32x4_t  vthreshold8 = vdupq_n_f32(threshold);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const f32* src = internal::getRowPtr(srcBase, srcStride, i);
        f32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            float32x4_t v0  = vld1q_f32(src + j);
            float32x4_t v1  = vld1q_f32(src + j + 4);
            uint32x4_t r0 = vcgtq_f32(v0, vthreshold8);
            uint32x4_t r1 = vcgtq_f32(v1, vthreshold8);
            uint32x4_t r0a = vandq_u32(vreinterpretq_u32_f32(v0), r0);
            uint32x4_t r1a = vandq_u32(vreinterpretq_u32_f32(v1), r1);
            vst1q_u32((u32*)dst + j, r0a);
            vst1q_u32((u32*)dst + j + 4, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? *(src + j) : 0;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

void thresholdToZeroInv(const Size2D &size,
                        const f32 *srcBase, ptrdiff_t srcStride,
                        f32 *dstBase, ptrdiff_t dstStride,
                        f32 threshold)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    float32x4_t  vthreshold8 = vdupq_n_f32(threshold);
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const f32* src = internal::getRowPtr(srcBase, srcStride, i);
        f32* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw8; j += 8)
        {
            internal::prefetch(src + j);
            float32x4_t v0  = vld1q_f32(src + j);
            float32x4_t v1  = vld1q_f32(src + j + 4);
            uint32x4_t r0 = vcgtq_f32(v0, vthreshold8);
            uint32x4_t r1 = vcgtq_f32(v1, vthreshold8);
            uint32x4_t r0a = vbicq_u32(vreinterpretq_u32_f32(v0), r0);
            uint32x4_t r1a = vbicq_u32(vreinterpretq_u32_f32(v1), r1);
            vst1q_u32((u32*)dst + j, r0a);
            vst1q_u32((u32*)dst + j + 4, r1a);
        }
        for (; j < size.width; j++)
        {
            *(dst + j) = *(src + j) > threshold ? 0 : *(src + j);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)threshold;
#endif
}

} // namespace CAROTENE_NS
