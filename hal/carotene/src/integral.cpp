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
 * Copyright (C) 2012-2014, NVIDIA Corporation, all rights reserved.
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

void integral(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u32 * sumBase, ptrdiff_t sumStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint32x4_t v_zero = vmovq_n_u32(0u);

    // the first iteration
    const u8 * src = internal::getRowPtr(srcBase, srcStride, 0);
    u32 * sum = internal::getRowPtr(sumBase, sumStride, 0);

    uint32x4_t prev = v_zero;
    size_t j = 0u;

    for ( ; j + 7 < size.width; j += 8)
    {
        internal::prefetch(sum + j);
        internal::prefetch(src + j);

        uint8x8_t el8shr0 = vld1_u8(src + j);
        uint8x8_t el8shr1 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(el8shr0), 8));
        uint8x8_t el8shr2 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(el8shr0), 16));
        uint8x8_t el8shr3 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(el8shr0), 24));

        uint16x8_t el8shr12 =  vaddl_u8(el8shr1, el8shr2);
        uint16x8_t el8shr03 =  vaddl_u8(el8shr0, el8shr3);

        uint16x8_t el8 = vaddq_u16(el8shr12, el8shr03);
        uint16x4_t el4h = vadd_u16(vget_low_u16(el8), vget_high_u16(el8));

        uint32x4_t vsuml = vaddw_u16(prev, vget_low_u16(el8));
        uint32x4_t vsumh = vaddw_u16(prev, el4h);

        vst1q_u32(sum + j, vsuml);
        vst1q_u32(sum + j + 4, vsumh);

        prev = vaddw_u16(prev, vdup_lane_u16(el4h, 3));
    }

    for (u32 v = vgetq_lane_u32(prev, 3); j < size.width; ++j)
        sum[j] = (v += src[j]);

    // the others
    for (size_t i = 1; i < size.height ; ++i)
    {
        src = internal::getRowPtr(srcBase, srcStride, i);
        u32 * prevSum = internal::getRowPtr(sumBase, sumStride, i - 1);
        sum = internal::getRowPtr(sumBase, sumStride, i);

        prev = v_zero;
        j = 0u;

        for ( ; j + 7 < size.width; j += 8)
        {
            internal::prefetch(sum + j);
            internal::prefetch(src + j);

            uint32x4_t vsuml = vld1q_u32(prevSum + j);
            uint32x4_t vsumh = vld1q_u32(prevSum + j + 4);

            uint8x8_t el8shr0 = vld1_u8(src + j);
            uint8x8_t el8shr1 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(el8shr0), 8));
            uint8x8_t el8shr2 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(el8shr0), 16));
            uint8x8_t el8shr3 = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(el8shr0), 24));

            vsuml = vaddq_u32(vsuml, prev);
            vsumh = vaddq_u32(vsumh, prev);

            uint16x8_t el8shr12 =  vaddl_u8(el8shr1, el8shr2);
            uint16x8_t el8shr03 =  vaddl_u8(el8shr0, el8shr3);

            uint16x8_t el8 = vaddq_u16(el8shr12, el8shr03);
            uint16x4_t el4h = vadd_u16(vget_low_u16(el8), vget_high_u16(el8));

            vsuml = vaddw_u16(vsuml, vget_low_u16(el8));
            vsumh = vaddw_u16(vsumh, el4h);

            vst1q_u32(sum + j, vsuml);
            vst1q_u32(sum + j + 4, vsumh);

            prev = vaddw_u16(prev, vdup_lane_u16(el4h, 3));
        }

        for (u32 v = vgetq_lane_u32(prev, 3); j < size.width; ++j)
            sum[j] = (v += src[j]) + prevSum[j];
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)sumBase;
    (void)sumStride;
#endif
}

void sqrIntegral(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 f64 * sqsumBase, ptrdiff_t sqsumStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    uint16x8_t v_zero8 = vmovq_n_u16(0u);

    // the first iteration
    const u8 * src = internal::getRowPtr(srcBase, srcStride, 0);
    f64 * sqsum = internal::getRowPtr(sqsumBase, sqsumStride, 0);

    double prev = 0.;
    size_t j = 0u;

    for ( ; j + 7 < size.width; j += 8)
    {
        internal::prefetch(sqsum + j);
        internal::prefetch(src + j);

        uint8x8_t vsrc = vld1_u8(src + j);

        uint16x8_t el8shr0 = vmull_u8(vsrc, vsrc);
        uint16x8_t el8shr1 = vextq_u16(v_zero8, el8shr0, 7);

        uint32x4_t el8shr01l =  vaddl_u16(vget_low_u16(el8shr0), vget_low_u16(el8shr1));
        uint32x4_t el8shr01h =  vaddl_u16(vget_high_u16(el8shr0), vget_high_u16(el8shr1));

        uint32x4_t el4h = vaddq_u32(el8shr01l, el8shr01h);

        uint32x2_t el2l = vadd_u32(vget_low_u32(el8shr01l), vget_high_u32(el8shr01l));
        uint32x2_t el2hl = vadd_u32(vget_low_u32(el4h), vget_high_u32(el8shr01l));
        uint32x2_t el2hh = vadd_u32(vget_low_u32(el4h), vget_high_u32(el4h));

        u32 buf[8];
        vst1_u32(buf, vget_low_u32(el8shr01l));
        vst1_u32(buf+2, el2l);
        vst1_u32(buf+4, el2hl);
        vst1_u32(buf+6, el2hh);
        for(u32 k=0; k < 8; k++)
            sqsum[j+k] = prev + buf[k];
        prev += buf[7];
    }

    for (; j < size.width; ++j)
        sqsum[j] = (prev += src[j]*src[j]);

    // the others
    for (size_t i = 1; i < size.height ; ++i)
    {
        src = internal::getRowPtr(srcBase, srcStride, i);
        f64 * prevSqSum = internal::getRowPtr(sqsumBase, sqsumStride, i - 1);
        sqsum = internal::getRowPtr(sqsumBase, sqsumStride, i);

        prev = 0.;
        j = 0u;

        for ( ; j + 7 < size.width; j += 8)
        {
            internal::prefetch(sqsum + j);
            internal::prefetch(src + j);

            uint8x8_t vsrc = vld1_u8(src + j);

            uint16x8_t el8shr0 = vmull_u8(vsrc, vsrc);
            uint16x8_t el8shr1 = vextq_u16(v_zero8, el8shr0, 7);

            uint32x4_t el8shr01l =  vaddl_u16(vget_low_u16(el8shr0), vget_low_u16(el8shr1));
            uint32x4_t el8shr01h =  vaddl_u16(vget_high_u16(el8shr0), vget_high_u16(el8shr1));

            uint32x4_t el4h = vaddq_u32(el8shr01l, el8shr01h);

            uint32x2_t el2l = vadd_u32(vget_low_u32(el8shr01l), vget_high_u32(el8shr01l));
            uint32x2_t el2hl = vadd_u32(vget_low_u32(el4h), vget_high_u32(el8shr01l));
            uint32x2_t el2hh = vadd_u32(vget_low_u32(el4h), vget_high_u32(el4h));

            u32 buf[8];
            vst1_u32(buf, vget_low_u32(el8shr01l));
            vst1_u32(buf+2, el2l);
            vst1_u32(buf+4, el2hl);
            vst1_u32(buf+6, el2hh);
            for(u32 k=0; k < 8; k++)
                sqsum[j+k] = prev + prevSqSum[j+k] + buf[k];
            prev += buf[7];
        }

        for (; j < size.width; ++j)
            sqsum[j] = (prev += src[j]*src[j]) + prevSqSum[j];
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)sqsumBase;
    (void)sqsumStride;
#endif
}

} // namespace CAROTENE_NS
