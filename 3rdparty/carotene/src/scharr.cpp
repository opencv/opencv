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

#include <vector>

#include "common.hpp"

namespace CAROTENE_NS {

bool isScharr3x3Supported(const Size2D &size, BORDER_MODE border, s32 dx, s32 dy, Margin borderMargin)
{
    return (dx == 0 && dy == 1 &&
                   isSeparableFilter3x3Supported(size, border, 3, 1, borderMargin)) ||
           (dx == 1 && dy == 0 &&
                   isSeparableFilter3x3Supported(size, border, 1, 3, borderMargin));
}

void Scharr3x3(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               s16 * dstBase, ptrdiff_t dstStride,
               s32 dx, s32 dy,
               BORDER_MODE border, u8 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isScharr3x3Supported(size, border, dx, dy, borderMargin));
#ifdef CAROTENE_NEON
    static s16 dw[] = {3, 10, 3};

    if (dy == 1)
        SeparableFilter3x3(size, srcBase, srcStride, dstBase, dstStride,
                           3, 1, dw, 0,
                           border, borderValue, borderMargin);
    else
        SeparableFilter3x3(size, srcBase, srcStride, dstBase, dstStride,
                           1, 3, 0, dw,
                           border, borderValue, borderMargin);
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
#endif
}

void ScharrDeriv(const Size2D &size, s32 cn,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t colsn = size.width*cn;
    size_t roiw8 = colsn > 7 ? colsn - 7 : 0;

    ptrdiff_t delta = (ptrdiff_t)(((size.width + 2)*cn + 15) & -16);//align size
    std::vector<s16> _tempBuf((delta << 1) + 64);
    s16 *trow0 = internal::alignPtr(&_tempBuf[cn], 16), *trow1 = internal::alignPtr(trow0 + delta, 16);

    int16x8_t vc3 = vmovq_n_s16(3);
    int16x8_t vc10 = vmovq_n_s16(10);
    uint8x8_t v8c10 = vmov_n_u8(10);

    for(size_t y = 0; y < size.height; y++ )
    {
        const u8* srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : size.height > 1 ? 1 : 0);
        const u8* srow1 = internal::getRowPtr(srcBase, srcStride, y);
        const u8* srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height > 1 ? size.height-2 : 0);
        s16* drow = internal::getRowPtr(dstBase, dstStride, y);

        // do vertical convolution
        size_t x = 0;
        for( ; x < roiw8; x += 8 )
        {
            internal::prefetch(srow0 + x);
            internal::prefetch(srow1 + x);
            internal::prefetch(srow2 + x);
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
            __asm__ (
                "vld1.8 {d0}, [%[src0]]                                \n\t"
                "vld1.8 {d2}, [%[src2]]                                \n\t"
                "vld1.8 {d1}, [%[src1]]                                \n\t"
                "vaddl.u8 q2, d2, d0                                   \n\t"
                "vmull.u8 q3, d1, %[vc10]                              \n\t"
                "vsubl.u8 q4, d2, d0                                   \n\t"
                "vmla.s16 q3, q2, %q[vc3]                              \n\t"
                "vst1.16 {d8-d9}, [%[out1],:128]                       \n\t"
                "vst1.16 {d6-d7}, [%[out0],:128]                       \n\t"
                :
                : [out0] "r" (trow0 + x),
                  [out1] "r" (trow1 + x),
                  [src0] "r" (srow0 + x),
                  [src1] "r" (srow1 + x),
                  [src2] "r" (srow2 + x),
                  [vc10] "w" (v8c10), [vc3] "w" (vc3)
                : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15"
            );
#else
            uint8x8_t s0 = vld1_u8(srow0 + x);
            uint8x8_t s1 = vld1_u8(srow1 + x);
            uint8x8_t s2 = vld1_u8(srow2 + x);

            int16x8_t s1x10 = vreinterpretq_s16_u16(vmull_u8(s1, v8c10));
            int16x8_t s02 = vreinterpretq_s16_u16(vaddl_u8(s2, s0));
            int16x8_t t1 = vreinterpretq_s16_u16(vsubl_u8(s2, s0));
            int16x8_t t0 = vmlaq_s16(s1x10, s02, vc3);

            vst1q_s16(trow1 + x, t1);
            vst1q_s16(trow0 + x, t0);
#endif
        }
        for( ; x < colsn; x++ )
        {
            trow0[x] = (s16)((srow0[x] + srow2[x])*3 + srow1[x]*10);
            trow1[x] = (s16)(srow2[x] - srow0[x]);
        }

        // make border
        size_t x0 = (size.width > 1 ? cn : 0), x1 = (size.width > 1 ? (size.width-2)*cn : 0);
        for( s32 k = 0; k < cn; k++ )
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
        for( ; x < roiw8; x += 8 )
        {
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
            __asm__ (
                "vld1.16 {d4-d5}, [%[s2ptr]]                           \n\t"
                "vld1.16 {d8-d9}, [%[s4ptr]]                           \n\t"
                "vld1.16 {d6-d7}, [%[s3ptr],:128]                      \n\t"
                "vld1.16 {d0-d1}, [%[s0ptr]]                           \n\t"
                "vld1.16 {d2-d3}, [%[s1ptr]]                           \n\t"
                "vadd.i16 q7, q2, q4                                   \n\t"
                "vmul.s16 q6, q3, %q[vc10]                             \n\t"
                "vsub.s16 q5, q1, q0                                   \n\t"
                "vmla.s16 q6, q7, %q[vc3]                              \n\t"
                "vst2.16 {d10-d13}, [%[out]]                           \n\t"
                :
                : [out] "r" (drow + x * 2),
                  [s0ptr] "r" (trow0 + x - cn),
                  [s1ptr] "r" (trow0 + x + cn),
                  [s2ptr] "r" (trow1 + x - cn),
                  [s3ptr] "r" (trow1 + x),
                  [s4ptr] "r" (trow1 + x + cn),
                  [vc10] "w" (vc10), [vc3] "w" (vc3)
                : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15"
            );
#else
            int16x8_t s0 = vld1q_s16(trow0 + x - cn);
            int16x8_t s1 = vld1q_s16(trow0 + x + cn);
            int16x8_t s2 = vld1q_s16(trow1 + x - cn);
            int16x8_t s3 = vld1q_s16(trow1 + x);
            int16x8_t s4 = vld1q_s16(trow1 + x + cn);

            int16x8_t s3x10 = vmulq_s16(s3, vc10);
            int16x8_t s24 = vaddq_s16(s2, s4);

            int16x8x2_t vr;
            vr.val[0] = vsubq_s16(s1, s0);
            vr.val[1] = vmlaq_s16(s3x10, s24, vc3);

            vst2q_s16(drow + x*2, vr);
#endif
        }
        for( ; x < colsn; x++ )
        {
            drow[x*2] = (s16)(trow0[x+cn] - trow0[x-cn]);
            drow[x*2+1] = (s16)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
        }
    }
#else
    (void)size;
    (void)cn;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

} // namespace CAROTENE_NS
