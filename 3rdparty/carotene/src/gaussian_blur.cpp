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
#include "saturate_cast.hpp"
#include "separable_filter.hpp"

namespace CAROTENE_NS {

bool isGaussianBlur3x3Supported(const Size2D &size, BORDER_MODE border)
{
    return isSupportedConfiguration() && size.width >= 8 &&
        (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REPLICATE);
}

void gaussianBlur3x3(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isGaussianBlur3x3Supported(size, border));
#ifdef CAROTENE_NEON
    const uint16x8_t v_border_x4 = vdupq_n_u16(borderValue << 2);
    const uint16x8_t v_zero = vdupq_n_u16(0);
    const uint8x8_t v_border = vdup_n_u8(borderValue);

    uint16x8_t tprev = v_zero, tcurr = v_zero, tnext = v_zero;
    uint16x8_t t0 = v_zero, t1 = v_zero, t2 = v_zero;

    ptrdiff_t width = (ptrdiff_t)size.width, height = (ptrdiff_t)size.height;

    for (ptrdiff_t y = 0; y < height; ++y)
    {
        const u8 * srow0 = y == 0 && border == BORDER_MODE_CONSTANT ? NULL : internal::getRowPtr(srcBase, srcStride, std::max<ptrdiff_t>(y - 1, 0));
        const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, y);
        const u8 * srow2 = y + 1 == height && border == BORDER_MODE_CONSTANT ? NULL : internal::getRowPtr(srcBase, srcStride, std::min(y + 1, height - 1));
        u8 * drow = internal::getRowPtr(dstBase, dstStride, y);

        s16 prevx = 0, currx = 0, nextx = 0;
        ptrdiff_t x = 0;
        const ptrdiff_t bwidth = y + 2 < height ? width : (width - 8);

        // perform vertical convolution
        for ( ; x <= bwidth; x += 8)
        {
            internal::prefetch(srow0 + x);
            internal::prefetch(srow1 + x);
            internal::prefetch(srow2 + x);

            uint8x8_t x0 = !srow0 ? v_border : vld1_u8(srow0 + x);
            uint8x8_t x1 = vld1_u8(srow1 + x);
            uint8x8_t x2 = !srow2 ? v_border : vld1_u8(srow2 + x);

            // calculate values for plain CPU part below if needed
            if (x + 8 >= bwidth)
            {
                ptrdiff_t x3 = x == width ? width - 1 : x;
                ptrdiff_t x4 = border == BORDER_MODE_CONSTANT ? x3 - 1 : std::max<ptrdiff_t>(x3 - 1, 0);

                if (border == BORDER_MODE_CONSTANT && x4 < 0)
                    prevx = borderValue;
                else
                    prevx = (srow2 ? srow2[x4] : borderValue) + (srow1[x4] << 1) + (srow0 ? srow0[x4] : borderValue);

                currx = (srow2 ? srow2[x3] : borderValue) + (srow1[x3] << 1) + (srow0 ? srow0[x3] : borderValue);
            }

            // make shift
            if (x)
            {
                tprev = tcurr;
                tcurr = tnext;
            }

            // and calculate next value
            tnext = vaddq_u16(vaddl_u8(x0, x2), vshll_n_u8(x1, 1));

            // make extrapolation for the first elements
            if (!x)
            {
                // make border
                if (border == BORDER_MODE_CONSTANT)
                    tcurr = v_border_x4;
                else if (border == BORDER_MODE_REPLICATE)
                    tcurr = vdupq_n_u16(vgetq_lane_u16(tnext, 0));

                continue;
            }

            // combine 3 "shifted" vectors
            t0 = vextq_u16(tprev, tcurr, 7);
            t1 = tcurr;
            t2 = vextq_u16(tcurr, tnext, 1);

            // and add them
            t0 = vqaddq_u16(vshlq_n_u16(t1, 1), vqaddq_u16(t0, t2));
            vst1_u8(drow + x - 8, vshrn_n_u16(t0, 4));
        }

        x -= 8;
        if (x == width)
            --x;

        for ( ; x < width; ++x)
        {
            // make extrapolation for the last elements
            if (x + 1 >= width)
            {
                if (border == BORDER_MODE_CONSTANT)
                    nextx = borderValue << 2;
                else if (border == BORDER_MODE_REPLICATE)
                    nextx = srow2[x] + (srow1[x] << 1) + srow0[x];
            }
            else
                nextx = (srow2 ? srow2[x + 1] : borderValue) +
                                (srow1[x + 1] << 1) +
                        (srow0 ? srow0[x + 1] : borderValue);

            f32 val = (prevx + (currx << 1) + nextx) >> 4;
            drow[x] = internal::saturate_cast<u8>((s32)val);

            // make shift
            prevx = currx;
            currx = nextx;
        }
    }
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
#endif
}

bool isGaussianBlur3x3MarginSupported(const Size2D &size, BORDER_MODE border, Margin borderMargin)
{
    return isSeparableFilter3x3Supported(size, border, 0, 0, borderMargin);
}

void gaussianBlur3x3Margin(const Size2D &size,
                           const u8 * srcBase, ptrdiff_t srcStride,
                           u8 * dstBase, ptrdiff_t dstStride,
                           BORDER_MODE border, u8 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isGaussianBlur3x3MarginSupported(size, border, borderMargin));
#ifdef CAROTENE_NEON
    internal::sepFilter3x3<internal::RowFilter3x3S16_121, internal::ColFilter3x3U8_121>::process(
                           size, srcBase, srcStride, dstBase, dstStride,
                           0, 0, border, borderValue, borderMargin);
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
#endif
}

bool isGaussianBlur5x5Supported(const Size2D &size, s32 cn, BORDER_MODE border)
{
    return isSupportedConfiguration() &&
           cn > 0 && cn <= 4 &&
           size.width >= 8 && size.height >= 2 &&
           (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REFLECT101 ||
            border == BORDER_MODE_REFLECT ||
            border == BORDER_MODE_REPLICATE ||
            border == BORDER_MODE_WRAP);
}

void gaussianBlur5x5(const Size2D &size, s32 cn,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE borderType, u8 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isGaussianBlur5x5Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
    size_t colsn = size.width * cn;

    std::vector<u8> _tmp;
    u8 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 4*cn, borderValue);
        tmp = &_tmp[cn << 1];
    }

    ptrdiff_t idx_l1 = internal::borderInterpolate(-1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_l2 = internal::borderInterpolate(-2, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r1 = internal::borderInterpolate(size.width + 0, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r2 = internal::borderInterpolate(size.width + 1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;

    //1-line buffer
    std::vector<u16> _buf(cn * (size.width + 4) + 32 / sizeof(u16));
    u16* lane = internal::alignPtr(&_buf[cn << 1], 32);

    if (borderType == BORDER_MODE_CONSTANT)
        for (s32 k = 0; k < cn; ++k)
        {
            lane[-cn+k] = borderValue;
            lane[-cn-cn+k] = borderValue;
            lane[colsn+k] = borderValue;
            lane[colsn+cn+k] = borderValue;
        }

    uint8x8_t vc6u8 = vmov_n_u8(6);
    uint16x8_t vc6u16 = vmovq_n_u16(6);
    uint16x8_t vc4u16 = vmovq_n_u16(4);

    for (size_t i = 0; i < size.height; ++i)
    {
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        ptrdiff_t idx_rm2 = internal::borderInterpolate(i - 2, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp1 = internal::borderInterpolate(i + 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp2 = internal::borderInterpolate(i + 2, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const u8* ln0 = idx_rm2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm2) : tmp;
        const u8* ln1 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const u8* ln2 = internal::getRowPtr(srcBase, srcStride, i);
        const u8* ln3 = idx_rp1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp1) : tmp;
        const u8* ln4 = idx_rp2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp2) : tmp;

        size_t x = 0;
        for (; x <= colsn - 8; x += 8)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, x % 5 - 2));
            uint8x8_t v0 = vld1_u8(ln0+x);
            uint8x8_t v1 = vld1_u8(ln1+x);
            uint8x8_t v2 = vld1_u8(ln2+x);
            uint8x8_t v3 = vld1_u8(ln3+x);
            uint8x8_t v4 = vld1_u8(ln4+x);

            uint16x8_t v = vaddl_u8(v0, v4);
            uint16x8_t v13 = vaddl_u8(v1, v3);

            v = vmlal_u8(v, v2, vc6u8);
            v = vmlaq_u16(v, v13, vc4u16);

            vst1q_u16(lane + x, v);
        }
        for (; x < colsn; ++x)
            lane[x] = ln0[x] + ln4[x] + u16(4) * (ln1[x] + ln3[x]) + u16(6) * ln2[x];

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lane[-cn+k] = lane[idx_l1 + k];
                lane[-cn-cn+k] = lane[idx_l2 + k];

                lane[colsn+k] = lane[idx_r1 + k];
                lane[colsn+cn+k] = lane[idx_r2 + k];
            }

        //horizontal convolution
        x = 0;
        switch(cn)
        {
        case 1:
            for (; x <= colsn - 8; x += 8)
            {
                internal::prefetch(lane + x);

                uint16x8_t lane0 = vld1q_u16(lane + x - 2);
                uint16x8_t lane4 = vld1q_u16(lane + x + 2);
                uint16x8_t lane1 = vld1q_u16(lane + x - 1);
                uint16x8_t lane3 = vld1q_u16(lane + x + 1);
                uint16x8_t lane2 = vld1q_u16(lane + x + 0);

                uint16x8_t ln04 = vaddq_u16(lane0, lane4);
                uint16x8_t ln13 = vaddq_u16(lane1, lane3);

                uint16x8_t ln042 = vmlaq_u16(ln04, lane2, vc6u16);
                uint16x8_t lsw = vmlaq_u16(ln042, ln13, vc4u16);

                uint8x8_t ls = vrshrn_n_u16(lsw, 8);

                vst1_u8(dst + x, ls);
            }
            break;
        case 2:
            for (; x <= colsn - 8*2; x += 8*2)
            {
                internal::prefetch(lane + x);

                u16* lidx0 = lane + x - 2*2;
                u16* lidx1 = lane + x - 1*2;
                u16* lidx3 = lane + x + 1*2;
                u16* lidx4 = lane + x + 2*2;
#if __GNUC_MINOR__ < 7
                __asm__ __volatile__ (
                    "vld2.16 {d0, d2}, [%[in0]]!                              \n\t"
                    "vld2.16 {d1, d3}, [%[in0]]                               \n\t"
                    "vld2.16 {d8, d10}, [%[in4]]!                             \n\t"
                    "vld2.16 {d9, d11}, [%[in4]]                              \n\t"
                    "vadd.i16 q0, q4                                          \n\t"
                    "vadd.i16 q1, q5                                          \n\t"
                    "vld2.16 {d16, d18}, [%[in1]]!                            \n\t"
                    "vld2.16 {d17, d19}, [%[in1]]                             \n\t"
                    "vld2.16 {d8, d10}, [%[in3]]!                             \n\t"
                    "vld2.16 {d9, d11}, [%[in3]]                              \n\t"
                    "vadd.i16 q4, q8                                          \n\t"
                    "vadd.i16 q5, q9                                          \n\t"
                    "vld2.16 {d16, d18}, [%[in2]]                             \n\t"
                    "vld2.16 {d17, d19}, [%[in22]]                            \n\t"
                    "vmla.i16 q0, q4, %q[c4]                                  \n\t"
                    "vmla.i16 q1, q5, %q[c4]                                  \n\t"
                    "vmla.i16 q0, q8, %q[c6]                                  \n\t"
                    "vmla.i16 q1, q9, %q[c6]                                  \n\t"
                    "vrshrn.u16 d8, q0, #8                                    \n\t"
                    "vrshrn.u16 d9, q1, #8                                    \n\t"
                    "vst2.8 {d8-d9}, [%[out]]                                 \n\t"
                    : [in0] "=r" (lidx0),
                      [in1] "=r" (lidx1),
                      [in3] "=r" (lidx3),
                      [in4] "=r" (lidx4)
                    : [out] "r" (dst + x),
                      "0" (lidx0),
                      "1" (lidx1),
                      "2" (lidx3),
                      "3" (lidx4),
                      [in2] "r" (lane + x),
                      [in22] "r" (lane + x + 4*2),
                      [c4] "w" (vc4u16), [c6] "w" (vc6u16)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23"
                );
#else
                uint16x8x2_t vLane0 = vld2q_u16(lidx0);
                uint16x8x2_t vLane1 = vld2q_u16(lidx1);
                uint16x8x2_t vLane2 = vld2q_u16(lane + x);
                uint16x8x2_t vLane3 = vld2q_u16(lidx3);
                uint16x8x2_t vLane4 = vld2q_u16(lidx4);

                uint16x8_t vSum_0_4 = vaddq_u16(vLane0.val[0], vLane4.val[0]);
                uint16x8_t vSum_1_5 = vaddq_u16(vLane0.val[1], vLane4.val[1]);

                uint16x8_t vSum_4_8 = vaddq_u16(vLane1.val[0], vLane3.val[0]);
                uint16x8_t vSum_5_9 = vaddq_u16(vLane1.val[1], vLane3.val[1]);

                vSum_0_4 = vmlaq_u16(vSum_0_4, vSum_4_8, vc4u16);
                vSum_1_5 = vmlaq_u16(vSum_1_5, vSum_5_9, vc4u16);
                vSum_0_4 = vmlaq_u16(vSum_0_4, vLane2.val[0], vc6u16);
                vSum_1_5 = vmlaq_u16(vSum_1_5, vLane2.val[1], vc6u16);

                uint8x8x2_t vRes;
                vRes.val[0] = vrshrn_n_u16(vSum_0_4, 8);
                vRes.val[1] = vrshrn_n_u16(vSum_1_5, 8);
                vst2_u8(dst + x, vRes);
#endif
            }
            break;
        case 3:
            for (; x <= colsn - 8*3; x += 8*3)
            {
                internal::prefetch(lane + x);

                u16* lidx0 = lane + x - 2*3;
                u16* lidx1 = lane + x - 1*3;
                u16* lidx3 = lane + x + 1*3;
                u16* lidx4 = lane + x + 2*3;
#if defined(__GNUC__) && defined(__arm__)
                __asm__ __volatile__ (
                    "vld3.16 {d0, d2, d4}, [%[in0]]!                          \n\t"
                    "vld3.16 {d1, d3, d5}, [%[in0]]                           \n\t"
                    "vld3.16 {d8, d10, d12}, [%[in4]]!                        \n\t"
                    "vld3.16 {d9, d11, d13}, [%[in4]]                         \n\t"
                    "vadd.i16 q0, q4                                          \n\t"
                    "vadd.i16 q1, q5                                          \n\t"
                    "vadd.i16 q2, q6                                          \n\t"
                    "vld3.16 {d16, d18, d20}, [%[in1]]!                       \n\t"
                    "vld3.16 {d17, d19, d21}, [%[in1]]                        \n\t"
                    "vld3.16 {d8, d10, d12}, [%[in3]]!                        \n\t"
                    "vld3.16 {d9, d11, d13}, [%[in3]]                         \n\t"
                    "vadd.i16 q4, q8                                          \n\t"
                    "vadd.i16 q5, q9                                          \n\t"
                    "vadd.i16 q6, q10                                         \n\t"
                    "vld3.16 {d16, d18, d20}, [%[in2]]                        \n\t"
                    "vld3.16 {d17, d19, d21}, [%[in22]]                       \n\t"
                    "vmla.i16 q0, q4, %q[c4]                                  \n\t"
                    "vmla.i16 q1, q5, %q[c4]                                  \n\t"
                    "vmla.i16 q2, q6, %q[c4]                                  \n\t"
                    "vmla.i16 q0, q8, %q[c6]                                  \n\t"
                    "vmla.i16 q1, q9, %q[c6]                                  \n\t"
                    "vmla.i16 q2, q10, %q[c6]                                 \n\t"
                    "vrshrn.u16 d8, q0, #8                                    \n\t"
                    "vrshrn.u16 d9, q1, #8                                    \n\t"
                    "vrshrn.u16 d10, q2, #8                                   \n\t"
                    "vst3.8 {d8-d10}, [%[out]]                                \n\t"
                    : [in0] "=r" (lidx0),
                      [in1] "=r" (lidx1),
                      [in3] "=r" (lidx3),
                      [in4] "=r" (lidx4)
                    : [out] "r" (dst + x),
                      "0" (lidx0),
                      "1" (lidx1),
                      "2" (lidx3),
                      "3" (lidx4),
                      [in2] "r" (lane + x),
                      [in22] "r" (lane + x + 4*3),
                      [c4] "w" (vc4u16), [c6] "w" (vc6u16)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23"
                );
#else
                uint16x8x3_t vLane0 = vld3q_u16(lidx0);
                uint16x8x3_t vLane1 = vld3q_u16(lidx1);
                uint16x8x3_t vLane2 = vld3q_u16(lane + x);
                uint16x8x3_t vLane3 = vld3q_u16(lidx3);
                uint16x8x3_t vLane4 = vld3q_u16(lidx4);

                uint16x8_t vSum_0_4 = vaddq_u16(vLane0.val[0], vLane4.val[0]);
                uint16x8_t vSum_1_5 = vaddq_u16(vLane0.val[1], vLane4.val[1]);
                uint16x8_t vSum_2_6 = vaddq_u16(vLane0.val[2], vLane4.val[2]);

                uint16x8_t vSum_3_1 = vaddq_u16(vLane3.val[0], vLane1.val[0]);
                uint16x8_t vSum_4_2 = vaddq_u16(vLane3.val[1], vLane1.val[1]);
                uint16x8_t vSum_5_6 = vaddq_u16(vLane3.val[2], vLane1.val[2]);

                vSum_0_4 = vmlaq_u16(vSum_0_4, vSum_3_1, vc4u16);
                vSum_1_5 = vmlaq_u16(vSum_1_5, vSum_4_2, vc4u16);
                vSum_2_6 = vmlaq_u16(vSum_2_6, vSum_5_6, vc4u16);

                vSum_0_4 = vmlaq_u16(vSum_0_4, vLane2.val[0], vc6u16);
                vSum_1_5 = vmlaq_u16(vSum_1_5, vLane2.val[1], vc6u16);
                vSum_2_6 = vmlaq_u16(vSum_2_6, vLane2.val[2], vc6u16);

                uint8x8x3_t vRes;
                vRes.val[0] = vrshrn_n_u16(vSum_0_4, 8);
                vRes.val[1] = vrshrn_n_u16(vSum_1_5, 8);
                vRes.val[2] = vrshrn_n_u16(vSum_2_6, 8);

                vst3_u8(dst + x, vRes);
#endif
            }
            break;
        case 4:
            for (; x <= colsn - 8*4; x += 8*4)
            {
                internal::prefetch(lane + x);
                internal::prefetch(lane + x + 16);

                u16* lidx0 = lane + x - 2*4;
                u16* lidx1 = lane + x - 1*4;
                u16* lidx3 = lane + x + 1*4;
                u16* lidx4 = lane + x + 2*4;
#if defined(__GNUC__) && defined(__arm__)
                __asm__ __volatile__ (
                    "vld4.16 {d0, d2, d4, d6}, [%[in0]]!                      \n\t"
                    "vld4.16 {d1, d3, d5, d7}, [%[in0]]                       \n\t"
                    "vld4.16 {d8, d10, d12, d14}, [%[in4]]!                   \n\t"
                    "vld4.16 {d9, d11, d13, d15}, [%[in4]]                    \n\t"
                    "vadd.i16 q0, q4                                          \n\t"
                    "vadd.i16 q1, q5                                          \n\t"
                    "vadd.i16 q2, q6                                          \n\t"
                    "vadd.i16 q3, q7                                          \n\t"
                    "vld4.16 {d16, d18, d20, d22}, [%[in1]]!                  \n\t"
                    "vld4.16 {d17, d19, d21, d23}, [%[in1]]                   \n\t"
                    "vld4.16 {d8, d10, d12, d14}, [%[in3]]!                   \n\t"
                    "vld4.16 {d9, d11, d13, d15}, [%[in3]]                    \n\t"
                    "vadd.i16 q4, q8                                          \n\t"
                    "vadd.i16 q5, q9                                          \n\t"
                    "vadd.i16 q6, q10                                         \n\t"
                    "vadd.i16 q7, q11                                         \n\t"
                    "vld4.16 {d16, d18, d20, d22}, [%[in2],:256]              \n\t"
                    "vld4.16 {d17, d19, d21, d23}, [%[in22],:256]             \n\t"
                    "vmla.i16 q0, q4, %q[c4]                                  \n\t"
                    "vmla.i16 q1, q5, %q[c4]                                  \n\t"
                    "vmla.i16 q2, q6, %q[c4]                                  \n\t"
                    "vmla.i16 q3, q7, %q[c4]                                  \n\t"
                    "vmla.i16 q0, q8, %q[c6]                                  \n\t"
                    "vmla.i16 q1, q9, %q[c6]                                  \n\t"
                    "vmla.i16 q2, q10, %q[c6]                                 \n\t"
                    "vmla.i16 q3, q11, %q[c6]                                 \n\t"
                    "vrshrn.u16 d8, q0, #8                                    \n\t"
                    "vrshrn.u16 d9, q1, #8                                    \n\t"
                    "vrshrn.u16 d10, q2, #8                                   \n\t"
                    "vrshrn.u16 d11, q3, #8                                   \n\t"
                    "vst4.8 {d8-d11}, [%[out]]                                \n\t"
                    : [in0] "=r" (lidx0),
                      [in1] "=r" (lidx1),
                      [in3] "=r" (lidx3),
                      [in4] "=r" (lidx4)
                    : [out] "r" (dst + x),
                      "0" (lidx0),
                      "1" (lidx1),
                      "2" (lidx3),
                      "3" (lidx4),
                      [in2] "r" (lane + x),
                      [in22] "r" (lane + x + 4*4),
                      [c4] "w" (vc4u16), [c6] "w" (vc6u16)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23"
                );
#else
                uint16x8x4_t vLane0 = vld4q_u16(lidx0);
                uint16x8x4_t vLane2 = vld4q_u16(lidx4);
                uint16x8x4_t vLane4 = vld4q_u16(lidx1);
                uint16x8x4_t vLane6 = vld4q_u16(lidx3);
                uint16x8x4_t vLane8 = vld4q_u16(lane + x);

                uint16x8_t vSum_0_4  = vaddq_u16(vLane0.val[0], vLane2.val[0]);
                uint16x8_t vSum_1_5  = vaddq_u16(vLane0.val[1], vLane2.val[1]);
                uint16x8_t vSum_2_6  = vaddq_u16(vLane0.val[2], vLane2.val[2]);
                uint16x8_t vSum_3_7  = vaddq_u16(vLane0.val[3], vLane2.val[3]);

                uint16x8_t vSum_4_8  = vaddq_u16(vLane4.val[0], vLane6.val[0]);
                uint16x8_t vSum_5_9  = vaddq_u16(vLane4.val[1], vLane6.val[1]);
                uint16x8_t vSum_6_10 = vaddq_u16(vLane4.val[2], vLane6.val[2]);
                uint16x8_t vSum_7_11 = vaddq_u16(vLane4.val[3], vLane6.val[3]);

                vSum_0_4 = vmlaq_u16(vSum_0_4, vSum_4_8, vc4u16);
                vSum_1_5 = vmlaq_u16(vSum_1_5, vSum_5_9, vc4u16);
                vSum_2_6 = vmlaq_u16(vSum_2_6, vSum_6_10, vc4u16);
                vSum_3_7 = vmlaq_u16(vSum_3_7, vSum_7_11, vc4u16);

                vSum_0_4 = vmlaq_u16(vSum_0_4, vLane8.val[0], vc6u16);
                vSum_1_5 = vmlaq_u16(vSum_1_5, vLane8.val[1], vc6u16);
                vSum_2_6 = vmlaq_u16(vSum_2_6, vLane8.val[2], vc6u16);
                vSum_3_7 = vmlaq_u16(vSum_3_7, vLane8.val[3], vc6u16);

                uint8x8x4_t vRes;
                vRes.val[0] = vrshrn_n_u16(vSum_0_4, 8);
                vRes.val[1] = vrshrn_n_u16(vSum_1_5, 8);
                vRes.val[2] = vrshrn_n_u16(vSum_2_6, 8);
                vRes.val[3] = vrshrn_n_u16(vSum_3_7, 8);

                vst4_u8(dst + x, vRes);
#endif
            }
            break;
        }
        for (s32 h = 0; h < cn; ++h)
        {
            u16* ln = lane + h;
            u8* dt = dst + h;
            for (size_t k = x; k < colsn; k += cn)
            {
                dt[k] = (u8)((ln[k-2*cn] + ln[k+2*cn]
                               + u16(4) * (ln[k-cn] + ln[k+cn])
                               + u16(6) * ln[k] + (1 << 7)) >> 8);
            }
        }
    }
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
    (void)borderMargin;
#endif
}

void gaussianBlur5x5(const Size2D &size, s32 cn,
                     const u16 * srcBase, ptrdiff_t srcStride,
                     u16 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE borderType, u16 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isGaussianBlur5x5Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
    size_t colsn = size.width * cn;

    std::vector<u16> _tmp;
    u16 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 4*cn, borderValue);
        tmp = &_tmp[cn << 1];
    }

    ptrdiff_t idx_l1 = internal::borderInterpolate(-1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_l2 = internal::borderInterpolate(-2, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r1 = internal::borderInterpolate(size.width + 0, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r2 = internal::borderInterpolate(size.width + 1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;

    //1-line buffer
    std::vector<u32> _buf(cn * (size.width + 4) + 32 / sizeof(u32));
    u32* lane = internal::alignPtr(&_buf[cn << 1], 32);

    if (borderType == BORDER_MODE_CONSTANT)
        for (s32 k = 0; k < cn; ++k)
        {
            lane[-cn+k] = borderValue;
            lane[-cn-cn+k] = borderValue;
            lane[colsn+k] = borderValue;
            lane[colsn+cn+k] = borderValue;
        }

    uint16x4_t vc6u16 = vmov_n_u16(6);
    uint32x4_t vc6u32 = vmovq_n_u32(6);
    uint32x4_t vc4u32 = vmovq_n_u32(4);

    for (size_t i = 0; i < size.height; ++i)
    {
        u16* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        ptrdiff_t idx_rm2 = internal::borderInterpolate(i - 2, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp1 = internal::borderInterpolate(i + 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp2 = internal::borderInterpolate(i + 2, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const u16* ln0 = idx_rm2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm2) : tmp;
        const u16* ln1 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const u16* ln2 = internal::getRowPtr(srcBase, srcStride, i);
        const u16* ln3 = idx_rp1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp1) : tmp;
        const u16* ln4 = idx_rp2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp2) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, x % 5 - 2));
            uint16x4_t v0 = vld1_u16(ln0+x);
            uint16x4_t v1 = vld1_u16(ln1+x);
            uint16x4_t v2 = vld1_u16(ln2+x);
            uint16x4_t v3 = vld1_u16(ln3+x);
            uint16x4_t v4 = vld1_u16(ln4+x);

            uint32x4_t v = vaddl_u16(v0, v4);
            uint32x4_t v13 = vaddl_u16(v1, v3);

            v = vmlal_u16(v, v2, vc6u16);
            v = vmlaq_u32(v, v13, vc4u32);

            vst1q_u32(lane + x, v);
        }
        for (; x < colsn; ++x)
            lane[x] = ln0[x] + ln4[x] + 4*(ln1[x] + ln3[x]) + 6*ln2[x];

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lane[-cn+k] = lane[idx_l1 + k];
                lane[-cn-cn+k] = lane[idx_l2 + k];

                lane[colsn+k] = lane[idx_r1 + k];
                lane[colsn+cn+k] = lane[idx_r2 + k];
            }

        //horizontal convolution
        x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(lane + x);

            uint32x4_t lane0 = vld1q_u32(lane + x - 2);
            uint32x4_t lane4 = vld1q_u32(lane + x + 2);
            uint32x4_t lane1 = vld1q_u32(lane + x - 1);
            uint32x4_t lane3 = vld1q_u32(lane + x + 1);
            uint32x4_t lane2 = vld1q_u32(lane + x + 0);

            uint32x4_t ln04 = vaddq_u32(lane0, lane4);
            uint32x4_t ln13 = vaddq_u32(lane1, lane3);

            uint32x4_t ln042 = vmlaq_u32(ln04, lane2, vc6u32);
            uint32x4_t lsw = vmlaq_u32(ln042, ln13, vc4u32);

            uint16x4_t ls = vrshrn_n_u32(lsw, 8);

            vst1_u16(dst + x, ls);
        }
        for (s32 h = 0; h < cn; ++h)
        {
            u32* ln = lane + h;
            u16* dt = dst + h;
            for (size_t k = x; k < colsn; k += cn)
            {
                dt[k] = (u16)((ln[k-2*cn] + ln[k+2*cn] + 4*(ln[k-cn] + ln[k+cn]) + 6*ln[k] + (1<<7))>>8);
            }
        }
    }
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
    (void)borderMargin;
#endif
}

void gaussianBlur5x5(const Size2D &size, s32 cn,
                     const s16 * srcBase, ptrdiff_t srcStride,
                     s16 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE borderType, s16 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isGaussianBlur5x5Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
    size_t colsn = size.width * cn;

    std::vector<s16> _tmp;
    s16 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 4*cn, borderValue);
        tmp = &_tmp[cn << 1];
    }

    ptrdiff_t idx_l1 = internal::borderInterpolate(-1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_l2 = internal::borderInterpolate(-2, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r1 = internal::borderInterpolate(size.width + 0, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r2 = internal::borderInterpolate(size.width + 1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;

    //1-line buffer
    std::vector<s32> _buf(cn * (size.width + 4) + 32 / sizeof(s32));
    s32* lane = internal::alignPtr(&_buf[cn << 1], 32);

    if (borderType == BORDER_MODE_CONSTANT)
        for (s32 k = 0; k < cn; ++k)
        {
            lane[-cn+k] = borderValue;
            lane[-cn-cn+k] = borderValue;
            lane[colsn+k] = borderValue;
            lane[colsn+cn+k] = borderValue;
        }

    int16x4_t vc6s16 = vmov_n_s16(6);
    int32x4_t vc6s32 = vmovq_n_s32(6);
    int32x4_t vc4s32 = vmovq_n_s32(4);

    for (size_t i = 0; i < size.height; ++i)
    {
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        ptrdiff_t idx_rm2 = internal::borderInterpolate(i - 2, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp1 = internal::borderInterpolate(i + 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp2 = internal::borderInterpolate(i + 2, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const s16* ln0 = idx_rm2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm2) : tmp;
        const s16* ln1 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const s16* ln2 = internal::getRowPtr(srcBase, srcStride, i);
        const s16* ln3 = idx_rp1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp1) : tmp;
        const s16* ln4 = idx_rp2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp2) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, x % 5 - 2));
            int16x4_t v0 = vld1_s16(ln0+x);
            int16x4_t v1 = vld1_s16(ln1+x);
            int16x4_t v2 = vld1_s16(ln2+x);
            int16x4_t v3 = vld1_s16(ln3+x);
            int16x4_t v4 = vld1_s16(ln4+x);

            int32x4_t v = vaddl_s16(v0, v4);
            int32x4_t v13 = vaddl_s16(v1, v3);

            v = vmlal_s16(v, v2, vc6s16);
            v = vmlaq_s32(v, v13, vc4s32);

            vst1q_s32(lane + x, v);
        }
        for (; x < colsn; ++x)
            lane[x] = ln0[x] + ln4[x] + 4*(ln1[x] + ln3[x]) + 6*ln2[x];

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lane[-cn+k] = lane[idx_l1 + k];
                lane[-cn-cn+k] = lane[idx_l2 + k];

                lane[colsn+k] = lane[idx_r1 + k];
                lane[colsn+cn+k] = lane[idx_r2 + k];
            }

        //horizontal convolution
        x = 0;
       switch(cn)
        {
        case 1:
        case 2:
        case 3:
            for (; x <= colsn - 4; x += 4)
            {
                internal::prefetch(lane + x);

                int32x4_t lane0 = vld1q_s32(lane + x - 2);
                int32x4_t lane4 = vld1q_s32(lane + x + 2);
                int32x4_t lane1 = vld1q_s32(lane + x - 1);
                int32x4_t lane3 = vld1q_s32(lane + x + 1);
                int32x4_t lane2 = vld1q_s32(lane + x + 0);

                int32x4_t ln04 = vaddq_s32(lane0, lane4);
                int32x4_t ln13 = vaddq_s32(lane1, lane3);

                int32x4_t ln042 = vmlaq_s32(ln04, lane2, vc6s32);
                int32x4_t lsw = vmlaq_s32(ln042, ln13, vc4s32);

                int16x4_t ls = vrshrn_n_s32(lsw, 8);

                vst1_s16(dst + x, ls);
           }
            break;
        case 4:
/*            for (; x <= colsn - 4*4; x += 4*4)
            {
                internal::prefetch(lane + x);
                internal::prefetch(lane + x + 16);

                ptrdiff_t* lidx0 = lane + x - 2*4;
                ptrdiff_t* lidx1 = lane + x - 1*4;
                ptrdiff_t* lidx3 = lane + x + 1*4;
                ptrdiff_t* lidx4 = lane + x + 2*4;

                __asm__ __volatile__ (
                    "vld4.32 {d0, d2, d4, d6}, [%[in0]]!                      \n\t"
                    "vld4.32 {d1, d3, d5, d7}, [%[in0]]                       \n\t"
                    "vld4.32 {d8, d10, d12, d14}, [%[in4]]!                   \n\t"
                    "vld4.32 {d9, d11, d13, d15}, [%[in4]]                    \n\t"
                    "vadd.i32 q0, q4                                          \n\t"
                    "vadd.i32 q1, q5                                          \n\t"
                    "vadd.i32 q2, q6                                          \n\t"
                    "vadd.i32 q3, q7                                          \n\t"
                    "vld4.32 {d16, d18, d20, d22}, [%[in1]]!                  \n\t"
                    "vld4.32 {d17, d19, d21, d23}, [%[in1]]                   \n\t"
                    "vld4.32 {d8, d10, d12, d14}, [%[in3]]!                   \n\t"
                    "vld4.32 {d9, d11, d13, d15}, [%[in3]]                    \n\t"
                    "vadd.i32 q4, q8                                          \n\t"
                    "vadd.i32 q5, q9                                          \n\t"
                    "vadd.i32 q6, q10                                         \n\t"
                    "vadd.i32 q7, q11                                         \n\t"
                    "vld4.32 {d16, d18, d20, d22}, [%[in2],:256]              \n\t"
                    "vld4.32 {d17, d19, d21, d23}, [%[in22],:256]             \n\t"
                    "vmla.i32 q0, q4, %q[c4]                                  \n\t"
                    "vmla.i32 q1, q5, %q[c4]                                  \n\t"
                    "vmla.i32 q2, q6, %q[c4]                                  \n\t"
                    "vmla.i32 q3, q7, %q[c4]                                  \n\t"
                    "vmla.i32 q0, q8, %q[c6]                                  \n\t"
                    "vmla.i32 q1, q9, %q[c6]                                  \n\t"
                    "vmla.i32 q2, q10, %q[c6]                                 \n\t"
                    "vmla.i32 q3, q11, %q[c6]                                 \n\t"
                    "vrshrn.i32 d8, q0, #8                                    \n\t"
                    "vrshrn.i32 d9, q1, #8                                    \n\t"
                    "vrshrn.i32 d10, q2, #8                                   \n\t"
                    "vrshrn.i32 d11, q3, #8                                   \n\t"
                   "vst4.16 {d8-d11}, [%[out]]                                \n\t"
                    : [in0] "=r" (lidx0),
                      [in1] "=r" (lidx1),
                      [in3] "=r" (lidx3),
                      [in4] "=r" (lidx4)
                    : [out] "r" (dst + x),
                      "0" (lidx0),
                      "1" (lidx1),
                      "2" (lidx3),
                      "3" (lidx4),
                      [in2] "r" (lane + x),
                      [in22] "r" (lane + x + 4*2),
                      [c4] "w" (vc4s32), [c6] "w" (vc6s32)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23"
                );
*/
            for (; x <= colsn - 4; x += 4)
            {
                internal::prefetch(lane + x);

                int32x4_t lane0 = vld1q_s32(lane + x - 2);
                int32x4_t lane4 = vld1q_s32(lane + x + 2);
                int32x4_t lane1 = vld1q_s32(lane + x - 1);
                int32x4_t lane3 = vld1q_s32(lane + x + 1);
                int32x4_t lane2 = vld1q_s32(lane + x + 0);

                int32x4_t ln04 = vaddq_s32(lane0, lane4);
                int32x4_t ln13 = vaddq_s32(lane1, lane3);

                int32x4_t ln042 = vmlaq_s32(ln04, lane2, vc6s32);
                int32x4_t lsw = vmlaq_s32(ln042, ln13, vc4s32);

                int16x4_t ls = vrshrn_n_s32(lsw, 8);

                vst1_s16(dst + x, ls);
            }
            break;
        }
        for (s32 h = 0; h < cn; ++h)
        {
            s32* ln = lane + h;
            s16* dt = dst + h;
            for (size_t k = x; k < colsn; k += cn)
            {
                dt[k] = (s16)((ln[k-2*cn] + ln[k+2*cn] + 4*(ln[k-cn] + ln[k+cn]) + 6*ln[k] + (1<<7))>>8);
            }
        }
    }
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
    (void)borderMargin;
#endif
}

void gaussianBlur5x5(const Size2D &size, s32 cn,
                     const s32 * srcBase, ptrdiff_t srcStride,
                     s32 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE borderType, s32 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isGaussianBlur5x5Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
    size_t colsn = size.width * cn;

    std::vector<s32> _tmp;
    s32 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 4*cn, borderValue);
        tmp = &_tmp[cn << 1];
    }

    ptrdiff_t idx_l1 = internal::borderInterpolate(-1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_l2 = internal::borderInterpolate(-2, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r1 = internal::borderInterpolate(size.width + 0, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r2 = internal::borderInterpolate(size.width + 1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;

    //1-line buffer
    std::vector<s32> _buf(cn * (size.width + 4) + 32 / sizeof(s32));
    s32* lane = internal::alignPtr(&_buf[cn << 1], 32);

    if (borderType == BORDER_MODE_CONSTANT)
        for (s32 k = 0; k < cn; ++k)
        {
            lane[-cn+k] = borderValue;
            lane[-cn-cn+k] = borderValue;
            lane[colsn+k] = borderValue;
            lane[colsn+cn+k] = borderValue;
        }

    int32x4_t vc6s32 = vmovq_n_s32(6);
    int32x4_t vc4s32 = vmovq_n_s32(4);

    for (size_t i = 0; i < size.height; ++i)
    {
        s32* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        ptrdiff_t idx_rm2 = internal::borderInterpolate(i - 2, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp1 = internal::borderInterpolate(i + 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp2 = internal::borderInterpolate(i + 2, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const s32* ln0 = idx_rm2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm2) : tmp;
        const s32* ln1 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const s32* ln2 = internal::getRowPtr(srcBase, srcStride, i);
        const s32* ln3 = idx_rp1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp1) : tmp;
        const s32* ln4 = idx_rp2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp2) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, x % 5 - 2));
            int32x4_t v0 = vld1q_s32(ln0+x);
            int32x4_t v1 = vld1q_s32(ln1+x);
            int32x4_t v2 = vld1q_s32(ln2+x);
            int32x4_t v3 = vld1q_s32(ln3+x);
            int32x4_t v4 = vld1q_s32(ln4+x);

            int32x4_t v = vaddq_s32(v0, v4);
            int32x4_t v13 = vaddq_s32(v1, v3);

            v = vmlaq_s32(v, v2, vc6s32);
            v = vmlaq_s32(v, v13, vc4s32);

            vst1q_s32(lane + x, v);
        }
        for (; x < colsn; ++x)
            lane[x] = ln0[x] + ln4[x] + 4*(ln1[x] + ln3[x]) + 6*ln2[x];

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lane[-cn+k] = lane[idx_l1 + k];
                lane[-cn-cn+k] = lane[idx_l2 + k];

                lane[colsn+k] = lane[idx_r1 + k];
                lane[colsn+cn+k] = lane[idx_r2 + k];
            }

        //horizontal convolution
        x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(lane + x);

            int32x4_t lane0 = vld1q_s32(lane + x - 2);
            int32x4_t lane4 = vld1q_s32(lane + x + 2);
            int32x4_t lane1 = vld1q_s32(lane + x - 1);
            int32x4_t lane3 = vld1q_s32(lane + x + 1);
            int32x4_t lane2 = vld1q_s32(lane + x + 0);

            int32x4_t ln04 = vaddq_s32(lane0, lane4);
            int32x4_t ln13 = vaddq_s32(lane1, lane3);

            int32x4_t ln042 = vmlaq_s32(ln04, lane2, vc6s32);
            int32x4_t lsw = vmlaq_s32(ln042, ln13, vc4s32);

            vst1q_s32(dst + x, lsw);
        }
        for (s32 h = 0; h < cn; ++h)
        {
            s32* ln = lane + h;
            s32* dt = dst + h;
            for (size_t k = x; k < colsn; k += cn)
            {
                dt[k] = ln[k-2*cn] + ln[k+2*cn] + 4*(ln[k-cn] + ln[k+cn]) + 6*ln[k];
            }
        }
    }
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
    (void)borderMargin;
#endif
}

} // namespace CAROTENE_NS
