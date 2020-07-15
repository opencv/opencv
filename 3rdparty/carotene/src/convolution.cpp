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
#include "saturate_cast.hpp"

namespace CAROTENE_NS {

bool isConvolutionSupported(const Size2D &size, const Size2D &ksize,
                            BORDER_MODE border)
{
    return isSupportedConfiguration() && size.width >= 8 &&
        (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REPLICATE) &&
        (ksize.width == 3) && (ksize.height == 3);
}

#ifdef CAROTENE_NEON

namespace {

template <int shift>
int32x4_t vshrq_s32(int32x4_t value)
{
    return vshrq_n_s32(value, shift);
}

template <>
int32x4_t vshrq_s32<0>(int32x4_t value)
{
    return value;
}

} // namespace

typedef int32x4_t (* vshrq_s32_func)(int32x4_t value);

#endif

void convolution(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE border, u8 borderValue,
                 const Size2D & ksize, s16 * kernelBase, u32 scale)
{
    internal::assertSupportedConfiguration(isConvolutionSupported(size, ksize, border));
#ifdef CAROTENE_NEON
    const uint8x8_t v_zero_u8 = vdup_n_u8(0);
    const uint8x8_t v_border = vdup_n_u8(borderValue);
    const int32x4_t v_zero_s32 = vdupq_n_s32(0);

    uint8x8_t tprev[3] = { v_zero_u8, v_zero_u8, v_zero_u8 },
              tcurr[3] = { v_zero_u8, v_zero_u8, v_zero_u8 },
              tnext[3] = { v_zero_u8, v_zero_u8, v_zero_u8 };
    uint8x8_t t0 = v_zero_u8, t1 = v_zero_u8, t2 = v_zero_u8;

    ptrdiff_t width = (ptrdiff_t)size.width, height = (ptrdiff_t)size.height;
    static const vshrq_s32_func vshrq_s32_a[33] =
    {
        vshrq_s32<0>,
        vshrq_s32<1>,
        vshrq_s32<2>,
        vshrq_s32<3>,
        vshrq_s32<4>,
        vshrq_s32<5>,
        vshrq_s32<6>,
        vshrq_s32<7>,
        vshrq_s32<8>,
        vshrq_s32<9>,
        vshrq_s32<10>,
        vshrq_s32<11>,
        vshrq_s32<12>,
        vshrq_s32<13>,
        vshrq_s32<14>,
        vshrq_s32<15>,
        vshrq_s32<16>,
        vshrq_s32<17>,
        vshrq_s32<18>,
        vshrq_s32<19>,
        vshrq_s32<20>,
        vshrq_s32<21>,
        vshrq_s32<22>,
        vshrq_s32<23>,
        vshrq_s32<24>,
        vshrq_s32<25>,
        vshrq_s32<26>,
        vshrq_s32<27>,
        vshrq_s32<28>,
        vshrq_s32<29>,
        vshrq_s32<30>,
        vshrq_s32<31>,
        vshrq_s32<32>
    };
    vshrq_s32_func vshrq_s32_p = vshrq_s32_a[scale];

    for (ptrdiff_t y = 0; y < height; ++y)
    {
        const u8 * srow0 = y == 0 && border == BORDER_MODE_CONSTANT ? NULL : internal::getRowPtr(srcBase, srcStride, std::max<ptrdiff_t>(y - 1, 0));
        const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, y);
        const u8 * srow2 = y + 1 == height && border == BORDER_MODE_CONSTANT ? NULL : internal::getRowPtr(srcBase, srcStride, std::min(y + 1, height - 1));
        u8 * drow = internal::getRowPtr(dstBase, dstStride, y);

        u8 prevx[3] = { 0, 0, 0 },
           currx[3] = { 0, 0, 0 },
           nextx[3] = { 0, 0, 0 };
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
                    prevx[0] = prevx[1] = prevx[2] = borderValue;
                else
                {
                    prevx[0] = srow0 ? srow0[x4] : borderValue;
                    prevx[1] =         srow1[x4]              ;
                    prevx[2] = srow2 ? srow2[x4] : borderValue;
                }

                currx[0] = srow0 ? srow0[x3] : borderValue;
                currx[1] =         srow1[x3]              ;
                currx[2] = srow2 ? srow2[x3] : borderValue;
            }

            // make shift
            if (x)
            {
                tprev[0] = tcurr[0];
                tcurr[0] = tnext[0];

                tprev[1] = tcurr[1];
                tcurr[1] = tnext[1];

                tprev[2] = tcurr[2];
                tcurr[2] = tnext[2];
            }

            tnext[0] = x0;
            tnext[1] = x1;
            tnext[2] = x2;

            // make extrapolation for the first elements
            if (!x)
            {
                // make border
                if (border == BORDER_MODE_CONSTANT)
                    tcurr[0] = tcurr[1] = tcurr[2] = v_border;
                else if (border == BORDER_MODE_REPLICATE)
                {
                    tcurr[0] = vdup_n_u8(vget_lane_u8(tnext[0], 0));
                    tcurr[1] = vdup_n_u8(vget_lane_u8(tnext[1], 0));
                    tcurr[2] = vdup_n_u8(vget_lane_u8(tnext[2], 0));
                }

                continue;
            }

            int32x4_t v_dst0 = v_zero_s32, v_dst1 = v_zero_s32;

            {
                // combine 3 "shifted" vectors
                t0 = vext_u8(tprev[0], tcurr[0], 7);
                t1 = tcurr[0];
                t2 = vext_u8(tcurr[0], tnext[0], 1);

                int16x8_t t0_16s = vreinterpretq_s16_u16(vmovl_u8(t0));
                int16x8_t t1_16s = vreinterpretq_s16_u16(vmovl_u8(t1));
                int16x8_t t2_16s = vreinterpretq_s16_u16(vmovl_u8(t2));

                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t0_16s), kernelBase[8]);
                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t1_16s), kernelBase[7]);
                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t2_16s), kernelBase[6]);

                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t0_16s), kernelBase[8]);
                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t1_16s), kernelBase[7]);
                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t2_16s), kernelBase[6]);
            }

            {
                // combine 3 "shifted" vectors
                t0 = vext_u8(tprev[1], tcurr[1], 7);
                t1 = tcurr[1];
                t2 = vext_u8(tcurr[1], tnext[1], 1);

                int16x8_t t0_16s = vreinterpretq_s16_u16(vmovl_u8(t0));
                int16x8_t t1_16s = vreinterpretq_s16_u16(vmovl_u8(t1));
                int16x8_t t2_16s = vreinterpretq_s16_u16(vmovl_u8(t2));

                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t0_16s), kernelBase[5]);
                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t1_16s), kernelBase[4]);
                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t2_16s), kernelBase[3]);

                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t0_16s), kernelBase[5]);
                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t1_16s), kernelBase[4]);
                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t2_16s), kernelBase[3]);
            }

            {
                // combine 3 "shifted" vectors
                t0 = vext_u8(tprev[2], tcurr[2], 7);
                t1 = tcurr[2];
                t2 = vext_u8(tcurr[2], tnext[2], 1);

                int16x8_t t0_16s = vreinterpretq_s16_u16(vmovl_u8(t0));
                int16x8_t t1_16s = vreinterpretq_s16_u16(vmovl_u8(t1));
                int16x8_t t2_16s = vreinterpretq_s16_u16(vmovl_u8(t2));

                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t0_16s), kernelBase[2]);
                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t1_16s), kernelBase[1]);
                v_dst0 = vmlal_n_s16(v_dst0, vget_low_s16(t2_16s), kernelBase[0]);

                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t0_16s), kernelBase[2]);
                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t1_16s), kernelBase[1]);
                v_dst1 = vmlal_n_s16(v_dst1, vget_high_s16(t2_16s), kernelBase[0]);
            }


            // make scale
            v_dst0 = vshrq_s32_p(v_dst0);
            v_dst1 = vshrq_s32_p(v_dst1);

            // and add them
            vst1_u8(drow + x - 8, vqmovn_u16(vcombine_u16(vqmovun_s32(v_dst0),
                                                          vqmovun_s32(v_dst1))));
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
                {
                    nextx[0] = borderValue;
                    nextx[1] = borderValue;
                    nextx[2] = borderValue;
                }
                else if (border == BORDER_MODE_REPLICATE)
                {
                    nextx[0] = srow0[x];
                    nextx[1] = srow1[x];
                    nextx[2] = srow2[x];
                }
            }
            else
            {
                nextx[0] = srow0 ? srow0[x + 1] : borderValue;
                nextx[1] =         srow1[x + 1]              ;
                nextx[2] = srow2 ? srow2[x + 1] : borderValue;
            }

            s32 val = 0;
            for (s32 _y = 0; _y < 3; ++_y)
                val += prevx[_y] * kernelBase[(2 - _y) * 3 + 2] +
                       currx[_y] * kernelBase[(2 - _y) * 3 + 1] +
                       nextx[_y] * kernelBase[(2 - _y) * 3 + 0];

            drow[x] = internal::saturate_cast<u8>(val >> scale);

            // make shift
            prevx[0] = currx[0];
            currx[0] = nextx[0];

            prevx[1] = currx[1];
            currx[1] = nextx[1];

            prevx[2] = currx[2];
            currx[2] = nextx[2];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)border;
    (void)borderValue;
    (void)ksize;
    (void)kernelBase;
    (void)scale;
#endif
}

} // namespace CAROTENE_NS
