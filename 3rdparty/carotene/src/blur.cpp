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
#include "saturate_cast.hpp"

namespace CAROTENE_NS {

bool isBlur3x3Supported(const Size2D &size, BORDER_MODE border)
{
    return isSupportedConfiguration() && size.width >= 8 &&
        (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REPLICATE);
}

void blur3x3(const Size2D &size,
             const u8 * srcBase, ptrdiff_t srcStride,
             u8 * dstBase, ptrdiff_t dstStride,
             BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isBlur3x3Supported(size, border));
#ifdef CAROTENE_NEON
    const int16x8_t v_scale = vmovq_n_s16(3640);
    const uint16x8_t v_border_x3 = vdupq_n_u16(borderValue * 3);
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
                    prevx = (srow2 ? srow2[x4] : borderValue) + srow1[x4] + (srow0 ? srow0[x4] : borderValue);

                currx = (srow2 ? srow2[x3] : borderValue) + srow1[x3] + (srow0 ? srow0[x3] : borderValue);
            }

            // make shift
            if (x)
            {
                tprev = tcurr;
                tcurr = tnext;
            }

            // and calculate next value
            tnext = vaddw_u8(vaddl_u8(x0, x1), x2);

            // make extrapolation for the first elements
            if (!x)
            {
                // make border
                if (border == BORDER_MODE_CONSTANT)
                    tcurr = v_border_x3;
                else if (border == BORDER_MODE_REPLICATE)
                    tcurr = vdupq_n_u16(vgetq_lane_u16(tnext, 0));

                continue;
            }

            // combine 3 "shifted" vectors
            t0 = vextq_u16(tprev, tcurr, 7);
            t1 = tcurr;
            t2 = vextq_u16(tcurr, tnext, 1);

            // and add them
            t0 = vqaddq_u16(t0, vqaddq_u16(t1, t2));

            int16x8_t tt0 = vqrdmulhq_s16(vreinterpretq_s16_u16(t0), v_scale);
            uint8x8_t it0 = vmovn_u16(vreinterpretq_u16_s16(tt0));
            vst1_u8(drow + x - 8, it0);
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
                    nextx = borderValue * 3;
                else if (border == BORDER_MODE_REPLICATE)
                    nextx = srow2[x] + srow1[x] + srow0[x];
            }
            else
                nextx = (srow2 ? srow2[x + 1] : borderValue) +
                                 srow1[x + 1] +
                        (srow0 ? srow0[x + 1] : borderValue);

            f32 val = (prevx + currx + nextx) * (1 / 9.f) + 0.5f;
            drow[x] = internal::saturate_cast<u8>((s32)val);

            // make shift
            prevx = currx;
            currx = nextx;
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
#endif
}

bool isBlurU8Supported(const Size2D &size, s32 cn, BORDER_MODE border)
{
    return isSupportedConfiguration() &&
           cn > 0 && cn <= 4 &&
           size.width*cn >= 8 && size.height >= 2 &&
           (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REFLECT101 ||
            border == BORDER_MODE_REFLECT ||
            border == BORDER_MODE_REPLICATE);
}

void blur3x3(const Size2D &size, s32 cn,
             const u8 * srcBase, ptrdiff_t srcStride,
             u8 * dstBase, ptrdiff_t dstStride,
             BORDER_MODE borderType, u8 borderValue)
{
    internal::assertSupportedConfiguration(isBlurU8Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
//#define FLOAT_VARIANT_1_9
#ifdef FLOAT_VARIANT_1_9
    float32x4_t v1_9 = vdupq_n_f32 (1.0/9.0);
    float32x4_t v0_5 = vdupq_n_f32 (.5);
#else
    const int16x8_t vScale = vmovq_n_s16(3640);
#endif

    size_t colsn = size.width*cn;

    std::vector<u8> _tmp;
    u8 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 2*cn, borderValue);
        tmp = &_tmp[cn];
    }

    uint16x8_t tprev = vdupq_n_u16(0x0);
    uint16x8_t tcurr = tprev;
    uint16x8_t tnext = tprev;
    uint16x8_t t0, t1, t2;
    if(cn == 1)
    {
        for( size_t y = 0; y < size.height; y++ )
        {
            const u8* srow0;
            const u8* srow1 = internal::getRowPtr(srcBase, srcStride, y);
            const u8* srow2;
            u8* drow = internal::getRowPtr(dstBase, dstStride, y);
            if (borderType == BORDER_MODE_REFLECT101) {
                srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 1);
                srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-2);
            } else  if (borderType == BORDER_MODE_CONSTANT) {
                srow0 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
                srow2 =  y < size.height-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
            } else { // BORDER_MODE_REFLECT || BORDER_MODE_REPLICATE
                srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
                srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-1);
            }

            // do vertical convolution
            size_t x = 0;
            const size_t bcols = y + 2 < size.height ? colsn : (colsn - 8);
            for( ; x <= bcols; x += 8 )
            {
                internal::prefetch(srow0 + x);
                internal::prefetch(srow1 + x);
                internal::prefetch(srow2 + x);

                uint8x8_t x0 = vld1_u8(srow0 + x);
                uint8x8_t x1 = vld1_u8(srow1 + x);
                uint8x8_t x2 = vld1_u8(srow2 + x);

                tprev = tcurr;
                tcurr = tnext;
                tnext = vaddw_u8(vaddl_u8(x0, x1), x2);

                if(!x) {
                    tcurr = tnext;

                    // make border
                        if (borderType == BORDER_MODE_CONSTANT)
                        {
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                        }
                        else if (borderType == BORDER_MODE_REFLECT101)
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 7);
                        }
                        else // borderType == BORDER_MODE_REFLECT || borderType == BORDER_MODE_REPLICATE
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 7);
                        }
                    continue;
                }

                t0 = vextq_u16(tprev, tcurr, 7);
                t1 = tcurr;
                t2 = vextq_u16(tcurr, tnext, 1);

                t0 = vqaddq_u16(t0, vqaddq_u16(t1, t2));

#ifdef FLOAT_VARIANT_1_9
                uint32x4_t tres1 = vmovl_u16(vget_low_u16(t0));
                uint32x4_t tres2 = vmovl_u16(vget_high_u16(t0));
                float32x4_t vf1 = vmulq_f32(v1_9, vcvtq_f32_u32(tres1));
                float32x4_t vf2 = vmulq_f32(v1_9, vcvtq_f32_u32(tres2));
                tres1 = vcvtq_u32_f32(vaddq_f32(vf1, v0_5));
                tres2 = vcvtq_u32_f32(vaddq_f32(vf2, v0_5));
                t0 = vcombine_u16(vmovn_u32(tres1),vmovn_u32(tres2));
                vst1_u8(drow + x - 8, vmovn_u16(t0));
#else
                int16x8_t tt0 = vqrdmulhq_s16(vreinterpretq_s16_u16(t0), vScale);
                uint8x8_t it0 = vmovn_u16(vreinterpretq_u16_s16(tt0));
                vst1_u8(drow + x - 8, it0);
#endif
            }

            x -= 8;
            if(x == colsn){
                x--;
            }
            s16 prevx, rowx, nextx;
            prevx = srow2[x-1] + srow1[x-1] + srow0[x-1];
            rowx = srow2[x] + srow1[x] + srow0[x];
            for( ; x < colsn; x++ )
            {
                if(x+1 >= colsn) {
                    // make border
                    if (borderType == BORDER_MODE_CONSTANT)
                    {
                        nextx = borderValue;
                    } else if (borderType == BORDER_MODE_REFLECT101)
                    {
                        nextx = srow2[x-1] + srow1[x-1] + srow0[x-1];
                    } else {
                        nextx = srow2[x] + srow1[x] + srow0[x];
                    }
                } else {
                    nextx = srow2[x+1] + srow1[x+1] + srow0[x+1];
                }
                *(drow+x) = internal::saturate_cast<u8>((prevx + rowx + nextx)*(1/9.));
                prevx = rowx;
                rowx = nextx;
            }
        }
    }
    else
    {
        for( size_t y = 0; y < size.height; y++ )
        {
            const u8* srow0;
            const u8* srow1 = internal::getRowPtr(srcBase, srcStride, y);
            const u8* srow2;
            u8* drow = internal::getRowPtr(dstBase, dstStride, y);
            if (borderType == BORDER_MODE_REFLECT101) {
                srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 1);
                srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-2);
            } else  if (borderType == BORDER_MODE_CONSTANT) {
                srow0 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
                srow2 =  y < size.height-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
            } else { // BORDER_MODE_REFLECT || BORDER_MODE_REPLICATE
                srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
                srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-1);
            }

            // do vertical convolution
            size_t x = 0;
            const size_t bcols = y + 2 < size.height ? colsn : (colsn - 8);
            for( ; x <= bcols; x += 8 )
            {
                internal::prefetch(srow0 + x);
                internal::prefetch(srow1 + x);
                internal::prefetch(srow2 + x);

                uint8x8_t x0 = vld1_u8(srow0 + x);
                uint8x8_t x1 = vld1_u8(srow1 + x);
                uint8x8_t x2 = vld1_u8(srow2 + x);

                tprev = tcurr;
                tcurr = tnext;
                tnext = vaddw_u8(vaddl_u8(x0, x1), x2);

                if(!x) {
                    tcurr = tnext;

                    // make border
                    switch(cn)
                    {
                    case 2:
                        if (borderType == BORDER_MODE_CONSTANT)
                        {
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                        }
                        else if (borderType == BORDER_MODE_REFLECT101)
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tcurr, 6);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tcurr, 6);
                        }
                        else
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 6);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 7);
                        }
                        break;
                    case 3:
                        if (borderType == BORDER_MODE_CONSTANT)
                        {
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 5);
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                        }
                        else if (borderType == BORDER_MODE_REFLECT101)
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 4),tcurr, 6);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 5),tcurr, 7);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tcurr, 5);
                        }
                        else
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 5);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 6);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tcurr, 7);
                        }
                        break;
                    case 4:
                        if (borderType == BORDER_MODE_CONSTANT)
                        {
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 4);
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 5);
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                            tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                        }
                        else if (borderType != BORDER_MODE_REFLECT101)
                        {
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 4);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 5);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tcurr, 6);
                            tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tcurr, 7);
                        }
                        break;
                    }
                    continue;
                }

                if(cn==2)
                    t0 = vextq_u16(tprev, tcurr, 6);
                else if(cn==3)
                    t0 = vextq_u16(tprev, tcurr, 5);
                else if(cn==4)
                    t0 = vextq_u16(tprev, tcurr, 4);

                t1 = tcurr;

                if(cn==2)
                    t2 = vextq_u16(tcurr, tnext, 2);
                else if(cn==3)
                    t2 = vextq_u16(tcurr, tnext, 3);
                else if(cn==4)
                    t2 = vextq_u16(tcurr, tnext, 4);

                t0 = vqaddq_u16(t0, vqaddq_u16(t1, t2));

#ifdef FLOAT_VARIANT_1_9
                uint32x4_t tres1 = vmovl_u16(vget_low_u16(t0));
                uint32x4_t tres2 = vmovl_u16(vget_high_u16(t0));
                float32x4_t vf1 = vmulq_f32(v1_9, vcvtq_f32_u32(tres1));
                float32x4_t vf2 = vmulq_f32(v1_9, vcvtq_f32_u32(tres2));
                tres1 = vcvtq_u32_f32(vaddq_f32(vf1, v0_5));
                tres2 = vcvtq_u32_f32(vaddq_f32(vf2, v0_5));
                t0 = vcombine_u16(vmovn_u32(tres1),vmovn_u32(tres2));
                vst1_u8(drow + x - 8, vmovn_u16(t0));
#else
                int16x8_t tt0 = vqrdmulhq_s16(vreinterpretq_s16_u16(t0), vScale);
                uint8x8_t it0 = vmovn_u16(vreinterpretq_u16_s16(tt0));
                vst1_u8(drow + x - 8, it0);
#endif
            }

            x -= 8;
            if(x == colsn){
                x -= cn;
            }
            s16 prevx[4], rowx[4], nextx[4];
            for( s32 k = 0; k < cn; k++ )
            {
                prevx[(k + x%cn)%cn] = srow2[x+k-cn] + srow1[x+k-cn] + srow0[x+k-cn];
                rowx[(k + x%cn)%cn] = srow2[x+k] + srow1[x+k] + srow0[x+k];
            }
            for( ; x < colsn; x++ )
            {
                size_t xx = x%cn;
                if(x+cn >= colsn) {
                    // make border
                    if (borderType == BORDER_MODE_CONSTANT)
                    {
                        nextx[xx] = borderValue;
                    } else if (borderType == BORDER_MODE_REFLECT101)
                    {
                        nextx[xx] = srow2[x-cn] + srow1[x-cn] + srow0[x-cn];
                    } else {
                        nextx[xx] = srow2[x] + srow1[x] + srow0[x];
                    }
                } else {
                    nextx[xx] = srow2[x+cn] + srow1[x+cn] + srow0[x+cn];
                }
                *(drow+x) = internal::saturate_cast<u8>((prevx[xx] + rowx[xx] + nextx[xx])*(1/9.));
                prevx[xx] = rowx[xx];
                rowx[xx] = nextx[xx];
            }
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

void blur5x5(const Size2D &size, s32 cn,
             const u8 * srcBase, ptrdiff_t srcStride,
             u8 * dstBase, ptrdiff_t dstStride,
             BORDER_MODE borderType, u8 borderValue)
{
    internal::assertSupportedConfiguration(isBlurU8Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
#define FLOAT_VARIANT_1_25
#ifdef FLOAT_VARIANT_1_25
    float32x4_t v1_25 = vdupq_n_f32 (1.0f/25.0f);
    float32x4_t v0_5 = vdupq_n_f32 (.5f);
#else
    const int16x8_t vScale = vmovq_n_s16(1310);
#endif
    size_t colsn = size.width*cn;

    std::vector<u8> _tmp;
    u8 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 2*cn, borderValue);
        tmp = &_tmp[cn];
    }

    uint16x8_t tprev = vdupq_n_u16(0x0);
    uint16x8_t tcurr = tprev;
    uint16x8_t tnext = tprev;
    uint16x8_t t0, t1, t2, t3, t4;
    for( size_t y = 0; y < size.height; y++ )
    {
        const u8 *srow0, *srow1;
        const u8 *srow2 = internal::getRowPtr(srcBase, srcStride, y);
        const u8 *srow3, *srow4;
        u8 *drow = internal::getRowPtr(dstBase, dstStride, y);
        if (borderType == BORDER_MODE_REFLECT101) {
            srow0 = internal::getRowPtr(srcBase, srcStride, y > 1 ? y-2 : 2-y);
            srow1 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 1);
            srow3 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-2);
            srow4 = internal::getRowPtr(srcBase, srcStride, y < size.height-2 ? y+2 : (size.height<<1)-4-y);
        } else  if (borderType == BORDER_MODE_CONSTANT) {
            srow0 = y > 1 ? internal::getRowPtr(srcBase, srcStride, y-2) : tmp;
            srow1 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
            srow3 =  y < size.height-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
            srow4 =  y < size.height-2 ? internal::getRowPtr(srcBase, srcStride, y+2) : tmp;
        } else  if (borderType == BORDER_MODE_REFLECT) {
            srow0 = internal::getRowPtr(srcBase, srcStride, y > 1 ? y-2 : 1-y);
            srow1 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            srow3 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-1);
            srow4 = internal::getRowPtr(srcBase, srcStride, y < size.height-2 ? y+2 : (size.height<<1)-3-y);
        } else { // BORDER_MODE_REPLICATE
            srow0 = internal::getRowPtr(srcBase, srcStride, y > 1 ? y-2 : 0);
            srow1 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            srow3 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-1);
            srow4 = internal::getRowPtr(srcBase, srcStride, y < size.height-2 ? y+2 : size.height-1);
        }

        // do vertical convolution
        size_t x = 0;
        const size_t bcols = y + 3 < size.height ? colsn : (colsn - 8);
        for( ; x <= bcols; x += 8 )
        {
            internal::prefetch(srow0 + x);
            internal::prefetch(srow1 + x);
            internal::prefetch(srow2 + x);
            internal::prefetch(srow3 + x);
            internal::prefetch(srow4 + x);

            uint8x8_t x0 = vld1_u8(srow0 + x);
            uint8x8_t x1 = vld1_u8(srow1 + x);
            uint8x8_t x2 = vld1_u8(srow2 + x);
            uint8x8_t x3 = vld1_u8(srow3 + x);
            uint8x8_t x4 = vld1_u8(srow4 + x);

            tprev = tcurr;
            tcurr = tnext;
            tnext = vaddw_u8(vaddq_u16(vaddl_u8(x0, x1), vaddl_u8(x2, x3)), x4);

            if(!x) {
                tcurr = tnext;

                if(borderType == BORDER_MODE_REFLECT101 && size.width < 3)
                {
                    x = 8;
                    break;
                }

                // make border
                switch(cn)
                {
                case 1:
                    if (borderType == BORDER_MODE_CONSTANT)
                    {
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                    }
                    else if (borderType == BORDER_MODE_REFLECT101)
                    {
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tcurr, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 7);
                    }
                    else if (borderType == BORDER_MODE_REFLECT)
                    {
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 7);
                    }
                    else
                    {
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 7);
                    }
                    break;
                case 2:
                    if (borderType == BORDER_MODE_CONSTANT)
                    {
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 4);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 5);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                    }
                    else if (borderType == BORDER_MODE_REFLECT101)
                    {
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tcurr, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tcurr, 7);
                    }
                    else if (borderType == BORDER_MODE_REFLECT)
                    {
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tcurr, 4);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tcurr, 5);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 7);
                    }
                    else
                    {
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 4);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 5);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tcurr, 7);
                    }
                    break;
                case 3:
                    if (borderType == BORDER_MODE_CONSTANT)
                    {
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 2);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 3);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 4);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 5);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                    }
                    else if (borderType == BORDER_MODE_REFLECT101)
                    {
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 6),tcurr, 2);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 7),tprev, 3);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tprev, 5);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 4),tprev, 6);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 5),tprev, 7);
                        s16 lane8 = srow4[8] + srow3[8] + srow2[8] + srow1[8] + srow0[8];
                        tcurr = vsetq_lane_u16(lane8,tprev, 4);
                    }
                    else if (borderType == BORDER_MODE_REFLECT)
                    {
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 3),tcurr, 2);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 4),tprev, 3);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 5),tprev, 4);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tprev, 5);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tprev, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tprev, 7);
                    }
                    else
                    {
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tcurr, 2);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tprev, 3);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tprev, 4);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 0),tprev, 5);
                        tprev = vsetq_lane_u16(vgetq_lane_u16(tcurr, 1),tprev, 6);
                        tcurr = vsetq_lane_u16(vgetq_lane_u16(tcurr, 2),tprev, 7);
                    }
                    break;
                case 4:
                    if (borderType == BORDER_MODE_CONSTANT)
                    {
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 0);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 1);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 2);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 3);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 4);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 5);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 6);
                        tcurr = vsetq_lane_u16(borderValue, tcurr, 7);
                    }
                    else if (borderType == BORDER_MODE_REFLECT101)
                    {
                        s16 lane8  = srow4[ 8] + srow3[ 8] + srow2[ 8] + srow1[ 8] + srow0[ 8];
                        s16 lane9  = srow4[ 9] + srow3[ 9] + srow2[ 9] + srow1[ 9] + srow0[ 9];
                        s16 lane10 = srow4[10] + srow3[10] + srow2[10] + srow1[10] + srow0[10];
                        s16 lane11 = srow4[11] + srow3[11] + srow2[11] + srow1[11] + srow0[11];
                        tprev = vsetq_lane_u16( lane8,tcurr, 0);
                        tprev = vsetq_lane_u16( lane9,tprev, 1);
                        tprev = vsetq_lane_u16(lane10,tprev, 2);
                        tcurr = vsetq_lane_u16(lane11,tprev, 3);
                    }
                    else if (borderType == BORDER_MODE_REFLECT)
                    {
                        tcurr = vcombine_u16(vget_high_u16(tcurr),vget_low_u16(tcurr));//swap 64-bit parts
                    }
                    else
                    {
                        tcurr = vcombine_u16(vget_low_u16(tcurr),vget_low_u16(tcurr));//double 64-bit part
                    }
                    break;
                }
                continue;
            }
            switch(cn)
            {
            case 1:
                t0 = vextq_u16(tprev, tcurr, 6);
                t1 = vextq_u16(tprev, tcurr, 7);
                t2 = tcurr;
                t3 = vextq_u16(tcurr, tnext, 1);
                t4 = vextq_u16(tcurr, tnext, 2);
                break;
            case 2:
                t0 = vextq_u16(tprev, tcurr, 4);
                t1 = vextq_u16(tprev, tcurr, 6);
                t2 = tcurr;
                t3 = vextq_u16(tcurr, tnext, 2);
                t4 = vextq_u16(tcurr, tnext, 4);
                break;
            case 3:
                t0 = vextq_u16(tprev, tcurr, 2);
                t1 = vextq_u16(tprev, tcurr, 5);
                t2 = tcurr;
                t3 = vextq_u16(tcurr, tnext, 3);
                t4 = vextq_u16(tcurr, tnext, 6);
                break;
            case 4:
                t0 = tprev;
                t1 = vextq_u16(tprev, tcurr, 4);
                t2 = tcurr;
                t3 = vextq_u16(tcurr, tnext, 4);
                t4 = tnext;
                break;
            default:
                internal::assertSupportedConfiguration(false);//Unsupported channels number
                return;
            }
            t0 = vqaddq_u16(vqaddq_u16(vqaddq_u16(t0, t1), vqaddq_u16(t2, t3)), t4);

#ifdef FLOAT_VARIANT_1_25
            uint32x4_t tres1 = vmovl_u16(vget_low_u16(t0));
            uint32x4_t tres2 = vmovl_u16(vget_high_u16(t0));
            float32x4_t vf1 = vmulq_f32(v1_25, vcvtq_f32_u32(tres1));
            float32x4_t vf2 = vmulq_f32(v1_25, vcvtq_f32_u32(tres2));
            tres1 = vcvtq_u32_f32(vaddq_f32(vf1, v0_5));
            tres2 = vcvtq_u32_f32(vaddq_f32(vf2, v0_5));
            t0 = vcombine_u16(vmovn_u32(tres1),vmovn_u32(tres2));
            vst1_u8(drow + x - 8, vmovn_u16(t0));
#else
            int16x8_t tt0 = vqrdmulhq_s16(vreinterpretq_s16_u16(t0), vScale);
            uint8x8_t it0 = vmovn_u16(vreinterpretq_u16_s16(tt0));
            vst1_u8(drow + x - 8, it0);
#endif
        }

        x -= 8;
        if(x == colsn){
            x -= cn;
        }
        s16 pprevx[4], prevx[4], rowx[4], nextx[4], nnextx[4];
        ptrdiff_t px = x / cn;
        for( s32 k = 0; k < cn; k++ )
        {
            ptrdiff_t ploc;
            ploc = internal::borderInterpolate(px-2, size.width, borderType);
            pprevx[k] = ploc < 0 ? 5*borderValue :
                                   srow4[ploc*cn+k] + srow3[ploc*cn+k] + srow2[ploc*cn+k] + srow1[ploc*cn+k] + srow0[ploc*cn+k];

            ploc = internal::borderInterpolate(px-1, size.width, borderType);
            prevx[k]  = ploc < 0 ? 5*borderValue :
                                   srow4[ploc*cn+k] + srow3[ploc*cn+k] + srow2[ploc*cn+k] + srow1[ploc*cn+k] + srow0[ploc*cn+k];

            rowx[k]   = srow4[px*cn+k] + srow3[px*cn+k] + srow2[px*cn+k] + srow1[px*cn+k] + srow0[px*cn+k];

            ploc = internal::borderInterpolate(px+1, size.width, borderType);
            nextx[k]  = ploc < 0 ? 5*borderValue :
                                   srow4[ploc*cn+k] + srow3[ploc*cn+k] + srow2[ploc*cn+k] + srow1[ploc*cn+k] + srow0[ploc*cn+k];
        }
        x = px*cn;
        for( ; x < colsn; x+=cn, px++ )
        {
            for( s32 k = 0; k < cn; k++ )
            {
                ptrdiff_t ploc = internal::borderInterpolate(px+2, size.width, borderType);
                nnextx[k] = ploc < 0 ? 5*borderValue :
                                       srow4[ploc*cn+k] + srow3[ploc*cn+k] + srow2[ploc*cn+k] + srow1[ploc*cn+k] + srow0[ploc*cn+k];
                *(drow+x+k) = internal::saturate_cast<u8>((pprevx[k] + prevx[k] + rowx[k] + nextx[k] +nnextx[k])*(1/25.));
                pprevx[k] = prevx[k];
                prevx[k]  = rowx[k];
                rowx[k]   = nextx[k];
                nextx[k]  = nnextx[k];
            }
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

bool isBlurF32Supported(const Size2D &size, s32 cn, BORDER_MODE border)
{
    return isSupportedConfiguration() &&
           cn > 0 && cn <= 4 &&
           size.width*cn >= 4 && size.height >= 2 &&
           (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REFLECT101 ||
            border == BORDER_MODE_REFLECT ||
            border == BORDER_MODE_REPLICATE ||
            border == BORDER_MODE_WRAP);
}

void blur3x3(const Size2D &size, s32 cn,
             const f32 * srcBase, ptrdiff_t srcStride,
             f32 * dstBase, ptrdiff_t dstStride,
             BORDER_MODE borderType, f32 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isBlurF32Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
    size_t colsn = size.width * cn;

    std::vector<f32> _tmp;
    f32 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 2*cn, borderValue);
        tmp = &_tmp[cn];
    }

    ptrdiff_t idx_l = internal::borderInterpolate(-1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r = internal::borderInterpolate(size.width, size.width, borderType, borderMargin.left, borderMargin.right) * cn;

    //2-line buffer
    std::vector<f32> _buf(4*(cn * (size.width + 2) + 32 / sizeof(f32)));
    f32* lanea = internal::alignPtr(&_buf[cn], 32);
    f32* laneA = internal::alignPtr(lanea + cn * (size.width + 2), 32);

    f32* laneb = internal::alignPtr(laneA + cn * (size.width + 2), 32);
    f32* laneB = internal::alignPtr(laneb + cn * (size.width + 2), 32);

    if (borderType == BORDER_MODE_CONSTANT)
        for (s32 k = 0; k < cn; ++k)
        {
            lanea[-cn+k] = borderValue;
            lanea[colsn+k] = borderValue;
            laneA[-cn+k] = borderValue;
            laneA[colsn+k] = borderValue;
            laneb[-cn+k] = borderValue;
            laneb[colsn+k] = borderValue;
            laneB[-cn+k] = borderValue;
            laneB[colsn+k] = borderValue;
        }

    size_t i = 0;
    f32* dsta = internal::getRowPtr(dstBase, dstStride, 0);
    for (; i < size.height-1; i+=2)
    {
        //vertical convolution
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp2 = internal::borderInterpolate(i + 2, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const f32* ln0 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const f32* ln1 = internal::getRowPtr(srcBase, srcStride, i);
        const f32* ln2 = internal::getRowPtr(srcBase, srcStride, i + 1);
        const f32* ln3 = idx_rp2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp2) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(ln1 + x);
            internal::prefetch(ln2 + x);
            internal::prefetch(ln0 + x);
            internal::prefetch(ln3 + x);
box3x3f32_vert:
            float32x4_t v1 = vld1q_f32(ln1 + x);
            float32x4_t v2 = vld1q_f32(ln2 + x);
            float32x4_t v0 = vld1q_f32(ln0 + x);
            float32x4_t v3 = vld1q_f32(ln3 + x);

            float32x4_t v = vaddq_f32(v1, v2);
            float32x4_t w0 = vaddq_f32(v, v0);
            float32x4_t w1 = vaddq_f32(v, v3);

            vst1q_f32(lanea + x, w0);
            vst1q_f32(laneb + x, w1);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3f32_vert;
        }

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lanea[-cn+k] = lanea[idx_l + k];
                lanea[colsn+k] = lanea[idx_r + k];
                laneb[-cn+k] = laneb[idx_l + k];
                laneb[colsn+k] = laneb[idx_r + k];
            }

        //horizontal convolution (2 lines from previous iteration)
        if (i > 0)
        {
            f32* dstb = internal::getRowPtr(dstBase, dstStride, i-1);
            x = 0;
            for (; x <= colsn - 4; x += 4)
            {
                internal::prefetch(laneA + x + cn);
                internal::prefetch(laneB + x + cn);
box3x3f32_horiz:
                float32x4_t lane0a = vld1q_f32(laneA + x - cn);
                float32x4_t lane2a = vld1q_f32(laneA + x + cn);
                float32x4_t lane1a = vld1q_f32(laneA + x);

                float32x4_t lane0b = vld1q_f32(laneB + x - cn);
                float32x4_t lane2b = vld1q_f32(laneB + x + cn);
                float32x4_t lane1b = vld1q_f32(laneB + x);

                float32x4_t va = vaddq_f32(lane0a, lane2a);
                float32x4_t vb = vaddq_f32(lane0b, lane2b);
                float32x4_t wa = vaddq_f32(va, lane1a);
                float32x4_t wb = vaddq_f32(vb, lane1b);

                vst1q_f32(dsta + x, wa);
                vst1q_f32(dstb + x, wb);
            }
            if(x < colsn)
            {
                x = colsn-4;
                goto box3x3f32_horiz;
            }
            dsta = internal::getRowPtr(dstBase, dstStride, i);
        }

        std::swap(lanea, laneA);
        std::swap(laneb, laneB);
    }

    //last line
    if(i < size.height)
    {
        //vertical convolution
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp1 = internal::borderInterpolate(i + 1, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const f32* ln0 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const f32* ln1 = internal::getRowPtr(srcBase, srcStride, i);
        const f32* ln2 = idx_rp1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp1) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(ln0 + x);
            internal::prefetch(ln1 + x);
            internal::prefetch(ln2 + x);
box3x3f32_vert_ll:
            float32x4_t v0 = vld1q_f32(ln0+x);
            float32x4_t v1 = vld1q_f32(ln1+x);
            float32x4_t v2 = vld1q_f32(ln2+x);

            float32x4_t v = vaddq_f32(v0, v1);
            float32x4_t w = vaddq_f32(v, v2);

            vst1q_f32(lanea + x, w);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3f32_vert_ll;
        }

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lanea[-cn+k] = lanea[idx_l + k];
                lanea[colsn+k] = lanea[idx_r + k];
            }

        //horizontal convolution (last 3 lines)
        x = 0;
        f32* dstb = internal::getRowPtr(dstBase, dstStride, i-1);
        f32* dstc = internal::getRowPtr(dstBase, dstStride, i);
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(laneA + x + cn);
            internal::prefetch(laneB + x + cn);
            internal::prefetch(lanea + x + cn);
box3x3f32_horiz_ll:
            float32x4_t lane0a = vld1q_f32(laneA + x - cn);
            float32x4_t lane2a = vld1q_f32(laneA + x + cn);
            float32x4_t lane1a = vld1q_f32(laneA + x);

            float32x4_t lane0b = vld1q_f32(laneB + x - cn);
            float32x4_t lane2b = vld1q_f32(laneB + x + cn);
            float32x4_t lane1b = vld1q_f32(laneB + x);

            float32x4_t lane0c = vld1q_f32(lanea + x - cn);
            float32x4_t lane2c = vld1q_f32(lanea + x + cn);
            float32x4_t lane1c = vld1q_f32(lanea + x);

            float32x4_t va = vaddq_f32(lane0a, lane2a);
            float32x4_t vb = vaddq_f32(lane0b, lane2b);
            float32x4_t vc = vaddq_f32(lane0c, lane2c);
            float32x4_t wa = vaddq_f32(va, lane1a);
            float32x4_t wb = vaddq_f32(vb, lane1b);
            float32x4_t wc = vaddq_f32(vc, lane1c);

            vst1q_f32(dsta + x, wa);
            vst1q_f32(dstb + x, wb);
            vst1q_f32(dstc + x, wc);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3f32_horiz_ll;
        }
    }
    else
    {
        //horizontal convolution (last 2 lines)
        f32* dstb = internal::getRowPtr(dstBase, dstStride, i-1);
        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(laneA + x + cn);
            internal::prefetch(laneB + x + cn);
box3x3f32_horiz_last2:
            float32x4_t lane0a = vld1q_f32(laneA + x - cn);
            float32x4_t lane2a = vld1q_f32(laneA + x + cn);
            float32x4_t lane1a = vld1q_f32(laneA + x);

            float32x4_t lane0b = vld1q_f32(laneB + x - cn);
            float32x4_t lane2b = vld1q_f32(laneB + x + cn);
            float32x4_t lane1b = vld1q_f32(laneB + x);

            float32x4_t va = vaddq_f32(lane0a, lane2a);
            float32x4_t vb = vaddq_f32(lane0b, lane2b);
            float32x4_t wa = vaddq_f32(va, lane1a);
            float32x4_t wb = vaddq_f32(vb, lane1b);

            vst1q_f32(dsta + x, wa);
            vst1q_f32(dstb + x, wb);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3f32_horiz_last2;
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

bool isBlurS32Supported(const Size2D &size, s32 cn, BORDER_MODE border)
{
    return isSupportedConfiguration() &&
           cn > 0 && cn <= 4 &&
           size.width*cn >= 4 && size.height >= 2 &&
           (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REFLECT101 ||
            border == BORDER_MODE_REFLECT ||
            border == BORDER_MODE_REPLICATE ||
            border == BORDER_MODE_WRAP);
}

void blur3x3(const Size2D &size, s32 cn,
             const s32 * srcBase, ptrdiff_t srcStride,
             s32 * dstBase, ptrdiff_t dstStride,
             BORDER_MODE borderType, s32 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isBlurS32Supported(size, cn, borderType));
#ifdef CAROTENE_NEON
    size_t colsn = size.width * cn;

    std::vector<s32> _tmp;
    s32 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(colsn + 2*cn, borderValue);
        tmp = &_tmp[cn];
    }

    ptrdiff_t idx_l = internal::borderInterpolate(-1, size.width, borderType, borderMargin.left, borderMargin.right) * cn;
    ptrdiff_t idx_r = internal::borderInterpolate(size.width, size.width, borderType, borderMargin.left, borderMargin.right) * cn;

    //2-line buffer
    std::vector<s32> _buf(4*(cn * (size.width + 2) + 32 / sizeof(s32)));
    s32* lanea = internal::alignPtr(&_buf[cn], 32);
    s32* laneA = internal::alignPtr(lanea + cn * (size.width + 2), 32);

    s32* laneb = internal::alignPtr(laneA + cn * (size.width + 2), 32);
    s32* laneB = internal::alignPtr(laneb + cn * (size.width + 2), 32);

    if (borderType == BORDER_MODE_CONSTANT)
        for (s32 k = 0; k < cn; ++k)
        {
            lanea[-cn+k] = borderValue;
            lanea[colsn+k] = borderValue;
            laneA[-cn+k] = borderValue;
            laneA[colsn+k] = borderValue;
            laneb[-cn+k] = borderValue;
            laneb[colsn+k] = borderValue;
            laneB[-cn+k] = borderValue;
            laneB[colsn+k] = borderValue;
        }

    size_t i = 0;
    s32* dsta = internal::getRowPtr(dstBase, dstStride, 0);
    for (; i < size.height-1; i+=2)
    {
        //vertical convolution
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp2 = internal::borderInterpolate(i + 2, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const s32* ln0 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const s32* ln1 = internal::getRowPtr(srcBase, srcStride, i);
        const s32* ln2 = internal::getRowPtr(srcBase, srcStride, i + 1);
        const s32* ln3 = idx_rp2 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp2) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(ln1 + x);
            internal::prefetch(ln2 + x);
            internal::prefetch(ln0 + x);
            internal::prefetch(ln3 + x);
box3x3s32_vert:
            int32x4_t v1 = vld1q_s32(ln1 + x);
            int32x4_t v2 = vld1q_s32(ln2 + x);
            int32x4_t v0 = vld1q_s32(ln0 + x);
            int32x4_t v3 = vld1q_s32(ln3 + x);

            int32x4_t v = vaddq_s32(v1, v2);
            int32x4_t w0 = vaddq_s32(v, v0);
            int32x4_t w1 = vaddq_s32(v, v3);

            vst1q_s32(lanea + x, w0);
            vst1q_s32(laneb + x, w1);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3s32_vert;
        }

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lanea[-cn+k] = lanea[idx_l + k];
                lanea[colsn+k] = lanea[idx_r + k];
                laneb[-cn+k] = laneb[idx_l + k];
                laneb[colsn+k] = laneb[idx_r + k];
            }

        //horizontal convolution (2 lines from previous iteration)
        if (i > 0)
        {
            s32* dstb = internal::getRowPtr(dstBase, dstStride, i-1);
            x = 0;
            for (; x <= colsn - 4; x += 4)
            {
                internal::prefetch(laneA + x + cn);
                internal::prefetch(laneB + x + cn);
box3x3s32_horiz:
                int32x4_t lane0a = vld1q_s32(laneA + x - cn);
                int32x4_t lane2a = vld1q_s32(laneA + x + cn);
                int32x4_t lane1a = vld1q_s32(laneA + x);

                int32x4_t lane0b = vld1q_s32(laneB + x - cn);
                int32x4_t lane2b = vld1q_s32(laneB + x + cn);
                int32x4_t lane1b = vld1q_s32(laneB + x);

                int32x4_t va = vaddq_s32(lane0a, lane2a);
                int32x4_t vb = vaddq_s32(lane0b, lane2b);
                int32x4_t wa = vaddq_s32(va, lane1a);
                int32x4_t wb = vaddq_s32(vb, lane1b);

                vst1q_s32(dsta + x, wa);
                vst1q_s32(dstb + x, wb);
            }
            if(x < colsn)
            {
                x = colsn-4;
                goto box3x3s32_horiz;
            }
            dsta = internal::getRowPtr(dstBase, dstStride, i);
        }

        std::swap(lanea, laneA);
        std::swap(laneb, laneB);
    }
    //last line
    if(i < size.height)
    {
        //vertical convolution
        ptrdiff_t idx_rm1 = internal::borderInterpolate(i - 1, size.height, borderType, borderMargin.top, borderMargin.bottom);
        ptrdiff_t idx_rp1 = internal::borderInterpolate(i + 1, size.height, borderType, borderMargin.top, borderMargin.bottom);

        const s32* ln0 = idx_rm1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rm1) : tmp;
        const s32* ln1 = internal::getRowPtr(srcBase, srcStride, i);
        const s32* ln2 = idx_rp1 >= -(ptrdiff_t)borderMargin.top ? internal::getRowPtr(srcBase, srcStride, idx_rp1) : tmp;

        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(ln0 + x);
            internal::prefetch(ln1 + x);
            internal::prefetch(ln2 + x);
box3x3s32_vert_ll:
            int32x4_t v0 = vld1q_s32(ln0+x);
            int32x4_t v1 = vld1q_s32(ln1+x);
            int32x4_t v2 = vld1q_s32(ln2+x);

            int32x4_t v = vaddq_s32(v0, v1);
            int32x4_t w = vaddq_s32(v, v2);

            vst1q_s32(lanea + x, w);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3s32_vert_ll;
        }

        //left&right borders
        if (borderType != BORDER_MODE_CONSTANT)
            for (s32 k = 0; k < cn; ++k)
            {
                lanea[-cn+k] = lanea[idx_l + k];
                lanea[colsn+k] = lanea[idx_r + k];
            }

        //horizontal convolution (last 3 lines)
        x = 0;
        s32* dstb = internal::getRowPtr(dstBase, dstStride, i-1);
        s32* dstc = internal::getRowPtr(dstBase, dstStride, i);
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(laneA + x + cn);
            internal::prefetch(laneB + x + cn);
            internal::prefetch(lanea + x + cn);
box3x3s32_horiz_ll:
            int32x4_t lane0a = vld1q_s32(laneA + x - cn);
            int32x4_t lane2a = vld1q_s32(laneA + x + cn);
            int32x4_t lane1a = vld1q_s32(laneA + x);

            int32x4_t lane0b = vld1q_s32(laneB + x - cn);
            int32x4_t lane2b = vld1q_s32(laneB + x + cn);
            int32x4_t lane1b = vld1q_s32(laneB + x);

            int32x4_t lane0c = vld1q_s32(lanea + x - cn);
            int32x4_t lane2c = vld1q_s32(lanea + x + cn);
            int32x4_t lane1c = vld1q_s32(lanea + x);

            int32x4_t va = vaddq_s32(lane0a, lane2a);
            int32x4_t vb = vaddq_s32(lane0b, lane2b);
            int32x4_t vc = vaddq_s32(lane0c, lane2c);
            int32x4_t wa = vaddq_s32(va, lane1a);
            int32x4_t wb = vaddq_s32(vb, lane1b);
            int32x4_t wc = vaddq_s32(vc, lane1c);

            vst1q_s32(dsta + x, wa);
            vst1q_s32(dstb + x, wb);
            vst1q_s32(dstc + x, wc);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3s32_horiz_ll;
        }
    }
    else
    {
        //horizontal convolution (last 2 lines)
        s32* dstb = internal::getRowPtr(dstBase, dstStride, i-1);
        size_t x = 0;
        for (; x <= colsn - 4; x += 4)
        {
            internal::prefetch(laneA + x + cn);
            internal::prefetch(laneB + x + cn);
box3x3s32_horiz_last2:
            int32x4_t lane0a = vld1q_s32(laneA + x - cn);
            int32x4_t lane2a = vld1q_s32(laneA + x + cn);
            int32x4_t lane1a = vld1q_s32(laneA + x);

            int32x4_t lane0b = vld1q_s32(laneB + x - cn);
            int32x4_t lane2b = vld1q_s32(laneB + x + cn);
            int32x4_t lane1b = vld1q_s32(laneB + x);

            int32x4_t va = vaddq_s32(lane0a, lane2a);
            int32x4_t vb = vaddq_s32(lane0b, lane2b);
            int32x4_t wa = vaddq_s32(va, lane1a);
            int32x4_t wb = vaddq_s32(vb, lane1b);

            vst1q_s32(dsta + x, wa);
            vst1q_s32(dstb + x, wb);
        }
        if(x < colsn)
        {
            x = colsn-4;
            goto box3x3s32_horiz_last2;
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

} //namespace CAROTENE_NS
