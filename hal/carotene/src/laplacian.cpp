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
 * Copyright (C) 2015, NVIDIA Corporation, all rights reserved.
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

#include <vector>

namespace CAROTENE_NS {

bool isLaplacian3x3Supported(const Size2D &size, BORDER_MODE border)
{
    return isSupportedConfiguration() && size.width >= 8 &&
        (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REPLICATE);
}

void Laplacian3x3(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isLaplacian3x3Supported(size, border));
#ifdef CAROTENE_NEON
    const uint16x8_t v_border_x3 = vdupq_n_u16(borderValue * 3);
    const uint16x8_t v_zero = vdupq_n_u16(0);
    const uint8x8_t v_border = vdup_n_u8(borderValue);

    uint8x8_t vsub;
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

                vsub = x1;

                continue;
            }

            // combine 3 "shifted" vectors
            t0 = vextq_u16(tprev, tcurr, 7);
            t1 = tcurr;
            t2 = vextq_u16(tcurr, tnext, 1);

            // and add them
            t0 = vqaddq_u16(t0, vqaddq_u16(t1, t2));

            int16x8_t tt0 = vsubq_s16(vreinterpretq_s16_u16(t0),
                                      vreinterpretq_s16_u16(vaddw_u8(vshll_n_u8(vsub, 3), vsub)));
            uint8x8_t it0 = vqmovun_s16(tt0);
            vst1_u8(drow + x - 8, it0);

            vsub = x1;
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
            {
                nextx = (srow2 ? srow2[x + 1] : borderValue) +
                                 srow1[x + 1] +
                        (srow0 ? srow0[x + 1] : borderValue);
            }

            s32 val = (prevx + currx + nextx) - 9 * srow1[x];
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

bool isLaplacianOpenCVSupported(const Size2D &size, BORDER_MODE border)
{
    return isSupportedConfiguration() &&
        size.width >= 8 && size.height >= 1 &&
        (border == BORDER_MODE_CONSTANT   ||
         border == BORDER_MODE_REFLECT    ||
         border == BORDER_MODE_REFLECT101 ||
         border == BORDER_MODE_REPLICATE);
}

void Laplacian1OpenCV(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isLaplacianOpenCVSupported(size, border));
#ifdef CAROTENE_NEON
    ptrdiff_t rows = size.height, cols = size.width;

    std::vector<u8> _tmp;
    u8 *tmp = 0;
    if (border == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(cols + 4,borderValue);
        tmp = &_tmp[2];
    }

    for( ptrdiff_t y = 0; y < rows; y++ )
    {
        const u8* v0 = 0;
        const u8* v1 = internal::getRowPtr(srcBase, srcStride, y);
        const u8* v2 = 0;
        // make border
        if (border == BORDER_MODE_REFLECT101) {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : y+1);
            v2 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        } else  if (border == BORDER_MODE_CONSTANT) {
            v0 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
            v2 =  y < rows-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
        } else {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            v2 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 0 ? rows-1 : 0);
        }
        s16* drow = internal::getRowPtr(dstBase, dstStride, y);

        int16x8_t tcurr = vmovq_n_s16(0x0);
        int16x8_t tnext = vmovq_n_s16(0x0);
        int16x8_t t0, t2;
        uint8x8_t xx0 = vmov_n_u8(0x0);
        uint8x8_t xx1 = vmov_n_u8(0x0);
        uint8x8_t xx2 = vmov_n_u8(0x0);
        ptrdiff_t x = 0;
        const ptrdiff_t bcols = y + 2 < rows ? cols : (cols - 8);
        for( ; x <= bcols; x += 8 )
        {
            internal::prefetch(v0 + x);
            internal::prefetch(v1 + x);
            internal::prefetch(v2 + x);

            uint8x8_t x0 = vld1_u8(v0 + x);
            uint8x8_t x1 = vld1_u8(v1 + x);
            uint8x8_t x2 = vld1_u8(v2 + x);

            if(x) {
                xx0 = xx1;
                xx1 = xx2;
            } else {
                xx1 = x1;
                // make border
                    if (border == BORDER_MODE_REPLICATE || border == BORDER_MODE_REFLECT)
                    {
                        xx1 = vset_lane_u8(vget_lane_u8(x1, 0),x1, 7);
                    }
                    else if (border == BORDER_MODE_CONSTANT)
                    {
                        xx1 = vset_lane_u8(borderValue, x1, 7);
                    }
                    else if (border == BORDER_MODE_REFLECT101)
                    {
                        xx1 = vset_lane_u8(vget_lane_u8(x1, 1),x1, 7);
                    }
            }
            xx2 = x1;

            if(x) {
                tcurr = tnext;
            }
            tnext = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(x0, x2)),
                              vreinterpretq_s16_u16(vshll_n_u8(x1, 2)));

            if(!x) {
                tcurr = tnext;
                continue;
            }
            t0 = vreinterpretq_s16_u16(vmovl_u8(vext_u8(xx0, xx1, 7)));
            t2 = vreinterpretq_s16_u16(vmovl_u8(vext_u8(xx1, xx2, 1)));
            t0 = vaddq_s16(vqaddq_s16(t0, t2), tcurr);

            vst1q_s16(drow + x - 8, t0);
        }

        x -= 8;
        if(x == cols){
            x--;
        }

        for( ; x < cols; x++ )
        {
            s16 nextx;
            s16 prevx;
            // make border
            if (border == BORDER_MODE_REPLICATE || border == BORDER_MODE_REFLECT)
            {
                prevx = x == 0 ? v1[0] : v1[x-1];
                nextx = x == cols-1 ? v1[x] : v1[x+1];
            }
            else if (border == BORDER_MODE_REFLECT101)
            {
                prevx = x == 0 ? v1[1] : v1[x-1];
                nextx = x == cols-1 ? v1[x-1] : v1[x+1];
            }
            else //if (border == BORDER_MODE_CONSTANT)
            {
                prevx = x == 0 ? borderValue : v1[x-1];
                nextx = x == cols-1 ? borderValue : v1[x+1];
            }
            *(drow+x) = prevx + nextx - 4*v1[x] + v0[x] + v2[x];
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

void Laplacian3OpenCV(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isLaplacianOpenCVSupported(size, border));
#ifdef CAROTENE_NEON
    ptrdiff_t rows = size.height, cols = size.width;

    std::vector<u8> _tmp;
    u8 *tmp = 0;
    if (border == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(cols + 4,borderValue);
        tmp = &_tmp[2];
    }

    for( ptrdiff_t y = 0; y < rows; y++ )
    {
        const u8* v0 = 0;
        const u8* v1 = internal::getRowPtr(srcBase, srcStride, y);
        const u8* v2 = 0;
        // make border
        if (border == BORDER_MODE_REFLECT101) {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : y+1);
            v2 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        } else  if (border == BORDER_MODE_CONSTANT) {
            v0 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
            v2 = y < rows-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
        } else {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            v2 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 0 ? rows-1 : 0);
        }
        s16* drow = internal::getRowPtr(dstBase, dstStride, y);

        int16x8_t tprev = vmovq_n_s16(0x0);
        int16x8_t tcurr = vmovq_n_s16(0x0);
        int16x8_t tnext = vmovq_n_s16(0x0);
        int16x8_t tc = vmovq_n_s16(0x0);
        int16x8_t t0, t2, tcnext;
        ptrdiff_t x = 0;
        const ptrdiff_t bcols = y + 2 < rows ? cols : (cols - 8);
        for( ; x <= bcols; x += 8 )
        {
            internal::prefetch(v0 + x);
            internal::prefetch(v1 + x);
            internal::prefetch(v2 + x);

            uint8x8_t x0 = vld1_u8(v0 + x);
            uint8x8_t x1 = vld1_u8(v1 + x);
            uint8x8_t x2 = vld1_u8(v2 + x);
            tcnext = vreinterpretq_s16_u16(vshll_n_u8(x1, 2));

            if(x) {
                tprev = tcurr;
                tcurr = tnext;
            }
            tnext = vreinterpretq_s16_u16(vaddl_u8(x0, x2));

            if(!x) {
                tcurr = tnext;
                tc = tcnext;

                // make border
                    if (border == BORDER_MODE_REPLICATE || border == BORDER_MODE_REFLECT)
                    {
                        tcurr = vsetq_lane_s16(vgetq_lane_s16(tcurr, 0),tcurr, 7);
                    }
                    else if (border == BORDER_MODE_CONSTANT)
                    {
                        tcurr = vsetq_lane_s16(borderValue, tcurr, 7);
                    }
                    else if (border == BORDER_MODE_REFLECT101)
                    {
                        tcurr = vsetq_lane_s16(vgetq_lane_s16(tcurr, 1),tcurr, 7);
                    }
                continue;
            }

            t0 = vextq_s16(tprev, tcurr, 7);
            t2 = vextq_s16(tcurr, tnext, 1);

            t0 = vsubq_s16(vqaddq_s16(t0, t2), tc);
            tc = tcnext;

            t0 = vshlq_n_s16(t0, 1);
            vst1q_s16(drow + x - 8, t0);
        }
        x -= 8;
        if(x == cols){
            x--;
        }

        for( ; x < cols; x++ )
        {
            s16 nextx, nextx2;
            s16 prevx, prevx2;
            // make border
            if (border == BORDER_MODE_REPLICATE || border == BORDER_MODE_REFLECT)
            {
                prevx = x == 0 ? v0[0] : v0[x-1];
                prevx2 = x == 0 ? v2[0] : v2[x-1];
                nextx = x == cols-1 ? v0[x] : v0[x+1];
                nextx2 = x == cols-1 ? v2[x] : v2[x+1];
            }
            else if (border == BORDER_MODE_REFLECT101)
            {
                prevx = x == 0 ? v0[1] : v0[x-1];
                prevx2 = x == 0 ? v2[1] : v2[x-1];
                nextx = x == cols-1 ? v0[x-1] : v0[x+1];
                nextx2 = x == cols-1 ? v2[x-1] : v2[x+1];
            }
            else //if (border == BORDER_MODE_CONSTANT)
            {
                prevx = x == 0 ? borderValue : v0[x-1];
                prevx2 = x == 0 ? borderValue : v2[x-1];
                nextx = x == cols-1 ? borderValue : v0[x+1];
                nextx2 = x == cols-1 ? borderValue : v2[x+1];
            }
            s16 res = prevx + nextx - 4*v1[x] + prevx2 + nextx2;
            *(drow+x) = 2*res;
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

void Laplacian5OpenCV(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isLaplacianOpenCVSupported(size, border));
#ifdef CAROTENE_NEON
    ptrdiff_t rows = size.height, cols = size.width;

    std::vector<u8> _tmp;
    u8 *tmp = 0;
    if (border == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(cols + 4,borderValue);
        tmp = &_tmp[2];
    }

    for( ptrdiff_t y = 0; y < rows; y++ )
    {
        const u8* v0 = 0;
        const u8* v1 = 0;
        const u8* v2 = internal::getRowPtr(srcBase, srcStride, y);
        const u8* v3 = 0;
        const u8* v4 = 0;
        // make border
        if (border == BORDER_MODE_REPLICATE) {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 1 ? y-2 : 0);
            v1 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            v3 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 0 ? rows-1 : 0);
            v4 = internal::getRowPtr(srcBase, srcStride, y < rows-2 ? y+2 : rows > 0 ? rows-1 : 0);
        } else if (border == BORDER_MODE_REFLECT) {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 1 ? y-2 : rows > 1 ? 1-y : 0);
            v1 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            v3 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 0 ? rows-1 : 0);
            v4 = internal::getRowPtr(srcBase, srcStride, y < rows-2 ? y+2 : rows > 1 ? 2*rows-(y+3) : 0);
        } else if (border == BORDER_MODE_REFLECT101) {
            v0 = internal::getRowPtr(srcBase, srcStride, y > 1 ? y-2 : rows > 2-y ? 2-y : 0); ///check
            v1 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : rows > 1 ? 1 : 0);
            v3 = internal::getRowPtr(srcBase, srcStride, y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
            v4 = internal::getRowPtr(srcBase, srcStride, y < rows-2 ? y+2 : rows > 2 ? 2*rows-(y+4) : 0);///bad if rows=2 y=1   rows - 4 + (2,1)
        } else if (border == BORDER_MODE_CONSTANT) {
            v0 = y > 1 ? internal::getRowPtr(srcBase, srcStride, y-2) : tmp;
            v1 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
            v3 = y < rows-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
            v4 = y < rows-2 ? internal::getRowPtr(srcBase, srcStride, y+2) : tmp;
        }
        s16* drow = internal::getRowPtr(dstBase, dstStride, y);

        int16x8_t tnext, tc, t0;
        int16x8_t tnext2, tnext3;
        int16x8_t tnext1Old, tnext2Old, tnext3Old;
        int16x8_t tnext4OldOldOld, tnext5OldOldOld;

        int16x8_t tcurr1 = vmovq_n_s16(0x0);
        int16x8_t tnext1 = vmovq_n_s16(0x0);
        int16x8_t tprev1 = vmovq_n_s16(0x0);
        int16x8_t tpprev1 = vmovq_n_s16(0x0);
        int16x8_t tppprev1 = vmovq_n_s16(0x0);

        int16x8_t tnext4Old = vmovq_n_s16(0x0);
        int16x8_t tnext5Old = vmovq_n_s16(0x0);
        int16x8_t tnext1OldOld = vmovq_n_s16(0x0);
        int16x8_t tnext2OldOld = vmovq_n_s16(0x0);
        int16x8_t tnext3OldOld = vmovq_n_s16(0x0);
        int16x8_t tnext4OldOld = vmovq_n_s16(0x0);
        int16x8_t tnext5OldOld = vmovq_n_s16(0x0);

        // do vertical convolution
        ptrdiff_t x = 0;
        const ptrdiff_t bcols = y + 3 < rows ? cols : (cols - 8);
        for( ; x <= bcols; x += 8 )
        {
            internal::prefetch(v0 + x);
            internal::prefetch(v1 + x);
            internal::prefetch(v2 + x);
            internal::prefetch(v3 + x);
            internal::prefetch(v4 + x);

            uint8x8_t x0 = vld1_u8(v0 + x);
            uint8x8_t x1 = vld1_u8(v1 + x);
            uint8x8_t x2 = vld1_u8(v2 + x);
            uint8x8_t x3 = vld1_u8(v3 + x);
            uint8x8_t x4 = vld1_u8(v4 + x);
            if(x) {
                tcurr1 = tnext1;
            }

            tnext4OldOldOld = tnext4Old;
            tnext5OldOldOld = tnext5Old;
            tnext1Old = tnext1OldOld;
            tnext2Old = tnext2OldOld;
            tnext3Old = tnext3OldOld;
            tnext4Old = tnext4OldOld;
            tnext5Old = tnext5OldOld;

            tnext3 = vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(x3, x2),vaddl_u8(x2, x1)));
            tnext3 = vshlq_n_s16(tnext3, 1);

            tc = vreinterpretq_s16_u16(vsubl_u8(x4, x2));
            tnext = vreinterpretq_s16_u16(vsubl_u8(x2, x0));
            tnext2 = vsubq_s16(tc, tnext);

            tnext1 = vaddq_s16(tnext3, tnext2);
            // tnext1 = x0 + 2*x1 + 2*x2 + 2*x3 + x4

            tnext2 = vshlq_n_s16(tnext2, 1);
            // tnext2 = 2*x4 - 4*x2 + 2*x0

            tnext3 = vsubq_s16(tnext2, vshlq_n_s16(tnext3, 1));
            // tnext3 = 2*x0 - 4*x1 - 12*x2 - 4*x3  + 2*x4

            tnext1OldOld = tnext1;
            tnext2OldOld = tnext2;
            tnext3OldOld = tnext3;
            tnext4OldOld = tnext2;
            tnext5OldOld = tnext1;

            if(x) {
                tnext1 = vextq_s16(tnext1Old, tnext1, 2);
                tcurr1 = vextq_s16(tnext2Old, tnext2, 1);
                tprev1 = tnext3Old;

                if(x!=8) {
                    tpprev1 = vextq_s16(tnext4OldOldOld, tnext4Old, 7);
                    tppprev1 = vextq_s16(tnext5OldOldOld, tnext5Old, 6);
                }
            }

            if(!x) {
                // make border
                if (border == BORDER_MODE_REPLICATE) {
                    tpprev1 = vextq_s16(tnext2, tnext2, 7);
                    tpprev1 = vsetq_lane_s16(vgetq_lane_s16(tpprev1, 1),tpprev1, 0);

                    tprev1 = vextq_s16(tnext1, tnext1, 6);
                    tprev1 = vsetq_lane_s16(vgetq_lane_s16(tprev1, 2),tprev1, 0);
                    tprev1 = vsetq_lane_s16(vgetq_lane_s16(tprev1, 2),tprev1, 1);
                } else if (border == BORDER_MODE_REFLECT) {
                    tpprev1 = vextq_s16(tnext2, tnext2, 7);
                    tpprev1 = vsetq_lane_s16(vgetq_lane_s16(tpprev1, 1),tpprev1, 0);

                    tprev1 = vextq_s16(tnext1, tnext1, 6);
                    tprev1 = vsetq_lane_s16(vgetq_lane_s16(tprev1, 3),tprev1, 0);
                    tprev1 = vsetq_lane_s16(vgetq_lane_s16(tprev1, 2),tprev1, 1);
                } else if (border == BORDER_MODE_REFLECT101) {
                    tpprev1 = vextq_s16(tnext2, tnext2, 7);
                    tpprev1 = vsetq_lane_s16(vgetq_lane_s16(tpprev1, 2),tpprev1, 0);

                    tprev1 = vextq_s16(tnext1, tnext1, 6);
                    tprev1 = vsetq_lane_s16(vgetq_lane_s16(tprev1, 3),tprev1, 1);
                    tprev1 = vsetq_lane_s16(vgetq_lane_s16(tprev1, 4),tprev1, 0);
                } else if (border == BORDER_MODE_CONSTANT) {
                    tpprev1 = vextq_s16(tnext2, tnext2, 7);
                    tpprev1 = vsetq_lane_s16(borderValue, tpprev1, 0);

                    tprev1 = vextq_s16(tnext1, tnext1, 6);
                    tprev1 = vsetq_lane_s16(borderValue, tprev1, 0);
                    tprev1 = vsetq_lane_s16(borderValue, tprev1, 1);
                }
                tppprev1 = tprev1;
                continue;
            }

            t0 = vaddq_s16(vaddq_s16(vqaddq_s16(tcurr1, tprev1), vqaddq_s16(tpprev1, tppprev1)), tnext1);
            t0 = vaddq_s16(t0, t0);
            vst1q_s16(drow + x - 8, t0);
        }
        x -= 8;
        if(x >= cols - 1)
            x = cols-2;

        s16 pprevx = 0;
        s16 prevx = 0;
        s16 nextx = 0;
        s16 nnextx = 0;

        for( ; x < cols; x++ )
        {
            if (x == 0) {
                // make border
                if (border == BORDER_MODE_REPLICATE) {
                    pprevx = v0[0] + 2*v1[0] + 2*v2[0] + 2*v3[0] + v4[0];
                    prevx = 2*v0[0] - 4*v2[0] + 2*v4[0];
                } else if (border == BORDER_MODE_REFLECT) {
                    pprevx = v0[1] + 2*v1[1] + 2*v2[1] + 2*v3[1] + v4[1];
                    prevx = 2*v0[0] - 4*v2[0] + 2*v4[0];
                } else if (border == BORDER_MODE_REFLECT101) {
                    pprevx = v0[2] + 2*v1[2] + 2*v2[2] + 2*v3[2] + v4[2];
                    prevx = 2*v0[1] - 4*v2[1] + 2*v4[1];
                } else if (border == BORDER_MODE_CONSTANT) {
                    pprevx = 8 * borderValue;
                    prevx = 0;
                }
            } else if (x == 1) {
                // make border
                if (border == BORDER_MODE_REPLICATE || border == BORDER_MODE_REFLECT) {
                    pprevx = v0[0] + 2*v1[0] + 2*v2[0] + 2*v3[0] + v4[0];
                } else if (border == BORDER_MODE_REFLECT101) {
                    pprevx = v0[1] + 2*v1[1] + 2*v2[1] + 2*v3[1] + v4[1];
                } else if (border == BORDER_MODE_CONSTANT) {
                    pprevx = 8 * borderValue;
                }
                prevx = 2*v0[0] - 4*v2[0] + 2*v4[0];
            } else {
                pprevx = v0[x-2] + 2*v1[x-2] + 2*v2[x-2] + 2*v3[x-2] + v4[x-2];
                prevx = 2*v0[x-1] - 4*v2[x-1] + 2*v4[x-1];
            }
            s16 currx = 2*v0[x] - 4*v1[x] - 12*v2[x] - 4*v3[x] + 2*v4[x];
            if (x == cols-1) {
                // make border
                if (border == BORDER_MODE_REPLICATE) {
                    nextx = 2*v0[x] - 4*v2[x] + 2*v4[x];
                    nnextx = v0[x] + 2*v1[x] + 2*v2[x] + 2*v3[x] + v4[x];
                } else if (border == BORDER_MODE_REFLECT) {
                    nextx = 2*v0[x] - 4*v2[x] + 2*v4[x];
                    nnextx = v0[x-1] + 2*v1[x-1] + 2*v2[x-1] + 2*v3[x-1] + v4[x-1];
                } else if (border == BORDER_MODE_REFLECT101) {
                    nextx = 2*v0[x-1] - 4*v2[x-1] + 2*v4[x-1];
                    nnextx = v0[x-2] + 2*v1[x-2] + 2*v2[x-2] + 2*v3[x-2] + v4[x-2];
                } else if (border == BORDER_MODE_CONSTANT) {
                    nextx = 0;
                    nnextx = 8 * borderValue;
                }
            } else if (x == cols-2) {
                // make border
                if (border == BORDER_MODE_REPLICATE || border == BORDER_MODE_REFLECT) {
                    nnextx = v0[x+1] + 2*v1[x+1] + 2*v2[x+1] + 2*v3[x+1] + v4[x+1];
                } else if (border == BORDER_MODE_REFLECT101) {
                    nnextx = v0[x] + 2*v1[x] + 2*v2[x] + 2*v3[x] + v4[x];
                } else if (border == BORDER_MODE_CONSTANT) {
                    nnextx = 8 * borderValue;
                }
                nextx = 2*v0[x+1] - 4*v2[x+1] + 2*v4[x+1];
            } else {
                nextx = 2*v0[x+1] - 4*v2[x+1] + 2*v4[x+1];
                nnextx = v0[x+2] + 2*v1[x+2] + 2*v2[x+2] + 2*v3[x+2] + v4[x+2];
            }
            s16 res = pprevx + prevx + currx + nextx + nnextx;
            *(drow+x) = 2*res;
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

} // namespace CAROTENE_NS
