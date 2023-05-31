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

#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>

namespace CAROTENE_NS {

bool isMorph3x3Supported(const Size2D &size, BORDER_MODE border)
{
    return isSupportedConfiguration() && size.width >= 16 &&
        (border == BORDER_MODE_CONSTANT ||
            border == BORDER_MODE_REPLICATE);
}

#ifdef CAROTENE_NEON

namespace {

struct ErodeVecOp
{
    ErodeVecOp():borderValue(0){}

    ErodeVecOp(BORDER_MODE border, u8 borderValue_) :
        borderValue(borderValue_)
    {
        if (border == BORDER_MODE_REPLICATE)
            borderValue = std::numeric_limits<u8>::max();
    }

    inline uint8x16_t operator()(uint8x16_t a, uint8x16_t b) const
    {
        return vminq_u8(a, b);
    }

    inline uint8x8_t operator()(uint8x8_t a, uint8x8_t b) const
    {
        return vmin_u8(a, b);
    }

    inline u8 operator()(u8 a, u8 b) const
    {
        return std::min(a, b);
    }

    u8 borderValue;
};

struct DilateVecOp
{
    DilateVecOp():borderValue(0){}

    DilateVecOp(BORDER_MODE border, u8 borderValue_) :
        borderValue(borderValue_)
    {
        if (border == BORDER_MODE_REPLICATE)
            borderValue = std::numeric_limits<u8>::min();
    }

    inline uint8x16_t operator()(uint8x16_t a, uint8x16_t b) const
    {
        return vmaxq_u8(a, b);
    }

    inline uint8x8_t operator()(uint8x8_t a, uint8x8_t b) const
    {
        return vmax_u8(a, b);
    }

    inline u8 operator()(u8 a, u8 b) const
    {
        return std::max(a, b);
    }

    u8 borderValue;
};

template <typename VecOp>
void morph3x3(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              BORDER_MODE border, const VecOp & vop)
{
    u8 borderValue = vop.borderValue;
    ptrdiff_t width = (ptrdiff_t)size.width, height = (ptrdiff_t)size.height;

    const uint8x16_t v_zero = vdupq_n_u8(0);
    const uint8x16_t v_border = vdupq_n_u8(borderValue);

    uint8x16_t tprev = v_zero, tcurr = v_zero, tnext = v_zero;
    uint8x16_t t0 = v_zero, t1 = v_zero, t2 = v_zero;

    for (ptrdiff_t y = 0; y < height; ++y)
    {
        const u8 * srow0 = y == 0 && border == BORDER_MODE_CONSTANT ? NULL : internal::getRowPtr(srcBase, srcStride, std::max<ptrdiff_t>(y - 1, 0));
        const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, y);
        const u8 * srow2 = y + 1 == height && border == BORDER_MODE_CONSTANT ? NULL : internal::getRowPtr(srcBase, srcStride, std::min(y + 1, height - 1));
        u8 * drow = internal::getRowPtr(dstBase, dstStride, y);

        u8 prevx = 0, currx = 0, nextx = 0;
        ptrdiff_t x = 0;
        const ptrdiff_t bwidth = y + 2 < height ? width : (width - 16);

        // perform vertical convolution
        for ( ; x <= bwidth; x += 16)
        {
            internal::prefetch(srow0 + x);
            internal::prefetch(srow1 + x);
            internal::prefetch(srow2 + x);

            uint8x16_t x0 = !srow0 ? v_border : vld1q_u8(srow0 + x);
            uint8x16_t x1 = vld1q_u8(srow1 + x);
            uint8x16_t x2 = !srow2 ? v_border : vld1q_u8(srow2 + x);

            // calculate values for plain CPU part below if needed
            if (x + 16 >= bwidth)
            {
                ptrdiff_t x3 = x == width ? width - 1 : x;
                ptrdiff_t x4 = border == BORDER_MODE_CONSTANT ? x3 - 1 : std::max<ptrdiff_t>(x3 - 1, 0);

                if (border == BORDER_MODE_CONSTANT && x4 < 0)
                    prevx = borderValue;
                else
                    prevx = vop(srow1[x4],
                                vop(srow2 ? srow2[x4] : borderValue,
                                    srow0 ? srow0[x4] : borderValue));

                currx = vop(srow2 ? srow2[x3] : borderValue, vop(srow1[x3], srow0 ? srow0[x3] : borderValue));
            }

            // make shift
            if (x)
            {
                tprev = tcurr;
                tcurr = tnext;
            }

            // and calculate next value
            tnext = vop(vop(x0, x1), x2);

            // make extrapolation for the first elements
            if (!x)
            {
                // make border
                if (border == BORDER_MODE_CONSTANT)
                    tcurr = v_border;
                else if (border == BORDER_MODE_REPLICATE)
                    tcurr = vdupq_n_u8(vgetq_lane_u8(tnext, 0));

                continue;
            }

            // combine 3 "shifted" vectors
            t0 = vextq_u8(tprev, tcurr, 15);
            t1 = tcurr;
            t2 = vextq_u8(tcurr, tnext, 1);

            // and add them
            t0 = vop(t0, vop(t1, t2));

            vst1q_u8(drow + x - 16, t0);
        }

        x -= 16;
        if (x == width)
            --x;

        for ( ; x < width; ++x)
        {
            // make extrapolation for the last elements
            if (x + 1 >= width)
            {
                if (border == BORDER_MODE_CONSTANT)
                    nextx = borderValue;
                else if (border == BORDER_MODE_REPLICATE)
                    nextx = vop(srow2[x], vop(srow1[x], srow0[x]));
            }
            else
                nextx = vop(vop(srow2 ? srow2[x + 1] : borderValue,
                                srow0 ? srow0[x + 1] : borderValue),
                            srow1[x + 1]);

            drow[x] = vop(prevx, vop(currx, nextx));

            // make shift
            prevx = currx;
            currx = nextx;
        }
    }
}

} // namespace

#endif

void erode3x3(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isMorph3x3Supported(size, border));
#ifdef CAROTENE_NEON
    morph3x3(size,
             srcBase, srcStride,
             dstBase, dstStride,
             border, ErodeVecOp(border, borderValue));
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

void dilate3x3(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride,
               BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isMorph3x3Supported(size, border));
#ifdef CAROTENE_NEON
    morph3x3(size,
             srcBase, srcStride,
             dstBase, dstStride,
             border, DilateVecOp(border, borderValue));
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

#ifdef CAROTENE_NEON
namespace {

template<class VecUpdate>
void MorphRow(const u8* src, u8* dst, size_t width, s32 cn, size_t ksize)
{
    size_t i, j, k;
    size_t width16 = (width & -16) * cn;
    size_t width8 = (width & -8) * cn;
    width *= cn;

    if (ksize == 1)
    {
        for (i = 0; i < width; i++)
            dst[i] = src[i];
        return;
    }

    ksize = ksize*cn;
    VecUpdate updateOp;
    switch(cn)
    {
    case 1:
        for (i = 0; i < width16; i += 16)
        {
            const u8* sptr = src + i;
            uint8x16_t s = vld1q_u8(sptr);
            internal::prefetch(sptr);

            for( k = 1; k < ksize; ++k)
                s = updateOp(s, vld1q_u8(sptr + k));

            vst1q_u8(dst + i, s);
        }

        for (; i < width8; i += 8)
        {
            const u8* sptr = src + i;
            uint8x8_t s = vld1_u8(sptr);
            internal::prefetch(sptr);

            for( k = 1; k < ksize; ++k)
                s = updateOp(s, vld1_u8(sptr + k));

            vst1_u8(dst + i, s);
        }
        break;
    default:
        for (i = 0; i < width16; i += 16)
        {
            uint8x16_t s = vld1q_u8(src + i);
            internal::prefetch(src + i);

            for (k = cn; k < ksize; k += cn)
                s = updateOp(s, vld1q_u8(src + i + k));

            vst1q_u8(dst + i, s);
        }

        for (; i < width8; i += 8)
        {
            uint8x8_t s = vld1_u8(src + i);
            internal::prefetch(src + i);

            for (k = cn; k < ksize; k += cn)
                s = updateOp(s, vld1_u8(src + i + k));

            vst1_u8(dst + i, s);
        }
        break;
    }

    ptrdiff_t i0 = i;
    for( k = 0; k < (size_t)cn; k++, src++, dst++ )
    {
        for( i = i0; i <= width - cn*2; i += cn*2 )
        {
            const u8* s = src + i;
            u8 m = s[cn];
            for( j = cn*2; j < ksize; j += cn )
                m = updateOp(m, s[j]);
            dst[i] = updateOp(m, s[0]);
            dst[i+cn] = updateOp(m, s[j]);
        }

        for( ; i < width; i += cn )
        {
            const u8* s = src + i;
            u8 m = s[0];
            for( j = cn; j < ksize; j += cn )
                m = updateOp(m, s[j]);
            dst[i] = m;
        }
    }
}

template<class VecUpdate>
void MorphColumn(const u8** src, u8* dst, ptrdiff_t dststep, size_t count, size_t width, size_t ksize)
{
    size_t i, k;
    size_t width32 = width & -32;
    VecUpdate updateOp;

    uint8x16_t x0,x1,s0,s1;
    if (ksize == 3)
    {
        for (; count > 1; count -= 2, dst += dststep * 2, src += 2)
        {
            for (i = 0; i < width32; i += 32)
            {
                const u8* sptr = src[1] + i;
                s0 = vld1q_u8(sptr);
                s1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);

                sptr = src[2] + i;
                x0 = vld1q_u8(sptr);
                x1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);

                s0 = updateOp(s0, x0);
                s1 = updateOp(s1, x1);

                sptr = src[0] + i;
                x0 = vld1q_u8(sptr);
                x1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);

                vst1q_u8(dst+i, updateOp(s0, x0));
                vst1q_u8(dst+i+16, updateOp(s1, x1));

                sptr = src[3] + i;
                x0 = vld1q_u8(sptr);
                x1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);
                vst1q_u8(dst + dststep + i, updateOp(s0, x0));
                vst1q_u8(dst + dststep + i + 16, updateOp(s1, x1));

            }
            for(; i < width; i++ )
            {
                u8 s = src[1][i];

                for( k = 2; k < ksize; k++ )
                    s = updateOp(s, src[k][i]);

                dst[i] = updateOp(s, src[0][i]);
                dst[i+dststep] = updateOp(s, src[k][i]);
            }
        }
    }
    else if (ksize > 1)
        for (; count > 1; count -= 2, dst += dststep*2, src += 2)
        {
            for (i = 0; i < width32; i += 32)
            {
                const u8* sptr = src[1] + i;
                s0 = vld1q_u8(sptr);
                s1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);
                for (k = 2; k < ksize; k++)
                {
                    sptr = src[k] + i;
                    x0 = vld1q_u8(sptr);
                    x1 = vld1q_u8(sptr + 16);
                    internal::prefetch(sptr);

                    s0 = updateOp(s0, x0);
                    s1 = updateOp(s1, x1);
                }

                sptr = src[0] + i;
                x0 = vld1q_u8(sptr);
                x1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);

                vst1q_u8(dst+i, updateOp(s0, x0));
                vst1q_u8(dst+i+16, updateOp(s1, x1));

                sptr = src[k] + i;
                x0 = vld1q_u8(sptr);
                x1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);
                vst1q_u8(dst + dststep + i, updateOp(s0, x0));
                vst1q_u8(dst + dststep + i + 16, updateOp(s1, x1));
            }
            for(; i < width; i++ )
            {
                u8 s = src[1][i];

                for( k = 2; k < ksize; k++ )
                    s = updateOp(s, src[k][i]);

                dst[i] = updateOp(s, src[0][i]);
                dst[i+dststep] = updateOp(s, src[k][i]);
            }
        }

    for (; count > 0; count--, dst += dststep, src++)
    {
        for (i = 0; i < width32; i += 32)
        {
            const u8* sptr = src[0] + i;
            s0 = vld1q_u8(sptr);
            s1 = vld1q_u8(sptr + 16);
            internal::prefetch(sptr);

            for (k = 1; k < ksize; k++)
            {
                sptr = src[k] + i;
                x0 = vld1q_u8(sptr);
                x1 = vld1q_u8(sptr + 16);
                internal::prefetch(sptr);
                s0 = updateOp(s0, x0);
                s1 = updateOp(s1, x1);
            }

            vst1q_u8(dst + i, s0);
            vst1q_u8(dst + i + 16, s1);
        }
        for(; i < width; i++ )
        {
            u8 s = src[0][i];
            for( k = 1; k < ksize; k++ )
                s = updateOp(s, src[k][i]);
            dst[i] = s;
        }
    }
}

template <class Op>
inline void morphology(const Size2D &ssize, u32 cn,
                       const u8 * srcBase, ptrdiff_t srcStride,
                       u8 * dstBase, ptrdiff_t dstStride,
                       const Size2D &ksize,
                       size_t anchorX, size_t anchorY,
                       BORDER_MODE rowBorderType, BORDER_MODE columnBorderType,
                       const u8 * borderValues, Margin borderMargin)
{
    //Temporary buffers common for all iterations
    std::vector<u8> _srcRow(cn*(ssize.width + ksize.width - 1));
    u8* srcRow = &_srcRow[0];

    size_t bufRows = std::max<size_t>(ksize.height + 3, std::max<size_t>(anchorY, ksize.height-anchorY-1)*2+1);
    std::vector<u8*> _rows(bufRows);
    u8** rows = &_rows[0];

    // adjust swidthcn so that the used part of buffers stays compact in memory
    ptrdiff_t swidthcn = cn*((ssize.width + 15) & -16);// cn * (aligned ssize.width size)
    std::vector<u8> _ringBuf(swidthcn*bufRows+16);
    u8 * ringBuf = internal::alignPtr(&_ringBuf[0], 16);

    size_t borderLength = std::max<size_t>(ksize.width - 1, 1) * cn;
    std::vector<ptrdiff_t> _borderTab(borderLength);
    ptrdiff_t * borderTab = &_borderTab[0];

    std::vector<u8> _constBorderValue;
    std::vector<u8> _constBorderRow;
    u8 * constBorderValue = NULL;
    u8 * constBorderRow = NULL;
    if( rowBorderType == BORDER_MODE_CONSTANT || columnBorderType == BORDER_MODE_CONSTANT )
    {
        _constBorderValue.resize(borderLength);
        constBorderValue = &_constBorderValue[0];
        size_t i;
        for(i = 0; i < cn; i++)
            constBorderValue[i] = borderValues[i];
        for(; i < borderLength; i++)
            constBorderValue[i] = constBorderValue[i-cn];

        if( columnBorderType == BORDER_MODE_CONSTANT )
        {
            _constBorderRow.resize(cn*(ssize.width + ksize.width - 1 + 16));
            constBorderRow = internal::alignPtr(&_constBorderRow[0], 16);
            size_t N = (ssize.width + ksize.width - 1)*cn;
            for( i = 0; i < N; i += borderLength )
            {
                size_t n = std::min( borderLength, N - i );
                for(size_t j = 0; j < n; j++)
                    srcRow[i+j] = constBorderValue[j];
            }
            MorphRow<Op>(srcRow, constBorderRow, ssize.width, cn, ksize.width);
        }
    }

    Size2D wholeSize(ssize.width + borderMargin.left + borderMargin.right,
                     ssize.height + borderMargin.top + borderMargin.bottom);

    ptrdiff_t dx1 = std::max<ptrdiff_t>(anchorX - (ptrdiff_t)borderMargin.left, 0);
    ptrdiff_t dx2 = std::max<ptrdiff_t>((ptrdiff_t)ksize.width - anchorX - 1 - (ptrdiff_t)borderMargin.right, 0);
    // recompute border tables
    if( dx1 > 0 || dx2 > 0 )
    {
        if( rowBorderType == BORDER_MODE_CONSTANT )
        {
            memcpy( srcRow, &constBorderValue[0], dx1*cn );
            memcpy( srcRow + (ssize.width + ksize.width - 1 - dx2)*cn, &constBorderValue[0], dx2*cn );
        }
        else
        {
            ptrdiff_t xofs1 = std::min<ptrdiff_t>(borderMargin.left, anchorX) - borderMargin.left;

            ptrdiff_t wholeWidth = wholeSize.width;

            ptrdiff_t i, j;
            for( i = 0; i < dx1; i++ )
            {
                ptrdiff_t p0 = (internal::borderInterpolate(i-dx1, wholeWidth, rowBorderType) + xofs1)*cn;
                for( j = 0; j < (ptrdiff_t)cn; j++ )
                    borderTab[i*cn + j] = p0 + j;
            }

            for( i = 0; i < dx2; i++ )
            {
                ptrdiff_t p0 = (internal::borderInterpolate(wholeWidth + i, wholeWidth, rowBorderType) + xofs1)*cn;
                for( j = 0; j < (ptrdiff_t)cn; j++ )
                    borderTab[(i + dx1)*cn + j] = p0 + j;
            }
        }
    }

    ptrdiff_t startY, startY0, endY, rowCount;
    startY = startY0 = std::max<ptrdiff_t>(borderMargin.top - anchorY, 0);
    endY = std::min<ptrdiff_t>(borderMargin.top + ssize.height + ksize.height - anchorY - 1, wholeSize.height);

    const u8* src = srcBase + (startY - borderMargin.top)*srcStride;
    u8* dst = dstBase;

    ptrdiff_t width = ssize.width, kwidth = ksize.width;
    ptrdiff_t kheight = ksize.height, ay = anchorY;
    ptrdiff_t width1 = ssize.width + kwidth - 1;
    ptrdiff_t xofs1 = std::min<ptrdiff_t>(borderMargin.left, anchorX);
    bool makeBorder = (dx1 > 0 || dx2 > 0) && rowBorderType != BORDER_MODE_CONSTANT;
    ptrdiff_t dy = 0, i = 0;

    src -= xofs1*cn;
    ptrdiff_t count = endY - startY;

    rowCount = 0;
    for(;; dst += dstStride*i, dy += i)
    {
        ptrdiff_t dcount = bufRows - ay - startY - rowCount + borderMargin.top;
        dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for( ; dcount-- > 0; src += srcStride )
        {
            ptrdiff_t bi = (startY - startY0 + rowCount) % bufRows;
            u8* brow = ringBuf + bi*swidthcn;

            if( (size_t)(++rowCount) > bufRows )
            {
                --rowCount;
                ++startY;
            }

            memcpy( srcRow + dx1*cn, src, (width1 - dx2 - dx1)*cn );

            if( makeBorder )
            {
                    for( i = 0; i < (ptrdiff_t)(dx1*cn); i++ )
                        srcRow[i] = src[borderTab[i]];
                    for( i = 0; i < (ptrdiff_t)(dx2*cn); i++ )
                        srcRow[i + (width1 - dx2)*cn] = src[borderTab[i+dx1*cn]];
            }

            MorphRow<Op>(srcRow, brow, width, cn, ksize.width);
        }

        ptrdiff_t max_i = std::min<ptrdiff_t>(bufRows, ssize.height - dy + (kheight - 1));
        for( i = 0; i < max_i; i++ )
        {
            ptrdiff_t srcY = internal::borderInterpolate(dy + i + borderMargin.top - ay,
                                               wholeSize.height, columnBorderType);
            if( srcY < 0 ) // can happen only with constant border type
                rows[i] = constBorderRow;
            else
            {
                if( srcY >= startY + rowCount )
                    break;
                ptrdiff_t bi = (srcY - startY0) % bufRows;
                rows[i] = ringBuf + bi*swidthcn;
            }
        }
        if( i < kheight )
            break;
        i -= kheight - 1;
        MorphColumn<Op>((const u8**)rows, dst, dstStride, i, ssize.width*cn, ksize.height);
    }
}

} // namespace
#endif // CAROTENE_NEON

void erode(const Size2D &ssize, u32 cn,
           const u8 * srcBase, ptrdiff_t srcStride,
           u8 * dstBase, ptrdiff_t dstStride,
           const Size2D &ksize,
           size_t anchorX, size_t anchorY,
           BORDER_MODE rowBorderType, BORDER_MODE columnBorderType,
           const u8 * borderValues, Margin borderMargin)
{
    internal::assertSupportedConfiguration(ssize.width > 0 && ssize.height > 0 &&
                                           anchorX < ksize.width && anchorY < ksize.height);
#ifdef CAROTENE_NEON
    morphology<ErodeVecOp>(ssize, cn, srcBase, srcStride, dstBase, dstStride,
                           ksize, anchorX, anchorY, rowBorderType, columnBorderType,
                           borderValues, borderMargin);
#else
    (void)cn;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)rowBorderType;
    (void)columnBorderType;
    (void)borderValues;
    (void)borderMargin;
#endif
}

void dilate(const Size2D &ssize, u32 cn,
            const u8 * srcBase, ptrdiff_t srcStride,
            u8 * dstBase, ptrdiff_t dstStride,
            const Size2D &ksize,
            size_t anchorX, size_t anchorY,
            BORDER_MODE rowBorderType, BORDER_MODE columnBorderType,
            const u8 * borderValues, Margin borderMargin)
{
    internal::assertSupportedConfiguration(ssize.width > 0 && ssize.height > 0 &&
                                           anchorX < ksize.width && anchorY < ksize.height);
#ifdef CAROTENE_NEON
    morphology<DilateVecOp>(ssize, cn, srcBase, srcStride, dstBase, dstStride,
                            ksize, anchorX, anchorY, rowBorderType, columnBorderType,
                            borderValues, borderMargin);
#else
    (void)cn;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)rowBorderType;
    (void)columnBorderType;
    (void)borderValues;
    (void)borderMargin;
#endif
}

} // namespace CAROTENE_NS
