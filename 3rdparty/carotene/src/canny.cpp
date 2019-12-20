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
#include <vector>
#include <cstring>

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON
namespace {
struct RowFilter3x3Canny
{
    inline RowFilter3x3Canny(const ptrdiff_t borderxl, const ptrdiff_t borderxr)
    {
        vfmask = vreinterpret_u8_u64(vmov_n_u64(borderxl ? 0x0000FFffFFffFFffULL : 0x0100FFffFFffFFffULL));
        vtmask = vreinterpret_u8_u64(vmov_n_u64(borderxr ? 0x0707060504030201ULL : 0x0706050403020100ULL));
        lookLeft = offsetk - borderxl;
        lookRight = offsetk - borderxr;
    }

    inline void operator()(const u8* src, s16* dstx, s16* dsty, ptrdiff_t width)
    {
        uint8x8_t l = vtbl1_u8(vld1_u8(src - lookLeft), vfmask);
        ptrdiff_t i = 0;
        for (; i < width - 8 + lookRight; i += 8)
        {
            internal::prefetch(src + i);
            uint8x8_t l18u = vld1_u8(src + i + 1);

            uint8x8_t l2 = l18u;
            uint8x8_t l0 = vext_u8(l, l18u, 6);
            int16x8_t l1x2 = vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l, l18u, 7), 1));

            l = l18u;

            int16x8_t l02 = vreinterpretq_s16_u16(vaddl_u8(l2, l0));
            int16x8_t ldx = vreinterpretq_s16_u16(vsubl_u8(l2, l0));
            int16x8_t ldy = vaddq_s16(l02, l1x2);

            vst1q_s16(dstx + i, ldx);
            vst1q_s16(dsty + i, ldy);
        }

        //tail
        if (lookRight == 0 || i != width)
        {
            uint8x8_t tail0 = vld1_u8(src + (width - 9));//can't get left 1 pixel another way if width==8*k+1
            uint8x8_t tail2 = vtbl1_u8(vld1_u8(src + (width - 8 + lookRight)), vtmask);
            uint8x8_t tail1 = vext_u8(vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(tail0), 8*6)), tail2, 7);

            int16x8_t tail02 = vreinterpretq_s16_u16(vaddl_u8(tail2, tail0));
            int16x8_t tail1x2 = vreinterpretq_s16_u16(vshll_n_u8(tail1, 1));
            int16x8_t taildx = vreinterpretq_s16_u16(vsubl_u8(tail2, tail0));
            int16x8_t taildy = vqaddq_s16(tail02, tail1x2);

            vst1q_s16(dstx + (width - 8), taildx);
            vst1q_s16(dsty + (width - 8), taildy);
        }
    }

    uint8x8_t vfmask;
    uint8x8_t vtmask;
    enum { offsetk = 1};
    ptrdiff_t lookLeft;
    ptrdiff_t lookRight;
};

template <bool L2gradient>
inline void ColFilter3x3Canny(const s16* src0, const s16* src1, const s16* src2, s16* dstx, s16* dsty, s32* mag, ptrdiff_t width)
{
    ptrdiff_t j = 0;
    for (; j <= width - 8; j += 8)
    {
        ColFilter3x3CannyL1Loop:
        int16x8_t line0x = vld1q_s16(src0 + j);
        int16x8_t line1x = vld1q_s16(src1 + j);
        int16x8_t line2x = vld1q_s16(src2 + j);
        int16x8_t line0y = vld1q_s16(src0 + j + width);
        int16x8_t line2y = vld1q_s16(src2 + j + width);

        int16x8_t l02 = vaddq_s16(line0x, line2x);
        int16x8_t l1x2 = vshlq_n_s16(line1x, 1);
        int16x8_t dy = vsubq_s16(line2y, line0y);
        int16x8_t dx = vaddq_s16(l1x2, l02);

        int16x8_t dya = vabsq_s16(dy);
        int16x8_t dxa = vabsq_s16(dx);
        int16x8_t norm = vaddq_s16(dya, dxa);

        int32x4_t normh = vmovl_s16(vget_high_s16(norm));
        int32x4_t norml = vmovl_s16(vget_low_s16(norm));

        vst1q_s16(dsty + j, dy);
        vst1q_s16(dstx + j, dx);
        vst1q_s32(mag + j + 4, normh);
        vst1q_s32(mag + j, norml);
    }
    if (j != width)
    {
        j = width - 8;
        goto ColFilter3x3CannyL1Loop;
    }
}
template <>
inline void ColFilter3x3Canny<true>(const s16* src0, const s16* src1, const s16* src2, s16* dstx, s16* dsty, s32* mag, ptrdiff_t width)
{
    ptrdiff_t j = 0;
    for (; j <= width - 8; j += 8)
    {
        ColFilter3x3CannyL2Loop:
        int16x8_t line0x = vld1q_s16(src0 + j);
        int16x8_t line1x = vld1q_s16(src1 + j);
        int16x8_t line2x = vld1q_s16(src2 + j);
        int16x8_t line0y = vld1q_s16(src0 + j + width);
        int16x8_t line2y = vld1q_s16(src2 + j + width);

        int16x8_t l02 = vaddq_s16(line0x, line2x);
        int16x8_t l1x2 = vshlq_n_s16(line1x, 1);
        int16x8_t dy = vsubq_s16(line2y, line0y);
        int16x8_t dx = vaddq_s16(l1x2, l02);

        int32x4_t norml = vmull_s16(vget_low_s16(dx), vget_low_s16(dx));
        int32x4_t normh = vmull_s16(vget_high_s16(dy), vget_high_s16(dy));

        norml = vmlal_s16(norml, vget_low_s16(dy), vget_low_s16(dy));
        normh = vmlal_s16(normh, vget_high_s16(dx), vget_high_s16(dx));

        vst1q_s16(dsty + j, dy);
        vst1q_s16(dstx + j, dx);
        vst1q_s32(mag + j, norml);
        vst1q_s32(mag + j + 4, normh);
    }
    if (j != width)
    {
        j = width - 8;
        goto ColFilter3x3CannyL2Loop;
    }
}

template <bool L2gradient>
inline void NormCanny(const ptrdiff_t colscn, s16* _dx, s16* _dy, s32* _norm)
{
    ptrdiff_t j = 0;
    if (colscn >= 8)
    {
        int16x8_t vx = vld1q_s16(_dx);
        int16x8_t vy = vld1q_s16(_dy);
        for (; j <= colscn - 16; j+=8)
        {
            internal::prefetch(_dx);
            internal::prefetch(_dy);

            int16x8_t vx2 = vld1q_s16(_dx + j + 8);
            int16x8_t vy2 = vld1q_s16(_dy + j + 8);

            int16x8_t vabsx = vabsq_s16(vx);
            int16x8_t vabsy = vabsq_s16(vy);

            int16x8_t norm = vaddq_s16(vabsx, vabsy);

            int32x4_t normh = vmovl_s16(vget_high_s16(norm));
            int32x4_t norml = vmovl_s16(vget_low_s16(norm));

            vst1q_s32(_norm + j + 4, normh);
            vst1q_s32(_norm + j + 0, norml);

            vx = vx2;
            vy = vy2;
        }
        int16x8_t vabsx = vabsq_s16(vx);
        int16x8_t vabsy = vabsq_s16(vy);

        int16x8_t norm = vaddq_s16(vabsx, vabsy);

        int32x4_t normh = vmovl_s16(vget_high_s16(norm));
        int32x4_t norml = vmovl_s16(vget_low_s16(norm));

        vst1q_s32(_norm + j + 4, normh);
        vst1q_s32(_norm + j + 0, norml);
    }
    for (; j < colscn; j++)
        _norm[j] = std::abs(s32(_dx[j])) + std::abs(s32(_dy[j]));
}

template <>
inline void NormCanny<true>(const ptrdiff_t colscn, s16* _dx, s16* _dy, s32* _norm)
{
    ptrdiff_t j = 0;
    if (colscn >= 8)
    {
        int16x8_t vx = vld1q_s16(_dx);
        int16x8_t vy = vld1q_s16(_dy);

        for (; j <= colscn - 16; j+=8)
        {
            internal::prefetch(_dx);
            internal::prefetch(_dy);

            int16x8_t vxnext = vld1q_s16(_dx + j + 8);
            int16x8_t vynext = vld1q_s16(_dy + j + 8);

            int32x4_t norml = vmull_s16(vget_low_s16(vx), vget_low_s16(vx));
            int32x4_t normh = vmull_s16(vget_high_s16(vy), vget_high_s16(vy));

            norml = vmlal_s16(norml, vget_low_s16(vy), vget_low_s16(vy));
            normh = vmlal_s16(normh, vget_high_s16(vx), vget_high_s16(vx));

            vst1q_s32(_norm + j + 0, norml);
            vst1q_s32(_norm + j + 4, normh);

            vx = vxnext;
            vy = vynext;
        }
        int32x4_t norml = vmull_s16(vget_low_s16(vx), vget_low_s16(vx));
        int32x4_t normh = vmull_s16(vget_high_s16(vy), vget_high_s16(vy));

        norml = vmlal_s16(norml, vget_low_s16(vy), vget_low_s16(vy));
        normh = vmlal_s16(normh, vget_high_s16(vx), vget_high_s16(vx));

        vst1q_s32(_norm + j + 0, norml);
        vst1q_s32(_norm + j + 4, normh);
    }
    for (; j < colscn; j++)
        _norm[j] = s32(_dx[j])*_dx[j] + s32(_dy[j])*_dy[j];
}

template <bool L2gradient>
inline void prepareThresh(f64 low_thresh, f64 high_thresh,
                          s32 &low, s32 &high)
{
    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);
#if defined __GNUC__
    low = (s32)low_thresh;
    high = (s32)high_thresh;
    low -= (low > low_thresh);
    high -= (high > high_thresh);
#else
    low = internal::round(low_thresh);
    high = internal::round(high_thresh);
    f32 ldiff = (f32)(low_thresh - low);
    f32 hdiff = (f32)(high_thresh - high);
    low -= (ldiff < 0);
    high -= (hdiff < 0);
#endif
}
template <>
inline void prepareThresh<true>(f64 low_thresh, f64 high_thresh,
                                s32 &low, s32 &high)
{
    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);
    if (low_thresh > 0) low_thresh *= low_thresh;
    if (high_thresh > 0) high_thresh *= high_thresh;
#if defined __GNUC__
    low = (s32)low_thresh;
    high = (s32)high_thresh;
    low -= (low > low_thresh);
    high -= (high > high_thresh);
#else
    low = internal::round(low_thresh);
    high = internal::round(high_thresh);
    f32 ldiff = (f32)(low_thresh - low);
    f32 hdiff = (f32)(high_thresh - high);
    low -= (ldiff < 0);
    high -= (hdiff < 0);
#endif
}

template <bool L2gradient, bool externalSobel>
struct _normEstimator
{
    ptrdiff_t magstep;
    ptrdiff_t dxOffset;
    ptrdiff_t dyOffset;
    ptrdiff_t shxOffset;
    ptrdiff_t shyOffset;
    std::vector<u8> buffer;
    const ptrdiff_t offsetk;
    ptrdiff_t borderyt, borderyb;
    RowFilter3x3Canny sobelRow;

    inline _normEstimator(const Size2D &size, s32, Margin borderMargin,
                          ptrdiff_t &mapstep, s32** mag_buf, u8* &map):
                          offsetk(1),
                          sobelRow(std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.left),
                                   std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.right))
    {
        mapstep = size.width + 2;
        magstep = size.width + 2 + size.width * (4 * sizeof(s16)/sizeof(s32));
        dxOffset = mapstep * sizeof(s32)/sizeof(s16);
        dyOffset = dxOffset + size.width * 1;
        shxOffset = dxOffset + size.width * 2;
        shyOffset = dxOffset + size.width * 3;
        buffer.resize( (size.width+2)*(size.height+2) + magstep*3*sizeof(s32) );
        mag_buf[0] = (s32*)&buffer[0];
        mag_buf[1] = mag_buf[0] + magstep;
        mag_buf[2] = mag_buf[1] + magstep;
        memset(mag_buf[0], 0, mapstep * sizeof(s32));

        map = (u8*)(mag_buf[2] + magstep);
        memset(map, 1, mapstep);
        memset(map + mapstep*(size.height + 1), 1, mapstep);
        borderyt = std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.top);
        borderyb = std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.bottom);
    }
    inline void firstRow(const Size2D &size, s32,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         s16*, ptrdiff_t,
                         s16*, ptrdiff_t,
                         s32** mag_buf)
    {
        //sobelH row #0
        const u8* _src = internal::getRowPtr(srcBase, srcStride, 0);
        sobelRow(_src, ((s16*)mag_buf[0]) + shxOffset, ((s16*)mag_buf[0]) + shyOffset, size.width);
        //sobelH row #1
        _src = internal::getRowPtr(srcBase, srcStride, 1);
        sobelRow(_src, ((s16*)mag_buf[1]) + shxOffset, ((s16*)mag_buf[1]) + shyOffset, size.width);

        mag_buf[1][0] = mag_buf[1][size.width+1] = 0;
        if (borderyt == 0)
        {
            //sobelH row #-1
            _src = internal::getRowPtr(srcBase, srcStride, -1);
            sobelRow(_src, ((s16*)mag_buf[2]) + shxOffset, ((s16*)mag_buf[2]) + shyOffset, size.width);

            ColFilter3x3Canny<L2gradient>( ((s16*)mag_buf[2]) + shxOffset, ((s16*)mag_buf[0]) + shxOffset, ((s16*)mag_buf[1]) + shxOffset,
                                           ((s16*)mag_buf[1]) + dxOffset,  ((s16*)mag_buf[1]) + dyOffset, mag_buf[1] + 1, size.width);
        }
        else
        {
            ColFilter3x3Canny<L2gradient>( ((s16*)mag_buf[0]) + shxOffset, ((s16*)mag_buf[0]) + shxOffset, ((s16*)mag_buf[1]) + shxOffset,
                                           ((s16*)mag_buf[1]) + dxOffset,  ((s16*)mag_buf[1]) + dyOffset, mag_buf[1] + 1, size.width);
        }
    }
    inline void nextRow(const Size2D &size, s32,
                        const u8 *srcBase, ptrdiff_t srcStride,
                        s16*, ptrdiff_t,
                        s16*, ptrdiff_t,
                        const ptrdiff_t &mapstep, s32** mag_buf,
                        size_t i, const s16* &_x, const s16* &_y)
    {
        mag_buf[2][0] = mag_buf[2][size.width+1] = 0;
        if (i < size.height - borderyb)
        {
            const u8* _src = internal::getRowPtr(srcBase, srcStride, i+1);
            //sobelH row #i+1
            sobelRow(_src, ((s16*)mag_buf[2]) + shxOffset, ((s16*)mag_buf[2]) + shyOffset, size.width);

            ColFilter3x3Canny<L2gradient>( ((s16*)mag_buf[0]) + shxOffset, ((s16*)mag_buf[1]) + shxOffset, ((s16*)mag_buf[2]) + shxOffset,
                                           ((s16*)mag_buf[2]) + dxOffset,  ((s16*)mag_buf[2]) + dyOffset, mag_buf[2] + 1, size.width);
        }
        else if (i < size.height)
        {
            ColFilter3x3Canny<L2gradient>( ((s16*)mag_buf[0]) + shxOffset, ((s16*)mag_buf[1]) + shxOffset, ((s16*)mag_buf[1]) + shxOffset,
                                           ((s16*)mag_buf[2]) + dxOffset,  ((s16*)mag_buf[2]) + dyOffset, mag_buf[2] + 1, size.width);
        }
        else
            memset(mag_buf[2], 0, mapstep*sizeof(s32));
        _x = ((s16*)mag_buf[1]) + dxOffset;
        _y = ((s16*)mag_buf[1]) + dyOffset;
    }
};
template <bool L2gradient>
struct _normEstimator<L2gradient, true>
{
    std::vector<u8> buffer;

    inline _normEstimator(const Size2D &size, s32 cn, Margin,
                          ptrdiff_t &mapstep, s32** mag_buf, u8* &map)
    {
        mapstep = size.width + 2;
        buffer.resize( (size.width+2)*(size.height+2) + cn*mapstep*3*sizeof(s32) );
        mag_buf[0] = (s32*)&buffer[0];
        mag_buf[1] = mag_buf[0] + mapstep*cn;
        mag_buf[2] = mag_buf[1] + mapstep*cn;
        memset(mag_buf[0], 0, /* cn* */mapstep * sizeof(s32));

        map = (u8*)(mag_buf[2] + mapstep*cn);
        memset(map, 1, mapstep);
        memset(map + mapstep*(size.height + 1), 1, mapstep);
    }
    inline void firstRow(const Size2D &size, s32 cn,
                         const u8 *, ptrdiff_t,
                         s16* dxBase, ptrdiff_t dxStride,
                         s16* dyBase, ptrdiff_t dyStride,
                         s32** mag_buf)
    {
        s32* _norm = mag_buf[1] + 1;

        s16* _dx = internal::getRowPtr(dxBase, dxStride, 0);
        s16* _dy = internal::getRowPtr(dyBase, dyStride, 0);

        NormCanny<L2gradient>(size.width*cn, _dx, _dy, _norm);

        if(cn > 1)
        {
            for(size_t j = 0, jn = 0; j < size.width; ++j, jn += cn)
            {
                size_t maxIdx = jn;
                for(s32 k = 1; k < cn; ++k)
                    if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                _norm[j] = _norm[maxIdx];
                _dx[j] = _dx[maxIdx];
                _dy[j] = _dy[maxIdx];
            }
        }

        _norm[-1] = _norm[size.width] = 0;
    }
    inline void nextRow(const Size2D &size, s32 cn,
                        const u8 *, ptrdiff_t,
                        s16* dxBase, ptrdiff_t dxStride,
                        s16* dyBase, ptrdiff_t dyStride,
                        const ptrdiff_t &mapstep, s32** mag_buf,
                        size_t i, const s16* &_x, const s16* &_y)
    {
        s32* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < size.height)
        {
            s16* _dx = internal::getRowPtr(dxBase, dxStride, i);
            s16* _dy = internal::getRowPtr(dyBase, dyStride, i);

            NormCanny<L2gradient>(size.width*cn, _dx, _dy, _norm);

            if(cn > 1)
            {
                for(size_t j = 0, jn = 0; j < size.width; ++j, jn += cn)
                {
                    size_t maxIdx = jn;
                    for(s32 k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }

            _norm[-1] = _norm[size.width] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(s32));

        _x = internal::getRowPtr(dxBase, dxStride, i-1);
        _y = internal::getRowPtr(dyBase, dyStride, i-1);
    }
};

template <bool L2gradient, bool externalSobel>
inline void Canny3x3(const Size2D &size, s32 cn,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     s16 * dxBase, ptrdiff_t dxStride,
                     s16 * dyBase, ptrdiff_t dyStride,
                     f64 low_thresh, f64 high_thresh,
                     Margin borderMargin)
{
    s32 low, high;
    prepareThresh<L2gradient>(low_thresh, high_thresh, low, high);

    ptrdiff_t mapstep;
    s32* mag_buf[3];
    u8* map;
    _normEstimator<L2gradient, externalSobel> normEstimator(size, cn, borderMargin, mapstep, mag_buf, map);

    size_t maxsize = std::max<size_t>( 1u << 10, size.width * size.height / 10 );
    std::vector<u8*> stack( maxsize );
    u8 **stack_top = &stack[0];
    u8 **stack_bottom = &stack[0];

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = u8(2), *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    //i == 0
    normEstimator.firstRow(size, cn, srcBase, srcStride, dxBase, dxStride, dyBase, dyStride, mag_buf);
    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (size_t i = 1; i <= size.height; i++)
    {
        const s16 *_x, *_y;
        normEstimator.nextRow(size, cn, srcBase, srcStride, dxBase, dxStride, dyBase, dyStride, mapstep, mag_buf, i, _x, _y);

        u8* _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;

        s32* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        if ((stack_top - stack_bottom) + size.width > maxsize)
        {
            ptrdiff_t sz = (ptrdiff_t)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        s32 prev_flag = 0;
        for (ptrdiff_t j = 0; j < (ptrdiff_t)size.width; j++)
        {
            #define CANNY_SHIFT 15
            const s32 TG22 = (s32)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

            s32 m = _mag[j];

            if (m > low)
            {
                s32 xs = _x[j];
                s32 ys = _y[j];
                s32 x = abs(xs);
                s32 y = abs(ys) << CANNY_SHIFT;

                s32 tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j-1] && m >= _mag[j+1]) goto __push;
                }
                else
                {
                    s32 tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto __push;
                    }
                    else
                    {
                        s32 s = (xs ^ ys) < 0 ? -1 : 1;
                        if(m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto __push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = u8(1);
            continue;
            __push:
            if (!prev_flag && m > high && _map[j-mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        u8* m;
        if ((size_t)(stack_top - stack_bottom) + 8u > maxsize)
        {
            ptrdiff_t sz = (ptrdiff_t)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    uint8x16_t v2 = vmovq_n_u8(2);
    const u8* ptrmap = map + mapstep + 1;
    for (size_t i = 0; i < size.height; i++, ptrmap += mapstep)
    {
        u8* _dst = internal::getRowPtr(dstBase, dstStride, i);
        ptrdiff_t j = 0;
        for (; j < (ptrdiff_t)size.width - 16; j += 16)
        {
            internal::prefetch(ptrmap);
            uint8x16_t vmap = vld1q_u8(ptrmap + j);
            uint8x16_t vdst = vceqq_u8(vmap, v2);
            vst1q_u8(_dst+j, vdst);
        }
        for (; j < (ptrdiff_t)size.width; j++)
            _dst[j] = (u8)-(ptrmap[j] >> 1);
    }
}

} // namespace
#endif

bool isCanny3x3Supported(const Size2D &size)
{
    return isSupportedConfiguration() &&
           size.height >= 2 && size.width >= 9;
}

void Canny3x3L1(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                f64 low_thresh, f64 high_thresh,
                Margin borderMargin)
{
    internal::assertSupportedConfiguration(isCanny3x3Supported(size));
#ifdef CAROTENE_NEON
    Canny3x3<false, false>(size, 1,
                           srcBase, srcStride,
                           dstBase, dstStride,
                           NULL, 0,
                           NULL, 0,
                           low_thresh, high_thresh,
                           borderMargin);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)low_thresh;
    (void)high_thresh;
    (void)borderMargin;
#endif
}

void Canny3x3L2(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                f64 low_thresh, f64 high_thresh,
                Margin borderMargin)
{
    internal::assertSupportedConfiguration(isCanny3x3Supported(size));
#ifdef CAROTENE_NEON
    Canny3x3<true, false>(size, 1,
                          srcBase, srcStride,
                          dstBase, dstStride,
                          NULL, 0,
                          NULL, 0,
                          low_thresh, high_thresh,
                          borderMargin);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)low_thresh;
    (void)high_thresh;
    (void)borderMargin;
#endif
}

void Canny3x3L1(const Size2D &size, s32 cn,
                     s16 * dxBase, ptrdiff_t dxStride,
                     s16 * dyBase, ptrdiff_t dyStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     f64 low_thresh, f64 high_thresh)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Canny3x3<false, true>(size, cn,
                          NULL, 0,
                          dstBase, dstStride,
                          dxBase, dxStride,
                          dyBase, dyStride,
                          low_thresh, high_thresh,
                          Margin());
#else
    (void)size;
    (void)cn;
    (void)dstBase;
    (void)dstStride;
    (void)dxBase;
    (void)dxStride;
    (void)dyBase;
    (void)dyStride;
    (void)low_thresh;
    (void)high_thresh;
#endif
}

void Canny3x3L2(const Size2D &size, s32 cn,
                     s16 * dxBase, ptrdiff_t dxStride,
                     s16 * dyBase, ptrdiff_t dyStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     f64 low_thresh, f64 high_thresh)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Canny3x3<true, true>(size, cn,
                         NULL, 0,
                         dstBase, dstStride,
                         dxBase, dxStride,
                         dyBase, dyStride,
                         low_thresh, high_thresh,
                         Margin());
#else
    (void)size;
    (void)cn;
    (void)dstBase;
    (void)dstStride;
    (void)dxBase;
    (void)dxStride;
    (void)dyBase;
    (void)dyStride;
    (void)low_thresh;
    (void)high_thresh;
#endif
}

} // namespace CAROTENE_NS
