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
 * Copyright (C) 2014-2015, NVIDIA Corporation, all rights reserved.
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

#ifndef CAROTENE_SRC_SEPARABLE_FILTER_HPP
#define CAROTENE_SRC_SEPARABLE_FILTER_HPP

#include "common.hpp"

#include <carotene/types.hpp>

#include <vector>

#ifdef CAROTENE_NEON

namespace CAROTENE_NS {

namespace internal {

struct RowFilter3x3S16Base
{
    typedef u8 srcType;
     /*
     Various border types, image boundaries are denoted with '|'

     * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
     * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
     * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
     * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
     * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
     */
    inline RowFilter3x3S16Base(const BORDER_MODE _borderType, const srcType _borderValue, const ptrdiff_t borderxl, const ptrdiff_t borderxr):
                               borderType(_borderType),borderValue(_borderValue)
    {
        if (borderType == BORDER_MODE_CONSTANT)
        {
            vfmask = vreinterpret_u8_u64(vmov_n_u64(borderxl ? 0x00ffFFffFFffFFffULL : 0x0100FFffFFffFFffULL));
            vtmask = vreinterpret_u8_u64(vmov_n_u64(borderxr ? 0xFF07060504030201ULL : 0x0706050403020100ULL));
        }
        else if (borderType == BORDER_MODE_REFLECT101)
        {
            vfmask = vreinterpret_u8_u64(vmov_n_u64(borderxl ? 0x0001FFffFFffFFffULL : 0x0100FFffFFffFFffULL));
            vtmask = vreinterpret_u8_u64(vmov_n_u64(borderxr ? 0x0607060504030201ULL : 0x0706050403020100ULL));
        }
        else //if (borderType == BORDER_MODE_REFLECT || borderType == BORDER_MODE_REPLICATE)
        {
            vfmask = vreinterpret_u8_u64(vmov_n_u64(borderxl ? 0x0000FFffFFffFFffULL : 0x0100FFffFFffFFffULL));
            vtmask = vreinterpret_u8_u64(vmov_n_u64(borderxr ? 0x0707060504030201ULL : 0x0706050403020100ULL));
        }
        lookLeft = offsetk - borderxl;
        lookRight = offsetk - borderxr;
    }

    uint8x8_t vfmask;
    uint8x8_t vtmask;
    enum { offsetk = 1};
    ptrdiff_t lookLeft;
    ptrdiff_t lookRight;
    const BORDER_MODE borderType;
    const srcType borderValue;
};

struct ColFilter3x3S16Base
{
    typedef s16 srcType;

    inline ColFilter3x3S16Base(const BORDER_MODE _borderType, const srcType _borderValue):
                               borderType(_borderType),borderValue(_borderValue) {}

    enum { offsetk = 1};
    const BORDER_MODE borderType;
    const srcType borderValue;
};

struct RowFilter3x3S16Generic : public RowFilter3x3S16Base
{
    typedef s16 dstType;

    inline RowFilter3x3S16Generic(BORDER_MODE _borderType, const srcType _borderValue, ptrdiff_t borderxl, ptrdiff_t borderxr, const s16 *w):
                                  RowFilter3x3S16Base(_borderType, _borderValue, borderxl, borderxr), borderFilter( (w[0]+w[1]+w[2]) * borderValue )
    {
        vw0 = vdupq_n_s16(w[0]);
        vw1 = vdupq_n_s16(w[1]);
        vw2 = vdupq_n_s16(w[2]);
    }

    int16x8_t vw0;
    int16x8_t vw1;
    int16x8_t vw2;
    const dstType borderFilter;

    inline void operator()(const u8* src, s16* dst, ptrdiff_t width)
    {
        uint8x8_t l = vtbl1_u8(vld1_u8(src - lookLeft), vfmask);
        if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
            l = vset_lane_u8(borderValue, l, 6);

        ptrdiff_t i = 0;
        for (; i < width - 16 + lookRight; i += 16)
        {
            internal::prefetch(src + i);
            uint8x8_t l18u = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vaddq_s16(vmlaq_s16(vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vext_u8(l, l18u, 6))), vw0),
                                                   vreinterpretq_s16_u16(vmovl_u8(vext_u8(l, l18u, 7))), vw1),
                                                   vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(l18u)), vw2)));
            l = vld1_u8(src + i + 9);
            vst1q_s16(dst + i + 8, vaddq_s16(vmlaq_s16(vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vext_u8(l18u, l, 6))), vw0),
                                                   vreinterpretq_s16_u16(vmovl_u8(vext_u8(l18u, l, 7))), vw1),
                                                   vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(l)), vw2)));
        }
        if (i < width - 8 + lookRight)
        {
            uint8x8_t l18u = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vaddq_s16(vmlaq_s16(vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vext_u8(l, l18u, 6))), vw0),
                                                   vreinterpretq_s16_u16(vmovl_u8(vext_u8(l, l18u, 7))), vw1),
                                                   vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(l18u)), vw2)));
            i += 8;
        }

        //tail
        if (lookRight == 0 || i != width)
        {
            uint8x8_t tail0 = vld1_u8(src + (width - 9));//can't get left 1 pixel another way if width==8*k+1
            uint8x8_t tail2 = vtbl1_u8(vld1_u8(src + (width - 8 + lookRight)), vtmask);
            if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
                tail2 = vset_lane_u8(borderValue, tail2, 7);
            uint8x8_t tail1 = vext_u8(vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(tail0), 8*6)), tail2, 7);

            int16x8_t l0 = vreinterpretq_s16_u16(vmovl_u8(tail0));
            int16x8_t l1 = vreinterpretq_s16_u16(vmovl_u8(tail1));
            int16x8_t l2 = vreinterpretq_s16_u16(vmovl_u8(tail2));

            int16x8_t l0w = vmulq_s16(l0, vw0);
            int16x8_t l2w = vmulq_s16(l2, vw2);
            int16x8_t ls = vaddq_s16(vmlaq_s16(l0w, l1, vw1), l2w);

            vst1q_s16(dst + (width - 8), ls);
        }
    }
};

struct RowFilter3x3S16_m101 : public RowFilter3x3S16Base
{
    typedef s16 dstType;

    inline RowFilter3x3S16_m101(const BORDER_MODE _borderType, const srcType _borderValue, ptrdiff_t borderxl, ptrdiff_t borderxr, const s16*):
                                RowFilter3x3S16Base(_borderType, _borderValue, borderxl, borderxr), borderFilter(0) {}

    const dstType borderFilter;

    inline void operator()(const u8* src, s16* dst, ptrdiff_t width)
    {
        uint8x8_t l = vtbl1_u8(vld1_u8(src - lookLeft), vfmask);
        if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
            l = vset_lane_u8(borderValue, l, 6);

        ptrdiff_t i = 0;
        for (; i < width - 16 + lookRight; i += 16)
        {
            internal::prefetch(src + i);

            uint8x8_t l2 = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vreinterpretq_s16_u16(vsubl_u8(l2, vext_u8(l, l2, 6))));

            l = vld1_u8(src + i + 9);
            vst1q_s16(dst + i + 8, vreinterpretq_s16_u16(vsubl_u8(l, vext_u8(l2, l, 6))));
        }

        if (i < width - 8 + lookRight)
        {
            uint8x8_t l2 = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vreinterpretq_s16_u16(vsubl_u8(l2, vext_u8(l, l2, 6))));
            i += 8;
        }

        //tail
        if (lookRight == 0 || i != width)
        {
            uint8x8_t tail0 = vld1_u8(src + (width - 9));//can't get left 1 pixel another way if width==8*k+1
            uint8x8_t tail2 = vtbl1_u8(vld1_u8(src + (width - 8 + lookRight)), vtmask);
            if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
                tail2 = vset_lane_u8(borderValue, tail2, 7);

            int16x8_t ls = vreinterpretq_s16_u16(vsubl_u8(tail2, tail0));

            vst1q_s16(dst + (width - 8), ls);
        }
    }
};

struct RowFilter3x3S16_121 : public RowFilter3x3S16Base
{
    typedef s16 dstType;

    inline RowFilter3x3S16_121(const BORDER_MODE _borderType, const srcType _borderValue, ptrdiff_t borderxl, ptrdiff_t borderxr, const s16*):
                               RowFilter3x3S16Base(_borderType, _borderValue, borderxl, borderxr), borderFilter(borderValue << 2) {}

    const dstType borderFilter;

    inline void operator()(const u8* src, s16* dst, ptrdiff_t width)
    {
        uint8x8_t l = vtbl1_u8(vld1_u8(src - lookLeft), vfmask);
        if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
            l = vset_lane_u8(borderValue, l, 6);

        ptrdiff_t i = 0;
        for (; i < width - 16 + lookRight; i += 16)
        {
            internal::prefetch(src + i);

            uint8x8_t l2 = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vqaddq_s16(vreinterpretq_s16_u16(vaddl_u8(vext_u8(l, l2, 6), l2)),
                                          vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l, l2, 7), 1))));

            l = vld1_u8(src + i + 9);
            vst1q_s16(dst + i + 8, vqaddq_s16(vreinterpretq_s16_u16(vaddl_u8(vext_u8(l2, l, 6), l)),
                                              vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l2, l, 7), 1))));
        }

        if (i < width - 8 + lookRight)
        {
            uint8x8_t l2 = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vqaddq_s16(vreinterpretq_s16_u16(vaddl_u8(vext_u8(l, l2, 6), l2)),
                                          vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l, l2, 7), 1))));
            i += 8;
        }

        //tail
        if (lookRight == 0 || i != width)
        {
            uint8x8_t tail0 = vld1_u8(src + (width - 9));//can't get left 1 pixel another way if width==8*k+1
            uint8x8_t tail2 = vtbl1_u8(vld1_u8(src + (width - 8 + lookRight)), vtmask);
            if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
                tail2 = vset_lane_u8(borderValue, tail2, 7);
            uint8x8_t tail1 = vext_u8(vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(tail0), 8*6)), tail2, 7);

            int16x8_t tail02 = vreinterpretq_s16_u16(vaddl_u8(tail0, tail2));
            int16x8_t tail1x2 = vreinterpretq_s16_u16(vshll_n_u8(tail1, 1));

            int16x8_t ls = vqaddq_s16(tail02, tail1x2);

            vst1q_s16(dst + (width - 8), ls);
        }
    }
};

struct RowFilter3x3S16_1m21 : public RowFilter3x3S16Base
{
    typedef s16 dstType;

    inline RowFilter3x3S16_1m21(const BORDER_MODE _borderType, const srcType _borderValue, ptrdiff_t borderxl, ptrdiff_t borderxr, const s16*):
                                RowFilter3x3S16Base(_borderType, _borderValue, borderxl, borderxr), borderFilter(0) {}

    const dstType borderFilter;

    inline void operator()(const u8* src, s16* dst, ptrdiff_t width)
    {
        uint8x8_t l = vtbl1_u8(vld1_u8(src - lookLeft), vfmask);
        if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
            l = vset_lane_u8(borderValue, l, 6);

        ptrdiff_t i = 0;
        for (; i < width - 16 + lookRight; i += 16)
        {
            internal::prefetch(src + i);

            uint8x8_t l2 = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vqsubq_s16(vreinterpretq_s16_u16(vaddl_u8(vext_u8(l, l2, 6), l2)),
                                          vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l, l2, 7), 1))));

            l = vld1_u8(src + i + 9);
            vst1q_s16(dst + i + 8, vqsubq_s16(vreinterpretq_s16_u16(vaddl_u8(vext_u8(l2, l, 6), l)),
                                              vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l2, l, 7), 1))));
        }

        if (i < width - 8 + lookRight)
        {
            uint8x8_t l2 = vld1_u8(src + i + 1);
            vst1q_s16(dst + i, vqsubq_s16(vreinterpretq_s16_u16(vaddl_u8(vext_u8(l, l2, 6), l2)),
                                          vreinterpretq_s16_u16(vshll_n_u8(vext_u8(l, l2, 7), 1))));
            i += 8;
        }

        //tail
        if (lookRight == 0 || i != width)
        {
            uint8x8_t tail0 = vld1_u8(src + (width - 9));//can't get left 1 pixel another way if width==8*k+1
            uint8x8_t tail2 = vtbl1_u8(vld1_u8(src + (width - 8 + lookRight)), vtmask);
            if (lookLeft == 0 && borderType == BORDER_MODE_CONSTANT)
                tail2 = vset_lane_u8(borderValue, tail2, 7);
            uint8x8_t tail1 = vext_u8(vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(tail0), 8*6)), tail2, 7);

            int16x8_t tail02 = vreinterpretq_s16_u16(vaddl_u8(tail0, tail2));
            int16x8_t tail1x2 = vreinterpretq_s16_u16(vshll_n_u8(tail1, 1));

            int16x8_t ls = vqsubq_s16(tail02, tail1x2);

            vst1q_s16(dst + (width - 8), ls);
        }
    }
};

struct ColFilter3x3S16Generic : public ColFilter3x3S16Base
{
    typedef s16 dstType;

    inline ColFilter3x3S16Generic(const BORDER_MODE _borderType, const srcType _borderValue, const s16 *w):
                                  ColFilter3x3S16Base(_borderType, _borderValue)
    {
        vw0 = vdupq_n_s16(w[0]);
        vw1 = vdupq_n_s16(w[1]);
        vw2 = vdupq_n_s16(w[2]);
    }

    int16x8_t vw0;
    int16x8_t vw1;
    int16x8_t vw2;

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, const s16* src3, s16* dst0, s16* dst1, ptrdiff_t width)
    {
        ptrdiff_t j = 0;
        for (; j <= width - 16; j += 16)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);
            vst1q_s16(dst0 + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j), vw0), line1, vw1), line2, vw2));
            vst1q_s16(dst1 + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src3 + j), vw2), line1, vw0), line2, vw1));

            line1 = vld1q_s16(src1 + j + 8);
            line2 = vld1q_s16(src2 + j + 8);
            vst1q_s16(dst0 + j + 8, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j + 8), vw0), line1, vw1), line2, vw2));
            vst1q_s16(dst1 + j + 8, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src3 + j + 8), vw2), line1, vw0), line2, vw1));
        }
        if (j <= width - 8)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);
            vst1q_s16(dst0 + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j), vw0), line1, vw1), line2, vw2));
            vst1q_s16(dst1 + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src3 + j), vw2), line1, vw0), line2, vw1));
            j += 8;
        }
        if (j != width)
        {
            j = width - 8;
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);
            vst1q_s16(dst0 + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j), vw0), line1, vw1), line2, vw2));
            vst1q_s16(dst1 + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src3 + j), vw2), line1, vw0), line2, vw1));
        }
    }

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, s16* dst, ptrdiff_t width)
    {
        if (src0 == 0 || src2 == 0)
        {
            int16x8_t vwl1 = vw0;
            int16x8_t vwl2 = vw2;
            if (src2 == 0)
            {
                src2 = src0;
                vwl1 = vw2;
                vwl2 = vw0;
            }

            int16x8_t v_border = vdupq_n_s16(0);
            if (borderType == BORDER_MODE_CONSTANT)
            {
                v_border = vmulq_s16(vdupq_n_s16(borderValue), vwl1);
                vwl1 = vw1;
            }
            else if (borderType == BORDER_MODE_REFLECT101)
            {
                vwl1 = vw1;
                vwl2 = vaddq_s16(vw0, vw2);
            }
            else //replicate\reflect
                vwl1 = vaddq_s16(vwl1, vw1);

            ptrdiff_t j = 0;
            for (; j <= width - 16; j += 16)
            {
                vst1q_s16(dst + j, vaddq_s16(vmlaq_s16(v_border, vld1q_s16(src1 + j), vwl1),
                                             vmulq_s16(vld1q_s16(src2 + j), vwl2)));
                vst1q_s16(dst + j + 8, vaddq_s16(vmlaq_s16(v_border, vld1q_s16(src1 + j + 8), vwl1),
                                             vmulq_s16(vld1q_s16(src2 + j + 8), vwl2)));
            }
            if (j <= width - 8)
            {
                vst1q_s16(dst + j, vaddq_s16(vmlaq_s16(v_border, vld1q_s16(src1 + j), vwl1),
                                             vmulq_s16(vld1q_s16(src2 + j), vwl2)));
                j += 8;
            }
            if (j != width)
            {
                j = width - 8;
                vst1q_s16(dst + j, vaddq_s16(vmlaq_s16(v_border, vld1q_s16(src1 + j), vwl1),
                                             vmulq_s16(vld1q_s16(src2 + j), vwl2)));
            }
        }
        else
        {
            ptrdiff_t j = 0;
            for (; j <= width - 16; j += 16)
            {
                vst1q_s16(dst + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j), vw0),
                                                                 vld1q_s16(src1 + j), vw1),
                                                                 vld1q_s16(src2 + j), vw2));
                vst1q_s16(dst + j + 8, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j + 8), vw0),
                                                                     vld1q_s16(src1 + j + 8), vw1),
                                                                     vld1q_s16(src2 + j + 8), vw2));
            }
            if (j <= width - 8)
            {
                vst1q_s16(dst + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j), vw0),
                                                                 vld1q_s16(src1 + j), vw1),
                                                                 vld1q_s16(src2 + j), vw2));
                j += 8;
            }
            if (j != width)
            {
                j = width - 8;
                vst1q_s16(dst + j, vmlaq_s16(vmlaq_s16(vmulq_s16(vld1q_s16(src0 + j), vw0),
                                                                 vld1q_s16(src1 + j), vw1),
                                                                 vld1q_s16(src2 + j), vw2));
            }
        }
    }
};

struct ColFilter3x3S16_m101 : public ColFilter3x3S16Base
{
    typedef s16 dstType;

    inline ColFilter3x3S16_m101(const BORDER_MODE _borderType, const srcType _borderValue, const s16 *):
                                ColFilter3x3S16Base(_borderType, _borderValue) {}

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, const s16* src3, s16* dst0, s16* dst1, ptrdiff_t width)
    {
        ptrdiff_t j = 0;
        for (; j <= width - 16; j += 16)
        {
            vst1q_s16(dst0 + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
            vst1q_s16(dst1 + j, vqsubq_s16(vld1q_s16(src3 + j), vld1q_s16(src1 + j)));
            vst1q_s16(dst0 + j + 8, vqsubq_s16(vld1q_s16(src2 + j + 8), vld1q_s16(src0 + j + 8)));
            vst1q_s16(dst1 + j + 8, vqsubq_s16(vld1q_s16(src3 + j + 8), vld1q_s16(src1 + j + 8)));
        }
        if (j <= width - 8)
        {
            vst1q_s16(dst0 + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
            vst1q_s16(dst1 + j, vqsubq_s16(vld1q_s16(src3 + j), vld1q_s16(src1 + j)));
            j += 8;
        }
        if (j != width)
        {
            j = width - 8;
            vst1q_s16(dst0 + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
            vst1q_s16(dst1 + j, vqsubq_s16(vld1q_s16(src3 + j), vld1q_s16(src1 + j)));
        }
    }

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, s16* dst, ptrdiff_t width)
    {
        if (src0 == 0 || src2 == 0)
        {
            if (borderType == BORDER_MODE_CONSTANT)
            {
                int16x8_t v_border = vdupq_n_s16(borderValue);
                if (src0 == 0)
                {
                    ptrdiff_t j = 0;
                    for (; j <= width - 16; j += 16)
                    {
                        vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), v_border));
                        vst1q_s16(dst + j + 8, vqsubq_s16(vld1q_s16(src2 + j + 8), v_border));
                    }
                    if (j <= width - 8)
                    {
                        vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), v_border));
                        j += 8;
                    }
                    if (j != width)
                    {
                        j = width - 8;
                        vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), v_border));
                    }
                }
                else
                {
                    ptrdiff_t j = 0;
                    for (; j <= width - 16; j += 16)
                    {
                        vst1q_s16(dst + j, vqsubq_s16(v_border, vld1q_s16(src0 + j)));
                        vst1q_s16(dst + j + 8, vqsubq_s16(v_border, vld1q_s16(src0 + j + 8)));
                    }
                    if (j <= width - 8)
                    {
                        vst1q_s16(dst + j, vqsubq_s16(v_border, vld1q_s16(src0 + j)));
                        j += 8;
                    }
                    if (j != width)
                    {
                        j = width - 8;
                        vst1q_s16(dst + j, vqsubq_s16(v_border, vld1q_s16(src0 + j)));
                    }
                }
            }
            else if (borderType == BORDER_MODE_REFLECT101)
            {
                int16x8_t vzero = vmovq_n_s16(0);
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vzero);
                    vst1q_s16(dst + j + 8, vzero);
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vzero);
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vzero);
                }
            }
            else //replicate\reflect
            {
                if (src0 == 0) src0 = src1; else src2 = src1;
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
                    vst1q_s16(dst + j + 8, vqsubq_s16(vld1q_s16(src2 + j + 8), vld1q_s16(src0 + j + 8)));
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
                }
            }
        }
        else
        {
            ptrdiff_t j = 0;
            for (; j <= width - 16; j += 16)
            {
                vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
                vst1q_s16(dst + j + 8, vqsubq_s16(vld1q_s16(src2 + j + 8), vld1q_s16(src0 + j + 8)));
            }
            if (j <= width - 8)
            {
                vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
                j += 8;
            }
            if (j != width)
            {
                j = width - 8;
                vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src0 + j)));
            }
        }
    }
};

struct ColFilter3x3S16_121 : public ColFilter3x3S16Base
{
    typedef s16 dstType;

    inline ColFilter3x3S16_121(const BORDER_MODE _borderType, const srcType _borderValue, const s16*):
                               ColFilter3x3S16Base(_borderType, _borderValue) {}

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, const s16* src3, s16* dst0, s16* dst1, ptrdiff_t width)
    {
        ptrdiff_t j = 0;
        //int16x8_t line0 = vld1q_s16(src0 + j);//1
        //int16x8_t line1 = vld1q_s16(src1 + j);//11
        //int16x8_t line2 = vld1q_s16(src2 + j);// 11
        //int16x8_t line3 = vld1q_s16(src3 + j);//  1
        for (; j <= width - 16; j += 16)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vqaddq_s16(line1, line2);

            vst1q_s16(dst0 + j, vqaddq_s16(vqaddq_s16(vld1q_s16(src0 + j), line1), l12));
            vst1q_s16(dst1 + j, vqaddq_s16(l12, vqaddq_s16(line2, vld1q_s16(src3 + j))));

            line1 = vld1q_s16(src1 + j + 8);
            line2 = vld1q_s16(src2 + j + 8);

            l12 = vqaddq_s16(line1, line2);

            vst1q_s16(dst0 + j + 8, vqaddq_s16(vqaddq_s16(vld1q_s16(src0 + j + 8), line1), l12));
            vst1q_s16(dst1 + j + 8, vqaddq_s16(l12, vqaddq_s16(line2, vld1q_s16(src3 + j + 8))));
        }
        if (j <= width - 8)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vqaddq_s16(line1, line2);

            vst1q_s16(dst0 + j, vqaddq_s16(vqaddq_s16(vld1q_s16(src0 + j), line1), l12));
            vst1q_s16(dst1 + j, vqaddq_s16(l12, vqaddq_s16(line2, vld1q_s16(src3 + j))));
            j += 8;
        }
        if (j != width)
        {
            j = width - 8;
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vqaddq_s16(line1, line2);

            vst1q_s16(dst0 + j, vqaddq_s16(vqaddq_s16(vld1q_s16(src0 + j), line1), l12));
            vst1q_s16(dst1 + j, vqaddq_s16(l12, vqaddq_s16(line2, vld1q_s16(src3 + j))));
        }
    }

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, s16* dst, ptrdiff_t width)
    {
        if (src0 == 0 || src2 == 0)
        {
            if (src2 == 0)
                src2 = src0;

            if (borderType == BORDER_MODE_CONSTANT)
            {
                int16x8_t v_border = vdupq_n_s16(borderValue);
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                  vqaddq_s16(v_border, vld1q_s16(src2 + j))));
                    vst1q_s16(dst + j + 8, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j + 8), 1),
                                                      vqaddq_s16(v_border, vld1q_s16(src2 + j + 8))));
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                  vqaddq_s16(v_border, vld1q_s16(src2 + j))));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                  vqaddq_s16(v_border, vld1q_s16(src2 + j))));
                }
            }
            else if (borderType == BORDER_MODE_REFLECT101)
            {
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vqshlq_n_s16(vqaddq_s16(vld1q_s16(src1 + j),
                                                               vld1q_s16(src2 + j)), 1));
                    vst1q_s16(dst + j + 8, vqshlq_n_s16(vqaddq_s16(vld1q_s16(src1 + j + 8),
                                                                   vld1q_s16(src2 + j + 8)), 1));
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vqshlq_n_s16(vqaddq_s16(vld1q_s16(src1 + j),
                                                               vld1q_s16(src2 + j)), 1));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vqshlq_n_s16(vqaddq_s16(vld1q_s16(src1 + j),
                                                               vld1q_s16(src2 + j)), 1));
                }
            }
            else //replicate\reflect
            {
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    int16x8_t line1 = vld1q_s16(src1 + j);
                    vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(line1, 1),
                                                  vqaddq_s16(line1, vld1q_s16(src2 + j))));

                    line1 = vld1q_s16(src1 + j + 8);
                    vst1q_s16(dst + j + 8, vqaddq_s16(vqshlq_n_s16(line1, 1),
                                                      vqaddq_s16(line1, vld1q_s16(src2 + j + 8))));
                }
                if (j <= width - 8)
                {
                    int16x8_t line1 = vld1q_s16(src1 + j);
                    vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(line1, 1),
                                                  vqaddq_s16(line1, vld1q_s16(src2 + j))));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    int16x8_t line1 = vld1q_s16(src1 + j);
                    vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(line1, 1),
                                                  vqaddq_s16(line1, vld1q_s16(src2 + j))));
                }
            }
        }
        else
        {
            ptrdiff_t j = 0;
            for (; j <= width - 16; j += 16)
            {
                vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j), 1),
                                              vqaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j))));

                vst1q_s16(dst + j + 8, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j + 8), 1),
                                              vqaddq_s16(vld1q_s16(src0 + j + 8), vld1q_s16(src2 + j + 8))));
            }
            if (j <= width - 8)
            {
                vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j), 1),
                                              vqaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j))));
                j += 8;
            }
            if (j != width)
            {
                j = width - 8;
                vst1q_s16(dst + j, vqaddq_s16(vqshlq_n_s16(vld1q_s16(src1 + j), 1),
                                              vqaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j))));
            }
        }
    }
};

struct ColFilter3x3U8_121 : public ColFilter3x3S16Base
{
    typedef u8 dstType;

    inline ColFilter3x3U8_121(const BORDER_MODE _borderType, const srcType _borderValue, const s16*):
                              ColFilter3x3S16Base(_borderType, _borderValue) {}

    inline void operator()(const srcType* src0, const srcType* src1, const srcType* src2, const srcType* src3, dstType* dst0, dstType* dst1, ptrdiff_t width)
    {
        ptrdiff_t j = 0;
        //int16x8_t line0 = vld1q_s16(src0 + j);//1
        //int16x8_t line1 = vld1q_s16(src1 + j);//11
        //int16x8_t line2 = vld1q_s16(src2 + j);// 11
        //int16x8_t line3 = vld1q_s16(src3 + j);//  1
        for (; j <= width - 16; j += 16)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vaddq_s16(line1, line2);

            vst1_u8(dst0 + j, vqrshrun_n_s16(vaddq_s16(vaddq_s16(vld1q_s16(src0 + j), line1), l12), 4));
            vst1_u8(dst1 + j, vqrshrun_n_s16(vaddq_s16(l12, vaddq_s16(line2, vld1q_s16(src3 + j))), 4));

            line1 = vld1q_s16(src1 + j + 8);
            line2 = vld1q_s16(src2 + j + 8);

            l12 = vaddq_s16(line1, line2);

            vst1_u8(dst0 + j + 8, vqrshrun_n_s16(vaddq_s16(vaddq_s16(vld1q_s16(src0 + j + 8), line1), l12), 4));
            vst1_u8(dst1 + j + 8, vqrshrun_n_s16(vaddq_s16(l12, vaddq_s16(line2, vld1q_s16(src3 + j + 8))), 4));
        }
        if (j <= width - 8)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vaddq_s16(line1, line2);

            vst1_u8(dst0 + j, vqrshrun_n_s16(vaddq_s16(vaddq_s16(vld1q_s16(src0 + j), line1), l12), 4));
            vst1_u8(dst1 + j, vqrshrun_n_s16(vaddq_s16(l12, vaddq_s16(line2, vld1q_s16(src3 + j))), 4));
            j += 8;
        }
        if (j != width)
        {
            j = width - 8;
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vaddq_s16(line1, line2);

            vst1_u8(dst0 + j, vqrshrun_n_s16(vaddq_s16(vaddq_s16(vld1q_s16(src0 + j), line1), l12), 4));
            vst1_u8(dst1 + j, vqrshrun_n_s16(vaddq_s16(l12, vaddq_s16(line2, vld1q_s16(src3 + j))), 4));
        }
    }

    inline void operator()(const srcType* src0, const srcType* src1, const srcType* src2, dstType* dst, ptrdiff_t width)
    {
        if (src0 == 0 || src2 == 0)
        {
            if (src2 == 0)
                src2 = src0;

            if (borderType == BORDER_MODE_CONSTANT)
            {
                ptrdiff_t j = 0;
                int16x8_t v_border = vdupq_n_s16(borderValue);
                for (; j <= width - 16; j += 16)
                {
                    //Store normalized result, essential for gaussianBlur
                    vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                              vaddq_s16(v_border, vld1q_s16(src2 + j))), 4));

                    vst1_u8(dst + j + 8, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j + 8), 1),
                                                                  vaddq_s16(v_border, vld1q_s16(src2 + j + 8))), 4));
                }
                if (j <= width - 8)
                {
                    vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                              vaddq_s16(v_border, vld1q_s16(src2 + j))), 4));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                              vaddq_s16(v_border, vld1q_s16(src2 + j))), 4));
                }
            }
            else if (borderType == BORDER_MODE_REFLECT101)
            {
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1_u8(dst + j, vqrshrun_n_s16(vshlq_n_s16(vaddq_s16(vld1q_s16(src1 + j),
                                                                          vld1q_s16(src2 + j)), 1), 4));
                    vst1_u8(dst + j + 8, vqrshrun_n_s16(vshlq_n_s16(vaddq_s16(vld1q_s16(src1 + j + 8),
                                                                          vld1q_s16(src2 + j + 8)), 1), 4));
                }
                if (j <= width - 8)
                {
                    vst1_u8(dst + j, vqrshrun_n_s16(vshlq_n_s16(vaddq_s16(vld1q_s16(src1 + j),
                                                                          vld1q_s16(src2 + j)), 1), 4));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1_u8(dst + j, vqrshrun_n_s16(vshlq_n_s16(vaddq_s16(vld1q_s16(src1 + j),
                                                                          vld1q_s16(src2 + j)), 1), 4));
                }
            }
            else //replicate\reflect
            {
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    int16x8_t line1 = vld1q_s16(src1 + j);
                    vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(line1, 1),
                                                              vaddq_s16(line1, vld1q_s16(src2 + j))), 4));

                    line1 = vld1q_s16(src1 + j + 8);
                    vst1_u8(dst + j + 8, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(line1, 1),
                                                              vaddq_s16(line1, vld1q_s16(src2 + j + 8))), 4));
                }
                if (j <= width - 8)
                {
                    int16x8_t line1 = vld1q_s16(src1 + j);
                    vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(line1, 1),
                                                              vaddq_s16(line1, vld1q_s16(src2 + j))), 4));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    int16x8_t line1 = vld1q_s16(src1 + j);
                    vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(line1, 1),
                                                              vaddq_s16(line1, vld1q_s16(src2 + j))), 4));
                }
            }
        }
        else
        {
            ptrdiff_t j = 0;
            for (; j <= width - 16; j += 16)
            {
                vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                          vaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j))), 4));
                vst1_u8(dst + j + 8, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j + 8), 1),
                                                          vaddq_s16(vld1q_s16(src0 + j + 8), vld1q_s16(src2 + j + 8))), 4));
            }
            if (j <= width - 8)
            {
                vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                          vaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j))), 4));
                j += 8;
            }
            if (j != width)
            {
                j = width - 8;
                vst1_u8(dst + j, vqrshrun_n_s16(vaddq_s16(vshlq_n_s16(vld1q_s16(src1 + j), 1),
                                                          vaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j))), 4));
            }
        }
    }
};

struct ColFilter3x3S16_1m21 : public ColFilter3x3S16Base
{
    typedef s16 dstType;

    inline ColFilter3x3S16_1m21(const BORDER_MODE _borderType, const srcType _borderValue, const s16*):
                                ColFilter3x3S16Base(_borderType, _borderValue) {}

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, const s16* src3, s16* dst0, s16* dst1, ptrdiff_t width)
    {
        ptrdiff_t j = 0;
        //int16x8_t line0 = vld1q_s16(src0 + j);// 1
        //int16x8_t line1 = vld1q_s16(src1 + j);//-1 1
        //int16x8_t line2 = vld1q_s16(src2 + j);//  -1 -1
        //int16x8_t line3 = vld1q_s16(src3 + j);//      1
        for (; j <= width - 16; j += 16)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vqsubq_s16(line1, line2);

            vst1q_s16(dst0 + j, vqsubq_s16(vqsubq_s16(vld1q_s16(src0 + j), line1), l12));
            vst1q_s16(dst1 + j, vqaddq_s16(vqsubq_s16(vld1q_s16(src3 + j), line2), l12));

            line1 = vld1q_s16(src1 + j + 8);
            line2 = vld1q_s16(src2 + j + 8);

            l12 = vqsubq_s16(line1, line2);

            vst1q_s16(dst0 + j + 8, vqsubq_s16(vqsubq_s16(vld1q_s16(src0 + j + 8), line1), l12));
            vst1q_s16(dst1 + j + 8, vqaddq_s16(vqsubq_s16(vld1q_s16(src3 + j + 8), line2), l12));
        }
        if (j <= width - 8)
        {
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vqsubq_s16(line1, line2);

            vst1q_s16(dst0 + j, vqsubq_s16(vqsubq_s16(vld1q_s16(src0 + j), line1), l12));
            vst1q_s16(dst1 + j, vqaddq_s16(vqsubq_s16(vld1q_s16(src3 + j), line2), l12));
            j += 8;
        }
        if (j != width)
        {
            j = width - 8;
            int16x8_t line1 = vld1q_s16(src1 + j);
            int16x8_t line2 = vld1q_s16(src2 + j);

            int16x8_t l12 = vqsubq_s16(line1, line2);

            vst1q_s16(dst0 + j, vqsubq_s16(vqsubq_s16(vld1q_s16(src0 + j), line1), l12));
            vst1q_s16(dst1 + j, vqaddq_s16(vqsubq_s16(vld1q_s16(src3 + j), line2), l12));
        }
    }

    inline void operator()(const s16* src0, const s16* src1, const s16* src2, s16* dst, ptrdiff_t width)
    {
        if (src0 == 0 || src2 == 0)
        {
            if (src2 == 0)
                src2 = src0;

            if (borderType == BORDER_MODE_CONSTANT)
            {
                ptrdiff_t j = 0;
                int16x8_t v_border = vdupq_n_s16(borderValue);
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vqsubq_s16(vqaddq_s16(v_border, vld1q_s16(src2 + j)), vshlq_n_s16(vld1q_s16(src1 + j), 1)));
                    vst1q_s16(dst + j + 8, vqsubq_s16(vqaddq_s16(v_border, vld1q_s16(src2 + j + 8)), vshlq_n_s16(vld1q_s16(src1 + j + 8), 1)));
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vqsubq_s16(vqaddq_s16(v_border, vld1q_s16(src2 + j)), vshlq_n_s16(vld1q_s16(src1 + j), 1)));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vqsubq_s16(vqaddq_s16(v_border, vld1q_s16(src2 + j)), vshlq_n_s16(vld1q_s16(src1 + j), 1)));
                }
            }
            else if (borderType == BORDER_MODE_REFLECT101)
            {
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vqshlq_n_s16(vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src1 + j)), 1));
                    vst1q_s16(dst + j + 8, vqshlq_n_s16(vqsubq_s16(vld1q_s16(src2 + j + 8), vld1q_s16(src1 + j + 8)), 1));
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vqshlq_n_s16(vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src1 + j)), 1));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vqshlq_n_s16(vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src1 + j)), 1));
                }
            }
            else //replicate\reflect
            {
                ptrdiff_t j = 0;
                for (; j <= width - 16; j += 16)
                {
                    vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src1 + j)));
                    vst1q_s16(dst + j + 8, vqsubq_s16(vld1q_s16(src2 + j + 8), vld1q_s16(src1 + j + 8)));
                }
                if (j <= width - 8)
                {
                    vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src1 + j)));
                    j += 8;
                }
                if (j != width)
                {
                    j = width - 8;
                    vst1q_s16(dst + j, vqsubq_s16(vld1q_s16(src2 + j), vld1q_s16(src1 + j)));
                }
            }
        }
        else
        {
            ptrdiff_t j = 0;
            for (; j <= width - 16; j += 16)
            {
                vst1q_s16(dst + j, vqsubq_s16(vqaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j)),
                                              vqshlq_n_s16(vld1q_s16(src1 + j), 1)));
                vst1q_s16(dst + j + 8, vqsubq_s16(vqaddq_s16(vld1q_s16(src0 + j + 8), vld1q_s16(src2 + j + 8)),
                                              vqshlq_n_s16(vld1q_s16(src1 + j + 8), 1)));
            }
            if (j <= width - 8)
            {
                vst1q_s16(dst + j, vqsubq_s16(vqaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j)),
                                              vqshlq_n_s16(vld1q_s16(src1 + j), 1)));
                j += 8;
            }
            if (j != width)
            {
                j = width - 8;
                vst1q_s16(dst + j, vqsubq_s16(vqaddq_s16(vld1q_s16(src0 + j), vld1q_s16(src2 + j)),
                                              vqshlq_n_s16(vld1q_s16(src1 + j), 1)));
            }
        }
    }
};

template<class RowFilter, class ColFilter> struct sepFilter3x3
{
    typedef typename RowFilter::srcType srcType;
    typedef typename RowFilter::dstType tmpType;
    typedef typename ColFilter::dstType dstType;

    static void process(const Size2D &ssize,
                        const srcType * srcBase, ptrdiff_t srcStride,
                        dstType * dstBase, ptrdiff_t dstStride,
                        const s16 *xw, const s16 *yw,
                        BORDER_MODE borderType, srcType borderValue, Margin borderMargin)
    {
        const ptrdiff_t offsetk = 1;
        ptrdiff_t borderxl, borderxr, borderyt, borderyb;
        borderxl = std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.left);
        borderyt = std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.top);
        borderxr = std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.right);
        borderyb = std::max<ptrdiff_t>(0, offsetk - (ptrdiff_t)borderMargin.bottom);

        std::vector<tmpType> _buf(ssize.width << 2);
        tmpType * buf = &_buf[0];

        RowFilter filterX(borderType, borderValue, borderxl, borderxr, xw);
        ColFilter filterY(borderType, filterX.borderFilter, yw);
        const ptrdiff_t lookTop = offsetk - borderyt;
        const ptrdiff_t lookBottom = offsetk - borderyb;

        const srcType* src = srcBase - lookTop * srcStride / sizeof(srcType);
        dstType* dst = dstBase;

        ptrdiff_t ridx = -lookTop;
        for (; ridx <= (ptrdiff_t)ssize.height + lookBottom - 2; ridx += 2)
        {
            for (ptrdiff_t bidx = 0; bidx < 2; ++bidx, src += srcStride / sizeof(srcType))
                filterX(src, buf + ssize.width * ((4 + ridx + bidx) % 4), ssize.width);

            if (ridx <= 0)
            {
                if (ridx == 0) //first row
                {
                    filterY(0, buf + ssize.width * ((ridx + 4) % 4), buf + ssize.width * ((ridx + 1) % 4), dst, ssize.width);
                    dst += dstStride / sizeof(dstType);
                }
                continue;
            }

            filterY(buf + ssize.width * ((ridx + 2) % 4),
                    buf + ssize.width * ((ridx + 3) % 4),
                    buf + ssize.width * ((ridx + 4) % 4),
                    buf + ssize.width * ((ridx + 1) % 4),
                    dst, dst + dstStride / sizeof(dstType),  ssize.width);

            dst += dstStride * 2 / sizeof(dstType);
        }

        if (ridx < (ptrdiff_t)ssize.height + lookBottom)
        {
            filterX(src, buf + ssize.width * ((4 + ridx) % 4), ssize.width);
            filterY(buf + ssize.width * ((2 + ridx) % 4),
                    buf + ssize.width * ((3 + ridx) % 4),
                    buf + ssize.width * ((4 + ridx) % 4), dst, ssize.width);
            dst += dstStride / sizeof(dstType);
            ridx++;
        }
        if (lookBottom == 0)
            filterY(buf + ssize.width * ((ridx + 2) % 4), buf + ssize.width * ((ridx + 3) % 4), 0, dst, ssize.width);
    }
};

} //namespace internal

} //namespace CAROTENE_NS

#endif // CAROTENE_NEON

#endif // CAROTENE_SRC_REMAP_HPP
