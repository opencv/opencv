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
#include "vtransform.hpp"

#include <cstring>

namespace CAROTENE_NS {

void accumulate(const Size2D &size,
                const u8 *srcBase, ptrdiff_t srcStride,
                s16 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            internal::prefetch(dst + j);
            uint8x16_t v_src = vld1q_u8(src + j);
            int16x8_t v_dst0 = vld1q_s16(dst + j);
            int16x8_t v_dst1 = vld1q_s16(dst + j + 8);
            int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src)));
            int16x8_t v_src1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src)));
            v_dst0 = vqaddq_s16(v_dst0, v_src0);
            v_dst1 = vqaddq_s16(v_dst1, v_src1);
            vst1q_s16(dst + j, v_dst0);
            vst1q_s16(dst + j + 8, v_dst1);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v_src = vld1_u8(src + j);
            int16x8_t v_src16 = vreinterpretq_s16_u16(vmovl_u8(v_src));
            int16x8_t v_dst = vld1q_s16(dst + j);
            v_dst = vqaddq_s16(v_dst, v_src16);
            vst1q_s16(dst + j, v_dst);
        }

        for (; j < size.width; j++)
            dst[j] = internal::saturate_cast<s16>(src[j] + dst[j]);
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

#ifdef CAROTENE_NEON

namespace {

template <int shift>
void accumulateSquareConst(const Size2D &size,
                           const u8 *srcBase, ptrdiff_t srcStride,
                           s16 *dstBase, ptrdiff_t dstStride)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            internal::prefetch(dst + j);
            uint8x16_t v_src = vld1q_u8(src + j);
            int16x8_t v_dst0 = vld1q_s16(dst + j), v_dst1 = vld1q_s16(dst + j + 8);
            int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src)));
            int16x8_t v_src1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src)));

            int16x4_t v_srclo = vget_low_s16(v_src0), v_srchi = vget_high_s16(v_src0);
            v_dst0 = vcombine_s16(vqmovn_s32(vaddw_s16(vshrq_n_s32(vmull_s16(v_srclo, v_srclo), shift), vget_low_s16(v_dst0))),
                                  vqmovn_s32(vaddw_s16(vshrq_n_s32(vmull_s16(v_srchi, v_srchi), shift), vget_high_s16(v_dst0))));

            v_srclo = vget_low_s16(v_src1);
            v_srchi = vget_high_s16(v_src1);
            v_dst1 = vcombine_s16(vqmovn_s32(vaddw_s16(vshrq_n_s32(vmull_s16(v_srclo, v_srclo), shift), vget_low_s16(v_dst1))),
                                  vqmovn_s32(vaddw_s16(vshrq_n_s32(vmull_s16(v_srchi, v_srchi), shift), vget_high_s16(v_dst1))));

            vst1q_s16(dst + j, v_dst0);
            vst1q_s16(dst + j + 8, v_dst1);
        }
        for (; j < roiw8; j += 8)
        {
            int16x8_t v_src = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + j)));
            int16x8_t v_dst = vld1q_s16(dst + j);
            int16x4_t v_srclo = vget_low_s16(v_src), v_srchi = vget_high_s16(v_src);
            v_dst = vcombine_s16(vqmovn_s32(vaddw_s16(vshrq_n_s32(vmull_s16(v_srclo, v_srclo), shift), vget_low_s16(v_dst))),
                                 vqmovn_s32(vaddw_s16(vshrq_n_s32(vmull_s16(v_srchi, v_srchi), shift), vget_high_s16(v_dst))));
            vst1q_s16(dst + j, v_dst);
        }

        for (; j < size.width; j++)
        {
            s32 srcVal = src[j];
            dst[j] = internal::saturate_cast<s16>(dst[j] + ((srcVal * srcVal) >> shift));
        }
    }
}

template <>
void accumulateSquareConst<0>(const Size2D &size,
                              const u8 *srcBase, ptrdiff_t srcStride,
                              s16 *dstBase, ptrdiff_t dstStride)
{
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            internal::prefetch(dst + j);
            uint8x16_t v_src = vld1q_u8(src + j);
            int16x8_t v_dst0 = vld1q_s16(dst + j), v_dst1 = vld1q_s16(dst + j + 8);
            int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src)));
            int16x8_t v_src1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src)));

            int16x4_t v_srclo = vget_low_s16(v_src0), v_srchi = vget_high_s16(v_src0);
            v_dst0 = vcombine_s16(vqmovn_s32(vaddw_s16(vmull_s16(v_srclo, v_srclo), vget_low_s16(v_dst0))),
                                  vqmovn_s32(vaddw_s16(vmull_s16(v_srchi, v_srchi), vget_high_s16(v_dst0))));

            v_srclo = vget_low_s16(v_src1);
            v_srchi = vget_high_s16(v_src1);
            v_dst1 = vcombine_s16(vqmovn_s32(vaddw_s16(vmull_s16(v_srclo, v_srclo), vget_low_s16(v_dst1))),
                                  vqmovn_s32(vaddw_s16(vmull_s16(v_srchi, v_srchi), vget_high_s16(v_dst1))));

            vst1q_s16(dst + j, v_dst0);
            vst1q_s16(dst + j + 8, v_dst1);
        }
        for (; j < roiw8; j += 8)
        {
            int16x8_t v_src = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + j)));
            int16x8_t v_dst = vld1q_s16(dst + j);
            int16x4_t v_srclo = vget_low_s16(v_src), v_srchi = vget_high_s16(v_src);
            v_dst = vcombine_s16(vqmovn_s32(vaddw_s16(vmull_s16(v_srclo, v_srclo), vget_low_s16(v_dst))),
                                 vqmovn_s32(vaddw_s16(vmull_s16(v_srchi, v_srchi), vget_high_s16(v_dst))));
            vst1q_s16(dst + j, v_dst);
        }

        for (; j < size.width; j++)
        {
            s32 srcVal = src[j];
            dst[j] = internal::saturate_cast<s16>(dst[j] + srcVal * srcVal);
        }
    }
}

typedef void (* accumulateSquareConstFunc)(const Size2D &size,
                                           const u8 *srcBase, ptrdiff_t srcStride,
                                           s16 *dstBase, ptrdiff_t dstStride);

} // namespace

#endif

void accumulateSquare(const Size2D &size,
                      const u8 *srcBase, ptrdiff_t srcStride,
                      s16 *dstBase, ptrdiff_t dstStride,
                      u32 shift)
{
    if (shift >= 16)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            s16 * dst = internal::getRowPtr(dstBase, dstStride, i);
            std::memset(dst, 0, sizeof(s16) * size.width);
        }
        return;
    }

    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    // this ugly contruction is needed to avoid:
    // /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/arm_neon.h:3581:59: error: argument must be a constant
    // return (int16x8_t)__builtin_neon_vshr_nv8hi (__a, __b, 1);

    accumulateSquareConstFunc funcs[16] =
    {
        accumulateSquareConst<0>,
        accumulateSquareConst<1>,
        accumulateSquareConst<2>,
        accumulateSquareConst<3>,
        accumulateSquareConst<4>,
        accumulateSquareConst<5>,
        accumulateSquareConst<6>,
        accumulateSquareConst<7>,
        accumulateSquareConst<8>,
        accumulateSquareConst<9>,
        accumulateSquareConst<10>,
        accumulateSquareConst<11>,
        accumulateSquareConst<12>,
        accumulateSquareConst<13>,
        accumulateSquareConst<14>,
        accumulateSquareConst<15>
    }, func = funcs[shift];

    func(size, srcBase, srcStride, dstBase, dstStride);
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)shift;
#endif
}

#ifdef CAROTENE_NEON

namespace {

struct AccumulateWeightedHalf
{
    typedef u8 type;

    void operator() (const uint8x16_t & v_src0, const uint8x16_t & v_src1,
                     uint8x16_t & v_dst) const
    {
        v_dst = vhaddq_u8(v_src0, v_src1);
    }

    void operator() (const uint8x8_t & v_src0, const uint8x8_t & v_src1,
                     uint8x8_t & v_dst) const
    {
        v_dst = vhadd_u8(v_src0, v_src1);
    }

    void operator() (const u8 * src0, const u8 * src1, u8 * dst) const
    {
        dst[0] = ((u16)(src0[0]) + src1[0]) >> 1;
    }
};

struct AccumulateWeighted
{
    typedef u8 type;

    float alpha, beta;
    float32x4_t v_alpha, v_beta;

    explicit AccumulateWeighted(float _alpha) :
        alpha(_alpha), beta(1 - _alpha)
    {
        v_alpha = vdupq_n_f32(alpha);
        v_beta = vdupq_n_f32(beta);
    }

    void operator() (const uint8x16_t & v_src0, const uint8x16_t & v_src1,
                     uint8x16_t & v_dst) const
    {
        uint16x8_t v_src0_p = vmovl_u8(vget_low_u8(v_src0));
        uint16x8_t v_src1_p = vmovl_u8(vget_low_u8(v_src1));
        float32x4_t v_dst0f = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p))), v_beta),
                                        v_alpha, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))));
        float32x4_t v_dst1f = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p))), v_beta),
                                        v_alpha, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))));
        uint16x8_t v_dst0 = vcombine_u16(vmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                         vmovn_u32(vcvtq_u32_f32(v_dst1f)));

        v_src0_p = vmovl_u8(vget_high_u8(v_src0));
        v_src1_p = vmovl_u8(vget_high_u8(v_src1));
        v_dst0f = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1_p))), v_beta),
                            v_alpha, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0_p))));
        v_dst1f = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1_p))), v_beta),
                            v_alpha, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0_p))));
        uint16x8_t v_dst1 = vcombine_u16(vmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                         vmovn_u32(vcvtq_u32_f32(v_dst1f)));

        v_dst = vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1));
    }

    void operator() (const uint8x8_t & _v_src0, const uint8x8_t & _v_src1,
                     uint8x8_t & v_dst) const
    {
        uint16x8_t v_src0 = vmovl_u8(_v_src0), v_src1 = vmovl_u8(_v_src1);

        float32x4_t v_dst0f = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))), v_beta),
                                        v_alpha, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src0))));
        float32x4_t v_dst1f = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))), v_beta),
                                        v_alpha, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src0))));
        uint16x8_t _v_dst = vcombine_u16(vmovn_u32(vcvtq_u32_f32(v_dst0f)),
                                        vmovn_u32(vcvtq_u32_f32(v_dst1f)));

        v_dst = vmovn_u16(_v_dst);
    }

    void operator() (const u8 * src0, const u8 * src1, u8 * dst) const
    {
        dst[0] = beta * src1[0] + alpha * src0[0];
    }
};

} // namespace

#endif

void accumulateWeighted(const Size2D &size,
                        const u8 *srcBase, ptrdiff_t srcStride,
                        u8 *dstBase, ptrdiff_t dstStride,
                        f32 alpha)
{
    if (alpha == 0.0f)
        return;
    if (alpha == 1.0f)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
            u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
            std::memcpy(dst, src, sizeof(u8) * size.width);
        }
        return;
    }

    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    // in this case we can use the following scheme:
    // dst[p] = (src[p] + dst[p]) >> 1
    // which is faster
    if (alpha == 0.5f)
    {
        internal::vtransform(size,
                             srcBase, srcStride,
                             dstBase, dstStride,
                             dstBase, dstStride,
                             AccumulateWeightedHalf());

        return;
    }

    internal::vtransform(size,
                     srcBase, srcStride,
                     dstBase, dstStride,
                     dstBase, dstStride,
                     AccumulateWeighted(alpha));
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)alpha;
#endif
}

} //namespace CAROTENE_NS
