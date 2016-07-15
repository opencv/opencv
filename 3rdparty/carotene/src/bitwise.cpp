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

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

struct BitwiseAnd
{
    typedef u8 type;

    void operator() (const uint8x16_t & v_src0, const uint8x16_t & v_src1,
                     uint8x16_t & v_dst) const
    {
        v_dst = vandq_u8(v_src0, v_src1);
    }

    void operator() (const uint8x8_t & v_src0, const uint8x8_t & v_src1,
                     uint8x8_t & v_dst) const
    {
        v_dst = vand_u8(v_src0, v_src1);
    }

    void operator() (const u8 * src0, const u8 * src1, u8 * dst) const
    {
        dst[0] = src0[0] & src1[0];
    }
};

struct BitwiseOr
{
    typedef u8 type;

    void operator() (const uint8x16_t & v_src0, const uint8x16_t & v_src1,
                     uint8x16_t & v_dst) const
    {
        v_dst = vorrq_u8(v_src0, v_src1);
    }

    void operator() (const uint8x8_t & v_src0, const uint8x8_t & v_src1,
                     uint8x8_t & v_dst) const
    {
        v_dst = vorr_u8(v_src0, v_src1);
    }

    void operator() (const u8 * src0, const u8 * src1, u8 * dst) const
    {
        dst[0] = src0[0] | src1[0];
    }
};

struct BitwiseXor
{
    typedef u8 type;

    void operator() (const uint8x16_t & v_src0, const uint8x16_t & v_src1,
                     uint8x16_t & v_dst) const
    {
        v_dst = veorq_u8(v_src0, v_src1);
    }

    void operator() (const uint8x8_t & v_src0, const uint8x8_t & v_src1,
                     uint8x8_t & v_dst) const
    {
        v_dst = veor_u8(v_src0, v_src1);
    }

    void operator() (const u8 * src0, const u8 * src1, u8 * dst) const
    {
        dst[0] = src0[0] ^ src1[0];
    }
};

#endif

void bitwiseNot(const Size2D &size,
                const u8 *srcBase, ptrdiff_t srcStride,
                u8 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw32; j += 32)
        {
            internal::prefetch(src + j);
            uint8x16_t v_src0 = vld1q_u8(src + j), v_src1 = vld1q_u8(src + j + 16);
            uint8x16_t v_dst0 = vmvnq_u8(v_src0), v_dst1 = vmvnq_u8(v_src1);
            vst1q_u8(dst + j, v_dst0);
            vst1q_u8(dst + j + 16, v_dst1);
        }
        for (; j < roiw8; j += 8)
        {
            uint8x8_t v_src = vld1_u8(src + j);
            uint8x8_t v_dst = vmvn_u8(v_src);
            vst1_u8(dst + j, v_dst);
        }

        for (; j < size.width; j++)
        {
            dst[j] = ~src[j];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bitwiseAnd(const Size2D &size,
                const u8 *src0Base, ptrdiff_t src0Stride,
                const u8 *src1Base, ptrdiff_t src1Stride,
                u8 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, BitwiseAnd());
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bitwiseOr(const Size2D &size,
               const u8 *src0Base, ptrdiff_t src0Stride,
               const u8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, BitwiseOr());
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bitwiseXor(const Size2D &size,
                const u8 *src0Base, ptrdiff_t src0Stride,
                const u8 *src1Base, ptrdiff_t src1Stride,
                u8 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, BitwiseXor());
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
#endif
}

} // namespace CAROTENE_NS
