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

#include "common.hpp"
#include "vtransform.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

inline void vnst(u8* dst, uint8x16_t v1, uint8x16_t v2) { vst1q_u8(dst, v1); vst1q_u8(dst+16, v2); }
inline void vnst(u8* dst, uint16x8_t v1, uint16x8_t v2) { vst1q_u8(dst, vcombine_u8(vmovn_u16(v1), vmovn_u16(v2))); }
inline void vnst(u8* dst, uint32x4_t v1, uint32x4_t v2) { vst1_u8(dst, vmovn_u16(vcombine_u16(vmovn_u32(v1), vmovn_u32(v2)))); }

template <typename Op, int elsize> struct vtail
{
    static inline void compare(const typename Op::type * src0, const typename Op::type * src1,
                               u8 * dst, const Op & op,
                               size_t &x, size_t width)
    {
        //do nothing since there couldn't be enough data
        (void)src0;
        (void)src1;
        (void)dst;
        (void)op;
        (void)x;
        (void)width;
    }
};
template <typename Op> struct vtail<Op, 2>
{
    static inline void compare(const typename Op::type * src0, const typename Op::type * src1,
                               u8 * dst, const Op & op,
                               size_t &x, size_t width)
    {
        typedef typename Op::type type;
        typedef typename internal::VecTraits<type>::vec128 vec128;
        typedef typename internal::VecTraits<type>::unsign::vec128 uvec128;
        //There no more than 15 elements in the tail, so we could handle 8 element vector only once
        if( x + 8 < width)
        {
            vec128  v_src0, v_src1;
            uvec128 v_dst;

            v_src0 = internal::vld1q(src0 + x);
            v_src1 = internal::vld1q(src1 + x);
            op(v_src0, v_src1, v_dst);
            internal::vst1(dst + x, internal::vmovn(v_dst));
            x+=8;
        }
    }
};
template <typename Op> struct vtail<Op, 1>
{
    static inline void compare(const typename Op::type * src0, const typename Op::type * src1,
                               u8 * dst, const Op & op,
                               size_t &x, size_t width)
    {
        typedef typename Op::type type;
        typedef typename internal::VecTraits<type>::vec128 vec128;
        typedef typename internal::VecTraits<type>::unsign::vec128 uvec128;
        typedef typename internal::VecTraits<type>::vec64 vec64;
        typedef typename internal::VecTraits<type>::unsign::vec64 uvec64;
        //There no more than 31 elements in the tail, so we could handle once 16+8 or 16 or 8 elements
        if( x + 16 < width)
        {
            vec128  v_src0, v_src1;
            uvec128 v_dst;

            v_src0 = internal::vld1q(src0 + x);
            v_src1 = internal::vld1q(src1 + x);
            op(v_src0, v_src1, v_dst);
            internal::vst1q(dst + x, v_dst);
            x+=16;
        }
        if( x + 8 < width)
        {
            vec64  v_src0, v_src1;
            uvec64 v_dst;

            v_src0 = internal::vld1(src0 + x);
            v_src1 = internal::vld1(src1 + x);
            op(v_src0, v_src1, v_dst);
            internal::vst1(dst + x, v_dst);
            x+=8;
        }
    }
};

template <typename Op>
void vcompare(Size2D size,
              const typename Op::type * src0Base, ptrdiff_t src0Stride,
              const typename Op::type * src1Base, ptrdiff_t src1Stride,
              u8 * dstBase, ptrdiff_t dstStride, const Op & op)
{
    typedef typename Op::type type;
    typedef typename internal::VecTraits<type>::vec128 vec128;
    typedef typename internal::VecTraits<type>::unsign::vec128 uvec128;

    if (src0Stride == src1Stride && src0Stride == dstStride &&
        src0Stride == (ptrdiff_t)(size.width * sizeof(type)))
    {
        size.width *= size.height;
        size.height = 1;
    }

    const u32 step_base = 32 / sizeof(type);
    size_t roiw_base = size.width >= (step_base - 1) ? size.width - step_base + 1 : 0;

    for (size_t y = 0; y < size.height; ++y)
    {
        const type * src0 = internal::getRowPtr(src0Base, src0Stride, y);
        const type * src1 = internal::getRowPtr(src1Base, src1Stride, y);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, y);
        size_t x = 0;

        for( ; x < roiw_base; x += step_base )
        {
            internal::prefetch(src0 + x);
            internal::prefetch(src1 + x);

            vec128 v_src00 = internal::vld1q(src0 + x), v_src01 = internal::vld1q(src0 + x + 16 / sizeof(type));
            vec128 v_src10 = internal::vld1q(src1 + x), v_src11 = internal::vld1q(src1 + x + 16 / sizeof(type));
            uvec128 v_dst0;
            uvec128 v_dst1;

            op(v_src00, v_src10, v_dst0);
            op(v_src01, v_src11, v_dst1);

            vnst(dst + x, v_dst0, v_dst1);
        }

        vtail<Op, sizeof(type)>::compare(src0, src1, dst, op, x, size.width);

        for (; x < size.width; ++x)
        {
            op(src0 + x, src1 + x, dst + x);
        }
    }
}

template<typename T>
struct OpCmpEQ
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0, const typename internal::VecTraits<T>::vec128 & v_src1,
              typename internal::VecTraits<T>::unsign::vec128 & v_dst) const
    {
        v_dst = internal::vceqq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0, const typename internal::VecTraits<T>::vec64 & v_src1,
              typename internal::VecTraits<T>::unsign::vec64 & v_dst) const
    {
        v_dst = internal::vceq(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, u8 * dst) const
    {
        dst[0] = src0[0] == src1[0] ? 255 : 0;
    }
};

template<typename T>
struct OpCmpNE
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0, const typename internal::VecTraits<T>::vec128 & v_src1,
              typename internal::VecTraits<T>::unsign::vec128 & v_dst) const
    {
        v_dst = internal::vmvnq(internal::vceqq(v_src0, v_src1));
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0, const typename internal::VecTraits<T>::vec64 & v_src1,
              typename internal::VecTraits<T>::unsign::vec64 & v_dst) const
    {
        v_dst = internal::vmvn(internal::vceq(v_src0, v_src1));
    }

    void operator() (const T * src0, const T * src1, u8 * dst) const
    {
        dst[0] = src0[0] == src1[0] ? 0 : 255;
    }
};

template<typename T>
struct OpCmpGT
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0, const typename internal::VecTraits<T>::vec128 & v_src1,
              typename internal::VecTraits<T>::unsign::vec128 & v_dst) const
    {
        v_dst = internal::vcgtq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0, const typename internal::VecTraits<T>::vec64 & v_src1,
              typename internal::VecTraits<T>::unsign::vec64 & v_dst) const
    {
        v_dst = internal::vcgt(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, u8 * dst) const
    {
        dst[0] = src0[0] > src1[0] ? 255 : 0;
    }
};

template<typename T>
struct OpCmpGE
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0, const typename internal::VecTraits<T>::vec128 & v_src1,
              typename internal::VecTraits<T>::unsign::vec128 & v_dst) const
    {
        v_dst = internal::vcgeq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0, const typename internal::VecTraits<T>::vec64 & v_src1,
              typename internal::VecTraits<T>::unsign::vec64 & v_dst) const
    {
        v_dst = internal::vcge(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, u8 * dst) const
    {
        dst[0] = src0[0] >= src1[0] ? 255 : 0;
    }
};

}

#define IMPL_CMPOP(op, type)                              \
void cmp##op(const Size2D &size,                          \
             const type * src0Base, ptrdiff_t src0Stride, \
             const type * src1Base, ptrdiff_t src1Stride, \
                       u8 *dstBase, ptrdiff_t dstStride)  \
{                                                         \
    internal::assertSupportedConfiguration();             \
    vcompare(size,                                        \
             src0Base, src0Stride,                        \
             src1Base, src1Stride,                        \
             dstBase, dstStride,                          \
             OpCmp##op<type>());                          \
}

#else

#define IMPL_CMPOP(op, type)                              \
void cmp##op(const Size2D &size,                          \
             const type * src0Base, ptrdiff_t src0Stride, \
             const type * src1Base, ptrdiff_t src1Stride, \
             u8 *dstBase, ptrdiff_t dstStride)            \
{                                                         \
    internal::assertSupportedConfiguration();             \
    (void)size;                                           \
    (void)src0Base;                                       \
    (void)src0Stride;                                     \
    (void)src1Base;                                       \
    (void)src1Stride;                                     \
    (void)dstBase;                                        \
    (void)dstStride;                                      \
}

#endif

IMPL_CMPOP(EQ, u8)
IMPL_CMPOP(EQ, s8)
IMPL_CMPOP(EQ, u16)
IMPL_CMPOP(EQ, s16)
IMPL_CMPOP(EQ, u32)
IMPL_CMPOP(EQ, s32)
IMPL_CMPOP(EQ, f32)

IMPL_CMPOP(NE, u8)
IMPL_CMPOP(NE, s8)
IMPL_CMPOP(NE, u16)
IMPL_CMPOP(NE, s16)
IMPL_CMPOP(NE, u32)
IMPL_CMPOP(NE, s32)
IMPL_CMPOP(NE, f32)

IMPL_CMPOP(GT, u8)
IMPL_CMPOP(GT, s8)
IMPL_CMPOP(GT, u16)
IMPL_CMPOP(GT, s16)
IMPL_CMPOP(GT, u32)
IMPL_CMPOP(GT, s32)
IMPL_CMPOP(GT, f32)

IMPL_CMPOP(GE, u8)
IMPL_CMPOP(GE, s8)
IMPL_CMPOP(GE, u16)
IMPL_CMPOP(GE, s16)
IMPL_CMPOP(GE, u32)
IMPL_CMPOP(GE, s32)
IMPL_CMPOP(GE, f32)

} // namespace CAROTENE_NS
