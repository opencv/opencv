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

#include <algorithm>

#include "common.hpp"
#include "vtransform.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

template <typename T>
struct Min
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0,
                     const typename internal::VecTraits<T>::vec128 & v_src1,
                     typename internal::VecTraits<T>::vec128 & v_dst) const
    {
        v_dst = internal::vminq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0,
                     const typename internal::VecTraits<T>::vec64 & v_src1,
                     typename internal::VecTraits<T>::vec64 & v_dst) const
    {
        v_dst = internal::vmin(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = std::min(src0[0], src1[0]);
    }
};

template <typename T>
struct Max
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0,
                     const typename internal::VecTraits<T>::vec128 & v_src1,
                     typename internal::VecTraits<T>::vec128 & v_dst) const
    {
        v_dst = internal::vmaxq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0,
                     const typename internal::VecTraits<T>::vec64 & v_src1,
                     typename internal::VecTraits<T>::vec64 & v_dst) const
    {
        v_dst = internal::vmax(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = std::max(src0[0], src1[0]);
    }
};

} // namespace

#define IMPL_OP(fun, op, type)                                         \
void fun(const Size2D &size,                                           \
         const type * src0Base, ptrdiff_t src0Stride,                  \
         const type * src1Base, ptrdiff_t src1Stride,                  \
         type * dstBase, ptrdiff_t dstStride)                          \
{                                                                      \
    internal::assertSupportedConfiguration();                          \
    internal::vtransform(size,                                         \
                         src0Base, src0Stride,                         \
                         src1Base, src1Stride,                         \
                         dstBase, dstStride, op<type>());              \
}

#else

#define IMPL_OP(fun, op, type)                    \
void fun(const Size2D &,                          \
         const type *, ptrdiff_t,                 \
         const type *, ptrdiff_t,                 \
         type *, ptrdiff_t)                       \
{                                                 \
    internal::assertSupportedConfiguration();     \
}

#endif

#define IMPL_MINMAX(type) IMPL_OP(min, Min, type) IMPL_OP(max, Max, type)

IMPL_MINMAX(u8)
IMPL_MINMAX(s8)
IMPL_MINMAX(u16)
IMPL_MINMAX(s16)
IMPL_MINMAX(u32)
IMPL_MINMAX(s32)
IMPL_MINMAX(f32)

} // namespace CAROTENE_NS
