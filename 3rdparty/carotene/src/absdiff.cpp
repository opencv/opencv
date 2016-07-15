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

#include <algorithm>

#include "common.hpp"
#include "vtransform.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

template <typename T>
struct AbsDiff
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0,
                     const typename internal::VecTraits<T>::vec128 & v_src1,
                     typename internal::VecTraits<T>::vec128 & v_dst) const
    {
        v_dst = internal::vabdq(v_src0, v_src1);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0,
                     const typename internal::VecTraits<T>::vec64 & v_src1,
                     typename internal::VecTraits<T>::vec64 & v_dst) const
    {
        v_dst = internal::vabd(v_src0, v_src1);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = src0[0] >= src1[0] ? src0[0] - src1[0] : src1[0] - src0[0];
    }
};

template <typename T>
struct AbsDiffSigned
{
    typedef T type;

    void operator() (const typename internal::VecTraits<T>::vec128 & v_src0,
                     const typename internal::VecTraits<T>::vec128 & v_src1,
                     typename internal::VecTraits<T>::vec128 & v_dst) const
    {
        typename internal::VecTraits<T>::vec128 v_min = internal::vminq(v_src0, v_src1);
        typename internal::VecTraits<T>::vec128 v_max = internal::vmaxq(v_src0, v_src1);
        v_dst = internal::vqsubq(v_max, v_min);
    }

    void operator() (const typename internal::VecTraits<T>::vec64 & v_src0,
                     const typename internal::VecTraits<T>::vec64 & v_src1,
                     typename internal::VecTraits<T>::vec64 & v_dst) const
    {
        typename internal::VecTraits<T>::vec64 v_min = internal::vmin(v_src0, v_src1);
        typename internal::VecTraits<T>::vec64 v_max = internal::vmax(v_src0, v_src1);
        v_dst = internal::vqsub(v_max, v_min);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = internal::saturate_cast<T>(src0[0] >= src1[0] ? (s64)src0[0] - src1[0] : (s64)src1[0] - src0[0]);
    }
};

} // namespace

#endif

void absDiff(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             u8 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, AbsDiff<u8>());
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

void absDiff(const Size2D &size,
             const u16 *src0Base, ptrdiff_t src0Stride,
             const u16 *src1Base, ptrdiff_t src1Stride,
             u16 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, AbsDiff<u16>());
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

void absDiff(const Size2D &size,
             const s8 *src0Base, ptrdiff_t src0Stride,
             const s8 *src1Base, ptrdiff_t src1Stride,
             s8 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, AbsDiffSigned<s8>());
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

void absDiff(const Size2D &size,
             const s16 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, AbsDiffSigned<s16>());
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

void absDiff(const Size2D &size,
             const s32 *src0Base, ptrdiff_t src0Stride,
             const s32 *src1Base, ptrdiff_t src1Stride,
             s32 *dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, AbsDiffSigned<s32>());
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

void absDiff(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride, AbsDiff<f32>());
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
