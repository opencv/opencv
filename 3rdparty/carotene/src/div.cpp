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
 * Copyright (C) 2016, NVIDIA Corporation, all rights reserved.
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
#include <cfloat>
#include <cmath>
#include <limits>

namespace CAROTENE_NS {

namespace {

#ifdef CAROTENE_NEON

template <typename T>
inline T divSaturateQ(const T &v1, const T &v2, const float scale)
{
    return internal::vcombine(internal::vqmovn(divSaturateQ(internal::vmovl(internal::vget_low(v1)),
                                                            internal::vmovl(internal::vget_low(v2)), scale)),
                              internal::vqmovn(divSaturateQ(internal::vmovl(internal::vget_high(v1)),
                                                            internal::vmovl(internal::vget_high(v2)), scale))
                             );
}
template <>
inline int32x4_t divSaturateQ<int32x4_t>(const int32x4_t &v1, const int32x4_t &v2, const float scale)
{ return vcvtq_s32_f32(vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(v1), scale), internal::vrecpq_f32(vcvtq_f32_s32(v2)))); }
template <>
inline uint32x4_t divSaturateQ<uint32x4_t>(const uint32x4_t &v1, const uint32x4_t &v2, const float scale)
{ return vcvtq_u32_f32(vmulq_f32(vmulq_n_f32(vcvtq_f32_u32(v1), scale), internal::vrecpq_f32(vcvtq_f32_u32(v2)))); }

template <typename T>
inline T divSaturate(const T &v1, const T &v2, const float scale)
{
    return internal::vqmovn(divSaturateQ(internal::vmovl(v1), internal::vmovl(v2), scale));
}
template <>
inline int32x2_t divSaturate<int32x2_t>(const int32x2_t &v1, const int32x2_t &v2, const float scale)
{ return vcvt_s32_f32(vmul_f32(vmul_n_f32(vcvt_f32_s32(v1), scale), internal::vrecp_f32(vcvt_f32_s32(v2)))); }
template <>
inline uint32x2_t divSaturate<uint32x2_t>(const uint32x2_t &v1, const uint32x2_t &v2, const float scale)
{ return vcvt_u32_f32(vmul_f32(vmul_n_f32(vcvt_f32_u32(v1), scale), internal::vrecp_f32(vcvt_f32_u32(v2)))); }


template <typename T>
inline T divWrapQ(const T &v1, const T &v2, const float scale)
{
    return internal::vcombine(internal::vmovn(divWrapQ(internal::vmovl(internal::vget_low(v1)),
                                                       internal::vmovl(internal::vget_low(v2)), scale)),
                              internal::vmovn(divWrapQ(internal::vmovl(internal::vget_high(v1)),
                                                       internal::vmovl(internal::vget_high(v2)), scale))
                             );
}
template <>
inline int32x4_t divWrapQ<int32x4_t>(const int32x4_t &v1, const int32x4_t &v2, const float scale)
{ return vcvtq_s32_f32(vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(v1), scale), internal::vrecpq_f32(vcvtq_f32_s32(v2)))); }
template <>
inline uint32x4_t divWrapQ<uint32x4_t>(const uint32x4_t &v1, const uint32x4_t &v2, const float scale)
{ return vcvtq_u32_f32(vmulq_f32(vmulq_n_f32(vcvtq_f32_u32(v1), scale), internal::vrecpq_f32(vcvtq_f32_u32(v2)))); }

template <typename T>
inline T divWrap(const T &v1, const T &v2, const float scale)
{
    return internal::vmovn(divWrapQ(internal::vmovl(v1), internal::vmovl(v2), scale));
}
template <>
inline int32x2_t divWrap<int32x2_t>(const int32x2_t &v1, const int32x2_t &v2, const float scale)
{ return vcvt_s32_f32(vmul_f32(vmul_n_f32(vcvt_f32_s32(v1), scale), internal::vrecp_f32(vcvt_f32_s32(v2)))); }
template <>
inline uint32x2_t divWrap<uint32x2_t>(const uint32x2_t &v1, const uint32x2_t &v2, const float scale)
{ return vcvt_u32_f32(vmul_f32(vmul_n_f32(vcvt_f32_u32(v1), scale), internal::vrecp_f32(vcvt_f32_u32(v2)))); }

inline  uint8x16_t vtstq(const uint8x16_t  & v0, const uint8x16_t  & v1) { return vtstq_u8 (v0, v1); }
inline  uint16x8_t vtstq(const uint16x8_t  & v0, const uint16x8_t  & v1) { return vtstq_u16(v0, v1); }
inline  uint32x4_t vtstq(const uint32x4_t  & v0, const uint32x4_t  & v1) { return vtstq_u32(v0, v1); }
inline   int8x16_t vtstq(const int8x16_t   & v0, const int8x16_t   & v1) { return vreinterpretq_s8_u8  (vtstq_s8 (v0, v1)); }
inline   int16x8_t vtstq(const int16x8_t   & v0, const int16x8_t   & v1) { return vreinterpretq_s16_u16(vtstq_s16(v0, v1)); }
inline   int32x4_t vtstq(const int32x4_t   & v0, const int32x4_t   & v1) { return vreinterpretq_s32_u32(vtstq_s32(v0, v1)); }

inline   uint8x8_t vtst(const uint8x8_t   & v0, const uint8x8_t   & v1) { return vtst_u8 (v0, v1); }
inline  uint16x4_t vtst(const uint16x4_t  & v0, const uint16x4_t  & v1) { return vtst_u16(v0, v1); }
inline  uint32x2_t vtst(const uint32x2_t  & v0, const uint32x2_t  & v1) { return vtst_u32(v0, v1); }
inline    int8x8_t vtst(const int8x8_t    & v0, const int8x8_t    & v1) { return vreinterpret_s8_u8  (vtst_s8 (v0, v1)); }
inline   int16x4_t vtst(const int16x4_t   & v0, const int16x4_t   & v1) { return vreinterpret_s16_u16(vtst_s16(v0, v1)); }
inline   int32x2_t vtst(const int32x2_t   & v0, const int32x2_t   & v1) { return vreinterpret_s32_u32(vtst_s32(v0, v1)); }
#endif

template <typename T>
void div(const Size2D &size,
         const T * src0Base, ptrdiff_t src0Stride,
         const T * src1Base, ptrdiff_t src1Stride,
         T * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    typedef typename internal::VecTraits<T>::vec128 vec128;
    typedef typename internal::VecTraits<T>::vec64 vec64;

    if (scale == 0.0f ||
        (std::numeric_limits<T>::is_integer &&
         (scale * std::numeric_limits<T>::max()) <  1.0f &&
         (scale * std::numeric_limits<T>::max()) > -1.0f))
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            T * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(T) * size.width);
        }
        return;
    }

    const size_t step128 = 16 / sizeof(T);
    size_t roiw128 = size.width >= (step128 - 1) ? size.width - step128 + 1 : 0;
    const size_t step64 = 8 / sizeof(T);
    size_t roiw64 = size.width >= (step64 - 1) ? size.width - step64 + 1 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src0 = internal::getRowPtr(src0Base, src0Stride, i);
        const T * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        T * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);

                vec128 v_src0 = internal::vld1q(src0 + j);
                vec128 v_src1 = internal::vld1q(src1 + j);

                vec128 v_mask = vtstq(v_src1,v_src1);
                internal::vst1q(dst + j, internal::vandq(v_mask, divSaturateQ(v_src0, v_src1, scale)));
            }
            for (; j < roiw64; j += step64)
            {
                vec64 v_src0 = internal::vld1(src0 + j);
                vec64 v_src1 = internal::vld1(src1 + j);

                vec64 v_mask = vtst(v_src1,v_src1);
                internal::vst1(dst + j, internal::vand(v_mask,divSaturate(v_src0, v_src1, scale)));
            }
            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? internal::saturate_cast<T>(scale * src0[j] / src1[j]) : 0;
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);

                vec128 v_src0 = internal::vld1q(src0 + j);
                vec128 v_src1 = internal::vld1q(src1 + j);

                vec128 v_mask = vtstq(v_src1,v_src1);
                internal::vst1q(dst + j, internal::vandq(v_mask, divWrapQ(v_src0, v_src1, scale)));
            }
            for (; j < roiw64; j += step64)
            {
                vec64 v_src0 = internal::vld1(src0 + j);
                vec64 v_src1 = internal::vld1(src1 + j);

                vec64 v_mask = vtst(v_src1,v_src1);
                internal::vst1(dst + j, internal::vand(v_mask,divWrap(v_src0, v_src1, scale)));
            }
            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? (T)((s32)trunc(scale * src0[j] / src1[j])) : 0;
            }
        }
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)cpolicy;
    (void)scale;
#endif
}

#ifdef CAROTENE_NEON

template <typename T>
inline T recipSaturateQ(const T &v2, const float scale)
{
    return internal::vcombine(internal::vqmovn(recipSaturateQ(internal::vmovl(internal::vget_low(v2)), scale)),
                              internal::vqmovn(recipSaturateQ(internal::vmovl(internal::vget_high(v2)), scale))
                             );
}
template <>
inline int32x4_t recipSaturateQ<int32x4_t>(const int32x4_t &v2, const float scale)
{ return vcvtq_s32_f32(vmulq_n_f32(internal::vrecpq_f32(vcvtq_f32_s32(v2)), scale)); }
template <>
inline uint32x4_t recipSaturateQ<uint32x4_t>(const uint32x4_t &v2, const float scale)
{ return vcvtq_u32_f32(vmulq_n_f32(internal::vrecpq_f32(vcvtq_f32_u32(v2)), scale)); }

template <typename T>
inline T recipSaturate(const T &v2, const float scale)
{
    return internal::vqmovn(recipSaturateQ(internal::vmovl(v2), scale));
}
template <>
inline int32x2_t recipSaturate<int32x2_t>(const int32x2_t &v2, const float scale)
{ return vcvt_s32_f32(vmul_n_f32(internal::vrecp_f32(vcvt_f32_s32(v2)), scale)); }
template <>
inline uint32x2_t recipSaturate<uint32x2_t>(const uint32x2_t &v2, const float scale)
{ return vcvt_u32_f32(vmul_n_f32(internal::vrecp_f32(vcvt_f32_u32(v2)), scale)); }


template <typename T>
inline T recipWrapQ(const T &v2, const float scale)
{
    return internal::vcombine(internal::vmovn(recipWrapQ(internal::vmovl(internal::vget_low(v2)), scale)),
                              internal::vmovn(recipWrapQ(internal::vmovl(internal::vget_high(v2)), scale))
                             );
}
template <>
inline int32x4_t recipWrapQ<int32x4_t>(const int32x4_t &v2, const float scale)
{ return vcvtq_s32_f32(vmulq_n_f32(internal::vrecpq_f32(vcvtq_f32_s32(v2)), scale)); }
template <>
inline uint32x4_t recipWrapQ<uint32x4_t>(const uint32x4_t &v2, const float scale)
{ return vcvtq_u32_f32(vmulq_n_f32(internal::vrecpq_f32(vcvtq_f32_u32(v2)), scale)); }

template <typename T>
inline T recipWrap(const T &v2, const float scale)
{
    return internal::vmovn(recipWrapQ(internal::vmovl(v2), scale));
}
template <>
inline int32x2_t recipWrap<int32x2_t>(const int32x2_t &v2, const float scale)
{ return vcvt_s32_f32(vmul_n_f32(internal::vrecp_f32(vcvt_f32_s32(v2)), scale)); }
template <>
inline uint32x2_t recipWrap<uint32x2_t>(const uint32x2_t &v2, const float scale)
{ return vcvt_u32_f32(vmul_n_f32(internal::vrecp_f32(vcvt_f32_u32(v2)), scale)); }
#endif

template <typename T>
void recip(const Size2D &size,
           const T * src1Base, ptrdiff_t src1Stride,
           T * dstBase, ptrdiff_t dstStride,
           f32 scale,
           CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    typedef typename internal::VecTraits<T>::vec128 vec128;
    typedef typename internal::VecTraits<T>::vec64 vec64;

    if (scale == 0.0f ||
        (std::numeric_limits<T>::is_integer &&
         scale <  1.0f &&
         scale > -1.0f))
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            T * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(T) * size.width);
        }
        return;
    }

    const size_t step128 = 16 / sizeof(T);
    size_t roiw128 = size.width >= (step128 - 1) ? size.width - step128 + 1 : 0;
    const size_t step64 = 8 / sizeof(T);
    size_t roiw64 = size.width >= (step64 - 1) ? size.width - step64 + 1 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src1 = internal::getRowPtr(src1Base, src1Stride, i);
        T * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(src1 + j);

                vec128 v_src1 = internal::vld1q(src1 + j);

                vec128 v_mask = vtstq(v_src1,v_src1);
                internal::vst1q(dst + j, internal::vandq(v_mask, recipSaturateQ(v_src1, scale)));
            }
            for (; j < roiw64; j += step64)
            {
                vec64 v_src1 = internal::vld1(src1 + j);

                vec64 v_mask = vtst(v_src1,v_src1);
                internal::vst1(dst + j, internal::vand(v_mask, recipSaturate(v_src1, scale)));
            }
            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? internal::saturate_cast<T>(scale / src1[j]) : 0;
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(src1 + j);

                vec128 v_src1 = internal::vld1q(src1 + j);

                vec128 v_mask = vtstq(v_src1,v_src1);
                internal::vst1q(dst + j, internal::vandq(v_mask, recipWrapQ(v_src1, scale)));
            }
            for (; j < roiw64; j += step64)
            {
                vec64 v_src1 = internal::vld1(src1 + j);

                vec64 v_mask = vtst(v_src1,v_src1);
                internal::vst1(dst + j, internal::vand(v_mask, recipWrap(v_src1, scale)));
            }
            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? (T)((s32)trunc(scale / src1[j])) : 0;
            }
        }
    }
#else
    (void)size;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)cpolicy;
    (void)scale;
#endif
}

}

void div(const Size2D &size,
         const u8 * src0Base, ptrdiff_t src0Stride,
         const u8 * src1Base, ptrdiff_t src1Stride,
         u8 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    div<u8>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void div(const Size2D &size,
         const s8 * src0Base, ptrdiff_t src0Stride,
         const s8 * src1Base, ptrdiff_t src1Stride,
         s8 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    div<s8>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void div(const Size2D &size,
         const u16 * src0Base, ptrdiff_t src0Stride,
         const u16 * src1Base, ptrdiff_t src1Stride,
         u16 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    div<u16>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void div(const Size2D &size,
         const s16 * src0Base, ptrdiff_t src0Stride,
         const s16 * src1Base, ptrdiff_t src1Stride,
         s16 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    div<s16>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void div(const Size2D &size,
         const s32 * src0Base, ptrdiff_t src0Stride,
         const s32 * src1Base, ptrdiff_t src1Stride,
         s32 * dstBase, ptrdiff_t dstStride,
         f32 scale,
         CONVERT_POLICY cpolicy)
{
    div<s32>(size, src0Base, src0Stride, src1Base, src1Stride, dstBase, dstStride, scale, cpolicy);
}

void div(const Size2D &size,
         const f32 * src0Base, ptrdiff_t src0Stride,
         const f32 * src1Base, ptrdiff_t src1Stride,
         f32 * dstBase, ptrdiff_t dstStride,
         f32 scale)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (scale == 0.0f)
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            f32 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(f32) * size.width);
        }
        return;
    }

    float32x4_t v_zero = vdupq_n_f32(0.0f);

    size_t roiw128 = size.width >= 3 ? size.width - 3 : 0;
    size_t roiw64 = size.width >= 1 ? size.width - 1 : 0;

    if (std::fabs(scale - 1.0f) < FLT_EPSILON)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const f32 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
            const f32 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
            f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw128; j += 4)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);

                float32x4_t v_src0 = vld1q_f32(src0 + j);
                float32x4_t v_src1 = vld1q_f32(src1 + j);

                uint32x4_t v_mask = vceqq_f32(v_src1,v_zero);
                vst1q_f32(dst + j, vreinterpretq_f32_u32(vbicq_u32(
                                   vreinterpretq_u32_f32(vmulq_f32(v_src0, internal::vrecpq_f32(v_src1))), v_mask)));
            }

            for (; j < roiw64; j += 2)
            {
                float32x2_t v_src0 = vld1_f32(src0 + j);
                float32x2_t v_src1 = vld1_f32(src1 + j);

                uint32x2_t v_mask = vceq_f32(v_src1,vget_low_f32(v_zero));
                vst1_f32(dst + j, vreinterpret_f32_u32(vbic_u32(
                                  vreinterpret_u32_f32(vmul_f32(v_src0, internal::vrecp_f32(v_src1))), v_mask)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? src0[j] / src1[j] : 0.0f;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const f32 * src0 = internal::getRowPtr(src0Base, src0Stride, i);
            const f32 * src1 = internal::getRowPtr(src1Base, src1Stride, i);
            f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw128; j += 4)
            {
                internal::prefetch(src0 + j);
                internal::prefetch(src1 + j);

                float32x4_t v_src0 = vld1q_f32(src0 + j);
                float32x4_t v_src1 = vld1q_f32(src1 + j);

                uint32x4_t v_mask = vceqq_f32(v_src1,v_zero);
                vst1q_f32(dst + j, vreinterpretq_f32_u32(vbicq_u32(
                                   vreinterpretq_u32_f32(vmulq_f32(vmulq_n_f32(v_src0, scale),
                                                         internal::vrecpq_f32(v_src1))), v_mask)));
            }

            for (; j < roiw64; j += 2)
            {
                float32x2_t v_src0 = vld1_f32(src0 + j);
                float32x2_t v_src1 = vld1_f32(src1 + j);

                uint32x2_t v_mask = vceq_f32(v_src1,vget_low_f32(v_zero));
                vst1_f32(dst + j, vreinterpret_f32_u32(vbic_u32(
                                  vreinterpret_u32_f32(vmul_f32(vmul_n_f32(v_src0, scale),
                                                                internal::vrecp_f32(v_src1))), v_mask)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? src0[j] * scale / src1[j] : 0.0f;
            }
        }
    }
#else
    (void)size;
    (void)src0Base;
    (void)src0Stride;
    (void)src1Base;
    (void)src1Stride;
    (void)dstBase;
    (void)dstStride;
    (void)scale;
#endif
}

void reciprocal(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                f32 scale,
                CONVERT_POLICY cpolicy)
{
    recip<u8>(size, srcBase, srcStride, dstBase, dstStride, scale, cpolicy);
}

void reciprocal(const Size2D &size,
                const s8 * srcBase, ptrdiff_t srcStride,
                s8 * dstBase, ptrdiff_t dstStride,
                f32 scale,
                CONVERT_POLICY cpolicy)
{
    recip<s8>(size, srcBase, srcStride, dstBase, dstStride, scale, cpolicy);
}

void reciprocal(const Size2D &size,
                const u16 * srcBase, ptrdiff_t srcStride,
                u16 * dstBase, ptrdiff_t dstStride,
                f32 scale,
                CONVERT_POLICY cpolicy)
{
    recip<u16>(size, srcBase, srcStride, dstBase, dstStride, scale, cpolicy);
}

void reciprocal(const Size2D &size,
                const s16 * srcBase, ptrdiff_t srcStride,
                s16 * dstBase, ptrdiff_t dstStride,
                f32 scale,
                CONVERT_POLICY cpolicy)
{
    recip<s16>(size, srcBase, srcStride, dstBase, dstStride, scale, cpolicy);
}

void reciprocal(const Size2D &size,
                const s32 * srcBase, ptrdiff_t srcStride,
                s32 * dstBase, ptrdiff_t dstStride,
                f32 scale,
                CONVERT_POLICY cpolicy)
{
    recip<s32>(size, srcBase, srcStride, dstBase, dstStride, scale, cpolicy);
}

void reciprocal(const Size2D &size,
                const f32 * srcBase, ptrdiff_t srcStride,
                f32 * dstBase, ptrdiff_t dstStride,
                f32 scale)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    if (scale == 0.0f)
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            f32 * dst = internal::getRowPtr(dstBase, dstStride, y);
            std::memset(dst, 0, sizeof(f32) * size.width);
        }
        return;
    }

    float32x4_t v_zero = vdupq_n_f32(0.0f);

    size_t roiw128 = size.width >= 3 ? size.width - 3 : 0;
    size_t roiw64 = size.width >= 1 ? size.width - 1 : 0;

    if (std::fabs(scale - 1.0f) < FLT_EPSILON)
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const f32 * src1 = internal::getRowPtr(srcBase, srcStride, i);
            f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw128; j += 4)
            {
                internal::prefetch(src1 + j);

                float32x4_t v_src1 = vld1q_f32(src1 + j);

                uint32x4_t v_mask = vceqq_f32(v_src1,v_zero);
                vst1q_f32(dst + j, vreinterpretq_f32_u32(vbicq_u32(
                                   vreinterpretq_u32_f32(internal::vrecpq_f32(v_src1)), v_mask)));
            }

            for (; j < roiw64; j += 2)
            {
                float32x2_t v_src1 = vld1_f32(src1 + j);

                uint32x2_t v_mask = vceq_f32(v_src1,vget_low_f32(v_zero));
                vst1_f32(dst + j, vreinterpret_f32_u32(vbic_u32(
                                  vreinterpret_u32_f32(internal::vrecp_f32(v_src1)), v_mask)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? 1.0f / src1[j] : 0;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < size.height; ++i)
        {
            const f32 * src1 = internal::getRowPtr(srcBase, srcStride, i);
            f32 * dst = internal::getRowPtr(dstBase, dstStride, i);
            size_t j = 0;

            for (; j < roiw128; j += 4)
            {
                internal::prefetch(src1 + j);

                float32x4_t v_src1 = vld1q_f32(src1 + j);

                uint32x4_t v_mask = vceqq_f32(v_src1,v_zero);
                vst1q_f32(dst + j, vreinterpretq_f32_u32(vbicq_u32(
                                   vreinterpretq_u32_f32(vmulq_n_f32(internal::vrecpq_f32(v_src1),
                                                                     scale)),v_mask)));
            }

            for (; j < roiw64; j += 2)
            {
                float32x2_t v_src1 = vld1_f32(src1 + j);

                uint32x2_t v_mask = vceq_f32(v_src1,vget_low_f32(v_zero));
                vst1_f32(dst + j, vreinterpret_f32_u32(vbic_u32(
                                  vreinterpret_u32_f32(vmul_n_f32(internal::vrecp_f32(v_src1),
                                                                  scale)), v_mask)));
            }

            for (; j < size.width; j++)
            {
                dst[j] = src1[j] ? scale / src1[j] : 0;
            }
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)scale;
#endif
}

} // namespace CAROTENE_NS
