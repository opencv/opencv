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
#include "vtransform.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

using namespace internal;

template <typename T> struct TypeTraits;
template <> struct TypeTraits< u8> { typedef u16 wide;                     typedef  u8 unsign; typedef  uint8x16_t vec128; };
template <> struct TypeTraits< s8> { typedef s16 wide;                     typedef  u8 unsign; typedef   int8x16_t vec128; };
template <> struct TypeTraits<u16> { typedef u32 wide; typedef  u8 narrow; typedef u16 unsign; typedef  uint16x8_t vec128; };
template <> struct TypeTraits<s16> { typedef s32 wide; typedef  s8 narrow; typedef u16 unsign; typedef   int16x8_t vec128; };
template <> struct TypeTraits<u32> { typedef u64 wide; typedef u16 narrow; typedef u32 unsign; typedef  uint32x4_t vec128; };
template <> struct TypeTraits<s32> { typedef s64 wide; typedef s16 narrow; typedef u32 unsign; typedef   int32x4_t vec128; };
template <> struct TypeTraits<f32> { typedef f64 wide;                                         typedef float32x4_t vec128; };

template <typename T> struct wAdd
{
    typedef T type;

    f32 alpha, beta, gamma;
    typedef typename TypeTraits<T>::wide wtype;
    wAdd<wtype> wideAdd;
    wAdd(f32 _alpha, f32 _beta, f32 _gamma):
        alpha(_alpha), beta(_beta), gamma(_gamma),
        wideAdd(_alpha, _beta, _gamma) {}

    void operator() (const typename VecTraits<T>::vec128 & v_src0,
                     const typename VecTraits<T>::vec128 & v_src1,
                     typename VecTraits<T>::vec128 & v_dst) const
    {
        typename VecTraits<wtype>::vec128 vrl, vrh;
        wideAdd(vmovl( vget_low(v_src0)), vmovl( vget_low(v_src1)), vrl);
        wideAdd(vmovl(vget_high(v_src0)), vmovl(vget_high(v_src1)), vrh);

        v_dst = vcombine(vqmovn(vrl), vqmovn(vrh));
    }

    void operator() (const typename VecTraits<T>::vec64 & v_src0,
                     const typename VecTraits<T>::vec64 & v_src1,
                     typename VecTraits<T>::vec64 & v_dst) const
    {
        typename VecTraits<wtype>::vec128 vr;
        wideAdd(vmovl(v_src0), vmovl(v_src1), vr);

        v_dst = vqmovn(vr);
    }

    void operator() (const T * src0, const T * src1, T * dst) const
    {
        dst[0] = saturate_cast<T>(alpha*src0[0] + beta*src1[0] + gamma);
    }
};

template <> struct wAdd<s32>
{
    typedef s32 type;

    f32 alpha, beta, gamma;
    float32x4_t valpha, vbeta, vgamma;
    wAdd(f32 _alpha, f32 _beta, f32 _gamma):
        alpha(_alpha), beta(_beta), gamma(_gamma)
    {
        valpha = vdupq_n_f32(_alpha);
        vbeta = vdupq_n_f32(_beta);
        vgamma = vdupq_n_f32(_gamma + 0.5);
    }

    void operator() (const VecTraits<s32>::vec128 & v_src0,
                     const VecTraits<s32>::vec128 & v_src1,
                     VecTraits<s32>::vec128 & v_dst) const
    {
        float32x4_t vs1 = vcvtq_f32_s32(v_src0);
        float32x4_t vs2 = vcvtq_f32_s32(v_src1);

        vs1 = vmlaq_f32(vgamma, vs1, valpha);
        vs1 = vmlaq_f32(vs1, vs2, vbeta);
        v_dst = vcvtq_s32_f32(vs1);
    }

    void operator() (const VecTraits<s32>::vec64 & v_src0,
                     const VecTraits<s32>::vec64 & v_src1,
                     VecTraits<s32>::vec64 & v_dst) const
    {
        float32x2_t vs1 = vcvt_f32_s32(v_src0);
        float32x2_t vs2 = vcvt_f32_s32(v_src1);

        vs1 = vmla_f32(vget_low(vgamma), vs1, vget_low(valpha));
        vs1 = vmla_f32(vs1, vs2, vget_low(vbeta));
        v_dst = vcvt_s32_f32(vs1);
    }

    void operator() (const s32 * src0, const s32 * src1, s32 * dst) const
    {
        dst[0] = saturate_cast<s32>(alpha*src0[0] + beta*src1[0] + gamma);
    }
};

template <> struct wAdd<u32>
{
    typedef u32 type;

    f32 alpha, beta, gamma;
    float32x4_t valpha, vbeta, vgamma;
    wAdd(f32 _alpha, f32 _beta, f32 _gamma):
        alpha(_alpha), beta(_beta), gamma(_gamma)
    {
        valpha = vdupq_n_f32(_alpha);
        vbeta = vdupq_n_f32(_beta);
        vgamma = vdupq_n_f32(_gamma + 0.5);
    }

    void operator() (const VecTraits<u32>::vec128 & v_src0,
                     const VecTraits<u32>::vec128 & v_src1,
                     VecTraits<u32>::vec128 & v_dst) const
    {
        float32x4_t vs1 = vcvtq_f32_u32(v_src0);
        float32x4_t vs2 = vcvtq_f32_u32(v_src1);

        vs1 = vmlaq_f32(vgamma, vs1, valpha);
        vs1 = vmlaq_f32(vs1, vs2, vbeta);
        v_dst = vcvtq_u32_f32(vs1);
    }

    void operator() (const VecTraits<u32>::vec64 & v_src0,
                     const VecTraits<u32>::vec64 & v_src1,
                     VecTraits<u32>::vec64 & v_dst) const
    {
        float32x2_t vs1 = vcvt_f32_u32(v_src0);
        float32x2_t vs2 = vcvt_f32_u32(v_src1);

        vs1 = vmla_f32(vget_low(vgamma), vs1, vget_low(valpha));
        vs1 = vmla_f32(vs1, vs2, vget_low(vbeta));
        v_dst = vcvt_u32_f32(vs1);
    }

    void operator() (const u32 * src0, const u32 * src1, u32 * dst) const
    {
        dst[0] = saturate_cast<u32>(alpha*src0[0] + beta*src1[0] + gamma);
    }
};

template <> struct wAdd<f32>
{
    typedef f32 type;

    f32 alpha, beta, gamma;
    float32x4_t valpha, vbeta, vgamma;
    wAdd(f32 _alpha, f32 _beta, f32 _gamma):
        alpha(_alpha), beta(_beta), gamma(_gamma)
    {
        valpha = vdupq_n_f32(_alpha);
        vbeta = vdupq_n_f32(_beta);
        vgamma = vdupq_n_f32(_gamma + 0.5);
    }

    void operator() (const VecTraits<f32>::vec128 & v_src0,
                     const VecTraits<f32>::vec128 & v_src1,
                     VecTraits<f32>::vec128 & v_dst) const
    {
        float32x4_t vs1 = vmlaq_f32(vgamma, v_src0, valpha);
        v_dst = vmlaq_f32(vs1, v_src1, vbeta);
    }

    void operator() (const VecTraits<f32>::vec64 & v_src0,
                     const VecTraits<f32>::vec64 & v_src1,
                     VecTraits<f32>::vec64 & v_dst) const
    {
        float32x2_t vs1 = vmla_f32(vget_low(vgamma), v_src0, vget_low(valpha));
        v_dst = vmla_f32(vs1, v_src1, vget_low(vbeta));

    }

    void operator() (const f32 * src0, const f32 * src1, f32 * dst) const
    {
        dst[0] = alpha*src0[0] + beta*src1[0] + gamma;
    }
};

} // namespace

#define IMPL_ADDWEIGHTED(type)                                \
void addWeighted(const Size2D &size,                          \
                 const type * src0Base, ptrdiff_t src0Stride, \
                 const type * src1Base, ptrdiff_t src1Stride, \
                 type * dstBase, ptrdiff_t dstStride,         \
                 f32 alpha, f32 beta, f32 gamma)              \
{                                                             \
    internal::assertSupportedConfiguration();                 \
    wAdd<type> wgtAdd(alpha,                                  \
                      beta,                                   \
                      gamma);                                 \
    internal::vtransform(size,                                \
                         src0Base, src0Stride,                \
                         src1Base, src1Stride,                \
                         dstBase, dstStride,                  \
                         wgtAdd);                             \
}

#else

#define IMPL_ADDWEIGHTED(type)                                \
void addWeighted(const Size2D &,                              \
                 const type *, ptrdiff_t,                     \
                 const type *, ptrdiff_t,                     \
                 type *, ptrdiff_t,                           \
                 f32, f32, f32)                               \
{                                                             \
    internal::assertSupportedConfiguration();                 \
}

#endif

IMPL_ADDWEIGHTED(u8)
IMPL_ADDWEIGHTED(s8)
IMPL_ADDWEIGHTED(u16)
IMPL_ADDWEIGHTED(s16)
IMPL_ADDWEIGHTED(u32)
IMPL_ADDWEIGHTED(s32)
IMPL_ADDWEIGHTED(f32)

} // namespace CAROTENE_NS
