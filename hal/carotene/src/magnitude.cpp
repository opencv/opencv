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

#include <cmath>

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

struct Magnitude
{
    typedef s16 type;

    void operator() (const int16x8_t & v_src0, const int16x8_t & v_src1,
              int16x8_t & v_dst) const
    {
        int16x4_t v_src0_p = vget_low_s16(v_src0), v_src1_p = vget_low_s16(v_src1);
        float32x4_t v_sqr0 = vaddq_f32(vcvtq_f32_s32(vmull_s16(v_src0_p, v_src0_p)),
                                       vcvtq_f32_s32(vmull_s16(v_src1_p, v_src1_p)));
        v_src0_p = vget_high_s16(v_src0);
        v_src1_p = vget_high_s16(v_src1);
        float32x4_t v_sqr1 = vaddq_f32(vcvtq_f32_s32(vmull_s16(v_src0_p, v_src0_p)),
                                       vcvtq_f32_s32(vmull_s16(v_src1_p, v_src1_p)));

        int32x4_t v_sqrt0 = vcvtq_s32_f32(internal::vsqrtq_f32(v_sqr0));
        int32x4_t v_sqrt1 = vcvtq_s32_f32(internal::vsqrtq_f32(v_sqr1));

        v_dst = vcombine_s16(vqmovn_s32(v_sqrt0), vqmovn_s32(v_sqrt1));
    }

    void operator() (const int16x4_t & v_src0, const int16x4_t & v_src1,
              int16x4_t & v_dst) const
    {
        float32x4_t v_tmp = vaddq_f32(vcvtq_f32_s32(vmull_s16(v_src0, v_src0)),
                                      vcvtq_f32_s32(vmull_s16(v_src1, v_src1)));
        int32x4_t v_sqrt = vcvtq_s32_f32(internal::vsqrtq_f32(v_tmp));
        v_dst = vqmovn_s32(v_sqrt);
    }

    void operator() (const short * src0, const short * src1, short * dst) const
    {
        f32 src0val = (f32)src0[0], src1val = (f32)src1[0];
        dst[0] = internal::saturate_cast<s16>((s32)sqrtf(src0val * src0val + src1val * src1val));
    }
};

struct MagnitudeF32
{
    typedef f32 type;

    void operator() (const float32x4_t & v_src0, const float32x4_t & v_src1,
              float32x4_t & v_dst) const
    {
        v_dst = internal::vsqrtq_f32(vaddq_f32(vmulq_f32(v_src0, v_src0), vmulq_f32(v_src1, v_src1)));
    }

    void operator() (const float32x2_t & v_src0, const float32x2_t & v_src1,
              float32x2_t & v_dst) const
    {
        v_dst = internal::vsqrt_f32(vadd_f32(vmul_f32(v_src0, v_src0), vmul_f32(v_src1, v_src1)));
    }

    void operator() (const f32 * src0, const f32 * src1, f32 * dst) const
    {
        dst[0] = sqrtf(src0[0] * src0[0] + src1[0] * src1[0]);
    }
};

} // namespace

#endif

void magnitude(const Size2D &size,
               const s16 * src0Base, ptrdiff_t src0Stride,
               const s16 * src1Base, ptrdiff_t src1Stride,
               s16 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride,
                         Magnitude());
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

void magnitude(const Size2D &size,
               const f32 * src0Base, ptrdiff_t src0Stride,
               const f32 * src1Base, ptrdiff_t src1Stride,
               f32 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    internal::vtransform(size,
                         src0Base, src0Stride,
                         src1Base, src1Stride,
                         dstBase, dstStride,
                         MagnitudeF32());
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
