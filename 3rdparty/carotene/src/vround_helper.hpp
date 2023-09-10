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

#ifndef CAROTENE_SRC_VROUND_HELPER_HPP
#define CAROTENE_SRC_VROUND_HELPER_HPP

#include "common.hpp"
#include "vtransform.hpp"
#include <iostream>

#ifdef CAROTENE_NEON

/**
 * This helper header is for rounding from float32xN to uin32xN or int32xN to nearest, ties to even.
 * See https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
 */

namespace CAROTENE_NS { namespace internal {

#if ( defined(__aarch64__) || defined(__aarch32__) )
#    undef  CAROTENE_ROUNDING_ARMV7
#    define CAROTENE_ROUNDING_ARMV8
#else
#    define CAROTENE_ROUNDING_ARMV7
#    if defined( CAROTENE_ROUNDING_ARMV8 )
#        error " ARMv7 doesn't support A32/A64 Neon Instructions."
#    endif
#endif

#if defined(CAROTENE_ROUNDING_LEGACY)
#    undef CAROTENE_ROUNDING_ARMV7
#    undef CAROTENE_ROUNDING_ARMV8
#endif

inline uint32x4_t vroundq_u32_f32(const float32x4_t val)
{
#ifdef CAROTENE_ROUNDING_ARMV7
    static const float32x4_t v0_5_f32 = vdupq_n_f32(0.5);
    static const uint32x4_t  v1_0_u32 = vdupq_n_u32(1);

    const uint32x4_t  round      = vcvtq_u32_f32( vaddq_f32(val, v0_5_f32 ) );
    const uint32x4_t  odd        = vandq_u32( round, v1_0_u32 );
    const float32x4_t diff       = vsubq_f32( vcvtq_f32_u32(round), val );
    const uint32x4_t  round_down = vandq_u32( odd, vceqq_f32(diff, v0_5_f32 ) );
    const uint32x4_t  ret        = vsubq_u32( round, round_down );

    return ret;
#elif defined( CAROTENE_ROUNDING_ARMV8 )
    return vcvtnq_u32_f32(val);
#else // CAROTENE_ROUNDING_LEGACY
    static const float32x4_t v0_5_f32 = vdupq_n_f32(0.5);
    return vcvtq_u32_f32( vaddq_f32(val, v0_5_f32 ));
#endif
}

inline uint32x2_t vround_u32_f32(const float32x2_t val)
{
#ifdef CAROTENE_ROUNDING_ARMV7
    static const float32x2_t v0_5_f32 = vdup_n_f32(0.5);
    static const uint32x2_t  v1_0_u32 = vdup_n_u32(1);

    const uint32x2_t  round      = vcvt_u32_f32( vadd_f32(val, v0_5_f32 ) );
    const uint32x2_t  odd        = vand_u32( round, v1_0_u32 );
    const float32x2_t diff       = vsub_f32( vcvt_f32_u32(round), val );
    const uint32x2_t  round_down = vand_u32( odd, vceq_f32(diff, v0_5_f32 ) );
    const uint32x2_t  ret        = vsub_u32( round, round_down );

    return ret;

#elif defined( CAROTENE_ROUNDING_ARMV8 )
    return vcvtn_u32_f32(val);

#else
    static const float32x2_t v0_5_f32 = vdup_n_f32(0.5);
    return vcvt_u32_f32( vadd_f32(val, v0_5_f32) );
#endif
}

inline int32x4_t vroundq_s32_f32(const float32x4_t val)
{
#ifdef CAROTENE_ROUNDING_ARMV7
    static const float32x4_t v0_0_f32  = vdupq_n_f32(0.0);
    static const float32x4_t v0_5_f32  = vdupq_n_f32(0.5);
    static const int32x4_t   v1_0_s32  = vdupq_n_s32(1);

    const int32x4_t val_positive = vreinterpretq_s32_u32( vcgtq_f32( val, v0_0_f32 ) );
    const int32x4_t ret_signs    = vsubq_s32(
                                       vandq_s32( v1_0_s32, val_positive ),
                                       vbicq_s32( v1_0_s32, val_positive ) );

    const float32x4_t val_abs    = vabsq_f32( val );
    const int32x4_t   round      = vcvtq_s32_f32( vaddq_f32( val_abs, v0_5_f32 ) );
    const int32x4_t   odd        = vandq_s32( round, v1_0_s32 );
    const float32x4_t diff       = vsubq_f32( vcvtq_f32_s32(round), val_abs);
    const int32x4_t   round_down = vandq_s32( odd, vreinterpretq_s32_u32( vceqq_f32( diff,v0_5_f32 ) ) );
    const int32x4_t   ret_abs    = vsubq_s32( round, round_down );

    const int32x4_t   ret        = vmulq_s32( ret_abs, ret_signs );

    return ret;

#elif defined( CAROTENE_ROUNDING_ARMV8 )
    return vcvtnq_s32_f32(val);

#else
    static const float32x4_t v0_0_f32  = vdupq_n_f32(0.0);
    static const float32x4_t v0_5_f32  = vdupq_n_f32(0.5);
    static const int32x4_t   v1_0_s32  = vdupq_n_s32(1);

    const int32x4_t val_positive = vreinterpretq_s32_u32( vcgtq_f32( val, v0_0_f32 ) );
    const int32x4_t ret_signs    = vsubq_s32(
                                       vandq_s32( v1_0_s32, val_positive ),
                                       vbicq_s32( v1_0_s32, val_positive ) );

    const float32x4_t val_abs    = vabsq_f32( val );
    const int32x4_t   ret_abs    = vcvtq_s32_f32( vaddq_f32( val_abs, v0_5_f32 ) );

    const int32x4_t   ret        = vmulq_s32( ret_abs, ret_signs );

    return ret;


#endif
}

inline int32x2_t vround_s32_f32(const float32x2_t val)
{
#ifdef CAROTENE_ROUNDING_ARMV7
    static const float32x2_t v0_0_f32  = vdup_n_f32(0.0);
    static const float32x2_t v0_5_f32  = vdup_n_f32(0.5);
    static const int32x2_t   v1_0_s32  = vdup_n_s32(1);

    const int32x2_t val_positive = vreinterpret_s32_u32( vcgt_f32( val, v0_0_f32 ) );
    const int32x2_t ret_signs    = vsub_s32(
                                       vand_s32( v1_0_s32, val_positive ),
                                       vbic_s32( v1_0_s32, val_positive ) );

    const float32x2_t val_abs    = vabs_f32( val );
    const int32x2_t   round      = vcvt_s32_f32( vadd_f32( val_abs, v0_5_f32 ) );
    const int32x2_t   odd        = vand_s32( round, v1_0_s32 );
    const float32x2_t diff       = vsub_f32( vcvt_f32_s32(round), val_abs);
    const int32x2_t   round_down = vand_s32( odd, vreinterpret_s32_u32( vceq_f32( diff,v0_5_f32 ) ) );
    const int32x2_t   ret_abs    = vsub_s32( round, round_down );

    const int32x2_t   ret        = vmul_s32( ret_abs, ret_signs );

    return ret;

#elif defined( CAROTENE_ROUNDING_ARMV8 )
    return vcvtn_s32_f32(val);

#else
    static const float32x2_t v0_0_f32  = vdup_n_f32(0.0);
    static const float32x2_t v0_5_f32  = vdup_n_f32(0.5);
    static const int32x2_t   v1_0_s32  = vdup_n_s32(1);

    const int32x2_t val_positive = vreinterpret_s32_u32( vcgt_f32( val, v0_0_f32 ) );
    const int32x2_t ret_signs    = vsub_s32(
                                       vand_s32( v1_0_s32, val_positive ),
                                       vbic_s32( v1_0_s32, val_positive ) );

    const float32x2_t val_abs    = vabs_f32( val );
    const int32x2_t   ret_abs    = vcvt_s32_f32( vadd_f32( val_abs, v0_5_f32 ) );

    const int32x2_t   ret        = vmul_s32( ret_abs, ret_signs );
    return ret;

#endif
}

} }

#endif // CAROTENE_NEON

#endif
