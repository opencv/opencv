/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HAL_NEON_UTILS_HPP
#define OPENCV_HAL_NEON_UTILS_HPP

#include "opencv2/core/cvdef.h"

//! @addtogroup core_utils_neon
//! @{

#if CV_NEON

inline int32x2_t cv_vrnd_s32_f32(float32x2_t v)
{
    static int32x2_t v_sign = vdup_n_s32(1 << 31),
        v_05 = vreinterpret_s32_f32(vdup_n_f32(0.5f));

    int32x2_t v_addition = vorr_s32(v_05, vand_s32(v_sign, vreinterpret_s32_f32(v)));
    return vcvt_s32_f32(vadd_f32(v, vreinterpret_f32_s32(v_addition)));
}

inline int32x4_t cv_vrndq_s32_f32(float32x4_t v)
{
    static int32x4_t v_sign = vdupq_n_s32(1 << 31),
        v_05 = vreinterpretq_s32_f32(vdupq_n_f32(0.5f));

    int32x4_t v_addition = vorrq_s32(v_05, vandq_s32(v_sign, vreinterpretq_s32_f32(v)));
    return vcvtq_s32_f32(vaddq_f32(v, vreinterpretq_f32_s32(v_addition)));
}

inline uint32x2_t cv_vrnd_u32_f32(float32x2_t v)
{
    static float32x2_t v_05 = vdup_n_f32(0.5f);
    return vcvt_u32_f32(vadd_f32(v, v_05));
}

inline uint32x4_t cv_vrndq_u32_f32(float32x4_t v)
{
    static float32x4_t v_05 = vdupq_n_f32(0.5f);
    return vcvtq_u32_f32(vaddq_f32(v, v_05));
}

inline float32x4_t cv_vrecpq_f32(float32x4_t val)
{
    float32x4_t reciprocal = vrecpeq_f32(val);
    reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
    return reciprocal;
}

inline float32x2_t cv_vrecp_f32(float32x2_t val)
{
    float32x2_t reciprocal = vrecpe_f32(val);
    reciprocal = vmul_f32(vrecps_f32(val, reciprocal), reciprocal);
    reciprocal = vmul_f32(vrecps_f32(val, reciprocal), reciprocal);
    return reciprocal;
}

inline float32x4_t cv_vrsqrtq_f32(float32x4_t val)
{
    float32x4_t e = vrsqrteq_f32(val);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);
    return e;
}

inline float32x2_t cv_vrsqrt_f32(float32x2_t val)
{
    float32x2_t e = vrsqrte_f32(val);
    e = vmul_f32(vrsqrts_f32(vmul_f32(e, e), val), e);
    e = vmul_f32(vrsqrts_f32(vmul_f32(e, e), val), e);
    return e;
}

inline float32x4_t cv_vsqrtq_f32(float32x4_t val)
{
    return cv_vrecpq_f32(cv_vrsqrtq_f32(val));
}

inline float32x2_t cv_vsqrt_f32(float32x2_t val)
{
    return cv_vrecp_f32(cv_vrsqrt_f32(val));
}

#endif

//! @}

#endif // OPENCV_HAL_NEON_UTILS_HPP
