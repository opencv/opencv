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

#ifdef CAROTENE_NEON

/**
 * This helper header is for rounding from float32xN to uin32xN or int32xN to nearest, ties to even.
 * See https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
 */

// See https://github.com/opencv/opencv/pull/24271#issuecomment-1867318007
#define CAROTENE_ROUND_DELTA (12582912.0f)

namespace CAROTENE_NS { namespace internal {

inline uint32x4_t vroundq_u32_f32(const float32x4_t val)
{
#if CAROTENE_NEON_ARCH >= 8 /* get ready for ARMv9 */
    return vcvtnq_u32_f32(val);
#else
    const float32x4_t delta = vdupq_n_f32(CAROTENE_ROUND_DELTA);
    return vcvtq_u32_f32(vsubq_f32(vaddq_f32(val, delta), delta));
#endif
}

inline uint32x2_t vround_u32_f32(const float32x2_t val)
{
#if CAROTENE_NEON_ARCH >= 8 /* get ready for ARMv9 */
    return vcvtn_u32_f32(val);
#else
    const float32x2_t delta = vdup_n_f32(CAROTENE_ROUND_DELTA);
    return vcvt_u32_f32(vsub_f32(vadd_f32(val, delta), delta));
#endif
}

inline int32x4_t vroundq_s32_f32(const float32x4_t val)
{
#if CAROTENE_NEON_ARCH >= 8 /* get ready for ARMv9 */
    return vcvtnq_s32_f32(val);
#else
    const float32x4_t delta = vdupq_n_f32(CAROTENE_ROUND_DELTA);
    return vcvtq_s32_f32(vsubq_f32(vaddq_f32(val, delta), delta));
#endif
}

inline int32x2_t vround_s32_f32(const float32x2_t val)
{
#if CAROTENE_NEON_ARCH >= 8 /* get ready for ARMv9 */
    return vcvtn_s32_f32(val);
#else
    const float32x2_t delta = vdup_n_f32(CAROTENE_ROUND_DELTA);
    return vcvt_s32_f32(vsub_f32(vadd_f32(val, delta), delta));
#endif
}

} }

#endif // CAROTENE_NEON

#endif
