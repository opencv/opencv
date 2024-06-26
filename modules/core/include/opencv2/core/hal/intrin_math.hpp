// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This header is not standalone. Don't include directly, use "intrin.hpp" instead.
#ifdef OPENCV_HAL_INTRIN_HPP  // defined in intrin.hpp

namespace CV__SIMD_NAMESPACE {

/* Universal Intrinsics implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2010,2011  RJVB - extensions */
/* Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef OPENCV_HAL_MATH_HAVE_EXP

//! @name Exponential
//! @{
#if defined(CV_SIMD_FP16) && CV_SIMD_FP16
    // Implementation is the same as float32 vector.
    inline v_float16 v_exp(const v_float16 &x) {
        const v_float16 _vexp_lo_f16 = vx_setall_f16(-10.7421875f);
        const v_float16 _vexp_hi_f16 = vx_setall_f16(11.f);
        const v_float16 _vexp_half_fp16 = vx_setall_f16(0.5f);
        const v_float16 _vexp_one_fp16 = vx_setall_f16(1.f);
        const v_float16 _vexp_LOG2EF_f16 = vx_setall_f16(1.44269504088896341f);
        const v_float16 _vexp_C1_f16 = vx_setall_f16(-6.93359375E-1f);
        const v_float16 _vexp_C2_f16 = vx_setall_f16(2.12194440E-4f);
        const v_float16 _vexp_p0_f16 = vx_setall_f16(1.9875691500E-4f);
        const v_float16 _vexp_p1_f16 = vx_setall_f16(1.3981999507E-3f);
        const v_float16 _vexp_p2_f16 = vx_setall_f16(8.3334519073E-3f);
        const v_float16 _vexp_p3_f16 = vx_setall_f16(4.1665795894E-2f);
        const v_float16 _vexp_p4_f16 = vx_setall_f16(1.6666665459E-1f);
        const v_float16 _vexp_p5_f16 = vx_setall_f16(5.0000001201E-1f);
        const v_int16 _vexp_bias_s16 = vx_setall_s16(0xf);

        v_float16 _vexp_, _vexp_x, _vexp_y, _vexp_xx;
        v_int16 _vexp_mm;

        // compute exponential of x
        _vexp_x = v_max(x, _vexp_lo_f16);
        _vexp_x = v_min(_vexp_x, _vexp_hi_f16);

        _vexp_ = v_fma(_vexp_x, _vexp_LOG2EF_f16, _vexp_half_fp16);
        _vexp_mm = v_floor(_vexp_);
        _vexp_ = v_cvt_f16(_vexp_mm);
        _vexp_mm = v_add(_vexp_mm, _vexp_bias_s16);
        _vexp_mm = v_shl(_vexp_mm, 10);

        _vexp_x = v_fma(_vexp_, _vexp_C1_f16, _vexp_x);
        _vexp_x = v_fma(_vexp_, _vexp_C2_f16, _vexp_x);
        _vexp_xx = v_mul(_vexp_x, _vexp_x);

        _vexp_y = v_fma(_vexp_x, _vexp_p0_f16, _vexp_p1_f16);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p2_f16);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p3_f16);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p4_f16);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p5_f16);

        _vexp_y = v_fma(_vexp_y, _vexp_xx, _vexp_x);
        _vexp_y = v_add(_vexp_y, _vexp_one_fp16);
        _vexp_y = v_mul(_vexp_y, v_reinterpret_as_f16(_vexp_mm));

        // exp(NAN) -> NAN
        v_float16 mask_not_nan = v_not_nan(x);
        return v_select(mask_not_nan, _vexp_y, v_reinterpret_as_f16(vx_setall_s16(0x7e00)));
    }
#endif

    inline v_float32 v_exp(const v_float32 &x) {
        const v_float32 _vexp_lo_f32 = vx_setall_f32(-88.3762626647949f);
        const v_float32 _vexp_hi_f32 = vx_setall_f32(89.f);
        const v_float32 _vexp_half_fp32 = vx_setall_f32(0.5f);
        const v_float32 _vexp_one_fp32 = vx_setall_f32(1.f);
        const v_float32 _vexp_LOG2EF_f32 = vx_setall_f32(1.44269504088896341f);
        const v_float32 _vexp_C1_f32 = vx_setall_f32(-6.93359375E-1f);
        const v_float32 _vexp_C2_f32 = vx_setall_f32(2.12194440E-4f);
        const v_float32 _vexp_p0_f32 = vx_setall_f32(1.9875691500E-4f);
        const v_float32 _vexp_p1_f32 = vx_setall_f32(1.3981999507E-3f);
        const v_float32 _vexp_p2_f32 = vx_setall_f32(8.3334519073E-3f);
        const v_float32 _vexp_p3_f32 = vx_setall_f32(4.1665795894E-2f);
        const v_float32 _vexp_p4_f32 = vx_setall_f32(1.6666665459E-1f);
        const v_float32 _vexp_p5_f32 = vx_setall_f32(5.0000001201E-1f);
        const v_int32 _vexp_bias_s32 = vx_setall_s32(0x7f);

        v_float32 _vexp_, _vexp_x, _vexp_y, _vexp_xx;
        v_int32 _vexp_mm;

        // compute exponential of x
        _vexp_x = v_max(x, _vexp_lo_f32);
        _vexp_x = v_min(_vexp_x, _vexp_hi_f32);

        _vexp_ = v_fma(_vexp_x, _vexp_LOG2EF_f32, _vexp_half_fp32);
        _vexp_mm = v_floor(_vexp_);
        _vexp_ = v_cvt_f32(_vexp_mm);
        _vexp_mm = v_add(_vexp_mm, _vexp_bias_s32);
        _vexp_mm = v_shl(_vexp_mm, 23);

        _vexp_x = v_fma(_vexp_, _vexp_C1_f32, _vexp_x);
        _vexp_x = v_fma(_vexp_, _vexp_C2_f32, _vexp_x);
        _vexp_xx = v_mul(_vexp_x, _vexp_x);

        _vexp_y = v_fma(_vexp_x, _vexp_p0_f32, _vexp_p1_f32);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p2_f32);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p3_f32);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p4_f32);
        _vexp_y = v_fma(_vexp_y, _vexp_x, _vexp_p5_f32);

        _vexp_y = v_fma(_vexp_y, _vexp_xx, _vexp_x);
        _vexp_y = v_add(_vexp_y, _vexp_one_fp32);
        _vexp_y = v_mul(_vexp_y, v_reinterpret_as_f32(_vexp_mm));

        // exp(NAN) -> NAN
        v_float32 mask_not_nan = v_not_nan(x);
        return v_select(mask_not_nan, _vexp_y, v_reinterpret_as_f32(vx_setall_s32(0x7fc00000)));
    }

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    inline v_float64 v_exp(const v_float64 &x) {
        const v_float64 _vexp_lo_f64 = vx_setall_f64(-709.43613930310391424428);
        const v_float64 _vexp_hi_f64 = vx_setall_f64(710.);
        const v_float64 _vexp_half_f64 = vx_setall_f64(0.5);
        const v_float64 _vexp_one_f64 = vx_setall_f64(1.0);
        const v_float64 _vexp_two_f64 = vx_setall_f64(2.0);
        const v_float64 _vexp_LOG2EF_f64 = vx_setall_f64(1.44269504088896340736);
        const v_float64 _vexp_C1_f64 = vx_setall_f64(-6.93145751953125E-1);
        const v_float64 _vexp_C2_f64 = vx_setall_f64(-1.42860682030941723212E-6);
        const v_float64 _vexp_p0_f64 = vx_setall_f64(1.26177193074810590878E-4);
        const v_float64 _vexp_p1_f64 = vx_setall_f64(3.02994407707441961300E-2);
        const v_float64 _vexp_p2_f64 = vx_setall_f64(9.99999999999999999910E-1);
        const v_float64 _vexp_q0_f64 = vx_setall_f64(3.00198505138664455042E-6);
        const v_float64 _vexp_q1_f64 = vx_setall_f64(2.52448340349684104192E-3);
        const v_float64 _vexp_q2_f64 = vx_setall_f64(2.27265548208155028766E-1);
        const v_float64 _vexp_q3_f64 = vx_setall_f64(2.00000000000000000009E0);
        const v_int64 _vexp_bias_s64 = vx_setall_s64(0x3ff);

        v_float64 _vexp_, _vexp_x, _vexp_y, _vexp_z, _vexp_xx;
        v_int64 _vexp_mm;

        // compute exponential of x
        _vexp_x = v_max(x, _vexp_lo_f64);
        _vexp_x = v_min(_vexp_x, _vexp_hi_f64);

        _vexp_ = v_fma(_vexp_x, _vexp_LOG2EF_f64, _vexp_half_f64);
        _vexp_mm = v_expand_low(v_floor(_vexp_));
        _vexp_ = v_cvt_f64(_vexp_mm);
        _vexp_mm = v_add(_vexp_mm, _vexp_bias_s64);
        _vexp_mm = v_shl(_vexp_mm, 52);

        _vexp_x = v_fma(_vexp_, _vexp_C1_f64, _vexp_x);
        _vexp_x = v_fma(_vexp_, _vexp_C2_f64, _vexp_x);
        _vexp_xx = v_mul(_vexp_x, _vexp_x);

        _vexp_y = v_fma(_vexp_xx, _vexp_p0_f64, _vexp_p1_f64);
        _vexp_y = v_fma(_vexp_y, _vexp_xx, _vexp_p2_f64);
        _vexp_y = v_mul(_vexp_y, _vexp_x);

        _vexp_z = v_fma(_vexp_xx, _vexp_q0_f64, _vexp_q1_f64);
        _vexp_z = v_fma(_vexp_xx, _vexp_z, _vexp_q2_f64);
        _vexp_z = v_fma(_vexp_xx, _vexp_z, _vexp_q3_f64);

        _vexp_z = v_div(_vexp_y, v_sub(_vexp_z, _vexp_y));
        _vexp_z = v_fma(_vexp_two_f64, _vexp_z, _vexp_one_f64);
        _vexp_z = v_mul(_vexp_z, v_reinterpret_as_f64(_vexp_mm));

        // exp(NAN) -> NAN
        v_float64 mask_not_nan = v_not_nan(x);
        return v_select(mask_not_nan, _vexp_z, v_reinterpret_as_f64(vx_setall_s64(0x7FF8000000000000)));
    }
#endif

#define OPENCV_HAL_MATH_HAVE_EXP 1
//! @}

#endif
}
#endif  // OPENCV_HAL_INTRIN_HPP
