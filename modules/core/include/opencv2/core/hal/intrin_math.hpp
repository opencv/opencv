// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


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
#ifndef OPENCV_HAL_INTRIN_MATH_HPP
#define OPENCV_HAL_INTRIN_MATH_HPP

//! @name Exponential
//! @{
// Implementation is the same as float32 vector.
template<typename _TpVec16F, typename _TpVec16S>
inline _TpVec16F v_exp_default_16f(const _TpVec16F &x) {
    const _TpVec16F _vexp_lo_f16 = v_setall_<_TpVec16F>(-10.7421875f);
    const _TpVec16F _vexp_hi_f16 = v_setall_<_TpVec16F>(11.f);
    const _TpVec16F _vexp_half_fp16 = v_setall_<_TpVec16F>(0.5f);
    const _TpVec16F _vexp_one_fp16 = v_setall_<_TpVec16F>(1.f);
    const _TpVec16F _vexp_LOG2EF_f16 = v_setall_<_TpVec16F>(1.44269504088896341f);
    const _TpVec16F _vexp_C1_f16 = v_setall_<_TpVec16F>(-6.93359375E-1f);
    const _TpVec16F _vexp_C2_f16 = v_setall_<_TpVec16F>(2.12194440E-4f);
    const _TpVec16F _vexp_p0_f16 = v_setall_<_TpVec16F>(1.9875691500E-4f);
    const _TpVec16F _vexp_p1_f16 = v_setall_<_TpVec16F>(1.3981999507E-3f);
    const _TpVec16F _vexp_p2_f16 = v_setall_<_TpVec16F>(8.3334519073E-3f);
    const _TpVec16F _vexp_p3_f16 = v_setall_<_TpVec16F>(4.1665795894E-2f);
    const _TpVec16F _vexp_p4_f16 = v_setall_<_TpVec16F>(1.6666665459E-1f);
    const _TpVec16F _vexp_p5_f16 = v_setall_<_TpVec16F>(5.0000001201E-1f);

    _TpVec16F _vexp_, _vexp_x, _vexp_y, _vexp_xx;
    _TpVec16S _vexp_mm;
    const _TpVec16S _vexp_bias_s16 = v_setall_<_TpVec16S>((short)0xf);

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
    _TpVec16F mask_not_nan = v_not_nan(x);
    return v_select(mask_not_nan, _vexp_y, v_reinterpret_as_f16(v_setall_<_TpVec16S>((short)0x7e00)));
}

template<typename _TpVec32F, typename _TpVec32S>
inline _TpVec32F v_exp_default_32f(const _TpVec32F &x) {
    const _TpVec32F _vexp_lo_f32 = v_setall_<_TpVec32F>(-88.3762626647949f);
    const _TpVec32F _vexp_hi_f32 = v_setall_<_TpVec32F>(89.f);
    const _TpVec32F _vexp_half_fp32 = v_setall_<_TpVec32F>(0.5f);
    const _TpVec32F _vexp_one_fp32 = v_setall_<_TpVec32F>(1.f);
    const _TpVec32F _vexp_LOG2EF_f32 = v_setall_<_TpVec32F>(1.44269504088896341f);
    const _TpVec32F _vexp_C1_f32 = v_setall_<_TpVec32F>(-6.93359375E-1f);
    const _TpVec32F _vexp_C2_f32 = v_setall_<_TpVec32F>(2.12194440E-4f);
    const _TpVec32F _vexp_p0_f32 = v_setall_<_TpVec32F>(1.9875691500E-4f);
    const _TpVec32F _vexp_p1_f32 = v_setall_<_TpVec32F>(1.3981999507E-3f);
    const _TpVec32F _vexp_p2_f32 = v_setall_<_TpVec32F>(8.3334519073E-3f);
    const _TpVec32F _vexp_p3_f32 = v_setall_<_TpVec32F>(4.1665795894E-2f);
    const _TpVec32F _vexp_p4_f32 = v_setall_<_TpVec32F>(1.6666665459E-1f);
    const _TpVec32F _vexp_p5_f32 = v_setall_<_TpVec32F>(5.0000001201E-1f);

    _TpVec32F _vexp_, _vexp_x, _vexp_y, _vexp_xx;
    _TpVec32S _vexp_mm;
    const _TpVec32S _vexp_bias_s32 = v_setall_<_TpVec32S>((int)0x7f);

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
    _TpVec32F mask_not_nan = v_not_nan(x);
    return v_select(mask_not_nan, _vexp_y, v_reinterpret_as_f32(v_setall_<_TpVec32S>((int)0x7fc00000)));
}

template<typename _TpVec64F, typename _TpVec64S>
inline _TpVec64F v_exp_default_64f(const _TpVec64F &x) {
    const _TpVec64F _vexp_lo_f64 = v_setall_<_TpVec64F>(-709.43613930310391424428);
    const _TpVec64F _vexp_hi_f64 = v_setall_<_TpVec64F>(710.);
    const _TpVec64F _vexp_half_f64 = v_setall_<_TpVec64F>(0.5);
    const _TpVec64F _vexp_one_f64 = v_setall_<_TpVec64F>(1.0);
    const _TpVec64F _vexp_two_f64 = v_setall_<_TpVec64F>(2.0);
    const _TpVec64F _vexp_LOG2EF_f64 = v_setall_<_TpVec64F>(1.44269504088896340736);
    const _TpVec64F _vexp_C1_f64 = v_setall_<_TpVec64F>(-6.93145751953125E-1);
    const _TpVec64F _vexp_C2_f64 = v_setall_<_TpVec64F>(-1.42860682030941723212E-6);
    const _TpVec64F _vexp_p0_f64 = v_setall_<_TpVec64F>(1.26177193074810590878E-4);
    const _TpVec64F _vexp_p1_f64 = v_setall_<_TpVec64F>(3.02994407707441961300E-2);
    const _TpVec64F _vexp_p2_f64 = v_setall_<_TpVec64F>(9.99999999999999999910E-1);
    const _TpVec64F _vexp_q0_f64 = v_setall_<_TpVec64F>(3.00198505138664455042E-6);
    const _TpVec64F _vexp_q1_f64 = v_setall_<_TpVec64F>(2.52448340349684104192E-3);
    const _TpVec64F _vexp_q2_f64 = v_setall_<_TpVec64F>(2.27265548208155028766E-1);
    const _TpVec64F _vexp_q3_f64 = v_setall_<_TpVec64F>(2.00000000000000000009E0);

    _TpVec64F _vexp_, _vexp_x, _vexp_y, _vexp_z, _vexp_xx;
    _TpVec64S _vexp_mm;
    const _TpVec64S _vexp_bias_s64 = v_setall_<_TpVec64S>((int64)0x3ff);

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
    _TpVec64F mask_not_nan = v_not_nan(x);
    return v_select(mask_not_nan, _vexp_z, v_reinterpret_as_f64(v_setall_<_TpVec64S>((int64)0x7FF8000000000000)));
}
//! @}

//! @name Natural Logarithm
//! @{
template<typename _TpVec16F, typename _TpVec16S>
inline _TpVec16F v_log_default_16f(const _TpVec16F &x) {
    const _TpVec16F _vlog_one_fp16 = v_setall_<_TpVec16F>(1.0f);
    const _TpVec16F _vlog_SQRTHF_fp16 = v_setall_<_TpVec16F>(0.707106781186547524f);
    const _TpVec16F _vlog_q1_fp16 = v_setall_<_TpVec16F>(-2.12194440E-4f);
    const _TpVec16F _vlog_q2_fp16 = v_setall_<_TpVec16F>(0.693359375f);
    const _TpVec16F _vlog_p0_fp16 = v_setall_<_TpVec16F>(7.0376836292E-2f);
    const _TpVec16F _vlog_p1_fp16 = v_setall_<_TpVec16F>(-1.1514610310E-1f);
    const _TpVec16F _vlog_p2_fp16 = v_setall_<_TpVec16F>(1.1676998740E-1f);
    const _TpVec16F _vlog_p3_fp16 = v_setall_<_TpVec16F>(-1.2420140846E-1f);
    const _TpVec16F _vlog_p4_fp16 = v_setall_<_TpVec16F>(1.4249322787E-1f);
    const _TpVec16F _vlog_p5_fp16 = v_setall_<_TpVec16F>(-1.6668057665E-1f);
    const _TpVec16F _vlog_p6_fp16 = v_setall_<_TpVec16F>(2.0000714765E-1f);
    const _TpVec16F _vlog_p7_fp16 = v_setall_<_TpVec16F>(-2.4999993993E-1f);
    const _TpVec16F _vlog_p8_fp16 = v_setall_<_TpVec16F>(3.3333331174E-1f);

    _TpVec16F _vlog_x, _vlog_e, _vlog_y, _vlog_z, _vlog_tmp;
    _TpVec16S _vlog_ux, _vlog_emm0;
    const _TpVec16S _vlog_inv_mant_mask_s16 = v_setall_<_TpVec16S>((short)~0x7c00);

    _vlog_ux = v_reinterpret_as_s16(x);
    _vlog_emm0 = v_shr(_vlog_ux, 10);

    _vlog_ux = v_and(_vlog_ux, _vlog_inv_mant_mask_s16);
    _vlog_ux = v_or(_vlog_ux, v_reinterpret_as_s16(v_setall_<_TpVec16F>(0.5f)));
    _vlog_x = v_reinterpret_as_f16(_vlog_ux);

    _vlog_emm0 = v_sub(_vlog_emm0, v_setall_<_TpVec16S>((short)0xf));
    _vlog_e = v_cvt_f16(_vlog_emm0);

    _vlog_e = v_add(_vlog_e, _vlog_one_fp16);

    _TpVec16F _vlog_mask = v_lt(_vlog_x, _vlog_SQRTHF_fp16);
    _vlog_tmp = v_and(_vlog_x, _vlog_mask);
    _vlog_x = v_sub(_vlog_x, _vlog_one_fp16);
    _vlog_e = v_sub(_vlog_e, v_and(_vlog_one_fp16, _vlog_mask));
    _vlog_x = v_add(_vlog_x, _vlog_tmp);

    _vlog_z = v_mul(_vlog_x, _vlog_x);

    _vlog_y = v_fma(_vlog_p0_fp16, _vlog_x, _vlog_p1_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p2_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p3_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p4_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p5_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p6_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p7_fp16);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p8_fp16);
    _vlog_y = v_mul(_vlog_y, _vlog_x);
    _vlog_y = v_mul(_vlog_y, _vlog_z);

    _vlog_y = v_fma(_vlog_e, _vlog_q1_fp16, _vlog_y);

    _vlog_y = v_sub(_vlog_y, v_mul(_vlog_z, v_setall_<_TpVec16F>(0.5f)));

    _vlog_x = v_add(_vlog_x, _vlog_y);
    _vlog_x = v_fma(_vlog_e, _vlog_q2_fp16, _vlog_x);
    // log(0) -> -INF
    _TpVec16F mask_zero = v_eq(x, v_setzero_<_TpVec16F>());
    _vlog_x = v_select(mask_zero, v_reinterpret_as_f16(v_setall_<_TpVec16S>((short)0xfc00)), _vlog_x);
    // log(NEG), log(NAN) -> NAN
    _TpVec16F mask_not_nan = v_ge(x, v_setzero_<_TpVec16F>());
    _vlog_x = v_select(mask_not_nan, _vlog_x, v_reinterpret_as_f16(v_setall_<_TpVec16S>((short)0x7e00)));
    // log(INF) -> INF
    _TpVec16F mask_inf = v_eq(x, v_reinterpret_as_f16(v_setall_<_TpVec16S>((short)0x7c00)));
    _vlog_x = v_select(mask_inf, x, _vlog_x);
    return _vlog_x;
}

template<typename _TpVec32F, typename _TpVec32S>
inline _TpVec32F v_log_default_32f(const _TpVec32F &x) {
    const _TpVec32F _vlog_one_fp32 = v_setall_<_TpVec32F>(1.0f);
    const _TpVec32F _vlog_SQRTHF_fp32 = v_setall_<_TpVec32F>(0.707106781186547524f);
    const _TpVec32F _vlog_q1_fp32 = v_setall_<_TpVec32F>(-2.12194440E-4f);
    const _TpVec32F _vlog_q2_fp32 = v_setall_<_TpVec32F>(0.693359375f);
    const _TpVec32F _vlog_p0_fp32 = v_setall_<_TpVec32F>(7.0376836292E-2f);
    const _TpVec32F _vlog_p1_fp32 = v_setall_<_TpVec32F>(-1.1514610310E-1f);
    const _TpVec32F _vlog_p2_fp32 = v_setall_<_TpVec32F>(1.1676998740E-1f);
    const _TpVec32F _vlog_p3_fp32 = v_setall_<_TpVec32F>(-1.2420140846E-1f);
    const _TpVec32F _vlog_p4_fp32 = v_setall_<_TpVec32F>(1.4249322787E-1f);
    const _TpVec32F _vlog_p5_fp32 = v_setall_<_TpVec32F>(-1.6668057665E-1f);
    const _TpVec32F _vlog_p6_fp32 = v_setall_<_TpVec32F>(2.0000714765E-1f);
    const _TpVec32F _vlog_p7_fp32 = v_setall_<_TpVec32F>(-2.4999993993E-1f);
    const _TpVec32F _vlog_p8_fp32 = v_setall_<_TpVec32F>(3.3333331174E-1f);

    _TpVec32F _vlog_x, _vlog_e, _vlog_y, _vlog_z, _vlog_tmp;
    _TpVec32S _vlog_ux, _vlog_emm0;
    const _TpVec32S _vlog_inv_mant_mask_s32 = v_setall_<_TpVec32S>((int)~0x7f800000);

    _vlog_ux = v_reinterpret_as_s32(x);
    _vlog_emm0 = v_shr(_vlog_ux, 23);

    _vlog_ux = v_and(_vlog_ux, _vlog_inv_mant_mask_s32);
    _vlog_ux = v_or(_vlog_ux, v_reinterpret_as_s32(v_setall_<_TpVec32F>(0.5f)));
    _vlog_x = v_reinterpret_as_f32(_vlog_ux);

    _vlog_emm0 = v_sub(_vlog_emm0, v_setall_<_TpVec32S>((int)0x7f));
    _vlog_e = v_cvt_f32(_vlog_emm0);

    _vlog_e = v_add(_vlog_e, _vlog_one_fp32);

    _TpVec32F _vlog_mask = v_lt(_vlog_x, _vlog_SQRTHF_fp32);
    _vlog_tmp = v_and(_vlog_x, _vlog_mask);
    _vlog_x = v_sub(_vlog_x, _vlog_one_fp32);
    _vlog_e = v_sub(_vlog_e, v_and(_vlog_one_fp32, _vlog_mask));
    _vlog_x = v_add(_vlog_x, _vlog_tmp);

    _vlog_z = v_mul(_vlog_x, _vlog_x);

    _vlog_y = v_fma(_vlog_p0_fp32, _vlog_x, _vlog_p1_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p2_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p3_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p4_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p5_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p6_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p7_fp32);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p8_fp32);
    _vlog_y = v_mul(_vlog_y, _vlog_x);
    _vlog_y = v_mul(_vlog_y, _vlog_z);

    _vlog_y = v_fma(_vlog_e, _vlog_q1_fp32, _vlog_y);

    _vlog_y = v_sub(_vlog_y, v_mul(_vlog_z, v_setall_<_TpVec32F>(0.5f)));

    _vlog_x = v_add(_vlog_x, _vlog_y);
    _vlog_x = v_fma(_vlog_e, _vlog_q2_fp32, _vlog_x);
    // log(0) -> -INF
    _TpVec32F mask_zero = v_eq(x, v_setzero_<_TpVec32F>());
    _vlog_x = v_select(mask_zero, v_reinterpret_as_f32(v_setall_<_TpVec32S>((int)0xff800000)), _vlog_x);
    // log(NEG), log(NAN) -> NAN
    _TpVec32F mask_not_nan = v_ge(x, v_setzero_<_TpVec32F>());
    _vlog_x = v_select(mask_not_nan, _vlog_x, v_reinterpret_as_f32(v_setall_<_TpVec32S>((int)0x7fc00000)));
    // log(INF) -> INF
    _TpVec32F mask_inf = v_eq(x, v_reinterpret_as_f32(v_setall_<_TpVec32S>((int)0x7f800000)));
    _vlog_x = v_select(mask_inf, x, _vlog_x);
    return _vlog_x;
}

template<typename _TpVec64F, typename _TpVec64S>
inline _TpVec64F v_log_default_64f(const _TpVec64F &x) {
    const _TpVec64F _vlog_one_fp64 = v_setall_<_TpVec64F>(1.0);
    const _TpVec64F _vlog_SQRTHF_fp64 = v_setall_<_TpVec64F>(0.7071067811865475244);
    const _TpVec64F _vlog_p0_fp64 = v_setall_<_TpVec64F>(1.01875663804580931796E-4);
    const _TpVec64F _vlog_p1_fp64 = v_setall_<_TpVec64F>(4.97494994976747001425E-1);
    const _TpVec64F _vlog_p2_fp64 = v_setall_<_TpVec64F>(4.70579119878881725854);
    const _TpVec64F _vlog_p3_fp64 = v_setall_<_TpVec64F>(1.44989225341610930846E1);
    const _TpVec64F _vlog_p4_fp64 = v_setall_<_TpVec64F>(1.79368678507819816313E1);
    const _TpVec64F _vlog_p5_fp64 = v_setall_<_TpVec64F>(7.70838733755885391666);
    const _TpVec64F _vlog_q0_fp64 = v_setall_<_TpVec64F>(1.12873587189167450590E1);
    const _TpVec64F _vlog_q1_fp64 = v_setall_<_TpVec64F>(4.52279145837532221105E1);
    const _TpVec64F _vlog_q2_fp64 = v_setall_<_TpVec64F>(8.29875266912776603211E1);
    const _TpVec64F _vlog_q3_fp64 = v_setall_<_TpVec64F>(7.11544750618563894466E1);
    const _TpVec64F _vlog_q4_fp64 = v_setall_<_TpVec64F>(2.31251620126765340583E1);

    const _TpVec64F _vlog_C0_fp64 = v_setall_<_TpVec64F>(2.121944400546905827679e-4);
    const _TpVec64F _vlog_C1_fp64 = v_setall_<_TpVec64F>(0.693359375);

    _TpVec64F _vlog_x, _vlog_e, _vlog_y, _vlog_z, _vlog_tmp, _vlog_xx;
    _TpVec64S _vlog_ux, _vlog_emm0;
    const _TpVec64S _vlog_inv_mant_mask_s64 = v_setall_<_TpVec64S>((int64)~0x7ff0000000000000);

    _vlog_ux = v_reinterpret_as_s64(x);
    _vlog_emm0 = v_shr(_vlog_ux, 52);

    _vlog_ux = v_and(_vlog_ux, _vlog_inv_mant_mask_s64);
    _vlog_ux = v_or(_vlog_ux, v_reinterpret_as_s64(v_setall_<_TpVec64F>(0.5)));
    _vlog_x = v_reinterpret_as_f64(_vlog_ux);

    _vlog_emm0 = v_sub(_vlog_emm0, v_setall_<_TpVec64S>((int64)0x3ff));
    _vlog_e = v_cvt_f64(_vlog_emm0);

    _vlog_e = v_add(_vlog_e, _vlog_one_fp64);

    _TpVec64F _vlog_mask = v_lt(_vlog_x, _vlog_SQRTHF_fp64);
    _vlog_tmp = v_and(_vlog_x, _vlog_mask);
    _vlog_x = v_sub(_vlog_x, _vlog_one_fp64);
    _vlog_e = v_sub(_vlog_e, v_and(_vlog_one_fp64, _vlog_mask));
    _vlog_x = v_add(_vlog_x, _vlog_tmp);

    _vlog_xx = v_mul(_vlog_x, _vlog_x);

    _vlog_y = v_fma(_vlog_p0_fp64, _vlog_x, _vlog_p1_fp64);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p2_fp64);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p3_fp64);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p4_fp64);
    _vlog_y = v_fma(_vlog_y, _vlog_x, _vlog_p5_fp64);
    _vlog_y = v_mul(_vlog_y, _vlog_x);
    _vlog_y = v_mul(_vlog_y, _vlog_xx);

    _vlog_z = v_add(_vlog_x, _vlog_q0_fp64);
    _vlog_z = v_fma(_vlog_z, _vlog_x, _vlog_q1_fp64);
    _vlog_z = v_fma(_vlog_z, _vlog_x, _vlog_q2_fp64);
    _vlog_z = v_fma(_vlog_z, _vlog_x, _vlog_q3_fp64);
    _vlog_z = v_fma(_vlog_z, _vlog_x, _vlog_q4_fp64);

    _vlog_z = v_div(_vlog_y, _vlog_z);
    _vlog_z = v_sub(_vlog_z, v_mul(_vlog_e, _vlog_C0_fp64));
    _vlog_z = v_sub(_vlog_z, v_mul(_vlog_xx, v_setall_<_TpVec64F>(0.5)));

    _vlog_z = v_add(_vlog_z, _vlog_x);
    _vlog_z = v_fma(_vlog_e, _vlog_C1_fp64, _vlog_z);

    // log(0) -> -INF
    _TpVec64F mask_zero = v_eq(x, v_setzero_<_TpVec64F>());
    _vlog_z = v_select(mask_zero, v_reinterpret_as_f64(v_setall_<_TpVec64S>((int64)0xfff0000000000000)), _vlog_z);
    // log(NEG), log(NAN) -> NAN
    _TpVec64F mask_not_nan = v_ge(x, v_setzero_<_TpVec64F>());
    _vlog_z = v_select(mask_not_nan, _vlog_z, v_reinterpret_as_f64(v_setall_<_TpVec64S>((int64)0x7ff8000000000000)));
    // log(INF) -> INF
    _TpVec64F mask_inf = v_eq(x, v_reinterpret_as_f64(v_setall_<_TpVec64S>((int64)0x7ff0000000000000)));
    _vlog_z = v_select(mask_inf, x, _vlog_z);
    return _vlog_z;
}
//! @}

//! @name Sine and Cosine
//! @{
template<typename _TpVec16F, typename _TpVec16S>
inline void v_sincos_default_16f(const _TpVec16F &x, _TpVec16F &ysin, _TpVec16F &ycos) {
    const _TpVec16F v_cephes_FOPI = v_setall_<_TpVec16F>(hfloat(1.27323954473516f)); // 4 / M_PI
    const _TpVec16F v_minus_DP1 = v_setall_<_TpVec16F>(hfloat(-0.78515625f));
    const _TpVec16F v_minus_DP2 = v_setall_<_TpVec16F>(hfloat(-2.4187564849853515625E-4f));
    const _TpVec16F v_minus_DP3 = v_setall_<_TpVec16F>(hfloat(-3.77489497744594108E-8f));
    const _TpVec16F v_sincof_p0 = v_setall_<_TpVec16F>(hfloat(-1.9515295891E-4f));
    const _TpVec16F v_sincof_p1 = v_setall_<_TpVec16F>(hfloat(8.3321608736E-3f));
    const _TpVec16F v_sincof_p2 = v_setall_<_TpVec16F>(hfloat(-1.6666654611E-1f));
    const _TpVec16F v_coscof_p0 = v_setall_<_TpVec16F>(hfloat(2.443315711809948E-5f));
    const _TpVec16F v_coscof_p1 = v_setall_<_TpVec16F>(hfloat(-1.388731625493765E-3f));
    const _TpVec16F v_coscof_p2 = v_setall_<_TpVec16F>(hfloat(4.166664568298827E-2f));
    const _TpVec16F v_nan = v_reinterpret_as_f16(v_setall_<_TpVec16S>((short)0x7e00));
    const _TpVec16F v_neg_zero = v_setall_<_TpVec16F>(hfloat(-0.f));

    _TpVec16F _vx, _vy, sign_mask_sin, sign_mask_cos;
    _TpVec16S emm2;

    sign_mask_sin = v_lt(x, v_setzero_<_TpVec16F>());
    _vx = v_abs(x);
    _vy = v_mul(_vx, v_cephes_FOPI);

    emm2 = v_trunc(_vy);
    emm2 = v_add(emm2, v_setall_<_TpVec16S>((short)1));
    emm2 = v_and(emm2, v_setall_<_TpVec16S>((short)~1));
    _vy = v_cvt_f16(emm2);

    _TpVec16F poly_mask = v_reinterpret_as_f16(v_eq(v_and(emm2, v_setall_<_TpVec16S>((short)2)), v_setall_<_TpVec16S>((short)0)));

    _vx = v_fma(_vy, v_minus_DP1, _vx);
    _vx = v_fma(_vy, v_minus_DP2, _vx);
    _vx = v_fma(_vy, v_minus_DP3, _vx);

    sign_mask_sin = v_xor(sign_mask_sin, v_reinterpret_as_f16(v_eq(v_and(emm2, v_setall_<_TpVec16S>((short)4)), v_setall_<_TpVec16S>((short)0))));
    sign_mask_cos = v_reinterpret_as_f16(v_eq(v_and(v_sub(emm2, v_setall_<_TpVec16S>((short)2)), v_setall_<_TpVec16S>((short)4)), v_setall_<_TpVec16S>((short)0)));

    _TpVec16F _vxx = v_mul(_vx, _vx);
    _TpVec16F y1, y2;

    y1 = v_fma(v_coscof_p0, _vxx, v_coscof_p1);
    y1 = v_fma(y1, _vxx, v_coscof_p2);
    y1 = v_fma(y1, _vxx, v_setall_<_TpVec16F>(hfloat(-0.5f)));
    y1 = v_fma(y1, _vxx, v_setall_<_TpVec16F>(hfloat(1.f)));

    y2 = v_fma(v_sincof_p0, _vxx, v_sincof_p1);
    y2 = v_fma(y2, _vxx, v_sincof_p2);
    y2 = v_mul(y2, _vxx);
    y2 = v_fma(y2, _vx, _vx);

    ysin = v_select(poly_mask, y2, y1);
    ycos = v_select(poly_mask, y1, y2);
    ysin = v_select(sign_mask_sin, ysin, v_xor(v_neg_zero, ysin));
    ycos = v_select(sign_mask_cos, v_xor(v_neg_zero, ycos), ycos);

    // sincos(NAN) -> NAN, sincos(±INF) -> NAN
    _TpVec16F mask_inf = v_eq(_vx, v_reinterpret_as_f16(v_setall_<_TpVec16S>((short)0x7c00)));
    _TpVec16F mask_nan = v_or(mask_inf, v_ne(x, x));
    ysin = v_select(mask_nan, v_nan, ysin);
    ycos = v_select(mask_nan, v_nan, ycos);
}

template<typename _TpVec16F, typename _TpVec16S>
inline _TpVec16F v_sin_default_16f(const _TpVec16F &x) {
    _TpVec16F ysin, ycos;
    v_sincos_default_16f<_TpVec16F, _TpVec16S>(x, ysin, ycos);
    return ysin;
}

template<typename _TpVec16F, typename _TpVec16S>
inline _TpVec16F v_cos_default_16f(const _TpVec16F &x) {
    _TpVec16F ysin, ycos;
    v_sincos_default_16f<_TpVec16F, _TpVec16S>(x, ysin, ycos);
    return ycos;
}


template<typename _TpVec32F, typename _TpVec32S>
inline void v_sincos_default_32f(const _TpVec32F &x, _TpVec32F &ysin, _TpVec32F &ycos) {
    const _TpVec32F v_cephes_FOPI = v_setall_<_TpVec32F>(1.27323954473516f); // 4 / M_PI
    const _TpVec32F v_minus_DP1 = v_setall_<_TpVec32F>(-0.78515625f);
    const _TpVec32F v_minus_DP2 = v_setall_<_TpVec32F>(-2.4187564849853515625E-4f);
    const _TpVec32F v_minus_DP3 = v_setall_<_TpVec32F>(-3.77489497744594108E-8f);
    const _TpVec32F v_sincof_p0 = v_setall_<_TpVec32F>(-1.9515295891E-4f);
    const _TpVec32F v_sincof_p1 = v_setall_<_TpVec32F>(8.3321608736E-3f);
    const _TpVec32F v_sincof_p2 = v_setall_<_TpVec32F>(-1.6666654611E-1f);
    const _TpVec32F v_coscof_p0 = v_setall_<_TpVec32F>(2.443315711809948E-5f);
    const _TpVec32F v_coscof_p1 = v_setall_<_TpVec32F>(-1.388731625493765E-3f);
    const _TpVec32F v_coscof_p2 = v_setall_<_TpVec32F>(4.166664568298827E-2f);
    const _TpVec32F v_nan = v_reinterpret_as_f32(v_setall_<_TpVec32S>((int)0x7fc00000));
    const _TpVec32F v_neg_zero = v_setall_<_TpVec32F>(-0.f);

    _TpVec32F _vx, _vy, sign_mask_sin, sign_mask_cos;
    _TpVec32S emm2;

    sign_mask_sin = v_lt(x, v_setzero_<_TpVec32F>());
    _vx = v_abs(x);
    _vy = v_mul(_vx, v_cephes_FOPI);

    emm2 = v_trunc(_vy);
    emm2 = v_add(emm2, v_setall_<_TpVec32S>(1));
    emm2 = v_and(emm2, v_setall_<_TpVec32S>(~1));
    _vy = v_cvt_f32(emm2);

    _TpVec32F poly_mask = v_reinterpret_as_f32(v_eq(v_and(emm2, v_setall_<_TpVec32S>(2)), v_setall_<_TpVec32S>(0)));

    _vx = v_fma(_vy, v_minus_DP1, _vx);
    _vx = v_fma(_vy, v_minus_DP2, _vx);
    _vx = v_fma(_vy, v_minus_DP3, _vx);

    sign_mask_sin = v_xor(sign_mask_sin, v_reinterpret_as_f32(v_eq(v_and(emm2, v_setall_<_TpVec32S>(4)), v_setall_<_TpVec32S>(0))));
    sign_mask_cos = v_reinterpret_as_f32(v_eq(v_and(v_sub(emm2, v_setall_<_TpVec32S>(2)), v_setall_<_TpVec32S>(4)), v_setall_<_TpVec32S>(0)));

    _TpVec32F _vxx = v_mul(_vx, _vx);
    _TpVec32F y1, y2;

    y1 = v_fma(v_coscof_p0, _vxx, v_coscof_p1);
    y1 = v_fma(y1, _vxx, v_coscof_p2);
    y1 = v_fma(y1, _vxx, v_setall_<_TpVec32F>(-0.5f));
    y1 = v_fma(y1, _vxx, v_setall_<_TpVec32F>(1.f));

    y2 = v_fma(v_sincof_p0, _vxx, v_sincof_p1);
    y2 = v_fma(y2, _vxx, v_sincof_p2);
    y2 = v_mul(y2, _vxx);
    y2 = v_fma(y2, _vx, _vx);

    ysin = v_select(poly_mask, y2, y1);
    ycos = v_select(poly_mask, y1, y2);
    ysin = v_select(sign_mask_sin, ysin, v_xor(v_neg_zero, ysin));
    ycos = v_select(sign_mask_cos, v_xor(v_neg_zero, ycos), ycos);

    // sincos(NAN) -> NAN, sincos(±INF) -> NAN
    _TpVec32F mask_inf = v_eq(_vx, v_reinterpret_as_f32(v_setall_<_TpVec32S>((int)0x7f800000)));
    _TpVec32F mask_nan = v_or(mask_inf, v_ne(x, x));
    ysin = v_select(mask_nan, v_nan, ysin);
    ycos = v_select(mask_nan, v_nan, ycos);
}

template<typename _TpVec32F, typename _TpVec32S>
inline _TpVec32F v_sin_default_32f(const _TpVec32F &x) {
    _TpVec32F ysin, ycos;
    v_sincos_default_32f<_TpVec32F, _TpVec32S>(x, ysin, ycos);
    return ysin;
}

template<typename _TpVec32F, typename _TpVec32S>
inline _TpVec32F v_cos_default_32f(const _TpVec32F &x) {
    _TpVec32F ysin, ycos;
    v_sincos_default_32f<_TpVec32F, _TpVec32S>(x, ysin, ycos);
    return ycos;
}

template<typename _TpVec64F, typename _TpVec64S>
inline void v_sincos_default_64f(const _TpVec64F &x, _TpVec64F &ysin, _TpVec64F &ycos) {
    const _TpVec64F v_cephes_FOPI = v_setall_<_TpVec64F>(1.2732395447351626861510701069801148); // 4 / M_PI
    const _TpVec64F v_minus_DP1 = v_setall_<_TpVec64F>(-7.853981554508209228515625E-1);
    const _TpVec64F v_minus_DP2 = v_setall_<_TpVec64F>(-7.94662735614792836714E-9);
    const _TpVec64F v_minus_DP3 = v_setall_<_TpVec64F>(-3.06161699786838294307E-17);
    const _TpVec64F v_sin_C1 = v_setall_<_TpVec64F>(1.58962301576546568060E-10);
    const _TpVec64F v_sin_C2 = v_setall_<_TpVec64F>(-2.50507477628578072866E-8);
    const _TpVec64F v_sin_C3 = v_setall_<_TpVec64F>(2.75573136213857245213E-6);
    const _TpVec64F v_sin_C4 = v_setall_<_TpVec64F>(-1.98412698295895385996E-4);
    const _TpVec64F v_sin_C5 = v_setall_<_TpVec64F>(8.33333333332211858878E-3);
    const _TpVec64F v_sin_C6 = v_setall_<_TpVec64F>(-1.66666666666666307295E-1);
    const _TpVec64F v_cos_C1 = v_setall_<_TpVec64F>(-1.13585365213876817300E-11);
    const _TpVec64F v_cos_C2 = v_setall_<_TpVec64F>(2.08757008419747316778E-9);
    const _TpVec64F v_cos_C3 = v_setall_<_TpVec64F>(-2.75573141792967388112E-7);
    const _TpVec64F v_cos_C4 = v_setall_<_TpVec64F>(2.48015872888517045348E-5);
    const _TpVec64F v_cos_C5 = v_setall_<_TpVec64F>(-1.38888888888730564116E-3);
    const _TpVec64F v_cos_C6 = v_setall_<_TpVec64F>(4.16666666666665929218E-2);
    const _TpVec64F v_nan = v_reinterpret_as_f64(v_setall_<_TpVec64S>((int64)0x7ff8000000000000));
    const _TpVec64F v_neg_zero = v_setall_<_TpVec64F>(-0.0);

    _TpVec64F _vx, _vy, sign_mask_sin, sign_mask_cos;
    _TpVec64S emm2;

    sign_mask_sin = v_lt(x, v_setzero_<_TpVec64F>());
    _vx = v_abs(x);
    _vy = v_mul(_vx, v_cephes_FOPI);

    emm2 = v_expand_low(v_trunc(_vy));
    emm2 = v_add(emm2, v_setall_<_TpVec64S>((int64)1));
    emm2 = v_and(emm2, v_setall_<_TpVec64S>((int64)~1));
    _vy = v_cvt_f64(emm2);

    _TpVec64F poly_mask = v_reinterpret_as_f64(v_eq(v_and(emm2, v_setall_<_TpVec64S>((int64)2)), v_setall_<_TpVec64S>((int64)0)));

    _vx = v_fma(_vy, v_minus_DP1, _vx);
    _vx = v_fma(_vy, v_minus_DP2, _vx);
    _vx = v_fma(_vy, v_minus_DP3, _vx);

    sign_mask_sin = v_xor(sign_mask_sin, v_reinterpret_as_f64(v_eq(v_and(emm2, v_setall_<_TpVec64S>((int64)4)), v_setall_<_TpVec64S>((int64)0))));
    sign_mask_cos = v_reinterpret_as_f64(v_eq(v_and(v_sub(emm2, v_setall_<_TpVec64S>((int64)2)), v_setall_<_TpVec64S>((int64)4)), v_setall_<_TpVec64S>((int64)0)));

    _TpVec64F _vxx = v_mul(_vx, _vx);
    _TpVec64F y1, y2;

    y1 = v_fma(v_cos_C1, _vxx, v_cos_C2);
    y1 = v_fma(y1, _vxx, v_cos_C3);
    y1 = v_fma(y1, _vxx, v_cos_C4);
    y1 = v_fma(y1, _vxx, v_cos_C5);
    y1 = v_fma(y1, _vxx, v_cos_C6);
    y1 = v_fma(y1, _vxx, v_setall_<_TpVec64F>(-0.5));
    y1 = v_fma(y1, _vxx, v_setall_<_TpVec64F>(1.0));

    y2 = v_fma(v_sin_C1, _vxx, v_sin_C2);
    y2 = v_fma(y2, _vxx, v_sin_C3);
    y2 = v_fma(y2, _vxx, v_sin_C4);
    y2 = v_fma(y2, _vxx, v_sin_C5);
    y2 = v_fma(y2, _vxx, v_sin_C6);
    y2 = v_mul(y2, _vxx);
    y2 = v_fma(y2, _vx, _vx);

    ysin = v_select(poly_mask, y2, y1);
    ycos = v_select(poly_mask, y1, y2);
    ysin = v_select(sign_mask_sin, ysin, v_xor(v_neg_zero, ysin));
    ycos = v_select(sign_mask_cos, v_xor(v_neg_zero, ycos), ycos);

    // sincos(NAN) -> NAN, sincos(±INF) -> NAN
    _TpVec64F mask_inf = v_eq(_vx, v_reinterpret_as_f64(v_setall_<_TpVec64S>((int64)0x7ff0000000000000)));
    _TpVec64F mask_nan = v_or(mask_inf, v_ne(x, x));
    ysin = v_select(mask_nan, v_nan, ysin);
    ycos = v_select(mask_nan, v_nan, ycos);
}

template<typename _TpVec64F, typename _TpVec64S>
inline _TpVec64F v_sin_default_64f(const _TpVec64F &x) {
    _TpVec64F ysin, ycos;
    v_sincos_default_64f<_TpVec64F, _TpVec64S>(x, ysin, ycos);
    return ysin;
}

template<typename _TpVec64F, typename _TpVec64S>
inline _TpVec64F v_cos_default_64f(const _TpVec64F &x) {
    _TpVec64F ysin, ycos;
    v_sincos_default_64f<_TpVec64F, _TpVec64S>(x, ysin, ycos);
    return ycos;
}
//! @}


/* This implementation is derived from the approximation approach of Error Function (Erf) from PyTorch
   https://github.com/pytorch/pytorch/blob/9c50ecc84b9a6e699a7f058891b889aafbf976c7/aten/src/ATen/cpu/vec/vec512/vec512_float.h#L189-L220
*/

//! @name Error Function
//! @{
template<typename _TpVec32F, typename _TpVec32S>
inline _TpVec32F v_erf_default_32f(const _TpVec32F &v) {
    const _TpVec32F coef0 = v_setall_<_TpVec32F>(0.3275911f),
            coef1 = v_setall_<_TpVec32F>(1.061405429f),
            coef2 = v_setall_<_TpVec32F>(-1.453152027f),
            coef3 = v_setall_<_TpVec32F>(1.421413741f),
            coef4 = v_setall_<_TpVec32F>(-0.284496736f),
            coef5 = v_setall_<_TpVec32F>(0.254829592f),
            ones = v_setall_<_TpVec32F>(1.0f),
            neg_zeros = v_setall_<_TpVec32F>(-0.f);
    _TpVec32F t = v_abs(v);
    // sign(v)
    _TpVec32F sign_mask = v_and(neg_zeros, v);

    t = v_div(ones, v_fma(coef0, t, ones));
    _TpVec32F r = v_fma(coef1, t, coef2);
    r = v_fma(r, t, coef3);
    r = v_fma(r, t, coef4);
    r = v_fma(r, t, coef5);
    // - v * v
    _TpVec32F v2 = v_mul(v, v);
    _TpVec32F mv2 = v_xor(neg_zeros, v2);
    // - exp(- v * v)
    _TpVec32F exp = v_exp_default_32f<_TpVec32F, _TpVec32S>(mv2);
    _TpVec32F neg_exp = v_xor(neg_zeros, exp);
    _TpVec32F res = v_mul(t, neg_exp);
    res = v_fma(r, res, ones);
    return v_xor(sign_mask, res);
}
//! @}

#endif // OPENCV_HAL_INTRIN_MATH_HPP
