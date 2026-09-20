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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CORE_FAST_MATH_HPP
#define OPENCV_CORE_FAST_MATH_HPP

#include "opencv2/core/cvdef.h"

//! @addtogroup core_utils
//! @{

/****************************************************************************************\
*                                      fast math                                         *
\****************************************************************************************/

#ifdef __cplusplus
#  include <cmath>
#  include <climits>
#else
#  ifdef __BORLANDC__
#    include <fastmath.h>
#  else
#    include <math.h>
#  endif
#  include <limits.h>
#endif

#if defined(__CUDACC__)
  // nothing, intrinsics/asm code is not supported
#else
  #if ((defined _MSC_VER && (defined _M_X64 || (defined _M_IX86_FP && _M_IX86_FP >= 2))) \
      || (defined __GNUC__ && defined __SSE2__)) \
      && !defined(OPENCV_SKIP_INCLUDE_EMMINTRIN_H)
    #include <emmintrin.h>
    // SSE2 is guaranteed by the compiler flags: 32-bit results can use cvtss2si/cvtsd2si
    #define CV__FASTMATH_HAVE_SSE2 1
    #if defined _M_X64 || defined __x86_64__
      // 64-bit mode: cvtss2si/cvtsd2si with a 64-bit destination are available too
      #define CV__FASTMATH_HAVE_SSE2_X64 1
    #endif
  #endif

  #if defined __PPC64__ && defined __GNUC__ && defined _ARCH_PWR8 \
      && !defined(OPENCV_SKIP_INCLUDE_ALTIVEC_H)
    #include <altivec.h>
    #undef vector
    #undef bool
    #undef pixel
  #endif

  #if defined(__aarch64__) && defined(__GNUC__)
    // fcvt{ns,ms,ps,zs} saturate to the destination integer range
    #define CV__FASTMATH_HAVE_AARCH64_ASM 1
  #endif

  #if defined(__riscv) && defined(__riscv_flen) && defined(__GNUC__)
    // fcvt.w.{s,d} / fcvt.l.{s,d} saturate to the destination integer range
    #define CV__FASTMATH_HAVE_RISCV_ASM_FLT 1
    #if __riscv_flen >= 64
      #define CV__FASTMATH_HAVE_RISCV_ASM_DBL 1
    #endif
    #if __riscv_xlen >= 64
      #define CV__FASTMATH_HAVE_RISCV_ASM_64 1
    #endif
  #endif

  #if defined(__loongarch64) && defined(__GNUC__)
    // ftint*.{w,l}.{s,d} saturate to the destination integer range
    #define CV__FASTMATH_HAVE_LOONGARCH_ASM 1
  #endif

  #if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
    // ACLE scalar conversions (fcvt*) saturate to the destination integer range
    #define CV__FASTMATH_HAVE_MSVC_ARM64 1
  #endif

  #if defined(CV_INLINE_ROUND_FLT)
    // user-specified version
    // CV_INLINE_ROUND_DBL should be defined too
  #elif defined __GNUC__ && defined __arm__ && (defined __ARM_PCS_VFP || defined __ARM_VFPV3__ || defined __ARM_NEON) && !defined __SOFTFP__
    // 1. general scheme
    #define ARM_ROUND(_value, _asm_string) \
        int res; \
        float temp; \
        CV_UNUSED(temp); \
        __asm__(_asm_string : [res] "=r" (res), [temp] "=w" (temp) : [value] "w" (_value)); \
        return res
    // 2. version for double
    #ifdef __clang__
        #define CV_INLINE_ROUND_DBL(value) ARM_ROUND(value, "vcvtr.s32.f64 %[temp], %[value] \n vmov %[res], %[temp]")
    #else
        #define CV_INLINE_ROUND_DBL(value) ARM_ROUND(value, "vcvtr.s32.f64 %[temp], %P[value] \n vmov %[res], %[temp]")
    #endif
    // 3. version for float
    #define CV_INLINE_ROUND_FLT(value) ARM_ROUND(value, "vcvtr.s32.f32 %[temp], %[value]\n vmov %[res], %[temp]")
  #elif defined __PPC64__ && defined __GNUC__ && defined _ARCH_PWR8
    // P8 and newer machines can convert fp32/64 to int quickly.
    #define CV_INLINE_ROUND_DBL(value) \
        int out; \
        double temp; \
        __asm__( "fctiw %[temp],%[in]\n\tmfvsrwz %[out],%[temp]\n\t" : [out] "=r" (out), [temp] "=d" (temp) : [in] "d" ((double)(value)) : ); \
        return out;

    // FP32 also works with FP64 routine above
    #define CV_INLINE_ROUND_FLT(value) CV_INLINE_ROUND_DBL(value)
  #endif

  #ifdef CV_INLINE_ISINF_FLT
    // user-specified version
    // CV_INLINE_ISINF_DBL should be defined too
  #elif defined __PPC64__ && defined _ARCH_PWR9 && defined(scalar_test_data_class)
    #define CV_INLINE_ISINF_DBL(value) return scalar_test_data_class(value, 0x30);
    #define CV_INLINE_ISINF_FLT(value) CV_INLINE_ISINF_DBL(value)
  #endif

  #ifdef CV_INLINE_ISNAN_FLT
    // user-specified version
    // CV_INLINE_ISNAN_DBL should be defined too
  #elif defined __PPC64__ && defined _ARCH_PWR9 && defined(scalar_test_data_class)
    #define CV_INLINE_ISNAN_DBL(value) return scalar_test_data_class(value, 0x40);
    #define CV_INLINE_ISNAN_FLT(value) CV_INLINE_ISNAN_DBL(value)
  #endif

  #if !defined(OPENCV_USE_FASTMATH_BUILTINS) \
    && ( \
        defined(__x86_64__) || defined(__i686__) \
        || defined(__arm__) \
        || defined(__PPC64__) \
    )
    /* Let builtin C math functions when available. Dedicated hardware is available to
       round and convert FP values. */
    #define OPENCV_USE_FASTMATH_BUILTINS 1
  #endif

  /* Enable builtin math functions if possible, desired, and available.
     Note, not all math functions inline equally. E.g lrint will not inline
     without the -fno-math-errno option. */
  #if defined(CV_ICC)
    // nothing
  #elif defined(OPENCV_USE_FASTMATH_BUILTINS) && OPENCV_USE_FASTMATH_BUILTINS
    #if defined(__clang__)
      #define CV__FASTMATH_ENABLE_CLANG_MATH_BUILTINS
      #if !defined(CV_INLINE_ISNAN_DBL) && __has_builtin(__builtin_isnan)
        #define CV_INLINE_ISNAN_DBL(value) return __builtin_isnan(value);
      #endif
      #if !defined(CV_INLINE_ISNAN_FLT) && __has_builtin(__builtin_isnan)
        #define CV_INLINE_ISNAN_FLT(value) return __builtin_isnan(value);
      #endif
      #if !defined(CV_INLINE_ISINF_DBL) && __has_builtin(__builtin_isinf)
        #define CV_INLINE_ISINF_DBL(value) return __builtin_isinf(value);
      #endif
      #if !defined(CV_INLINE_ISINF_FLT) && __has_builtin(__builtin_isinf)
        #define CV_INLINE_ISINF_FLT(value) return __builtin_isinf(value);
      #endif
    #elif defined(__GNUC__)
      #define CV__FASTMATH_ENABLE_GCC_MATH_BUILTINS
      #if !defined(CV_INLINE_ISNAN_DBL)
        #define CV_INLINE_ISNAN_DBL(value) return __builtin_isnan(value);
      #endif
      #if !defined(CV_INLINE_ISNAN_FLT)
        #define CV_INLINE_ISNAN_FLT(value) return __builtin_isnanf(value);
      #endif
      #if !defined(CV_INLINE_ISINF_DBL)
        #define CV_INLINE_ISINF_DBL(value) return __builtin_isinf(value);
      #endif
      #if !defined(CV_INLINE_ISINF_FLT)
        #define CV_INLINE_ISINF_FLT(value) return __builtin_isinff(value);
      #endif
    #elif defined(_MSC_VER)
      #if !defined(CV_INLINE_ISNAN_DBL)
        #define CV_INLINE_ISNAN_DBL(value) return isnan(value);
      #endif
      #if !defined(CV_INLINE_ISNAN_FLT)
        #define CV_INLINE_ISNAN_FLT(value) return isnan(value);
      #endif
      #if !defined(CV_INLINE_ISINF_DBL)
        #define CV_INLINE_ISINF_DBL(value) return isinf(value);
      #endif
      #if !defined(CV_INLINE_ISINF_FLT)
        #define CV_INLINE_ISINF_FLT(value) return isinf(value);
      #endif
    #endif
  #endif

#endif // defined(__CUDACC__)

/*
 Saturation limits used by the conversion functions below.

 The double-precision functions saturate exactly to INT_MIN/INT_MAX (INT64_MIN/INT64_MAX).
 INT_MAX is not representable as float: the single-precision functions clamp the input to
 2^31-128 = 2147483520 (the largest float below 2^31) where the conversion instruction does not
 saturate by itself; where it does (e.g. AArch64, RISC-V), INT_MAX is returned. Both variants are
 accepted by OpenCV tests; the result is guaranteed to be in [2147483520, INT_MAX] for
 values >= 2^31 and INT_MIN for values <= -2^31.
*/
#define CV__FLT2INT_MAX_F  2147483520.f
#define CV__FLT2INT_MIN_F  -2147483648.f
#define CV__FLT2INT_MAX_D  2147483647.0
#define CV__FLT2INT_MIN_D  -2147483648.0
#define CV__FLT2INT_2P63_D  9223372036854775808.0
#define CV__FLT2INT_2P63_F  9223372036854775808.f

/** @brief Rounds floating-point number to the nearest integer

 The function uses the round-half-to-even rule (the default IEEE 754 rounding mode).
 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated to INT_MIN or INT_MAX.
 The result for NaN is platform-specific.
 @param value floating-point number.
 */
inline int
cvRound( double value )
{
#if defined CV_INLINE_ROUND_DBL
    CV_INLINE_ROUND_DBL(value);
#elif defined CV__FASTMATH_HAVE_MSVC_ARM64
    float64x1_t v = vdup_n_f64(value);
    int64_t r = vget_lane_s64(vcvtn_s64_f64(v), 0);
    return r >= INT_MAX ? INT_MAX : r <= INT_MIN ? INT_MIN : (int)r;
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtns %w[i], %d[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_DBL
    int i;
    __asm__("fcvt.w.d %[i], %[in], rne" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    double tmp;
    __asm__ ("ftintrne.w.d    %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    // cvtsd2si returns INT_MIN for anything below INT_MIN, so only the upper bound needs a clamp
    __m128d t = _mm_min_sd(_mm_set_sd(value), _mm_set_sd(CV__FLT2INT_MAX_D));
    return _mm_cvtsd_si32(t);
#else
    value = value >= CV__FLT2INT_MAX_D ? CV__FLT2INT_MAX_D : value <= CV__FLT2INT_MIN_D ? CV__FLT2INT_MIN_D : value;
    return (int)lrint(value);
#endif
}


/** @brief Rounds floating-point number to the nearest integer not larger than the original.

 The function computes an integer i such that:
 \f[i \le \texttt{value} < i+1\f]
 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated to INT_MIN or INT_MAX.
 The result for NaN is platform-specific.
 @param value floating-point number.
 */
inline int cvFloor( double value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    int64_t r = vcvtmd_s64_f64(value);
    return r >= INT_MAX ? INT_MAX : r <= INT_MIN ? INT_MIN : (int)r;
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtms %w[i], %d[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_DBL
    int i;
    __asm__("fcvt.w.d %[i], %[in], rdn" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    double tmp;
    __asm__ ("ftintrm.w.d     %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    __m128d v = _mm_set_sd(value);
    v = _mm_max_sd(v, _mm_set_sd(CV__FLT2INT_MIN_D));
    v = _mm_min_sd(v, _mm_set_sd(CV__FLT2INT_MAX_D));
    int i = _mm_cvtsd_si32(v);
    __m128d r = _mm_cvtsi32_sd(v, i);
    return i - _mm_comilt_sd(v, r);
#else
    value = value >= CV__FLT2INT_MAX_D ? CV__FLT2INT_MAX_D : value <= CV__FLT2INT_MIN_D ? CV__FLT2INT_MIN_D : value;
    int i = (int)value;
    return i - (i > value);
#endif
}

/** @brief Rounds floating-point number to the nearest integer not smaller than the original.

 The function computes an integer i such that:
 \f[i-1 < \texttt{value} \le i\f]
 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated to INT_MIN or INT_MAX.
 The result for NaN is platform-specific.
 @param value floating-point number.
 */
inline int cvCeil( double value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    int64_t r = vcvtpd_s64_f64(value);
    return r >= INT_MAX ? INT_MAX : r <= INT_MIN ? INT_MIN : (int)r;
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtps %w[i], %d[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_DBL
    int i;
    __asm__("fcvt.w.d %[i], %[in], rup" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    double tmp;
    __asm__ ("ftintrp.w.d     %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    // cvtsd2si returns INT_MIN for anything below INT_MIN, so only the upper bound needs a clamp
    __m128d v = _mm_min_sd(_mm_set_sd(value), _mm_set_sd(CV__FLT2INT_MAX_D));
    int i = _mm_cvtsd_si32(v);
    __m128d r = _mm_cvtsi32_sd(v, i);
    return i + _mm_comigt_sd(v, r);
#else
    value = value >= CV__FLT2INT_MAX_D ? CV__FLT2INT_MAX_D : value <= CV__FLT2INT_MIN_D ? CV__FLT2INT_MIN_D : value;
    int i = (int)value;
    return i + (i < value);
#endif
}

/** @brief Truncates floating-point number to integer (rounds towards zero).

 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated to INT_MIN or INT_MAX.
 The result for NaN is platform-specific.
 @param value floating-point number.
 */
inline int cvTrunc( double value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    int64_t r = vcvtd_s64_f64(value);
    return r >= INT_MAX ? INT_MAX : r <= INT_MIN ? INT_MIN : (int)r;
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtzs %w[i], %d[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_DBL
    int i;
    __asm__("fcvt.w.d %[i], %[in], rtz" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    double tmp;
    __asm__ ("ftintrz.w.d     %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    // cvttsd2si returns INT_MIN for anything below INT_MIN, so only the upper bound needs a clamp
    return _mm_cvttsd_si32(_mm_min_sd(_mm_set_sd(value), _mm_set_sd(CV__FLT2INT_MAX_D)));
#else
    value = value >= CV__FLT2INT_MAX_D ? CV__FLT2INT_MAX_D : value <= CV__FLT2INT_MIN_D ? CV__FLT2INT_MIN_D : value;
    return (int)value;
#endif
}

/** @brief Rounds floating-point number to the nearest 64-bit integer

 The function uses the round-half-to-even rule (the default IEEE 754 rounding mode).
 If the value is outside of INT64_MIN ... INT64_MAX range, the result is saturated to INT64_MIN or INT64_MAX.
 The result for NaN is platform-specific.
 @param value floating-point number.
 */
inline int64_t cvRound64( double value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    return vcvtnd_s64_f64(value);
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int64_t i;
    __asm__("fcvtns %x[i], %d[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_DBL && defined CV__FASTMATH_HAVE_RISCV_ASM_64
    int64_t i;
    __asm__("fcvt.l.d %[i], %[in], rne" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int64_t i;
    double tmp;
    __asm__ ("ftintrne.l.d    %[tmp],    %[in]       \n\t"
             "movfr2gr.d      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2_X64
    // cvtsd2si returns INT64_MIN for anything below INT64_MIN, so only the upper bound needs a check
    return value >= CV__FLT2INT_2P63_D ? INT64_MAX : (int64_t)_mm_cvtsd_si64(_mm_set_sd(value));
#else
    return value >= CV__FLT2INT_2P63_D ? INT64_MAX :
           value <= -CV__FLT2INT_2P63_D ? INT64_MIN : (int64_t)llrint(value);
#endif
}

/** @brief Determines if the argument is Not A Number.

 @param value The input floating-point value

 The function returns 1 if the argument is Not A Number (as defined by IEEE754 standard), 0
 otherwise. */
inline int cvIsNaN( double value )
{
#if defined CV_INLINE_ISNAN_DBL
    CV_INLINE_ISNAN_DBL(value);
#else
    Cv64suf ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
           ((unsigned)ieee754.u != 0) > 0x7ff00000;
#endif
}

/** @brief Determines if the argument is Infinity.

 @param value The input floating-point value

 The function returns 1 if the argument is a plus or minus infinity (as defined by IEEE754 standard)
 and 0 otherwise. */
inline int cvIsInf( double value )
{
#if defined CV_INLINE_ISINF_DBL
    CV_INLINE_ISINF_DBL(value);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__PPC64__) || defined(__loongarch64)
    Cv64suf ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffffffffffff) ==
                        0x7ff0000000000000;
#else
    Cv64suf ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) == 0x7ff00000 &&
            (unsigned)ieee754.u == 0;
#endif
}

#ifdef __cplusplus

/** @overload

 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated: INT_MIN for values
 not greater than -2^31; for values not less than 2^31 the result is in [2147483520, INT_MAX]
 (INT_MAX is not representable as float, and the exact value depends on the platform).
 */
inline int cvRound(float value)
{
#if defined CV_INLINE_ROUND_FLT
    CV_INLINE_ROUND_FLT(value);
#elif defined CV__FASTMATH_HAVE_MSVC_ARM64
    float32x2_t v = vdup_n_f32(value);
    int32x2_t r = vcvtn_s32_f32(v);
    return vget_lane_s32(r, 0);
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtns %w[i], %s[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_FLT
    int i;
    __asm__("fcvt.w.s %[i], %[in], rne" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    float tmp;
    __asm__ ("ftintrne.w.s    %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    // cvtss2si returns INT_MIN for anything below INT_MIN, so only the upper bound needs a clamp
    __m128 t = _mm_min_ss(_mm_set_ss(value), _mm_set_ss(CV__FLT2INT_MAX_F));
    return _mm_cvtss_si32(t);
#else
    value = value >= CV__FLT2INT_MAX_F ? CV__FLT2INT_MAX_F : value <= CV__FLT2INT_MIN_F ? CV__FLT2INT_MIN_F : value;
    return (int)lrintf(value);
#endif
}

/** @overload */
inline int cvRound( int value )
{
    return value;
}

/** @overload

 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated: INT_MIN for values
 not greater than -2^31; for values not less than 2^31 the result is in [2147483520, INT_MAX]
 (INT_MAX is not representable as float, and the exact value depends on the platform).
 */
inline int cvFloor( float value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    return vcvtms_s32_f32(value);
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtms %w[i], %s[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_FLT
    int i;
    __asm__("fcvt.w.s %[i], %[in], rdn" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    float tmp;
    __asm__ ("ftintrm.w.s     %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    __m128 v = _mm_set_ss(value);
    v = _mm_max_ss(v, _mm_set_ss(CV__FLT2INT_MIN_F));
    v = _mm_min_ss(v, _mm_set_ss(CV__FLT2INT_MAX_F));
    int i = _mm_cvtss_si32(v);
    __m128 r = _mm_cvtsi32_ss(v, i);
    return i - _mm_comilt_ss(v, r);
#else
    value = value >= CV__FLT2INT_MAX_F ? CV__FLT2INT_MAX_F : value <= CV__FLT2INT_MIN_F ? CV__FLT2INT_MIN_F : value;
    int i = (int)value;
    return i - (i > value);
#endif
}

/** @overload */
inline int cvFloor( int value )
{
    return value;
}

/** @overload

 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated: INT_MIN for values
 not greater than -2^31; for values not less than 2^31 the result is in [2147483520, INT_MAX]
 (INT_MAX is not representable as float, and the exact value depends on the platform).
 */
inline int cvCeil( float value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    return vcvtps_s32_f32(value);
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtps %w[i], %s[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_FLT
    int i;
    __asm__("fcvt.w.s %[i], %[in], rup" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    float tmp;
    __asm__ ("ftintrp.w.s     %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    // cvtss2si returns INT_MIN for anything below INT_MIN, so only the upper bound needs a clamp
    __m128 v = _mm_min_ss(_mm_set_ss(value), _mm_set_ss(CV__FLT2INT_MAX_F));
    int i = _mm_cvtss_si32(v);
    __m128 r = _mm_cvtsi32_ss(v, i);
    return i + _mm_comigt_ss(v, r);
#else
    value = value >= CV__FLT2INT_MAX_F ? CV__FLT2INT_MAX_F : value <= CV__FLT2INT_MIN_F ? CV__FLT2INT_MIN_F : value;
    int i = (int)value;
    return i + (i < value);
#endif
}

/** @overload */
inline int cvCeil( int value )
{
    return value;
}

/** @overload

 If the value is outside of INT_MIN ... INT_MAX range, the result is saturated: INT_MIN for values
 not greater than -2^31; for values not less than 2^31 the result is in [2147483520, INT_MAX]
 (INT_MAX is not representable as float, and the exact value depends on the platform).
 */
inline int cvTrunc( float value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    return vcvts_s32_f32(value);
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int i;
    __asm__("fcvtzs %w[i], %s[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_FLT
    int i;
    __asm__("fcvt.w.s %[i], %[in], rtz" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int i;
    float tmp;
    __asm__ ("ftintrz.w.s     %[tmp],    %[in]       \n\t"
             "movfr2gr.s      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2
    // cvttss2si returns INT_MIN for anything below INT_MIN, so only the upper bound needs a clamp
    return _mm_cvttss_si32(_mm_min_ss(_mm_set_ss(value), _mm_set_ss(CV__FLT2INT_MAX_F)));
#else
    value = value >= CV__FLT2INT_MAX_F ? CV__FLT2INT_MAX_F : value <= CV__FLT2INT_MIN_F ? CV__FLT2INT_MIN_F : value;
    return (int)value;
#endif
}

/** @overload */
inline int cvTrunc( int value )
{
    return value;
}

/** @overload */
inline int64_t cvRound64( float value )
{
#if defined CV__FASTMATH_HAVE_MSVC_ARM64
    return vcvtnd_s64_f64((double)value);
#elif defined CV__FASTMATH_HAVE_AARCH64_ASM
    int64_t i;
    __asm__("fcvtns %x[i], %s[in]" : [i] "=r" (i) : [in] "w" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_RISCV_ASM_FLT && defined CV__FASTMATH_HAVE_RISCV_ASM_64
    int64_t i;
    __asm__("fcvt.l.s %[i], %[in], rne" : [i] "=r" (i) : [in] "f" (value));
    return i;
#elif defined CV__FASTMATH_HAVE_LOONGARCH_ASM
    int64_t i;
    float tmp;
    __asm__ ("ftintrne.l.s    %[tmp],    %[in]       \n\t"
             "movfr2gr.d      %[i],      %[tmp]      \n\t"
             : [i] "=r" (i), [tmp] "=f" (tmp)
             : [in] "f" (value)
             :);
    return i;
#elif defined CV__FASTMATH_HAVE_SSE2_X64
    // cvtss2si returns INT64_MIN for anything below INT64_MIN, so only the upper bound needs a check
    return value >= CV__FLT2INT_2P63_F ? INT64_MAX : (int64_t)_mm_cvtss_si64(_mm_set_ss(value));
#else
    return value >= CV__FLT2INT_2P63_F ? INT64_MAX :
           value <= -CV__FLT2INT_2P63_F ? INT64_MIN : (int64_t)llrintf(value);
#endif
}

/** @overload */
inline int64_t cvRound64( int value )
{
    return value;
}

/** @overload */
inline int cvIsNaN( float value )
{
#if defined CV_INLINE_ISNAN_FLT
    CV_INLINE_ISNAN_FLT(value);
#else
    Cv32suf ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) > 0x7f800000;
#endif
}

/** @overload */
inline int cvIsInf( float value )
{
#if defined CV_INLINE_ISINF_FLT
    CV_INLINE_ISINF_FLT(value);
#else
    Cv32suf ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) == 0x7f800000;
#endif
}

#endif // __cplusplus

//! @} core_utils

#endif
