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

#ifndef __OPENCV_CORE_CVDEF_H__
#define __OPENCV_CORE_CVDEF_H__

#if !defined _CRT_SECURE_NO_DEPRECATE && defined _MSC_VER && _MSC_VER > 1300
#  define _CRT_SECURE_NO_DEPRECATE /* to avoid multiple Visual Studio warnings */
#endif

// undef problematic defines sometimes defined by system headers (windows.h in particular)
#undef small
#undef min
#undef max
#undef abs
#undef Complex

#include "opencv2/hal/defs.h"

#ifdef __OPENCV_BUILD
#  define DISABLE_OPENCV_24_COMPATIBILITY
#endif

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined CVAPI_EXPORTS
#  define CV_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CV_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define CV_EXPORTS
#endif

#ifndef CV_INLINE
#  if defined __cplusplus
#    define CV_INLINE static inline
#  elif defined _MSC_VER
#    define CV_INLINE __inline
#  else
#    define CV_INLINE static
#  endif
#endif

#ifndef CV_EXTERN_C
#  ifdef __cplusplus
#    define CV_EXTERN_C extern "C"
#  else
#    define CV_EXTERN_C
#  endif
#endif

/* special informative macros for wrapper generators */
#define CV_EXPORTS_W CV_EXPORTS
#define CV_EXPORTS_W_SIMPLE CV_EXPORTS
#define CV_EXPORTS_AS(synonym) CV_EXPORTS
#define CV_EXPORTS_W_MAP CV_EXPORTS
#define CV_IN_OUT
#define CV_OUT
#define CV_PROP
#define CV_PROP_RW
#define CV_WRAP
#define CV_WRAP_AS(synonym)

/****************************************************************************************\
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_USRTYPE1 7

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
#define CV_MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)
#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)
#define CV_MAT_CONT_FLAG_SHIFT  14
#define CV_MAT_CONT_FLAG        (1 << CV_MAT_CONT_FLAG_SHIFT)
#define CV_IS_MAT_CONT(flags)   ((flags) & CV_MAT_CONT_FLAG)
#define CV_IS_CONT_MAT          CV_IS_MAT_CONT
#define CV_SUBMAT_FLAG_SHIFT    15
#define CV_SUBMAT_FLAG          (1 << CV_SUBMAT_FLAG_SHIFT)
#define CV_IS_SUBMAT(flags)     ((flags) & CV_MAT_SUBMAT_FLAG)

/* Size of each channel item,
   0x124489 = 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define CV_ELEM_SIZE1(type) \
    ((((sizeof(size_t)<<28)|0x8442211) >> CV_MAT_DEPTH(type)*4) & 15)

/* 0x3a50 = 11 10 10 01 01 00 00 ~ array of log2(sizeof(arr_type_elem)) */
#define CV_ELEM_SIZE(type) \
    (CV_MAT_CN(type) << ((((sizeof(size_t)/4+1)*16384|0x3a50) >> CV_MAT_DEPTH(type)*2) & 3))


/****************************************************************************************\
*                                      fast math                                         *
\****************************************************************************************/

#if defined __BORLANDC__
#  include <fastmath.h>
#elif defined __cplusplus
#  include <cmath>
#else
#  include <math.h>
#endif

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifdef HAVE_TEGRA_OPTIMIZATION
#  include "tegra_round.hpp"
#endif

//! @addtogroup core_utils
//! @{

#if CV_VFP
// 1. general scheme
#define ARM_ROUND(_value, _asm_string) \
    int res; \
    float temp; \
    asm(_asm_string : [res] "=r" (res), [temp] "=w" (temp) : [value] "w" (_value)); \
    return res;
// 2. version for double
#ifdef __clang__
#define ARM_ROUND_DBL(value) ARM_ROUND(value, "vcvtr.s32.f64 %[temp], %[value] \n vmov %[res], %[temp]")
#else
#define ARM_ROUND_DBL(value) ARM_ROUND(value, "vcvtr.s32.f64 %[temp], %P[value] \n vmov %[res], %[temp]")
#endif
// 3. version for float
#define ARM_ROUND_FLT(value) ARM_ROUND(value, "vcvtr.s32.f32 %[temp], %[value]\n vmov %[res], %[temp]")
#endif // CV_VFP

/** @brief Rounds floating-point number to the nearest integer

@param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
result is not defined.
 */
CV_INLINE int cvRound( double value )
{
#if ((defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ && defined __SSE2__ && !defined __APPLE__)) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd( value );
    return _mm_cvtsd_si32(t);
#elif defined _MSC_VER && defined _M_IX86
    int t;
    __asm
    {
        fld value;
        fistp t;
    }
    return t;
#elif ((defined _MSC_VER && defined _M_ARM) || defined CV_ICC || defined __GNUC__) && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND_DBL(value);
#elif defined CV_ICC || defined __GNUC__
# if CV_VFP
    ARM_ROUND_DBL(value)
# else
    return (int)lrint(value);
# endif
#else
    double intpart, fractpart;
    fractpart = modf(value, &intpart);
    if ((fabs(fractpart) != 0.5) || ((((int)intpart) % 2) != 0))
        return (int)(value + (value >= 0 ? 0.5 : -0.5));
    else
        return (int)intpart;
#endif
}

#ifdef __cplusplus

/** @overload */
CV_INLINE int cvRound(float value)
{
#if defined ANDROID && (defined CV_ICC || defined __GNUC__) && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND_FLT(value);
#elif CV_VFP && !defined HAVE_TEGRA_OPTIMIZATION
    ARM_ROUND_FLT(value)
#else
    return cvRound((double)value);
#endif
}

/** @overload */
CV_INLINE int cvRound(int value)
{
    return value;
}

#endif // __cplusplus

/** @brief Rounds floating-point number to the nearest integer not larger than the original.

The function computes an integer i such that:
\f[i \le \texttt{value} < i+1\f]
@param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
result is not defined.
 */
CV_INLINE int cvFloor( double value )
{
#if (defined _MSC_VER && defined _M_X64 || (defined __GNUC__ && defined __SSE2__ && !defined __APPLE__)) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd( value );
    int i = _mm_cvtsd_si32(t);
    return i - _mm_movemask_pd(_mm_cmplt_sd(t, _mm_cvtsi32_sd(t,i)));
#elif defined __GNUC__
    int i = (int)value;
    return i - (i > value);
#else
    int i = cvRound(value);
    float diff = (float)(value - i);
    return i - (diff < 0);
#endif
}

/** @brief Rounds floating-point number to the nearest integer not larger than the original.

The function computes an integer i such that:
\f[i \le \texttt{value} < i+1\f]
@param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
result is not defined.
*/
CV_INLINE int cvCeil( double value )
{
#if (defined _MSC_VER && defined _M_X64 || (defined __GNUC__ && defined __SSE2__&& !defined __APPLE__)) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd( value );
    int i = _mm_cvtsd_si32(t);
    return i + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t,i), t));
#elif defined __GNUC__
    int i = (int)value;
    return i + (i < value);
#else
    int i = cvRound(value);
    float diff = (float)(i - value);
    return i + (diff < 0);
#endif
}

/** @brief Determines if the argument is Not A Number.

@param value The input floating-point value

The function returns 1 if the argument is Not A Number (as defined by IEEE754 standard), 0
otherwise. */
CV_INLINE int cvIsNaN( double value )
{
    union { uint64 u; double f; } ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
           ((unsigned)ieee754.u != 0) > 0x7ff00000;
}

/** @brief Determines if the argument is Infinity.

@param value The input floating-point value

The function returns 1 if the argument is a plus or minus infinity (as defined by IEEE754 standard)
and 0 otherwise. */
CV_INLINE int cvIsInf( double value )
{
    union { uint64 u; double f; } ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) == 0x7ff00000 &&
           (unsigned)ieee754.u == 0;
}

//! @} core_utils

/****************************************************************************************\
*          exchange-add operation for atomic operations on reference counters            *
\****************************************************************************************/

#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
   // atomic increment on the linux version of the Intel(tm) compiler
#  define CV_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define CV_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
       // version for gcc >= 4.7
#      define CV_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define CV_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
   CV_INLINE CV_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif


/****************************************************************************************\
*                                  CV_NORETURN attribute                                 *
\****************************************************************************************/

#ifndef CV_NORETURN
#  if defined(__GNUC__)
#    define CV_NORETURN __attribute__((__noreturn__))
#  elif defined(_MSC_VER) && (_MSC_VER >= 1300)
#    define CV_NORETURN __declspec(noreturn)
#  else
#    define CV_NORETURN /* nothing by default */
#  endif
#endif

#endif // __OPENCV_CORE_CVDEF_H__
