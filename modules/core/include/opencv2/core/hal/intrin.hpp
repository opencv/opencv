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

#ifndef OPENCV_HAL_INTRIN_HPP
#define OPENCV_HAL_INTRIN_HPP

#include <cmath>
#include <float.h>
#include <stdlib.h>
#include "opencv2/core/cvdef.h"

#define OPENCV_HAL_ADD(a, b) ((a) + (b))
#define OPENCV_HAL_AND(a, b) ((a) & (b))
#define OPENCV_HAL_NOP(a) (a)
#define OPENCV_HAL_1ST(a, b) (a)

// unlike HAL API, which is in cv::hal,
// we put intrinsics into cv namespace to make its
// access from within opencv code more accessible
namespace cv {

#ifndef CV_DOXYGEN

#ifdef CV_CPU_DISPATCH_MODE
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE __CV_CAT(hal_, CV_CPU_DISPATCH_MODE)
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN namespace __CV_CAT(hal_, CV_CPU_DISPATCH_MODE) {
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END }
#else
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE hal_baseline
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN namespace hal_baseline {
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END }
#endif


CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
using namespace CV_CPU_OPTIMIZATION_HAL_NAMESPACE;
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
#endif

//! @addtogroup core_hal_intrin
//! @{

//! @cond IGNORED
template<typename _Tp> struct V_TypeTraits
{
    typedef _Tp int_type;
    typedef _Tp uint_type;
    typedef _Tp abs_type;
    typedef _Tp sum_type;

    enum { delta = 0, shift = 0 };

    static int_type reinterpret_int(_Tp x) { return x; }
    static uint_type reinterpet_uint(_Tp x) { return x; }
    static _Tp reinterpret_from_int(int_type x) { return (_Tp)x; }
};

template<> struct V_TypeTraits<uchar>
{
    typedef uchar value_type;
    typedef schar int_type;
    typedef uchar uint_type;
    typedef uchar abs_type;
    typedef int sum_type;

    typedef ushort w_type;
    typedef unsigned q_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<schar>
{
    typedef schar value_type;
    typedef schar int_type;
    typedef uchar uint_type;
    typedef uchar abs_type;
    typedef int sum_type;

    typedef short w_type;
    typedef int q_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<ushort>
{
    typedef ushort value_type;
    typedef short int_type;
    typedef ushort uint_type;
    typedef ushort abs_type;
    typedef int sum_type;

    typedef unsigned w_type;
    typedef uchar nu_type;

    enum { delta = 32768, shift = 16 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<short>
{
    typedef short value_type;
    typedef short int_type;
    typedef ushort uint_type;
    typedef ushort abs_type;
    typedef int sum_type;

    typedef int w_type;
    typedef uchar nu_type;
    typedef schar n_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<unsigned>
{
    typedef unsigned value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef unsigned abs_type;
    typedef unsigned sum_type;

    typedef uint64 w_type;
    typedef ushort nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<int>
{
    typedef int value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef unsigned abs_type;
    typedef int sum_type;

    typedef int64 w_type;
    typedef short n_type;
    typedef ushort nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<uint64>
{
    typedef uint64 value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef uint64 abs_type;
    typedef uint64 sum_type;

    typedef unsigned nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<int64>
{
    typedef int64 value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef uint64 abs_type;
    typedef int64 sum_type;

    typedef int nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};


template<> struct V_TypeTraits<float>
{
    typedef float value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef float abs_type;
    typedef float sum_type;

    typedef double w_type;

    static int_type reinterpret_int(value_type x)
    {
        Cv32suf u;
        u.f = x;
        return u.i;
    }
    static uint_type reinterpet_uint(value_type x)
    {
        Cv32suf u;
        u.f = x;
        return u.u;
    }
    static value_type reinterpret_from_int(int_type x)
    {
        Cv32suf u;
        u.i = x;
        return u.f;
    }
};

template<> struct V_TypeTraits<double>
{
    typedef double value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef double abs_type;
    typedef double sum_type;
    static int_type reinterpret_int(value_type x)
    {
        Cv64suf u;
        u.f = x;
        return u.i;
    }
    static uint_type reinterpet_uint(value_type x)
    {
        Cv64suf u;
        u.f = x;
        return u.u;
    }
    static value_type reinterpret_from_int(int_type x)
    {
        Cv64suf u;
        u.i = x;
        return u.f;
    }
};

template <typename T> struct V_SIMD128Traits
{
    enum { nlanes = 16 / sizeof(T) };
};

//! @endcond

//! @}

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
#endif
}

#ifdef CV_DOXYGEN
#   undef CV_SSE2
#   undef CV_NEON
#   undef CV_VSX
#endif

#if CV_SSE2

#include "opencv2/core/hal/intrin_sse.hpp"

#elif CV_NEON

#include "opencv2/core/hal/intrin_neon.hpp"

#elif CV_VSX

#include "opencv2/core/hal/intrin_vsx.hpp"

#else

#include "opencv2/core/hal/intrin_cpp.hpp"

#endif

//! @addtogroup core_hal_intrin
//! @{

#ifndef CV_SIMD128
//! Set to 1 if current compiler supports vector extensions (NEON or SSE is enabled)
#define CV_SIMD128 0
#endif

#ifndef CV_SIMD128_64F
//! Set to 1 if current intrinsics implementation supports 64-bit float vectors
#define CV_SIMD128_64F 0
#endif

//! @}

//==================================================================================================

//! @cond IGNORED

namespace cv {

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
#endif

template <typename R> struct V_RegTrait128;

template <> struct V_RegTrait128<uchar> {
    typedef v_uint8x16 reg;
    typedef v_uint16x8 w_reg;
    typedef v_uint32x4 q_reg;
    typedef v_uint8x16 u_reg;
    static v_uint8x16 zero() { return v_setzero_u8(); }
    static v_uint8x16 all(uchar val) { return v_setall_u8(val); }
};

template <> struct V_RegTrait128<schar> {
    typedef v_int8x16 reg;
    typedef v_int16x8 w_reg;
    typedef v_int32x4 q_reg;
    typedef v_uint8x16 u_reg;
    static v_int8x16 zero() { return v_setzero_s8(); }
    static v_int8x16 all(schar val) { return v_setall_s8(val); }
};

template <> struct V_RegTrait128<ushort> {
    typedef v_uint16x8 reg;
    typedef v_uint32x4 w_reg;
    typedef v_int16x8 int_reg;
    typedef v_uint16x8 u_reg;
    static v_uint16x8 zero() { return v_setzero_u16(); }
    static v_uint16x8 all(ushort val) { return v_setall_u16(val); }
};

template <> struct V_RegTrait128<short> {
    typedef v_int16x8 reg;
    typedef v_int32x4 w_reg;
    typedef v_uint16x8 u_reg;
    static v_int16x8 zero() { return v_setzero_s16(); }
    static v_int16x8 all(short val) { return v_setall_s16(val); }
};

template <> struct V_RegTrait128<unsigned> {
    typedef v_uint32x4 reg;
    typedef v_uint64x2 w_reg;
    typedef v_int32x4 int_reg;
    typedef v_uint32x4 u_reg;
    static v_uint32x4 zero() { return v_setzero_u32(); }
    static v_uint32x4 all(unsigned val) { return v_setall_u32(val); }
};

template <> struct V_RegTrait128<int> {
    typedef v_int32x4 reg;
    typedef v_int64x2 w_reg;
    typedef v_uint32x4 u_reg;
    static v_int32x4 zero() { return v_setzero_s32(); }
    static v_int32x4 all(int val) { return v_setall_s32(val); }
};

template <> struct V_RegTrait128<uint64> {
    typedef v_uint64x2 reg;
    static v_uint64x2 zero() { return v_setzero_u64(); }
    static v_uint64x2 all(uint64 val) { return v_setall_u64(val); }
};

template <> struct V_RegTrait128<int64> {
    typedef v_int64x2 reg;
    static v_int64x2 zero() { return v_setzero_s64(); }
    static v_int64x2 all(int64 val) { return v_setall_s64(val); }
};

template <> struct V_RegTrait128<float> {
    typedef v_float32x4 reg;
    typedef v_int32x4 int_reg;
    typedef v_float32x4 u_reg;
    static v_float32x4 zero() { return v_setzero_f32(); }
    static v_float32x4 all(float val) { return v_setall_f32(val); }
};

#if CV_SIMD128_64F
template <> struct V_RegTrait128<double> {
    typedef v_float64x2 reg;
    typedef v_int32x4 int_reg;
    typedef v_float64x2 u_reg;
    static v_float64x2 zero() { return v_setzero_f64(); }
    static v_float64x2 all(double val) { return v_setall_f64(val); }
};
#endif

inline unsigned int trailingZeros32(unsigned int value) {
#if defined(_MSC_VER)
#if (_MSC_VER < 1700) || defined(_M_ARM)
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return (unsigned int)index;
#else
    return _tzcnt_u32(value);
#endif
#elif defined(__GNUC__) || defined(__GNUG__)
    return __builtin_ctz(value);
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    return _bit_scan_forward(value);
#elif defined(__clang__)
    return llvm.cttz.i32(value, true);
#else
    static const int MultiplyDeBruijnBitPosition[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };
    return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) * 0x077CB531U)) >> 27];
#endif
}

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
#endif

} // cv::

//! @endcond

#endif
