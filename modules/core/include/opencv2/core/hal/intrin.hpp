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

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
#endif
}

#ifdef CV_DOXYGEN
#   undef CV_AVX2
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

#if CV_AVX2

#include "opencv2/core/hal/intrin_avx.hpp"

#endif

//! @addtogroup core_hal_intrin
//! @{

#ifdef CV_SIMD128
#undef CV_SIMD
#define CV_SIMD 1
#else
#define CV_SIMD128 0
#endif

#ifdef CV_SIMD128_64F
#undef CV_SIMD_64F
#define CV_SIMD_64F 1
#else
#define CV_SIMD128_64F 0
#endif

#ifdef CV_SIMD256
#undef CV_SIMD
#define CV_SIMD 1
#else
#define CV_SIMD256 0
#endif

#ifdef CV_SIMD256_64F
#undef CV_SIMD_64F
#define CV_SIMD_64F 1
#else
#define CV_SIMD256_64F 0
#endif


#ifndef CV_SIMD
//! Set to 1 if current compiler supports vector extensions (AVX2 is enabled)
#define CV_SIMD 0
#endif

#ifndef CV_SIMD_64F
//! Set to 1 if current intrinsics implementation supports 64-bit float vectors
#define CV_SIMD_64F 0
#endif

//! @}

//==================================================================================================

//! @cond IGNORED

#define CV_INTRIN_DEFINE_WIDE_INTRIN(typ, vtyp, short_typ, prefix, loadsfx) \
    inline vtyp vx_setall_##short_typ(typ v) { return prefix##_setall_##short_typ(v); } \
    inline vtyp vx_setzero_##short_typ() { return prefix##_setzero_##short_typ(); } \
    inline vtyp vx_##loadsfx(const typ* ptr) { return prefix##_##loadsfx(ptr); } \
    inline vtyp vx_##loadsfx##_aligned(const typ* ptr) { return prefix##_##loadsfx##_aligned(ptr); } \
    inline void vx_store(typ* ptr, vtyp v) { return v_store(ptr, v); } \
    inline void vx_store_aligned(typ* ptr, vtyp v) { return v_store_aligned(ptr, v); }

#define CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(uchar, v_uint8, u8, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(schar, v_int8, s8, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(ushort, v_uint16, u16, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(short, v_int16, s16, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(int, v_int32, s32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(float, v_float32, f32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(double, v_float64, f64, prefix, load)

namespace cv {

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
#endif

template<typename _Tp> struct V_RegTraits
{
};

#define CV_DEF_REG_TRAITS(prefix, _reg, lane_type, suffix, _u_reg, _w_reg, _q_reg, _int_reg, _round_reg) \
    template<> struct V_RegTraits<_reg> \
    { \
        typedef _reg reg; \
        typedef _u_reg u_reg; \
        typedef _w_reg w_reg; \
        typedef _q_reg q_reg; \
        typedef _int_reg int_reg; \
        typedef _round_reg round_reg; \
        static reg zero() { return prefix##_setzero_##suffix(); } \
        static reg all(lane_type v) { return prefix##_setall_##suffix(v); } \
    }

#if CV_SIMD128
    CV_DEF_REG_TRAITS(v, v_uint8x16, uchar, u8, v_uint8x16, v_uint16x8, v_uint32x4, v_int8x16, void);
    CV_DEF_REG_TRAITS(v, v_int8x16, schar, s8, v_uint8x16, v_int16x8, v_int32x4, v_int8x16, void);
    CV_DEF_REG_TRAITS(v, v_uint16x8, ushort, u16, v_uint16x8, v_uint32x4, v_uint64x2, v_int16x8, void);
    CV_DEF_REG_TRAITS(v, v_int16x8, short, s16, v_uint16x8, v_int32x4, v_int64x2, v_int16x8, void);
    CV_DEF_REG_TRAITS(v, v_uint32x4, unsigned, u32, v_uint32x4, v_uint64x2, void, v_int32x4, void);
    CV_DEF_REG_TRAITS(v, v_int32x4, int, s32, v_uint32x4, v_int64x2, void, v_int32x4, void);
    CV_DEF_REG_TRAITS(v, v_float32x4, float, f32, v_float32x4, v_float64x2, void, v_int32x4, v_int32x4);
    CV_DEF_REG_TRAITS(v, v_uint64x2, uint64, u64, v_uint64x2, void, void, v_int64x2, void);
    CV_DEF_REG_TRAITS(v, v_int64x2, int64, s64, v_uint64x2, void, void, v_int64x2, void);
    CV_DEF_REG_TRAITS(v, v_float64x2, double, f64, v_float64x2, void, void, v_int64x2, v_int32x4);
#if CV_FP16
    CV_DEF_REG_TRAITS(v, v_float16x8, short, f16, v_float32x4, void, void, v_int16x8, v_int16x8);
#endif
#endif

#if CV_SIMD256
    CV_DEF_REG_TRAITS(v256, v_uint8x32, uchar, u8, v_uint8x32, v_uint16x16, v_uint32x8, v_int8x32, void);
    CV_DEF_REG_TRAITS(v256, v_int8x32, schar, s8, v_uint8x32, v_int16x16, v_int32x8, v_int8x32, void);
    CV_DEF_REG_TRAITS(v256, v_uint16x16, ushort, u16, v_uint16x16, v_uint32x8, v_uint64x4, v_int16x16, void);
    CV_DEF_REG_TRAITS(v256, v_int16x16, short, s16, v_uint16x16, v_int32x8, v_int64x4, v_int16x16, void);
    CV_DEF_REG_TRAITS(v256, v_uint32x8, unsigned, u32, v_uint32x8, v_uint64x4, void, v_int32x8, void);
    CV_DEF_REG_TRAITS(v256, v_int32x8, int, s32, v_uint32x8, v_int64x4, void, v_int32x8, void);
    CV_DEF_REG_TRAITS(v256, v_float32x8, float, f32, v_float32x8, v_float64x4, void, v_int32x8, v_int32x8);
    CV_DEF_REG_TRAITS(v256, v_uint64x4, uint64, u64, v_uint64x4, void, void, v_int64x4, void);
    CV_DEF_REG_TRAITS(v256, v_int64x4, int64, s64, v_uint64x4, void, void, v_int64x4, void);
    CV_DEF_REG_TRAITS(v256, v_float64x4, double, f64, v_float64x4, void, void, v_int64x4, v_int32x8);
#if CV_FP16
    CV_DEF_REG_TRAITS(v256, v_float16x16, short, f16, v_float32x8, void, void, v_int16x16, void);
#endif
#endif

#if CV_SIMD256
    typedef v_uint8x32   v_uint8;
    typedef v_int8x32    v_int8;
    typedef v_uint16x16  v_uint16;
    typedef v_int16x16   v_int16;
    typedef v_uint32x8   v_uint32;
    typedef v_int32x8    v_int32;
    typedef v_uint64x4   v_uint64;
    typedef v_int64x4    v_int64;
    typedef v_float32x8  v_float32;
    #if CV_SIMD256_64F
    typedef v_float64x4  v_float64;
    #endif
    #if CV_FP16
    typedef v_float16x16  v_float16;
    CV_INTRIN_DEFINE_WIDE_INTRIN(short, v_float16, f16, v256, load_f16)
    #endif
    CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(v256)
    inline void vx_cleanup() { v256_cleanup(); }
#elif CV_SIMD128
    typedef v_uint8x16  v_uint8;
    typedef v_int8x16   v_int8;
    typedef v_uint16x8  v_uint16;
    typedef v_int16x8   v_int16;
    typedef v_uint32x4  v_uint32;
    typedef v_int32x4   v_int32;
    typedef v_uint64x2  v_uint64;
    typedef v_int64x2   v_int64;
    typedef v_float32x4 v_float32;
    #if CV_SIMD128_64F
    typedef v_float64x2 v_float64;
    #endif
    #if CV_FP16
    typedef v_float16x8  v_float16;
    CV_INTRIN_DEFINE_WIDE_INTRIN(short, v_float16, f16, v, load_f16)
    #endif
    CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(v)
    inline void vx_cleanup() { v_cleanup(); }
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
