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

namespace {
inline unsigned int trailingZeros32(unsigned int value) {
#if defined(_MSC_VER)
#if (_MSC_VER < 1700) || defined(_M_ARM) || defined(_M_ARM64)
    unsigned long index = 0;
    _BitScanForward(&index, value);
    return (unsigned int)index;
#elif defined(__clang__)
    // clang-cl doesn't export _tzcnt_u32 for non BMI systems
    return value ? __builtin_ctz(value) : 32;
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
}

// unlike HAL API, which is in cv::hal,
// we put intrinsics into cv namespace to make its
// access from within opencv code more accessible
namespace cv {

namespace hal {

enum StoreMode
{
    STORE_UNALIGNED = 0,
    STORE_ALIGNED = 1,
    STORE_ALIGNED_NOCACHE = 2
};

}

// TODO FIXIT: Don't use "God" traits. Split on separate cases.
template<typename _Tp> struct V_TypeTraits
{
};

#define CV_INTRIN_DEF_TYPE_TRAITS(type, int_type_, uint_type_, abs_type_, w_type_, q_type_, sum_type_, nlanes128_) \
    template<> struct V_TypeTraits<type> \
    { \
        typedef type value_type; \
        typedef int_type_ int_type; \
        typedef abs_type_ abs_type; \
        typedef uint_type_ uint_type; \
        typedef w_type_ w_type; \
        typedef q_type_ q_type; \
        typedef sum_type_ sum_type; \
        enum { nlanes128 = nlanes128_ }; \
    \
        static inline int_type reinterpret_int(type x) \
        { \
            union { type l; int_type i; } v; \
            v.l = x; \
            return v.i; \
        } \
    \
        static inline type reinterpret_from_int(int_type x) \
        { \
            union { type l; int_type i; } v; \
            v.i = x; \
            return v.l; \
        } \
    }

#define CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(type, int_type_, uint_type_, abs_type_, w_type_, sum_type_, nlanes128_) \
    template<> struct V_TypeTraits<type> \
    { \
        typedef type value_type; \
        typedef int_type_ int_type; \
        typedef abs_type_ abs_type; \
        typedef uint_type_ uint_type; \
        typedef w_type_ w_type; \
        typedef sum_type_ sum_type; \
        enum { nlanes128 = nlanes128_ }; \
    \
        static inline int_type reinterpret_int(type x) \
        { \
            union { type l; int_type i; } v; \
            v.l = x; \
            return v.i; \
        } \
    \
        static inline type reinterpret_from_int(int_type x) \
        { \
            union { type l; int_type i; } v; \
            v.i = x; \
            return v.l; \
        } \
    }

CV_INTRIN_DEF_TYPE_TRAITS(uchar, schar, uchar, uchar, ushort, unsigned, unsigned, 16);
CV_INTRIN_DEF_TYPE_TRAITS(schar, schar, uchar, uchar, short, int, int, 16);
CV_INTRIN_DEF_TYPE_TRAITS(ushort, short, ushort, ushort, unsigned, uint64, unsigned, 8);
CV_INTRIN_DEF_TYPE_TRAITS(short, short, ushort, ushort, int, int64, int, 8);
CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(unsigned, int, unsigned, unsigned, uint64, unsigned, 4);
CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(int, int, unsigned, unsigned, int64, int, 4);
CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(float, int, unsigned, float, double, float, 4);
CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(uint64, int64, uint64, uint64, void, uint64, 2);
CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(int64, int64, uint64, uint64, void, int64, 2);
CV_INTRIN_DEF_TYPE_TRAITS_NO_Q_TYPE(double, int64, uint64, double, void, double, 2);

#ifndef CV_DOXYGEN

#ifndef CV_CPU_OPTIMIZATION_HAL_NAMESPACE
#ifdef CV_FORCE_SIMD128_CPP
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE hal_EMULATOR_CPP
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN namespace hal_EMULATOR_CPP {
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END }
#elif defined(CV_CPU_DISPATCH_MODE)
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE __CV_CAT(hal_, CV_CPU_DISPATCH_MODE)
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN namespace __CV_CAT(hal_, CV_CPU_DISPATCH_MODE) {
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END }
#else
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE hal_baseline
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN namespace hal_baseline {
    #define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END }
#endif
#endif // CV_CPU_OPTIMIZATION_HAL_NAMESPACE

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
using namespace CV_CPU_OPTIMIZATION_HAL_NAMESPACE;
#endif
}

#ifdef CV_DOXYGEN
#   undef CV_AVX2
#   undef CV_SSE2
#   undef CV_NEON
#   undef CV_VSX
#   undef CV_FP16
#   undef CV_MSA
#   undef CV_RVV
#endif

#if (CV_SSE2 || CV_NEON || CV_VSX || CV_MSA || CV_WASM_SIMD || CV_RVV) && !defined(CV_FORCE_SIMD128_CPP)
#define CV__SIMD_FORWARD 128
#include "opencv2/core/hal/intrin_forward.hpp"
#endif

#if CV_SSE2 && !defined(CV_FORCE_SIMD128_CPP)

#include "opencv2/core/hal/intrin_sse_em.hpp"
#include "opencv2/core/hal/intrin_sse.hpp"

#elif CV_NEON && !defined(CV_FORCE_SIMD128_CPP)

#include "opencv2/core/hal/intrin_neon.hpp"

#elif CV_VSX && !defined(CV_FORCE_SIMD128_CPP)

#include "opencv2/core/hal/intrin_vsx.hpp"

#elif CV_MSA && !defined(CV_FORCE_SIMD128_CPP)

#include "opencv2/core/hal/intrin_msa.hpp"

#elif CV_WASM_SIMD && !defined(CV_FORCE_SIMD128_CPP)
#include "opencv2/core/hal/intrin_wasm.hpp"

#elif CV_RVV && !defined(CV_FORCE_SIMD128_CPP)
#include "opencv2/core/hal/intrin_rvv.hpp"

#else

#include "opencv2/core/hal/intrin_cpp.hpp"

#endif

// AVX2 can be used together with SSE2, so
// we define those two sets of intrinsics at once.
// Most of the intrinsics do not conflict (the proper overloaded variant is
// resolved by the argument types, e.g. v_float32x4 ~ SSE2, v_float32x8 ~ AVX2),
// but some of AVX2 intrinsics get v256_ prefix instead of v_, e.g. v256_load() vs v_load().
// Correspondingly, the wide intrinsics (which are mapped to the "widest"
// available instruction set) will get vx_ prefix
// (and will be mapped to v256_ counterparts) (e.g. vx_load() => v256_load())
#if CV_AVX2

#define CV__SIMD_FORWARD 256
#include "opencv2/core/hal/intrin_forward.hpp"
#include "opencv2/core/hal/intrin_avx.hpp"

#endif

// AVX512 can be used together with SSE2 and AVX2, so
// we define those sets of intrinsics at once.
// For some of AVX512 intrinsics get v512_ prefix instead of v_, e.g. v512_load() vs v_load().
// Wide intrinsics will be mapped to v512_ counterparts in this case(e.g. vx_load() => v512_load())
#if CV_AVX512_SKX

#define CV__SIMD_FORWARD 512
#include "opencv2/core/hal/intrin_forward.hpp"
#include "opencv2/core/hal/intrin_avx512.hpp"

#endif

//! @cond IGNORED

namespace cv {

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
#endif

#ifndef CV_SIMD128
#define CV_SIMD128 0
#endif

#ifndef CV_SIMD128_CPP
#define CV_SIMD128_CPP 0
#endif

#ifndef CV_SIMD128_64F
#define CV_SIMD128_64F 0
#endif

#ifndef CV_SIMD256
#define CV_SIMD256 0
#endif

#ifndef CV_SIMD256_64F
#define CV_SIMD256_64F 0
#endif

#ifndef CV_SIMD512
#define CV_SIMD512 0
#endif

#ifndef CV_SIMD512_64F
#define CV_SIMD512_64F 0
#endif

#ifndef CV_SIMD128_FP16
#define CV_SIMD128_FP16 0
#endif

#ifndef CV_SIMD256_FP16
#define CV_SIMD256_FP16 0
#endif

#ifndef CV_SIMD512_FP16
#define CV_SIMD512_FP16 0
#endif

//==================================================================================================

#define CV_INTRIN_DEFINE_WIDE_INTRIN(typ, vtyp, short_typ, prefix, loadsfx) \
    inline vtyp vx_setall_##short_typ(typ v) { return prefix##_setall_##short_typ(v); } \
    inline vtyp vx_setzero_##short_typ() { return prefix##_setzero_##short_typ(); } \
    inline vtyp vx_##loadsfx(const typ* ptr) { return prefix##_##loadsfx(ptr); } \
    inline vtyp vx_##loadsfx##_aligned(const typ* ptr) { return prefix##_##loadsfx##_aligned(ptr); } \
    inline vtyp vx_##loadsfx##_low(const typ* ptr) { return prefix##_##loadsfx##_low(ptr); } \
    inline vtyp vx_##loadsfx##_halves(const typ* ptr0, const typ* ptr1) { return prefix##_##loadsfx##_halves(ptr0, ptr1); } \
    inline void vx_store(typ* ptr, const vtyp& v) { return v_store(ptr, v); } \
    inline void vx_store_aligned(typ* ptr, const vtyp& v) { return v_store_aligned(ptr, v); } \
    inline vtyp vx_lut(const typ* ptr, const int* idx) { return prefix##_lut(ptr, idx); } \
    inline vtyp vx_lut_pairs(const typ* ptr, const int* idx) { return prefix##_lut_pairs(ptr, idx); }

#define CV_INTRIN_DEFINE_WIDE_LUT_QUAD(typ, vtyp, prefix) \
    inline vtyp vx_lut_quads(const typ* ptr, const int* idx) { return prefix##_lut_quads(ptr, idx); }

#define CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(typ, wtyp, prefix) \
    inline wtyp vx_load_expand(const typ* ptr) { return prefix##_load_expand(ptr); }

#define CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND_Q(typ, qtyp, prefix) \
    inline qtyp vx_load_expand_q(const typ* ptr) { return prefix##_load_expand_q(ptr); }

#define CV_INTRIN_DEFINE_WIDE_INTRIN_WITH_EXPAND(typ, vtyp, short_typ, wtyp, qtyp, prefix, loadsfx) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(typ, vtyp, short_typ, prefix, loadsfx) \
    CV_INTRIN_DEFINE_WIDE_LUT_QUAD(typ, vtyp, prefix) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(typ, wtyp, prefix) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND_Q(typ, qtyp, prefix)

#define CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN_WITH_EXPAND(uchar, v_uint8, u8, v_uint16, v_uint32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN_WITH_EXPAND(schar, v_int8, s8, v_int16, v_int32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(ushort, v_uint16, u16, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_LUT_QUAD(ushort, v_uint16, prefix) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(ushort, v_uint32, prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(short, v_int16, s16, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_LUT_QUAD(short, v_int16, prefix) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(short, v_int32, prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(int, v_int32, s32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_LUT_QUAD(int, v_int32, prefix) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(int, v_int64, prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(unsigned, v_uint32, u32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_LUT_QUAD(unsigned, v_uint32, prefix) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(unsigned, v_uint64, prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(float, v_float32, f32, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_LUT_QUAD(float, v_float32, prefix) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(int64, v_int64, s64, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_INTRIN(uint64, v_uint64, u64, prefix, load) \
    CV_INTRIN_DEFINE_WIDE_LOAD_EXPAND(float16_t, v_float32, prefix)

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
    }

#if CV_SIMD128 || CV_SIMD128_CPP
    CV_DEF_REG_TRAITS(v, v_uint8x16, uchar, u8, v_uint8x16, v_uint16x8, v_uint32x4, v_int8x16, void);
    CV_DEF_REG_TRAITS(v, v_int8x16, schar, s8, v_uint8x16, v_int16x8, v_int32x4, v_int8x16, void);
    CV_DEF_REG_TRAITS(v, v_uint16x8, ushort, u16, v_uint16x8, v_uint32x4, v_uint64x2, v_int16x8, void);
    CV_DEF_REG_TRAITS(v, v_int16x8, short, s16, v_uint16x8, v_int32x4, v_int64x2, v_int16x8, void);
    CV_DEF_REG_TRAITS(v, v_uint32x4, unsigned, u32, v_uint32x4, v_uint64x2, void, v_int32x4, void);
    CV_DEF_REG_TRAITS(v, v_int32x4, int, s32, v_uint32x4, v_int64x2, void, v_int32x4, void);
#if CV_SIMD128_64F || CV_SIMD128_CPP
    CV_DEF_REG_TRAITS(v, v_float32x4, float, f32, v_float32x4, v_float64x2, void, v_int32x4, v_int32x4);
#else
    CV_DEF_REG_TRAITS(v, v_float32x4, float, f32, v_float32x4, void, void, v_int32x4, v_int32x4);
#endif
    CV_DEF_REG_TRAITS(v, v_uint64x2, uint64, u64, v_uint64x2, void, void, v_int64x2, void);
    CV_DEF_REG_TRAITS(v, v_int64x2, int64, s64, v_uint64x2, void, void, v_int64x2, void);
#if CV_SIMD128_64F
    CV_DEF_REG_TRAITS(v, v_float64x2, double, f64, v_float64x2, void, void, v_int64x2, v_int32x4);
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
#endif

#if CV_SIMD512
    CV_DEF_REG_TRAITS(v512, v_uint8x64, uchar, u8, v_uint8x64, v_uint16x32, v_uint32x16, v_int8x64, void);
    CV_DEF_REG_TRAITS(v512, v_int8x64, schar, s8, v_uint8x64, v_int16x32, v_int32x16, v_int8x64, void);
    CV_DEF_REG_TRAITS(v512, v_uint16x32, ushort, u16, v_uint16x32, v_uint32x16, v_uint64x8, v_int16x32, void);
    CV_DEF_REG_TRAITS(v512, v_int16x32, short, s16, v_uint16x32, v_int32x16, v_int64x8, v_int16x32, void);
    CV_DEF_REG_TRAITS(v512, v_uint32x16, unsigned, u32, v_uint32x16, v_uint64x8, void, v_int32x16, void);
    CV_DEF_REG_TRAITS(v512, v_int32x16, int, s32, v_uint32x16, v_int64x8, void, v_int32x16, void);
    CV_DEF_REG_TRAITS(v512, v_float32x16, float, f32, v_float32x16, v_float64x8, void, v_int32x16, v_int32x16);
    CV_DEF_REG_TRAITS(v512, v_uint64x8, uint64, u64, v_uint64x8, void, void, v_int64x8, void);
    CV_DEF_REG_TRAITS(v512, v_int64x8, int64, s64, v_uint64x8, void, void, v_int64x8, void);
    CV_DEF_REG_TRAITS(v512, v_float64x8, double, f64, v_float64x8, void, void, v_int64x8, v_int32x16);
#endif

#if CV_SIMD512 && (!defined(CV__SIMD_FORCE_WIDTH) || CV__SIMD_FORCE_WIDTH == 512)
#define CV__SIMD_NAMESPACE simd512
namespace CV__SIMD_NAMESPACE {
    #define CV_SIMD 1
    #define CV_SIMD_64F CV_SIMD512_64F
    #define CV_SIMD_FP16 CV_SIMD512_FP16
    #define CV_SIMD_WIDTH 64
    typedef v_uint8x64    v_uint8;
    typedef v_int8x64     v_int8;
    typedef v_uint16x32   v_uint16;
    typedef v_int16x32    v_int16;
    typedef v_uint32x16   v_uint32;
    typedef v_int32x16    v_int32;
    typedef v_uint64x8    v_uint64;
    typedef v_int64x8     v_int64;
    typedef v_float32x16  v_float32;
    CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(v512)
#if CV_SIMD512_64F
    typedef v_float64x8   v_float64;
    CV_INTRIN_DEFINE_WIDE_INTRIN(double, v_float64, f64, v512, load)
#endif
        inline void vx_cleanup() { v512_cleanup(); }
} // namespace
using namespace CV__SIMD_NAMESPACE;
#elif CV_SIMD256 && (!defined(CV__SIMD_FORCE_WIDTH) || CV__SIMD_FORCE_WIDTH == 256)
#define CV__SIMD_NAMESPACE simd256
namespace CV__SIMD_NAMESPACE {
    #define CV_SIMD 1
    #define CV_SIMD_64F CV_SIMD256_64F
    #define CV_SIMD_FP16 CV_SIMD256_FP16
    #define CV_SIMD_WIDTH 32
    typedef v_uint8x32   v_uint8;
    typedef v_int8x32    v_int8;
    typedef v_uint16x16  v_uint16;
    typedef v_int16x16   v_int16;
    typedef v_uint32x8   v_uint32;
    typedef v_int32x8    v_int32;
    typedef v_uint64x4   v_uint64;
    typedef v_int64x4    v_int64;
    typedef v_float32x8  v_float32;
    CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(v256)
    #if CV_SIMD256_64F
    typedef v_float64x4  v_float64;
    CV_INTRIN_DEFINE_WIDE_INTRIN(double, v_float64, f64, v256, load)
    #endif
    inline void vx_cleanup() { v256_cleanup(); }
} // namespace
using namespace CV__SIMD_NAMESPACE;
#elif (CV_SIMD128 || CV_SIMD128_CPP) && (!defined(CV__SIMD_FORCE_WIDTH) || CV__SIMD_FORCE_WIDTH == 128)
#if defined CV_SIMD128_CPP
#define CV__SIMD_NAMESPACE simd128_cpp
#else
#define CV__SIMD_NAMESPACE simd128
#endif
namespace CV__SIMD_NAMESPACE {
    #define CV_SIMD CV_SIMD128
    #define CV_SIMD_64F CV_SIMD128_64F
    #define CV_SIMD_WIDTH 16
    typedef v_uint8x16  v_uint8;
    typedef v_int8x16   v_int8;
    typedef v_uint16x8  v_uint16;
    typedef v_int16x8   v_int16;
    typedef v_uint32x4  v_uint32;
    typedef v_int32x4   v_int32;
    typedef v_uint64x2  v_uint64;
    typedef v_int64x2   v_int64;
    typedef v_float32x4 v_float32;
    CV_INTRIN_DEFINE_WIDE_INTRIN_ALL_TYPES(v)
    #if CV_SIMD128_64F
    typedef v_float64x2 v_float64;
    CV_INTRIN_DEFINE_WIDE_INTRIN(double, v_float64, f64, v, load)
    #endif
    inline void vx_cleanup() { v_cleanup(); }
} // namespace
using namespace CV__SIMD_NAMESPACE;
#endif

#ifndef CV_SIMD_64F
#define CV_SIMD_64F 0
#endif

#ifndef CV_SIMD_FP16
#define CV_SIMD_FP16 0  //!< Defined to 1 on native support of operations with float16x8_t / float16x16_t (SIMD256) types
#endif

#ifndef CV_SIMD
#define CV_SIMD 0
#endif

#include "simd_utils.impl.hpp"

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
#endif

} // cv::

//! @endcond

#endif
