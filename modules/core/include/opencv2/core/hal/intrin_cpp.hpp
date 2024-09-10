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

#ifndef OPENCV_HAL_INTRIN_CPP_HPP
#define OPENCV_HAL_INTRIN_CPP_HPP

#include <limits>
#include <cstring>
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include "opencv2/core/saturate.hpp"

//! @cond IGNORED
#define CV_SIMD128_CPP 1
#if defined(CV_FORCE_SIMD128_CPP)
#define CV_SIMD128 1
#define CV_SIMD128_64F 1
#endif
#if defined(CV_DOXYGEN)
#define CV_SIMD128 1
#define CV_SIMD128_64F 1
#define CV_SIMD256 1
#define CV_SIMD256_64F 1
#define CV_SIMD512 1
#define CV_SIMD512_64F 1
#else
#define CV_SIMD256 0 // Explicitly disable SIMD256 and SIMD512 support for scalar intrinsic implementation
#define CV_SIMD512 0 // to avoid warnings during compilation
#endif
//! @endcond

namespace cv
{

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
#endif

/** @addtogroup core_hal_intrin

"Universal intrinsics" is a types and functions set intended to simplify vectorization of code on
different platforms. Currently a few different SIMD extensions on different architectures are supported.
128 bit registers of various types support is implemented for a wide range of architectures
including x86(__SSE/SSE2/SSE4.2__), ARM(__NEON__), PowerPC(__VSX__), MIPS(__MSA__).
256 bit long registers are supported on x86(__AVX2__) and 512 bit long registers are supported on x86(__AVX512__).
In case when there is no SIMD extension available during compilation, fallback C++ implementation of intrinsics
will be chosen and code will work as expected although it could be slower.

### Types

There are several types representing packed values vector registers, each type is
implemented as a structure based on a one SIMD register.

- cv::v_uint8 and cv::v_int8: 8-bit integer values (unsigned/signed) - char
- cv::v_uint16 and cv::v_int16: 16-bit integer values (unsigned/signed) - short
- cv::v_uint32 and cv::v_int32: 32-bit integer values (unsigned/signed) - int
- cv::v_uint64 and cv::v_int64: 64-bit integer values (unsigned/signed) - int64
- cv::v_float32: 32-bit floating point values (signed) - float
- cv::v_float64: 64-bit floating point values (signed) - double

Exact bit length(and value quantity) of listed types is compile time deduced and depends on architecture SIMD
capabilities chosen as available during compilation of the library. All the types contains __nlanes__ enumeration
to check for exact value quantity of the type.

In case the exact bit length of the type is important it is possible to use specific fixed length register types.

There are several types representing 128-bit registers.

- cv::v_uint8x16 and cv::v_int8x16: sixteen 8-bit integer values (unsigned/signed) - char
- cv::v_uint16x8 and cv::v_int16x8: eight 16-bit integer values (unsigned/signed) - short
- cv::v_uint32x4 and cv::v_int32x4: four 32-bit integer values (unsigned/signed) - int
- cv::v_uint64x2 and cv::v_int64x2: two 64-bit integer values (unsigned/signed) - int64
- cv::v_float32x4: four 32-bit floating point values (signed) - float
- cv::v_float64x2: two 64-bit floating point values (signed) - double

There are several types representing 256-bit registers.

- cv::v_uint8x32 and cv::v_int8x32: thirty two 8-bit integer values (unsigned/signed) - char
- cv::v_uint16x16 and cv::v_int16x16: sixteen 16-bit integer values (unsigned/signed) - short
- cv::v_uint32x8 and cv::v_int32x8: eight 32-bit integer values (unsigned/signed) - int
- cv::v_uint64x4 and cv::v_int64x4: four 64-bit integer values (unsigned/signed) - int64
- cv::v_float32x8: eight 32-bit floating point values (signed) - float
- cv::v_float64x4: four 64-bit floating point values (signed) - double

@note
256 bit registers at the moment implemented for AVX2 SIMD extension only, if you want to use this type directly,
don't forget to check the CV_SIMD256 preprocessor definition:
@code
#if CV_SIMD256
//...
#endif
@endcode

There are several types representing 512-bit registers.

- cv::v_uint8x64 and cv::v_int8x64: sixty four 8-bit integer values (unsigned/signed) - char
- cv::v_uint16x32 and cv::v_int16x32: thirty two 16-bit integer values (unsigned/signed) - short
- cv::v_uint32x16 and cv::v_int32x16: sixteen 32-bit integer values (unsigned/signed) - int
- cv::v_uint64x8 and cv::v_int64x8: eight 64-bit integer values (unsigned/signed) - int64
- cv::v_float32x16: sixteen 32-bit floating point values (signed) - float
- cv::v_float64x8: eight 64-bit floating point values (signed) - double
@note
512 bit registers at the moment implemented for AVX512 SIMD extension only, if you want to use this type directly,
don't forget to check the CV_SIMD512 preprocessor definition.

@note
cv::v_float64x2 is not implemented in NEON variant, if you want to use this type, don't forget to
check the CV_SIMD128_64F preprocessor definition.

### Load and store operations

These operations allow to set contents of the register explicitly or by loading it from some memory
block and to save contents of the register to memory block.

There are variable size register load operations that provide result of maximum available size
depending on chosen platform capabilities.
- Constructors:
@ref v_reg::v_reg(const _Tp *ptr) "from memory",
- Other create methods:
vx_setall_s8, vx_setall_u8, ...,
vx_setzero_u8, vx_setzero_s8, ...
- Memory load operations:
vx_load, vx_load_aligned, vx_load_low, vx_load_halves,
- Memory operations with expansion of values:
vx_load_expand, vx_load_expand_q

Also there are fixed size register load/store operations.

For 128 bit registers
- Constructors:
@ref v_reg::v_reg(const _Tp *ptr) "from memory",
@ref v_reg::v_reg(_Tp s0, _Tp s1) "from two values", ...
- Other create methods:
@ref v_setall_s8, @ref v_setall_u8, ...,
@ref v_setzero_u8, @ref v_setzero_s8, ...
- Memory load operations:
@ref v_load, @ref v_load_aligned, @ref v_load_low, @ref v_load_halves,
- Memory operations with expansion of values:
@ref v_load_expand, @ref v_load_expand_q

For 256 bit registers(check CV_SIMD256 preprocessor definition)
- Constructors:
@ref v_reg::v_reg(const _Tp *ptr) "from memory",
@ref v_reg::v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3) "from four values", ...
- Other create methods:
@ref v256_setall_s8, @ref v256_setall_u8, ...,
@ref v256_setzero_u8, @ref v256_setzero_s8, ...
- Memory load operations:
@ref v256_load, @ref v256_load_aligned, @ref v256_load_low, @ref v256_load_halves,
- Memory operations with expansion of values:
@ref v256_load_expand, @ref v256_load_expand_q

For 512 bit registers(check CV_SIMD512 preprocessor definition)
- Constructors:
@ref v_reg::v_reg(const _Tp *ptr) "from memory",
@ref v_reg::v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3, _Tp s4, _Tp s5, _Tp s6, _Tp s7) "from eight values", ...
- Other create methods:
@ref v512_setall_s8, @ref v512_setall_u8, ...,
@ref v512_setzero_u8, @ref v512_setzero_s8, ...
- Memory load operations:
@ref v512_load, @ref v512_load_aligned, @ref v512_load_low, @ref v512_load_halves,
- Memory operations with expansion of values:
@ref v512_load_expand, @ref v512_load_expand_q

Store to memory operations are similar across different platform capabilities:
@ref v_store, @ref v_store_aligned,
@ref v_store_high, @ref v_store_low

### Value reordering

These operations allow to reorder or recombine elements in one or multiple vectors.

- Interleave, deinterleave (2, 3 and 4 channels): @ref v_load_deinterleave, @ref v_store_interleave
- Expand: @ref v_expand, @ref v_expand_low, @ref v_expand_high
- Pack: @ref v_pack, @ref v_pack_u, @ref v_pack_b, @ref v_rshr_pack, @ref v_rshr_pack_u,
@ref v_pack_store, @ref v_pack_u_store, @ref v_rshr_pack_store, @ref v_rshr_pack_u_store
- Recombine: @ref v_zip, @ref v_recombine, @ref v_combine_low, @ref v_combine_high
- Reverse: @ref v_reverse
- Extract: @ref v_extract


### Arithmetic, bitwise and comparison operations

Element-wise binary and unary operations.

- Arithmetics:
@ref v_add(const v_reg &a, const v_reg &b) "+",
@ref v_sub(const v_reg &a, const v_reg &b) "-",
@ref v_mul(const v_reg &a, const v_reg &b) "*",
@ref v_div(const v_reg &a, const v_reg &b) "/",
@ref v_mul_expand

- Non-saturating arithmetics: @ref v_add_wrap, @ref v_sub_wrap

- Bitwise shifts:
@ref v_shl(const v_reg &a, int s) "<<",
@ref v_shr(const v_reg &a, int s) ">>",
@ref v_shl, @ref v_shr

- Bitwise logic:
@ref v_and(const v_reg &a, const v_reg &b) "&",
@ref v_or(const v_reg &a, const v_reg &b) "|",
@ref v_xor(const v_reg &a, const v_reg &b) "^",
@ref v_not(const v_reg &a) "~"

- Comparison:
@ref v_gt(const v_reg &a, const v_reg &b) ">",
@ref v_ge(const v_reg &a, const v_reg &b) ">=",
@ref v_lt(const v_reg &a, const v_reg &b) "<",
@ref v_le(const v_reg &a, const v_reg &b) "<=",
@ref v_eq(const v_reg &a, const v_reg &b) "==",
@ref v_ne(const v_reg &a, const v_reg &b) "!="

- min/max: @ref v_min, @ref v_max

### Reduce and mask

Most of these operations return only one value.

- Reduce: @ref v_reduce_min, @ref v_reduce_max, @ref v_reduce_sum, @ref v_popcount
- Mask: @ref v_signmask, @ref v_check_all, @ref v_check_any, @ref v_select

### Other math

- Some frequent operations: @ref v_sqrt, @ref v_invsqrt, @ref v_magnitude, @ref v_sqr_magnitude, @ref v_exp, @ref v_log,
                            @ref v_erf
- Absolute values: @ref v_abs, @ref v_absdiff, @ref v_absdiffs

### Conversions

Different type conversions and casts:

- Rounding: @ref v_round, @ref v_floor, @ref v_ceil, @ref v_trunc,
- To float: @ref v_cvt_f32, @ref v_cvt_f64
- Reinterpret: @ref v_reinterpret_as_u8, @ref v_reinterpret_as_s8, ...

### Matrix operations

In these operations vectors represent matrix rows/columns: @ref v_dotprod, @ref v_dotprod_fast,
@ref v_dotprod_expand, @ref v_dotprod_expand_fast, @ref v_matmul, @ref v_transpose4x4

### Usability

Most operations are implemented only for some subset of the available types, following matrices
shows the applicability of different operations to the types.

Regular integers:

| Operations\\Types | uint 8 | int 8 | uint 16 | int 16 | uint 32 | int 32 |
|-------------------|:-:|:-:|:-:|:-:|:-:|:-:|
|load, store        | x | x | x | x | x | x |
|interleave         | x | x | x | x | x | x |
|expand             | x | x | x | x | x | x |
|expand_low         | x | x | x | x | x | x |
|expand_high        | x | x | x | x | x | x |
|expand_q           | x | x |   |   |   |   |
|add, sub           | x | x | x | x | x | x |
|add_wrap, sub_wrap | x | x | x | x |   |   |
|mul_wrap           | x | x | x | x |   |   |
|mul                | x | x | x | x | x | x |
|mul_expand         | x | x | x | x | x |   |
|compare            | x | x | x | x | x | x |
|shift              |   |   | x | x | x | x |
|dotprod            |   |   |   | x |   | x |
|dotprod_fast       |   |   |   | x |   | x |
|dotprod_expand     | x | x | x | x |   | x |
|dotprod_expand_fast| x | x | x | x |   | x |
|logical            | x | x | x | x | x | x |
|min, max           | x | x | x | x | x | x |
|absdiff            | x | x | x | x | x | x |
|absdiffs           |   | x |   | x |   |   |
|reduce             | x | x | x | x | x | x |
|mask               | x | x | x | x | x | x |
|pack               | x | x | x | x | x | x |
|pack_u             | x |   | x |   |   |   |
|pack_b             | x |   |   |   |   |   |
|unpack             | x | x | x | x | x | x |
|extract            | x | x | x | x | x | x |
|rotate (lanes)     | x | x | x | x | x | x |
|cvt_flt32          |   |   |   |   |   | x |
|cvt_flt64          |   |   |   |   |   | x |
|transpose4x4       |   |   |   |   | x | x |
|reverse            | x | x | x | x | x | x |
|extract_n          | x | x | x | x | x | x |
|broadcast_element  |   |   |   |   | x | x |

Big integers:

| Operations\\Types | uint 64 | int 64 |
|-------------------|:-:|:-:|
|load, store        | x | x |
|add, sub           | x | x |
|shift              | x | x |
|logical            | x | x |
|reverse            | x | x |
|extract            | x | x |
|rotate (lanes)     | x | x |
|cvt_flt64          |   | x |
|extract_n          | x | x |

Floating point:

| Operations\\Types | float 32 | float 64 |
|-------------------|:-:|:-:|
|load, store        | x | x |
|interleave         | x |   |
|add, sub           | x | x |
|mul                | x | x |
|div                | x | x |
|compare            | x | x |
|min, max           | x | x |
|absdiff            | x | x |
|reduce             | x |   |
|mask               | x | x |
|unpack             | x | x |
|cvt_flt32          |   | x |
|cvt_flt64          | x |   |
|sqrt, abs          | x | x |
|float math         | x | x |
|transpose4x4       | x |   |
|extract            | x | x |
|rotate (lanes)     | x | x |
|reverse            | x | x |
|extract_n          | x | x |
|broadcast_element  | x |   |
|exp                | x | x |
|log                | x | x |

 @{ */

template<typename _Tp, int n> struct v_reg
{
//! @cond IGNORED
    typedef _Tp lane_type;
    enum { nlanes = n };
// !@endcond

    /** @brief Constructor

    Initializes register with data from memory
    @param ptr pointer to memory block with data for register */
    explicit v_reg(const _Tp* ptr) { for( int i = 0; i < n; i++ ) s[i] = ptr[i]; }

    /** @brief Constructor

    Initializes register with two 64-bit values */
    v_reg(_Tp s0, _Tp s1) { s[0] = s0; s[1] = s1; }

    /** @brief Constructor

    Initializes register with four 32-bit values */
    v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3) { s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; }

    /** @brief Constructor

    Initializes register with eight 16-bit values */
    v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3,
           _Tp s4, _Tp s5, _Tp s6, _Tp s7)
    {
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        s[4] = s4; s[5] = s5; s[6] = s6; s[7] = s7;
    }

    /** @brief Constructor

    Initializes register with sixteen 8-bit values */
    v_reg(_Tp s0, _Tp s1, _Tp s2, _Tp s3,
           _Tp s4, _Tp s5, _Tp s6, _Tp s7,
           _Tp s8, _Tp s9, _Tp s10, _Tp s11,
           _Tp s12, _Tp s13, _Tp s14, _Tp s15)
    {
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        s[4] = s4; s[5] = s5; s[6] = s6; s[7] = s7;
        s[8] = s8; s[9] = s9; s[10] = s10; s[11] = s11;
        s[12] = s12; s[13] = s13; s[14] = s14; s[15] = s15;
    }

    /** @brief Default constructor

    Does not initialize anything*/
    v_reg() {}

    /** @brief Copy constructor */
    v_reg(const v_reg<_Tp, n> & r)
    {
        for( int i = 0; i < n; i++ )
            s[i] = r.s[i];
    }
    /** @brief Access first value

    Returns value of the first lane according to register type, for example:
    @code{.cpp}
    v_int32x4 r(1, 2, 3, 4);
    int v = r.get0(); // returns 1
    v_uint64x2 r(1, 2);
    uint64_t v = r.get0(); // returns 1
    @endcode
    */
    _Tp get0() const { return s[0]; }

//! @cond IGNORED
    _Tp get(const int i) const { return s[i]; }
    v_reg<_Tp, n> high() const
    {
        v_reg<_Tp, n> c;
        int i;
        for( i = 0; i < n/2; i++ )
        {
            c.s[i] = s[i+(n/2)];
            c.s[i+(n/2)] = 0;
        }
        return c;
    }

    static v_reg<_Tp, n> zero()
    {
        v_reg<_Tp, n> c;
        for( int i = 0; i < n; i++ )
            c.s[i] = (_Tp)0;
        return c;
    }

    static v_reg<_Tp, n> all(_Tp s)
    {
        v_reg<_Tp, n> c;
        for( int i = 0; i < n; i++ )
            c.s[i] = s;
        return c;
    }

    template<typename _Tp2, int n2> v_reg<_Tp2, n2> reinterpret_as() const
    {
        size_t bytes = std::min(sizeof(_Tp2)*n2, sizeof(_Tp)*n);
        v_reg<_Tp2, n2> c;
        std::memcpy(&c.s[0], &s[0], bytes);
        return c;
    }

    v_reg& operator=(const v_reg<_Tp, n> & r)
    {
        for( int i = 0; i < n; i++ )
            s[i] = r.s[i];
        return *this;
    }

    _Tp s[n];
//! @endcond
};

/** @brief Sixteen 8-bit unsigned integer values */
typedef v_reg<uchar, 16> v_uint8x16;
/** @brief Sixteen 8-bit signed integer values */
typedef v_reg<schar, 16> v_int8x16;
/** @brief Eight 16-bit unsigned integer values */
typedef v_reg<ushort, 8> v_uint16x8;
/** @brief Eight 16-bit signed integer values */
typedef v_reg<short, 8> v_int16x8;
/** @brief Four 32-bit unsigned integer values */
typedef v_reg<unsigned, 4> v_uint32x4;
/** @brief Four 32-bit signed integer values */
typedef v_reg<int, 4> v_int32x4;
/** @brief Four 32-bit floating point values (single precision) */
typedef v_reg<float, 4> v_float32x4;
/** @brief Two 64-bit floating point values (double precision) */
typedef v_reg<double, 2> v_float64x2;
/** @brief Two 64-bit unsigned integer values */
typedef v_reg<uint64, 2> v_uint64x2;
/** @brief Two 64-bit signed integer values */
typedef v_reg<int64, 2> v_int64x2;

#if CV_SIMD256
/** @brief Thirty two 8-bit unsigned integer values */
typedef v_reg<uchar, 32> v_uint8x32;
/** @brief Thirty two 8-bit signed integer values */
typedef v_reg<schar, 32> v_int8x32;
/** @brief Sixteen 16-bit unsigned integer values */
typedef v_reg<ushort, 16> v_uint16x16;
/** @brief Sixteen 16-bit signed integer values */
typedef v_reg<short, 16> v_int16x16;
/** @brief Eight 32-bit unsigned integer values */
typedef v_reg<unsigned, 8> v_uint32x8;
/** @brief Eight 32-bit signed integer values */
typedef v_reg<int, 8> v_int32x8;
/** @brief Eight 32-bit floating point values (single precision) */
typedef v_reg<float, 8> v_float32x8;
/** @brief Four 64-bit floating point values (double precision) */
typedef v_reg<double, 4> v_float64x4;
/** @brief Four 64-bit unsigned integer values */
typedef v_reg<uint64, 4> v_uint64x4;
/** @brief Four 64-bit signed integer values */
typedef v_reg<int64, 4> v_int64x4;
#endif

#if CV_SIMD512
/** @brief Sixty four 8-bit unsigned integer values */
typedef v_reg<uchar, 64> v_uint8x64;
/** @brief Sixty four 8-bit signed integer values */
typedef v_reg<schar, 64> v_int8x64;
/** @brief Thirty two 16-bit unsigned integer values */
typedef v_reg<ushort, 32> v_uint16x32;
/** @brief Thirty two 16-bit signed integer values */
typedef v_reg<short, 32> v_int16x32;
/** @brief Sixteen 32-bit unsigned integer values */
typedef v_reg<unsigned, 16> v_uint32x16;
/** @brief Sixteen 32-bit signed integer values */
typedef v_reg<int, 16> v_int32x16;
/** @brief Sixteen 32-bit floating point values (single precision) */
typedef v_reg<float, 16> v_float32x16;
/** @brief Eight 64-bit floating point values (double precision) */
typedef v_reg<double, 8> v_float64x8;
/** @brief Eight 64-bit unsigned integer values */
typedef v_reg<uint64, 8> v_uint64x8;
/** @brief Eight 64-bit signed integer values */
typedef v_reg<int64, 8> v_int64x8;
#endif

enum {
    simd128_width = 16,
#if CV_SIMD256
    simd256_width = 32,
#endif
#if CV_SIMD512
    simd512_width = 64,
    simdmax_width = simd512_width
#elif CV_SIMD256
    simdmax_width = simd256_width
#else
    simdmax_width = simd128_width
#endif
};

/** @brief Add values

For all types. */
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_add(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

/** @brief Subtract values

For all types. */
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_sub(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

/** @brief Multiply values

For 16- and 32-bit integer types and floating types. */
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_mul(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

/** @brief Divide values

For floating types only. */
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_div(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);


/** @brief Bitwise AND

Only for integer types. */
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_and(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

/** @brief Bitwise OR

Only for integer types. */
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_or(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

/** @brief Bitwise XOR

Only for integer types.*/
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_xor(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b);

/** @brief Bitwise NOT

Only for integer types.*/
template<typename _Tp, int n> CV_INLINE v_reg<_Tp, n> v_not(const v_reg<_Tp, n>& a);


#ifndef CV_DOXYGEN

#define CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(macro_name, ...) \
__CV_EXPAND(macro_name(uchar, __VA_ARGS__)) \
__CV_EXPAND(macro_name(schar, __VA_ARGS__)) \
__CV_EXPAND(macro_name(ushort, __VA_ARGS__)) \
__CV_EXPAND(macro_name(short, __VA_ARGS__)) \
__CV_EXPAND(macro_name(unsigned, __VA_ARGS__)) \
__CV_EXPAND(macro_name(int, __VA_ARGS__)) \
__CV_EXPAND(macro_name(uint64, __VA_ARGS__)) \
__CV_EXPAND(macro_name(int64, __VA_ARGS__)) \

#define CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(macro_name, ...) \
__CV_EXPAND(macro_name(float, __VA_ARGS__)) \
__CV_EXPAND(macro_name(double, __VA_ARGS__)) \

#define CV__HAL_INTRIN_EXPAND_WITH_ALL_TYPES(macro_name, ...) \
CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(macro_name, __VA_ARGS__) \
CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(macro_name, __VA_ARGS__) \

#define CV__HAL_INTRIN_IMPL_BIN_OP_(_Tp, bin_op, func) \
template<int n> inline \
v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return c; \
}

#define CV__HAL_INTRIN_IMPL_BIN_OP(bin_op, func) CV__HAL_INTRIN_EXPAND_WITH_ALL_TYPES(CV__HAL_INTRIN_IMPL_BIN_OP_, bin_op, func)

CV__HAL_INTRIN_IMPL_BIN_OP(+, v_add)
CV__HAL_INTRIN_IMPL_BIN_OP(-, v_sub)
CV__HAL_INTRIN_IMPL_BIN_OP(*, v_mul)
CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(CV__HAL_INTRIN_IMPL_BIN_OP_, /, v_div)

#define CV__HAL_INTRIN_IMPL_BIT_OP_(_Tp, bit_op, func) \
template<int n> CV_INLINE \
v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return c; \
}

#define CV__HAL_INTRIN_IMPL_BIT_OP(bit_op, func) \
CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(CV__HAL_INTRIN_IMPL_BIT_OP_, bit_op, func) \
CV__HAL_INTRIN_EXPAND_WITH_FP_TYPES(CV__HAL_INTRIN_IMPL_BIT_OP_, bit_op, func) /* TODO: FIXIT remove this after masks refactoring */


CV__HAL_INTRIN_IMPL_BIT_OP(&, v_and)
CV__HAL_INTRIN_IMPL_BIT_OP(|, v_or)
CV__HAL_INTRIN_IMPL_BIT_OP(^, v_xor)

#define CV__HAL_INTRIN_IMPL_BITWISE_NOT_(_Tp, dummy, dummy2) \
template<int n> CV_INLINE \
v_reg<_Tp, n> v_not(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int(~V_TypeTraits<_Tp>::reinterpret_int(a.s[i])); \
    return c; \
} \

CV__HAL_INTRIN_EXPAND_WITH_INTEGER_TYPES(CV__HAL_INTRIN_IMPL_BITWISE_NOT_, ~, v_not)

#endif  // !CV_DOXYGEN


//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_MATH_FUNC(func, cfunc, _Tp2) \
template<typename _Tp, int n> inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp2, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cfunc(a.s[i]); \
    return c; \
}

/** @brief Square root of elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_sqrt, std::sqrt, _Tp)

/**
 * @brief Exponential \f$ e^x \f$ of elements
 *
 * Only for floating point types. Core implementation steps:
 * 1. Decompose Input: Convert the input to \f$ 2^{x \cdot \log_2e} \f$ and split its exponential into integer and fractional parts:
 *    \f$ x \cdot \log_2e = n + f \f$, where \f$ n \f$ is the integer part and \f$ f \f$ is the fractional part.
 * 2. Compute \f$ 2^n \f$: Calculated by shifting the bits.
 * 3. Adjust Fractional Part: Compute \f$ f \cdot \ln2 \f$ to convert the fractional part to base \f$ e \f$.
 *    \f$ C1 \f$ and \f$ C2 \f$ are used to adjust the fractional part.
 * 4. Polynomial Approximation for \f$ e^{f \cdot \ln2} \f$: The closer the fractional part is to 0, the more accurate the result.
 *    - For float16 and float32, use a Taylor Series with 6 terms.
 *    - For float64, use Pade Polynomials Approximation with 4 terms.
 * 5. Combine Results: Multiply the two parts together to get the final result:
 *    \f$ e^x = 2^n \cdot e^{f \cdot \ln2} \f$.
 *
 * @note The precision of the calculation depends on the implementation and the data type of the input vector.
 */
OPENCV_HAL_IMPL_MATH_FUNC(v_exp, std::exp, _Tp)
#define OPENCV_HAL_MATH_HAVE_EXP 1

/**
 * @brief Natural logarithm \f$ \log(x) \f$ of elements
 *
 * Only for floating point types. Core implementation steps:
 * 1. Decompose Input: Use binary representation to decompose the input into mantissa part \f$ m \f$ and exponent part \f$ e \f$. Such that \f$ \log(x) = \log(m \cdot 2^e) = \log(m) + e \cdot \ln(2) \f$.
 * 2. Adjust Mantissa and Exponent Parts: If the mantissa is less than \f$ \sqrt{0.5} \f$, adjust the exponent and mantissa to ensure the mantissa is in the range \f$ (\sqrt{0.5}, \sqrt{2}) \f$ for better approximation.
 * 3. Polynomial Approximation for \f$ \log(m) \f$: The closer the \f$ m \f$ is to 1, the more accurate the result.
 *    - For float16 and float32, use a Taylor Series with 9 terms.
 *    - For float64, use Pade Polynomials Approximation with 6 terms.
 * 4. Combine Results: Add the two parts together to get the final result.
 *
 * @note The precision of the calculation depends on the implementation and the data type of the input.
 *
 * @note Similar to the behavior of std::log(), \f$ \ln(0) = -\infty \f$.
 */
OPENCV_HAL_IMPL_MATH_FUNC(v_log, std::log, _Tp)

/**
 * @brief Error function.
 *
 * @note Support FP32 precision for now.
 */
OPENCV_HAL_IMPL_MATH_FUNC(v_erf, std::erf, _Tp)

//! @cond IGNORED
OPENCV_HAL_IMPL_MATH_FUNC(v_sin, std::sin, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_cos, std::cos, _Tp)
//! @endcond

/** @brief Absolute value of elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_abs, (typename V_TypeTraits<_Tp>::abs_type)std::abs,
                          typename V_TypeTraits<_Tp>::abs_type)

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_MINMAX_FUNC(func, cfunc) \
template<typename _Tp, int n> inline v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cfunc(a.s[i], b.s[i]); \
    return c; \
}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(func, cfunc) \
template<typename _Tp, int n> inline _Tp func(const v_reg<_Tp, n>& a) \
{ \
    _Tp c = a.s[0]; \
    for( int i = 1; i < n; i++ ) \
        c = cfunc(c, a.s[i]); \
    return c; \
}

/** @brief Choose min values for each pair

Scheme:
@code
{A1 A2 ...}
{B1 B2 ...}
--------------
{min(A1,B1) min(A2,B2) ...}
@endcode
For all types except 64-bit integer. */
OPENCV_HAL_IMPL_MINMAX_FUNC(v_min, std::min)

/** @brief Choose max values for each pair

Scheme:
@code
{A1 A2 ...}
{B1 B2 ...}
--------------
{max(A1,B1) max(A2,B2) ...}
@endcode
For all types except 64-bit integer. */
OPENCV_HAL_IMPL_MINMAX_FUNC(v_max, std::max)

/** @brief Find one min value

Scheme:
@code
{A1 A2 A3 ...} => min(A1,A2,A3,...)
@endcode
For all types except 64-bit integer and 64-bit floating point types. */
OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(v_reduce_min, std::min)

/** @brief Find one max value

Scheme:
@code
{A1 A2 A3 ...} => max(A1,A2,A3,...)
@endcode
For all types except 64-bit integer and 64-bit floating point types. */
OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(v_reduce_max, std::max)

static const unsigned char popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};
/** @brief Count the 1 bits in the vector lanes and return result as corresponding unsigned type

Scheme:
@code
{A1 A2 A3 ...} => {popcount(A1), popcount(A2), popcount(A3), ...}
@endcode
For all integer types. */
template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::abs_type, n> v_popcount(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::abs_type, n> b = v_reg<typename V_TypeTraits<_Tp>::abs_type, n>::zero();
    for (int i = 0; i < n*(int)sizeof(_Tp); i++)
        b.s[i/sizeof(_Tp)] += popCountTable[v_reinterpret_as_u8(a).s[i]];
    return b;
}


//! @cond IGNORED
template<typename _Tp, int n>
inline void v_minmax( const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                      v_reg<_Tp, n>& minval, v_reg<_Tp, n>& maxval )
{
    for( int i = 0; i < n; i++ )
    {
        minval.s[i] = std::min(a.s[i], b.s[i]);
        maxval.s[i] = std::max(a.s[i], b.s[i]);
    }
}
//! @endcond

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_CMP_OP(cmp_op, func) \
template<typename _Tp, int n> \
inline v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)-(int)(a.s[i] cmp_op b.s[i])); \
    return c; \
}

/** @brief Less-than comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(<, v_lt)

/** @brief Greater-than comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(>, v_gt)

/** @brief Less-than or equal comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(<=, v_le)

/** @brief Greater-than or equal comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(>=, v_ge)

/** @brief Equal comparison */
OPENCV_HAL_IMPL_CMP_OP(==, v_eq)

/** @brief Not equal comparison */
OPENCV_HAL_IMPL_CMP_OP(!=, v_ne)

template<int n>
inline v_reg<float, n> v_not_nan(const v_reg<float, n>& a)
{
    typedef typename V_TypeTraits<float>::int_type itype;
    v_reg<float, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = V_TypeTraits<float>::reinterpret_from_int((itype)-(int)(a.s[i] == a.s[i]));
    return c;
}
template<int n>
inline v_reg<double, n> v_not_nan(const v_reg<double, n>& a)
{
    typedef typename V_TypeTraits<double>::int_type itype;
    v_reg<double, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = V_TypeTraits<double>::reinterpret_from_int((itype)-(int)(a.s[i] == a.s[i]));
    return c;
}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_ARITHM_OP(func, bin_op, cast_op, _Tp2) \
template<typename _Tp, int n> \
inline v_reg<_Tp2, n> func(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef _Tp2 rtype; \
    v_reg<rtype, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = cast_op(a.s[i] bin_op b.s[i]); \
    return c; \
}

/** @brief Add values without saturation

For 8- and 16-bit integer values. */
OPENCV_HAL_IMPL_ARITHM_OP(v_add_wrap, +, (_Tp), _Tp)

/** @brief Subtract values without saturation

For 8- and 16-bit integer values. */
OPENCV_HAL_IMPL_ARITHM_OP(v_sub_wrap, -, (_Tp), _Tp)

/** @brief Multiply values without saturation

For 8- and 16-bit integer values. */
OPENCV_HAL_IMPL_ARITHM_OP(v_mul_wrap, *, (_Tp), _Tp)

//! @cond IGNORED
template<typename T> inline T _absdiff(T a, T b)
{
    return a > b ? a - b : b - a;
}
//! @endcond

/** @brief Absolute difference

Returns \f$ |a - b| \f$ converted to corresponding unsigned type.
Example:
@code{.cpp}
v_int32x4 a, b; // {1, 2, 3, 4} and {4, 3, 2, 1}
v_uint32x4 c = v_absdiff(a, b); // result is {3, 1, 1, 3}
@endcode
For 8-, 16-, 32-bit integer source types. */
template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::abs_type, n> v_absdiff(const v_reg<_Tp, n>& a, const v_reg<_Tp, n> & b)
{
    typedef typename V_TypeTraits<_Tp>::abs_type rtype;
    v_reg<rtype, n> c;
    const rtype mask = (rtype)(std::numeric_limits<_Tp>::is_signed ? (1 << (sizeof(rtype)*8 - 1)) : 0);
    for( int i = 0; i < n; i++ )
    {
        rtype ua = a.s[i] ^ mask;
        rtype ub = b.s[i] ^ mask;
        c.s[i] = _absdiff(ua, ub);
    }
    return c;
}

/** @overload

For 32-bit floating point values */
template<int n> inline v_reg<float, n> v_absdiff(const v_reg<float, n>& a, const v_reg<float, n>& b)
{
    v_reg<float, n> c;
    for( int i = 0; i < c.nlanes; i++ )
        c.s[i] = _absdiff(a.s[i], b.s[i]);
    return c;
}

/** @overload

For 64-bit floating point values */
template<int n> inline v_reg<double, n> v_absdiff(const v_reg<double, n>& a, const v_reg<double, n>& b)
{
    v_reg<double, n> c;
    for( int i = 0; i < c.nlanes; i++ )
        c.s[i] = _absdiff(a.s[i], b.s[i]);
    return c;
}

/** @brief Saturating absolute difference

Returns \f$ saturate(|a - b|) \f$ .
For 8-, 16-bit signed integer source types. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_absdiffs(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++)
        c.s[i] = saturate_cast<_Tp>(std::abs(a.s[i] - b.s[i]));
    return c;
}

/** @brief Inversed square root

Returns \f$ 1/sqrt(a) \f$
For floating point types only. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_invsqrt(const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = 1.f/std::sqrt(a.s[i]);
    return c;
}

/** @brief Magnitude

Returns \f$ sqrt(a^2 + b^2) \f$
For floating point types only. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_magnitude(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = std::sqrt(a.s[i]*a.s[i] + b.s[i]*b.s[i]);
    return c;
}

/** @brief Square of the magnitude

Returns \f$ a^2 + b^2 \f$
For floating point types only. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_sqr_magnitude(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = a.s[i]*a.s[i] + b.s[i]*b.s[i];
    return c;
}

/** @brief Multiply and add

 Returns \f$ a*b + c \f$
 For floating point types and signed 32bit int only. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_fma(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                           const v_reg<_Tp, n>& c)
{
    v_reg<_Tp, n> d;
    for( int i = 0; i < n; i++ )
        d.s[i] = a.s[i]*b.s[i] + c.s[i];
    return d;
}

/** @brief A synonym for v_fma */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_muladd(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                              const v_reg<_Tp, n>& c)
{
    return v_fma(a, b, c);
}

/** @brief Dot product of elements

Multiply values in two registers and sum adjacent result pairs.

Scheme:
@code
  {A1 A2 ...} // 16-bit
x {B1 B2 ...} // 16-bit
-------------
{A1B1+A2B2 ...} // 32-bit

@endcode
*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n/2> c;
    for( int i = 0; i < (n/2); i++ )
        c.s[i] = (w_type)a.s[i*2]*b.s[i*2] + (w_type)a.s[i*2+1]*b.s[i*2+1];
    return c;
}

/** @brief Dot product of elements

Same as cv::v_dotprod, but add a third element to the sum of adjacent pairs.
Scheme:
@code
  {A1 A2 ...} // 16-bit
x {B1 B2 ...} // 16-bit
-------------
  {A1B1+A2B2+C1 ...} // 32-bit
@endcode
*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
          const v_reg<typename V_TypeTraits<_Tp>::w_type, n / 2>& c)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, n/2> s;
    for( int i = 0; i < (n/2); i++ )
        s.s[i] = (w_type)a.s[i*2]*b.s[i*2] + (w_type)a.s[i*2+1]*b.s[i*2+1] + c.s[i];
    return s;
}

/** @brief Fast Dot product of elements

Same as cv::v_dotprod, but it may perform unorder sum between result pairs in some platforms,
this intrinsic can be used if the sum among all lanes is only matters
and also it should be yielding better performance on the affected platforms.

*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{ return v_dotprod(a, b); }

/** @brief Fast Dot product of elements

Same as cv::v_dotprod_fast, but add a third element to the sum of adjacent pairs.
*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_dotprod_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
               const v_reg<typename V_TypeTraits<_Tp>::w_type, n / 2>& c)
{ return v_dotprod(a, b, c); }

/** @brief Dot product of elements and expand

Multiply values in two registers and expand the sum of adjacent result pairs.

Scheme:
@code
  {A1 A2 A3 A4 ...} // 8-bit
x {B1 B2 B3 B4 ...} // 8-bit
-------------
  {A1B1+A2B2+A3B3+A4B4 ...} // 32-bit

@endcode
*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, n/4> s;
    for( int i = 0; i < (n/4); i++ )
        s.s[i] = (q_type)a.s[i*4    ]*b.s[i*4    ] + (q_type)a.s[i*4 + 1]*b.s[i*4 + 1] +
                 (q_type)a.s[i*4 + 2]*b.s[i*4 + 2] + (q_type)a.s[i*4 + 3]*b.s[i*4 + 3];
    return s;
}

/** @brief Dot product of elements

Same as cv::v_dotprod_expand, but add a third element to the sum of adjacent pairs.
Scheme:
@code
  {A1 A2 A3 A4 ...} // 8-bit
x {B1 B2 B3 B4 ...} // 8-bit
-------------
  {A1B1+A2B2+A3B3+A4B4+C1 ...} // 32-bit
@endcode
*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                 const v_reg<typename V_TypeTraits<_Tp>::q_type, n / 4>& c)
{
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, n/4> s;
    for( int i = 0; i < (n/4); i++ )
        s.s[i] = (q_type)a.s[i*4    ]*b.s[i*4    ] + (q_type)a.s[i*4 + 1]*b.s[i*4 + 1] +
                 (q_type)a.s[i*4 + 2]*b.s[i*4 + 2] + (q_type)a.s[i*4 + 3]*b.s[i*4 + 3] + c.s[i];
    return s;
}

/** @brief Fast Dot product of elements and expand

Multiply values in two registers and expand the sum of adjacent result pairs.

Same as cv::v_dotprod_expand, but it may perform unorder sum between result pairs in some platforms,
this intrinsic can be used if the sum among all lanes is only matters
and also it should be yielding better performance on the affected platforms.

*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{ return v_dotprod_expand(a, b); }

/** @brief Fast Dot product of elements

Same as cv::v_dotprod_expand_fast, but add a third element to the sum of adjacent pairs.
*/
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::q_type, n/4>
v_dotprod_expand_fast(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                      const v_reg<typename V_TypeTraits<_Tp>::q_type, n / 4>& c)
{ return v_dotprod_expand(a, b, c); }

/** @brief Multiply and expand

Multiply values two registers and store results in two registers with wider pack type.
Scheme:
@code
  {A B C D} // 32-bit
x {E F G H} // 32-bit
---------------
{AE BF}         // 64-bit
        {CG DH} // 64-bit
@endcode
Example:
@code{.cpp}
v_uint32x4 a, b; // {1,2,3,4} and {2,2,2,2}
v_uint64x2 c, d; // results
v_mul_expand(a, b, c, d); // c, d = {2,4}, {6, 8}
@endcode
Implemented only for 16- and unsigned 32-bit source types (v_int16x8, v_uint16x8, v_uint32x4).
*/
template<typename _Tp, int n> inline void v_mul_expand(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                                                       v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& c,
                                                       v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& d)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = (w_type)a.s[i]*b.s[i];
        d.s[i] = (w_type)a.s[i+(n/2)]*b.s[i+(n/2)];
    }
}

/** @brief Multiply and extract high part

Multiply values two registers and store high part of the results.
Implemented only for 16-bit source types (v_int16x8, v_uint16x8). Returns \f$ a*b >> 16 \f$
*/
template<typename _Tp, int n> inline v_reg<_Tp, n> v_mul_hi(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<_Tp, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = (_Tp)(((w_type)a.s[i] * b.s[i]) >> sizeof(_Tp)*8);
    return c;
}

//! @cond IGNORED
template<typename _Tp, int n> inline void v_hsum(const v_reg<_Tp, n>& a,
                                                 v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& c)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = (w_type)a.s[i*2] + a.s[i*2+1];
    }
}
//! @endcond

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_SHIFT_OP(shift_op, func) \
template<typename _Tp, int n> inline v_reg<_Tp, n> func(const v_reg<_Tp, n>& a, int imm) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = (_Tp)(a.s[i] shift_op imm); \
    return c; \
}

/** @brief Bitwise shift left

For 16-, 32- and 64-bit integer values. */
OPENCV_HAL_IMPL_SHIFT_OP(<<, v_shl)

/** @brief Bitwise shift right

For 16-, 32- and 64-bit integer values. */
OPENCV_HAL_IMPL_SHIFT_OP(>>, v_shr)

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_ROTATE_SHIFT_OP(suffix,opA,opB) \
template<int imm, typename _Tp, int n> inline v_reg<_Tp, n> v_rotate_##suffix(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp, n> b; \
    for (int i = 0; i < n; i++) \
    { \
        int sIndex = i opA imm; \
        if (0 <= sIndex && sIndex < n) \
        { \
            b.s[i] = a.s[sIndex]; \
        } \
        else \
        { \
            b.s[i] = 0; \
        } \
    } \
    return b; \
} \
template<int imm, typename _Tp, int n> inline v_reg<_Tp, n> v_rotate_##suffix(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for (int i = 0; i < n; i++) \
    { \
        int aIndex = i opA imm; \
        int bIndex = i opA imm opB n; \
        if (0 <= bIndex && bIndex < n) \
        { \
            c.s[i] = b.s[bIndex]; \
        } \
        else if (0 <= aIndex && aIndex < n) \
        { \
            c.s[i] = a.s[aIndex]; \
        } \
        else \
        { \
            c.s[i] = 0; \
        } \
    } \
    return c; \
}

/** @brief Element shift left among vector

For all type */
OPENCV_HAL_IMPL_ROTATE_SHIFT_OP(left,  -, +)

/** @brief Element shift right among vector

For all type */
OPENCV_HAL_IMPL_ROTATE_SHIFT_OP(right, +, -)

/** @brief Sum packed values

Scheme:
@code
{A1 A2 A3 ...} => sum{A1,A2,A3,...}
@endcode
*/
template<typename _Tp, int n> inline typename V_TypeTraits<_Tp>::sum_type v_reduce_sum(const v_reg<_Tp, n>& a)
{
    typename V_TypeTraits<_Tp>::sum_type c = a.s[0];
    for( int i = 1; i < n; i++ )
        c += a.s[i];
    return c;
}

/** @brief Sums all elements of each input vector, returns the vector of sums

 Scheme:
 @code
 result[0] = a[0] + a[1] + a[2] + a[3]
 result[1] = b[0] + b[1] + b[2] + b[3]
 result[2] = c[0] + c[1] + c[2] + c[3]
 result[3] = d[0] + d[1] + d[2] + d[3]
 @endcode
*/
template<int n> inline v_reg<float, n> v_reduce_sum4(const v_reg<float, n>& a, const v_reg<float, n>& b,
    const v_reg<float, n>& c, const v_reg<float, n>& d)
{
    v_reg<float, n> r;
    for(int i = 0; i < (n/4); i++)
    {
        r.s[i*4 + 0] = a.s[i*4 + 0] + a.s[i*4 + 1] + a.s[i*4 + 2] + a.s[i*4 + 3];
        r.s[i*4 + 1] = b.s[i*4 + 0] + b.s[i*4 + 1] + b.s[i*4 + 2] + b.s[i*4 + 3];
        r.s[i*4 + 2] = c.s[i*4 + 0] + c.s[i*4 + 1] + c.s[i*4 + 2] + c.s[i*4 + 3];
        r.s[i*4 + 3] = d.s[i*4 + 0] + d.s[i*4 + 1] + d.s[i*4 + 2] + d.s[i*4 + 3];
    }
    return r;
}

/** @brief Sum absolute differences of values

Scheme:
@code
{A1 A2 A3 ...} {B1 B2 B3 ...} => sum{ABS(A1-B1),abs(A2-B2),abs(A3-B3),...}
@endcode
For all types except 64-bit types.*/
template<typename _Tp, int n> inline typename V_TypeTraits< typename V_TypeTraits<_Tp>::abs_type >::sum_type v_reduce_sad(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typename V_TypeTraits< typename V_TypeTraits<_Tp>::abs_type >::sum_type c = _absdiff(a.s[0], b.s[0]);
    for (int i = 1; i < n; i++)
        c += _absdiff(a.s[i], b.s[i]);
    return c;
}

/** @brief Get negative values mask
@deprecated v_signmask depends on a lane count heavily and therefore isn't universal enough

Returned value is a bit mask with bits set to 1 on places corresponding to negative packed values indexes.
Example:
@code{.cpp}
v_int32x4 r; // set to {-1, -1, 1, 1}
int mask = v_signmask(r); // mask = 3 <== 00000000 00000000 00000000 00000011
@endcode
*/
template<typename _Tp, int n> inline int v_signmask(const v_reg<_Tp, n>& a)
{
    int mask = 0;
    for( int i = 0; i < n; i++ )
        mask |= (V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0) << i;
    return mask;
}

/** @brief Get first negative lane index

Returned value is an index of first negative lane (undefined for input of all positive values)
Example:
@code{.cpp}
v_int32x4 r; // set to {0, 0, -1, -1}
int idx = v_heading_zeros(r); // idx = 2
@endcode
*/
template <typename _Tp, int n> inline int v_scan_forward(const v_reg<_Tp, n>& a)
{
    for (int i = 0; i < n; i++)
        if(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0)
            return i;
    return 0;
}

/** @brief Check if all packed values are less than zero

Unsigned values will be casted to signed: `uchar 254 => char -2`.
*/
template<typename _Tp, int n> inline bool v_check_all(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) >= 0 )
            return false;
    return true;
}

/** @brief Check if any of packed values is less than zero

Unsigned values will be casted to signed: `uchar 254 => char -2`.
*/
template<typename _Tp, int n> inline bool v_check_any(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0 )
            return true;
    return false;
}

/** @brief Per-element select (blend operation)

Return value will be built by combining values _a_ and _b_ using the following scheme:
    result[i] = mask[i] ? a[i] : b[i];

@note: _mask_ element values are restricted to these values:
- 0: select element from _b_
- 0xff/0xffff/etc: select element from _a_
(fully compatible with bitwise-based operator)
*/
template<typename _Tp, int n> inline v_reg<_Tp, n> v_select(const v_reg<_Tp, n>& mask,
                                                           const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef V_TypeTraits<_Tp> Traits;
    typedef typename Traits::int_type int_type;
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
    {
        int_type m = Traits::reinterpret_int(mask.s[i]);
        CV_DbgAssert(m == 0 || m == (~(int_type)0));  // restrict mask values: 0 or 0xff/0xffff/etc
        c.s[i] = m ? a.s[i] : b.s[i];
    }
    return c;
}

/** @brief Expand values to the wider pack type

Copy contents of register to two registers with 2x wider pack type.
Scheme:
@code
 int32x4     int64x2 int64x2
{A B C D} ==> {A B} , {C D}
@endcode */
template<typename _Tp, int n> inline void v_expand(const v_reg<_Tp, n>& a,
                            v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& b0,
                            v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>& b1)
{
    for( int i = 0; i < (n/2); i++ )
    {
        b0.s[i] = a.s[i];
        b1.s[i] = a.s[i+(n/2)];
    }
}

/** @brief Expand lower values to the wider pack type

Same as cv::v_expand, but return lower half of the vector.

Scheme:
@code
 int32x4     int64x2
{A B C D} ==> {A B}
@endcode */
template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_expand_low(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::w_type, n/2> b;
    for( int i = 0; i < (n/2); i++ )
        b.s[i] = a.s[i];
    return b;
}

/** @brief Expand higher values to the wider pack type

Same as cv::v_expand_low, but expand higher half of the vector instead.

Scheme:
@code
 int32x4     int64x2
{A B C D} ==> {C D}
@endcode */
template<typename _Tp, int n>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, n/2>
v_expand_high(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::w_type, n/2> b;
    for( int i = 0; i < (n/2); i++ )
        b.s[i] = a.s[i+(n/2)];
    return b;
}

//! @cond IGNORED
template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::int_type, n>
    v_reinterpret_as_int(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::int_type, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_int(a.s[i]);
    return c;
}

template<typename _Tp, int n> inline v_reg<typename V_TypeTraits<_Tp>::uint_type, n>
    v_reinterpret_as_uint(const v_reg<_Tp, n>& a)
{
    v_reg<typename V_TypeTraits<_Tp>::uint_type, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_uint(a.s[i]);
    return c;
}
//! @endcond

/** @brief Interleave two vectors

Scheme:
@code
  {A1 A2 A3 A4}
  {B1 B2 B3 B4}
---------------
  {A1 B1 A2 B2} and {A3 B3 A4 B4}
@endcode
For all types except 64-bit.
*/
template<typename _Tp, int n> inline void v_zip( const v_reg<_Tp, n>& a0, const v_reg<_Tp, n>& a1,
                                               v_reg<_Tp, n>& b0, v_reg<_Tp, n>& b1 )
{
    int i;
    for( i = 0; i < n/2; i++ )
    {
        b0.s[i*2] = a0.s[i];
        b0.s[i*2+1] = a1.s[i];
    }
    for( ; i < n; i++ )
    {
        b1.s[i*2-n] = a0.s[i];
        b1.s[i*2-n+1] = a1.s[i];
    }
}

/** @brief Load register contents from memory

@param ptr pointer to memory block with data
@return register object

@note Returned type will be detected from passed pointer type, for example uchar ==> cv::v_uint8x16, int ==> cv::v_int32x4, etc.

@note Use vx_load version to get maximum available register length result

@note Alignment requirement:
if CV_STRONG_ALIGNMENT=1 then passed pointer must be aligned (`sizeof(lane type)` should be enough).
Do not cast pointer types without runtime check for pointer alignment (like `uchar*` => `int*`).
 */
template<typename _Tp>
inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_load(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    return v_reg<_Tp, simd128_width / sizeof(_Tp)>(ptr);
}

#if CV_SIMD256
/** @brief Load 256-bit length register contents from memory

@param ptr pointer to memory block with data
@return register object

@note Returned type will be detected from passed pointer type, for example uchar ==> cv::v_uint8x32, int ==> cv::v_int32x8, etc.

@note Check CV_SIMD256 preprocessor definition prior to use.
Use vx_load version to get maximum available register length result

@note Alignment requirement:
if CV_STRONG_ALIGNMENT=1 then passed pointer must be aligned (`sizeof(lane type)` should be enough).
Do not cast pointer types without runtime check for pointer alignment (like `uchar*` => `int*`).
 */
template<typename _Tp>
inline v_reg<_Tp, simd256_width / sizeof(_Tp)> v256_load(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    return v_reg<_Tp, simd256_width / sizeof(_Tp)>(ptr);
}
#endif

#if CV_SIMD512
/** @brief Load 512-bit length register contents from memory

@param ptr pointer to memory block with data
@return register object

@note Returned type will be detected from passed pointer type, for example uchar ==> cv::v_uint8x64, int ==> cv::v_int32x16, etc.

@note Check CV_SIMD512 preprocessor definition prior to use.
Use vx_load version to get maximum available register length result

@note Alignment requirement:
if CV_STRONG_ALIGNMENT=1 then passed pointer must be aligned (`sizeof(lane type)` should be enough).
Do not cast pointer types without runtime check for pointer alignment (like `uchar*` => `int*`).
 */
template<typename _Tp>
inline v_reg<_Tp, simd512_width / sizeof(_Tp)> v512_load(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    return v_reg<_Tp, simd512_width / sizeof(_Tp)>(ptr);
}
#endif

/** @brief Load register contents from memory (aligned)

similar to cv::v_load, but source memory block should be aligned (to 16-byte boundary in case of SIMD128, 32-byte - SIMD256, etc)

@note Use vx_load_aligned version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_load_aligned(const _Tp* ptr)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, simd128_width / sizeof(_Tp)>)>(ptr));
    return v_reg<_Tp, simd128_width / sizeof(_Tp)>(ptr);
}

#if CV_SIMD256
/** @brief Load register contents from memory (aligned)

similar to cv::v256_load, but source memory block should be aligned (to 32-byte boundary in case of SIMD256, 64-byte - SIMD512, etc)

@note Check CV_SIMD256 preprocessor definition prior to use.
Use vx_load_aligned version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd256_width / sizeof(_Tp)> v256_load_aligned(const _Tp* ptr)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, simd256_width / sizeof(_Tp)>)>(ptr));
    return v_reg<_Tp, simd256_width / sizeof(_Tp)>(ptr);
}
#endif

#if CV_SIMD512
/** @brief Load register contents from memory (aligned)

similar to cv::v512_load, but source memory block should be aligned (to 64-byte boundary in case of SIMD512, etc)

@note Check CV_SIMD512 preprocessor definition prior to use.
Use vx_load_aligned version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd512_width / sizeof(_Tp)> v512_load_aligned(const _Tp* ptr)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, simd512_width / sizeof(_Tp)>)>(ptr));
    return v_reg<_Tp, simd512_width / sizeof(_Tp)>(ptr);
}
#endif

/** @brief Load 64-bits of data to lower part (high part is undefined).

@param ptr memory block containing data for first half (0..n/2)

@code{.cpp}
int lo[2] = { 1, 2 };
v_int32x4 r = v_load_low(lo);
@endcode

@note Use vx_load_low version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_load_low(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    v_reg<_Tp, simd128_width / sizeof(_Tp)> c;
    for( int i = 0; i < c.nlanes/2; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

#if CV_SIMD256
/** @brief Load 128-bits of data to lower part (high part is undefined).

@param ptr memory block containing data for first half (0..n/2)

@code{.cpp}
int lo[4] = { 1, 2, 3, 4 };
v_int32x8 r = v256_load_low(lo);
@endcode

@note Check CV_SIMD256 preprocessor definition prior to use.
Use vx_load_low version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd256_width / sizeof(_Tp)> v256_load_low(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    v_reg<_Tp, simd256_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes / 2; i++)
    {
        c.s[i] = ptr[i];
    }
    return c;
}
#endif

#if CV_SIMD512
/** @brief Load 256-bits of data to lower part (high part is undefined).

@param ptr memory block containing data for first half (0..n/2)

@code{.cpp}
int lo[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
v_int32x16 r = v512_load_low(lo);
@endcode

@note Check CV_SIMD512 preprocessor definition prior to use.
Use vx_load_low version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd512_width / sizeof(_Tp)> v512_load_low(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    v_reg<_Tp, simd512_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes / 2; i++)
    {
        c.s[i] = ptr[i];
    }
    return c;
}
#endif

/** @brief Load register contents from two memory blocks

@param loptr memory block containing data for first half (0..n/2)
@param hiptr memory block containing data for second half (n/2..n)

@code{.cpp}
int lo[2] = { 1, 2 }, hi[2] = { 3, 4 };
v_int32x4 r = v_load_halves(lo, hi);
@endcode

@note Use vx_load_halves version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_load_halves(const _Tp* loptr, const _Tp* hiptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(loptr));
    CV_Assert(isAligned<sizeof(_Tp)>(hiptr));
#endif
    v_reg<_Tp, simd128_width / sizeof(_Tp)> c;
    for( int i = 0; i < c.nlanes/2; i++ )
    {
        c.s[i] = loptr[i];
        c.s[i+c.nlanes/2] = hiptr[i];
    }
    return c;
}

#if CV_SIMD256
/** @brief Load register contents from two memory blocks

@param loptr memory block containing data for first half (0..n/2)
@param hiptr memory block containing data for second half (n/2..n)

@code{.cpp}
int lo[4] = { 1, 2, 3, 4 }, hi[4] = { 5, 6, 7, 8 };
v_int32x8 r = v256_load_halves(lo, hi);
@endcode

@note Check CV_SIMD256 preprocessor definition prior to use.
Use vx_load_halves version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd256_width / sizeof(_Tp)> v256_load_halves(const _Tp* loptr, const _Tp* hiptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(loptr));
    CV_Assert(isAligned<sizeof(_Tp)>(hiptr));
#endif
    v_reg<_Tp, simd256_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes / 2; i++)
    {
        c.s[i] = loptr[i];
        c.s[i + c.nlanes / 2] = hiptr[i];
    }
    return c;
}
#endif

#if CV_SIMD512
/** @brief Load register contents from two memory blocks

@param loptr memory block containing data for first half (0..n/2)
@param hiptr memory block containing data for second half (n/2..n)

@code{.cpp}
int lo[4] = { 1, 2, 3, 4, 5, 6, 7, 8 }, hi[4] = { 9, 10, 11, 12, 13, 14, 15, 16 };
v_int32x16 r = v512_load_halves(lo, hi);
@endcode

@note Check CV_SIMD512 preprocessor definition prior to use.
Use vx_load_halves version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<_Tp, simd512_width / sizeof(_Tp)> v512_load_halves(const _Tp* loptr, const _Tp* hiptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(loptr));
    CV_Assert(isAligned<sizeof(_Tp)>(hiptr));
#endif
    v_reg<_Tp, simd512_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes / 2; i++)
    {
        c.s[i] = loptr[i];
        c.s[i + c.nlanes / 2] = hiptr[i];
    }
    return c;
}
#endif

/** @brief Load register contents from memory with double expand

Same as cv::v_load, but result pack type will be 2x wider than memory type.

@code{.cpp}
short buf[4] = {1, 2, 3, 4}; // type is int16
v_int32x4 r = v_load_expand(buf); // r = {1, 2, 3, 4} - type is int32
@endcode
For 8-, 16-, 32-bit integer source types.

@note Use vx_load_expand version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, simd128_width / sizeof(typename V_TypeTraits<_Tp>::w_type)>
v_load_expand(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, simd128_width / sizeof(w_type)> c;
    for( int i = 0; i < c.nlanes; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

#if CV_SIMD256
/** @brief Load register contents from memory with double expand

Same as cv::v256_load, but result pack type will be 2x wider than memory type.

@code{.cpp}
short buf[8] = {1, 2, 3, 4, 5, 6, 7, 8}; // type is int16
v_int32x8 r = v256_load_expand(buf); // r = {1, 2, 3, 4, 5, 6, 7, 8} - type is int32
@endcode
For 8-, 16-, 32-bit integer source types.

@note Check CV_SIMD256 preprocessor definition prior to use.
Use vx_load_expand version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, simd256_width / sizeof(typename V_TypeTraits<_Tp>::w_type)>
v256_load_expand(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, simd256_width / sizeof(w_type)> c;
    for (int i = 0; i < c.nlanes; i++)
    {
        c.s[i] = ptr[i];
    }
    return c;
}
#endif

#if CV_SIMD512
/** @brief Load register contents from memory with double expand

Same as cv::v512_load, but result pack type will be 2x wider than memory type.

@code{.cpp}
short buf[8] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // type is int16
v_int32x16 r = v512_load_expand(buf); // r = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} - type is int32
@endcode
For 8-, 16-, 32-bit integer source types.

@note Check CV_SIMD512 preprocessor definition prior to use.
Use vx_load_expand version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, simd512_width / sizeof(typename V_TypeTraits<_Tp>::w_type)>
v512_load_expand(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, simd512_width / sizeof(w_type)> c;
    for (int i = 0; i < c.nlanes; i++)
    {
        c.s[i] = ptr[i];
    }
    return c;
}
#endif

/** @brief Load register contents from memory with quad expand

Same as cv::v_load_expand, but result type is 4 times wider than source.
@code{.cpp}
char buf[4] = {1, 2, 3, 4}; // type is int8
v_int32x4 r = v_load_expand_q(buf); // r = {1, 2, 3, 4} - type is int32
@endcode
For 8-bit integer source types.

@note Use vx_load_expand_q version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::q_type, simd128_width / sizeof(typename V_TypeTraits<_Tp>::q_type)>
v_load_expand_q(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, simd128_width / sizeof(q_type)> c;
    for( int i = 0; i < c.nlanes; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

#if CV_SIMD256
/** @brief Load register contents from memory with quad expand

Same as cv::v256_load_expand, but result type is 4 times wider than source.
@code{.cpp}
char buf[8] = {1, 2, 3, 4, 5, 6, 7, 8}; // type is int8
v_int32x8 r = v256_load_expand_q(buf); // r = {1, 2, 3, 4, 5, 6, 7, 8} - type is int32
@endcode
For 8-bit integer source types.

@note Check CV_SIMD256 preprocessor definition prior to use.
Use vx_load_expand_q version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::q_type, simd256_width / sizeof(typename V_TypeTraits<_Tp>::q_type)>
v256_load_expand_q(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, simd256_width / sizeof(q_type)> c;
    for (int i = 0; i < c.nlanes; i++)
    {
        c.s[i] = ptr[i];
    }
    return c;
}
#endif

#if CV_SIMD512
/** @brief Load register contents from memory with quad expand

Same as cv::v512_load_expand, but result type is 4 times wider than source.
@code{.cpp}
char buf[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // type is int8
v_int32x16 r = v512_load_expand_q(buf); // r = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} - type is int32
@endcode
For 8-bit integer source types.

@note Check CV_SIMD512 preprocessor definition prior to use.
Use vx_load_expand_q version to get maximum available register length result
*/
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::q_type, simd512_width / sizeof(typename V_TypeTraits<_Tp>::q_type)>
v512_load_expand_q(const _Tp* ptr)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, simd512_width / sizeof(q_type)> c;
    for (int i = 0; i < c.nlanes; i++)
    {
        c.s[i] = ptr[i];
    }
    return c;
}
#endif

/** @brief Load and deinterleave (2 channels)

Load data from memory deinterleave and store to 2 registers.
Scheme:
@code
{A1 B1 A2 B2 ...} ==> {A1 A2 ...}, {B1 B2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i2;
    for( i = i2 = 0; i < n; i++, i2 += 2 )
    {
        a.s[i] = ptr[i2];
        b.s[i] = ptr[i2+1];
    }
}

/** @brief Load and deinterleave (3 channels)

Load data from memory deinterleave and store to 3 registers.
Scheme:
@code
{A1 B1 C1 A2 B2 C2 ...} ==> {A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b, v_reg<_Tp, n>& c)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i3;
    for( i = i3 = 0; i < n; i++, i3 += 3 )
    {
        a.s[i] = ptr[i3];
        b.s[i] = ptr[i3+1];
        c.s[i] = ptr[i3+2];
    }
}

/** @brief Load and deinterleave (4 channels)

Load data from memory deinterleave and store to 4 registers.
Scheme:
@code
{A1 B1 C1 D1 A2 B2 C2 D2 ...} ==> {A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...}, {D1 D2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                v_reg<_Tp, n>& b, v_reg<_Tp, n>& c,
                                v_reg<_Tp, n>& d)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i4;
    for( i = i4 = 0; i < n; i++, i4 += 4 )
    {
        a.s[i] = ptr[i4];
        b.s[i] = ptr[i4+1];
        c.s[i] = ptr[i4+2];
        d.s[i] = ptr[i4+3];
    }
}

/** @brief Interleave and store (2 channels)

Interleave and store data from 2 registers to memory.
Scheme:
@code
{A1 A2 ...}, {B1 B2 ...} ==> {A1 B1 A2 B2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                               const v_reg<_Tp, n>& b,
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i2;
    for( i = i2 = 0; i < n; i++, i2 += 2 )
    {
        ptr[i2] = a.s[i];
        ptr[i2+1] = b.s[i];
    }
}

/** @brief Interleave and store (3 channels)

Interleave and store data from 3 registers to memory.
Scheme:
@code
{A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...} ==> {A1 B1 C1 A2 B2 C2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                                const v_reg<_Tp, n>& b, const v_reg<_Tp, n>& c,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i3;
    for( i = i3 = 0; i < n; i++, i3 += 3 )
    {
        ptr[i3] = a.s[i];
        ptr[i3+1] = b.s[i];
        ptr[i3+2] = c.s[i];
    }
}

/** @brief Interleave and store (4 channels)

Interleave and store data from 4 registers to memory.
Scheme:
@code
{A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...}, {D1 D2 ...} ==> {A1 B1 C1 D1 A2 B2 C2 D2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n> inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                                                            const v_reg<_Tp, n>& b, const v_reg<_Tp, n>& c,
                                                            const v_reg<_Tp, n>& d,
                                                            hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    int i, i4;
    for( i = i4 = 0; i < n; i++, i4 += 4 )
    {
        ptr[i4] = a.s[i];
        ptr[i4+1] = b.s[i];
        ptr[i4+2] = c.s[i];
        ptr[i4+3] = d.s[i];
    }
}

/** @brief Store data to memory

Store register contents to memory.
Scheme:
@code
  REG {A B C D} ==> MEM {A B C D}
@endcode
Pointer can be unaligned. */
template<typename _Tp, int n>
inline void v_store(_Tp* ptr, const v_reg<_Tp, n>& a)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
}

template<typename _Tp, int n>
inline void v_store(_Tp* ptr, const v_reg<_Tp, n>& a, hal::StoreMode /*mode*/)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    v_store(ptr, a);
}

/** @brief Store data to memory (lower half)

Store lower half of register contents to memory.
Scheme:
@code
  REG {A B C D} ==> MEM {A B}
@endcode */
template<typename _Tp, int n>
inline void v_store_low(_Tp* ptr, const v_reg<_Tp, n>& a)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i];
}

/** @brief Store data to memory (higher half)

Store higher half of register contents to memory.
Scheme:
@code
  REG {A B C D} ==> MEM {C D}
@endcode */
template<typename _Tp, int n>
inline void v_store_high(_Tp* ptr, const v_reg<_Tp, n>& a)
{
#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(_Tp)>(ptr));
#endif
    for( int i = 0; i < (n/2); i++ )
        ptr[i] = a.s[i+(n/2)];
}

/** @brief Store data to memory (aligned)

Store register contents to memory.
Scheme:
@code
  REG {A B C D} ==> MEM {A B C D}
@endcode
Pointer __should__ be aligned by 16-byte boundary. */
template<typename _Tp, int n>
inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, n>)>(ptr));
    v_store(ptr, a);
}

template<typename _Tp, int n>
inline void v_store_aligned_nocache(_Tp* ptr, const v_reg<_Tp, n>& a)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, n>)>(ptr));
    v_store(ptr, a);
}

template<typename _Tp, int n>
inline void v_store_aligned(_Tp* ptr, const v_reg<_Tp, n>& a, hal::StoreMode /*mode*/)
{
    CV_Assert(isAligned<sizeof(v_reg<_Tp, n>)>(ptr));
    v_store(ptr, a);
}

/** @brief Combine vector from first elements of two vectors

Scheme:
@code
  {A1 A2 A3 A4}
  {B1 B2 B3 B4}
---------------
  {A1 A2 B1 B2}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_combine_low(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = a.s[i];
        c.s[i+(n/2)] = b.s[i];
    }
    return c;
}

/** @brief Combine vector from last elements of two vectors

Scheme:
@code
  {A1 A2 A3 A4}
  {B1 B2 B3 B4}
---------------
  {A3 A4 B3 B4}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_combine_high(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < (n/2); i++ )
    {
        c.s[i] = a.s[i+(n/2)];
        c.s[i+(n/2)] = b.s[i+(n/2)];
    }
    return c;
}

/** @brief Combine two vectors from lower and higher parts of two other vectors

@code{.cpp}
low = cv::v_combine_low(a, b);
high = cv::v_combine_high(a, b);
@endcode */
template<typename _Tp, int n>
inline void v_recombine(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                        v_reg<_Tp, n>& low, v_reg<_Tp, n>& high)
{
    for( int i = 0; i < (n/2); i++ )
    {
        low.s[i] = a.s[i];
        low.s[i+(n/2)] = b.s[i];
        high.s[i] = a.s[i+(n/2)];
        high.s[i+(n/2)] = b.s[i+(n/2)];
    }
}

/** @brief Vector reverse order

Reverse the order of the vector
Scheme:
@code
  REG {A1 ... An} ==> REG {An ... A1}
@endcode
For all types. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_reverse(const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = a.s[n-i-1];
    return c;
}

/** @brief Vector extract

Scheme:
@code
  {A1 A2 A3 A4}
  {B1 B2 B3 B4}
========================
shift = 1  {A2 A3 A4 B1}
shift = 2  {A3 A4 B1 B2}
shift = 3  {A4 B1 B2 B3}
@endcode
Restriction: 0 <= shift < nlanes

Usage:
@code
v_int32x4 a, b, c;
c = v_extract<2>(a, b);
@endcode
For all types. */
template<int s, typename _Tp, int n>
inline v_reg<_Tp, n> v_extract(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    v_reg<_Tp, n> r;
    const int shift = n - s;
    int i = 0;
    for (; i < shift; ++i)
        r.s[i] = a.s[i+s];
    for (; i < n; ++i)
        r.s[i] = b.s[i-shift];
    return r;
}

/** @brief Vector extract

Scheme:
Return the s-th element of v.
Restriction: 0 <= s < nlanes

Usage:
@code
v_int32x4 a;
int r;
r = v_extract_n<2>(a);
@endcode
For all types. */
template<int s, typename _Tp, int n>
inline _Tp v_extract_n(const v_reg<_Tp, n>& v)
{
    CV_DbgAssert(s >= 0 && s < n);
    return v.s[s];
}

/** @brief Broadcast i-th element of vector

Scheme:
@code
{ v[0] v[1] v[2] ... v[SZ] } => { v[i], v[i], v[i] ... v[i] }
@endcode
Restriction: 0 <= i < nlanes
Supported types: 32-bit integers and floats (s32/u32/f32)
 */
template<int i, typename _Tp, int n>
inline v_reg<_Tp, n> v_broadcast_element(const v_reg<_Tp, n>& a)
{
    CV_DbgAssert(i >= 0 && i < n);
    return v_reg<_Tp, n>::all(a.s[i]);
}

/** @brief Round elements

Rounds each value. Input type is float vector ==> output type is int vector.
@note Only for floating point types.
*/
template<int n> inline v_reg<int, n> v_round(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvRound(a.s[i]);
    return c;
}

/** @overload */
template<int n> inline v_reg<int, n*2> v_round(const v_reg<double, n>& a, const v_reg<double, n>& b)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvRound(a.s[i]);
        c.s[i+n] = cvRound(b.s[i]);
    }
    return c;
}

/** @brief Floor elements

Floor each value. Input type is float vector ==> output type is int vector.
@note Only for floating point types.
*/
template<int n> inline v_reg<int, n> v_floor(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvFloor(a.s[i]);
    return c;
}

/** @brief Ceil elements

Ceil each value. Input type is float vector ==> output type is int vector.
@note Only for floating point types.
*/
template<int n> inline v_reg<int, n> v_ceil(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvCeil(a.s[i]);
    return c;
}

/** @brief Truncate elements

Truncate each value. Input type is float vector ==> output type is int vector.
@note Only for floating point types.
*/
template<int n> inline v_reg<int, n> v_trunc(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (int)(a.s[i]);
    return c;
}

/** @overload */
template<int n> inline v_reg<int, n*2> v_round(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvRound(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

/** @overload */
template<int n> inline v_reg<int, n*2> v_floor(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvFloor(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

/** @overload */
template<int n> inline v_reg<int, n*2> v_ceil(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvCeil(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

/** @overload */
template<int n> inline v_reg<int, n*2> v_trunc(const v_reg<double, n>& a)
{
    v_reg<int, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = (int)(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

/** @brief Convert to float

Supported input type is cv::v_int32. */
template<int n> inline v_reg<float, n> v_cvt_f32(const v_reg<int, n>& a)
{
    v_reg<float, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (float)a.s[i];
    return c;
}

/** @brief Convert lower half to float

Supported input type is cv::v_float64. */
template<int n> inline v_reg<float, n*2> v_cvt_f32(const v_reg<double, n>& a)
{
    v_reg<float, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = (float)a.s[i];
        c.s[i+n] = 0;
    }
    return c;
}

/** @brief Convert to float

Supported input type is cv::v_float64. */
template<int n> inline v_reg<float, n*2> v_cvt_f32(const v_reg<double, n>& a, const v_reg<double, n>& b)
{
    v_reg<float, n*2> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = (float)a.s[i];
        c.s[i+n] = (float)b.s[i];
    }
    return c;
}

/** @brief Convert lower half to double

Supported input type is cv::v_int32. */
template<int n> CV_INLINE v_reg<double, n/2> v_cvt_f64(const v_reg<int, n>& a)
{
    v_reg<double, (n/2)> c;
    for( int i = 0; i < (n/2); i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

/** @brief Convert to double high part of vector

Supported input type is cv::v_int32. */
template<int n> CV_INLINE v_reg<double, (n/2)> v_cvt_f64_high(const v_reg<int, n>& a)
{
    v_reg<double, (n/2)> c;
    for( int i = 0; i < (n/2); i++ )
        c.s[i] = (double)a.s[i + (n/2)];
    return c;
}

/** @brief Convert lower half to double

Supported input type is cv::v_float32. */
template<int n> CV_INLINE v_reg<double, (n/2)> v_cvt_f64(const v_reg<float, n>& a)
{
    v_reg<double, (n/2)> c;
    for( int i = 0; i < (n/2); i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

/** @brief Convert to double high part of vector

Supported input type is cv::v_float32. */
template<int n> CV_INLINE v_reg<double, (n/2)> v_cvt_f64_high(const v_reg<float, n>& a)
{
    v_reg<double, (n/2)> c;
    for( int i = 0; i < (n/2); i++ )
        c.s[i] = (double)a.s[i + (n/2)];
    return c;
}

/** @brief Convert to double

Supported input type is cv::v_int64. */
template<int n> CV_INLINE v_reg<double, n> v_cvt_f64(const v_reg<int64, n>& a)
{
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}


template<typename _Tp> inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_lut(const _Tp* tab, const int* idx)
{
    v_reg<_Tp, simd128_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes; i++)
        c.s[i] = tab[idx[i]];
    return c;
}
template<typename _Tp> inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_lut_pairs(const _Tp* tab, const int* idx)
{
    v_reg<_Tp, simd128_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes; i++)
        c.s[i] = tab[idx[i / 2] + i % 2];
    return c;
}
template<typename _Tp> inline v_reg<_Tp, simd128_width / sizeof(_Tp)> v_lut_quads(const _Tp* tab, const int* idx)
{
    v_reg<_Tp, simd128_width / sizeof(_Tp)> c;
    for (int i = 0; i < c.nlanes; i++)
        c.s[i] = tab[idx[i / 4] + i % 4];
    return c;
}

template<int n> inline v_reg<int, n> v_lut(const int* tab, const v_reg<int, n>& idx)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = tab[idx.s[i]];
    return c;
}

template<int n> inline v_reg<unsigned, n> v_lut(const unsigned* tab, const v_reg<int, n>& idx)
{
    v_reg<int, n> c;
    for (int i = 0; i < n; i++)
        c.s[i] = tab[idx.s[i]];
    return c;
}

template<int n> inline v_reg<float, n> v_lut(const float* tab, const v_reg<int, n>& idx)
{
    v_reg<float, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = tab[idx.s[i]];
    return c;
}

template<int n> inline v_reg<double, n/2> v_lut(const double* tab, const v_reg<int, n>& idx)
{
    v_reg<double, n/2> c;
    for( int i = 0; i < n/2; i++ )
        c.s[i] = tab[idx.s[i]];
    return c;
}


template<int n> inline void v_lut_deinterleave(const float* tab, const v_reg<int, n>& idx,
                                               v_reg<float, n>& x, v_reg<float, n>& y)
{
    for( int i = 0; i < n; i++ )
    {
        int j = idx.s[i];
        x.s[i] = tab[j];
        y.s[i] = tab[j+1];
    }
}

template<int n> inline void v_lut_deinterleave(const double* tab, const v_reg<int, n*2>& idx,
                                               v_reg<double, n>& x, v_reg<double, n>& y)
{
    for( int i = 0; i < n; i++ )
    {
        int j = idx.s[i];
        x.s[i] = tab[j];
        y.s[i] = tab[j+1];
    }
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_interleave_pairs(const v_reg<_Tp, n>& vec)
{
    v_reg<_Tp, n> c;
    for (int i = 0; i < n/4; i++)
    {
        c.s[4*i  ] = vec.s[4*i  ];
        c.s[4*i+1] = vec.s[4*i+2];
        c.s[4*i+2] = vec.s[4*i+1];
        c.s[4*i+3] = vec.s[4*i+3];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_interleave_quads(const v_reg<_Tp, n>& vec)
{
    v_reg<_Tp, n> c;
    for (int i = 0; i < n/8; i++)
    {
        c.s[8*i  ] = vec.s[8*i  ];
        c.s[8*i+1] = vec.s[8*i+4];
        c.s[8*i+2] = vec.s[8*i+1];
        c.s[8*i+3] = vec.s[8*i+5];
        c.s[8*i+4] = vec.s[8*i+2];
        c.s[8*i+5] = vec.s[8*i+6];
        c.s[8*i+6] = vec.s[8*i+3];
        c.s[8*i+7] = vec.s[8*i+7];
    }
    return c;
}

template<typename _Tp, int n> inline v_reg<_Tp, n> v_pack_triplets(const v_reg<_Tp, n>& vec)
{
    v_reg<_Tp, n> c;
    for (int i = 0; i < n/4; i++)
    {
        c.s[3*i  ] = vec.s[4*i  ];
        c.s[3*i+1] = vec.s[4*i+1];
        c.s[3*i+2] = vec.s[4*i+2];
    }
    return c;
}

/** @brief Transpose 4x4 matrix

Scheme:
@code
a0  {A1 A2 A3 A4}
a1  {B1 B2 B3 B4}
a2  {C1 C2 C3 C4}
a3  {D1 D2 D3 D4}
===============
b0  {A1 B1 C1 D1}
b1  {A2 B2 C2 D2}
b2  {A3 B3 C3 D3}
b3  {A4 B4 C4 D4}
@endcode
*/
template<typename _Tp, int n>
inline void v_transpose4x4( v_reg<_Tp, n>& a0, const v_reg<_Tp, n>& a1,
                            const v_reg<_Tp, n>& a2, const v_reg<_Tp, n>& a3,
                            v_reg<_Tp, n>& b0, v_reg<_Tp, n>& b1,
                            v_reg<_Tp, n>& b2, v_reg<_Tp, n>& b3 )
{
    for (int i = 0; i < n / 4; i++)
    {
        b0.s[0 + i*4] = a0.s[0 + i*4]; b0.s[1 + i*4] = a1.s[0 + i*4];
        b0.s[2 + i*4] = a2.s[0 + i*4]; b0.s[3 + i*4] = a3.s[0 + i*4];
        b1.s[0 + i*4] = a0.s[1 + i*4]; b1.s[1 + i*4] = a1.s[1 + i*4];
        b1.s[2 + i*4] = a2.s[1 + i*4]; b1.s[3 + i*4] = a3.s[1 + i*4];
        b2.s[0 + i*4] = a0.s[2 + i*4]; b2.s[1 + i*4] = a1.s[2 + i*4];
        b2.s[2 + i*4] = a2.s[2 + i*4]; b2.s[3 + i*4] = a3.s[2 + i*4];
        b3.s[0 + i*4] = a0.s[3 + i*4]; b3.s[1 + i*4] = a1.s[3 + i*4];
        b3.s[2 + i*4] = a2.s[3 + i*4]; b3.s[3 + i*4] = a3.s[3 + i*4];
    }
}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_INIT_ZERO(_Tpvec, prefix, suffix) \
inline _Tpvec prefix##_setzero_##suffix() { return _Tpvec::zero(); }

//! @name Init with zero
//! @{
//! @brief Create new vector with zero elements
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint8x16, v, u8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int8x16, v, s8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint16x8, v, u16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int16x8, v, s16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint32x4, v, u32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int32x4, v, s32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float32x4, v, f32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float64x2, v, f64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint64x2, v, u64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int64x2, v, s64)

#if CV_SIMD256
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint8x32, v256, u8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int8x32, v256, s8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint16x16, v256, u16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int16x16, v256, s16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint32x8, v256, u32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int32x8, v256, s32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float32x8, v256, f32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float64x4, v256, f64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint64x4, v256, u64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int64x4, v256, s64)
#endif

#if CV_SIMD512
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint8x64, v512, u8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int8x64, v512, s8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint16x32, v512, u16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int16x32, v512, s16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint32x16, v512, u32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int32x16, v512, s32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float32x16, v512, f32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float64x8, v512, f64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint64x8, v512, u64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int64x8, v512, s64)
#endif
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_INIT_VAL(_Tpvec, _Tp, prefix, suffix) \
inline _Tpvec prefix##_setall_##suffix(_Tp val) { return _Tpvec::all(val); }

//! @name Init with value
//! @{
//! @brief Create new vector with elements set to a specific value
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint8x16, uchar, v, u8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int8x16, schar, v, s8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint16x8, ushort, v, u16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int16x8, short, v, s16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint32x4, unsigned, v, u32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int32x4, int, v, s32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float32x4, float, v, f32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float64x2, double, v, f64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint64x2, uint64, v, u64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int64x2, int64, v, s64)

#if CV_SIMD256
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint8x32, uchar, v256, u8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int8x32, schar, v256, s8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint16x16, ushort, v256, u16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int16x16, short, v256, s16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint32x8, unsigned, v256, u32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int32x8, int, v256, s32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float32x8, float, v256, f32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float64x4, double, v256, f64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint64x4, uint64, v256, u64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int64x4, int64, v256, s64)
#endif

#if CV_SIMD512
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint8x64, uchar, v512, u8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int8x64, schar, v512, s8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint16x32, ushort, v512, u16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int16x32, short, v512, s16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint32x16, unsigned, v512, u32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int32x16, int, v512, s32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float32x16, float, v512, f32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float64x8, double, v512, f64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint64x8, uint64, v512, u64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int64x8, int64, v512, s64)
#endif
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_REINTERPRET(_Tp, suffix) \
template<typename _Tp0, int n0> inline v_reg<_Tp, n0*sizeof(_Tp0)/sizeof(_Tp)> \
    v_reinterpret_as_##suffix(const v_reg<_Tp0, n0>& a) \
{ return a.template reinterpret_as<_Tp, n0*sizeof(_Tp0)/sizeof(_Tp)>(); }

//! @name Reinterpret
//! @{
//! @brief Convert vector to different type without modifying underlying data.
OPENCV_HAL_IMPL_C_REINTERPRET(uchar, u8)
OPENCV_HAL_IMPL_C_REINTERPRET(schar, s8)
OPENCV_HAL_IMPL_C_REINTERPRET(ushort, u16)
OPENCV_HAL_IMPL_C_REINTERPRET(short, s16)
OPENCV_HAL_IMPL_C_REINTERPRET(unsigned, u32)
OPENCV_HAL_IMPL_C_REINTERPRET(int, s32)
OPENCV_HAL_IMPL_C_REINTERPRET(float, f32)
OPENCV_HAL_IMPL_C_REINTERPRET(double, f64)
OPENCV_HAL_IMPL_C_REINTERPRET(uint64, u64)
OPENCV_HAL_IMPL_C_REINTERPRET(int64, s64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_SHIFTL(_Tp) \
template<int shift, int n> inline v_reg<_Tp, n> v_shl(const v_reg<_Tp, n>& a) \
{ return v_shl(a, shift); }

//! @name Left shift
//! @{
//! @brief Shift left
OPENCV_HAL_IMPL_C_SHIFTL(ushort)
OPENCV_HAL_IMPL_C_SHIFTL(short)
OPENCV_HAL_IMPL_C_SHIFTL(unsigned)
OPENCV_HAL_IMPL_C_SHIFTL(int)
OPENCV_HAL_IMPL_C_SHIFTL(uint64)
OPENCV_HAL_IMPL_C_SHIFTL(int64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_SHIFTR(_Tp) \
template<int shift, int n> inline v_reg<_Tp, n> v_shr(const v_reg<_Tp, n>& a) \
{ return v_shr(a, shift); }

//! @name Right shift
//! @{
//! @brief Shift right
OPENCV_HAL_IMPL_C_SHIFTR(ushort)
OPENCV_HAL_IMPL_C_SHIFTR(short)
OPENCV_HAL_IMPL_C_SHIFTR(unsigned)
OPENCV_HAL_IMPL_C_SHIFTR(int)
OPENCV_HAL_IMPL_C_SHIFTR(uint64)
OPENCV_HAL_IMPL_C_SHIFTR(int64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_RSHIFTR(_Tp) \
template<int shift, int n> inline v_reg<_Tp, n> v_rshr(const v_reg<_Tp, n>& a) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = (_Tp)((a.s[i] + ((_Tp)1 << (shift - 1))) >> shift); \
    return c; \
}

//! @name Rounding shift
//! @{
//! @brief Rounding shift right
OPENCV_HAL_IMPL_C_RSHIFTR(ushort)
OPENCV_HAL_IMPL_C_RSHIFTR(short)
OPENCV_HAL_IMPL_C_RSHIFTR(unsigned)
OPENCV_HAL_IMPL_C_RSHIFTR(int)
OPENCV_HAL_IMPL_C_RSHIFTR(uint64)
OPENCV_HAL_IMPL_C_RSHIFTR(int64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_PACK(_Tp, _Tpn, pack_suffix, cast) \
template<int n> inline v_reg<_Tpn, 2*n> v_##pack_suffix(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tpn, 2*n> c; \
    for( int i = 0; i < n; i++ ) \
    { \
        c.s[i] = cast<_Tpn>(a.s[i]); \
        c.s[i+n] = cast<_Tpn>(b.s[i]); \
    } \
    return c; \
}

//! @name Pack
//! @{
//! @brief Pack values from two vectors to one
//!
//! Return vector type have twice more elements than input vector types. Variant with _u_ suffix also
//! converts to corresponding unsigned type.
//!
//! - pack: for 16-, 32- and 64-bit integer input types
//! - pack_u: for 16- and 32-bit signed integer input types
//!
//! @note All variants except 64-bit use saturation.
OPENCV_HAL_IMPL_C_PACK(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(int, ushort, pack_u, saturate_cast)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_RSHR_PACK(_Tp, _Tpn, pack_suffix, cast) \
template<int shift, int n> inline v_reg<_Tpn, 2*n> v_rshr_##pack_suffix(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tpn, 2*n> c; \
    for( int i = 0; i < n; i++ ) \
    { \
        c.s[i] = cast<_Tpn>((a.s[i] + ((_Tp)1 << (shift - 1))) >> shift); \
        c.s[i+n] = cast<_Tpn>((b.s[i] + ((_Tp)1 << (shift - 1))) >> shift); \
    } \
    return c; \
}

//! @name Pack with rounding shift
//! @{
//! @brief Pack values from two vectors to one with rounding shift
//!
//! Values from the input vectors will be shifted right by _n_ bits with rounding, converted to narrower
//! type and returned in the result vector. Variant with _u_ suffix converts to unsigned type.
//!
//! - pack: for 16-, 32- and 64-bit integer input types
//! - pack_u: for 16- and 32-bit signed integer input types
//!
//! @note All variants except 64-bit use saturation.
OPENCV_HAL_IMPL_C_RSHR_PACK(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(int, ushort, pack_u, saturate_cast)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_PACK_STORE(_Tp, _Tpn, pack_suffix, cast) \
template<int n> inline void v_##pack_suffix##_store(_Tpn* ptr, const v_reg<_Tp, n>& a) \
{ \
    for( int i = 0; i < n; i++ ) \
        ptr[i] = cast<_Tpn>(a.s[i]); \
}

//! @name Pack and store
//! @{
//! @brief Store values from the input vector into memory with pack
//!
//! Values will be stored into memory with conversion to narrower type.
//! Variant with _u_ suffix converts to corresponding unsigned type.
//!
//! - pack: for 16-, 32- and 64-bit integer input types
//! - pack_u: for 16- and 32-bit signed integer input types
//!
//! @note All variants except 64-bit use saturation.
OPENCV_HAL_IMPL_C_PACK_STORE(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(int, ushort, pack_u, saturate_cast)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(_Tp, _Tpn, pack_suffix, cast) \
template<int shift, int n> inline void v_rshr_##pack_suffix##_store(_Tpn* ptr, const v_reg<_Tp, n>& a) \
{ \
    for( int i = 0; i < n; i++ ) \
        ptr[i] = cast<_Tpn>((a.s[i] + ((_Tp)1 << (shift - 1))) >> shift); \
}

//! @name Pack and store with rounding shift
//! @{
//! @brief Store values from the input vector into memory with pack
//!
//! Values will be shifted _n_ bits right with rounding, converted to narrower type and stored into
//! memory. Variant with _u_ suffix converts to unsigned type.
//!
//! - pack: for 16-, 32- and 64-bit integer input types
//! - pack_u: for 16- and 32-bit signed integer input types
//!
//! @note All variants except 64-bit use saturation.
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(int, ushort, pack_u, saturate_cast)
//! @}

//! @cond IGNORED
template<typename _Tpm, typename _Tp, int n>
inline void _pack_b(_Tpm* mptr, const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    for (int i = 0; i < n; ++i)
    {
        mptr[i] = (_Tpm)a.s[i];
        mptr[i + n] = (_Tpm)b.s[i];
    }
}
//! @endcond

//! @name Pack boolean values
//! @{
//! @brief Pack boolean values from multiple vectors to one unsigned 8-bit integer vector
//!
//! @note Must provide valid boolean values to guarantee same result for all architectures.

/** @brief
//! For 16-bit boolean values

Scheme:
@code
a  {0xFFFF 0 0 0xFFFF 0 0xFFFF 0xFFFF 0}
b  {0xFFFF 0 0xFFFF 0 0 0xFFFF 0 0xFFFF}
===============
{
   0xFF 0 0 0xFF 0 0xFF 0xFF 0
   0xFF 0 0xFF 0 0 0xFF 0 0xFF
}
@endcode */

template<int n> inline v_reg<uchar, 2*n> v_pack_b(const v_reg<ushort, n>& a, const v_reg<ushort, n>& b)
{
    v_reg<uchar, 2*n> mask;
    _pack_b(mask.s, a, b);
    return mask;
}

/** @overload
For 32-bit boolean values

Scheme:
@code
a  {0xFFFF.. 0 0 0xFFFF..}
b  {0 0xFFFF.. 0xFFFF.. 0}
c  {0xFFFF.. 0 0xFFFF.. 0}
d  {0 0xFFFF.. 0 0xFFFF..}
===============
{
   0xFF 0 0 0xFF 0 0xFF 0xFF 0
   0xFF 0 0xFF 0 0 0xFF 0 0xFF
}
@endcode */

template<int n> inline v_reg<uchar, 4*n> v_pack_b(const v_reg<unsigned, n>& a, const v_reg<unsigned, n>& b,
                                                  const v_reg<unsigned, n>& c, const v_reg<unsigned, n>& d)
{
    v_reg<uchar, 4*n> mask;
    _pack_b(mask.s, a, b);
    _pack_b(mask.s + 2*n, c, d);
    return mask;
}

/** @overload
For 64-bit boolean values

Scheme:
@code
a  {0xFFFF.. 0}
b  {0 0xFFFF..}
c  {0xFFFF.. 0}
d  {0 0xFFFF..}

e  {0xFFFF.. 0}
f  {0xFFFF.. 0}
g  {0 0xFFFF..}
h  {0 0xFFFF..}
===============
{
   0xFF 0 0 0xFF 0xFF 0 0 0xFF
   0xFF 0 0xFF 0 0 0xFF 0 0xFF
}
@endcode */
template<int n> inline v_reg<uchar, 8*n> v_pack_b(const v_reg<uint64, n>& a, const v_reg<uint64, n>& b,
                                                  const v_reg<uint64, n>& c, const v_reg<uint64, n>& d,
                                                  const v_reg<uint64, n>& e, const v_reg<uint64, n>& f,
                                                  const v_reg<uint64, n>& g, const v_reg<uint64, n>& h)
{
    v_reg<uchar, 8*n> mask;
    _pack_b(mask.s, a, b);
    _pack_b(mask.s + 2*n, c, d);
    _pack_b(mask.s + 4*n, e, f);
    _pack_b(mask.s + 6*n, g, h);
    return mask;
}
//! @}

/** @brief Matrix multiplication

Scheme:
@code
{A0 A1 A2 A3}   |V0|
{B0 B1 B2 B3}   |V1|
{C0 C1 C2 C3}   |V2|
{D0 D1 D2 D3} x |V3|
====================
{R0 R1 R2 R3}, where:
R0 = A0V0 + B0V1 + C0V2 + D0V3,
R1 = A1V0 + B1V1 + C1V2 + D1V3
...
@endcode
*/
template<int n>
inline v_reg<float, n> v_matmul(const v_reg<float, n>& v,
                                const v_reg<float, n>& a, const v_reg<float, n>& b,
                                const v_reg<float, n>& c, const v_reg<float, n>& d)
{
    v_reg<float, n> res;
    for (int i = 0; i < n / 4; i++)
    {
        res.s[0 + i*4] = v.s[0 + i*4] * a.s[0 + i*4] + v.s[1 + i*4] * b.s[0 + i*4] + v.s[2 + i*4] * c.s[0 + i*4] + v.s[3 + i*4] * d.s[0 + i*4];
        res.s[1 + i*4] = v.s[0 + i*4] * a.s[1 + i*4] + v.s[1 + i*4] * b.s[1 + i*4] + v.s[2 + i*4] * c.s[1 + i*4] + v.s[3 + i*4] * d.s[1 + i*4];
        res.s[2 + i*4] = v.s[0 + i*4] * a.s[2 + i*4] + v.s[1 + i*4] * b.s[2 + i*4] + v.s[2 + i*4] * c.s[2 + i*4] + v.s[3 + i*4] * d.s[2 + i*4];
        res.s[3 + i*4] = v.s[0 + i*4] * a.s[3 + i*4] + v.s[1 + i*4] * b.s[3 + i*4] + v.s[2 + i*4] * c.s[3 + i*4] + v.s[3 + i*4] * d.s[3 + i*4];
    }
    return res;
}

/** @brief Matrix multiplication and add

Scheme:
@code
{A0 A1 A2 A3}   |V0|   |D0|
{B0 B1 B2 B3}   |V1|   |D1|
{C0 C1 C2 C3} x |V2| + |D2|
====================   |D3|
{R0 R1 R2 R3}, where:
R0 = A0V0 + B0V1 + C0V2 + D0,
R1 = A1V0 + B1V1 + C1V2 + D1
...
@endcode
*/
template<int n>
inline v_reg<float, n> v_matmuladd(const v_reg<float, n>& v,
                                   const v_reg<float, n>& a, const v_reg<float, n>& b,
                                   const v_reg<float, n>& c, const v_reg<float, n>& d)
{
    v_reg<float, n> res;
    for (int i = 0; i < n / 4; i++)
    {
        res.s[0 + i * 4] = v.s[0 + i * 4] * a.s[0 + i * 4] + v.s[1 + i * 4] * b.s[0 + i * 4] + v.s[2 + i * 4] * c.s[0 + i * 4] + d.s[0 + i * 4];
        res.s[1 + i * 4] = v.s[0 + i * 4] * a.s[1 + i * 4] + v.s[1 + i * 4] * b.s[1 + i * 4] + v.s[2 + i * 4] * c.s[1 + i * 4] + d.s[1 + i * 4];
        res.s[2 + i * 4] = v.s[0 + i * 4] * a.s[2 + i * 4] + v.s[1 + i * 4] * b.s[2 + i * 4] + v.s[2 + i * 4] * c.s[2 + i * 4] + d.s[2 + i * 4];
        res.s[3 + i * 4] = v.s[0 + i * 4] * a.s[3 + i * 4] + v.s[1 + i * 4] * b.s[3 + i * 4] + v.s[2 + i * 4] * c.s[3 + i * 4] + d.s[3 + i * 4];
    }
    return res;
}


template<int n> inline v_reg<double, n/2> v_dotprod_expand(const v_reg<int, n>& a, const v_reg<int, n>& b)
{ return v_fma(v_cvt_f64(a), v_cvt_f64(b), v_mul(v_cvt_f64_high(a), v_cvt_f64_high(b))); }
template<int n> inline v_reg<double, n/2> v_dotprod_expand(const v_reg<int, n>& a, const v_reg<int, n>& b,
                                                           const v_reg<double, n/2>& c)
{ return v_fma(v_cvt_f64(a), v_cvt_f64(b), v_fma(v_cvt_f64_high(a), v_cvt_f64_high(b), c)); }

template<int n> inline v_reg<double, n/2> v_dotprod_expand_fast(const v_reg<int, n>& a, const v_reg<int, n>& b)
{ return v_dotprod_expand(a, b); }
template<int n> inline v_reg<double, n/2> v_dotprod_expand_fast(const v_reg<int, n>& a, const v_reg<int, n>& b,
                                                                const v_reg<double, n/2>& c)
{ return v_dotprod_expand(a, b, c); }

////// FP16 support ///////

inline v_reg<float, simd128_width / sizeof(float)>
v_load_expand(const hfloat* ptr)
{
    v_reg<float, simd128_width / sizeof(float)> v;
    for( int i = 0; i < v.nlanes; i++ )
    {
        v.s[i] = ptr[i];
    }
    return v;
}
#if CV_SIMD256
inline v_reg<float, simd256_width / sizeof(float)>
v256_load_expand(const hfloat* ptr)
{
    v_reg<float, simd256_width / sizeof(float)> v;
    for (int i = 0; i < v.nlanes; i++)
    {
        v.s[i] = ptr[i];
    }
    return v;
}
#endif
#if CV_SIMD512
inline v_reg<float, simd512_width / sizeof(float)>
v512_load_expand(const hfloat* ptr)
{
    v_reg<float, simd512_width / sizeof(float)> v;
    for (int i = 0; i < v.nlanes; i++)
    {
        v.s[i] = ptr[i];
    }
    return v;
}
#endif

template<int n> inline void
v_pack_store(hfloat* ptr, const v_reg<float, n>& v)
{
    for( int i = 0; i < v.nlanes; i++ )
    {
        ptr[i] = hfloat(v.s[i]);
    }
}

inline void v_cleanup() {}
#if CV_SIMD256
inline void v256_cleanup() {}
#endif
#if CV_SIMD512
inline void v512_cleanup() {}
#endif

//! @}

#ifndef CV_DOXYGEN
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
#endif
}

#if !defined(CV_DOXYGEN)
#undef CV_SIMD256
#undef CV_SIMD512
#endif

#endif
