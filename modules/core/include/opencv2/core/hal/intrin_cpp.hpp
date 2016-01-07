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

#ifndef __OPENCV_HAL_INTRIN_CPP_HPP__
#define __OPENCV_HAL_INTRIN_CPP_HPP__

#include <limits>
#include <cstring>
#include <algorithm>
#include "opencv2/core/saturate.hpp"

namespace cv
{

/** @addtogroup core_hal_intrin

"Universal intrinsics" is a types and functions set intended to simplify vectorization of code on
different platforms. Currently there are two supported SIMD extensions: __SSE/SSE2__ on x86
architectures and __NEON__ on ARM architectures, both allow working with 128 bit registers
containing packed values of different types. In case when there is no SIMD extension available
during compilation, fallback C++ implementation of intrinsics will be chosen and code will work as
expected although it could be slower.

### Types

There are several types representing 128-bit register as a vector of packed values, each type is
implemented as a structure based on a one SIMD register.

- cv::v_uint8x16 and cv::v_int8x16: sixteen 8-bit integer values (unsigned/signed) - char
- cv::v_uint16x8 and cv::v_int16x8: eight 16-bit integer values (unsigned/signed) - short
- cv::v_uint32x4 and cv::v_int32x4: four 32-bit integer values (unsgined/signed) - int
- cv::v_uint64x2 and cv::v_int64x2: two 64-bit integer values (unsigned/signed) - int64
- cv::v_float32x4: four 32-bit floating point values (signed) - float
- cv::v_float64x2: two 64-bit floating point valies (signed) - double

@note
cv::v_float64x2 is not implemented in NEON variant, if you want to use this type, don't forget to
check the CV_SIMD128_64F preprocessor definition:
@code
#if CV_SIMD128_64F
//...
#endif
@endcode

### Load and store operations

These operations allow to set contents of the register explicitly or by loading it from some memory
block and to save contents of the register to memory block.

- Constructors:
@ref v_reg::v_reg(const _Tp *ptr) "from memory",
@ref v_reg::v_reg(_Tp s0, _Tp s1) "from two values", ...
- Other create methods:
@ref v_setall_s8, @ref v_setall_u8, ...,
@ref v_setzero_u8, @ref v_setzero_s8, ...
- Memory operations:
@ref v_load, @ref v_load_aligned, @ref v_load_halves,
@ref v_store, @ref v_store_aligned,
@ref v_store_high, @ref v_store_low

### Value reordering

These operations allow to reorder or recombine elements in one or multiple vectors.

- Interleave, deinterleave (3 and 4 channels): @ref v_load_deinterleave, @ref v_store_interleave
- Expand: @ref v_load_expand, @ref v_load_expand_q, @ref v_expand
- Pack: @ref v_pack, @ref v_pack_u, @ref v_rshr_pack, @ref v_rshr_pack_u,
@ref v_pack_store, @ref v_pack_u_store, @ref v_rshr_pack_store, @ref v_rshr_pack_u_store
- Recombine: @ref v_zip, @ref v_recombine, @ref v_combine_low, @ref v_combine_high
- Extract: @ref v_extract


### Arithmetic, bitwise and comparison operations

Element-wise binary and unary operations.

- Arithmetics:
@ref operator+(const v_reg &a, const v_reg &b) "+",
@ref operator-(const v_reg &a, const v_reg &b) "-",
@ref operator*(const v_reg &a, const v_reg &b) "*",
@ref operator/(const v_reg &a, const v_reg &b) "/",
@ref v_mul_expand

- Non-saturating arithmetics: @ref v_add_wrap, @ref v_sub_wrap

- Bitwise shifts:
@ref operator<<(const v_reg &a, int s) "<<",
@ref operator>>(const v_reg &a, int s) ">>",
@ref v_shl, @ref v_shr

- Bitwise logic:
@ref operator&(const v_reg &a, const v_reg &b) "&",
@ref operator|(const v_reg &a, const v_reg &b) "|",
@ref operator^(const v_reg &a, const v_reg &b) "^",
@ref operator~(const v_reg &a) "~"

- Comparison:
@ref operator>(const v_reg &a, const v_reg &b) ">",
@ref operator>=(const v_reg &a, const v_reg &b) ">=",
@ref operator<(const v_reg &a, const v_reg &b) "<",
@ref operator<=(const v_reg &a, const v_reg &b) "<=",
@ref operator==(const v_reg &a, const v_reg &b) "==",
@ref operator!=(const v_reg &a, const v_reg &b) "!="

- min/max: @ref v_min, @ref v_max

### Reduce and mask

Most of these operations return only one value.

- Reduce: @ref v_reduce_min, @ref v_reduce_max, @ref v_reduce_sum
- Mask: @ref v_signmask, @ref v_check_all, @ref v_check_any, @ref v_select

### Other math

- Some frequent operations: @ref v_sqrt, @ref v_invsqrt, @ref v_magnitude, @ref v_sqr_magnitude
- Absolute values: @ref v_abs, @ref v_absdiff

### Conversions

Different type conversions and casts:

- Rounding: @ref v_round, @ref v_floor, @ref v_ceil, @ref v_trunc,
- To float: @ref v_cvt_f32, @ref v_cvt_f64
- Reinterpret: @ref v_reinterpret_as_u8, @ref v_reinterpret_as_s8, ...

### Matrix operations

In these operations vectors represent matrix rows/columns: @ref v_dotprod, @ref v_matmul, @ref v_transpose4x4

### Usability

Most operations are implemented only for some subset of the available types, following matrices
shows the applicability of different operations to the types.

Regular integers:

| Operations\\Types | uint 8x16 | int 8x16 | uint 16x8 | int 16x8 | uint 32x4 | int 32x4 |
|-------------------|:-:|:-:|:-:|:-:|:-:|:-:|
|load, store        | x | x | x | x | x | x |
|interleave         | x | x | x | x | x | x |
|expand             | x | x | x | x | x | x |
|expand_q           | x | x |   |   |   |   |
|add, sub           | x | x | x | x | x | x |
|add_wrap, sub_wrap | x | x | x | x |   |   |
|mul                |   |   | x | x | x | x |
|mul_expand         |   |   | x | x | x |   |
|compare            | x | x | x | x | x | x |
|shift              |   |   | x | x | x | x |
|dotprod            |   |   |   | x |   |   |
|logical            | x | x | x | x | x | x |
|min, max           | x | x | x | x | x | x |
|absdiff            | x | x | x | x | x | x |
|reduce             |   |   |   |   | x | x |
|mask               | x | x | x | x | x | x |
|pack               | x | x | x | x | x | x |
|pack_u             | x |   | x |   |   |   |
|unpack             | x | x | x | x | x | x |
|extract            | x | x | x | x | x | x |
|cvt_flt32          |   |   |   |   |   | x |
|cvt_flt64          |   |   |   |   |   | x |
|transpose4x4       |   |   |   |   | x | x |

Big integers:

| Operations\\Types | uint 64x2 | int 64x2 |
|-------------------|:-:|:-:|
|load, store        | x | x |
|add, sub           | x | x |
|shift              | x | x |
|logical            | x | x |
|extract            | x | x |

Floating point:

| Operations\\Types | float 32x4 | float 64x2 |
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


 @{ */

template<typename _Tp, int n> struct v_reg
{
//! @cond IGNORED
    typedef _Tp lane_type;
    typedef v_reg<typename V_TypeTraits<_Tp>::int_type, n> int_vec;
    typedef v_reg<typename V_TypeTraits<_Tp>::abs_type, n> abs_vec;
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

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_BIN_OP(bin_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> \
    operator bin_op (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return c; \
} \
template<typename _Tp, int n> inline v_reg<_Tp, n>& \
    operator bin_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = saturate_cast<_Tp>(a.s[i] bin_op b.s[i]); \
    return a; \
}

/** @brief Add values

For all types. */
OPENCV_HAL_IMPL_BIN_OP(+)

/** @brief Subtract values

For all types. */
OPENCV_HAL_IMPL_BIN_OP(-)

/** @brief Multiply values

For 16- and 32-bit integer types and floating types. */
OPENCV_HAL_IMPL_BIN_OP(*)

/** @brief Divide values

For floating types only. */
OPENCV_HAL_IMPL_BIN_OP(/)

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_BIT_OP(bit_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> operator bit_op \
    (const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    v_reg<_Tp, n> c; \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return c; \
} \
template<typename _Tp, int n> inline v_reg<_Tp, n>& operator \
    bit_op##= (v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    for( int i = 0; i < n; i++ ) \
        a.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)(V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) bit_op \
                                                        V_TypeTraits<_Tp>::reinterpret_int(b.s[i]))); \
    return a; \
}

/** @brief Bitwise AND

Only for integer types. */
OPENCV_HAL_IMPL_BIT_OP(&)

/** @brief Bitwise OR

Only for integer types. */
OPENCV_HAL_IMPL_BIT_OP(|)

/** @brief Bitwise XOR

Only for integer types.*/
OPENCV_HAL_IMPL_BIT_OP(^)

/** @brief Bitwise NOT

Only for integer types.*/
template<typename _Tp, int n> inline v_reg<_Tp, n> operator ~ (const v_reg<_Tp, n>& a)
{
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int(~V_TypeTraits<_Tp>::reinterpret_int(a.s[i]));
        return c;
}

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

//! @cond IGNORED
OPENCV_HAL_IMPL_MATH_FUNC(v_sin, std::sin, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_cos, std::cos, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_exp, std::exp, _Tp)
OPENCV_HAL_IMPL_MATH_FUNC(v_log, std::log, _Tp)
//! @endcond

/** @brief Absolute value of elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_abs, (typename V_TypeTraits<_Tp>::abs_type)std::abs,
                          typename V_TypeTraits<_Tp>::abs_type)

/** @brief Round elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_round, cvRound, int)

/** @brief Floor elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_floor, cvFloor, int)

/** @brief Ceil elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_ceil, cvCeil, int)

/** @brief Truncate elements

Only for floating point types.*/
OPENCV_HAL_IMPL_MATH_FUNC(v_trunc, int, int)

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
For 32-bit integer and 32-bit floating point types. */
OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(v_reduce_min, std::min)

/** @brief Find one max value

Scheme:
@code
{A1 A2 A3 ...} => max(A1,A2,A3,...)
@endcode
For 32-bit integer and 32-bit floating point types. */
OPENCV_HAL_IMPL_REDUCE_MINMAX_FUNC(v_reduce_max, std::max)

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
#define OPENCV_HAL_IMPL_CMP_OP(cmp_op) \
template<typename _Tp, int n> \
inline v_reg<_Tp, n> operator cmp_op(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b) \
{ \
    typedef typename V_TypeTraits<_Tp>::int_type itype; \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = V_TypeTraits<_Tp>::reinterpret_from_int((itype)-(int)(a.s[i] cmp_op b.s[i])); \
    return c; \
}

/** @brief Less-than comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(<)

/** @brief Greater-than comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(>)

/** @brief Less-than or equal comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(<=)

/** @brief Greater-than or equal comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(>=)

/** @brief Equal comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(==)

/** @brief Not equal comparison

For all types except 64-bit integer values. */
OPENCV_HAL_IMPL_CMP_OP(!=)

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_ADD_SUB_OP(func, bin_op, cast_op, _Tp2) \
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
OPENCV_HAL_IMPL_ADD_SUB_OP(v_add_wrap, +, (_Tp), _Tp)

/** @brief Subtract values without saturation

For 8- and 16-bit integer values. */
OPENCV_HAL_IMPL_ADD_SUB_OP(v_sub_wrap, -, (_Tp), _Tp)

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
    const rtype mask = std::numeric_limits<_Tp>::is_signed ? (1 << (sizeof(rtype)*8 - 1)) : 0;
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
inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 c;
    for( int i = 0; i < c.nlanes; i++ )
        c.s[i] = _absdiff(a.s[i], b.s[i]);
    return c;
}

/** @overload

For 64-bit floating point values */
inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{
    v_float64x2 c;
    for( int i = 0; i < c.nlanes; i++ )
        c.s[i] = _absdiff(a.s[i], b.s[i]);
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
For floating point types only. */
template<typename _Tp, int n>
inline v_reg<_Tp, n> v_muladd(const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b,
                              const v_reg<_Tp, n>& c)
{
    v_reg<_Tp, n> d;
    for( int i = 0; i < n; i++ )
        d.s[i] = a.s[i]*b.s[i] + c.s[i];
    return d;
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
Implemented only for 16-bit signed source type (v_int16x8).
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
#define OPENCV_HAL_IMPL_SHIFT_OP(shift_op) \
template<typename _Tp, int n> inline v_reg<_Tp, n> operator shift_op(const v_reg<_Tp, n>& a, int imm) \
{ \
    v_reg<_Tp, n> c; \
    for( int i = 0; i < n; i++ ) \
        c.s[i] = (_Tp)(a.s[i] shift_op imm); \
    return c; \
}

/** @brief Bitwise shift left

For 16-, 32- and 64-bit integer values. */
OPENCV_HAL_IMPL_SHIFT_OP(<<)

/** @brief Bitwise shift right

For 16-, 32- and 64-bit integer values. */
OPENCV_HAL_IMPL_SHIFT_OP(>>)

/** @brief Sum packed values

Scheme:
@code
{A1 A2 A3 ...} => sum{A1,A2,A3,...}
@endcode
For 32-bit integer and 32-bit floating point types.*/
template<typename _Tp, int n> inline typename V_TypeTraits<_Tp>::sum_type v_reduce_sum(const v_reg<_Tp, n>& a)
{
    typename V_TypeTraits<_Tp>::sum_type c = a.s[0];
    for( int i = 1; i < n; i++ )
        c += a.s[i];
    return c;
}

/** @brief Get negative values mask

Returned value is a bit mask with bits set to 1 on places corresponding to negative packed values indexes.
Example:
@code{.cpp}
v_int32x4 r; // set to {-1, -1, 1, 1}
int mask = v_signmask(r); // mask = 3 <== 00000000 00000000 00000000 00000011
@endcode
For all types except 64-bit. */
template<typename _Tp, int n> inline int v_signmask(const v_reg<_Tp, n>& a)
{
    int mask = 0;
    for( int i = 0; i < n; i++ )
        mask |= (V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0) << i;
    return mask;
}

/** @brief Check if all packed values are less than zero

Unsigned values will be casted to signed: `uchar 254 => char -2`.
For all types except 64-bit. */
template<typename _Tp, int n> inline bool v_check_all(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) >= 0 )
            return false;
    return true;
}

/** @brief Check if any of packed values is less than zero

Unsigned values will be casted to signed: `uchar 254 => char -2`.
For all types except 64-bit. */
template<typename _Tp, int n> inline bool v_check_any(const v_reg<_Tp, n>& a)
{
    for( int i = 0; i < n; i++ )
        if( V_TypeTraits<_Tp>::reinterpret_int(a.s[i]) < 0 )
            return true;
    return false;
}

/** @brief Bitwise select

Return value will be built by combining values a and b using the following scheme:
If the i-th bit in _mask_ is 1
    select i-th bit from _a_
else
    select i-th bit from _b_ */
template<typename _Tp, int n> inline v_reg<_Tp, n> v_select(const v_reg<_Tp, n>& mask,
                                                           const v_reg<_Tp, n>& a, const v_reg<_Tp, n>& b)
{
    typedef V_TypeTraits<_Tp> Traits;
    typedef typename Traits::int_type int_type;
    v_reg<_Tp, n> c;
    for( int i = 0; i < n; i++ )
    {
        int_type m = Traits::reinterpret_int(mask.s[i]);
        c.s[i] =  Traits::reinterpret_from_int((Traits::reinterpret_int(a.s[i]) & m)
                                             | (Traits::reinterpret_int(b.s[i]) & ~m));
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
 */
template<typename _Tp>
inline v_reg<_Tp, V_SIMD128Traits<_Tp>::nlanes> v_load(const _Tp* ptr)
{
    return v_reg<_Tp, V_SIMD128Traits<_Tp>::nlanes>(ptr);
}

/** @brief Load register contents from memory (aligned)

similar to cv::v_load, but source memory block should be aligned (to 16-byte boundary)
 */
template<typename _Tp>
inline v_reg<_Tp, V_SIMD128Traits<_Tp>::nlanes> v_load_aligned(const _Tp* ptr)
{
    return v_reg<_Tp, V_SIMD128Traits<_Tp>::nlanes>(ptr);
}

/** @brief Load register contents from two memory blocks

@param loptr memory block containing data for first half (0..n/2)
@param hiptr memory block containing data for second half (n/2..n)

@code{.cpp}
int lo[2] = { 1, 2 }, hi[2] = { 3, 4 };
v_int32x4 r = v_load_halves(lo, hi);
@endcode
 */
template<typename _Tp>
inline v_reg<_Tp, V_SIMD128Traits<_Tp>::nlanes> v_load_halves(const _Tp* loptr, const _Tp* hiptr)
{
    v_reg<_Tp, V_SIMD128Traits<_Tp>::nlanes> c;
    for( int i = 0; i < c.nlanes/2; i++ )
    {
        c.s[i] = loptr[i];
        c.s[i+c.nlanes/2] = hiptr[i];
    }
    return c;
}

/** @brief Load register contents from memory with double expand

Same as cv::v_load, but result pack type will be 2x wider than memory type.

@code{.cpp}
short buf[4] = {1, 2, 3, 4}; // type is int16
v_int32x4 r = v_load_expand(buf); // r = {1, 2, 3, 4} - type is int32
@endcode
For 8-, 16-, 32-bit integer source types. */
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::w_type, V_SIMD128Traits<_Tp>::nlanes / 2>
v_load_expand(const _Tp* ptr)
{
    typedef typename V_TypeTraits<_Tp>::w_type w_type;
    v_reg<w_type, V_SIMD128Traits<w_type>::nlanes> c;
    for( int i = 0; i < c.nlanes; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

/** @brief Load register contents from memory with quad expand

Same as cv::v_load_expand, but result type is 4 times wider than source.
@code{.cpp}
char buf[4] = {1, 2, 3, 4}; // type is int8
v_int32x4 r = v_load_q(buf); // r = {1, 2, 3, 4} - type is int32
@endcode
For 8-bit integer source types. */
template<typename _Tp>
inline v_reg<typename V_TypeTraits<_Tp>::q_type, V_SIMD128Traits<_Tp>::nlanes / 4>
v_load_expand_q(const _Tp* ptr)
{
    typedef typename V_TypeTraits<_Tp>::q_type q_type;
    v_reg<q_type, V_SIMD128Traits<q_type>::nlanes> c;
    for( int i = 0; i < c.nlanes; i++ )
    {
        c.s[i] = ptr[i];
    }
    return c;
}

/** @brief Load and deinterleave (4 channels)

Load data from memory deinterleave and store to 4 registers.
Scheme:
@code
{A1 B1 C1 D1 A2 B2 C2 D2 ...} ==> {A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...}, {D1 D2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n> inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                                            v_reg<_Tp, n>& b, v_reg<_Tp, n>& c)
{
    int i, i3;
    for( i = i3 = 0; i < n; i++, i3 += 3 )
    {
        a.s[i] = ptr[i3];
        b.s[i] = ptr[i3+1];
        c.s[i] = ptr[i3+2];
    }
}

/** @brief Load and deinterleave (3 channels)

Load data from memory deinterleave and store to 3 registers.
Scheme:
@code
{A1 B1 C1 A2 B2 C2 ...} ==> {A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline void v_load_deinterleave(const _Tp* ptr, v_reg<_Tp, n>& a,
                                v_reg<_Tp, n>& b, v_reg<_Tp, n>& c,
                                v_reg<_Tp, n>& d)
{
    int i, i4;
    for( i = i4 = 0; i < n; i++, i4 += 4 )
    {
        a.s[i] = ptr[i4];
        b.s[i] = ptr[i4+1];
        c.s[i] = ptr[i4+2];
        d.s[i] = ptr[i4+3];
    }
}

/** @brief Interleave and store (3 channels)

Interleave and store data from 3 registers to memory.
Scheme:
@code
{A1 A2 ...}, {B1 B2 ...}, {C1 C2 ...}, {D1 D2 ...} ==> {A1 B1 C1 D1 A2 B2 C2 D2 ...}
@endcode
For all types except 64-bit. */
template<typename _Tp, int n>
inline void v_store_interleave( _Tp* ptr, const v_reg<_Tp, n>& a,
                                const v_reg<_Tp, n>& b, const v_reg<_Tp, n>& c)
{
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
                                                            const v_reg<_Tp, n>& d)
{
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
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
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
    for( int i = 0; i < n; i++ )
        ptr[i] = a.s[i];
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
For integer types only. */
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

/** @brief Round

Rounds each value. Input type is float vector ==> output type is int vector.*/
template<int n> inline v_reg<int, n> v_round(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvRound(a.s[i]);
    return c;
}

/** @brief Floor

Floor each value. Input type is float vector ==> output type is int vector.*/
template<int n> inline v_reg<int, n> v_floor(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvFloor(a.s[i]);
    return c;
}

/** @brief Ceil

Ceil each value. Input type is float vector ==> output type is int vector.*/
template<int n> inline v_reg<int, n> v_ceil(const v_reg<float, n>& a)
{
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = cvCeil(a.s[i]);
    return c;
}

/** @brief Trunc

Truncate each value. Input type is float vector ==> output type is int vector.*/
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
    v_reg<int, n> c;
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
    v_reg<int, n> c;
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
    v_reg<int, n> c;
    for( int i = 0; i < n; i++ )
    {
        c.s[i] = cvCeil(a.s[i]);
        c.s[i+n] = 0;
    }
    return c;
}

/** @brief Convert to float

Supported input type is cv::v_int32x4. */
template<int n> inline v_reg<float, n> v_cvt_f32(const v_reg<int, n>& a)
{
    v_reg<float, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (float)a.s[i];
    return c;
}

/** @brief Convert to double

Supported input type is cv::v_int32x4. */
template<int n> inline v_reg<double, n> v_cvt_f64(const v_reg<int, n*2>& a)
{
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
    return c;
}

/** @brief Convert to double

Supported input type is cv::v_float32x4. */
template<int n> inline v_reg<double, n> v_cvt_f64(const v_reg<float, n*2>& a)
{
    v_reg<double, n> c;
    for( int i = 0; i < n; i++ )
        c.s[i] = (double)a.s[i];
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
template<typename _Tp>
inline void v_transpose4x4( v_reg<_Tp, 4>& a0, const v_reg<_Tp, 4>& a1,
                            const v_reg<_Tp, 4>& a2, const v_reg<_Tp, 4>& a3,
                            v_reg<_Tp, 4>& b0, v_reg<_Tp, 4>& b1,
                            v_reg<_Tp, 4>& b2, v_reg<_Tp, 4>& b3 )
{
    b0 = v_reg<_Tp, 4>(a0.s[0], a1.s[0], a2.s[0], a3.s[0]);
    b1 = v_reg<_Tp, 4>(a0.s[1], a1.s[1], a2.s[1], a3.s[1]);
    b2 = v_reg<_Tp, 4>(a0.s[2], a1.s[2], a2.s[2], a3.s[2]);
    b3 = v_reg<_Tp, 4>(a0.s[3], a1.s[3], a2.s[3], a3.s[3]);
}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_INIT_ZERO(_Tpvec, _Tp, suffix) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec::zero(); }

//! @name Init with zero
//! @{
//! @brief Create new vector with zero elements
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_INIT_ZERO(v_int64x2, int64, s64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_INIT_VAL(_Tpvec, _Tp, suffix) \
inline _Tpvec v_setall_##suffix(_Tp val) { return _Tpvec::all(val); }

//! @name Init with value
//! @{
//! @brief Create new vector with elements set to a specific value
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_INIT_VAL(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_INIT_VAL(v_int64x2, int64, s64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_REINTERPRET(_Tpvec, _Tp, suffix) \
template<typename _Tp0, int n0> inline _Tpvec \
    v_reinterpret_as_##suffix(const v_reg<_Tp0, n0>& a) \
{ return a.template reinterpret_as<_Tp, _Tpvec::nlanes>(); }

//! @name Reinterpret
//! @{
//! @brief Convert vector to different type without modifying underlying data.
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int16x8, short, s16)
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int32x4, int, s32)
OPENCV_HAL_IMPL_C_REINTERPRET(v_float32x4, float, f32)
OPENCV_HAL_IMPL_C_REINTERPRET(v_float64x2, double, f64)
OPENCV_HAL_IMPL_C_REINTERPRET(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_C_REINTERPRET(v_int64x2, int64, s64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_SHIFTL(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return a << n; }

//! @name Left shift
//! @{
//! @brief Shift left
OPENCV_HAL_IMPL_C_SHIFTL(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_SHIFTL(v_int16x8, short)
OPENCV_HAL_IMPL_C_SHIFTL(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_SHIFTL(v_int32x4, int)
OPENCV_HAL_IMPL_C_SHIFTL(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_SHIFTL(v_int64x2, int64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_SHIFTR(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return a >> n; }

//! @name Right shift
//! @{
//! @brief Shift right
OPENCV_HAL_IMPL_C_SHIFTR(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_SHIFTR(v_int16x8, short)
OPENCV_HAL_IMPL_C_SHIFTR(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_SHIFTR(v_int32x4, int)
OPENCV_HAL_IMPL_C_SHIFTR(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_SHIFTR(v_int64x2, int64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_RSHIFTR(_Tpvec, _Tp) \
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ \
    _Tpvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        c.s[i] = (_Tp)((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
    return c; \
}

//! @name Rounding shift
//! @{
//! @brief Rounding shift right
OPENCV_HAL_IMPL_C_RSHIFTR(v_uint16x8, ushort)
OPENCV_HAL_IMPL_C_RSHIFTR(v_int16x8, short)
OPENCV_HAL_IMPL_C_RSHIFTR(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_C_RSHIFTR(v_int32x4, int)
OPENCV_HAL_IMPL_C_RSHIFTR(v_uint64x2, uint64)
OPENCV_HAL_IMPL_C_RSHIFTR(v_int64x2, int64)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_PACK(_Tpvec, _Tpnvec, _Tpn, pack_suffix) \
inline _Tpnvec v_##pack_suffix(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpnvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
    { \
        c.s[i] = saturate_cast<_Tpn>(a.s[i]); \
        c.s[i+_Tpvec::nlanes] = saturate_cast<_Tpn>(b.s[i]); \
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
OPENCV_HAL_IMPL_C_PACK(v_uint16x8, v_uint8x16, uchar, pack)
OPENCV_HAL_IMPL_C_PACK(v_int16x8, v_int8x16, schar, pack)
OPENCV_HAL_IMPL_C_PACK(v_uint32x4, v_uint16x8, ushort, pack)
OPENCV_HAL_IMPL_C_PACK(v_int32x4, v_int16x8, short, pack)
OPENCV_HAL_IMPL_C_PACK(v_uint64x2, v_uint32x4, unsigned, pack)
OPENCV_HAL_IMPL_C_PACK(v_int64x2, v_int32x4, int, pack)
OPENCV_HAL_IMPL_C_PACK(v_int16x8, v_uint8x16, uchar, pack_u)
OPENCV_HAL_IMPL_C_PACK(v_int32x4, v_uint16x8, ushort, pack_u)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_RSHR_PACK(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix) \
template<int n> inline _Tpnvec v_rshr_##pack_suffix(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpnvec c; \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
    { \
        c.s[i] = saturate_cast<_Tpn>((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
        c.s[i+_Tpvec::nlanes] = saturate_cast<_Tpn>((b.s[i] + ((_Tp)1 << (n - 1))) >> n); \
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
OPENCV_HAL_IMPL_C_RSHR_PACK(v_uint16x8, ushort, v_uint8x16, uchar, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int16x8, short, v_int8x16, schar, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_uint32x4, unsigned, v_uint16x8, ushort, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int32x4, int, v_int16x8, short, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_uint64x2, uint64, v_uint32x4, unsigned, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int64x2, int64, v_int32x4, int, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int16x8, short, v_uint8x16, uchar, pack_u)
OPENCV_HAL_IMPL_C_RSHR_PACK(v_int32x4, int, v_uint16x8, ushort, pack_u)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_PACK_STORE(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix) \
inline void v_##pack_suffix##_store(_Tpn* ptr, const _Tpvec& a) \
{ \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        ptr[i] = saturate_cast<_Tpn>(a.s[i]); \
}

//! @name Pack and store
//! @{
//! @brief Store values from the input vector into memory with pack
//!
//! Values will be stored into memory with saturating conversion to narrower type.
//! Variant with _u_ suffix converts to corresponding unsigned type.
//!
//! - pack: for 16-, 32- and 64-bit integer input types
//! - pack_u: for 16- and 32-bit signed integer input types
OPENCV_HAL_IMPL_C_PACK_STORE(v_uint16x8, ushort, v_uint8x16, uchar, pack)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int16x8, short, v_int8x16, schar, pack)
OPENCV_HAL_IMPL_C_PACK_STORE(v_uint32x4, unsigned, v_uint16x8, ushort, pack)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int32x4, int, v_int16x8, short, pack)
OPENCV_HAL_IMPL_C_PACK_STORE(v_uint64x2, uint64, v_uint32x4, unsigned, pack)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int64x2, int64, v_int32x4, int, pack)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int16x8, short, v_uint8x16, uchar, pack_u)
OPENCV_HAL_IMPL_C_PACK_STORE(v_int32x4, int, v_uint16x8, ushort, pack_u)
//! @}

//! @brief Helper macro
//! @ingroup core_hal_intrin_impl
#define OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(_Tpvec, _Tp, _Tpnvec, _Tpn, pack_suffix) \
template<int n> inline void v_rshr_##pack_suffix##_store(_Tpn* ptr, const _Tpvec& a) \
{ \
    for( int i = 0; i < _Tpvec::nlanes; i++ ) \
        ptr[i] = saturate_cast<_Tpn>((a.s[i] + ((_Tp)1 << (n - 1))) >> n); \
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
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_uint16x8, ushort, v_uint8x16, uchar, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int16x8, short, v_int8x16, schar, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_uint32x4, unsigned, v_uint16x8, ushort, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int32x4, int, v_int16x8, short, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_uint64x2, uint64, v_uint32x4, unsigned, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int64x2, int64, v_int32x4, int, pack)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int16x8, short, v_uint8x16, uchar, pack_u)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(v_int32x4, int, v_uint16x8, ushort, pack_u)
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
R0 = A0V0 + A1V1 + A2V2 + A3V3,
R1 = B0V0 + B1V1 + B2V2 + B3V3
...
@endcode
*/
inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    return v_float32x4(v.s[0]*m0.s[0] + v.s[1]*m1.s[0] + v.s[2]*m2.s[0] + v.s[3]*m3.s[0],
                       v.s[0]*m0.s[1] + v.s[1]*m1.s[1] + v.s[2]*m2.s[1] + v.s[3]*m3.s[1],
                       v.s[0]*m0.s[2] + v.s[1]*m1.s[2] + v.s[2]*m2.s[2] + v.s[3]*m3.s[2],
                       v.s[0]*m0.s[3] + v.s[1]*m1.s[3] + v.s[2]*m2.s[3] + v.s[3]*m3.s[3]);
}

//! @}

}

#endif
