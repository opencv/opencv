// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on files from package issued with the following license:

/*============================================================================

This C header file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#pragma once
#ifndef softfloat_h
#define softfloat_h 1

#include "cvdef.h"

// int32_t / uint32_t
#if defined(_MSC_VER) && _MSC_VER < 1600 /* MSVS 2010 */
namespace cv {
typedef signed int int32_t;
typedef unsigned int uint32_t;
}
#elif defined(_MSC_VER) || __cplusplus >= 201103L
#include <cstdint>
#else
#include <stdint.h>
#endif

namespace cv
{

/** @addtogroup core_utils_softfloat

  [SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) is a software implementation
  of floating-point calculations according to IEEE 754 standard.
  All calculations are done in integers, that's why they are machine-independent and bit-exact.
  This library can be useful in accuracy-critical parts like look-up tables generation, tests, etc.
  OpenCV contains a subset of SoftFloat partially rewritten to C++.

  ### Types

  There are two basic types: @ref softfloat and @ref softdouble.
  These types are binary compatible with float and double types respectively
  and support conversions to/from them.
  Other types from original SoftFloat library like fp16 or fp128 were thrown away
  as well as quiet/signaling NaN support, on-the-fly rounding mode switch
  and exception flags (though exceptions can be implemented in the future).

  ### Operations

  Both types support the following:
  - Construction from signed and unsigned 32-bit and 64 integers,
  float/double or raw binary representation
  - Conversions betweeen each other, to float or double and to int
  using @ref cvRound, @ref cvTrunc, @ref cvFloor, @ref cvCeil or a bunch of
  saturate_cast functions
  - Add, subtract, multiply, divide, remainder, square root, FMA with absolute precision
  - Comparison operations
  - Explicit sign, exponent and significand manipulation through get/set methods,
 number state indicators (isInf, isNan, isSubnormal)
  - Type-specific constants like eps, minimum/maximum value, best pi approximation, etc.
  - min(), max(), abs(), exp(), log() and pow() functions

*/
//! @{

struct softfloat;
struct softdouble;

struct CV_EXPORTS softfloat
{
public:
    /** @brief Default constructor */
    softfloat() { v = 0; }
    /** @brief Copy constructor */
    softfloat( const softfloat& c) { v = c.v; }
    /** @brief Assign constructor */
    softfloat& operator=( const softfloat& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    /** @brief Construct from raw

    Builds new value from raw binary representation
    */
    static const softfloat fromRaw( const uint32_t a ) { softfloat x; x.v = a; return x; }

    /** @brief Construct from integer */
    explicit softfloat( const uint32_t );
    explicit softfloat( const uint64_t );
    explicit softfloat( const int32_t );
    explicit softfloat( const int64_t );
    /** @brief Construct from float */
    explicit softfloat( const float a ) { Cv32suf s; s.f = a; v = s.u; }

    /** @brief Type casts  */
    operator softdouble() const;
    operator float() const { Cv32suf s; s.u = v; return s.f; }

    /** @brief Basic arithmetics */
    softfloat operator + (const softfloat&) const;
    softfloat operator - (const softfloat&) const;
    softfloat operator * (const softfloat&) const;
    softfloat operator / (const softfloat&) const;
    softfloat operator - () const { softfloat x; x.v = v ^ (1U << 31); return x; }

    /** @brief Remainder operator

    A quote from original SoftFloat manual:

    > The IEEE Standard remainder operation computes the value
    > a - n * b, where n is the integer closest to a / b.
    > If a / b is exactly halfway between two integers, n is the even integer
    > closest to a / b. The IEEE Standard’s remainder operation is always exact and so requires no rounding.
    > Depending on the relative magnitudes of the operands, the remainder functions
    > can take considerably longer to execute than the other SoftFloat functions.
    > This is an inherent characteristic of the remainder operation itself and is not a flaw
    > in the SoftFloat implementation.
    */
    softfloat operator % (const softfloat&) const;

    softfloat& operator += (const softfloat& a) { *this = *this + a; return *this; }
    softfloat& operator -= (const softfloat& a) { *this = *this - a; return *this; }
    softfloat& operator *= (const softfloat& a) { *this = *this * a; return *this; }
    softfloat& operator /= (const softfloat& a) { *this = *this / a; return *this; }
    softfloat& operator %= (const softfloat& a) { *this = *this % a; return *this; }

    /** @brief Comparison operations

     - Any operation with NaN produces false
       + The only exception is when x is NaN: x != y for any y.
     - Positive and negative zeros are equal
    */
    bool operator == ( const softfloat& ) const;
    bool operator != ( const softfloat& ) const;
    bool operator >  ( const softfloat& ) const;
    bool operator >= ( const softfloat& ) const;
    bool operator <  ( const softfloat& ) const;
    bool operator <= ( const softfloat& ) const;

    /** @brief NaN state indicator */
    inline bool isNaN() const { return (v & 0x7fffffff)  > 0x7f800000; }
    /** @brief Inf state indicator */
    inline bool isInf() const { return (v & 0x7fffffff) == 0x7f800000; }
    /** @brief Subnormal number indicator */
    inline bool isSubnormal() const { return ((v >> 23) & 0xFF) == 0; }

    /** @brief Get sign bit */
    inline bool getSign() const { return (v >> 31) != 0; }
    /** @brief Construct a copy with new sign bit */
    inline softfloat setSign(bool sign) const { softfloat x; x.v = (v & ((1U << 31) - 1)) | ((uint32_t)sign << 31); return x; }
    /** @brief Get 0-based exponent */
    inline int getExp() const { return ((v >> 23) & 0xFF) - 127; }
    /** @brief Construct a copy with new 0-based exponent */
    inline softfloat setExp(int e) const { softfloat x; x.v = (v & 0x807fffff) | (((e + 127) & 0xFF) << 23 ); return x; }

    /** @brief Get a fraction part

    Returns a number 1 <= x < 2 with the same significand
    */
    inline softfloat getFrac() const
    {
        uint_fast32_t vv = (v & 0x007fffff) | (127 << 23);
        return softfloat::fromRaw(vv);
    }
    /** @brief Construct a copy with provided significand

    Constructs a copy of a number with significand taken from parameter
    */
    inline softfloat setFrac(const softfloat& s) const
    {
        softfloat x;
        x.v = (v & 0xff800000) | (s.v & 0x007fffff);
        return x;
    }

    /** @brief Zero constant */
    static softfloat zero() { return softfloat::fromRaw( 0 ); }
    /** @brief Positive infinity constant */
    static softfloat  inf() { return softfloat::fromRaw( 0xFF << 23 ); }
    /** @brief Default NaN constant */
    static softfloat  nan() { return softfloat::fromRaw( 0x7fffffff ); }
    /** @brief One constant */
    static softfloat  one() { return softfloat::fromRaw(  127 << 23 ); }
    /** @brief Smallest normalized value */
    static softfloat  min() { return softfloat::fromRaw( 0x01 << 23 ); }
    /** @brief Difference between 1 and next representable value */
    static softfloat  eps() { return softfloat::fromRaw( (127 - 23) << 23 ); }
    /** @brief Biggest finite value */
    static softfloat  max() { return softfloat::fromRaw( (0xFF << 23) - 1 ); }
    /** @brief Correct pi approximation */
    static softfloat   pi() { return softfloat::fromRaw( 0x40490fdb ); }

    uint32_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct CV_EXPORTS softdouble
{
public:
    /** @brief Default constructor */
    softdouble() : v(0) { }
    /** @brief Copy constructor */
    softdouble( const softdouble& c) { v = c.v; }
    /** @brief Assign constructor */
    softdouble& operator=( const softdouble& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    /** @brief Construct from raw

    Builds new value from raw binary representation
    */
    static softdouble fromRaw( const uint64_t a ) { softdouble x; x.v = a; return x; }

    /** @brief Construct from integer */
    explicit softdouble( const uint32_t );
    explicit softdouble( const uint64_t );
    explicit softdouble( const  int32_t );
    explicit softdouble( const  int64_t );
    /** @brief Construct from double */
    explicit softdouble( const double a ) { Cv64suf s; s.f = a; v = s.u; }

    /** @brief Type casts  */
    operator softfloat() const;
    operator double() const { Cv64suf s; s.u = v; return s.f; }

    /** @brief Basic arithmetics */
    softdouble operator + (const softdouble&) const;
    softdouble operator - (const softdouble&) const;
    softdouble operator * (const softdouble&) const;
    softdouble operator / (const softdouble&) const;
    softdouble operator - () const { softdouble x; x.v = v ^ (1ULL << 63); return x; }

    /** @brief Remainder operator

    A quote from original SoftFloat manual:

    > The IEEE Standard remainder operation computes the value
    > a - n * b, where n is the integer closest to a / b.
    > If a / b is exactly halfway between two integers, n is the even integer
    > closest to a / b. The IEEE Standard’s remainder operation is always exact and so requires no rounding.
    > Depending on the relative magnitudes of the operands, the remainder functions
    > can take considerably longer to execute than the other SoftFloat functions.
    > This is an inherent characteristic of the remainder operation itself and is not a flaw
    > in the SoftFloat implementation.
    */
    softdouble operator % (const softdouble&) const;

    softdouble& operator += (const softdouble& a) { *this = *this + a; return *this; }
    softdouble& operator -= (const softdouble& a) { *this = *this - a; return *this; }
    softdouble& operator *= (const softdouble& a) { *this = *this * a; return *this; }
    softdouble& operator /= (const softdouble& a) { *this = *this / a; return *this; }
    softdouble& operator %= (const softdouble& a) { *this = *this % a; return *this; }

    /** @brief Comparison operations

     - Any operation with NaN produces false
       + The only exception is when x is NaN: x != y for any y.
     - Positive and negative zeros are equal
    */
    bool operator == ( const softdouble& ) const;
    bool operator != ( const softdouble& ) const;
    bool operator >  ( const softdouble& ) const;
    bool operator >= ( const softdouble& ) const;
    bool operator <  ( const softdouble& ) const;
    bool operator <= ( const softdouble& ) const;

    /** @brief NaN state indicator */
    inline bool isNaN() const { return (v & 0x7fffffffffffffff)  > 0x7ff0000000000000; }
    /** @brief Inf state indicator */
    inline bool isInf() const { return (v & 0x7fffffffffffffff) == 0x7ff0000000000000; }
    /** @brief Subnormal number indicator */
    inline bool isSubnormal() const { return ((v >> 52) & 0x7FF) == 0; }

    /** @brief Get sign bit */
    inline bool getSign() const { return (v >> 63) != 0; }
    /** @brief Construct a copy with new sign bit */
    softdouble setSign(bool sign) const { softdouble x; x.v = (v & ((1ULL << 63) - 1)) | ((uint_fast64_t)(sign) << 63); return x; }
    /** @brief Get 0-based exponent */
    inline int getExp() const { return ((v >> 52) & 0x7FF) - 1023; }
    /** @brief Construct a copy with new 0-based exponent */
    inline softdouble setExp(int e) const
    {
        softdouble x;
        x.v = (v & 0x800FFFFFFFFFFFFF) | ((uint_fast64_t)((e + 1023) & 0x7FF) << 52);
        return x;
    }

    /** @brief Get a fraction part

    Returns a number 1 <= x < 2 with the same significand
    */
    inline softdouble getFrac() const
    {
        uint_fast64_t vv = (v & 0x000FFFFFFFFFFFFF) | ((uint_fast64_t)(1023) << 52);
        return softdouble::fromRaw(vv);
    }
    /** @brief Construct a copy with provided significand

    Constructs a copy of a number with significand taken from parameter
    */
    inline softdouble setFrac(const softdouble& s) const
    {
        softdouble x;
        x.v = (v & 0xFFF0000000000000) | (s.v & 0x000FFFFFFFFFFFFF);
        return x;
    }

    /** @brief Zero constant */
    static softdouble zero() { return softdouble::fromRaw( 0 ); }
    /** @brief Positive infinity constant */
    static softdouble  inf() { return softdouble::fromRaw( (uint_fast64_t)(0x7FF) << 52 ); }
    /** @brief Default NaN constant */
    static softdouble  nan() { return softdouble::fromRaw( CV_BIG_INT(0x7FFFFFFFFFFFFFFF) ); }
    /** @brief One constant */
    static softdouble  one() { return softdouble::fromRaw( (uint_fast64_t)( 1023) << 52 ); }
    /** @brief Smallest normalized value */
    static softdouble  min() { return softdouble::fromRaw( (uint_fast64_t)( 0x01) << 52 ); }
    /** @brief Difference between 1 and next representable value */
    static softdouble  eps() { return softdouble::fromRaw( (uint_fast64_t)( 1023 - 52 ) << 52 ); }
    /** @brief Biggest finite value */
    static softdouble  max() { return softdouble::fromRaw( ((uint_fast64_t)(0x7FF) << 52) - 1 ); }
    /** @brief Correct pi approximation */
    static softdouble   pi() { return softdouble::fromRaw( CV_BIG_INT(0x400921FB54442D18) ); }

    uint64_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

/** @brief Fused Multiplication and Addition

Computes (a*b)+c with single rounding
*/
CV_EXPORTS softfloat  mulAdd( const softfloat&  a, const softfloat&  b, const softfloat & c);
CV_EXPORTS softdouble mulAdd( const softdouble& a, const softdouble& b, const softdouble& c);

/** @brief Square root */
CV_EXPORTS softfloat  sqrt( const softfloat&  a );
CV_EXPORTS softdouble sqrt( const softdouble& a );
}

/*----------------------------------------------------------------------------
| Ported from OpenCV and added for usability
*----------------------------------------------------------------------------*/

/** @brief Truncates number to integer with minimum magnitude */
CV_EXPORTS int cvTrunc(const cv::softfloat&  a);
CV_EXPORTS int cvTrunc(const cv::softdouble& a);

/** @brief Rounds a number to nearest even integer */
CV_EXPORTS int cvRound(const cv::softfloat&  a);
CV_EXPORTS int cvRound(const cv::softdouble& a);

/** @brief Rounds a number down to integer */
CV_EXPORTS int cvFloor(const cv::softfloat&  a);
CV_EXPORTS int cvFloor(const cv::softdouble& a);

/** @brief Rounds number up to integer */
CV_EXPORTS int  cvCeil(const cv::softfloat&  a);
CV_EXPORTS int  cvCeil(const cv::softdouble& a);

namespace cv
{
/** @brief Saturate casts */
template<typename _Tp> static inline _Tp saturate_cast(softfloat  a) { return _Tp(a); }
template<typename _Tp> static inline _Tp saturate_cast(softdouble a) { return _Tp(a); }

template<> inline uchar saturate_cast<uchar>(softfloat  a) { return (uchar)std::max(std::min(cvRound(a), (int)UCHAR_MAX), 0); }
template<> inline uchar saturate_cast<uchar>(softdouble a) { return (uchar)std::max(std::min(cvRound(a), (int)UCHAR_MAX), 0); }

template<> inline schar saturate_cast<schar>(softfloat  a) { return (schar)std::min(std::max(cvRound(a), (int)SCHAR_MIN), (int)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(softdouble a) { return (schar)std::min(std::max(cvRound(a), (int)SCHAR_MIN), (int)SCHAR_MAX); }

template<> inline ushort saturate_cast<ushort>(softfloat  a) { return (ushort)std::max(std::min(cvRound(a), (int)USHRT_MAX), 0); }
template<> inline ushort saturate_cast<ushort>(softdouble a) { return (ushort)std::max(std::min(cvRound(a), (int)USHRT_MAX), 0); }

template<> inline short saturate_cast<short>(softfloat  a) { return (short)std::min(std::max(cvRound(a), (int)SHRT_MIN), (int)SHRT_MAX); }
template<> inline short saturate_cast<short>(softdouble a) { return (short)std::min(std::max(cvRound(a), (int)SHRT_MIN), (int)SHRT_MAX); }

template<> inline int saturate_cast<int>(softfloat  a) { return cvRound(a); }
template<> inline int saturate_cast<int>(softdouble a) { return cvRound(a); }

/** @brief Saturate cast to unsigned integer
We intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
*/
template<> inline unsigned saturate_cast<unsigned>(softfloat  a) { return cvRound(a); }
template<> inline unsigned saturate_cast<unsigned>(softdouble a) { return cvRound(a); }

/** @brief Min and Max functions */
inline softfloat  min(const softfloat&  a, const softfloat&  b) { return (a > b) ? b : a; }
inline softdouble min(const softdouble& a, const softdouble& b) { return (a > b) ? b : a; }

inline softfloat  max(const softfloat&  a, const softfloat&  b) { return (a > b) ? a : b; }
inline softdouble max(const softdouble& a, const softdouble& b) { return (a > b) ? a : b; }

/** @brief Absolute value */
inline softfloat  abs( softfloat  a) { softfloat  x; x.v = a.v & ((1U   << 31) - 1); return x; }
inline softdouble abs( softdouble a) { softdouble x; x.v = a.v & ((1ULL << 63) - 1); return x; }

/** @brief Exponent

Special cases:
- exp(NaN) is NaN
- exp(-Inf) == 0
- exp(+Inf) == +Inf
*/
CV_EXPORTS softfloat  exp( const softfloat&  a);
CV_EXPORTS softdouble exp( const softdouble& a);

/** @brief Natural logarithm

Special cases:
- log(NaN), log(x < 0) are NaN
- log(0) == -Inf
*/
CV_EXPORTS softfloat  log( const softfloat&  a );
CV_EXPORTS softdouble log( const softdouble& a );

/** @brief Raising to the power

Special cases:
- x**NaN is NaN for any x
- ( |x| == 1 )**Inf is NaN
- ( |x|  > 1 )**+Inf or ( |x| < 1 )**-Inf is +Inf
- ( |x|  > 1 )**-Inf or ( |x| < 1 )**+Inf is 0
- x ** 0 == 1 for any x
- x ** 1 == 1 for any x
- NaN ** y is NaN for any other y
- Inf**(y < 0) == 0
- Inf ** y is +Inf for any other y
- (x < 0)**y is NaN for any other y if x can't be correctly rounded to integer
- 0 ** 0 == 1
- 0 ** (y < 0) is +Inf
- 0 ** (y > 0) is 0
*/
CV_EXPORTS softfloat  pow( const softfloat&  a, const softfloat&  b);
CV_EXPORTS softdouble pow( const softdouble& a, const softdouble& b);

/** @brief Cube root

Special cases:
- cbrt(NaN) is NaN
- cbrt(+/-Inf) is +/-Inf
*/
CV_EXPORTS softfloat cbrt( const softfloat& a );

/** @brief Sine

Special cases:
- sin(Inf) or sin(NaN) is NaN
- sin(x) == x when sin(x) is close to zero
*/
CV_EXPORTS softdouble sin( const softdouble& a );

/** @brief Cosine
 *
Special cases:
- cos(Inf) or cos(NaN) is NaN
- cos(x) == +/- 1 when cos(x) is close to +/- 1
*/
CV_EXPORTS softdouble cos( const softdouble& a );

}

//! @}

#endif
