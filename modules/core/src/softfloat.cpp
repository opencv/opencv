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

/*============================================================================

This C source file is part of the SoftFloat IEEE Floating-Point Arithmetic
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

#include "precomp.hpp"

namespace cv
{

/*----------------------------------------------------------------------------
| Software floating-point underflow tininess-detection mode.
*----------------------------------------------------------------------------*/
enum {
    tininess_beforeRounding = 0,
    tininess_afterRounding  = 1
};
//fixed to make softfloat code stateless
const uint_fast8_t globalDetectTininess = tininess_afterRounding;

/*----------------------------------------------------------------------------
| Software floating-point exception flags.
*----------------------------------------------------------------------------*/
enum {
    flag_inexact   =  1,
    flag_underflow =  2,
    flag_overflow  =  4,
    flag_infinite  =  8,
    flag_invalid   = 16
};

// Disabled to make softfloat code stateless
// This function may be changed in the future for better error handling
inline void raiseFlags( uint_fast8_t /* flags */)
{
    //exceptionFlags |= flags;
}

/*----------------------------------------------------------------------------
| Software floating-point rounding mode.
*----------------------------------------------------------------------------*/
enum {
    round_near_even   = 0,
    round_minMag      = 1,
    round_min         = 2,
    round_max         = 3,
    round_near_maxMag = 4,
    round_odd         = 5
};

//fixed to make softfloat code stateless
const uint_fast8_t globalRoundingMode = round_near_even;

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

#define signF32UI( a ) (((uint32_t) (a)>>31) != 0)
#define expF32UI( a ) ((int_fast16_t) ((a)>>23) & 0xFF)
#define fracF32UI( a ) ((a) & 0x007FFFFF)
#define packToF32UI( sign, exp, sig ) (((uint32_t) (sign)<<31) + ((uint32_t) (exp)<<23) + (sig))

#define isNaNF32UI( a ) (((~(a) & 0x7F800000) == 0) && ((a) & 0x007FFFFF))

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

#define signF64UI( a ) (((uint64_t) (a)>>63) != 0)
#define expF64UI( a ) ((int_fast16_t) ((a)>>52) & 0x7FF)
#define fracF64UI( a ) ((a) & UINT64_C( 0x000FFFFFFFFFFFFF ))
#define packToF64UI( sign, exp, sig ) ((uint64_t) (((uint_fast64_t) (sign)<<63) + ((uint_fast64_t) (exp)<<52) + (sig)))

#define isNaNF64UI( a ) (((~(a) & UINT64_C( 0x7FF0000000000000 )) == 0) && ((a) & UINT64_C( 0x000FFFFFFFFFFFFF )))

/*----------------------------------------------------------------------------
| Types used to pass 32-bit and 64-bit floating-point
| arguments and results to/from functions.  These types must be exactly
| 32 bits and 64 bits in size, respectively.  Where a
| platform has "native" support for IEEE-Standard floating-point formats,
| the types below may, if desired, be defined as aliases for the native types
| (typically 'float' and 'double').
*----------------------------------------------------------------------------*/
typedef softfloat float32_t;
typedef softdouble float64_t;

/*----------------------------------------------------------------------------
| Integer-to-floating-point conversion routines.
*----------------------------------------------------------------------------*/
float32_t ui32_to_f32( uint32_t );
float64_t ui32_to_f64( uint32_t );
float32_t ui64_to_f32( uint64_t );
float64_t ui64_to_f64( uint64_t );
float32_t i32_to_f32( int32_t );
float64_t i32_to_f64( int32_t );
float32_t i64_to_f32( int64_t );
float64_t i64_to_f64( int64_t );

/*----------------------------------------------------------------------------
| 32-bit (single-precision) floating-point operations.
*----------------------------------------------------------------------------*/
uint_fast32_t f32_to_ui32( float32_t, uint_fast8_t, bool );
uint_fast64_t f32_to_ui64( float32_t, uint_fast8_t, bool );
int_fast32_t f32_to_i32( float32_t, uint_fast8_t, bool );
int_fast64_t f32_to_i64( float32_t, uint_fast8_t, bool );
uint_fast32_t f32_to_ui32_r_minMag( float32_t, bool );
uint_fast64_t f32_to_ui64_r_minMag( float32_t, bool );
int_fast32_t f32_to_i32_r_minMag( float32_t, bool );
int_fast64_t f32_to_i64_r_minMag( float32_t, bool );
float64_t f32_to_f64( float32_t );
float32_t f32_roundToInt( float32_t, uint_fast8_t, bool );
float32_t f32_add( float32_t, float32_t );
float32_t f32_sub( float32_t, float32_t );
float32_t f32_mul( float32_t, float32_t );
float32_t f32_mulAdd( float32_t, float32_t, float32_t );
float32_t f32_div( float32_t, float32_t );
float32_t f32_rem( float32_t, float32_t );
float32_t f32_sqrt( float32_t );
bool f32_eq( float32_t, float32_t );
bool f32_le( float32_t, float32_t );
bool f32_lt( float32_t, float32_t );
bool f32_eq_signaling( float32_t, float32_t );
bool f32_le_quiet( float32_t, float32_t );
bool f32_lt_quiet( float32_t, float32_t );
bool f32_isSignalingNaN( float32_t );

/*----------------------------------------------------------------------------
| 64-bit (double-precision) floating-point operations.
*----------------------------------------------------------------------------*/
uint_fast32_t f64_to_ui32( float64_t, uint_fast8_t, bool );
uint_fast64_t f64_to_ui64( float64_t, uint_fast8_t, bool );
int_fast32_t f64_to_i32( float64_t, uint_fast8_t, bool );
int_fast64_t f64_to_i64( float64_t, uint_fast8_t, bool );
uint_fast32_t f64_to_ui32_r_minMag( float64_t, bool );
uint_fast64_t f64_to_ui64_r_minMag( float64_t, bool );
int_fast32_t f64_to_i32_r_minMag( float64_t, bool );
int_fast64_t f64_to_i64_r_minMag( float64_t, bool );
float32_t f64_to_f32( float64_t );
float64_t f64_roundToInt( float64_t, uint_fast8_t, bool );
float64_t f64_add( float64_t, float64_t );
float64_t f64_sub( float64_t, float64_t );
float64_t f64_mul( float64_t, float64_t );
float64_t f64_mulAdd( float64_t, float64_t, float64_t );
float64_t f64_div( float64_t, float64_t );
float64_t f64_rem( float64_t, float64_t );
float64_t f64_sqrt( float64_t );
bool f64_eq( float64_t, float64_t );
bool f64_le( float64_t, float64_t );
bool f64_lt( float64_t, float64_t );
bool f64_eq_signaling( float64_t, float64_t );
bool f64_le_quiet( float64_t, float64_t );
bool f64_lt_quiet( float64_t, float64_t );
bool f64_isSignalingNaN( float64_t );

/*----------------------------------------------------------------------------
| Ported from OpenCV and added for usability
*----------------------------------------------------------------------------*/

float32_t f32_powi( float32_t x, int y);
float64_t f64_powi( float64_t x, int y);

float32_t f32_exp( float32_t x);
float64_t f64_exp(float64_t x);
float32_t f32_log(float32_t x);
float64_t f64_log(float64_t x);
float32_t f32_cbrt(float32_t x);
float32_t f32_pow( float32_t x, float32_t y);
float64_t f64_pow( float64_t x, float64_t y);

/*----------------------------------------------------------------------------
| softfloat and softdouble methods and members
*----------------------------------------------------------------------------*/

softfloat::softfloat( const uint32_t a ) { *this = ui32_to_f32(a); }
softfloat::softfloat( const uint64_t a ) { *this = ui64_to_f32(a); }
softfloat::softfloat( const  int32_t a ) { *this =  i32_to_f32(a); }
softfloat::softfloat( const  int64_t a ) { *this =  i64_to_f32(a); }

softfloat::operator softdouble() const { return f32_to_f64(*this); }

softfloat softfloat::operator + (const softfloat& a) const { return f32_add(*this, a); }
softfloat softfloat::operator - (const softfloat& a) const { return f32_sub(*this, a); }
softfloat softfloat::operator * (const softfloat& a) const { return f32_mul(*this, a); }
softfloat softfloat::operator / (const softfloat& a) const { return f32_div(*this, a); }
softfloat softfloat::operator % (const softfloat& a) const { return f32_rem(*this, a); }

bool softfloat::operator == ( const softfloat& a ) const { return  f32_eq(*this, a); }
bool softfloat::operator != ( const softfloat& a ) const { return !f32_eq(*this, a); }
bool softfloat::operator >  ( const softfloat& a ) const { return  f32_lt(a, *this); }
bool softfloat::operator >= ( const softfloat& a ) const { return  f32_le(a, *this); }
bool softfloat::operator <  ( const softfloat& a ) const { return  f32_lt(*this, a); }
bool softfloat::operator <= ( const softfloat& a ) const { return  f32_le(*this, a); }

softdouble::softdouble( const uint32_t a ) { *this = ui32_to_f64(a); }
softdouble::softdouble( const uint64_t a ) { *this = ui64_to_f64(a); }
softdouble::softdouble( const  int32_t a ) { *this =  i32_to_f64(a); }
softdouble::softdouble( const  int64_t a ) { *this =  i64_to_f64(a); }

}

int cvTrunc(const cv::softfloat& a) { return cv::f32_to_i32_r_minMag(a, false); }
int cvRound(const cv::softfloat& a) { return cv::f32_to_i32(a, cv::round_near_even, false); }
int cvFloor(const cv::softfloat& a) { return cv::f32_to_i32(a, cv::round_min, false); }
int cvCeil (const cv::softfloat& a) { return cv::f32_to_i32(a, cv::round_max, false); }

int cvTrunc(const cv::softdouble& a) { return cv::f64_to_i32_r_minMag(a, false); }
int cvRound(const cv::softdouble& a) { return cv::f64_to_i32(a, cv::round_near_even, false); }
int cvFloor(const cv::softdouble& a) { return cv::f64_to_i32(a, cv::round_min, false); }
int cvCeil (const cv::softdouble& a) { return cv::f64_to_i32(a, cv::round_max, false); }

namespace cv
{
softdouble::operator softfloat() const { return f64_to_f32(*this); }

softdouble softdouble::operator + (const softdouble& a) const { return f64_add(*this, a); }
softdouble softdouble::operator - (const softdouble& a) const { return f64_sub(*this, a); }
softdouble softdouble::operator * (const softdouble& a) const { return f64_mul(*this, a); }
softdouble softdouble::operator / (const softdouble& a) const { return f64_div(*this, a); }
softdouble softdouble::operator % (const softdouble& a) const { return f64_rem(*this, a); }

bool softdouble::operator == (const softdouble& a) const { return  f64_eq(*this, a); }
bool softdouble::operator != (const softdouble& a) const { return !f64_eq(*this, a); }
bool softdouble::operator >  (const softdouble& a) const { return  f64_lt(a, *this); }
bool softdouble::operator >= (const softdouble& a) const { return  f64_le(a, *this); }
bool softdouble::operator <  (const softdouble& a) const { return  f64_lt(*this, a); }
bool softdouble::operator <= (const softdouble& a) const { return  f64_le(*this, a); }

/*----------------------------------------------------------------------------
| Overloads for math functions
*----------------------------------------------------------------------------*/

softfloat  mulAdd( const softfloat&  a, const softfloat&  b, const softfloat & c) { return f32_mulAdd(a, b, c); }
softdouble mulAdd( const softdouble& a, const softdouble& b, const softdouble& c) { return f64_mulAdd(a, b, c); }

softfloat  sqrt( const softfloat&  a ) { return f32_sqrt(a); }
softdouble sqrt( const softdouble& a ) { return f64_sqrt(a); }

softfloat  exp( const softfloat&  a) { return f32_exp(a); }
softdouble exp( const softdouble& a) { return f64_exp(a); }

softfloat  log( const softfloat&  a ) { return f32_log(a); }
softdouble log( const softdouble& a ) { return f64_log(a); }

softfloat  pow( const softfloat&  a, const softfloat&  b) { return f32_pow(a, b); }
softdouble pow( const softdouble& a, const softdouble& b) { return f64_pow(a, b); }

softfloat cbrt(const softfloat& a) { return f32_cbrt(a); }

/*----------------------------------------------------------------------------
| The values to return on conversions to 32-bit integer formats that raise an
| invalid exception.
*----------------------------------------------------------------------------*/
#define ui32_fromPosOverflow 0xFFFFFFFF
#define ui32_fromNegOverflow 0
#define ui32_fromNaN         0xFFFFFFFF
#define i32_fromPosOverflow  0x7FFFFFFF
#define i32_fromNegOverflow  (-0x7FFFFFFF - 1)
#define i32_fromNaN          0x7FFFFFFF

/*----------------------------------------------------------------------------
| The values to return on conversions to 64-bit integer formats that raise an
| invalid exception.
*----------------------------------------------------------------------------*/
#define ui64_fromPosOverflow UINT64_C( 0xFFFFFFFFFFFFFFFF )
#define ui64_fromNegOverflow 0
#define ui64_fromNaN         UINT64_C( 0xFFFFFFFFFFFFFFFF )
#define i64_fromPosOverflow  UINT64_C( 0x7FFFFFFFFFFFFFFF )
//fixed unsigned unary minus: -x == ~x + 1
//#define i64_fromNegOverflow (-UINT64_C( 0x7FFFFFFFFFFFFFFF ) - 1)
#define i64_fromNegOverflow  (~UINT64_C( 0x7FFFFFFFFFFFFFFF ) + 1 - 1)
#define i64_fromNaN          UINT64_C( 0x7FFFFFFFFFFFFFFF )

/*----------------------------------------------------------------------------
| "Common NaN" structure, used to transfer NaN representations from one format
| to another.
*----------------------------------------------------------------------------*/
struct commonNaN {
    bool sign;
#ifndef WORDS_BIGENDIAN
    uint64_t v0, v64;
#else
    uint64_t v64, v0;
#endif
};

/*----------------------------------------------------------------------------
| The bit pattern for a default generated 32-bit floating-point NaN.
*----------------------------------------------------------------------------*/
#define defaultNaNF32UI 0xFFC00000

/*----------------------------------------------------------------------------
| Returns true when 32-bit unsigned integer `uiA' has the bit pattern of a
| 32-bit floating-point signaling NaN.
| Note:  This macro evaluates its argument more than once.
*----------------------------------------------------------------------------*/
#define softfloat_isSigNaNF32UI( uiA ) ((((uiA) & 0x7FC00000) == 0x7F800000) && ((uiA) & 0x003FFFFF))

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 32-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
static void softfloat_f32UIToCommonNaN( uint_fast32_t uiA, struct commonNaN *zPtr );

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 32-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
static uint_fast32_t softfloat_commonNaNToF32UI( const struct commonNaN *aPtr );

/*----------------------------------------------------------------------------
| Interpreting `uiA' and `uiB' as the bit patterns of two 32-bit floating-
| point values, at least one of which is a NaN, returns the bit pattern of
| the combined NaN result.  If either `uiA' or `uiB' has the pattern of a
| signaling NaN, the invalid exception is raised.
*----------------------------------------------------------------------------*/
static uint_fast32_t softfloat_propagateNaNF32UI( uint_fast32_t uiA, uint_fast32_t uiB );

/*----------------------------------------------------------------------------
| The bit pattern for a default generated 64-bit floating-point NaN.
*----------------------------------------------------------------------------*/
#define defaultNaNF64UI UINT64_C( 0xFFF8000000000000 )

/*----------------------------------------------------------------------------
| Returns true when 64-bit unsigned integer `uiA' has the bit pattern of a
| 64-bit floating-point signaling NaN.
| Note:  This macro evaluates its argument more than once.
*----------------------------------------------------------------------------*/
#define softfloat_isSigNaNF64UI( uiA ) \
    ((((uiA) & UINT64_C( 0x7FF8000000000000 )) == UINT64_C( 0x7FF0000000000000 )) && \
      ((uiA) & UINT64_C( 0x0007FFFFFFFFFFFF )))

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 64-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
static void softfloat_f64UIToCommonNaN( uint_fast64_t uiA, struct commonNaN *zPtr );

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 64-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
static uint_fast64_t softfloat_commonNaNToF64UI( const struct commonNaN *aPtr );

/*----------------------------------------------------------------------------
| Interpreting `uiA' and `uiB' as the bit patterns of two 64-bit floating-
| point values, at least one of which is a NaN, returns the bit pattern of
| the combined NaN result.  If either `uiA' or `uiB' has the pattern of a
| signaling NaN, the invalid exception is raised.
*----------------------------------------------------------------------------*/
static uint_fast64_t softfloat_propagateNaNF64UI( uint_fast64_t uiA, uint_fast64_t uiB );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

#ifndef WORDS_BIGENDIAN
struct uint128 { uint64_t v0, v64; };
struct uint64_extra { uint64_t extra, v; };
struct uint128_extra { uint64_t extra; struct uint128 v; };
#else
struct uint128 { uint64_t v64, v0; };
struct uint64_extra { uint64_t v, extra; };
struct uint128_extra { struct uint128 v; uint64_t extra; };
#endif

/*----------------------------------------------------------------------------
| These macros are used to isolate the differences in word order between big-
| endian and little-endian platforms.
*----------------------------------------------------------------------------*/
#ifndef WORDS_BIGENDIAN
#define wordIncr 1
#define indexWord( total, n ) (n)
#define indexWordHi( total ) ((total) - 1)
#define indexWordLo( total ) 0
#define indexMultiword( total, m, n ) (n)
#define indexMultiwordHi( total, n ) ((total) - (n))
#define indexMultiwordLo( total, n ) 0
#define indexMultiwordHiBut( total, n ) (n)
#define indexMultiwordLoBut( total, n ) 0
#define INIT_UINTM4( v3, v2, v1, v0 ) { v0, v1, v2, v3 }
#else
#define wordIncr -1
#define indexWord( total, n ) ((total) - 1 - (n))
#define indexWordHi( total ) 0
#define indexWordLo( total ) ((total) - 1)
#define indexMultiword( total, m, n ) ((total) - 1 - (m))
#define indexMultiwordHi( total, n ) 0
#define indexMultiwordLo( total, n ) ((total) - (n))
#define indexMultiwordHiBut( total, n ) 0
#define indexMultiwordLoBut( total, n ) (n)
#define INIT_UINTM4( v3, v2, v1, v0 ) { v3, v2, v1, v0 }
#endif

enum {
    softfloat_mulAdd_subC    = 1,
    softfloat_mulAdd_subProd = 2
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
static uint_fast32_t softfloat_roundToUI32( bool, uint_fast64_t, uint_fast8_t, bool );
static uint_fast64_t softfloat_roundToUI64( bool, uint_fast64_t, uint_fast64_t, uint_fast8_t, bool );
static int_fast32_t softfloat_roundToI32( bool, uint_fast64_t, uint_fast8_t, bool );
static int_fast64_t softfloat_roundToI64( bool, uint_fast64_t, uint_fast64_t, uint_fast8_t, bool );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct exp16_sig32 { int_fast16_t exp; uint_fast32_t sig; };
static struct exp16_sig32 softfloat_normSubnormalF32Sig( uint_fast32_t );

static float32_t softfloat_roundPackToF32( bool, int_fast16_t, uint_fast32_t );
static float32_t softfloat_normRoundPackToF32( bool, int_fast16_t, uint_fast32_t );

static float32_t softfloat_addMagsF32( uint_fast32_t, uint_fast32_t );
static float32_t softfloat_subMagsF32( uint_fast32_t, uint_fast32_t );
static float32_t softfloat_mulAddF32(uint_fast32_t, uint_fast32_t, uint_fast32_t, uint_fast8_t );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct exp16_sig64 { int_fast16_t exp; uint_fast64_t sig; };
static struct exp16_sig64 softfloat_normSubnormalF64Sig( uint_fast64_t );

static float64_t softfloat_roundPackToF64( bool, int_fast16_t, uint_fast64_t );
static float64_t softfloat_normRoundPackToF64( bool, int_fast16_t, uint_fast64_t );

static float64_t softfloat_addMagsF64( uint_fast64_t, uint_fast64_t, bool );
static float64_t softfloat_subMagsF64( uint_fast64_t, uint_fast64_t, bool );
static float64_t softfloat_mulAddF64( uint_fast64_t, uint_fast64_t, uint_fast64_t, uint_fast8_t );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
| Shifts 'a' right by the number of bits given in 'dist', which must be in
| the range 1 to 63.  If any nonzero bits are shifted off, they are "jammed"
| into the least-significant bit of the shifted value by setting the least-
| significant bit to 1.  This shifted-and-jammed value is returned.
*----------------------------------------------------------------------------*/

inline uint64_t softfloat_shortShiftRightJam64( uint64_t a, uint_fast8_t dist )
{ return a>>dist | ((a & (((uint_fast64_t) 1<<dist) - 1)) != 0); }

/*----------------------------------------------------------------------------
| Shifts 'a' right by the number of bits given in 'dist', which must not
| be zero.  If any nonzero bits are shifted off, they are "jammed" into the
| least-significant bit of the shifted value by setting the least-significant
| bit to 1.  This shifted-and-jammed value is returned.
|   The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
| greater than 32, the result will be either 0 or 1, depending on whether 'a'
| is zero or nonzero.
*----------------------------------------------------------------------------*/

inline uint32_t softfloat_shiftRightJam32( uint32_t a, uint_fast16_t dist )
{
    //fixed unsigned unary minus: -x == ~x + 1
    return (dist < 31) ? a>>dist | ((uint32_t) (a<<((~dist + 1) & 31)) != 0) : (a != 0);
}

/*----------------------------------------------------------------------------
| Shifts 'a' right by the number of bits given in 'dist', which must not
| be zero.  If any nonzero bits are shifted off, they are "jammed" into the
| least-significant bit of the shifted value by setting the least-significant
| bit to 1.  This shifted-and-jammed value is returned.
|   The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
| greater than 64, the result will be either 0 or 1, depending on whether 'a'
| is zero or nonzero.
*----------------------------------------------------------------------------*/
inline uint64_t softfloat_shiftRightJam64( uint64_t a, uint_fast32_t dist )
{
    //fixed unsigned unary minus: -x == ~x + 1
    return (dist < 63) ? a>>dist | ((uint64_t) (a<<((~dist + 1) & 63)) != 0) : (a != 0);
}

/*----------------------------------------------------------------------------
| A constant table that translates an 8-bit unsigned integer (the array index)
| into the number of leading 0 bits before the most-significant 1 of that
| integer.  For integer zero (index 0), the corresponding table element is 8.
*----------------------------------------------------------------------------*/
static const uint_least8_t softfloat_countLeadingZeros8[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/*----------------------------------------------------------------------------
| Returns the number of leading 0 bits before the most-significant 1 bit of
| 'a'.  If 'a' is zero, 32 is returned.
*----------------------------------------------------------------------------*/
inline uint_fast8_t softfloat_countLeadingZeros32( uint32_t a )
{
    uint_fast8_t count = 0;
    if ( a < 0x10000 ) {
        count = 16;
        a <<= 16;
    }
    if ( a < 0x1000000 ) {
        count += 8;
        a <<= 8;
    }
    count += softfloat_countLeadingZeros8[a>>24];
    return count;
}

/*----------------------------------------------------------------------------
| Returns the number of leading 0 bits before the most-significant 1 bit of
| 'a'.  If 'a' is zero, 64 is returned.
*----------------------------------------------------------------------------*/
static uint_fast8_t softfloat_countLeadingZeros64( uint64_t a );

/*----------------------------------------------------------------------------
| Returns an approximation to the reciprocal of the number represented by 'a',
| where 'a' is interpreted as an unsigned fixed-point number with one integer
| bit and 31 fraction bits.  The 'a' input must be "normalized", meaning that
| its most-significant bit (bit 31) must be 1.  Thus, if A is the value of
| the fixed-point interpretation of 'a', then 1 <= A < 2.  The returned value
| is interpreted as a pure unsigned fraction, having no integer bits and 32
| fraction bits.  The approximation returned is never greater than the true
| reciprocal 1/A, and it differs from the true reciprocal by at most 2.006 ulp
| (units in the last place).
*----------------------------------------------------------------------------*/
#define softfloat_approxRecip32_1( a ) ((uint32_t) (UINT64_C( 0x7FFFFFFFFFFFFFFF ) / (uint32_t) (a)))

/*----------------------------------------------------------------------------
| Returns an approximation to the reciprocal of the square root of the number
| represented by 'a', where 'a' is interpreted as an unsigned fixed-point
| number either with one integer bit and 31 fraction bits or with two integer
| bits and 30 fraction bits.  The format of 'a' is determined by 'oddExpA',
| which must be either 0 or 1.  If 'oddExpA' is 1, 'a' is interpreted as
| having one integer bit, and if 'oddExpA' is 0, 'a' is interpreted as having
| two integer bits.  The 'a' input must be "normalized", meaning that its
| most-significant bit (bit 31) must be 1.  Thus, if A is the value of the
| fixed-point interpretation of 'a', it follows that 1 <= A < 2 when 'oddExpA'
| is 1, and 2 <= A < 4 when 'oddExpA' is 0.
|   The returned value is interpreted as a pure unsigned fraction, having
| no integer bits and 32 fraction bits.  The approximation returned is never
| greater than the true reciprocal 1/sqrt(A), and it differs from the true
| reciprocal by at most 2.06 ulp (units in the last place).  The approximation
| returned is also always within the range 0.5 to 1; thus, the most-
| significant bit of the result is always set.
*----------------------------------------------------------------------------*/
static uint32_t softfloat_approxRecipSqrt32_1( unsigned int oddExpA, uint32_t a );

static const uint16_t softfloat_approxRecipSqrt_1k0s[16] = {
    0xB4C9, 0xFFAB, 0xAA7D, 0xF11C, 0xA1C5, 0xE4C7, 0x9A43, 0xDA29,
    0x93B5, 0xD0E5, 0x8DED, 0xC8B7, 0x88C6, 0xC16D, 0x8424, 0xBAE1
};
static const uint16_t softfloat_approxRecipSqrt_1k1s[16] = {
    0xA5A5, 0xEA42, 0x8C21, 0xC62D, 0x788F, 0xAA7F, 0x6928, 0x94B6,
    0x5CC7, 0x8335, 0x52A6, 0x74E2, 0x4A3E, 0x68FE, 0x432B, 0x5EFD
};

/*----------------------------------------------------------------------------
| Shifts the 128 bits formed by concatenating 'a64' and 'a0' left by the
| number of bits given in 'dist', which must be in the range 1 to 63.
*----------------------------------------------------------------------------*/
inline struct uint128 softfloat_shortShiftLeft128( uint64_t a64, uint64_t a0, uint_fast8_t dist )
{
    struct uint128 z;
    z.v64 = a64<<dist | a0>>(-dist & 63);
    z.v0 = a0<<dist;
    return z;
}

/*----------------------------------------------------------------------------
| Shifts the 128 bits formed by concatenating 'a64' and 'a0' right by the
| number of bits given in 'dist', which must be in the range 1 to 63.  If any
| nonzero bits are shifted off, they are "jammed" into the least-significant
| bit of the shifted value by setting the least-significant bit to 1.  This
| shifted-and-jammed value is returned.
*----------------------------------------------------------------------------*/
inline struct uint128 softfloat_shortShiftRightJam128(uint64_t a64, uint64_t a0, uint_fast8_t dist )
{
    uint_fast8_t negDist = -dist;
    struct uint128 z;
    z.v64 = a64>>dist;
    z.v0 =
        a64<<(negDist & 63) | a0>>dist
            | ((uint64_t) (a0<<(negDist & 63)) != 0);
    return z;
}

/*----------------------------------------------------------------------------
| Shifts the 128 bits formed by concatenating 'a' and 'extra' right by 64
| _plus_ the number of bits given in 'dist', which must not be zero.  This
| shifted value is at most 64 nonzero bits and is returned in the 'v' field
| of the 'struct uint64_extra' result.  The 64-bit 'extra' field of the result
| contains a value formed as follows from the bits that were shifted off:  The
| _last_ bit shifted off is the most-significant bit of the 'extra' field, and
| the other 63 bits of the 'extra' field are all zero if and only if _all_but_
| _the_last_ bits shifted off were all zero.
|   (This function makes more sense if 'a' and 'extra' are considered to form
| an unsigned fixed-point number with binary point between 'a' and 'extra'.
| This fixed-point value is shifted right by the number of bits given in
| 'dist', and the integer part of this shifted value is returned in the 'v'
| field of the result.  The fractional part of the shifted value is modified
| as described above and returned in the 'extra' field of the result.)
*----------------------------------------------------------------------------*/
inline struct uint64_extra softfloat_shiftRightJam64Extra(uint64_t a, uint64_t extra, uint_fast32_t dist )
{
    struct uint64_extra z;
    if ( dist < 64 ) {
        z.v = a>>dist;
        //fixed unsigned unary minus: -x == ~x + 1
        z.extra = a<<((~dist + 1) & 63);
    } else {
        z.v = 0;
        z.extra = (dist == 64) ? a : (a != 0);
    }
    z.extra |= (extra != 0);
    return z;
}

/*----------------------------------------------------------------------------
| Shifts the 128 bits formed by concatenating 'a64' and 'a0' right by the
| number of bits given in 'dist', which must not be zero.  If any nonzero bits
| are shifted off, they are "jammed" into the least-significant bit of the
| shifted value by setting the least-significant bit to 1.  This shifted-and-
| jammed value is returned.
|   The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
| greater than 128, the result will be either 0 or 1, depending on whether the
| original 128 bits are all zeros.
*----------------------------------------------------------------------------*/
static struct uint128 softfloat_shiftRightJam128( uint64_t a64, uint64_t a0, uint_fast32_t dist );

/*----------------------------------------------------------------------------
| Returns the sum of the 128-bit integer formed by concatenating 'a64' and
| 'a0' and the 128-bit integer formed by concatenating 'b64' and 'b0'.  The
| addition is modulo 2^128, so any carry out is lost.
*----------------------------------------------------------------------------*/
inline struct uint128 softfloat_add128( uint64_t a64, uint64_t a0, uint64_t b64, uint64_t b0 )
{
    struct uint128 z;
    z.v0 = a0 + b0;
    z.v64 = a64 + b64 + (z.v0 < a0);
    return z;
}

/*----------------------------------------------------------------------------
| Returns the difference of the 128-bit integer formed by concatenating 'a64'
| and 'a0' and the 128-bit integer formed by concatenating 'b64' and 'b0'.
| The subtraction is modulo 2^128, so any borrow out (carry out) is lost.
*----------------------------------------------------------------------------*/
inline struct uint128 softfloat_sub128( uint64_t a64, uint64_t a0, uint64_t b64, uint64_t b0 )
{
    struct uint128 z;
    z.v0 = a0 - b0;
    z.v64 = a64 - b64;
    z.v64 -= (a0 < b0);
    return z;
}

/*----------------------------------------------------------------------------
| Returns the 128-bit product of 'a' and 'b'.
*----------------------------------------------------------------------------*/
static struct uint128 softfloat_mul64To128( uint64_t a, uint64_t b );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

float32_t f32_add( float32_t a, float32_t b )
{
    uint_fast32_t uiA = a.v;
    uint_fast32_t uiB = b.v;

    if ( signF32UI( uiA ^ uiB ) ) {
        return softfloat_subMagsF32( uiA, uiB );
    } else {
        return softfloat_addMagsF32( uiA, uiB );
    }
}

float32_t f32_div( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast32_t sigA;
    uint_fast32_t uiB;
    bool signB;
    int_fast16_t expB;
    uint_fast32_t sigB;
    bool signZ;
    struct exp16_sig32 normExpSig;
    int_fast16_t expZ;
    uint_fast64_t sig64A;
    uint_fast32_t sigZ;
    uint_fast32_t uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF32UI( uiA );
    expA  = expF32UI( uiA );
    sigA  = fracF32UI( uiA );
    uiB = b.v;
    signB = signF32UI( uiB );
    expB  = expF32UI( uiB );
    sigB  = fracF32UI( uiB );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0xFF ) {
        if ( sigA ) goto propagateNaN;
        if ( expB == 0xFF ) {
            if ( sigB ) goto propagateNaN;
            goto invalid;
        }
        goto infinity;
    }
    if ( expB == 0xFF ) {
        if ( sigB ) goto propagateNaN;
        goto zero;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expB ) {
        if ( ! sigB ) {
            if ( ! (expA | sigA) ) goto invalid;
            raiseFlags( flag_infinite );
            goto infinity;
        }
        normExpSig = softfloat_normSubnormalF32Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    if ( ! expA ) {
        if ( ! sigA ) goto zero;
        normExpSig = softfloat_normSubnormalF32Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA - expB + 0x7E;
    sigA |= 0x00800000;
    sigB |= 0x00800000;
    if ( sigA < sigB ) {
        --expZ;
        sig64A = (uint_fast64_t) sigA<<31;
    } else {
        sig64A = (uint_fast64_t) sigA<<30;
    }
    sigZ = (uint_fast32_t)(sig64A / sigB); // fixed warning on type cast
    if ( ! (sigZ & 0x3F) ) sigZ |= ((uint_fast64_t) sigB * sigZ != sig64A);
    return softfloat_roundPackToF32( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF32UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF32UI;
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infinity:
    uiZ = packToF32UI( signZ, 0xFF, 0 );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zero:
    uiZ = packToF32UI( signZ, 0, 0 );
 uiZ:
    return float32_t::fromRaw(uiZ);
}

bool f32_eq( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF32UI( uiA ) || isNaNF32UI( uiB ) ) {
        if (
            softfloat_isSigNaNF32UI( uiA ) || softfloat_isSigNaNF32UI( uiB )
        ) {
            raiseFlags( flag_invalid );
        }
        return false;
    }
    return (uiA == uiB) || ! (uint32_t) ((uiA | uiB)<<1);
}

bool f32_eq_signaling( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF32UI( uiA ) || isNaNF32UI( uiB ) ) {
        raiseFlags( flag_invalid );
        return false;
    }
    return (uiA == uiB) || ! (uint32_t) ((uiA | uiB)<<1);
}

bool f32_isSignalingNaN( float32_t a )
{
    return softfloat_isSigNaNF32UI( a.v );
}

bool f32_le( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF32UI( uiA ) || isNaNF32UI( uiB ) ) {
        raiseFlags( flag_invalid );
        return false;
    }
    signA = signF32UI( uiA );
    signB = signF32UI( uiB );
    return
        (signA != signB) ? signA || ! (uint32_t) ((uiA | uiB)<<1)
            : (uiA == uiB) || (signA ^ (uiA < uiB));
}

bool f32_le_quiet( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF32UI( uiA ) || isNaNF32UI( uiB ) ) {
        if (
            softfloat_isSigNaNF32UI( uiA ) || softfloat_isSigNaNF32UI( uiB )
        ) {
            raiseFlags( flag_invalid );
        }
        return false;
    }
    signA = signF32UI( uiA );
    signB = signF32UI( uiB );
    return
        (signA != signB) ? signA || ! (uint32_t) ((uiA | uiB)<<1)
            : (uiA == uiB) || (signA ^ (uiA < uiB));
}

bool f32_lt( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF32UI( uiA ) || isNaNF32UI( uiB ) ) {
        raiseFlags( flag_invalid );
        return false;
    }
    signA = signF32UI( uiA );
    signB = signF32UI( uiB );
    return
        (signA != signB) ? signA && ((uint32_t) ((uiA | uiB)<<1) != 0)
            : (uiA != uiB) && (signA ^ (uiA < uiB));
}

bool f32_lt_quiet( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF32UI( uiA ) || isNaNF32UI( uiB ) ) {
        if (
            softfloat_isSigNaNF32UI( uiA ) || softfloat_isSigNaNF32UI( uiB )
        ) {
            raiseFlags( flag_invalid );
        }
        return false;
    }
    signA = signF32UI( uiA );
    signB = signF32UI( uiB );
    return
        (signA != signB) ? signA && ((uint32_t) ((uiA | uiB)<<1) != 0)
            : (uiA != uiB) && (signA ^ (uiA < uiB));
}

float32_t f32_mulAdd( float32_t a, float32_t b, float32_t c )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;
    uint_fast32_t uiC;

    uiA = a.v;
    uiB = b.v;
    uiC = c.v;
    return softfloat_mulAddF32( uiA, uiB, uiC, 0 );
}

float32_t f32_mul( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast32_t sigA;
    uint_fast32_t uiB;
    bool signB;
    int_fast16_t expB;
    uint_fast32_t sigB;
    bool signZ;
    uint_fast32_t magBits;
    struct exp16_sig32 normExpSig;
    int_fast16_t expZ;
    uint_fast32_t sigZ, uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF32UI( uiA );
    expA  = expF32UI( uiA );
    sigA  = fracF32UI( uiA );
    uiB = b.v;
    signB = signF32UI( uiB );
    expB  = expF32UI( uiB );
    sigB  = fracF32UI( uiB );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0xFF ) {
        if ( sigA || ((expB == 0xFF) && sigB) ) goto propagateNaN;
        magBits = expB | sigB;
        goto infArg;
    }
    if ( expB == 0xFF ) {
        if ( sigB ) goto propagateNaN;
        magBits = expA | sigA;
        goto infArg;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) goto zero;
        normExpSig = softfloat_normSubnormalF32Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    if ( ! expB ) {
        if ( ! sigB ) goto zero;
        normExpSig = softfloat_normSubnormalF32Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA + expB - 0x7F;
    sigA = (sigA | 0x00800000)<<7;
    sigB = (sigB | 0x00800000)<<8;
    sigZ = (uint_fast32_t)softfloat_shortShiftRightJam64( (uint_fast64_t) sigA * sigB, 32 ); //fixed warning on type cast
    if ( sigZ < 0x40000000 ) {
        --expZ;
        sigZ <<= 1;
    }
    return softfloat_roundPackToF32( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF32UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infArg:
    if ( ! magBits ) {
        raiseFlags( flag_invalid );
        uiZ = defaultNaNF32UI;
    } else {
        uiZ = packToF32UI( signZ, 0xFF, 0 );
    }
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zero:
    uiZ = packToF32UI( signZ, 0, 0 );
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float32_t f32_rem( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast32_t sigA;
    uint_fast32_t uiB;
    int_fast16_t expB;
    uint_fast32_t sigB;
    struct exp16_sig32 normExpSig;
    uint32_t rem;
    int_fast16_t expDiff;
    uint32_t q, recip32, altRem, meanRem;
    bool signRem;
    uint_fast32_t uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF32UI( uiA );
    expA  = expF32UI( uiA );
    sigA  = fracF32UI( uiA );
    uiB = b.v;
    expB = expF32UI( uiB );
    sigB = fracF32UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0xFF ) {
        if ( sigA || ((expB == 0xFF) && sigB) ) goto propagateNaN;
        goto invalid;
    }
    if ( expB == 0xFF ) {
        if ( sigB ) goto propagateNaN;
        return a;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expB ) {
        if ( ! sigB ) goto invalid;
        normExpSig = softfloat_normSubnormalF32Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    if ( ! expA ) {
        if ( ! sigA ) return a;
        normExpSig = softfloat_normSubnormalF32Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    rem = sigA | 0x00800000;
    sigB |= 0x00800000;
    expDiff = expA - expB;
    if ( expDiff < 1 ) {
        if ( expDiff < -1 ) return a;
        sigB <<= 6;
        if ( expDiff ) {
            rem <<= 5;
            q = 0;
        } else {
            rem <<= 6;
            q = (sigB <= rem);
            if ( q ) rem -= sigB;
        }
    } else {
        recip32 = softfloat_approxRecip32_1( sigB<<8 );
        /*--------------------------------------------------------------------
        | Changing the shift of `rem' here requires also changing the initial
        | subtraction from `expDiff'.
        *--------------------------------------------------------------------*/
        rem <<= 7;
        expDiff -= 31;
        /*--------------------------------------------------------------------
        | The scale of `sigB' affects how many bits are obtained during each
        | cycle of the loop.  Currently this is 29 bits per loop iteration,
        | which is believed to be the maximum possible.
        *--------------------------------------------------------------------*/
        sigB <<= 6;
        for (;;) {
            q = (rem * (uint_fast64_t) recip32)>>32;
            if ( expDiff < 0 ) break;
            //fixed unsigned unary minus: -x == ~x + 1
            rem = ~(q * (uint32_t) sigB) + 1;
            expDiff -= 29;
        }
        /*--------------------------------------------------------------------
        | (`expDiff' cannot be less than -30 here.)
        *--------------------------------------------------------------------*/
        q >>= ~expDiff & 31;
        rem = (rem<<(expDiff + 30)) - q * (uint32_t) sigB;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    do {
        altRem = rem;
        ++q;
        rem -= sigB;
    } while ( ! (rem & 0x80000000) );
    meanRem = rem + altRem;
    if ( (meanRem & 0x80000000) || (! meanRem && (q & 1)) ) rem = altRem;
    signRem = signA;
    if ( 0x80000000 <= rem ) {
        signRem = ! signRem;
        //fixed unsigned unary minus: -x == ~x + 1
        rem = ~rem + 1;
    }
    return softfloat_normRoundPackToF32( signRem, expB, rem );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF32UI( uiA, uiB );
    goto uiZ;
 invalid:
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF32UI;
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float32_t f32_roundToInt( float32_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast32_t uiA;
    int_fast16_t exp;
    uint_fast32_t uiZ, lastBitMask, roundBitsMask;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp <= 0x7E ) {
        if ( ! (uint32_t) (uiA<<1) ) return a;
        if ( exact ) raiseFlags(flag_inexact);
        uiZ = uiA & packToF32UI( 1, 0, 0 );
        switch ( roundingMode ) {
         case round_near_even:
            if ( ! fracF32UI( uiA ) ) break;
         case round_near_maxMag:
            if ( exp == 0x7E ) uiZ |= packToF32UI( 0, 0x7F, 0 );
            break;
         case round_min:
            if ( uiZ ) uiZ = packToF32UI( 1, 0x7F, 0 );
            break;
         case round_max:
            if ( ! uiZ ) uiZ = packToF32UI( 0, 0x7F, 0 );
            break;
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( 0x96 <= exp ) {
        if ( (exp == 0xFF) && fracF32UI( uiA ) ) {
            uiZ = softfloat_propagateNaNF32UI( uiA, 0 );
            goto uiZ;
        }
        return a;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiZ = uiA;
    lastBitMask = (uint_fast32_t) 1<<(0x96 - exp);
    roundBitsMask = lastBitMask - 1;
    if ( roundingMode == round_near_maxMag ) {
        uiZ += lastBitMask>>1;
    } else if ( roundingMode == round_near_even ) {
        uiZ += lastBitMask>>1;
        if ( ! (uiZ & roundBitsMask) ) uiZ &= ~lastBitMask;
    } else if (
        roundingMode
            == (signF32UI( uiZ ) ? round_min : round_max)
    ) {
        uiZ += roundBitsMask;
    }
    uiZ &= ~roundBitsMask;
    if ( exact && (uiZ != uiA) ) {
        raiseFlags(flag_inexact);
    }
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float32_t f32_sqrt( float32_t a )
{
    uint_fast32_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast32_t sigA, uiZ;
    struct exp16_sig32 normExpSig;
    int_fast16_t expZ;
    uint_fast32_t sigZ, shiftedSigZ;
    uint32_t negRem;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF32UI( uiA );
    expA  = expF32UI( uiA );
    sigA  = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0xFF ) {
        if ( sigA ) {
            uiZ = softfloat_propagateNaNF32UI( uiA, 0 );
            goto uiZ;
        }
        if ( ! signA ) return a;
        goto invalid;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( signA ) {
        if ( ! (expA | sigA) ) return a;
        goto invalid;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) return a;
        normExpSig = softfloat_normSubnormalF32Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = ((expA - 0x7F)>>1) + 0x7E;
    expA &= 1;
    sigA = (sigA | 0x00800000)<<8;
    sigZ =
        ((uint_fast64_t) sigA * softfloat_approxRecipSqrt32_1( expA, sigA ))
            >>32;
    if ( expA ) sigZ >>= 1;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sigZ += 2;
    if ( (sigZ & 0x3F) < 2 ) {
        shiftedSigZ = sigZ>>2;
        negRem = shiftedSigZ * shiftedSigZ;
        sigZ &= ~3;
        if ( negRem & 0x80000000 ) {
            sigZ |= 1;
        } else {
            if ( negRem ) --sigZ;
        }
    }
    return softfloat_roundPackToF32( 0, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF32UI;
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float32_t f32_sub( float32_t a, float32_t b )
{
    uint_fast32_t uiA;
    uint_fast32_t uiB;

    uiA = a.v;
    uiB = b.v;
    if ( signF32UI( uiA ^ uiB ) ) {
        return softfloat_addMagsF32( uiA, uiB );
    } else {
        return softfloat_subMagsF32( uiA, uiB );
    }
}

float64_t f32_to_f64( float32_t a )
{
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t frac;
    struct commonNaN commonNaN;
    uint_fast64_t uiZ;
    struct exp16_sig32 normExpSig;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    frac = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp == 0xFF ) {
        if ( frac ) {
            softfloat_f32UIToCommonNaN( uiA, &commonNaN );
            uiZ = softfloat_commonNaNToF64UI( &commonNaN );
        } else {
            uiZ = packToF64UI( sign, 0x7FF, 0 );
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! exp ) {
        if ( ! frac ) {
            uiZ = packToF64UI( sign, 0, 0 );
            goto uiZ;
        }
        normExpSig = softfloat_normSubnormalF32Sig( frac );
        exp = normExpSig.exp - 1;
        frac = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiZ = packToF64UI( sign, exp + 0x380, (uint_fast64_t) frac<<29 );
 uiZ:
    return float64_t::fromRaw(uiZ);
}

int_fast32_t f32_to_i32( float32_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t sig;
    uint_fast64_t sig64;
    int_fast16_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    sig  = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#if (i32_fromNaN != i32_fromPosOverflow) || (i32_fromNaN != i32_fromNegOverflow)
    if ( (exp == 0xFF) && sig ) {
#if (i32_fromNaN == i32_fromPosOverflow)
        sign = 0;
#elif (i32_fromNaN == i32_fromNegOverflow)
        sign = 1;
#else
        raiseFlags( flag_invalid );
        return i32_fromNaN;
#endif
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<32;
    shiftDist = 0xAA - exp;
    if ( 0 < shiftDist ) sig64 = softfloat_shiftRightJam64( sig64, shiftDist );
    return softfloat_roundToI32( sign, sig64, roundingMode, exact );
}

int_fast32_t f32_to_i32_r_minMag( float32_t a, bool exact )
{
    uint_fast32_t uiA;
    int_fast16_t exp;
    uint_fast32_t sig;
    int_fast16_t shiftDist;
    bool sign;
    int_fast32_t absZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF32UI( uiA );
    sig = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0x9E - exp;
    if ( 32 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF32UI( uiA );
    if ( shiftDist <= 0 ) {
        if ( uiA == packToF32UI( 1, 0x9E, 0 ) ) return -0x7FFFFFFF - 1;
        raiseFlags( flag_invalid );
        return
            (exp == 0xFF) && sig ? i32_fromNaN
                : sign ? i32_fromNegOverflow : i32_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig = (sig | 0x00800000)<<8;
    absZ = sig>>shiftDist;
    if ( exact && ((uint_fast32_t) absZ<<shiftDist != sig) ) {
        raiseFlags(flag_inexact);
    }
    return sign ? -absZ : absZ;
}

int_fast64_t f32_to_i64( float32_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t sig;
    int_fast16_t shiftDist;
    uint_fast64_t sig64, extra;
    struct uint64_extra sig64Extra;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    sig  = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0xBE - exp;
    if ( shiftDist < 0 ) {
        raiseFlags( flag_invalid );
        return
            (exp == 0xFF) && sig ? i64_fromNaN
                : sign ? i64_fromNegOverflow : i64_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<40;
    extra = 0;
    if ( shiftDist ) {
        sig64Extra = softfloat_shiftRightJam64Extra( sig64, 0, shiftDist );
        sig64 = sig64Extra.v;
        extra = sig64Extra.extra;
    }
    return softfloat_roundToI64( sign, sig64, extra, roundingMode, exact );
}

int_fast64_t f32_to_i64_r_minMag( float32_t a, bool exact )
{
    uint_fast32_t uiA;
    int_fast16_t exp;
    uint_fast32_t sig;
    int_fast16_t shiftDist;
    bool sign;
    uint_fast64_t sig64;
    int_fast64_t absZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF32UI( uiA );
    sig = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0xBE - exp;
    if ( 64 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF32UI( uiA );
    if ( shiftDist <= 0 ) {
        if ( uiA == packToF32UI( 1, 0xBE, 0 ) ) {
            return -INT64_C( 0x7FFFFFFFFFFFFFFF ) - 1;
        }
        raiseFlags( flag_invalid );
        return
            (exp == 0xFF) && sig ? i64_fromNaN
                : sign ? i64_fromNegOverflow : i64_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<40;
    absZ = sig64>>shiftDist;
    shiftDist = 40 - shiftDist;
    if ( exact && (shiftDist < 0) && (uint32_t) (sig<<(shiftDist & 31)) ) {
        raiseFlags(flag_inexact);
    }
    return sign ? -absZ : absZ;
}

uint_fast32_t f32_to_ui32( float32_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t sig;
    uint_fast64_t sig64;
    int_fast16_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    sig  = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#if (ui32_fromNaN != ui32_fromPosOverflow) || (ui32_fromNaN != ui32_fromNegOverflow)
    if ( (exp == 0xFF) && sig ) {
#if (ui32_fromNaN == ui32_fromPosOverflow)
        sign = 0;
#elif (ui32_fromNaN == ui32_fromNegOverflow)
        sign = 1;
#else
        raiseFlags( flag_invalid );
        return ui32_fromNaN;
#endif
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<32;
    shiftDist = 0xAA - exp;
    if ( 0 < shiftDist ) sig64 = softfloat_shiftRightJam64( sig64, shiftDist );
    return softfloat_roundToUI32( sign, sig64, roundingMode, exact );
}

uint_fast32_t f32_to_ui32_r_minMag( float32_t a, bool exact )
{
    uint_fast32_t uiA;
    int_fast16_t exp;
    uint_fast32_t sig;
    int_fast16_t shiftDist;
    bool sign;
    uint_fast32_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF32UI( uiA );
    sig = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0x9E - exp;
    if ( 32 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF32UI( uiA );
    if ( sign || (shiftDist < 0) ) {
        raiseFlags( flag_invalid );
        return
            (exp == 0xFF) && sig ? ui32_fromNaN
                : sign ? ui32_fromNegOverflow : ui32_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig = (sig | 0x00800000)<<8;
    z = sig>>shiftDist;
    if ( exact && (z<<shiftDist != sig) ) {
        raiseFlags(flag_inexact);
    }
    return z;
}

uint_fast64_t f32_to_ui64( float32_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t sig;
    int_fast16_t shiftDist;
    uint_fast64_t sig64, extra;
    struct uint64_extra sig64Extra;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    sig  = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0xBE - exp;
    if ( shiftDist < 0 ) {
        raiseFlags( flag_invalid );
        return
            (exp == 0xFF) && sig ? ui64_fromNaN
                : sign ? ui64_fromNegOverflow : ui64_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<40;
    extra = 0;
    if ( shiftDist ) {
        sig64Extra = softfloat_shiftRightJam64Extra( sig64, 0, shiftDist );
        sig64 = sig64Extra.v;
        extra = sig64Extra.extra;
    }
    return softfloat_roundToUI64( sign, sig64, extra, roundingMode, exact );
}

uint_fast64_t f32_to_ui64_r_minMag( float32_t a, bool exact )
{
    uint_fast32_t uiA;
    int_fast16_t exp;
    uint_fast32_t sig;
    int_fast16_t shiftDist;
    bool sign;
    uint_fast64_t sig64, z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF32UI( uiA );
    sig = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0xBE - exp;
    if ( 64 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF32UI( uiA );
    if ( sign || (shiftDist < 0) ) {
        raiseFlags( flag_invalid );
        return
            (exp == 0xFF) && sig ? ui64_fromNaN
                : sign ? ui64_fromNegOverflow : ui64_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<40;
    z = sig64>>shiftDist;
    shiftDist = 40 - shiftDist;
    if ( exact && (shiftDist < 0) && (uint32_t) (sig<<(shiftDist & 31)) ) {
        raiseFlags(flag_inexact);
    }
    return z;
}

float64_t f64_add( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    bool signA;
    uint_fast64_t uiB;
    bool signB;

    uiA = a.v;
    signA = signF64UI( uiA );
    uiB = b.v;
    signB = signF64UI( uiB );
    if ( signA == signB ) {
        return softfloat_addMagsF64( uiA, uiB, signA );
    } else {
        return softfloat_subMagsF64( uiA, uiB, signA );
    }
}

float64_t f64_div( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast64_t sigA;
    uint_fast64_t uiB;
    bool signB;
    int_fast16_t expB;
    uint_fast64_t sigB;
    bool signZ;
    struct exp16_sig64 normExpSig;
    int_fast16_t expZ;
    uint32_t recip32, sig32Z, doubleTerm;
    uint_fast64_t rem;
    uint32_t q;
    uint_fast64_t sigZ;
    uint_fast64_t uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF64UI( uiA );
    expA  = expF64UI( uiA );
    sigA  = fracF64UI( uiA );
    uiB = b.v;
    signB = signF64UI( uiB );
    expB  = expF64UI( uiB );
    sigB  = fracF64UI( uiB );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x7FF ) {
        if ( sigA ) goto propagateNaN;
        if ( expB == 0x7FF ) {
            if ( sigB ) goto propagateNaN;
            goto invalid;
        }
        goto infinity;
    }
    if ( expB == 0x7FF ) {
        if ( sigB ) goto propagateNaN;
        goto zero;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expB ) {
        if ( ! sigB ) {
            if ( ! (expA | sigA) ) goto invalid;
            raiseFlags( flag_infinite );
            goto infinity;
        }
        normExpSig = softfloat_normSubnormalF64Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    if ( ! expA ) {
        if ( ! sigA ) goto zero;
        normExpSig = softfloat_normSubnormalF64Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA - expB + 0x3FE;
    sigA |= UINT64_C( 0x0010000000000000 );
    sigB |= UINT64_C( 0x0010000000000000 );
    if ( sigA < sigB ) {
        --expZ;
        sigA <<= 11;
    } else {
        sigA <<= 10;
    }
    sigB <<= 11;
    recip32 = softfloat_approxRecip32_1( sigB>>32 ) - 2;
    sig32Z = ((uint32_t) (sigA>>32) * (uint_fast64_t) recip32)>>32;
    doubleTerm = sig32Z<<1;
    rem =
        ((sigA - (uint_fast64_t) doubleTerm * (uint32_t) (sigB>>32))<<28)
            - (uint_fast64_t) doubleTerm * ((uint32_t) sigB>>4);
    q = (((uint32_t) (rem>>32) * (uint_fast64_t) recip32)>>32) + 4;
    sigZ = ((uint_fast64_t) sig32Z<<32) + ((uint_fast64_t) q<<4);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( (sigZ & 0x1FF) < 4<<4 ) {
        q &= ~7;
        sigZ &= ~(uint_fast64_t) 0x7F;
        doubleTerm = q<<1;
        rem =
            ((rem - (uint_fast64_t) doubleTerm * (uint32_t) (sigB>>32))<<28)
                - (uint_fast64_t) doubleTerm * ((uint32_t) sigB>>4);
        if ( rem & UINT64_C( 0x8000000000000000 ) ) {
            sigZ -= 1<<7;
        } else {
            if ( rem ) sigZ |= 1;
        }
    }
    return softfloat_roundPackToF64( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF64UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF64UI;
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infinity:
    uiZ = packToF64UI( signZ, 0x7FF, 0 );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zero:
    uiZ = packToF64UI( signZ, 0, 0 );
 uiZ:
    return float64_t::fromRaw(uiZ);
}

bool f64_eq( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF64UI( uiA ) || isNaNF64UI( uiB ) ) {
        if (
            softfloat_isSigNaNF64UI( uiA ) || softfloat_isSigNaNF64UI( uiB )
        ) {
            raiseFlags( flag_invalid );
        }
        return false;
    }
    return (uiA == uiB) || ! ((uiA | uiB) & UINT64_C( 0x7FFFFFFFFFFFFFFF ));
}

bool f64_eq_signaling( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF64UI( uiA ) || isNaNF64UI( uiB ) ) {
        raiseFlags( flag_invalid );
        return false;
    }
    return (uiA == uiB) || ! ((uiA | uiB) & UINT64_C( 0x7FFFFFFFFFFFFFFF ));
}

bool f64_isSignalingNaN( float64_t a )
{
    return softfloat_isSigNaNF64UI( a.v );
}

bool f64_le( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF64UI( uiA ) || isNaNF64UI( uiB ) ) {
        raiseFlags( flag_invalid );
        return false;
    }
    signA = signF64UI( uiA );
    signB = signF64UI( uiB );
    return
        (signA != signB)
            ? signA || ! ((uiA | uiB) & UINT64_C( 0x7FFFFFFFFFFFFFFF ))
            : (uiA == uiB) || (signA ^ (uiA < uiB));
}

bool f64_le_quiet( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF64UI( uiA ) || isNaNF64UI( uiB ) ) {
        if (
            softfloat_isSigNaNF64UI( uiA ) || softfloat_isSigNaNF64UI( uiB )
        ) {
            raiseFlags( flag_invalid );
        }
        return false;
    }
    signA = signF64UI( uiA );
    signB = signF64UI( uiB );
    return
        (signA != signB)
            ? signA || ! ((uiA | uiB) & UINT64_C( 0x7FFFFFFFFFFFFFFF ))
            : (uiA == uiB) || (signA ^ (uiA < uiB));
}

bool f64_lt( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF64UI( uiA ) || isNaNF64UI( uiB ) ) {
        raiseFlags( flag_invalid );
        return false;
    }
    signA = signF64UI( uiA );
    signB = signF64UI( uiB );
    return
        (signA != signB)
            ? signA && ((uiA | uiB) & UINT64_C( 0x7FFFFFFFFFFFFFFF ))
            : (uiA != uiB) && (signA ^ (uiA < uiB));
}

bool f64_lt_quiet( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;
    bool signA, signB;

    uiA = a.v;
    uiB = b.v;
    if ( isNaNF64UI( uiA ) || isNaNF64UI( uiB ) ) {
        if (
            softfloat_isSigNaNF64UI( uiA ) || softfloat_isSigNaNF64UI( uiB )
        ) {
            raiseFlags( flag_invalid );
        }
        return false;
    }
    signA = signF64UI( uiA );
    signB = signF64UI( uiB );
    return
        (signA != signB)
            ? signA && ((uiA | uiB) & UINT64_C( 0x7FFFFFFFFFFFFFFF ))
            : (uiA != uiB) && (signA ^ (uiA < uiB));
}

float64_t f64_mulAdd( float64_t a, float64_t b, float64_t c )
{
    uint_fast64_t uiA;
    uint_fast64_t uiB;
    uint_fast64_t uiC;

    uiA = a.v;
    uiB = b.v;
    uiC = c.v;
    return softfloat_mulAddF64( uiA, uiB, uiC, 0 );
}

float64_t f64_mul( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast64_t sigA;
    uint_fast64_t uiB;
    bool signB;
    int_fast16_t expB;
    uint_fast64_t sigB;
    bool signZ;
    uint_fast64_t magBits;
    struct exp16_sig64 normExpSig;
    int_fast16_t expZ;
    struct uint128 sig128Z;
    uint_fast64_t sigZ, uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF64UI( uiA );
    expA  = expF64UI( uiA );
    sigA  = fracF64UI( uiA );
    uiB = b.v;
    signB = signF64UI( uiB );
    expB  = expF64UI( uiB );
    sigB  = fracF64UI( uiB );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x7FF ) {
        if ( sigA || ((expB == 0x7FF) && sigB) ) goto propagateNaN;
        magBits = expB | sigB;
        goto infArg;
    }
    if ( expB == 0x7FF ) {
        if ( sigB ) goto propagateNaN;
        magBits = expA | sigA;
        goto infArg;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) goto zero;
        normExpSig = softfloat_normSubnormalF64Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    if ( ! expB ) {
        if ( ! sigB ) goto zero;
        normExpSig = softfloat_normSubnormalF64Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA + expB - 0x3FF;
    sigA = (sigA | UINT64_C( 0x0010000000000000 ))<<10;
    sigB = (sigB | UINT64_C( 0x0010000000000000 ))<<11;
    sig128Z = softfloat_mul64To128( sigA, sigB );
    sigZ = sig128Z.v64 | (sig128Z.v0 != 0);

    if ( sigZ < UINT64_C( 0x4000000000000000 ) ) {
        --expZ;
        sigZ <<= 1;
    }
    return softfloat_roundPackToF64( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF64UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infArg:
    if ( ! magBits ) {
        raiseFlags( flag_invalid );
        uiZ = defaultNaNF64UI;
    } else {
        uiZ = packToF64UI( signZ, 0x7FF, 0 );
    }
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zero:
    uiZ = packToF64UI( signZ, 0, 0 );
 uiZ:
    return float64_t::fromRaw(uiZ);
}

float64_t f64_rem( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast64_t sigA;
    uint_fast64_t uiB;
    int_fast16_t expB;
    uint_fast64_t sigB;
    struct exp16_sig64 normExpSig;
    uint64_t rem;
    int_fast16_t expDiff;
    uint32_t q, recip32;
    uint_fast64_t q64;
    uint64_t altRem, meanRem;
    bool signRem;
    uint_fast64_t uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF64UI( uiA );
    expA  = expF64UI( uiA );
    sigA  = fracF64UI( uiA );
    uiB = b.v;
    expB = expF64UI( uiB );
    sigB = fracF64UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x7FF ) {
        if ( sigA || ((expB == 0x7FF) && sigB) ) goto propagateNaN;
        goto invalid;
    }
    if ( expB == 0x7FF ) {
        if ( sigB ) goto propagateNaN;
        return a;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA < expB - 1 ) return a;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expB ) {
        if ( ! sigB ) goto invalid;
        normExpSig = softfloat_normSubnormalF64Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    if ( ! expA ) {
        if ( ! sigA ) return a;
        normExpSig = softfloat_normSubnormalF64Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    rem = sigA | UINT64_C( 0x0010000000000000 );
    sigB |= UINT64_C( 0x0010000000000000 );
    expDiff = expA - expB;
    if ( expDiff < 1 ) {
        if ( expDiff < -1 ) return a;
        sigB <<= 9;
        if ( expDiff ) {
            rem <<= 8;
            q = 0;
        } else {
            rem <<= 9;
            q = (sigB <= rem);
            if ( q ) rem -= sigB;
        }
    } else {
        recip32 = softfloat_approxRecip32_1( sigB>>21 );
        /*--------------------------------------------------------------------
        | Changing the shift of `rem' here requires also changing the initial
        | subtraction from `expDiff'.
        *--------------------------------------------------------------------*/
        rem <<= 9;
        expDiff -= 30;
        /*--------------------------------------------------------------------
        | The scale of `sigB' affects how many bits are obtained during each
        | cycle of the loop.  Currently this is 29 bits per loop iteration,
        | the maximum possible.
        *--------------------------------------------------------------------*/
        sigB <<= 9;
        for (;;) {
            q64 = (uint32_t) (rem>>32) * (uint_fast64_t) recip32;
            if ( expDiff < 0 ) break;
            q = (q64 + 0x80000000)>>32;
            rem <<= 29;
            rem -= q * (uint64_t) sigB;
            if ( rem & UINT64_C( 0x8000000000000000 ) ) rem += sigB;
            expDiff -= 29;
        }
        /*--------------------------------------------------------------------
        | (`expDiff' cannot be less than -29 here.)
        *--------------------------------------------------------------------*/
        q = (uint32_t) (q64>>32)>>(~expDiff & 31);
        rem = (rem<<(expDiff + 30)) - q * (uint64_t) sigB;
        if ( rem & UINT64_C( 0x8000000000000000 ) ) {
            altRem = rem + sigB;
            goto selectRem;
        }
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    do {
        altRem = rem;
        ++q;
        rem -= sigB;
    } while ( ! (rem & UINT64_C( 0x8000000000000000 )) );
 selectRem:
    meanRem = rem + altRem;
    if (
        (meanRem & UINT64_C( 0x8000000000000000 )) || (! meanRem && (q & 1))
    ) {
        rem = altRem;
    }
    signRem = signA;
    if ( rem & UINT64_C( 0x8000000000000000 ) ) {
        signRem = ! signRem;
        //fixed unsigned unary minus: -x == ~x + 1
        rem = ~rem + 1;
    }
    return softfloat_normRoundPackToF64( signRem, expB, rem );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF64UI( uiA, uiB );
    goto uiZ;
 invalid:
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF64UI;
 uiZ:
    return float64_t::fromRaw(uiZ);
}

float64_t f64_roundToInt( float64_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast64_t uiA;
    int_fast16_t exp;
    uint_fast64_t uiZ, lastBitMask, roundBitsMask;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp <= 0x3FE ) {
        if ( ! (uiA & UINT64_C( 0x7FFFFFFFFFFFFFFF )) ) return a;
        if ( exact ) raiseFlags(flag_inexact);
        uiZ = uiA & packToF64UI( 1, 0, 0 );
        switch ( roundingMode ) {
         case round_near_even:
            if ( ! fracF64UI( uiA ) ) break;
         case round_near_maxMag:
            if ( exp == 0x3FE ) uiZ |= packToF64UI( 0, 0x3FF, 0 );
            break;
         case round_min:
            if ( uiZ ) uiZ = packToF64UI( 1, 0x3FF, 0 );
            break;
         case round_max:
            if ( ! uiZ ) uiZ = packToF64UI( 0, 0x3FF, 0 );
            break;
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( 0x433 <= exp ) {
        if ( (exp == 0x7FF) && fracF64UI( uiA ) ) {
            uiZ = softfloat_propagateNaNF64UI( uiA, 0 );
            goto uiZ;
        }
        return a;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiZ = uiA;
    lastBitMask = (uint_fast64_t) 1<<(0x433 - exp);
    roundBitsMask = lastBitMask - 1;
    if ( roundingMode == round_near_maxMag ) {
        uiZ += lastBitMask>>1;
    } else if ( roundingMode == round_near_even ) {
        uiZ += lastBitMask>>1;
        if ( ! (uiZ & roundBitsMask) ) uiZ &= ~lastBitMask;
    } else if (
        roundingMode
            == (signF64UI( uiZ ) ? round_min : round_max)
    ) {
        uiZ += roundBitsMask;
    }
    uiZ &= ~roundBitsMask;
    if ( exact && (uiZ != uiA) ) {
        raiseFlags(flag_inexact);
    }
 uiZ:
    return float64_t::fromRaw(uiZ);
}

float64_t f64_sqrt( float64_t a )
{
    uint_fast64_t uiA;
    bool signA;
    int_fast16_t expA;
    uint_fast64_t sigA, uiZ;
    struct exp16_sig64 normExpSig;
    int_fast16_t expZ;
    uint32_t sig32A, recipSqrt32, sig32Z;
    uint_fast64_t rem;
    uint32_t q;
    uint_fast64_t sigZ, shiftedSigZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    signA = signF64UI( uiA );
    expA  = expF64UI( uiA );
    sigA  = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x7FF ) {
        if ( sigA ) {
            uiZ = softfloat_propagateNaNF64UI( uiA, 0 );
            goto uiZ;
        }
        if ( ! signA ) return a;
        goto invalid;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( signA ) {
        if ( ! (expA | sigA) ) return a;
        goto invalid;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) return a;
        normExpSig = softfloat_normSubnormalF64Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    | (`sig32Z' is guaranteed to be a lower bound on the square root of
    | `sig32A', which makes `sig32Z' also a lower bound on the square root of
    | `sigA'.)
    *------------------------------------------------------------------------*/
    expZ = ((expA - 0x3FF)>>1) + 0x3FE;
    expA &= 1;
    sigA |= UINT64_C( 0x0010000000000000 );
    sig32A = (uint32_t)(sigA>>21); //fixed warning on type cast
    recipSqrt32 = softfloat_approxRecipSqrt32_1( expA, sig32A );
    sig32Z = ((uint_fast64_t) sig32A * recipSqrt32)>>32;
    if ( expA ) {
        sigA <<= 8;
        sig32Z >>= 1;
    } else {
        sigA <<= 9;
    }
    rem = sigA - (uint_fast64_t) sig32Z * sig32Z;
    q = ((uint32_t) (rem>>2) * (uint_fast64_t) recipSqrt32)>>32;
    sigZ = ((uint_fast64_t) sig32Z<<32 | 1<<5) + ((uint_fast64_t) q<<3);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( (sigZ & 0x1FF) < 1<<5 ) {
        sigZ &= ~(uint_fast64_t) 0x3F;
        shiftedSigZ = sigZ>>6;
        rem = (sigA<<52) - shiftedSigZ * shiftedSigZ;
        if ( rem & UINT64_C( 0x8000000000000000 ) ) {
            --sigZ;
        } else {
            if ( rem ) sigZ |= 1;
        }
    }
    return softfloat_roundPackToF64( 0, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF64UI;
 uiZ:
    return float64_t::fromRaw(uiZ);
}

float64_t f64_sub( float64_t a, float64_t b )
{
    uint_fast64_t uiA;
    bool signA;
    uint_fast64_t uiB;
    bool signB;

    uiA = a.v;
    signA = signF64UI( uiA );
    uiB = b.v;
    signB = signF64UI( uiB );

    if ( signA == signB ) {
        return softfloat_subMagsF64( uiA, uiB, signA );
    } else {
        return softfloat_addMagsF64( uiA, uiB, signA );
    }
}

float32_t f64_to_f32( float64_t a )
{
    uint_fast64_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast64_t frac;
    struct commonNaN commonNaN;
    uint_fast32_t uiZ, frac32;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF64UI( uiA );
    exp  = expF64UI( uiA );
    frac = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp == 0x7FF ) {
        if ( frac ) {
            softfloat_f64UIToCommonNaN( uiA, &commonNaN );
            uiZ = softfloat_commonNaNToF32UI( &commonNaN );
        } else {
            uiZ = packToF32UI( sign, 0xFF, 0 );
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    frac32 = (uint_fast32_t)softfloat_shortShiftRightJam64( frac, 22 ); //fixed warning on type cast
    if ( ! (exp | frac32) ) {
        uiZ = packToF32UI( sign, 0, 0 );
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    return softfloat_roundPackToF32( sign, exp - 0x381, frac32 | 0x40000000 );
 uiZ:
    return float32_t::fromRaw(uiZ);
}

int_fast32_t f64_to_i32( float64_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast64_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF64UI( uiA );
    exp  = expF64UI( uiA );
    sig  = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#if (i32_fromNaN != i32_fromPosOverflow) || (i32_fromNaN != i32_fromNegOverflow)
    if ( (exp == 0x7FF) && sig ) {
#if (i32_fromNaN == i32_fromPosOverflow)
        sign = 0;
#elif (i32_fromNaN == i32_fromNegOverflow)
        sign = 1;
#else
        raiseFlags( flag_invalid );
        return i32_fromNaN;
#endif
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= UINT64_C( 0x0010000000000000 );
    shiftDist = 0x427 - exp;
    if ( 0 < shiftDist ) sig = softfloat_shiftRightJam64( sig, shiftDist );
    return softfloat_roundToI32( sign, sig, roundingMode, exact );
}

int_fast32_t f64_to_i32_r_minMag( float64_t a, bool exact )
{
    uint_fast64_t uiA;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;
    bool sign;
    int_fast32_t absZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF64UI( uiA );
    sig = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0x433 - exp;
    if ( 53 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF64UI( uiA );
    if ( shiftDist < 22 ) {
        if (
            sign && (exp == 0x41E) && (sig < UINT64_C( 0x0000000000200000 ))
        ) {
            if ( exact && sig ) {
                raiseFlags(flag_inexact);
            }
            return -0x7FFFFFFF - 1;
        }
        raiseFlags( flag_invalid );
        return
            (exp == 0x7FF) && sig ? i32_fromNaN
                : sign ? i32_fromNegOverflow : i32_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig |= UINT64_C( 0x0010000000000000 );
    absZ = (int_fast32_t)(sig>>shiftDist); //fixed warning on type cast
    if ( exact && ((uint_fast64_t) (uint_fast32_t) absZ<<shiftDist != sig) ) {
        raiseFlags(flag_inexact);
    }
    return sign ? -absZ : absZ;
}

int_fast64_t f64_to_i64( float64_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast64_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;
    struct uint64_extra sigExtra;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF64UI( uiA );
    exp  = expF64UI( uiA );
    sig  = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= UINT64_C( 0x0010000000000000 );
    shiftDist = 0x433 - exp;
    if ( shiftDist <= 0 ) {
        if ( shiftDist < -11 ) goto invalid;
        sigExtra.v = sig<<-shiftDist;
        sigExtra.extra = 0;
    } else {
        sigExtra = softfloat_shiftRightJam64Extra( sig, 0, shiftDist );
    }
    return
        softfloat_roundToI64(
            sign, sigExtra.v, sigExtra.extra, roundingMode, exact );

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return
        (exp == 0x7FF) && fracF64UI( uiA ) ? i64_fromNaN
            : sign ? i64_fromNegOverflow : i64_fromPosOverflow;
}

int_fast64_t f64_to_i64_r_minMag( float64_t a, bool exact )
{
    uint_fast64_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;
    int_fast64_t absZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF64UI( uiA );
    exp  = expF64UI( uiA );
    sig  = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0x433 - exp;
    if ( shiftDist <= 0 ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( shiftDist < -10 ) {
            if ( uiA == packToF64UI( 1, 0x43E, 0 ) ) {
                return -INT64_C( 0x7FFFFFFFFFFFFFFF ) - 1;
            }
            raiseFlags( flag_invalid );
            return
                (exp == 0x7FF) && sig ? i64_fromNaN
                    : sign ? i64_fromNegOverflow : i64_fromPosOverflow;
        }
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        sig |= UINT64_C( 0x0010000000000000 );
        absZ = sig<<-shiftDist;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( 53 <= shiftDist ) {
            if ( exact && (exp | sig) ) {
                raiseFlags(flag_inexact);
            }
            return 0;
        }
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        sig |= UINT64_C( 0x0010000000000000 );
        absZ = sig>>shiftDist;
        if ( exact && (absZ<<shiftDist != (int_fast64_t)sig) ) {
            raiseFlags(flag_inexact);
        }
    }
    return sign ? -absZ : absZ;
}

uint_fast32_t f64_to_ui32( float64_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast64_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF64UI( uiA );
    exp  = expF64UI( uiA );
    sig  = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#if (ui32_fromNaN != ui32_fromPosOverflow) || (ui32_fromNaN != ui32_fromNegOverflow)
    if ( (exp == 0x7FF) && sig ) {
#if (ui32_fromNaN == ui32_fromPosOverflow)
        sign = 0;
#elif (ui32_fromNaN == ui32_fromNegOverflow)
        sign = 1;
#else
        raiseFlags( flag_invalid );
        return ui32_fromNaN;
#endif
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= UINT64_C( 0x0010000000000000 );
    shiftDist = 0x427 - exp;
    if ( 0 < shiftDist ) sig = softfloat_shiftRightJam64( sig, shiftDist );
    return softfloat_roundToUI32( sign, sig, roundingMode, exact );
}

uint_fast32_t f64_to_ui32_r_minMag( float64_t a, bool exact )
{
    uint_fast64_t uiA;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;
    bool sign;
    uint_fast32_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF64UI( uiA );
    sig = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0x433 - exp;
    if ( 53 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF64UI( uiA );
    if ( sign || (shiftDist < 21) ) {
        raiseFlags( flag_invalid );
        return
            (exp == 0x7FF) && sig ? ui32_fromNaN
                : sign ? ui32_fromNegOverflow : ui32_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig |= UINT64_C( 0x0010000000000000 );
    z = (uint_fast32_t)(sig>>shiftDist); //fixed warning on type cast
    if ( exact && ((uint_fast64_t) z<<shiftDist != sig) ) {
        raiseFlags(flag_inexact);
    }
    return z;
}

uint_fast64_t f64_to_ui64( float64_t a, uint_fast8_t roundingMode, bool exact )
{
    uint_fast64_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;
    struct uint64_extra sigExtra;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    sign = signF64UI( uiA );
    exp  = expF64UI( uiA );
    sig  = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp ) sig |= UINT64_C( 0x0010000000000000 );
    shiftDist = 0x433 - exp;
    if ( shiftDist <= 0 ) {
        if ( shiftDist < -11 ) goto invalid;
        sigExtra.v = sig<<-shiftDist;
        sigExtra.extra = 0;
    } else {
        sigExtra = softfloat_shiftRightJam64Extra( sig, 0, shiftDist );
    }
    return
        softfloat_roundToUI64(
            sign, sigExtra.v, sigExtra.extra, roundingMode, exact );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return
        (exp == 0x7FF) && fracF64UI( uiA ) ? ui64_fromNaN
            : sign ? ui64_fromNegOverflow : ui64_fromPosOverflow;
}

uint_fast64_t f64_to_ui64_r_minMag( float64_t a, bool exact )
{
    uint_fast64_t uiA;
    int_fast16_t exp;
    uint_fast64_t sig;
    int_fast16_t shiftDist;
    bool sign;
    uint_fast64_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiA = a.v;
    exp = expF64UI( uiA );
    sig = fracF64UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    shiftDist = 0x433 - exp;
    if ( 53 <= shiftDist ) {
        if ( exact && (exp | sig) ) {
            raiseFlags(flag_inexact);
        }
        return 0;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF64UI( uiA );
    if ( sign ) goto invalid;
    if ( shiftDist <= 0 ) {
        if ( shiftDist < -11 ) goto invalid;
        z = (sig | UINT64_C( 0x0010000000000000 ))<<-shiftDist;
    } else {
        sig |= UINT64_C( 0x0010000000000000 );
        z = sig>>shiftDist;
        if ( exact && (uint64_t) (sig<<(-shiftDist & 63)) ) {
            raiseFlags(flag_inexact);
        }
    }
    return z;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return
        (exp == 0x7FF) && sig ? ui64_fromNaN
            : sign ? ui64_fromNegOverflow : ui64_fromPosOverflow;
}

float32_t i32_to_f32( int32_t a )
{
    bool sign;
    uint_fast32_t absA;

    sign = (a < 0);
    if ( ! (a & 0x7FFFFFFF) ) {
        return float32_t::fromRaw(sign ? packToF32UI( 1, 0x9E, 0 ) : 0);
    }
    //fixed unsigned unary minus: -x == ~x + 1
    absA = sign ? (~(uint_fast32_t) a + 1) : (uint_fast32_t) a;
    return softfloat_normRoundPackToF32( sign, 0x9C, absA );
}

float64_t i32_to_f64( int32_t a )
{
    uint_fast64_t uiZ;
    bool sign;
    uint_fast32_t absA;
    int_fast8_t shiftDist;

    if ( ! a ) {
        uiZ = 0;
    } else {
        sign = (a < 0);
        //fixed unsigned unary minus: -x == ~x + 1
        absA = sign ? (~(uint_fast32_t) a + 1) : (uint_fast32_t) a;
        shiftDist = softfloat_countLeadingZeros32( absA ) + 21;
        uiZ =
            packToF64UI(
                sign, 0x432 - shiftDist, (uint_fast64_t) absA<<shiftDist );
    }
    return float64_t::fromRaw(uiZ);
}

float32_t i64_to_f32( int64_t a )
{
    bool sign;
    uint_fast64_t absA;
    int_fast8_t shiftDist;
    uint_fast32_t sig;

    sign = (a < 0);
    //fixed unsigned unary minus: -x == ~x + 1
    absA = sign ? (~(uint_fast64_t) a + 1) : (uint_fast64_t) a;
    shiftDist = softfloat_countLeadingZeros64( absA ) - 40;
    if ( 0 <= shiftDist ) {
        return float32_t::fromRaw(a ? packToF32UI(sign, 0x95 - shiftDist, (uint_fast32_t) absA<<shiftDist ) : 0);
    } else {
        shiftDist += 7;
        sig =
            (shiftDist < 0)
                ? (uint_fast32_t) softfloat_shortShiftRightJam64( absA, -shiftDist ) //fixed warning on type cast
                : (uint_fast32_t) absA<<shiftDist;
        return softfloat_roundPackToF32( sign, 0x9C - shiftDist, sig );
    }
}

float64_t i64_to_f64( int64_t a )
{
    bool sign;
    uint_fast64_t absA;

    sign = (a < 0);
    if ( ! (a & UINT64_C( 0x7FFFFFFFFFFFFFFF )) ) {
        return float64_t::fromRaw(sign ? packToF64UI( 1, 0x43E, 0 ) : 0);
    }
    //fixed unsigned unary minus: -x == ~x + 1
    absA = sign ? (~(uint_fast64_t) a + 1) : (uint_fast64_t) a;
    return softfloat_normRoundPackToF64( sign, 0x43C, absA );
}

float32_t softfloat_addMagsF32( uint_fast32_t uiA, uint_fast32_t uiB )
{
    int_fast16_t expA;
    uint_fast32_t sigA;
    int_fast16_t expB;
    uint_fast32_t sigB;
    int_fast16_t expDiff;
    uint_fast32_t uiZ;
    bool signZ;
    int_fast16_t expZ;
    uint_fast32_t sigZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF32UI( uiA );
    sigA = fracF32UI( uiA );
    expB = expF32UI( uiB );
    sigB = fracF32UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( ! expA ) {
            uiZ = uiA + sigB;
            goto uiZ;
        }
        if ( expA == 0xFF ) {
            if ( sigA | sigB ) goto propagateNaN;
            uiZ = uiA;
            goto uiZ;
        }
        signZ = signF32UI( uiA );
        expZ = expA;
        sigZ = 0x01000000 + sigA + sigB;
        if ( ! (sigZ & 1) && (expZ < 0xFE) ) {
            uiZ = packToF32UI( signZ, expZ, sigZ>>1 );
            goto uiZ;
        }
        sigZ <<= 6;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        signZ = signF32UI( uiA );
        sigA <<= 6;
        sigB <<= 6;
        if ( expDiff < 0 ) {
            if ( expB == 0xFF ) {
                if ( sigB ) goto propagateNaN;
                uiZ = packToF32UI( signZ, 0xFF, 0 );
                goto uiZ;
            }
            expZ = expB;
            sigA += expA ? 0x20000000 : sigA;
            sigA = softfloat_shiftRightJam32( sigA, -expDiff );
        } else {
            if ( expA == 0xFF ) {
                if ( sigA ) goto propagateNaN;
                uiZ = uiA;
                goto uiZ;
            }
            expZ = expA;
            sigB += expB ? 0x20000000 : sigB;
            sigB = softfloat_shiftRightJam32( sigB, expDiff );
        }
        sigZ = 0x20000000 + sigA + sigB;
        if ( sigZ < 0x40000000 ) {
            --expZ;
            sigZ <<= 1;
        }
    }
    return softfloat_roundPackToF32( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF32UI( uiA, uiB );
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float64_t
 softfloat_addMagsF64( uint_fast64_t uiA, uint_fast64_t uiB, bool signZ )
{
    int_fast16_t expA;
    uint_fast64_t sigA;
    int_fast16_t expB;
    uint_fast64_t sigB;
    int_fast16_t expDiff;
    uint_fast64_t uiZ;
    int_fast16_t expZ;
    uint_fast64_t sigZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF64UI( uiA );
    sigA = fracF64UI( uiA );
    expB = expF64UI( uiB );
    sigB = fracF64UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( ! expA ) {
            uiZ = uiA + sigB;
            goto uiZ;
        }
        if ( expA == 0x7FF ) {
            if ( sigA | sigB ) goto propagateNaN;
            uiZ = uiA;
            goto uiZ;
        }
        expZ = expA;
        sigZ = UINT64_C( 0x0020000000000000 ) + sigA + sigB;
        sigZ <<= 9;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        sigA <<= 9;
        sigB <<= 9;
        if ( expDiff < 0 ) {
            if ( expB == 0x7FF ) {
                if ( sigB ) goto propagateNaN;
                uiZ = packToF64UI( signZ, 0x7FF, 0 );
                goto uiZ;
            }
            expZ = expB;
            if ( expA ) {
                sigA += UINT64_C( 0x2000000000000000 );
            } else {
                sigA <<= 1;
            }
            sigA = softfloat_shiftRightJam64( sigA, -expDiff );
        } else {
            if ( expA == 0x7FF ) {
                if ( sigA ) goto propagateNaN;
                uiZ = uiA;
                goto uiZ;
            }
            expZ = expA;
            if ( expB ) {
                sigB += UINT64_C( 0x2000000000000000 );
            } else {
                sigB <<= 1;
            }
            sigB = softfloat_shiftRightJam64( sigB, expDiff );
        }
        sigZ = UINT64_C( 0x2000000000000000 ) + sigA + sigB;
        if ( sigZ < UINT64_C( 0x4000000000000000 ) ) {
            --expZ;
            sigZ <<= 1;
        }
    }
    return softfloat_roundPackToF64( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF64UI( uiA, uiB );
 uiZ:
    return float64_t::fromRaw(uiZ);
}

uint32_t softfloat_approxRecipSqrt32_1( unsigned int oddExpA, uint32_t a )
{
    int index;
    uint16_t eps, r0;
    uint_fast32_t ESqrR0;
    uint32_t sigma0;
    uint_fast32_t r;
    uint32_t sqrSigma0;

    index = (a>>27 & 0xE) + oddExpA;
    eps = (uint16_t) (a>>12);
    r0 = softfloat_approxRecipSqrt_1k0s[index]
             - ((softfloat_approxRecipSqrt_1k1s[index] * (uint_fast32_t) eps)
                    >>20);
    ESqrR0 = (uint_fast32_t) r0 * r0;
    if ( ! oddExpA ) ESqrR0 <<= 1;
    sigma0 = ~(uint_fast32_t) (((uint32_t) ESqrR0 * (uint_fast64_t) a)>>23);
    r = (uint_fast32_t)(((uint_fast32_t) r0<<16) + ((r0 * (uint_fast64_t) sigma0)>>25)); //fixed warning on type cast
    sqrSigma0 = ((uint_fast64_t) sigma0 * sigma0)>>32;
    r += ((uint32_t) ((r>>1) + (r>>3) - ((uint_fast32_t) r0<<14))
              * (uint_fast64_t) sqrSigma0)
             >>48;
    if ( ! (r & 0x80000000) ) r = 0x80000000;
    return r;
}

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 32-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
uint_fast32_t softfloat_commonNaNToF32UI( const struct commonNaN *aPtr )
{
    return (uint_fast32_t) aPtr->sign<<31 | 0x7FC00000 | aPtr->v64>>41;
}

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 64-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
uint_fast64_t softfloat_commonNaNToF64UI( const struct commonNaN *aPtr )
{
    return
        (uint_fast64_t) aPtr->sign<<63 | UINT64_C( 0x7FF8000000000000 )
            | aPtr->v64>>12;
}

uint_fast8_t softfloat_countLeadingZeros64( uint64_t a )
{
    uint_fast8_t count;
    uint32_t a32;

    count = 0;
    a32 = a>>32;
    if ( ! a32 ) {
        count = 32;
        a32 = (uint32_t) a; //fixed warning on type cast
    }
    /*------------------------------------------------------------------------
    | From here, result is current count + count leading zeros of `a32'.
    *------------------------------------------------------------------------*/
    if ( a32 < 0x10000 ) {
        count += 16;
        a32 <<= 16;
    }
    if ( a32 < 0x1000000 ) {
        count += 8;
        a32 <<= 8;
    }
    count += softfloat_countLeadingZeros8[a32>>24];
    return count;
}

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 32-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
void softfloat_f32UIToCommonNaN( uint_fast32_t uiA, struct commonNaN *zPtr )
{
    if ( softfloat_isSigNaNF32UI( uiA ) ) {
        raiseFlags( flag_invalid );
    }
    zPtr->sign = (uiA>>31) != 0;
    zPtr->v64  = (uint_fast64_t) uiA<<41;
    zPtr->v0   = 0;
}

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 64-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
void softfloat_f64UIToCommonNaN( uint_fast64_t uiA, struct commonNaN *zPtr )
{
    if ( softfloat_isSigNaNF64UI( uiA ) ) {
        raiseFlags( flag_invalid );
    }
    zPtr->sign = (uiA>>63) != 0;
    zPtr->v64  = uiA<<12;
    zPtr->v0   = 0;
}

struct uint128 softfloat_mul64To128( uint64_t a, uint64_t b )
{
    uint32_t a32, a0, b32, b0;
    struct uint128 z;
    uint64_t mid1, mid;

    a32 = a>>32;
    a0 = (uint32_t)a; //fixed warning on type cast
    b32 = b>>32;
    b0 = (uint32_t) b; //fixed warning on type cast
    z.v0 = (uint_fast64_t) a0 * b0;
    mid1 = (uint_fast64_t) a32 * b0;
    mid = mid1 + (uint_fast64_t) a0 * b32;
    z.v64 = (uint_fast64_t) a32 * b32;
    z.v64 += (uint_fast64_t) (mid < mid1)<<32 | mid>>32;
    mid <<= 32;
    z.v0 += mid;
    z.v64 += (z.v0 < mid);
    return z;
}

float32_t
 softfloat_mulAddF32(
     uint_fast32_t uiA, uint_fast32_t uiB, uint_fast32_t uiC, uint_fast8_t op )
{
    bool signA;
    int_fast16_t expA;
    uint_fast32_t sigA;
    bool signB;
    int_fast16_t expB;
    uint_fast32_t sigB;
    bool signC;
    int_fast16_t expC;
    uint_fast32_t sigC;
    bool signProd;
    uint_fast32_t magBits, uiZ;
    struct exp16_sig32 normExpSig;
    int_fast16_t expProd;
    uint_fast64_t sigProd;
    bool signZ;
    int_fast16_t expZ;
    uint_fast32_t sigZ;
    int_fast16_t expDiff;
    uint_fast64_t sig64Z, sig64C;
    int_fast8_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signA = signF32UI( uiA );
    expA  = expF32UI( uiA );
    sigA  = fracF32UI( uiA );
    signB = signF32UI( uiB );
    expB  = expF32UI( uiB );
    sigB  = fracF32UI( uiB );
    signC = signF32UI( uiC ) ^ (op == softfloat_mulAdd_subC);
    expC  = expF32UI( uiC );
    sigC  = fracF32UI( uiC );
    signProd = signA ^ signB ^ (op == softfloat_mulAdd_subProd);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0xFF ) {
        if ( sigA || ((expB == 0xFF) && sigB) ) goto propagateNaN_ABC;
        magBits = expB | sigB;
        goto infProdArg;
    }
    if ( expB == 0xFF ) {
        if ( sigB ) goto propagateNaN_ABC;
        magBits = expA | sigA;
        goto infProdArg;
    }
    if ( expC == 0xFF ) {
        if ( sigC ) {
            uiZ = 0;
            goto propagateNaN_ZC;
        }
        uiZ = uiC;
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) goto zeroProd;
        normExpSig = softfloat_normSubnormalF32Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    if ( ! expB ) {
        if ( ! sigB ) goto zeroProd;
        normExpSig = softfloat_normSubnormalF32Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expProd = expA + expB - 0x7E;
    sigA = (sigA | 0x00800000)<<7;
    sigB = (sigB | 0x00800000)<<7;
    sigProd = (uint_fast64_t) sigA * sigB;
    if ( sigProd < UINT64_C( 0x2000000000000000 ) ) {
        --expProd;
        sigProd <<= 1;
    }
    signZ = signProd;
    if ( ! expC ) {
        if ( ! sigC ) {
            expZ = expProd - 1;
            sigZ = (uint_fast32_t) softfloat_shortShiftRightJam64( sigProd, 31 ); //fixed warning on type cast
            goto roundPack;
        }
        normExpSig = softfloat_normSubnormalF32Sig( sigC );
        expC = normExpSig.exp;
        sigC = normExpSig.sig;
    }
    sigC = (sigC | 0x00800000)<<6;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expProd - expC;
    if ( signProd == signC ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expDiff <= 0 ) {
            expZ = expC;
            sigZ = sigC + (uint_fast32_t) softfloat_shiftRightJam64( sigProd, 32 - expDiff ); //fixed warning on type cast
        } else {
            expZ = expProd;
            sig64Z =
                sigProd
                    + softfloat_shiftRightJam64(
                          (uint_fast64_t) sigC<<32, expDiff );
            sigZ = (uint_fast32_t) softfloat_shortShiftRightJam64( sig64Z, 32 ); //fixed warning on type cast
        }
        if ( sigZ < 0x40000000 ) {
            --expZ;
            sigZ <<= 1;
        }
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        sig64C = (uint_fast64_t) sigC<<32;
        if ( expDiff < 0 ) {
            signZ = signC;
            expZ = expC;
            sig64Z = sig64C - softfloat_shiftRightJam64( sigProd, -expDiff );
        } else if ( ! expDiff ) {
            expZ = expProd;
            sig64Z = sigProd - sig64C;
            if ( ! sig64Z ) goto completeCancellation;
            if ( sig64Z & UINT64_C( 0x8000000000000000 ) ) {
                signZ = ! signZ;
                //fixed unsigned unary minus: -x == ~x + 1
                sig64Z = ~sig64Z + 1;
            }
        } else {
            expZ = expProd;
            sig64Z = sigProd - softfloat_shiftRightJam64( sig64C, expDiff );
        }
        shiftDist = softfloat_countLeadingZeros64( sig64Z ) - 1;
        expZ -= shiftDist;
        shiftDist -= 32;
        if ( shiftDist < 0 ) {
            sigZ = (uint_fast32_t) softfloat_shortShiftRightJam64( sig64Z, -shiftDist ); //fixed warning on type cast
        } else {
            sigZ = (uint_fast32_t) sig64Z<<shiftDist;
        }
    }
 roundPack:
    return softfloat_roundPackToF32( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN_ABC:
    uiZ = softfloat_propagateNaNF32UI( uiA, uiB );
    goto propagateNaN_ZC;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infProdArg:
    if ( magBits ) {
        uiZ = packToF32UI( signProd, 0xFF, 0 );
        if ( expC != 0xFF ) goto uiZ;
        if ( sigC ) goto propagateNaN_ZC;
        if ( signProd == signC ) goto uiZ;
    }
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF32UI;
 propagateNaN_ZC:
    uiZ = softfloat_propagateNaNF32UI( uiZ, uiC );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zeroProd:
    uiZ = uiC;
    if ( ! (expC | sigC) && (signProd != signC) ) {
 completeCancellation:
        uiZ =
            packToF32UI((globalRoundingMode == round_min), 0, 0 );
    }
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float64_t
 softfloat_mulAddF64(
     uint_fast64_t uiA, uint_fast64_t uiB, uint_fast64_t uiC, uint_fast8_t op )
{
    bool signA;
    int_fast16_t expA;
    uint_fast64_t sigA;
    bool signB;
    int_fast16_t expB;
    uint_fast64_t sigB;
    bool signC;
    int_fast16_t expC;
    uint_fast64_t sigC;
    bool signZ;
    uint_fast64_t magBits, uiZ;
    struct exp16_sig64 normExpSig;
    int_fast16_t expZ;
    struct uint128 sig128Z;
    uint_fast64_t sigZ;
    int_fast16_t expDiff;
    struct uint128 sig128C;
    int_fast8_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signA = signF64UI( uiA );
    expA  = expF64UI( uiA );
    sigA  = fracF64UI( uiA );
    signB = signF64UI( uiB );
    expB  = expF64UI( uiB );
    sigB  = fracF64UI( uiB );
    signC = signF64UI( uiC ) ^ (op == softfloat_mulAdd_subC);
    expC  = expF64UI( uiC );
    sigC  = fracF64UI( uiC );
    signZ = signA ^ signB ^ (op == softfloat_mulAdd_subProd);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x7FF ) {
        if ( sigA || ((expB == 0x7FF) && sigB) ) goto propagateNaN_ABC;
        magBits = expB | sigB;
        goto infProdArg;
    }
    if ( expB == 0x7FF ) {
        if ( sigB ) goto propagateNaN_ABC;
        magBits = expA | sigA;
        goto infProdArg;
    }
    if ( expC == 0x7FF ) {
        if ( sigC ) {
            uiZ = 0;
            goto propagateNaN_ZC;
        }
        uiZ = uiC;
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) goto zeroProd;
        normExpSig = softfloat_normSubnormalF64Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    if ( ! expB ) {
        if ( ! sigB ) goto zeroProd;
        normExpSig = softfloat_normSubnormalF64Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA + expB - 0x3FE;
    sigA = (sigA | UINT64_C( 0x0010000000000000 ))<<10;
    sigB = (sigB | UINT64_C( 0x0010000000000000 ))<<10;
    sig128Z = softfloat_mul64To128( sigA, sigB );
    if ( sig128Z.v64 < UINT64_C( 0x2000000000000000 ) ) {
        --expZ;
        sig128Z =
            softfloat_add128(
                sig128Z.v64, sig128Z.v0, sig128Z.v64, sig128Z.v0 );
    }
    if ( ! expC ) {
        if ( ! sigC ) {
            --expZ;
            sigZ = sig128Z.v64<<1 | (sig128Z.v0 != 0);
            goto roundPack;
        }
        normExpSig = softfloat_normSubnormalF64Sig( sigC );
        expC = normExpSig.exp;
        sigC = normExpSig.sig;
    }
    sigC = (sigC | UINT64_C( 0x0010000000000000 ))<<9;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    //fixed initialization
    sig128C.v0 = sig128C.v64 = 0;
    expDiff = expZ - expC;
    if ( expDiff < 0 ) {
        expZ = expC;
        if ( (signZ == signC) || (expDiff < -1) ) {
            sig128Z.v64 = softfloat_shiftRightJam64( sig128Z.v64, -expDiff );
        } else {
            sig128Z =
                softfloat_shortShiftRightJam128( sig128Z.v64, sig128Z.v0, 1 );
        }
    } else if ( expDiff ) {
        sig128C = softfloat_shiftRightJam128( sigC, 0, expDiff );
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( signZ == signC ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expDiff <= 0 ) {
            sigZ = (sigC + sig128Z.v64) | (sig128Z.v0 != 0);
        } else {
            sig128Z =
                softfloat_add128(
                    sig128Z.v64, sig128Z.v0, sig128C.v64, sig128C.v0 );
            sigZ = sig128Z.v64 | (sig128Z.v0 != 0);
        }
        if ( sigZ < UINT64_C( 0x4000000000000000 ) ) {
            --expZ;
            sigZ <<= 1;
        }
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expDiff < 0 ) {
            signZ = signC;
            sig128Z = softfloat_sub128( sigC, 0, sig128Z.v64, sig128Z.v0 );
        } else if ( ! expDiff ) {
            sig128Z.v64 = sig128Z.v64 - sigC;
            if ( ! (sig128Z.v64 | sig128Z.v0) ) goto completeCancellation;
            if ( sig128Z.v64 & UINT64_C( 0x8000000000000000 ) ) {
                signZ = ! signZ;
                sig128Z = softfloat_sub128( 0, 0, sig128Z.v64, sig128Z.v0 );
            }
        } else {
            sig128Z =
                softfloat_sub128(
                    sig128Z.v64, sig128Z.v0, sig128C.v64, sig128C.v0 );
        }
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( ! sig128Z.v64 ) {
            expZ -= 64;
            sig128Z.v64 = sig128Z.v0;
            sig128Z.v0 = 0;
        }
        shiftDist = softfloat_countLeadingZeros64( sig128Z.v64 ) - 1;
        expZ -= shiftDist;
        if ( shiftDist < 0 ) {
            sigZ = softfloat_shortShiftRightJam64( sig128Z.v64, -shiftDist );
        } else {
            sig128Z =
                softfloat_shortShiftLeft128(
                    sig128Z.v64, sig128Z.v0, shiftDist );
            sigZ = sig128Z.v64;
        }
        sigZ |= (sig128Z.v0 != 0);
    }
 roundPack:
    return softfloat_roundPackToF64( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN_ABC:
    uiZ = softfloat_propagateNaNF64UI( uiA, uiB );
    goto propagateNaN_ZC;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infProdArg:
    if ( magBits ) {
        uiZ = packToF64UI( signZ, 0x7FF, 0 );
        if ( expC != 0x7FF ) goto uiZ;
        if ( sigC ) goto propagateNaN_ZC;
        if ( signZ == signC ) goto uiZ;
    }
    raiseFlags( flag_invalid );
    uiZ = defaultNaNF64UI;
 propagateNaN_ZC:
    uiZ = softfloat_propagateNaNF64UI( uiZ, uiC );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zeroProd:
    uiZ = uiC;
    if ( ! (expC | sigC) && (signZ != signC) ) {
 completeCancellation:
        uiZ =
            packToF64UI((globalRoundingMode == round_min), 0, 0 );
    }
 uiZ:
    return float64_t::fromRaw(uiZ);
}

float32_t
 softfloat_normRoundPackToF32( bool sign, int_fast16_t exp, uint_fast32_t sig )
{
    int_fast8_t shiftDist;

    shiftDist = softfloat_countLeadingZeros32( sig ) - 1;
    exp -= shiftDist;
    if ( (7 <= shiftDist) && ((unsigned int) exp < 0xFD) ) {
        return float32_t::fromRaw(packToF32UI( sign, sig ? exp : 0, sig<<(shiftDist - 7) ));
    } else {
        return softfloat_roundPackToF32( sign, exp, sig<<shiftDist );
    }
}

float64_t
 softfloat_normRoundPackToF64( bool sign, int_fast16_t exp, uint_fast64_t sig )
{
    int_fast8_t shiftDist;

    shiftDist = softfloat_countLeadingZeros64( sig ) - 1;
    exp -= shiftDist;
    if ( (10 <= shiftDist) && ((unsigned int) exp < 0x7FD) ) {
        return float64_t::fromRaw(packToF64UI( sign, sig ? exp : 0, sig<<(shiftDist - 10) ));
    } else {
        return softfloat_roundPackToF64( sign, exp, sig<<shiftDist );
    }
}

struct exp16_sig32 softfloat_normSubnormalF32Sig( uint_fast32_t sig )
{
    int_fast8_t shiftDist;
    struct exp16_sig32 z;

    shiftDist = softfloat_countLeadingZeros32( sig ) - 8;
    z.exp = 1 - shiftDist;
    z.sig = sig<<shiftDist;
    return z;
}

struct exp16_sig64 softfloat_normSubnormalF64Sig( uint_fast64_t sig )
{
    int_fast8_t shiftDist;
    struct exp16_sig64 z;

    shiftDist = softfloat_countLeadingZeros64( sig ) - 11;
    z.exp = 1 - shiftDist;
    z.sig = sig<<shiftDist;
    return z;
}

/*----------------------------------------------------------------------------
| Interpreting `uiA' and `uiB' as the bit patterns of two 32-bit floating-
| point values, at least one of which is a NaN, returns the bit pattern of
| the combined NaN result.  If either `uiA' or `uiB' has the pattern of a
| signaling NaN, the invalid exception is raised.
*----------------------------------------------------------------------------*/
uint_fast32_t
 softfloat_propagateNaNF32UI( uint_fast32_t uiA, uint_fast32_t uiB )
{
    bool isSigNaNA;

    isSigNaNA = softfloat_isSigNaNF32UI( uiA );
    if ( isSigNaNA || softfloat_isSigNaNF32UI( uiB ) ) {
        raiseFlags( flag_invalid );
        if ( isSigNaNA ) return uiA | 0x00400000;
    }
    return (isNaNF32UI( uiA ) ? uiA : uiB) | 0x00400000;
}

/*----------------------------------------------------------------------------
| Interpreting `uiA' and `uiB' as the bit patterns of two 64-bit floating-
| point values, at least one of which is a NaN, returns the bit pattern of
| the combined NaN result.  If either `uiA' or `uiB' has the pattern of a
| signaling NaN, the invalid exception is raised.
*----------------------------------------------------------------------------*/
uint_fast64_t
 softfloat_propagateNaNF64UI( uint_fast64_t uiA, uint_fast64_t uiB )
{
    bool isSigNaNA;

    isSigNaNA = softfloat_isSigNaNF64UI( uiA );
    if ( isSigNaNA || softfloat_isSigNaNF64UI( uiB ) ) {
        raiseFlags( flag_invalid );
        if ( isSigNaNA ) return uiA | UINT64_C( 0x0008000000000000 );
    }
    return (isNaNF64UI( uiA ) ? uiA : uiB) | UINT64_C( 0x0008000000000000 );
}

float32_t
 softfloat_roundPackToF32( bool sign, int_fast16_t exp, uint_fast32_t sig )
{
    uint_fast8_t roundingMode;
    bool roundNearEven;
    uint_fast8_t roundIncrement, roundBits;
    bool isTiny;
    uint_fast32_t uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundingMode = globalRoundingMode;
    roundNearEven = (roundingMode == round_near_even);
    roundIncrement = 0x40;
    if ( ! roundNearEven && (roundingMode != round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? round_min : round_max))
                ? 0x7F
                : 0;
    }
    roundBits = sig & 0x7F;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( 0xFD <= (unsigned int) exp ) {
        if ( exp < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            isTiny =
                (globalDetectTininess == tininess_beforeRounding)
                    || (exp < -1) || (sig + roundIncrement < 0x80000000);
            sig = softfloat_shiftRightJam32( sig, -exp );
            exp = 0;
            roundBits = sig & 0x7F;
            if ( isTiny && roundBits ) {
                raiseFlags( flag_underflow );
            }
        } else if ( (0xFD < exp) || (0x80000000 <= sig + roundIncrement) ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            raiseFlags(
                flag_overflow | flag_inexact );
            uiZ = packToF32UI( sign, 0xFF, 0 ) - ! roundIncrement;
            goto uiZ;
        }
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig = (sig + roundIncrement)>>7;
    if ( roundBits ) {
        raiseFlags(flag_inexact);
        if ( roundingMode == round_odd ) {
            sig |= 1;
            goto packReturn;
        }
    }
    sig &= ~(uint_fast32_t) (! (roundBits ^ 0x40) & roundNearEven);
    if ( ! sig ) exp = 0;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 packReturn:
    uiZ = packToF32UI( sign, exp, sig );
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float64_t
 softfloat_roundPackToF64( bool sign, int_fast16_t exp, uint_fast64_t sig )
{
    uint_fast8_t roundingMode;
    bool roundNearEven;
    uint_fast16_t roundIncrement, roundBits;
    bool isTiny;
    uint_fast64_t uiZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundingMode = globalRoundingMode;
    roundNearEven = (roundingMode == round_near_even);
    roundIncrement = 0x200;
    if ( ! roundNearEven && (roundingMode != round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? round_min : round_max))
                ? 0x3FF
                : 0;
    }
    roundBits = sig & 0x3FF;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( 0x7FD <= (uint16_t) exp ) {
        if ( exp < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            isTiny =
                (globalDetectTininess == tininess_beforeRounding)
                    || (exp < -1)
                    || (sig + roundIncrement < UINT64_C( 0x8000000000000000 ));
            sig = softfloat_shiftRightJam64( sig, -exp );
            exp = 0;
            roundBits = sig & 0x3FF;
            if ( isTiny && roundBits ) {
                raiseFlags( flag_underflow );
            }
        } else if (
            (0x7FD < exp)
                || (UINT64_C( 0x8000000000000000 ) <= sig + roundIncrement)
        ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            raiseFlags(
                flag_overflow | flag_inexact );
            uiZ = packToF64UI( sign, 0x7FF, 0 ) - ! roundIncrement;
            goto uiZ;
        }
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig = (sig + roundIncrement)>>10;
    if ( roundBits ) {
        raiseFlags(flag_inexact);
        if ( roundingMode == round_odd ) {
            sig |= 1;
            goto packReturn;
        }
    }
    sig &= ~(uint_fast64_t) (! (roundBits ^ 0x200) & roundNearEven);
    if ( ! sig ) exp = 0;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 packReturn:
    uiZ = packToF64UI( sign, exp, sig );
 uiZ:
    return float64_t::fromRaw(uiZ);
}

int_fast32_t
 softfloat_roundToI32(
     bool sign, uint_fast64_t sig, uint_fast8_t roundingMode, bool exact )
{
    bool roundNearEven;
    uint_fast16_t roundIncrement, roundBits;
    uint_fast32_t sig32;
    union { uint32_t ui; int32_t i; } uZ;
    int_fast32_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundNearEven = (roundingMode == round_near_even);
    roundIncrement = 0x800;
    if ( ! roundNearEven && (roundingMode != round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? round_min : round_max))
                ? 0xFFF
                : 0;
    }
    roundBits = sig & 0xFFF;
    sig += roundIncrement;
    if ( sig & UINT64_C( 0xFFFFF00000000000 ) ) goto invalid;
    sig32 = (uint_fast32_t)(sig>>12); //fixed warning on type cast
    sig32 &= ~(uint_fast32_t) (! (roundBits ^ 0x800) & roundNearEven);
    //fixed unsigned unary minus: -x == ~x + 1
    uZ.ui = sign ? (~sig32 + 1) : sig32;
    z = uZ.i;
    if ( z && ((z < 0) ^ sign) ) goto invalid;
    if ( exact && roundBits ) {
        raiseFlags(flag_inexact);
    }
    return z;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return sign ? i32_fromNegOverflow : i32_fromPosOverflow;
}

int_fast64_t
 softfloat_roundToI64(
     bool sign,
     uint_fast64_t sig,
     uint_fast64_t sigExtra,
     uint_fast8_t roundingMode,
     bool exact
 )
{
    bool roundNearEven, doIncrement;
    union { uint64_t ui; int64_t i; } uZ;
    int_fast64_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundNearEven = (roundingMode == round_near_even);
    doIncrement = (UINT64_C( 0x8000000000000000 ) <= sigExtra);
    if ( ! roundNearEven && (roundingMode != round_near_maxMag) ) {
        doIncrement =
            (roundingMode
                 == (sign ? round_min : round_max))
                && sigExtra;
    }
    if ( doIncrement ) {
        ++sig;
        if ( ! sig ) goto invalid;
        sig &=
            ~(uint_fast64_t)
                 (! (sigExtra & UINT64_C( 0x7FFFFFFFFFFFFFFF ))
                      & roundNearEven);
    }
    //fixed unsigned unary minus: -x == ~x + 1
    uZ.ui = sign ? (~sig + 1) : sig;
    z = uZ.i;
    if ( z && ((z < 0) ^ sign) ) goto invalid;
    if ( exact && sigExtra ) {
        raiseFlags(flag_inexact);
    }
    return z;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return sign ? i64_fromNegOverflow : i64_fromPosOverflow;
}

uint_fast32_t
 softfloat_roundToUI32(
     bool sign, uint_fast64_t sig, uint_fast8_t roundingMode, bool exact )
{
    bool roundNearEven;
    uint_fast16_t roundIncrement, roundBits;
    uint_fast32_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundNearEven = (roundingMode == round_near_even);
    roundIncrement = 0x800;
    if ( ! roundNearEven && (roundingMode != round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? round_min : round_max))
                ? 0xFFF
                : 0;
    }
    roundBits = sig & 0xFFF;
    sig += roundIncrement;
    if ( sig & UINT64_C( 0xFFFFF00000000000 ) ) goto invalid;
    z = (uint_fast32_t)(sig>>12); //fixed warning on type cast
    z &= ~(uint_fast32_t) (! (roundBits ^ 0x800) & roundNearEven);
    if ( sign && z ) goto invalid;
    if ( exact && roundBits ) {
        raiseFlags(flag_inexact);
    }
    return z;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return sign ? ui32_fromNegOverflow : ui32_fromPosOverflow;
}

uint_fast64_t
 softfloat_roundToUI64(
     bool sign,
     uint_fast64_t sig,
     uint_fast64_t sigExtra,
     uint_fast8_t roundingMode,
     bool exact
 )
{
    bool roundNearEven, doIncrement;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundNearEven = (roundingMode == round_near_even);
    doIncrement = (UINT64_C( 0x8000000000000000 ) <= sigExtra);
    if ( ! roundNearEven && (roundingMode != round_near_maxMag) ) {
        doIncrement =
            (roundingMode
                 == (sign ? round_min : round_max))
                && sigExtra;
    }
    if ( doIncrement ) {
        ++sig;
        if ( ! sig ) goto invalid;
        sig &=
            ~(uint_fast64_t)
                 (! (sigExtra & UINT64_C( 0x7FFFFFFFFFFFFFFF ))
                      & roundNearEven);
    }
    if ( sign && sig ) goto invalid;
    if ( exact && sigExtra ) {
        raiseFlags(flag_inexact);
    }
    return sig;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    raiseFlags( flag_invalid );
    return sign ? ui64_fromNegOverflow : ui64_fromPosOverflow;
}

struct uint128
 softfloat_shiftRightJam128( uint64_t a64, uint64_t a0, uint_fast32_t dist )
{
    uint_fast8_t u8NegDist;
    struct uint128 z;

    if ( dist < 64 ) {
        //fixed unsigned unary minus: -x == ~x + 1 , fixed type cast
        u8NegDist = (uint_fast8_t)(~dist + 1);
        z.v64 = a64>>dist;
        z.v0 =
            a64<<(u8NegDist & 63) | a0>>dist
                | ((uint64_t) (a0<<(u8NegDist & 63)) != 0);
    } else {
        z.v64 = 0;
        z.v0 =
            (dist < 127)
                ? a64>>(dist & 63)
                      | (((a64 & (((uint_fast64_t) 1<<(dist & 63)) - 1)) | a0)
                             != 0)
                : ((a64 | a0) != 0);
    }
    return z;
}

float32_t softfloat_subMagsF32( uint_fast32_t uiA, uint_fast32_t uiB )
{
    int_fast16_t expA;
    uint_fast32_t sigA;
    int_fast16_t expB;
    uint_fast32_t sigB;
    int_fast16_t expDiff;
    uint_fast32_t uiZ;
    int_fast32_t sigDiff;
    bool signZ;
    int_fast8_t shiftDist;
    int_fast16_t expZ;
    uint_fast32_t sigX, sigY;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF32UI( uiA );
    sigA = fracF32UI( uiA );
    expB = expF32UI( uiB );
    sigB = fracF32UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expA == 0xFF ) {
            if ( sigA | sigB ) goto propagateNaN;
            raiseFlags( flag_invalid );
            uiZ = defaultNaNF32UI;
            goto uiZ;
        }
        sigDiff = sigA - sigB;
        if ( ! sigDiff ) {
            uiZ =
                packToF32UI(
                    (globalRoundingMode == round_min), 0, 0 );
            goto uiZ;
        }
        if ( expA ) --expA;
        signZ = signF32UI( uiA );
        if ( sigDiff < 0 ) {
            signZ = ! signZ;
            sigDiff = -sigDiff;
        }
        shiftDist = softfloat_countLeadingZeros32( sigDiff ) - 8;
        expZ = expA - shiftDist;
        if ( expZ < 0 ) {
            shiftDist = (int_fast8_t)expA; //fixed type cast
            expZ = 0;
        }
        uiZ = packToF32UI( signZ, expZ, sigDiff<<shiftDist );
        goto uiZ;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        signZ = signF32UI( uiA );
        sigA <<= 7;
        sigB <<= 7;
        if ( expDiff < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            signZ = ! signZ;
            if ( expB == 0xFF ) {
                if ( sigB ) goto propagateNaN;
                uiZ = packToF32UI( signZ, 0xFF, 0 );
                goto uiZ;
            }
            expZ = expB - 1;
            sigX = sigB | 0x40000000;
            sigY = sigA + (expA ? 0x40000000 : sigA);
            expDiff = -expDiff;
        } else {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            if ( expA == 0xFF ) {
                if ( sigA ) goto propagateNaN;
                uiZ = uiA;
                goto uiZ;
            }
            expZ = expA - 1;
            sigX = sigA | 0x40000000;
            sigY = sigB + (expB ? 0x40000000 : sigB);
        }
        return
            softfloat_normRoundPackToF32(
                signZ, expZ, sigX - softfloat_shiftRightJam32( sigY, expDiff )
            );
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF32UI( uiA, uiB );
 uiZ:
    return float32_t::fromRaw(uiZ);
}

float64_t
 softfloat_subMagsF64( uint_fast64_t uiA, uint_fast64_t uiB, bool signZ )
{
    int_fast16_t expA;
    uint_fast64_t sigA;
    int_fast16_t expB;
    uint_fast64_t sigB;
    int_fast16_t expDiff;
    uint_fast64_t uiZ;
    int_fast64_t sigDiff;
    int_fast8_t shiftDist;
    int_fast16_t expZ;
    uint_fast64_t sigZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF64UI( uiA );
    sigA = fracF64UI( uiA );
    expB = expF64UI( uiB );
    sigB = fracF64UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expA == 0x7FF ) {
            if ( sigA | sigB ) goto propagateNaN;
            raiseFlags( flag_invalid );
            uiZ = defaultNaNF64UI;
            goto uiZ;
        }
        sigDiff = sigA - sigB;
        if ( ! sigDiff ) {
            uiZ =
                packToF64UI(
                    (globalRoundingMode == round_min), 0, 0 );
            goto uiZ;
        }
        if ( expA ) --expA;
        if ( sigDiff < 0 ) {
            signZ = ! signZ;
            sigDiff = -sigDiff;
        }
        shiftDist = softfloat_countLeadingZeros64( sigDiff ) - 11;
        expZ = expA - shiftDist;
        if ( expZ < 0 ) {
            shiftDist = (int_fast8_t)expA; //fixed type cast
            expZ = 0;
        }
        uiZ = packToF64UI( signZ, expZ, sigDiff<<shiftDist );
        goto uiZ;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        sigA <<= 10;
        sigB <<= 10;
        if ( expDiff < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            signZ = ! signZ;
            if ( expB == 0x7FF ) {
                if ( sigB ) goto propagateNaN;
                uiZ = packToF64UI( signZ, 0x7FF, 0 );
                goto uiZ;
            }
            sigA += expA ? UINT64_C( 0x4000000000000000 ) : sigA;
            sigA = softfloat_shiftRightJam64( sigA, -expDiff );
            sigB |= UINT64_C( 0x4000000000000000 );
            expZ = expB;
            sigZ = sigB - sigA;
        } else {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            if ( expA == 0x7FF ) {
                if ( sigA ) goto propagateNaN;
                uiZ = uiA;
                goto uiZ;
            }
            sigB += expB ? UINT64_C( 0x4000000000000000 ) : sigB;
            sigB = softfloat_shiftRightJam64( sigB, expDiff );
            sigA |= UINT64_C( 0x4000000000000000 );
            expZ = expA;
            sigZ = sigA - sigB;
        }
        return softfloat_normRoundPackToF64( signZ, expZ - 1, sigZ );
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF64UI( uiA, uiB );
 uiZ:
    return float64_t::fromRaw(uiZ);
}

float32_t ui32_to_f32( uint32_t a )
{
    if ( ! a ) {
        return float32_t::fromRaw(0);
    }
    if ( a & 0x80000000 ) {
        return softfloat_roundPackToF32( 0, 0x9D, a>>1 | (a & 1) );
    } else {
        return softfloat_normRoundPackToF32( 0, 0x9C, a );
    }
}

float64_t ui32_to_f64( uint32_t a )
{
    uint_fast64_t uiZ;
    int_fast8_t shiftDist;

    if ( ! a ) {
        uiZ = 0;
    } else {
        shiftDist = softfloat_countLeadingZeros32( a ) + 21;
        uiZ =
            packToF64UI( 0, 0x432 - shiftDist, (uint_fast64_t) a<<shiftDist );
    }
    return float64_t::fromRaw(uiZ);
}

float32_t ui64_to_f32( uint64_t a )
{
    int_fast8_t shiftDist;
    uint_fast32_t sig;

    shiftDist = softfloat_countLeadingZeros64( a ) - 40;
    if ( 0 <= shiftDist ) {
        return float32_t::fromRaw(a ? packToF32UI(0, 0x95 - shiftDist, (uint_fast32_t) a<<shiftDist ) : 0);
    } else {
        shiftDist += 7;
        sig =
            (shiftDist < 0) ? (uint_fast32_t) softfloat_shortShiftRightJam64( a, -shiftDist ) //fixed warning on type cast
                : (uint_fast32_t) a<<shiftDist;
        return softfloat_roundPackToF32( 0, 0x9C - shiftDist, sig );
    }
}

float64_t ui64_to_f64( uint64_t a )
{
    if ( ! a ) {
        return float64_t::fromRaw(0);
    }
    if ( a & UINT64_C( 0x8000000000000000 ) ) {
        return
            softfloat_roundPackToF64(
                0, 0x43D, softfloat_shortShiftRightJam64( a, 1 ) );
    } else {
        return softfloat_normRoundPackToF64( 0, 0x43C, a );
    }
}

/*----------------------------------------------------------------------------
| Ported from OpenCV.
*----------------------------------------------------------------------------*/

////////////////////////////////////// EXP /////////////////////////////////////

#define EXPTAB_SCALE 6
#define EXPTAB_MASK  ((1 << EXPTAB_SCALE) - 1)

static const softdouble EXPPOLY_32F_A0(.9670371139572337719125840413672004409288e-2);

static const double expTab[] = {
    1.0,
    1.0108892860517004600204097905619,
    1.0218971486541166782344801347833,
    1.0330248790212284225001082839705,
    1.0442737824274138403219664787399,
    1.0556451783605571588083413251529,
    1.0671404006768236181695211209928,
    1.0787607977571197937406800374385,
    1.0905077326652576592070106557607,
    1.1023825833078409435564142094256,
    1.1143867425958925363088129569196,
    1.126521618608241899794798643787 ,
    1.1387886347566916537038302838415,
    1.151189229952982705817759635202 ,
    1.1637248587775775138135735990922,
    1.1763969916502812762846457284838,
    1.1892071150027210667174999705605,
    1.2021567314527031420963969574978,
    1.2152473599804688781165202513388,
    1.2284805361068700056940089577928,
    1.2418578120734840485936774687266,
    1.2553807570246910895793906574423,
    1.2690509571917332225544190810323,
    1.2828700160787782807266697810215,
    1.2968395546510096659337541177925,
    1.3109612115247643419229917863308,
    1.3252366431597412946295370954987,
    1.3396675240533030053600306697244,
    1.3542555469368927282980147401407,
    1.3690024229745906119296011329822,
    1.3839098819638319548726595272652,
    1.3989796725383111402095281367152,
    1.4142135623730950488016887242097,
    1.4296133383919700112350657782751,
    1.4451808069770466200370062414717,
    1.4609177941806469886513028903106,
    1.476826145939499311386907480374 ,
    1.4929077282912648492006435314867,
    1.5091644275934227397660195510332,
    1.5255981507445383068512536895169,
    1.5422108254079408236122918620907,
    1.5590044002378369670337280894749,
    1.5759808451078864864552701601819,
    1.5931421513422668979372486431191,
    1.6104903319492543081795206673574,
    1.628027421857347766848218522014 ,
    1.6457554781539648445187567247258,
    1.6636765803267364350463364569764,
    1.6817928305074290860622509524664,
    1.7001063537185234695013625734975,
    1.7186192981224779156293443764563,
    1.7373338352737062489942020818722,
    1.7562521603732994831121606193753,
    1.7753764925265212525505592001993,
    1.7947090750031071864277032421278,
    1.8142521755003987562498346003623,
    1.8340080864093424634870831895883,
    1.8539791250833855683924530703377,
    1.8741676341102999013299989499544,
    1.8945759815869656413402186534269,
    1.9152065613971472938726112702958,
    1.9360617934922944505980559045667,
    1.9571441241754002690183222516269,
    1.9784560263879509682582499181312,
};

// 1 / ln(2) == 1.4426950408889634073599246810019
static const float64_t exp_prescale = float64_t(1.4426950408889634073599246810019) * float64_t(1 << EXPTAB_SCALE);
static const float64_t exp_postscale = float64_t::one()/float64_t(1 << EXPTAB_SCALE);
static const float64_t exp_max_val(3000*(1 << EXPTAB_SCALE)); // log10(DBL_MAX) < 3000

float32_t f32_exp( float32_t x)
{
    //special cases
    if(x.isNaN()) return float32_t::nan();
    if(x.isInf()) return (x == float32_t::inf()) ? x : float32_t::zero();

    static const float64_t
        A4 = float64_t(1.000000000000002438532970795181890933776) / EXPPOLY_32F_A0,
        A3 = float64_t(.6931471805521448196800669615864773144641) / EXPPOLY_32F_A0,
        A2 = float64_t(.2402265109513301490103372422686535526573) / EXPPOLY_32F_A0,
        A1 = float64_t(.5550339366753125211915322047004666939128e-1) / EXPPOLY_32F_A0;

    float64_t x0;
    if(expF32UI(x.v) > 127 + 10)
        x0 = signF32UI(x.v) ? -exp_max_val : exp_max_val;
    else
        x0 = f32_to_f64(x) * exp_prescale;

    int val0 = f64_to_i32(x0, round_near_even, false);
    int t = (val0 >> EXPTAB_SCALE) + 1023;
    t = t < 0 ? 0 : (t > 2047 ? 2047 : t);
    float64_t buf; buf.v = packToF64UI(0, t, 0);

    x0 = (x0 - f64_roundToInt(x0, round_near_even, false)) * exp_postscale;

    return (buf * EXPPOLY_32F_A0 * float64_t(expTab[val0 & EXPTAB_MASK]) * ((((x0 + A1)*x0 + A2)*x0 + A3)*x0 + A4));
}

float64_t f64_exp(float64_t x)
{
    //special cases
    if(x.isNaN()) return float64_t::nan();
    if(x.isInf()) return (x == float64_t::inf()) ? x : float64_t::zero();

    static const float64_t
    A5 = float64_t(.99999999999999999998285227504999) / EXPPOLY_32F_A0,
    A4 = float64_t(.69314718055994546743029643825322) / EXPPOLY_32F_A0,
    A3 = float64_t(.24022650695886477918181338054308) / EXPPOLY_32F_A0,
    A2 = float64_t(.55504108793649567998466049042729e-1) / EXPPOLY_32F_A0,
    A1 = float64_t(.96180973140732918010002372686186e-2) / EXPPOLY_32F_A0,
    A0 = float64_t(.13369713757180123244806654839424e-2) / EXPPOLY_32F_A0;

    float64_t x0;
    if(expF64UI(x.v) > 1023 + 10)
        x0 = signF64UI(x.v) ? -exp_max_val : exp_max_val;
    else
        x0 = x * exp_prescale;

    int val0 = cvRound(x0);
    int t = (val0 >> EXPTAB_SCALE) + 1023;
    t = t < 0 ? 0 : (t > 2047 ? 2047 : t);
    float64_t buf; buf.v = packToF64UI(0, t, 0);

    x0 = (x0 - f64_roundToInt(x0, round_near_even, false)) * exp_postscale;

    return buf * EXPPOLY_32F_A0 * float64_t(expTab[val0 & EXPTAB_MASK]) * (((((A0*x0 + A1)*x0 + A2)*x0 + A3)*x0 + A4)*x0 + A5);
}

#undef EXPTAB_SCALE
#undef EXPTAB_MASK
#undef EXPPOLY_32F_A0

/////////////////////////////////////////// LOG ///////////////////////////////////////

#define LOGTAB_SCALE    8

static const double CV_DECL_ALIGNED(16) icvLogTab[] = {
    0.0000000000000000000000000000000000000000,    1.000000000000000000000000000000000000000,
    .00389864041565732288852075271279318258166,    .9961089494163424124513618677042801556420,
    .00778214044205494809292034119607706088573,    .9922480620155038759689922480620155038760,
    .01165061721997527263705585198749759001657,    .9884169884169884169884169884169884169884,
    .01550418653596525274396267235488267033361,    .9846153846153846153846153846153846153846,
    .01934296284313093139406447562578250654042,    .9808429118773946360153256704980842911877,
    .02316705928153437593630670221500622574241,    .9770992366412213740458015267175572519084,
    .02697658769820207233514075539915211265906,    .9733840304182509505703422053231939163498,
    .03077165866675368732785500469617545604706,    .9696969696969696969696969696969696969697,
    .03455238150665972812758397481047722976656,    .9660377358490566037735849056603773584906,
    .03831886430213659461285757856785494368522,    .9624060150375939849624060150375939849624,
    .04207121392068705056921373852674150839447,    .9588014981273408239700374531835205992509,
    .04580953603129420126371940114040626212953,    .9552238805970149253731343283582089552239,
    .04953393512227662748292900118940451648088,    .9516728624535315985130111524163568773234,
    .05324451451881227759255210685296333394944,    .9481481481481481481481481481481481481481,
    .05694137640013842427411105973078520037234,    .9446494464944649446494464944649446494465,
    .06062462181643483993820353816772694699466,    .9411764705882352941176470588235294117647,
    .06429435070539725460836422143984236754475,    .9377289377289377289377289377289377289377,
    .06795066190850773679699159401934593915938,    .9343065693430656934306569343065693430657,
    .07159365318700880442825962290953611955044,    .9309090909090909090909090909090909090909,
    .07522342123758751775142172846244648098944,    .9275362318840579710144927536231884057971,
    .07884006170777602129362549021607264876369,    .9241877256317689530685920577617328519856,
    .08244366921107458556772229485432035289706,    .9208633093525179856115107913669064748201,
    .08603433734180314373940490213499288074675,    .9175627240143369175627240143369175627240,
    .08961215868968712416897659522874164395031,    .9142857142857142857142857142857142857143,
    .09317722485418328259854092721070628613231,    .9110320284697508896797153024911032028470,
    .09672962645855109897752299730200320482256,    .9078014184397163120567375886524822695035,
    .10026945316367513738597949668474029749630,    .9045936395759717314487632508833922261484,
    .10379679368164355934833764649738441221420,    .9014084507042253521126760563380281690141,
    .10731173578908805021914218968959175981580,    .8982456140350877192982456140350877192982,
    .11081436634029011301105782649756292812530,    .8951048951048951048951048951048951048951,
    .11430477128005862852422325204315711744130,    .8919860627177700348432055749128919860627,
    .11778303565638344185817487641543266363440,    .8888888888888888888888888888888888888889,
    .12124924363286967987640707633545389398930,    .8858131487889273356401384083044982698962,
    .12470347850095722663787967121606925502420,    .8827586206896551724137931034482758620690,
    .12814582269193003360996385708858724683530,    .8797250859106529209621993127147766323024,
    .13157635778871926146571524895989568904040,    .8767123287671232876712328767123287671233,
    .13499516453750481925766280255629681050780,    .8737201365187713310580204778156996587031,
    .13840232285911913123754857224412262439730,    .8707482993197278911564625850340136054422,
    .14179791186025733629172407290752744302150,    .8677966101694915254237288135593220338983,
    .14518200984449788903951628071808954700830,    .8648648648648648648648648648648648648649,
    .14855469432313711530824207329715136438610,    .8619528619528619528619528619528619528620,
    .15191604202584196858794030049466527998450,    .8590604026845637583892617449664429530201,
    .15526612891112392955683674244937719777230,    .8561872909698996655518394648829431438127,
    .15860503017663857283636730244325008243330,    .8533333333333333333333333333333333333333,
    .16193282026931324346641360989451641216880,    .8504983388704318936877076411960132890365,
    .16524957289530714521497145597095368430010,    .8476821192052980132450331125827814569536,
    .16855536102980664403538924034364754334090,    .8448844884488448844884488448844884488449,
    .17185025692665920060697715143760433420540,    .8421052631578947368421052631578947368421,
    .17513433212784912385018287750426679849630,    .8393442622950819672131147540983606557377,
    .17840765747281828179637841458315961062910,    .8366013071895424836601307189542483660131,
    .18167030310763465639212199675966985523700,    .8338762214983713355048859934853420195440,
    .18492233849401198964024217730184318497780,    .8311688311688311688311688311688311688312,
    .18816383241818296356839823602058459073300,    .8284789644012944983818770226537216828479,
    .19139485299962943898322009772527962923050,    .8258064516129032258064516129032258064516,
    .19461546769967164038916962454095482826240,    .8231511254019292604501607717041800643087,
    .19782574332991986754137769821682013571260,    .8205128205128205128205128205128205128205,
    .20102574606059073203390141770796617493040,    .8178913738019169329073482428115015974441,
    .20421554142869088876999228432396193966280,    .8152866242038216560509554140127388535032,
    .20739519434607056602715147164417430758480,    .8126984126984126984126984126984126984127,
    .21056476910734961416338251183333341032260,    .8101265822784810126582278481012658227848,
    .21372432939771812687723695489694364368910,    .8075709779179810725552050473186119873817,
    .21687393830061435506806333251006435602900,    .8050314465408805031446540880503144654088,
    .22001365830528207823135744547471404075630,    .8025078369905956112852664576802507836991,
    .22314355131420973710199007200571941211830,    .8000000000000000000000000000000000000000,
    .22626367865045338145790765338460914790630,    .7975077881619937694704049844236760124611,
    .22937410106484582006380890106811420992010,    .7950310559006211180124223602484472049689,
    .23247487874309405442296849741978803649550,    .7925696594427244582043343653250773993808,
    .23556607131276688371634975283086532726890,    .7901234567901234567901234567901234567901,
    .23864773785017498464178231643018079921600,    .7876923076923076923076923076923076923077,
    .24171993688714515924331749374687206000090,    .7852760736196319018404907975460122699387,
    .24478272641769091566565919038112042471760,    .7828746177370030581039755351681957186544,
    .24783616390458124145723672882013488560910,    .7804878048780487804878048780487804878049,
    .25088030628580937353433455427875742316250,    .7781155015197568389057750759878419452888,
    .25391520998096339667426946107298135757450,    .7757575757575757575757575757575757575758,
    .25694093089750041913887912414793390780680,    .7734138972809667673716012084592145015106,
    .25995752443692604627401010475296061486000,    .7710843373493975903614457831325301204819,
    .26296504550088134477547896494797896593800,    .7687687687687687687687687687687687687688,
    .26596354849713793599974565040611196309330,    .7664670658682634730538922155688622754491,
    .26895308734550393836570947314612567424780,    .7641791044776119402985074626865671641791,
    .27193371548364175804834985683555714786050,    .7619047619047619047619047619047619047619,
    .27490548587279922676529508862586226314300,    .7596439169139465875370919881305637982196,
    .27786845100345625159121709657483734190480,    .7573964497041420118343195266272189349112,
    .28082266290088775395616949026589281857030,    .7551622418879056047197640117994100294985,
    .28376817313064456316240580235898960381750,    .7529411764705882352941176470588235294118,
    .28670503280395426282112225635501090437180,    .7507331378299120234604105571847507331378,
    .28963329258304265634293983566749375313530,    .7485380116959064327485380116959064327485,
    .29255300268637740579436012922087684273730,    .7463556851311953352769679300291545189504,
    .29546421289383584252163927885703742504130,    .7441860465116279069767441860465116279070,
    .29836697255179722709783618483925238251680,    .7420289855072463768115942028985507246377,
    .30126133057816173455023545102449133992200,    .7398843930635838150289017341040462427746,
    .30414733546729666446850615102448500692850,    .7377521613832853025936599423631123919308,
    .30702503529491181888388950937951449304830,    .7356321839080459770114942528735632183908,
    .30989447772286465854207904158101882785550,    .7335243553008595988538681948424068767908,
    .31275571000389684739317885942000430077330,    .7314285714285714285714285714285714285714,
    .31560877898630329552176476681779604405180,    .7293447293447293447293447293447293447293,
    .31845373111853458869546784626436419785030,    .7272727272727272727272727272727272727273,
    .32129061245373424782201254856772720813750,    .7252124645892351274787535410764872521246,
    .32411946865421192853773391107097268104550,    .7231638418079096045197740112994350282486,
    .32694034499585328257253991068864706903700,    .7211267605633802816901408450704225352113,
    .32975328637246797969240219572384376078850,    .7191011235955056179775280898876404494382,
    .33255833730007655635318997155991382896900,    .7170868347338935574229691876750700280112,
    .33535554192113781191153520921943709254280,    .7150837988826815642458100558659217877095,
    .33814494400871636381467055798566434532400,    .7130919220055710306406685236768802228412,
    .34092658697059319283795275623560883104800,    .7111111111111111111111111111111111111111,
    .34370051385331840121395430287520866841080,    .7091412742382271468144044321329639889197,
    .34646676734620857063262633346312213689100,    .7071823204419889502762430939226519337017,
    .34922538978528827602332285096053965389730,    .7052341597796143250688705234159779614325,
    .35197642315717814209818925519357435405250,    .7032967032967032967032967032967032967033,
    .35471990910292899856770532096561510115850,    .7013698630136986301369863013698630136986,
    .35745588892180374385176833129662554711100,    .6994535519125683060109289617486338797814,
    .36018440357500774995358483465679455548530,    .6975476839237057220708446866485013623978,
    .36290549368936841911903457003063522279280,    .6956521739130434782608695652173913043478,
    .36561919956096466943762379742111079394830,    .6937669376693766937669376693766937669377,
    .36832556115870762614150635272380895912650,    .6918918918918918918918918918918918918919,
    .37102461812787262962487488948681857436900,    .6900269541778975741239892183288409703504,
    .37371640979358405898480555151763837784530,    .6881720430107526881720430107526881720430,
    .37640097516425302659470730759494472295050,    .6863270777479892761394101876675603217158,
    .37907835293496944251145919224654790014030,    .6844919786096256684491978609625668449198,
    .38174858149084833769393299007788300514230,    .6826666666666666666666666666666666666667,
    .38441169891033200034513583887019194662580,    .6808510638297872340425531914893617021277,
    .38706774296844825844488013899535872042180,    .6790450928381962864721485411140583554377,
    .38971675114002518602873692543653305619950,    .6772486772486772486772486772486772486772,
    .39235876060286384303665840889152605086580,    .6754617414248021108179419525065963060686,
    .39499380824086893770896722344332374632350,    .6736842105263157894736842105263157894737,
    .39762193064713846624158577469643205404280,    .6719160104986876640419947506561679790026,
    .40024316412701266276741307592601515352730,    .6701570680628272251308900523560209424084,
    .40285754470108348090917615991202183067800,    .6684073107049608355091383812010443864230,
    .40546510810816432934799991016916465014230,    .6666666666666666666666666666666666666667,
    .40806588980822172674223224930756259709600,    .6649350649350649350649350649350649350649,
    .41065992498526837639616360320360399782650,    .6632124352331606217616580310880829015544,
    .41324724855021932601317757871584035456180,    .6614987080103359173126614987080103359173,
    .41582789514371093497757669865677598863850,    .6597938144329896907216494845360824742268,
    .41840189913888381489925905043492093682300,    .6580976863753213367609254498714652956298,
    .42096929464412963239894338585145305842150,    .6564102564102564102564102564102564102564,
    .42353011550580327293502591601281892508280,    .6547314578005115089514066496163682864450,
    .42608439531090003260516141381231136620050,    .6530612244897959183673469387755102040816,
    .42863216738969872610098832410585600882780,    .6513994910941475826972010178117048346056,
    .43117346481837132143866142541810404509300,    .6497461928934010152284263959390862944162,
    .43370832042155937902094819946796633303180,    .6481012658227848101265822784810126582278,
    .43623676677491801667585491486534010618930,    .6464646464646464646464646464646464646465,
    .43875883620762790027214350629947148263450,    .6448362720403022670025188916876574307305,
    .44127456080487520440058801796112675219780,    .6432160804020100502512562814070351758794,
    .44378397241030093089975139264424797147500,    .6416040100250626566416040100250626566416,
    .44628710262841947420398014401143882423650,    .6400000000000000000000000000000000000000,
    .44878398282700665555822183705458883196130,    .6384039900249376558603491271820448877805,
    .45127464413945855836729492693848442286250,    .6368159203980099502487562189054726368159,
    .45375911746712049854579618113348260521900,    .6352357320099255583126550868486352357320,
    .45623743348158757315857769754074979573500,    .6336633663366336633663366336633663366337,
    .45870962262697662081833982483658473938700,    .6320987654320987654320987654320987654321,
    .46117571512217014895185229761409573256980,    .6305418719211822660098522167487684729064,
    .46363574096303250549055974261136725544930,    .6289926289926289926289926289926289926290,
    .46608972992459918316399125615134835243230,    .6274509803921568627450980392156862745098,
    .46853771156323925639597405279346276074650,    .6259168704156479217603911980440097799511,
    .47097971521879100631480241645476780831830,    .6243902439024390243902439024390243902439,
    .47341577001667212165614273544633761048330,    .6228710462287104622871046228710462287105,
    .47584590486996386493601107758877333253630,    .6213592233009708737864077669902912621359,
    .47827014848147025860569669930555392056700,    .6198547215496368038740920096852300242131,
    .48068852934575190261057286988943815231330,    .6183574879227053140096618357487922705314,
    .48310107575113581113157579238759353756900,    .6168674698795180722891566265060240963855,
    .48550781578170076890899053978500887751580,    .6153846153846153846153846153846153846154,
    .48790877731923892879351001283794175833480,    .6139088729016786570743405275779376498801,
    .49030398804519381705802061333088204264650,    .6124401913875598086124401913875598086124,
    .49269347544257524607047571407747454941280,    .6109785202863961813842482100238663484487,
    .49507726679785146739476431321236304938800,    .6095238095238095238095238095238095238095,
    .49745538920281889838648226032091770321130,    .6080760095011876484560570071258907363420,
    .49982786955644931126130359189119189977650,    .6066350710900473933649289099526066350711,
    .50219473456671548383667413872899487614650,    .6052009456264775413711583924349881796690,
    .50455601075239520092452494282042607665050,    .6037735849056603773584905660377358490566,
    .50691172444485432801997148999362252652650,    .6023529411764705882352941176470588235294,
    .50926190178980790257412536448100581765150,    .6009389671361502347417840375586854460094,
    .51160656874906207391973111953120678663250,    .5995316159250585480093676814988290398126,
    .51394575110223428282552049495279788970950,    .5981308411214953271028037383177570093458,
    .51627947444845445623684554448118433356300,    .5967365967365967365967365967365967365967,
    .51860776420804555186805373523384332656850,    .5953488372093023255813953488372093023256,
    .52093064562418522900344441950437612831600,    .5939675174013921113689095127610208816705,
    .52324814376454775732838697877014055848100,    .5925925925925925925925925925925925925926,
    .52556028352292727401362526507000438869000,    .5912240184757505773672055427251732101617,
    .52786708962084227803046587723656557500350,    .5898617511520737327188940092165898617512,
    .53016858660912158374145519701414741575700,    .5885057471264367816091954022988505747126,
    .53246479886947173376654518506256863474850,    .5871559633027522935779816513761467889908,
    .53475575061602764748158733709715306758900,    .5858123569794050343249427917620137299771,
    .53704146589688361856929077475797384977350,    .5844748858447488584474885844748858447489,
    .53932196859560876944783558428753167390800,    .5831435079726651480637813211845102505695,
    .54159728243274429804188230264117009937750,    .5818181818181818181818181818181818181818,
    .54386743096728351609669971367111429572100,    .5804988662131519274376417233560090702948,
    .54613243759813556721383065450936555862450,    .5791855203619909502262443438914027149321,
    .54839232556557315767520321969641372561450,    .5778781038374717832957110609480812641084,
    .55064711795266219063194057525834068655950,    .5765765765765765765765765765765765765766,
    .55289683768667763352766542084282264113450,    .5752808988764044943820224719101123595506,
    .55514150754050151093110798683483153581600,    .5739910313901345291479820627802690582960,
    .55738115013400635344709144192165695130850,    .5727069351230425055928411633109619686801,
    .55961578793542265941596269840374588966350,    .5714285714285714285714285714285714285714,
    .56184544326269181269140062795486301183700,    .5701559020044543429844097995545657015590,
    .56407013828480290218436721261241473257550,    .5688888888888888888888888888888888888889,
    .56628989502311577464155334382667206227800,    .5676274944567627494456762749445676274945,
    .56850473535266865532378233183408156037350,    .5663716814159292035398230088495575221239,
    .57071468100347144680739575051120482385150,    .5651214128035320088300220750551876379691,
    .57291975356178548306473885531886480748650,    .5638766519823788546255506607929515418502,
    .57511997447138785144460371157038025558000,    .5626373626373626373626373626373626373626,
    .57731536503482350219940144597785547375700,    .5614035087719298245614035087719298245614,
    .57950594641464214795689713355386629700650,    .5601750547045951859956236323851203501094,
    .58169173963462239562716149521293118596100,    .5589519650655021834061135371179039301310,
    .58387276558098266665552955601015128195300,    .5577342047930283224400871459694989106754,
    .58604904500357812846544902640744112432000,    .5565217391304347826086956521739130434783,
    .58822059851708596855957011939608491957200,    .5553145336225596529284164859002169197397,
    .59038744660217634674381770309992134571100,    .5541125541125541125541125541125541125541,
    .59254960960667157898740242671919986605650,    .5529157667386609071274298056155507559395,
    .59470710774669277576265358220553025603300,    .5517241379310344827586206896551724137931,
    .59685996110779382384237123915227130055450,    .5505376344086021505376344086021505376344,
    .59900818964608337768851242799428291618800,    .5493562231759656652360515021459227467811,
    .60115181318933474940990890900138765573500,    .5481798715203426124197002141327623126338,
    .60329085143808425240052883964381180703650,    .5470085470085470085470085470085470085470,
    .60542532396671688843525771517306566238400,    .5458422174840085287846481876332622601279,
    .60755525022454170969155029524699784815300,    .5446808510638297872340425531914893617021,
    .60968064953685519036241657886421307921400,    .5435244161358811040339702760084925690021,
    .61180154110599282990534675263916142284850,    .5423728813559322033898305084745762711864,
    .61391794401237043121710712512140162289150,    .5412262156448202959830866807610993657505,
    .61602987721551394351138242200249806046500,    .5400843881856540084388185654008438818565,
    .61813735955507864705538167982012964785100,    .5389473684210526315789473684210526315789,
    .62024040975185745772080281312810257077200,    .5378151260504201680672268907563025210084,
    .62233904640877868441606324267922900617100,    .5366876310272536687631027253668763102725,
    .62443328801189346144440150965237990021700,    .5355648535564853556485355648535564853556,
    .62652315293135274476554741340805776417250,    .5344467640918580375782881002087682672234,
    .62860865942237409420556559780379757285100,    .5333333333333333333333333333333333333333,
    .63068982562619868570408243613201193511500,    .5322245322245322245322245322245322245322,
    .63276666957103777644277897707070223987100,    .5311203319502074688796680497925311203320,
    .63483920917301017716738442686619237065300,    .5300207039337474120082815734989648033126,
    .63690746223706917739093569252872839570050,    .5289256198347107438016528925619834710744,
    .63897144645792069983514238629140891134750,    .5278350515463917525773195876288659793814,
    .64103117942093124081992527862894348800200,    .5267489711934156378600823045267489711934,
    .64308667860302726193566513757104985415950,    .5256673511293634496919917864476386036961,
    .64513796137358470073053240412264131009600,    .5245901639344262295081967213114754098361,
    .64718504499530948859131740391603671014300,    .5235173824130879345603271983640081799591,
    .64922794662510974195157587018911726772800,    .5224489795918367346938775510204081632653,
    .65126668331495807251485530287027359008800,    .5213849287169042769857433808553971486762,
    .65330127201274557080523663898929953575150,    .5203252032520325203252032520325203252033,
    .65533172956312757406749369692988693714150,    .5192697768762677484787018255578093306288,
    .65735807270835999727154330685152672231200,    .5182186234817813765182186234817813765182,
    .65938031808912778153342060249997302889800,    .5171717171717171717171717171717171717172,
    .66139848224536490484126716182800009846700,    .5161290322580645161290322580645161290323,
    .66341258161706617713093692145776003599150,    .5150905432595573440643863179074446680080,
    .66542263254509037562201001492212526500250,    .5140562248995983935742971887550200803213,
    .66742865127195616370414654738851822912700,    .5130260521042084168336673346693386773547,
    .66943065394262923906154583164607174694550,    .5120000000000000000000000000000000000000,
    .67142865660530226534774556057527661323550,    .5109780439121756487025948103792415169661,
    .67342267521216669923234121597488410770900,    .5099601593625498007968127490039840637450,
    .67541272562017662384192817626171745359900,    .5089463220675944333996023856858846918489,
    .67739882359180603188519853574689477682100,    .5079365079365079365079365079365079365079,
    .67938098479579733801614338517538271844400,    .5069306930693069306930693069306930693069,
    .68135922480790300781450241629499942064300,    .5059288537549407114624505928853754940711,
    .68333355911162063645036823800182901322850,    .5049309664694280078895463510848126232742,
    .68530400309891936760919861626462079584600,    .5039370078740157480314960629921259842520,
    .68727057207096020619019327568821609020250,    .5029469548133595284872298624754420432220,
    .68923328123880889251040571252815425395950,    .5019607843137254901960784313725490196078,
    .69314718055994530941723212145818, 5.0e-01,
};

static const float64_t ln_2(0.69314718055994530941723212145818);

float32_t f32_log(float32_t x)
{
    //special cases
    if(x.isNaN() || x < float32_t::zero()) return float32_t::nan();
    if(x == float32_t::zero()) return -float32_t::inf();

    //first 8 bits of mantissa
    int h0 = (x.v >> (23 - LOGTAB_SCALE)) & ((1 << LOGTAB_SCALE) - 1);
    //buf == 0.00000000_the_rest_mantissa_bits
    float64_t buf; buf.v = packToF64UI(0, 1023, ((uint64_t)x.v << 29) & ((1LL << (52 - LOGTAB_SCALE)) - 1));
    buf -= float64_t::one();

    float64_t tab0(icvLogTab[2*h0]);
    float64_t tab1(icvLogTab[2*h0+1]);

    float64_t x0 = buf * tab1;
    //if last elements of icvLogTab
    if(h0 == 255) x0 += float64_t(-1./512);

    float64_t y0 = ln_2 * float64_t(expF32UI(x.v) - 127) + tab0 + x0*x0*x0/float64_t(3) - x0*x0/float64_t(2) + x0;

    return y0;
}

float64_t f64_log(float64_t x)
{
    //special cases
    if(x.isNaN() || x < float64_t::zero()) return float64_t::nan();
    if(x == float64_t::zero()) return -float64_t::inf();

    static const float64_t
    A7(1.0),
    A6(-0.5),
    A5(0.333333333333333314829616256247390992939472198486328125),
    A4(-0.25),
    A3(0.2),
    A2(-0.1666666666666666574148081281236954964697360992431640625),
    A1(0.1428571428571428769682682968777953647077083587646484375),
    A0(-0.125);

    //first 8 bits of mantissa
    int h0 = (x.v >> (52 - LOGTAB_SCALE)) & ((1 << LOGTAB_SCALE) - 1);
    //buf == 0.00000000_the_rest_mantissa_bits
    float64_t buf; buf.v = packToF64UI(0, 1023, x.v & ((1LL << (52 - LOGTAB_SCALE)) - 1));
    buf -= float64_t::one();

    float64_t tab0(icvLogTab[2*h0]);
    float64_t tab1(icvLogTab[2*h0 + 1]);

    float64_t x0 = buf * tab1;
    //if last elements of icvLogTab
    if(h0 == 255) x0 += float64_t(-1./512);
    float64_t xq = x0*x0;

    return ln_2 * float64_t( expF64UI(x.v) - 1023) + tab0 + (((A0*xq + A2)*xq + A4)*xq + A6)*xq +
           (((A1*xq + A3)*xq + A5)*xq + A7)*x0;
}

/* ************************************************************************** *\
   Fast cube root by Ken Turkowski
   (http://www.worldserver.com/turk/computergraphics/papers.html)
\* ************************************************************************** */
float32_t f32_cbrt(float32_t x)
{
    //special cases
    if(x.isNaN()) return float32_t::nan();
    if(x.isInf()) return x;

    int s = signF32UI(x.v);
    int ex = expF32UI(x.v) - 127;
    int shx = ex % 3;
    shx -= shx >= 0 ? 3 : 0;
    ex = (ex - shx) / 3 - 1; /* exponent of cube root */
    float64_t fr; fr.v = packToF64UI(0, shx + 1023, ((uint64_t)fracF32UI(x.v)) << 29);

    /* 0.125 <= fr < 1.0 */
    /* Use quartic rational polynomial with error < 2^(-24) */
    const float64_t A1(45.2548339756803022511987494);
    const float64_t A2(192.2798368355061050458134625);
    const float64_t A3(119.1654824285581628956914143);
    const float64_t A4(13.43250139086239872172837314);
    const float64_t A5(0.1636161226585754240958355063);
    const float64_t A6(14.80884093219134573786480845);
    const float64_t A7(151.9714051044435648658557668);
    const float64_t A8(168.5254414101568283957668343);
    const float64_t A9(33.9905941350215598754191872);
    const float64_t A10(1.0);
    fr = ((((A1 * fr + A2) * fr + A3) * fr + A4) * fr + A5)/
         ((((A6 * fr + A7) * fr + A8) * fr + A9) * fr + A10);
    /* fr *= 2^ex * sign */

    // checks for "+0" and "-0", reset sign bit
    float32_t y; y.v = ((x.v & ((1u << 31) - 1)) != 0) ? packToF32UI(s, ex+127, (uint32_t)(fracF64UI(fr.v) >> 29)) : 0;
    return y;
}

/// POW functions ///

float32_t f32_pow( float32_t x, float32_t y)
{
    const float32_t zero = float32_t::zero(), one = float32_t::one(), inf = float32_t::inf(), nan = float32_t::nan();
    bool xinf = x.isInf(), yinf = y.isInf(), xnan = x.isNaN(), ynan = y.isNaN();
    float32_t ax = abs(x);
    bool useInf = (y > zero) == (ax > one);
    float32_t v;
    //special cases
    if(ynan) v = nan;
    else if(yinf) v = (ax == one || xnan) ? nan : (useInf ? inf : zero);
    else if(y == zero) v = one;
    else if(y == one ) v = x;
    else //here y is ok
    {
        if(xnan) v = nan;
        else if(xinf) v = (y < zero) ? zero : inf;
        else if(y == f32_roundToInt(y, round_near_even, false)) v = f32_powi(x, f32_to_i32(y, round_near_even, false));
        else if(x  < zero) v = nan;
        // (0 ** 0) == 1
        else if(x == zero) v = (y < zero) ? inf : (y == zero ? one : zero);
        // here x and y are ok
        else v = f32_exp(y * f32_log(x));
    }

    return v;
}

float64_t f64_pow( float64_t x, float64_t y)
{
    const float64_t zero = float64_t::zero(), one = float64_t::one(), inf = float64_t::inf(), nan = float64_t::nan();
    bool xinf = x.isInf(), yinf = y.isInf(), xnan = x.isNaN(), ynan = y.isNaN();
    float64_t ax = abs(x);
    bool useInf = (y > zero) == (ax > one);
    float64_t v;
    //special cases
    if(ynan) v = nan;
    else if(yinf) v = (ax == one || xnan) ? nan : (useInf ? inf : zero);
    else if(y == zero) v = one;
    else if(y == one ) v = x;
    else //here y is ok
    {
        if(xnan) v = nan;
        else if(xinf) v = (y < zero) ? zero : inf;
        else if(y == f64_roundToInt(y, round_near_even, false)) v = f64_powi(x, f64_to_i32(y, round_near_even, false));
        else if(x  < zero) v = nan;
        // (0 ** 0) == 1
        else if(x == zero) v = (y < zero) ? inf : (y == zero ? one : zero);
        // here x and y are ok
        else v = f64_exp(y * f64_log(x));
    }

    return v;
}

// These functions are for internal use only

float32_t f32_powi( float32_t x, int y)
{
    float32_t v;
    //special case: (0 ** 0) == 1
    if(x == float32_t::zero())
        v = (y < 0) ? float32_t::inf() : (y == 0 ? float32_t::one() : float32_t::zero());
    // here x and y are ok
    else
    {
        float32_t a = float32_t::one(), b = x;
        int p = std::abs(y);
        if( y < 0 )
            b = float32_t::one()/b;
        while( p > 1 )
        {
            if( p & 1 )
                a *= b;
            b *= b;
            p >>= 1;
        }
        v = a * b;
    }

    return v;
}

float64_t f64_powi( float64_t x, int y)
{
    float64_t v;
    //special case: (0 ** 0) == 1
    if(x == float64_t::zero())
        v = (y < 0) ? float64_t::inf() : (y == 0 ? float64_t::one() : float64_t::zero());
    // here x and y are ok
    else
    {
        float64_t a = float64_t::one(), b = x;
        int p = std::abs(y);
        if( y < 0 )
            b = float64_t::one()/b;
        while( p > 1 )
        {
            if( p & 1 )
                a *= b;
            b *= b;
            p >>= 1;
        }
        v = a * b;
    }

    return v;
}

}
