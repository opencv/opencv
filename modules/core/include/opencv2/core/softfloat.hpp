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

namespace cv
{

namespace softfloat
{

/*----------------------------------------------------------------------------
| Types used to pass 32-bit and 64-bit floating-point
| arguments and results to/from functions.  These types must be exactly
| 32 bits and 64 bits in size, respectively.  Where a
| platform has "native" support for IEEE-Standard floating-point formats,
| the types below may, if desired, be defined as aliases for the native types
| (typically 'float' and 'double').
*----------------------------------------------------------------------------*/
typedef struct { uint32_t v; } float32_t;
typedef struct { uint64_t v; } float64_t;

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
// The user has to check values manually with *_isSignalingNaN() functions
CV_INLINE void raiseFlags( uint_fast8_t /* flags */)
{
    //exceptionFlags |= flags;
}

/*----------------------------------------------------------------------------
| Integer-to-floating-point conversion routines.
*----------------------------------------------------------------------------*/
CV_EXPORTS float32_t ui32_to_f32( uint32_t );
CV_EXPORTS float64_t ui32_to_f64( uint32_t );
CV_EXPORTS float32_t ui64_to_f32( uint64_t );
CV_EXPORTS float64_t ui64_to_f64( uint64_t );
CV_EXPORTS float32_t i32_to_f32( int32_t );
CV_EXPORTS float64_t i32_to_f64( int32_t );
CV_EXPORTS float32_t i64_to_f32( int64_t );
CV_EXPORTS float64_t i64_to_f64( int64_t );

/*----------------------------------------------------------------------------
| 32-bit (single-precision) floating-point operations.
*----------------------------------------------------------------------------*/
CV_EXPORTS uint_fast32_t f32_to_ui32( float32_t, uint_fast8_t, bool );
CV_EXPORTS uint_fast64_t f32_to_ui64( float32_t, uint_fast8_t, bool );
CV_EXPORTS int_fast32_t f32_to_i32( float32_t, uint_fast8_t, bool );
CV_EXPORTS int_fast64_t f32_to_i64( float32_t, uint_fast8_t, bool );
CV_EXPORTS uint_fast32_t f32_to_ui32_r_minMag( float32_t, bool );
CV_EXPORTS uint_fast64_t f32_to_ui64_r_minMag( float32_t, bool );
CV_EXPORTS int_fast32_t f32_to_i32_r_minMag( float32_t, bool );
CV_EXPORTS int_fast64_t f32_to_i64_r_minMag( float32_t, bool );
CV_EXPORTS float64_t f32_to_f64( float32_t );
CV_EXPORTS float32_t f32_roundToInt( float32_t, uint_fast8_t, bool );
CV_EXPORTS float32_t f32_add( float32_t, float32_t );
CV_EXPORTS float32_t f32_sub( float32_t, float32_t );
CV_EXPORTS float32_t f32_mul( float32_t, float32_t );
CV_EXPORTS float32_t f32_mulAdd( float32_t, float32_t, float32_t );
CV_EXPORTS float32_t f32_div( float32_t, float32_t );
CV_EXPORTS float32_t f32_rem( float32_t, float32_t );
CV_EXPORTS float32_t f32_sqrt( float32_t );
CV_EXPORTS bool f32_eq( float32_t, float32_t );
CV_EXPORTS bool f32_le( float32_t, float32_t );
CV_EXPORTS bool f32_lt( float32_t, float32_t );
CV_EXPORTS bool f32_eq_signaling( float32_t, float32_t );
CV_EXPORTS bool f32_le_quiet( float32_t, float32_t );
CV_EXPORTS bool f32_lt_quiet( float32_t, float32_t );
CV_EXPORTS bool f32_isSignalingNaN( float32_t );

/*----------------------------------------------------------------------------
| 64-bit (double-precision) floating-point operations.
*----------------------------------------------------------------------------*/
CV_EXPORTS uint_fast32_t f64_to_ui32( float64_t, uint_fast8_t, bool );
CV_EXPORTS uint_fast64_t f64_to_ui64( float64_t, uint_fast8_t, bool );
CV_EXPORTS int_fast32_t f64_to_i32( float64_t, uint_fast8_t, bool );
CV_EXPORTS int_fast64_t f64_to_i64( float64_t, uint_fast8_t, bool );
CV_EXPORTS uint_fast32_t f64_to_ui32_r_minMag( float64_t, bool );
CV_EXPORTS uint_fast64_t f64_to_ui64_r_minMag( float64_t, bool );
CV_EXPORTS int_fast32_t f64_to_i32_r_minMag( float64_t, bool );
CV_EXPORTS int_fast64_t f64_to_i64_r_minMag( float64_t, bool );
CV_EXPORTS float32_t f64_to_f32( float64_t );
CV_EXPORTS float64_t f64_roundToInt( float64_t, uint_fast8_t, bool );
CV_EXPORTS float64_t f64_add( float64_t, float64_t );
CV_EXPORTS float64_t f64_sub( float64_t, float64_t );
CV_EXPORTS float64_t f64_mul( float64_t, float64_t );
CV_EXPORTS float64_t f64_mulAdd( float64_t, float64_t, float64_t );
CV_EXPORTS float64_t f64_div( float64_t, float64_t );
CV_EXPORTS float64_t f64_rem( float64_t, float64_t );
CV_EXPORTS float64_t f64_sqrt( float64_t );
CV_EXPORTS bool f64_eq( float64_t, float64_t );
CV_EXPORTS bool f64_le( float64_t, float64_t );
CV_EXPORTS bool f64_lt( float64_t, float64_t );
CV_EXPORTS bool f64_eq_signaling( float64_t, float64_t );
CV_EXPORTS bool f64_le_quiet( float64_t, float64_t );
CV_EXPORTS bool f64_lt_quiet( float64_t, float64_t );
CV_EXPORTS bool f64_isSignalingNaN( float64_t );

/*----------------------------------------------------------------------------
| Ported from OpenCV and added for usability
*----------------------------------------------------------------------------*/

CV_INLINE float32_t float_to_f32 (const float&);
CV_INLINE float64_t double_to_f64(const double&);
CV_INLINE float  f32_to_float (const float32_t&);
CV_INLINE double f64_to_double(const float64_t&);

CV_INLINE float32_t  operator + (const float32_t& a, const float32_t& b);
CV_INLINE float32_t  operator - (const float32_t& a, const float32_t& b);
CV_INLINE float32_t  operator * (const float32_t& a, const float32_t& b);
CV_INLINE float32_t  operator / (const float32_t& a, const float32_t& b);
CV_INLINE float32_t  operator - (const float32_t& a);

CV_INLINE float32_t& operator += (float32_t& a, const float32_t& b);
CV_INLINE float32_t& operator -= (float32_t& a, const float32_t& b);
CV_INLINE float32_t& operator *= (float32_t& a, const float32_t& b);
CV_INLINE float32_t& operator /= (float32_t& a, const float32_t& b);

CV_INLINE float64_t operator + (const float64_t& a, const float64_t& b);
CV_INLINE float64_t operator - (const float64_t& a, const float64_t& b);
CV_INLINE float64_t operator * (const float64_t& a, const float64_t& b);
CV_INLINE float64_t operator / (const float64_t& a, const float64_t& b);
CV_INLINE float64_t operator - (const float64_t& a);

CV_INLINE float64_t& operator += (float64_t& a, const float64_t& b);
CV_INLINE float64_t& operator -= (float64_t& a, const float64_t& b);
CV_INLINE float64_t& operator *= (float64_t& a, const float64_t& b);
CV_INLINE float64_t& operator /= (float64_t& a, const float64_t& b);

CV_INLINE bool operator == (const float32_t& a, const float32_t& b);
CV_INLINE bool operator != (const float32_t& a, const float32_t& b);
CV_INLINE bool operator >  (const float32_t& a, const float32_t& b);
CV_INLINE bool operator >= (const float32_t& a, const float32_t& b);
CV_INLINE bool operator <  (const float32_t& a, const float32_t& b);
CV_INLINE bool operator <= (const float32_t& a, const float32_t& b);

CV_INLINE bool operator == (const float64_t& a, const float64_t& b);
CV_INLINE bool operator != (const float64_t& a, const float64_t& b);
CV_INLINE bool operator >  (const float64_t& a, const float64_t& b);
CV_INLINE bool operator >= (const float64_t& a, const float64_t& b);
CV_INLINE bool operator <  (const float64_t& a, const float64_t& b);
CV_INLINE bool operator <= (const float64_t& a, const float64_t& b);

CV_INLINE float32_t float_to_f32 (const float&  a){ return *((float32_t*) &a); }
CV_INLINE float64_t double_to_f64(const double& a){ return *((float64_t*) &a); }
CV_INLINE float  f32_to_float (const float32_t& a){ return *((float*)  &a); }
CV_INLINE double f64_to_double(const float64_t& a){ return *((double*) &a); }

CV_INLINE float32_t  operator + (const float32_t& a, const float32_t& b) { return f32_add(a, b); }
CV_INLINE float32_t  operator - (const float32_t& a, const float32_t& b) { return f32_sub(a, b); }
CV_INLINE float32_t  operator * (const float32_t& a, const float32_t& b) { return f32_mul(a, b); }
CV_INLINE float32_t  operator / (const float32_t& a, const float32_t& b) { return f32_div(a, b); }
CV_INLINE float32_t  operator - (const float32_t& a) { float32_t x = {a.v ^ (1U << 31)}; return x; }

CV_INLINE float32_t& operator += (float32_t& a, const float32_t& b) { a = a + b; return a; }
CV_INLINE float32_t& operator -= (float32_t& a, const float32_t& b) { a = a - b; return a; }
CV_INLINE float32_t& operator *= (float32_t& a, const float32_t& b) { a = a * b; return a; }
CV_INLINE float32_t& operator /= (float32_t& a, const float32_t& b) { a = a / b; return a; }

CV_INLINE float64_t  operator + (const float64_t& a, const float64_t& b) { return f64_add(a, b); }
CV_INLINE float64_t  operator - (const float64_t& a, const float64_t& b) { return f64_sub(a, b); }
CV_INLINE float64_t  operator * (const float64_t& a, const float64_t& b) { return f64_mul(a, b); }
CV_INLINE float64_t  operator / (const float64_t& a, const float64_t& b) { return f64_div(a, b); }
CV_INLINE float64_t  operator - (const float64_t& a) { float64_t x = {a.v ^ (1ULL << 63)}; return x; }

CV_INLINE float64_t& operator += (float64_t& a, const float64_t& b) { a = a + b; return a; }
CV_INLINE float64_t& operator -= (float64_t& a, const float64_t& b) { a = a - b; return a; }
CV_INLINE float64_t& operator *= (float64_t& a, const float64_t& b) { a = a * b; return a; }
CV_INLINE float64_t& operator /= (float64_t& a, const float64_t& b) { a = a / b; return a; }

CV_INLINE bool operator == (const float32_t& a, const float32_t& b) { return  f32_eq(a, b); }
CV_INLINE bool operator != (const float32_t& a, const float32_t& b) { return !f32_eq(a, b); }
CV_INLINE bool operator >  (const float32_t& a, const float32_t& b) { return  f32_lt(b, a); }
CV_INLINE bool operator >= (const float32_t& a, const float32_t& b) { return  f32_le(b, a); }
CV_INLINE bool operator <  (const float32_t& a, const float32_t& b) { return  f32_lt(a, b); }
CV_INLINE bool operator <= (const float32_t& a, const float32_t& b) { return  f32_le(a, b); }

CV_INLINE bool operator == (const float64_t& a, const float64_t& b) { return  f64_eq(a, b); }
CV_INLINE bool operator != (const float64_t& a, const float64_t& b) { return !f64_eq(a, b); }
CV_INLINE bool operator >  (const float64_t& a, const float64_t& b) { return  f64_lt(b, a); }
CV_INLINE bool operator >= (const float64_t& a, const float64_t& b) { return  f64_le(b, a); }
CV_INLINE bool operator <  (const float64_t& a, const float64_t& b) { return  f64_lt(a, b); }
CV_INLINE bool operator <= (const float64_t& a, const float64_t& b) { return  f64_le(a, b); }

CV_EXPORTS float32_t f32_exp( float32_t );
CV_EXPORTS float32_t f32_log( float32_t );
CV_EXPORTS float32_t f32_pow( float32_t, float32_t );

CV_EXPORTS float64_t f64_exp( float64_t );
CV_EXPORTS float64_t f64_log( float64_t );
CV_EXPORTS float64_t f64_pow( float64_t, float64_t );

CV_EXPORTS float32_t f32_cbrt( float32_t );
}

}

#endif
