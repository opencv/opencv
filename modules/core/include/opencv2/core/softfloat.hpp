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

#include <stdbool.h>
#include <stdint.h>
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
    softfloat_tininess_beforeRounding = 0,
    softfloat_tininess_afterRounding  = 1
};
//fixed to make softfloat code stateless
const uint_fast8_t softfloat_detectTininess = softfloat_tininess_afterRounding;

/*----------------------------------------------------------------------------
| Software floating-point rounding mode.
*----------------------------------------------------------------------------*/
enum {
    softfloat_round_near_even   = 0,
    softfloat_round_minMag      = 1,
    softfloat_round_min         = 2,
    softfloat_round_max         = 3,
    softfloat_round_near_maxMag = 4,
    softfloat_round_odd         = 5
};
//fixed to make softfloat code stateless
const uint_fast8_t softfloat_roundingMode = softfloat_round_near_even;

/*----------------------------------------------------------------------------
| Software floating-point exception flags.
*----------------------------------------------------------------------------*/
enum {
    softfloat_flag_inexact   =  1,
    softfloat_flag_underflow =  2,
    softfloat_flag_overflow  =  4,
    softfloat_flag_infinite  =  8,
    softfloat_flag_invalid   = 16
};

// Disabled to make softfloat code stateless
// The user has to check values manually with *_isSignalingNaN() functions
/*----------------------------------------------------------------------------
| Raises the exceptions specified by `flags'.  Floating-point traps can be
| defined here if desired.  It is currently not possible for such a trap
| to substitute a result value.  If traps are not implemented, this routine
| should be simply `softfloat_exceptionFlags |= flags;'.
*----------------------------------------------------------------------------*/
inline void softfloat_raiseFlags( uint_fast8_t /* flags */)
{
    //softfloat_exceptionFlags |= flags;
}

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

inline float32_t  operator + (const float32_t& a, const float32_t& b);
inline float32_t  operator - (const float32_t& a, const float32_t& b);
inline float32_t  operator * (const float32_t& a, const float32_t& b);
inline float32_t  operator / (const float32_t& a, const float32_t& b);

inline float32_t operator += (const float32_t& a, const float32_t& b);
inline float32_t operator -= (const float32_t& a, const float32_t& b);
inline float32_t operator *= (const float32_t& a, const float32_t& b);
inline float32_t operator /= (const float32_t& a, const float32_t& b);

inline float64_t operator + (const float64_t& a, const float64_t& b);
inline float64_t operator - (const float64_t& a, const float64_t& b);
inline float64_t operator * (const float64_t& a, const float64_t& b);
inline float64_t operator / (const float64_t& a, const float64_t& b);

inline float64_t operator += (const float64_t& a, const float64_t& b);
inline float64_t operator -= (const float64_t& a, const float64_t& b);
inline float64_t operator *= (const float64_t& a, const float64_t& b);
inline float64_t operator /= (const float64_t& a, const float64_t& b);

//TODO: process flags
inline float32_t  operator + (const float32_t& a, const float32_t& b)
{
    return f32_add(a, b);
}
inline float32_t  operator - (const float32_t& a, const float32_t& b)
{
    return f32_sub(a, b);
}
inline float32_t  operator * (const float32_t& a, const float32_t& b)
{
    return f32_mul(a, b);
}
inline float32_t  operator / (const float32_t& a, const float32_t& b)
{
    return f32_div(a, b);
}

inline float32_t operator += (const float32_t& a, const float32_t& b)
{
    return (a + b);
}
inline float32_t operator -= (const float32_t& a, const float32_t& b)
{
    return (a - b);
}
inline float32_t operator *= (const float32_t& a, const float32_t& b)
{
    return (a * b);
}
inline float32_t operator /= (const float32_t& a, const float32_t& b)
{
    return (a / b);
}

inline float64_t  operator + (const float64_t& a, const float64_t& b)
{
    return f64_add(a, b);
}
inline float64_t  operator - (const float64_t& a, const float64_t& b)
{
    return f64_sub(a, b);
}
inline float64_t  operator * (const float64_t& a, const float64_t& b)
{
    return f64_mul(a, b);
}
inline float64_t  operator / (const float64_t& a, const float64_t& b)
{
    return f64_div(a, b);
}

inline float64_t operator += (const float64_t& a, const float64_t& b)
{
    return (a + b);
}
inline float64_t operator -= (const float64_t& a, const float64_t& b)
{
    return (a - b);
}
inline float64_t operator *= (const float64_t& a, const float64_t& b)
{
    return (a * b);
}
inline float64_t operator /= (const float64_t& a, const float64_t& b)
{
    return (a / b);
}

float32_t f32_exp( float32_t );
float32_t f32_log( float32_t );
float32_t f32_pow( float32_t, float32_t );

float64_t f64_exp( float64_t );
float64_t f64_log( float64_t );
float64_t f64_pow( float64_t, float64_t );

}

}

#endif
