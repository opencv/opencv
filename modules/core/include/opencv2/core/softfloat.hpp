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

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct softfloat32_t;
struct softfloat64_t;

struct CV_EXPORTS softfloat32_t
{
public:
    softfloat32_t() { v = 0; }
    softfloat32_t( const softfloat32_t& c) { v = c.v; }
    softfloat32_t& operator=( const softfloat32_t& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    static const softfloat32_t fromRaw( const uint32_t a ) { softfloat32_t x; x.v = a; return x; }

    softfloat32_t( const uint32_t );
    softfloat32_t( const uint64_t );
    softfloat32_t( const int32_t );
    softfloat32_t( const int64_t );
    softfloat32_t( const float a ) { Cv32suf s; s.f = a; v = s.u; }

    uint_fast32_t toUI32( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;
    uint_fast64_t toUI64( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;
    int_fast32_t   toI32( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;
    int_fast64_t   toI64( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;

    softfloat32_t  round( uint_fast8_t roundingMode = round_near_even, bool exact = false) const;
    softfloat64_t toF64() const;
    float toFloat() const { Cv32suf s; s.u = v; return s.f; }

    softfloat32_t operator + (const softfloat32_t&) const;
    softfloat32_t operator - (const softfloat32_t&) const;
    softfloat32_t operator * (const softfloat32_t&) const;
    softfloat32_t operator / (const softfloat32_t&) const;
    softfloat32_t operator % (const softfloat32_t&) const;
    softfloat32_t operator - () const { softfloat32_t x; x.v = v ^ (1U << 31); return x; }

    softfloat32_t& operator += (const softfloat32_t& a) { *this = *this + a; return *this; }
    softfloat32_t& operator -= (const softfloat32_t& a) { *this = *this - a; return *this; }
    softfloat32_t& operator *= (const softfloat32_t& a) { *this = *this * a; return *this; }
    softfloat32_t& operator /= (const softfloat32_t& a) { *this = *this / a; return *this; }
    softfloat32_t& operator %= (const softfloat32_t& a) { *this = *this % a; return *this; }

    bool operator == ( const softfloat32_t& ) const;
    bool operator != ( const softfloat32_t& ) const;
    bool operator >  ( const softfloat32_t& ) const;
    bool operator >= ( const softfloat32_t& ) const;
    bool operator <  ( const softfloat32_t& ) const;
    bool operator <= ( const softfloat32_t& ) const;

    bool isNaN() const { return (v & 0x7fffffff)  > 0x7f800000; }
    bool isInf() const { return (v & 0x7fffffff) == 0x7f800000; }

    static softfloat32_t zero() { return softfloat32_t::fromRaw( 0 ); }
    static softfloat32_t  inf() { return softfloat32_t::fromRaw( 0xFF << 23 ); }
    static softfloat32_t  nan() { return softfloat32_t::fromRaw( 0x7fffffff ); }
    static softfloat32_t  one() { return softfloat32_t::fromRaw(  127 << 23 ); }

    uint32_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct CV_EXPORTS softfloat64_t
{
public:
    softfloat64_t() { }
    softfloat64_t( const softfloat64_t& c) { v = c.v; }
    softfloat64_t& operator=( const softfloat64_t& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    static softfloat64_t fromRaw( const uint64_t a ) { softfloat64_t x; x.v = a; return x; }

    softfloat64_t( const uint32_t );
    softfloat64_t( const uint64_t );
    softfloat64_t( const  int32_t );
    softfloat64_t( const  int64_t );
    softfloat64_t( const double a ) { Cv64suf s; s.f = a; v = s.u; }

    uint_fast32_t toUI32( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;
    uint_fast64_t toUI64( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;
    int_fast32_t   toI32( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;
    int_fast64_t   toI64( uint_fast8_t roundingMode = round_near_even, bool exact = false ) const;

    softfloat64_t  round( uint_fast8_t roundingMode = round_near_even, bool exact = false) const;
    softfloat32_t toF32() const;
    double toDouble() const { Cv64suf s; s.u = v; return s.f; }

    softfloat64_t operator + (const softfloat64_t&) const;
    softfloat64_t operator - (const softfloat64_t&) const;
    softfloat64_t operator * (const softfloat64_t&) const;
    softfloat64_t operator / (const softfloat64_t&) const;
    softfloat64_t operator % (const softfloat64_t&) const;
    softfloat64_t operator - () const { softfloat64_t x; x.v = v ^ (1ULL << 63); return x; }

    softfloat64_t& operator += (const softfloat64_t& a) { *this = *this + a; return *this; }
    softfloat64_t& operator -= (const softfloat64_t& a) { *this = *this - a; return *this; }
    softfloat64_t& operator *= (const softfloat64_t& a) { *this = *this * a; return *this; }
    softfloat64_t& operator /= (const softfloat64_t& a) { *this = *this / a; return *this; }
    softfloat64_t& operator %= (const softfloat64_t& a) { *this = *this % a; return *this; }

    bool operator == ( const softfloat64_t& ) const;
    bool operator != ( const softfloat64_t& ) const;
    bool operator >  ( const softfloat64_t& ) const;
    bool operator >= ( const softfloat64_t& ) const;
    bool operator <  ( const softfloat64_t& ) const;
    bool operator <= ( const softfloat64_t& ) const;

    bool isNaN() const { return (v & 0x7fffffffffffffff)  > 0x7ff0000000000000; }
    bool isInf() const { return (v & 0x7fffffffffffffff) == 0x7ff0000000000000; }

    static softfloat64_t zero() { return softfloat64_t::fromRaw( 0 ); }
    static softfloat64_t  inf() { return softfloat64_t::fromRaw( (uint_fast64_t)(0x7FF) << 52 ); }
    static softfloat64_t  nan() { return softfloat64_t::fromRaw( CV_BIG_INT(0x7FFFFFFFFFFFFFFF) ); }
    static softfloat64_t  one() { return softfloat64_t::fromRaw( (uint_fast64_t)( 1023) << 52 ); }

    uint64_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

CV_EXPORTS softfloat32_t f32_mulAdd( softfloat32_t, softfloat32_t, softfloat32_t );
CV_EXPORTS softfloat32_t f32_sqrt( softfloat32_t );
CV_EXPORTS softfloat64_t f64_mulAdd( softfloat64_t, softfloat64_t, softfloat64_t );
CV_EXPORTS softfloat64_t f64_sqrt( softfloat64_t );

/*----------------------------------------------------------------------------
| Ported from OpenCV and added for usability
*----------------------------------------------------------------------------*/

inline softfloat32_t min(const softfloat32_t a, const softfloat32_t b);
inline softfloat64_t min(const softfloat64_t a, const softfloat64_t b);

inline softfloat32_t max(const softfloat32_t a, const softfloat32_t b);
inline softfloat64_t max(const softfloat64_t a, const softfloat64_t b);

inline softfloat32_t min(const softfloat32_t a, const softfloat32_t b) { return (a > b) ? b : a; }
inline softfloat64_t min(const softfloat64_t a, const softfloat64_t b) { return (a > b) ? b : a; }

inline softfloat32_t max(const softfloat32_t a, const softfloat32_t b) { return (a > b) ? a : b; }
inline softfloat64_t max(const softfloat64_t a, const softfloat64_t b) { return (a > b) ? a : b; }

inline softfloat32_t f32_abs( softfloat32_t a) { softfloat32_t x; x.v = a.v & ((1U   << 31) - 1); return x; }
inline softfloat64_t f64_abs( softfloat64_t a) { softfloat64_t x; x.v = a.v & ((1ULL << 63) - 1); return x; }

CV_EXPORTS softfloat32_t f32_exp( softfloat32_t );
CV_EXPORTS softfloat32_t f32_log( softfloat32_t );
CV_EXPORTS softfloat32_t f32_pow( softfloat32_t, softfloat32_t );

CV_EXPORTS softfloat64_t f64_exp( softfloat64_t );
CV_EXPORTS softfloat64_t f64_log( softfloat64_t );
CV_EXPORTS softfloat64_t f64_pow( softfloat64_t, softfloat64_t );

CV_EXPORTS softfloat32_t f32_cbrt( softfloat32_t );

}

#endif
