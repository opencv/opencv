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

struct softfloat;
struct softdouble;

struct CV_EXPORTS softfloat
{
public:
    softfloat() { v = 0; }
    softfloat( const softfloat& c) { v = c.v; }
    softfloat& operator=( const softfloat& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    static const softfloat fromRaw( const uint32_t a ) { softfloat x; x.v = a; return x; }

    explicit softfloat( const uint32_t );
    explicit softfloat( const uint64_t );
    explicit softfloat( const int32_t );
    explicit softfloat( const int64_t );
    explicit softfloat( const float a ) { Cv32suf s; s.f = a; v = s.u; }

    operator softdouble() const;
    operator float() const { Cv32suf s; s.u = v; return s.f; }

    softfloat operator + (const softfloat&) const;
    softfloat operator - (const softfloat&) const;
    softfloat operator * (const softfloat&) const;
    softfloat operator / (const softfloat&) const;
    softfloat operator % (const softfloat&) const;
    softfloat operator - () const { softfloat x; x.v = v ^ (1U << 31); return x; }

    softfloat& operator += (const softfloat& a) { *this = *this + a; return *this; }
    softfloat& operator -= (const softfloat& a) { *this = *this - a; return *this; }
    softfloat& operator *= (const softfloat& a) { *this = *this * a; return *this; }
    softfloat& operator /= (const softfloat& a) { *this = *this / a; return *this; }
    softfloat& operator %= (const softfloat& a) { *this = *this % a; return *this; }

    bool operator == ( const softfloat& ) const;
    bool operator != ( const softfloat& ) const;
    bool operator >  ( const softfloat& ) const;
    bool operator >= ( const softfloat& ) const;
    bool operator <  ( const softfloat& ) const;
    bool operator <= ( const softfloat& ) const;

    bool isNaN() const { return (v & 0x7fffffff)  > 0x7f800000; }
    bool isInf() const { return (v & 0x7fffffff) == 0x7f800000; }

    static softfloat zero() { return softfloat::fromRaw( 0 ); }
    static softfloat  inf() { return softfloat::fromRaw( 0xFF << 23 ); }
    static softfloat  nan() { return softfloat::fromRaw( 0x7fffffff ); }
    static softfloat  one() { return softfloat::fromRaw(  127 << 23 ); }

    uint32_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct CV_EXPORTS softdouble
{
public:
    softdouble() : v(0) { }
    softdouble( const softdouble& c) { v = c.v; }
    softdouble& operator=( const softdouble& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    static softdouble fromRaw( const uint64_t a ) { softdouble x; x.v = a; return x; }

    explicit softdouble( const uint32_t );
    explicit softdouble( const uint64_t );
    explicit softdouble( const  int32_t );
    explicit softdouble( const  int64_t );
    explicit softdouble( const double a ) { Cv64suf s; s.f = a; v = s.u; }

    operator softfloat() const;
    operator double() const { Cv64suf s; s.u = v; return s.f; }

    softdouble operator + (const softdouble&) const;
    softdouble operator - (const softdouble&) const;
    softdouble operator * (const softdouble&) const;
    softdouble operator / (const softdouble&) const;
    softdouble operator % (const softdouble&) const;
    softdouble operator - () const { softdouble x; x.v = v ^ (1ULL << 63); return x; }

    softdouble& operator += (const softdouble& a) { *this = *this + a; return *this; }
    softdouble& operator -= (const softdouble& a) { *this = *this - a; return *this; }
    softdouble& operator *= (const softdouble& a) { *this = *this * a; return *this; }
    softdouble& operator /= (const softdouble& a) { *this = *this / a; return *this; }
    softdouble& operator %= (const softdouble& a) { *this = *this % a; return *this; }

    bool operator == ( const softdouble& ) const;
    bool operator != ( const softdouble& ) const;
    bool operator >  ( const softdouble& ) const;
    bool operator >= ( const softdouble& ) const;
    bool operator <  ( const softdouble& ) const;
    bool operator <= ( const softdouble& ) const;

    bool isNaN() const { return (v & 0x7fffffffffffffff)  > 0x7ff0000000000000; }
    bool isInf() const { return (v & 0x7fffffffffffffff) == 0x7ff0000000000000; }

    static softdouble zero() { return softdouble::fromRaw( 0 ); }
    static softdouble  inf() { return softdouble::fromRaw( (uint_fast64_t)(0x7FF) << 52 ); }
    static softdouble  nan() { return softdouble::fromRaw( CV_BIG_INT(0x7FFFFFFFFFFFFFFF) ); }
    static softdouble  one() { return softdouble::fromRaw( (uint_fast64_t)( 1023) << 52 ); }

    uint64_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

CV_EXPORTS softfloat  mulAdd( const softfloat&  a, const softfloat&  b, const softfloat & c);
CV_EXPORTS softdouble mulAdd( const softdouble& a, const softdouble& b, const softdouble& c);

CV_EXPORTS softfloat  sqrt( const softfloat&  a );
CV_EXPORTS softdouble sqrt( const softdouble& a );
}

/*----------------------------------------------------------------------------
| Ported from OpenCV and added for usability
*----------------------------------------------------------------------------*/

CV_EXPORTS int cvTrunc(const cv::softfloat&  a);
CV_EXPORTS int cvTrunc(const cv::softdouble& a);

CV_EXPORTS int cvRound(const cv::softfloat&  a);
CV_EXPORTS int cvRound(const cv::softdouble& a);

CV_EXPORTS int cvFloor(const cv::softfloat&  a);
CV_EXPORTS int cvFloor(const cv::softdouble& a);

CV_EXPORTS int  cvCeil(const cv::softfloat&  a);
CV_EXPORTS int  cvCeil(const cv::softdouble& a);

namespace cv
{
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

// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template<> inline unsigned saturate_cast<unsigned>(softfloat  a) { return cvRound(a); }
template<> inline unsigned saturate_cast<unsigned>(softdouble a) { return cvRound(a); }

inline softfloat  min(const softfloat&  a, const softfloat&  b) { return (a > b) ? b : a; }
inline softdouble min(const softdouble& a, const softdouble& b) { return (a > b) ? b : a; }

inline softfloat  max(const softfloat&  a, const softfloat&  b) { return (a > b) ? a : b; }
inline softdouble max(const softdouble& a, const softdouble& b) { return (a > b) ? a : b; }

inline softfloat  abs( softfloat  a) { softfloat  x; x.v = a.v & ((1U   << 31) - 1); return x; }
inline softdouble abs( softdouble a) { softdouble x; x.v = a.v & ((1ULL << 63) - 1); return x; }

CV_EXPORTS softfloat  exp( const softfloat&  a);
CV_EXPORTS softdouble exp( const softdouble& a);

CV_EXPORTS softfloat  log( const softfloat&  a );
CV_EXPORTS softdouble log( const softdouble& a );

CV_EXPORTS softfloat  pow( const softfloat&  a, const softfloat&  b);
CV_EXPORTS softdouble pow( const softdouble& a, const softdouble& b);

CV_EXPORTS softfloat cbrt(const softfloat& a);

}

#endif
