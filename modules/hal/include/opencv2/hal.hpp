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

#ifndef __OPENCV_HAL_HPP__
#define __OPENCV_HAL_HPP__

#include "opencv2/hal/defs.h"
#include "opencv2/hal/interface.hpp"

/**
  @defgroup hal Hardware Acceleration Layer
  @{
    @defgroup hal_intrin Universal intrinsics
    @{
      @defgroup hal_intrin_impl Private implementation helpers
    @}
    @defgroup hal_utils Platform-dependent utils
  @}
*/

namespace cv { namespace hal {

//! @addtogroup hal
//! @{

class Failure
{
public:
    Failure(int code_ = Error::Unknown) : code(code_) {}
public:
    int code;
};

int normHamming(const uchar* a, int n);
int normHamming(const uchar* a, const uchar* b, int n);

int normHamming(const uchar* a, int n, int cellSize);
int normHamming(const uchar* a, const uchar* b, int n, int cellSize);

//////////////////////////////// low-level functions ////////////////////////////////

int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);
bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);

int normL1_(const uchar* a, const uchar* b, int n);
float normL1_(const float* a, const float* b, int n);
float normL2Sqr_(const float* a, const float* b, int n);

void exp(const float* src, float* dst, int n);
void exp(const double* src, double* dst, int n);
void log(const float* src, float* dst, int n);
void log(const double* src, double* dst, int n);

void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees);
void magnitude(const float* x, const float* y, float* dst, int n);
void magnitude(const double* x, const double* y, double* dst, int n);
void sqrt(const float* src, float* dst, int len);
void sqrt(const double* src, double* dst, int len);
void invSqrt(const float* src, float* dst, int len);
void invSqrt(const double* src, double* dst, int len);

void split8u(const uchar* src, uchar** dst, int len, int cn );
void split16u(const ushort* src, ushort** dst, int len, int cn );
void split32s(const int* src, int** dst, int len, int cn );
void split64s(const int64* src, int64** dst, int len, int cn );

void merge8u(const uchar** src, uchar* dst, int len, int cn );
void merge16u(const ushort** src, ushort* dst, int len, int cn );
void merge32s(const int** src, int* dst, int len, int cn );
void merge64s(const int64** src, int64* dst, int len, int cn );

void add8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void add8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
void add16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
void add16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
void add32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
void add32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
void add64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );

void sub8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void sub8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
void sub16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
void sub16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
void sub32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
void sub32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
void sub64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );

void max8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void max8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
void max16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
void max16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
void max32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
void max32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
void max64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );

void min8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void min8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
void min16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
void min16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
void min32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
void min32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
void min64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );

void absdiff8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void absdiff8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* );
void absdiff16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* );
void absdiff16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* );
void absdiff32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* );
void absdiff32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* );
void absdiff64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* );

void and8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void or8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void xor8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );
void not8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* );

void cmp8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
void cmp8s(const schar* src1, size_t step1, const schar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
void cmp16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
void cmp16s(const short* src1, size_t step1, const short* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
void cmp32s(const int* src1, size_t step1, const int* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
void cmp32f(const float* src1, size_t step1, const float* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);
void cmp64f(const double* src1, size_t step1, const double* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);

void mul8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* scale);
void mul8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scale);
void mul16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scale);
void mul16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scale);
void mul32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scale);
void mul32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scale);
void mul64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scale);

void div8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* scale);
void div8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scale);
void div16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scale);
void div16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scale);
void div32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scale);
void div32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scale);
void div64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scale);

void recip8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* scale);
void recip8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scale);
void recip16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scale);
void recip16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scale);
void recip32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scale);
void recip32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scale);
void recip64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scale);

void addWeighted8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _scalars );
void addWeighted8s( const schar* src1, size_t step1, const schar* src2, size_t step2, schar* dst, size_t step, int width, int height, void* scalars );
void addWeighted16u( const ushort* src1, size_t step1, const ushort* src2, size_t step2, ushort* dst, size_t step, int width, int height, void* scalars );
void addWeighted16s( const short* src1, size_t step1, const short* src2, size_t step2, short* dst, size_t step, int width, int height, void* scalars );
void addWeighted32s( const int* src1, size_t step1, const int* src2, size_t step2, int* dst, size_t step, int width, int height, void* scalars );
void addWeighted32f( const float* src1, size_t step1, const float* src2, size_t step2, float* dst, size_t step, int width, int height, void* scalars );
void addWeighted64f( const double* src1, size_t step1, const double* src2, size_t step2, double* dst, size_t step, int width, int height, void* scalars );
//! @}

}} //cv::hal

namespace cv {

template<typename T1, typename T2=T1, typename T3=T1> struct OpAdd
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(const T1 a, const T2 b) const { return saturate_cast<T3>(a + b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(const T1 a, const T2 b) const { return saturate_cast<T3>(a - b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpRSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(const T1 a, const T2 b) const { return saturate_cast<T3>(b - a); }
};

template<typename T> struct OpMin
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(const T a, const T b) const { return std::min(a, b); }
};

template<typename T> struct OpMax
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(const T a, const T b) const { return std::max(a, b); }
};

template<typename T> struct OpAbsDiff
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()(T a, T b) const { return a > b ? a - b : b - a; }
};

template<typename T> struct OpAnd
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a & b; }
};

template<typename T> struct OpOr
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a | b; }
};

template<typename T> struct OpXor
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a ^ b; }
};

template<typename T> struct OpNot
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T ) const { return ~a; }
};

}

#endif //__OPENCV_HAL_HPP__
