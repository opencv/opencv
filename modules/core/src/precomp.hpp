/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#if defined _MSC_VER && _MSC_VER >= 1200
    // disable warnings related to inline functions
    #pragma warning( disable: 4251 4711 4710 4514 )
#endif

#ifdef HAVE_CONFIG_H 
#include <cvconfig.h> 
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cv
{

// -128.f ... 255.f
extern const float g_8x32fTab[];
#define CV_8TO32F(x)  cv::g_8x32fTab[(x)+128]

extern const ushort g_8x16uSqrTab[];
#define CV_SQR_8U(x)  cv::g_8x16uSqrTab[(x)+255]

extern const char* g_HersheyGlyphs[];

extern const uchar g_Saturate8u[];
#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) && (t) <= 512), cv::g_Saturate8u[(t)+256])
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))


#if defined WIN32 || defined _WIN32
void deleteThreadAllocData();
void deleteThreadRNGData();
#endif

template<typename T1, typename T2=T1, typename T3=T1> struct OpAdd
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(a + b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(a - b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpRSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(b - a); }
};

template<typename T> struct OpMin
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(T a, T b) const { return std::min(a, b); }
};

template<typename T> struct OpMax
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(T a, T b) const { return std::max(a, b); }
};

inline Size getContinuousSize( const Mat& m1, int widthScale=1 )
{
    return m1.isContinuous() ? Size(m1.cols*m1.rows*widthScale, 1) :
        Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2, int widthScale=1 )
{
    return (m1.flags & m2.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & m4.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               const Mat& m5, int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & m4.flags & m5.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

struct NoVec
{
    size_t operator()(const void*, const void*, void*, size_t) const { return 0; }
};

extern volatile bool USE_SSE2;

typedef void (*BinaryFunc)(const uchar* src1, size_t step1,
                           const uchar* src2, size_t step2,
                           uchar* dst, size_t step, Size sz,
                           void*);

BinaryFunc getConvertFunc(int sdepth, int ddepth);
BinaryFunc getConvertScaleFunc(int sdepth, int ddepth);
BinaryFunc getCopyMaskFunc(size_t esz);

enum { BLOCK_SIZE = 1024 };

#ifdef HAVE_IPP
static inline IppiSize ippiSize(int width, int height) { IppiSize sz = { width, height}; return sz; }
static inline IppiSize ippiSize(Size _sz)              { IppiSize sz = { _sz.width, _sz.height}; return sz; }
#endif

#if defined HAVE_IPP && (IPP_VERSION_MAJOR >= 7)
#define ARITHM_USE_IPP 1
#define IF_IPP(then_call, else_call) then_call
#else
#define ARITHM_USE_IPP 0
#define IF_IPP(then_call, else_call) else_call
#endif

}

#endif /*_CXCORE_INTERNAL_H_*/
