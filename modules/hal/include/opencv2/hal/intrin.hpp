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

#ifndef __OPENCV_HAL_INTRIN_HPP__
#define __OPENCV_HAL_INTRIN_HPP__

#include <cmath>
#include <float.h>
#include <stdlib.h>

#define OPENCV_HAL_ADD(a, b) ((a) + (b))
#define OPENCV_HAL_AND(a, b) ((a) & (b))
#define OPENCV_HAL_NOP(a) (a)
#define OPENCV_HAL_1ST(a, b) (a)

// unlike HAL API, which is in cv::hall,
// we put intrinsics into cv namespace to make its
// access from within opencv code more accessible
namespace cv {

template<typename _Tp> struct V_TypeTraits
{
    typedef _Tp int_type;
    typedef _Tp uint_type;
    typedef _Tp abs_type;
    typedef _Tp sum_type;

    enum { delta = 0, shift = 0 };

    static int_type reinterpret_int(_Tp x) { return x; }
    static uint_type reinterpet_uint(_Tp x) { return x; }
    static _Tp reinterpret_from_int(int_type x) { return (_Tp)x; }
};

template<> struct V_TypeTraits<uchar>
{
    typedef uchar value_type;
    typedef schar int_type;
    typedef uchar uint_type;
    typedef uchar abs_type;
    typedef int sum_type;

    typedef ushort w_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<schar>
{
    typedef schar value_type;
    typedef schar int_type;
    typedef uchar uint_type;
    typedef uchar abs_type;
    typedef int sum_type;

    typedef short w_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<ushort>
{
    typedef ushort value_type;
    typedef short int_type;
    typedef ushort uint_type;
    typedef ushort abs_type;
    typedef int sum_type;

    typedef unsigned w_type;
    typedef uchar nu_type;

    enum { delta = 32768, shift = 16 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<short>
{
    typedef short value_type;
    typedef short int_type;
    typedef ushort uint_type;
    typedef ushort abs_type;
    typedef int sum_type;

    typedef int w_type;
    typedef uchar nu_type;
    typedef schar n_type;

    enum { delta = 128, shift = 8 };

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<unsigned>
{
    typedef unsigned value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef unsigned abs_type;
    typedef unsigned sum_type;

    typedef uint64 w_type;
    typedef ushort nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<int>
{
    typedef int value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef unsigned abs_type;
    typedef int sum_type;

    typedef int64 w_type;
    typedef short n_type;
    typedef ushort nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<uint64>
{
    typedef uint64 value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef uint64 abs_type;
    typedef uint64 sum_type;

    typedef unsigned nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};

template<> struct V_TypeTraits<int64>
{
    typedef int64 value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef uint64 abs_type;
    typedef int64 sum_type;

    typedef int nu_type;

    static int_type reinterpret_int(value_type x) { return (int_type)x; }
    static uint_type reinterpret_uint(value_type x) { return (uint_type)x; }
    static value_type reinterpret_from_int(int_type x) { return (value_type)x; }
};


template<> struct V_TypeTraits<float>
{
    typedef float value_type;
    typedef int int_type;
    typedef unsigned uint_type;
    typedef float abs_type;
    typedef float sum_type;

    typedef double w_type;

    static int_type reinterpret_int(value_type x)
    {
        Cv32suf u;
        u.f = x;
        return u.i;
    }
    static uint_type reinterpet_uint(value_type x)
    {
        Cv32suf u;
        u.f = x;
        return u.u;
    }
    static value_type reinterpret_from_int(int_type x)
    {
        Cv32suf u;
        u.i = x;
        return u.f;
    }
};

template<> struct V_TypeTraits<double>
{
    typedef double value_type;
    typedef int64 int_type;
    typedef uint64 uint_type;
    typedef double abs_type;
    typedef double sum_type;
    static int_type reinterpret_int(value_type x)
    {
        Cv64suf u;
        u.f = x;
        return u.i;
    }
    static uint_type reinterpet_uint(value_type x)
    {
        Cv64suf u;
        u.f = x;
        return u.u;
    }
    static value_type reinterpret_from_int(int_type x)
    {
        Cv64suf u;
        u.i = x;
        return u.f;
    }
};

}

#if CV_SSE2

#include "opencv2/hal/intrin_sse.hpp"

#elif CV_NEON

#include "opencv2/hal/intrin_neon.hpp"

#else

#include "opencv2/hal/intrin_cpp.hpp"

#endif

#ifndef CV_SIMD128
#define CV_SIMD128 0
#endif

#ifndef CV_SIMD128_64F
#define CV_SIMD128_64F 0
#endif

#endif
