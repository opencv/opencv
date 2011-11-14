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

#ifndef __OPENCV_GPU_LIMITS_GPU_HPP__
#define __OPENCV_GPU_LIMITS_GPU_HPP__

#include <limits>
#include "common.hpp"

namespace cv { namespace gpu { namespace device 
{
    template<class T> struct numeric_limits
    {
        typedef T type;
        __device__ __forceinline__ static type min()  { return type(); };
        __device__ __forceinline__ static type max() { return type(); };
        __device__ __forceinline__ static type epsilon() { return type(); }
        __device__ __forceinline__ static type round_error() { return type(); }
        __device__ __forceinline__ static type denorm_min()  { return type(); }
        __device__ __forceinline__ static type infinity() { return type(); }
        __device__ __forceinline__ static type quiet_NaN() { return type(); }
        __device__ __forceinline__ static type signaling_NaN() { return T(); }
        static const bool is_signed;
    };

    template<> struct numeric_limits<bool>
    {
        typedef bool type;
        __device__ __forceinline__ static type min() { return false; };
        __device__ __forceinline__ static type max() { return true;  };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = false;
    };

    template<> struct numeric_limits<char>
    {
        typedef char type;
        __device__ __forceinline__ static type min() { return CHAR_MIN; };
        __device__ __forceinline__ static type max() { return CHAR_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = (char)-1 == -1;
    };

    template<> struct numeric_limits<signed char>
    {
        typedef char type;
        __device__ __forceinline__ static type min() { return SCHAR_MIN; };
        __device__ __forceinline__ static type max() { return SCHAR_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = (signed char)-1 == -1;
    };

    template<> struct numeric_limits<unsigned char>
    {
        typedef unsigned char type;
        __device__ __forceinline__ static type min() { return 0; };
        __device__ __forceinline__ static type max() { return UCHAR_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = false;
    };

    template<> struct numeric_limits<short>
    {
        typedef short type;
        __device__ __forceinline__ static type min() { return SHRT_MIN; };
        __device__ __forceinline__ static type max() { return SHRT_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = true;
    };

    template<> struct numeric_limits<unsigned short>
    {
        typedef unsigned short type;
        __device__ __forceinline__ static type min() { return 0; };
        __device__ __forceinline__ static type max() { return USHRT_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = false;
    };

    template<> struct numeric_limits<int>
    {
        typedef int type;
        __device__ __forceinline__ static type min() { return INT_MIN; };
        __device__ __forceinline__ static type max() { return INT_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = true;
    };


    template<> struct numeric_limits<unsigned int>
    {
        typedef unsigned int type;
        __device__ __forceinline__ static type min() { return 0; };
        __device__ __forceinline__ static type max() { return UINT_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = false;
    };

    template<> struct numeric_limits<long>
    {
        typedef long type;
        __device__ __forceinline__ static type min() { return LONG_MIN; };
        __device__ __forceinline__ static type max() { return LONG_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = true;
    };

    template<> struct numeric_limits<unsigned long>
    {
        typedef unsigned long type;
        __device__ __forceinline__ static type min() { return 0; };
        __device__ __forceinline__ static type max() { return ULONG_MAX; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = false;
    };

    template<> struct numeric_limits<float>
    {
        typedef float type;
        __device__ __forceinline__ static type min() { return 1.175494351e-38f/*FLT_MIN*/; };
        __device__ __forceinline__ static type max() { return 3.402823466e+38f/*FLT_MAX*/; };
        __device__ __forceinline__ static type epsilon() { return 1.192092896e-07f/*FLT_EPSILON*/; };
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = true;
    };

    template<> struct numeric_limits<double>
    {
        typedef double type;
        __device__ __forceinline__ static type min() { return 2.2250738585072014e-308/*DBL_MIN*/; };
        __device__ __forceinline__ static type max() { return 1.7976931348623158e+308/*DBL_MAX*/; };
        __device__ __forceinline__ static type epsilon();
        __device__ __forceinline__ static type round_error();
        __device__ __forceinline__ static type denorm_min();
        __device__ __forceinline__ static type infinity();
        __device__ __forceinline__ static type quiet_NaN();
        __device__ __forceinline__ static type signaling_NaN();
        static const bool is_signed = true;
    };
}}} // namespace cv { namespace gpu { namespace device {

#endif // __OPENCV_GPU_LIMITS_GPU_HPP__
