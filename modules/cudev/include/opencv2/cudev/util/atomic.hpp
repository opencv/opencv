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

#pragma once

#ifndef OPENCV_CUDEV_UTIL_ATOMIC_HPP
#define OPENCV_CUDEV_UTIL_ATOMIC_HPP

#include "../common.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// atomicAdd

__device__ __forceinline__ int atomicAdd(int* address, int val)
{
    return ::atomicAdd(address, val);
}

__device__ __forceinline__ uint atomicAdd(uint* address, uint val)
{
    return ::atomicAdd(address, val);
}

__device__ __forceinline__ float atomicAdd(float* address, float val)
{
#if CV_CUDEV_ARCH >= 200
    return ::atomicAdd(address, val);
#else
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
#endif
}

__device__ static double atomicAdd(double* address, double val)
{
#if CV_CUDEV_ARCH >= 130
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    (void) address;
    (void) val;
    return 0.0;
#endif
}

// atomicMin

__device__ __forceinline__ int atomicMin(int* address, int val)
{
    return ::atomicMin(address, val);
}

__device__ __forceinline__ uint atomicMin(uint* address, uint val)
{
    return ::atomicMin(address, val);
}

__device__ static float atomicMin(float* address, float val)
{
#if CV_CUDEV_ARCH >= 120
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
#else
    (void) address;
    (void) val;
    return 0.0f;
#endif
}

__device__ static double atomicMin(double* address, double val)
{
#if CV_CUDEV_ARCH >= 130
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
            __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    (void) address;
    (void) val;
    return 0.0;
#endif
}

// atomicMax

__device__ __forceinline__ int atomicMax(int* address, int val)
{
    return ::atomicMax(address, val);
}

__device__ __forceinline__ uint atomicMax(uint* address, uint val)
{
    return ::atomicMax(address, val);
}

__device__ static float atomicMax(float* address, float val)
{
#if CV_CUDEV_ARCH >= 120
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
#else
    (void) address;
    (void) val;
    return 0.0f;
#endif
}

__device__ static double atomicMax(double* address, double val)
{
#if CV_CUDEV_ARCH >= 130
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
            __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    (void) address;
    (void) val;
    return 0.0;
#endif
}

//! @}

}}

#endif
