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

#ifndef OPENCV_CUDEV_UTIL_LIMITS_HPP
#define OPENCV_CUDEV_UTIL_LIMITS_HPP

#include <limits.h>
#include <float.h>
#include "../common.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

template <class T> struct numeric_limits;

template <> struct numeric_limits<bool>
{
    __device__ __forceinline__ static bool min() { return false; }
    __device__ __forceinline__ static bool max() { return true;  }
    static const bool is_signed = false;
};

template <> struct numeric_limits<schar>
{
    __device__ __forceinline__ static schar min() { return SCHAR_MIN; }
    __device__ __forceinline__ static schar max() { return SCHAR_MAX; }
    static const bool is_signed = true;
};

template <> struct numeric_limits<uchar>
{
    __device__ __forceinline__ static uchar min() { return 0; }
    __device__ __forceinline__ static uchar max() { return UCHAR_MAX; }
    static const bool is_signed = false;
};

template <> struct numeric_limits<short>
{
    __device__ __forceinline__ static short min() { return SHRT_MIN; }
    __device__ __forceinline__ static short max() { return SHRT_MAX; }
    static const bool is_signed = true;
};

template <> struct numeric_limits<ushort>
{
    __device__ __forceinline__ static ushort min() { return 0; }
    __device__ __forceinline__ static ushort max() { return USHRT_MAX; }
    static const bool is_signed = false;
};

template <> struct numeric_limits<int>
{
    __device__ __forceinline__ static int min() { return INT_MIN; }
    __device__ __forceinline__ static int max() { return INT_MAX; }
    static const bool is_signed = true;
};

template <> struct numeric_limits<uint>
{
    __device__ __forceinline__ static uint min() { return 0; }
    __device__ __forceinline__ static uint max() { return UINT_MAX; }
    static const bool is_signed = false;
};

template <> struct numeric_limits<float>
{
    __device__ __forceinline__ static float min() { return FLT_MIN; }
    __device__ __forceinline__ static float max() { return FLT_MAX; }
    __device__ __forceinline__ static float epsilon() { return FLT_EPSILON; }
    static const bool is_signed = true;
};

template <> struct numeric_limits<double>
{
    __device__ __forceinline__ static double min() { return DBL_MIN; }
    __device__ __forceinline__ static double max() { return DBL_MAX; }
    __device__ __forceinline__ static double epsilon() { return DBL_EPSILON; }
    static const bool is_signed = true;
};

//! @}

}}

#endif
