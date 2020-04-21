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

#ifndef OPENCV_CUDEV_UTIL_TYPE_TRAITS_DETAIL_HPP
#define OPENCV_CUDEV_UTIL_TYPE_TRAITS_DETAIL_HPP

#include "../../common.hpp"

namespace cv { namespace cudev {

namespace type_traits_detail
{
    template <typename T> struct IsSignedIntergral { enum {value = 0}; };
    template <> struct IsSignedIntergral<schar> { enum {value = 1}; };
    template <> struct IsSignedIntergral<short> { enum {value = 1}; };
    template <> struct IsSignedIntergral<int> { enum {value = 1}; };

    template <typename T> struct IsUnsignedIntegral { enum {value = 0}; };
    template <> struct IsUnsignedIntegral<uchar> { enum {value = 1}; };
    template <> struct IsUnsignedIntegral<ushort> { enum {value = 1}; };
    template <> struct IsUnsignedIntegral<uint> { enum {value = 1}; };

    template <typename T> struct IsIntegral { enum {value = IsSignedIntergral<T>::value || IsUnsignedIntegral<T>::value}; };
    template <> struct IsIntegral<char> { enum {value = 1}; };
    template <> struct IsIntegral<bool> { enum {value = 1}; };

    template <typename T> struct IsFloat { enum {value = 0}; };
    template <> struct IsFloat<float> { enum {value = 1}; };
    template <> struct IsFloat<double> { enum {value = 1}; };

    template <typename T> struct IsVec { enum {value = 0}; };
    template <> struct IsVec<uchar1> { enum {value = 1}; };
    template <> struct IsVec<uchar2> { enum {value = 1}; };
    template <> struct IsVec<uchar3> { enum {value = 1}; };
    template <> struct IsVec<uchar4> { enum {value = 1}; };
    template <> struct IsVec<char1> { enum {value = 1}; };
    template <> struct IsVec<char2> { enum {value = 1}; };
    template <> struct IsVec<char3> { enum {value = 1}; };
    template <> struct IsVec<char4> { enum {value = 1}; };
    template <> struct IsVec<ushort1> { enum {value = 1}; };
    template <> struct IsVec<ushort2> { enum {value = 1}; };
    template <> struct IsVec<ushort3> { enum {value = 1}; };
    template <> struct IsVec<ushort4> { enum {value = 1}; };
    template <> struct IsVec<short1> { enum {value = 1}; };
    template <> struct IsVec<short2> { enum {value = 1}; };
    template <> struct IsVec<short3> { enum {value = 1}; };
    template <> struct IsVec<short4> { enum {value = 1}; };
    template <> struct IsVec<uint1> { enum {value = 1}; };
    template <> struct IsVec<uint2> { enum {value = 1}; };
    template <> struct IsVec<uint3> { enum {value = 1}; };
    template <> struct IsVec<uint4> { enum {value = 1}; };
    template <> struct IsVec<int1> { enum {value = 1}; };
    template <> struct IsVec<int2> { enum {value = 1}; };
    template <> struct IsVec<int3> { enum {value = 1}; };
    template <> struct IsVec<int4> { enum {value = 1}; };
    template <> struct IsVec<float1> { enum {value = 1}; };
    template <> struct IsVec<float2> { enum {value = 1}; };
    template <> struct IsVec<float3> { enum {value = 1}; };
    template <> struct IsVec<float4> { enum {value = 1}; };
    template <> struct IsVec<double1> { enum {value = 1}; };
    template <> struct IsVec<double2> { enum {value = 1}; };
    template <> struct IsVec<double3> { enum {value = 1}; };
    template <> struct IsVec<double4> { enum {value = 1}; };

    template <class U> struct AddParameterType { typedef const U& type; };
    template <class U> struct AddParameterType<U&> { typedef U& type; };
    template <> struct AddParameterType<void> { typedef void type; };

    // ReferenceTraits

    template <class U> struct ReferenceTraits
    {
        enum { value = 0 };
        typedef U type;
    };
    template <class U> struct ReferenceTraits<U&>
    {
        enum { value = 1 };
        typedef U type;
    };

    // PointerTraits

    template <class U> struct PointerTraits
    {
        enum { value = 0 };
        typedef void type;
    };
    template <class U> struct PointerTraits<U*>
    {
        enum { value = 1 };
        typedef U type;
    };
    template <class U> struct PointerTraits<U*&>
    {
        enum { value = 1 };
        typedef U type;
    };

    // UnConst

    template <class U> struct UnConst
    {
        typedef U type;
        enum { value = 0 };
    };
    template <class U> struct UnConst<const U>
    {
        typedef U type;
        enum { value = 1 };
    };
    template <class U> struct UnConst<const U&>
    {
        typedef U& type;
        enum { value = 1 };
    };

    // UnVolatile

    template <class U> struct UnVolatile
    {
        typedef U type;
        enum { value = 0 };
    };
    template <class U> struct UnVolatile<volatile U>
    {
        typedef U type;
        enum { value = 1 };
    };
    template <class U> struct UnVolatile<volatile U&>
    {
        typedef U& type;
        enum { value = 1 };
    };

    // IsSimpleParameter

    template <typename T> struct IsSimpleParameter
    {
        enum { value = IsIntegral<T>::value
               || IsFloat<T>::value
               || PointerTraits<typename ReferenceTraits<T>::type>::value};
    };

    // LargerDepth

    template <bool, typename ThenType, typename ElseType> struct SelectIf
    {
        typedef ThenType type;
    };
    template <typename ThenType, typename ElseType> struct SelectIf<false, ThenType, ElseType>
    {
        typedef ElseType type;
    };

    template <typename A, typename B> struct LargerDepth
    {
        typedef typename SelectIf<sizeof(A) >= sizeof(B), A, B>::type type;
    };
    template <typename A> struct LargerDepth<A, float>
    {
        typedef float type;
    };
    template <typename A> struct LargerDepth<float, A>
    {
        typedef float type;
    };
    template <typename A> struct LargerDepth<A, double>
    {
        typedef double type;
    };
    template <typename A> struct LargerDepth<double, A>
    {
        typedef double type;
    };
    template <> struct LargerDepth<float, float>
    {
        typedef float type;
    };
    template <> struct LargerDepth<float, double>
    {
        typedef double type;
    };
    template <> struct LargerDepth<double, float>
    {
        typedef double type;
    };
    template <> struct LargerDepth<double, double>
    {
        typedef double type;
    };
}

}}

#endif
