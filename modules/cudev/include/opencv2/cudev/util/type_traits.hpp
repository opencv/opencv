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

#ifndef __OPENCV_CUDEV_UTIL_TYPE_TRAITS_HPP__
#define __OPENCV_CUDEV_UTIL_TYPE_TRAITS_HPP__

#include "../common.hpp"
#include "vec_traits.hpp"
#include "detail/type_traits.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// NullType

struct NullType {};

// Int2Type

template <int A> struct Int2Type
{
   enum { value = A };
};

// ArrayWrapper

template <typename T, int COUNT> struct ArrayWrapper
{
    T array[COUNT];
};

// Log2 (compile time calculation)

template <int N, int CURRENT_VAL = N, int COUNT = 0> struct Log2
{
    enum { value = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};
template <int N, int COUNT> struct Log2<N, 0, COUNT>
{
    enum { value = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

// IsPowerOf2

template <int N> struct IsPowerOf2
{
    enum { value = ((N != 0) && !(N & (N - 1))) };
};

// SelectIf

template <bool, typename ThenType, typename ElseType> struct SelectIf
{
    typedef ThenType type;
};
template <typename ThenType, typename ElseType> struct SelectIf<false, ThenType, ElseType>
{
    typedef ElseType type;
};

// EnableIf

template <bool, typename T = void> struct EnableIf {};
template <typename T> struct EnableIf<true, T> { typedef T type; };

// DisableIf

template <bool, typename T = void> struct DisableIf {};
template <typename T> struct DisableIf<false, T> { typedef T type; };

// TypesEquals

template <typename A, typename B> struct TypesEquals
{
    enum { value = 0 };
};
template <typename A> struct TypesEquals<A, A>
{
    enum { value = 1 };
};

// TypeTraits

template <typename T> struct TypeTraits
{
    typedef typename type_traits_detail::UnConst<T>::type                                                non_const_type;
    typedef typename type_traits_detail::UnVolatile<T>::type                                             non_volatile_type;
    typedef typename type_traits_detail::UnVolatile<typename type_traits_detail::UnConst<T>::type>::type unqualified_type;
    typedef typename type_traits_detail::PointerTraits<unqualified_type>::type                           pointee_type;
    typedef typename type_traits_detail::ReferenceTraits<T>::type                                        referred_type;

    enum { is_const          = type_traits_detail::UnConst<T>::value };
    enum { is_volatile       = type_traits_detail::UnVolatile<T>::value };

    enum { is_reference      = type_traits_detail::ReferenceTraits<unqualified_type>::value };
    enum { is_pointer        = type_traits_detail::PointerTraits<typename type_traits_detail::ReferenceTraits<unqualified_type>::type>::value };

    enum { is_unsigned_int   = type_traits_detail::IsUnsignedIntegral<unqualified_type>::value };
    enum { is_signed_int     = type_traits_detail::IsSignedIntergral<unqualified_type>::value };
    enum { is_integral       = type_traits_detail::IsIntegral<unqualified_type>::value };
    enum { is_float          = type_traits_detail::IsFloat<unqualified_type>::value };
    enum { is_scalar         = is_integral || is_float };
    enum { is_vec            = type_traits_detail::IsVec<unqualified_type>::value };

    typedef typename SelectIf<type_traits_detail::IsSimpleParameter<unqualified_type>::value,
        T, typename type_traits_detail::AddParameterType<T>::type>::type parameter_type;
};

// LargerType

template <typename A, typename B> struct LargerType
{
    typedef typename SelectIf<
        unsigned(VecTraits<A>::cn) != unsigned(VecTraits<B>::cn),
        void,
        typename MakeVec<
            typename type_traits_detail::LargerDepth<
                typename VecTraits<A>::elem_type,
                typename VecTraits<B>::elem_type
            >::type,
            VecTraits<A>::cn
        >::type
    >::type type;
};

//! @}

}}

#endif
