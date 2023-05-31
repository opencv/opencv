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

#ifndef OPENCV_CUDA_TYPE_TRAITS_HPP
#define OPENCV_CUDA_TYPE_TRAITS_HPP

#include "detail/type_traits_detail.hpp"

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    template <typename T> struct IsSimpleParameter
    {
        enum {value = type_traits_detail::IsIntegral<T>::value || type_traits_detail::IsFloat<T>::value ||
            type_traits_detail::PointerTraits<typename type_traits_detail::ReferenceTraits<T>::type>::value};
    };

    template <typename T> struct TypeTraits
    {
        typedef typename type_traits_detail::UnConst<T>::type                                                NonConstType;
        typedef typename type_traits_detail::UnVolatile<T>::type                                             NonVolatileType;
        typedef typename type_traits_detail::UnVolatile<typename type_traits_detail::UnConst<T>::type>::type UnqualifiedType;
        typedef typename type_traits_detail::PointerTraits<UnqualifiedType>::type                            PointeeType;
        typedef typename type_traits_detail::ReferenceTraits<T>::type                                        ReferredType;

        enum { isConst          = type_traits_detail::UnConst<T>::value };
        enum { isVolatile       = type_traits_detail::UnVolatile<T>::value };

        enum { isReference      = type_traits_detail::ReferenceTraits<UnqualifiedType>::value };
        enum { isPointer        = type_traits_detail::PointerTraits<typename type_traits_detail::ReferenceTraits<UnqualifiedType>::type>::value };

        enum { isUnsignedInt    = type_traits_detail::IsUnsignedIntegral<UnqualifiedType>::value };
        enum { isSignedInt      = type_traits_detail::IsSignedIntergral<UnqualifiedType>::value };
        enum { isIntegral       = type_traits_detail::IsIntegral<UnqualifiedType>::value };
        enum { isFloat          = type_traits_detail::IsFloat<UnqualifiedType>::value };
        enum { isArith          = isIntegral || isFloat };
        enum { isVec            = type_traits_detail::IsVec<UnqualifiedType>::value };

        typedef typename type_traits_detail::Select<IsSimpleParameter<UnqualifiedType>::value,
            T, typename type_traits_detail::AddParameterType<T>::type>::type ParameterType;
    };
}}}

//! @endcond

#endif // OPENCV_CUDA_TYPE_TRAITS_HPP
