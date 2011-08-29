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

#ifndef __OPENCV_GPU_TYPE_TRAITS_HPP__
#define __OPENCV_GPU_TYPE_TRAITS_HPP__

#include "detail/type_traits_detail.hpp"

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct IsSimpleParameter
    {
        enum {value = detail::IsIntegral<T>::value || detail::IsFloat<T>::value || detail::PointerTraits<typename detail::ReferenceTraits<T>::type>::value};
    };

    template <typename T> struct TypeTraits
    {
        typedef typename detail::UnConst<T>::type                                       NonConstType;
        typedef typename detail::UnVolatile<T>::type                                    NonVolatileType;
        typedef typename detail::UnVolatile<typename detail::UnConst<T>::type>::type    UnqualifiedType;
        typedef typename detail::PointerTraits<UnqualifiedType>::type                   PointeeType;
        typedef typename detail::ReferenceTraits<T>::type                               ReferredType;

        enum { isConst          = detail::UnConst<T>::value };
        enum { isVolatile       = detail::UnVolatile<T>::value };

        enum { isReference      = detail::ReferenceTraits<UnqualifiedType>::value };
        enum { isPointer        = detail::PointerTraits<typename detail::ReferenceTraits<UnqualifiedType>::type>::value };        

        enum { isUnsignedInt = detail::IsUnsignedIntegral<UnqualifiedType>::value };
        enum { isSignedInt   = detail::IsSignedIntergral<UnqualifiedType>::value };
        enum { isIntegral    = detail::IsIntegral<UnqualifiedType>::value };
        enum { isFloat       = detail::IsFloat<UnqualifiedType>::value  };
        enum { isArith       = isIntegral || isFloat };
        enum { isVec         = detail::IsVec<UnqualifiedType>::value  };
        
        typedef typename detail::Select<IsSimpleParameter<UnqualifiedType>::value, T, typename detail::AddParameterType<T>::type>::type ParameterType;
    };
}}}

#endif // __OPENCV_GPU_TYPE_TRAITS_HPP__
