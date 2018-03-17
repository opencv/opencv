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

#ifndef OPENCV_CUDEV_EXPR_BINARY_FUNC_HPP
#define OPENCV_CUDEV_EXPR_BINARY_FUNC_HPP

#include "../common.hpp"
#include "../util/type_traits.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/transform.hpp"
#include "../functional/functional.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

#define CV_CUDEV_EXPR_BINARY_FUNC(name) \
    template <class SrcPtr1, class SrcPtr2> \
    __host__ Expr<BinaryTransformPtrSz<typename PtrTraits<SrcPtr1>::ptr_type, typename PtrTraits<SrcPtr2>::ptr_type, name ## _func<typename LargerType<typename PtrTraits<SrcPtr1>::value_type, typename PtrTraits<SrcPtr2>::value_type>::type> > > \
    name ## _(const SrcPtr1& src1, const SrcPtr2& src2) \
    { \
        return makeExpr(transformPtr(src1, src2, name ## _func<typename LargerType<typename PtrTraits<SrcPtr1>::value_type, typename PtrTraits<SrcPtr2>::value_type>::type>())); \
    }

CV_CUDEV_EXPR_BINARY_FUNC(hypot)
CV_CUDEV_EXPR_BINARY_FUNC(magnitude)
CV_CUDEV_EXPR_BINARY_FUNC(atan2)
CV_CUDEV_EXPR_BINARY_FUNC(absdiff)

#undef CV_CUDEV_EXPR_BINARY_FUNC

//! @}

}}

#endif
