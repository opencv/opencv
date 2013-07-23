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

#ifndef __OPENCV_CUDEV_EXPR_UNARY_FUNC_HPP__
#define __OPENCV_CUDEV_EXPR_UNARY_FUNC_HPP__

#include "../common.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/transform.hpp"
#include "../functional/functional.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

#define CV_CUDEV_EXPR_UNARY_FUNC(name) \
    template <class SrcPtr> \
    __host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, name ## _func<typename PtrTraits<SrcPtr>::value_type> > > \
    name ## _(const SrcPtr& src) \
    { \
        return makeExpr(transformPtr(src, name ## _func<typename PtrTraits<SrcPtr>::value_type>())); \
    }

CV_CUDEV_EXPR_UNARY_FUNC(abs)
CV_CUDEV_EXPR_UNARY_FUNC(sqr)
CV_CUDEV_EXPR_UNARY_FUNC(sqrt)
CV_CUDEV_EXPR_UNARY_FUNC(exp)
CV_CUDEV_EXPR_UNARY_FUNC(exp2)
CV_CUDEV_EXPR_UNARY_FUNC(exp10)
CV_CUDEV_EXPR_UNARY_FUNC(log)
CV_CUDEV_EXPR_UNARY_FUNC(log2)
CV_CUDEV_EXPR_UNARY_FUNC(log10)
CV_CUDEV_EXPR_UNARY_FUNC(sin)
CV_CUDEV_EXPR_UNARY_FUNC(cos)
CV_CUDEV_EXPR_UNARY_FUNC(tan)
CV_CUDEV_EXPR_UNARY_FUNC(asin)
CV_CUDEV_EXPR_UNARY_FUNC(acos)
CV_CUDEV_EXPR_UNARY_FUNC(atan)
CV_CUDEV_EXPR_UNARY_FUNC(sinh)
CV_CUDEV_EXPR_UNARY_FUNC(cosh)
CV_CUDEV_EXPR_UNARY_FUNC(tanh)
CV_CUDEV_EXPR_UNARY_FUNC(asinh)
CV_CUDEV_EXPR_UNARY_FUNC(acosh)
CV_CUDEV_EXPR_UNARY_FUNC(atanh)

#undef CV_CUDEV_EXPR_UNARY_FUNC

template <class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, Binder2nd<pow_func<typename PtrTraits<SrcPtr>::value_type> > > >
pow_(const SrcPtr& src, float power)
{
    return makeExpr(transformPtr(src, bind2nd(pow_func<typename PtrTraits<SrcPtr>::value_type>(), power)));
}

}}

#endif
