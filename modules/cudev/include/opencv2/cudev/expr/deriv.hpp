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

#ifndef __OPENCV_CUDEV_EXPR_DERIV_HPP__
#define __OPENCV_CUDEV_EXPR_DERIV_HPP__

#include "../common.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/deriv.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// derivX

template <class SrcPtr>
__host__ Expr<DerivXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
derivX_(const SrcPtr& src)
{
    return makeExpr(derivXPtr(src));
}

// derivY

template <class SrcPtr>
__host__ Expr<DerivYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
derivY_(const SrcPtr& src)
{
    return makeExpr(derivYPtr(src));
}

// sobelX

template <class SrcPtr>
__host__ Expr<SobelXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
sobelX_(const SrcPtr& src)
{
    return makeExpr(sobelXPtr(src));
}

// sobelY

template <class SrcPtr>
__host__ Expr<SobelYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
sobelY_(const SrcPtr& src)
{
    return makeExpr(sobelYPtr(src));
}

// scharrX

template <class SrcPtr>
__host__ Expr<ScharrXPtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
scharrX_(const SrcPtr& src)
{
    return makeExpr(scharrXPtr(src));
}

// scharrY

template <class SrcPtr>
__host__ Expr<ScharrYPtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
scharrY_(const SrcPtr& src)
{
    return makeExpr(scharrYPtr(src));
}

// laplacian

template <int ksize, class SrcPtr>
__host__ Expr<LaplacianPtrSz<ksize, typename PtrTraits<SrcPtr>::ptr_type> >
laplacian_(const SrcPtr& src)
{
    return makeExpr(laplacianPtr<ksize>(src));
}

//! @}

}}

#endif
