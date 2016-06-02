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

#ifndef __OPENCV_CUDEV_EXPR_PER_ELEMENT_FUNC_HPP__
#define __OPENCV_CUDEV_EXPR_PER_ELEMENT_FUNC_HPP__

#include "../common.hpp"
#include "../util/type_traits.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/transform.hpp"
#include "../ptr2d/lut.hpp"
#include "../functional/functional.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// min/max

template <class SrcPtr1, class SrcPtr2>
__host__ Expr<BinaryTransformPtrSz<typename PtrTraits<SrcPtr1>::ptr_type, typename PtrTraits<SrcPtr2>::ptr_type, minimum<typename LargerType<typename PtrTraits<SrcPtr1>::value_type, typename PtrTraits<SrcPtr2>::value_type>::type> > >
min_(const SrcPtr1& src1, const SrcPtr2& src2)
{
    return makeExpr(transformPtr(src1, src2, minimum<typename LargerType<typename PtrTraits<SrcPtr1>::value_type, typename PtrTraits<SrcPtr2>::value_type>::type>()));
}

template <class SrcPtr1, class SrcPtr2>
__host__ Expr<BinaryTransformPtrSz<typename PtrTraits<SrcPtr1>::ptr_type, typename PtrTraits<SrcPtr2>::ptr_type, maximum<typename LargerType<typename PtrTraits<SrcPtr1>::value_type, typename PtrTraits<SrcPtr2>::value_type>::type> > >
max_(const SrcPtr1& src1, const SrcPtr2& src2)
{
    return makeExpr(transformPtr(src1, src2, maximum<typename LargerType<typename PtrTraits<SrcPtr1>::value_type, typename PtrTraits<SrcPtr2>::value_type>::type>()));
}

// threshold

template <class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, ThreshBinaryFunc<typename PtrTraits<SrcPtr>::value_type> > >
threshBinary_(const SrcPtr& src, typename PtrTraits<SrcPtr>::value_type thresh, typename PtrTraits<SrcPtr>::value_type maxVal)
{
    return makeExpr(transformPtr(src, thresh_binary_func(thresh, maxVal)));
}

template <class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, ThreshBinaryInvFunc<typename PtrTraits<SrcPtr>::value_type> > >
threshBinaryInv_(const SrcPtr& src, typename PtrTraits<SrcPtr>::value_type thresh, typename PtrTraits<SrcPtr>::value_type maxVal)
{
    return makeExpr(transformPtr(src, thresh_binary_inv_func(thresh, maxVal)));
}

template <class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, ThreshTruncFunc<typename PtrTraits<SrcPtr>::value_type> > >
threshTrunc_(const SrcPtr& src, typename PtrTraits<SrcPtr>::value_type thresh)
{
    return makeExpr(transformPtr(src, thresh_trunc_func(thresh)));
}

template <class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, ThreshToZeroFunc<typename PtrTraits<SrcPtr>::value_type> > >
threshToZero_(const SrcPtr& src, typename PtrTraits<SrcPtr>::value_type thresh)
{
    return makeExpr(transformPtr(src, thresh_to_zero_func(thresh)));
}

template <class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, ThreshToZeroInvFunc<typename PtrTraits<SrcPtr>::value_type> > >
threshToZeroInv_(const SrcPtr& src, typename PtrTraits<SrcPtr>::value_type thresh)
{
    return makeExpr(transformPtr(src, thresh_to_zero_inv_func(thresh)));
}

// cvt

template <typename D, class SrcPtr>
__host__ Expr<UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, saturate_cast_func<typename PtrTraits<SrcPtr>::value_type, D> > >
cvt_(const SrcPtr& src)
{
    return makeExpr(transformPtr(src, saturate_cast_func<typename PtrTraits<SrcPtr>::value_type, D>()));
}

// lut

template <class SrcPtr, class TablePtr>
__host__ Expr<LutPtrSz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<TablePtr>::ptr_type> >
lut_(const SrcPtr& src, const TablePtr& tbl)
{
    return makeExpr(lutPtr(src, tbl));
}

//! @}

}}

#endif
