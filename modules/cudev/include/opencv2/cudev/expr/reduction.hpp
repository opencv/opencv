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

#ifndef OPENCV_CUDEV_EXPR_REDUCTION_HPP
#define OPENCV_CUDEV_EXPR_REDUCTION_HPP

#include "../common.hpp"
#include "../grid/reduce.hpp"
#include "../grid/histogram.hpp"
#include "../grid/integral.hpp"
#include "../grid/reduce_to_vec.hpp"
#include "../ptr2d/traits.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// sum

template <class SrcPtr> struct SumExprBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCalcSum(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<SumExprBody<SrcPtr> >
sum_(const SrcPtr& src)
{
    SumExprBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// minVal

template <class SrcPtr> struct FindMinValExprBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridFindMinVal(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<FindMinValExprBody<SrcPtr> >
minVal_(const SrcPtr& src)
{
    FindMinValExprBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// maxVal

template <class SrcPtr> struct FindMaxValExprBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridFindMaxVal(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<FindMaxValExprBody<SrcPtr> >
maxVal_(const SrcPtr& src)
{
    FindMaxValExprBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// minMaxVal

template <class SrcPtr> struct FindMinMaxValExprBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridFindMinMaxVal(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<FindMinMaxValExprBody<SrcPtr> >
minMaxVal_(const SrcPtr& src)
{
    FindMinMaxValExprBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// countNonZero

template <class SrcPtr> struct CountNonZeroExprBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCountNonZero(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<CountNonZeroExprBody<SrcPtr> >
countNonZero_(const SrcPtr& src)
{
    CountNonZeroExprBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// reduceToRow

template <class Reductor, class SrcPtr> struct ReduceToRowBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridReduceToRow<Reductor>(src, dst, stream);
    }
};

template <class Reductor, class SrcPtr>
__host__ Expr<ReduceToRowBody<Reductor, SrcPtr> >
reduceToRow_(const SrcPtr& src)
{
    ReduceToRowBody<Reductor, SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// reduceToColumn

template <class Reductor, class SrcPtr> struct ReduceToColumnBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridReduceToColumn<Reductor>(src, dst, stream);
    }
};

template <class Reductor, class SrcPtr>
__host__ Expr<ReduceToColumnBody<Reductor, SrcPtr> >
reduceToColumn_(const SrcPtr& src)
{
    ReduceToColumnBody<Reductor, SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// histogram

template <int BIN_COUNT, class SrcPtr> struct HistogramBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridHistogram<BIN_COUNT>(src, dst, stream);
    }
};

template <int BIN_COUNT, class SrcPtr>
__host__ Expr<HistogramBody<BIN_COUNT, SrcPtr> >
histogram_(const SrcPtr& src)
{
    HistogramBody<BIN_COUNT, SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// integral

template <class SrcPtr> struct IntegralBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridIntegral(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<IntegralBody<SrcPtr> >
integral_(const SrcPtr& src)
{
    IntegralBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

//! @}

}}

#endif
