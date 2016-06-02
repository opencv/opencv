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

#ifndef __OPENCV_CUDEV_PTR2D_TRANSFORM_HPP__
#define __OPENCV_CUDEV_PTR2D_TRANSFORM_HPP__

#include "../common.hpp"
#include "../grid/copy.hpp"
#include "traits.hpp"
#include "gpumat.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// UnaryTransformPtr

template <class SrcPtr, class Op> struct UnaryTransformPtr
{
    typedef typename Op::result_type               value_type;
    typedef typename PtrTraits<SrcPtr>::index_type index_type;

    SrcPtr src;
    Op op;

    __device__ __forceinline__ typename Op::result_type operator ()(typename PtrTraits<SrcPtr>::index_type y, typename PtrTraits<SrcPtr>::index_type x) const
    {
        return op(src(y, x));
    }
};

template <class SrcPtr, class Op> struct UnaryTransformPtrSz : UnaryTransformPtr<SrcPtr, Op>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr, class Op>
__host__ UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, Op>
transformPtr(const SrcPtr& src, const Op& op)
{
    UnaryTransformPtrSz<typename PtrTraits<SrcPtr>::ptr_type, Op> ptr;
    ptr.src = shrinkPtr(src);
    ptr.op = op;
    ptr.rows = getRows(src);
    ptr.cols = getCols(src);
    return ptr;
}

template <class SrcPtr, class Op> struct PtrTraits< UnaryTransformPtrSz<SrcPtr, Op> > : PtrTraitsBase<UnaryTransformPtrSz<SrcPtr, Op>, UnaryTransformPtr<SrcPtr, Op> >
{
};

// BinaryTransformPtr

template <class Src1Ptr, class Src2Ptr, class Op> struct BinaryTransformPtr
{
    typedef typename Op::result_type                value_type;
    typedef typename PtrTraits<Src1Ptr>::index_type index_type;

    Src1Ptr src1;
    Src2Ptr src2;
    Op op;

    __device__ __forceinline__ typename Op::result_type operator ()(typename PtrTraits<Src1Ptr>::index_type y, typename PtrTraits<Src1Ptr>::index_type x) const
    {
        return op(src1(y, x), src2(y, x));
    }
};

template <class Src1Ptr, class Src2Ptr, class Op> struct BinaryTransformPtrSz : BinaryTransformPtr<Src1Ptr, Src2Ptr, Op>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class Src1Ptr, class Src2Ptr, class Op>
__host__ BinaryTransformPtrSz<typename PtrTraits<Src1Ptr>::ptr_type, typename PtrTraits<Src2Ptr>::ptr_type, Op>
transformPtr(const Src1Ptr& src1, const Src2Ptr& src2, const Op& op)
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );

    BinaryTransformPtrSz<typename PtrTraits<Src1Ptr>::ptr_type, typename PtrTraits<Src2Ptr>::ptr_type, Op> ptr;
    ptr.src1 = shrinkPtr(src1);
    ptr.src2 = shrinkPtr(src2);
    ptr.op = op;
    ptr.rows = rows;
    ptr.cols = cols;
    return ptr;
}

template <class Src1Ptr, class Src2Ptr, class Op> struct PtrTraits< BinaryTransformPtrSz<Src1Ptr, Src2Ptr, Op> > : PtrTraitsBase<BinaryTransformPtrSz<Src1Ptr, Src2Ptr, Op>, BinaryTransformPtr<Src1Ptr, Src2Ptr, Op> >
{
};

//! @}

}}

#endif
