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

#ifndef OPENCV_CUDEV_PTR2D_EXTRAPOLATION_HPP
#define OPENCV_CUDEV_PTR2D_EXTRAPOLATION_HPP

#include "../common.hpp"
#include "../util/vec_traits.hpp"
#include "traits.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// BrdConstant

template <class SrcPtr> struct BrdConstant
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;
    int rows, cols;
    typename PtrTraits<SrcPtr>::value_type val;

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        return (x >= 0 && x < cols && y >= 0 && y < rows) ? src(y, x) : val;
    }
};

template <class SrcPtr>
__host__ BrdConstant<typename PtrTraits<SrcPtr>::ptr_type> brdConstant(const SrcPtr& src, typename PtrTraits<SrcPtr>::value_type val)
{
    BrdConstant<typename PtrTraits<SrcPtr>::ptr_type> b;
    b.src = shrinkPtr(src);
    b.rows = getRows(src);
    b.cols = getCols(src);
    b.val = val;
    return b;
}

template <class SrcPtr>
__host__ BrdConstant<typename PtrTraits<SrcPtr>::ptr_type> brdConstant(const SrcPtr& src)
{
    return brdConstant(src, VecTraits<typename PtrTraits<SrcPtr>::value_type>::all(0));
}

// BrdBase

template <class BrdImpl, class SrcPtr> struct BrdBase
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef int                                    index_type;

    SrcPtr src;
    int rows, cols;

    __device__ __forceinline__ int idx_row(int y) const
    {
        return BrdImpl::idx_low(BrdImpl::idx_high(y, rows), rows);
    }

    __device__ __forceinline__ int idx_col(int x) const
    {
        return BrdImpl::idx_low(BrdImpl::idx_high(x, cols), cols);
    }

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(int y, int x) const
    {
        return src(idx_row(y), idx_col(x));
    }
};

// BrdReplicate

struct BrdReplicate
{
    __device__ __forceinline__ static int idx_low(int i, int len)
    {
        return ::max(i, 0);
    }

    __device__ __forceinline__ static int idx_high(int i, int len)
    {
        return ::min(i, len - 1);
    }
};

template <class SrcPtr>
__host__ BrdBase<BrdReplicate, typename PtrTraits<SrcPtr>::ptr_type> brdReplicate(const SrcPtr& src)
{
    BrdBase<BrdReplicate, typename PtrTraits<SrcPtr>::ptr_type> b;
    b.src = shrinkPtr(src);
    b.rows = getRows(src);
    b.cols = getCols(src);
    return b;
}

// BrdReflect101

struct BrdReflect101
{
    __device__ __forceinline__ static int idx_low(int i, int len)
    {
        return ::abs(i) % len;
    }

    __device__ __forceinline__ static int idx_high(int i, int len)
    {
        const int last_ind = len - 1;
        return ::abs(last_ind - ::abs(last_ind - i)) % len;
    }
};

template <class SrcPtr>
__host__ BrdBase<BrdReflect101, typename PtrTraits<SrcPtr>::ptr_type> brdReflect101(const SrcPtr& src)
{
    BrdBase<BrdReflect101, typename PtrTraits<SrcPtr>::ptr_type> b;
    b.src = shrinkPtr(src);
    b.rows = getRows(src);
    b.cols = getCols(src);
    return b;
}

// BrdReflect

struct BrdReflect
{
    __device__ __forceinline__ static int idx_low(int i, int len)
    {
        return (::abs(i) - (i < 0)) % len;
    }

    __device__ __forceinline__ static int idx_high(int i, int len)
    {
        const int last_ind = len - 1;
        return (last_ind - ::abs(last_ind - i) + (i > last_ind));
    }
};

template <class SrcPtr>
__host__ BrdBase<BrdReflect, typename PtrTraits<SrcPtr>::ptr_type> brdReflect(const SrcPtr& src)
{
    BrdBase<BrdReflect, typename PtrTraits<SrcPtr>::ptr_type> b;
    b.src = shrinkPtr(src);
    b.rows = getRows(src);
    b.cols = getCols(src);
    return b;
}

// BrdWrap

struct BrdWrap
{
    __device__ __forceinline__ static int idx_low(int i, int len)
    {
        return (i >= 0) ? i : (i - ((i - len + 1) / len) * len);
    }

    __device__ __forceinline__ static int idx_high(int i, int len)
    {
        return (i < len) ? i : (i % len);
    }
};

template <class SrcPtr>
__host__ BrdBase<BrdWrap, typename PtrTraits<SrcPtr>::ptr_type> brdWrap(const SrcPtr& src)
{
    BrdBase<BrdWrap, typename PtrTraits<SrcPtr>::ptr_type> b;
    b.src = shrinkPtr(src);
    b.rows = getRows(src);
    b.cols = getCols(src);
    return b;
}

//! @}

}}

#endif
