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

#ifndef __OPENCV_CUDEV_PTR2D_LUT_HPP__
#define __OPENCV_CUDEV_PTR2D_LUT_HPP__

#include "../common.hpp"
#include "../util/vec_traits.hpp"
#include "../grid/copy.hpp"
#include "traits.hpp"
#include "gpumat.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

template <class SrcPtr, class TablePtr> struct LutPtr
{
    typedef typename PtrTraits<TablePtr>::value_type value_type;
    typedef typename PtrTraits<SrcPtr>::index_type   index_type;

    SrcPtr src;
    TablePtr tbl;

    __device__ __forceinline__ typename PtrTraits<TablePtr>::value_type operator ()(typename PtrTraits<SrcPtr>::index_type y, typename PtrTraits<SrcPtr>::index_type x) const
    {
        typedef typename PtrTraits<TablePtr>::index_type tbl_index_type;
        return tbl(VecTraits<tbl_index_type>::all(0), src(y, x));
    }
};

template <class SrcPtr, class TablePtr> struct LutPtrSz : LutPtr<SrcPtr, TablePtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr, class TablePtr>
__host__ LutPtrSz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<TablePtr>::ptr_type> lutPtr(const SrcPtr& src, const TablePtr& tbl)
{
    LutPtrSz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<TablePtr>::ptr_type> ptr;
    ptr.src = shrinkPtr(src);
    ptr.tbl = shrinkPtr(tbl);
    ptr.rows = getRows(src);
    ptr.cols = getCols(src);
    return ptr;
}

template <class SrcPtr, class TablePtr> struct PtrTraits< LutPtrSz<SrcPtr, TablePtr> > : PtrTraitsBase<LutPtrSz<SrcPtr, TablePtr>, LutPtr<SrcPtr, TablePtr> >
{
};

//! @}

}}

#endif
