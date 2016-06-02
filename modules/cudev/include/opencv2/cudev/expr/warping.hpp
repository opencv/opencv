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

#ifndef __OPENCV_CUDEV_EXPR_WARPING_HPP__
#define __OPENCV_CUDEV_EXPR_WARPING_HPP__

#include "../common.hpp"
#include "../ptr2d/traits.hpp"
#include "../ptr2d/resize.hpp"
#include "../ptr2d/remap.hpp"
#include "../ptr2d/warping.hpp"
#include "../grid/pyramids.hpp"
#include "../grid/transpose.hpp"
#include "expr.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

// resize

template <class SrcPtr>
__host__ Expr<ResizePtrSz<typename PtrTraits<SrcPtr>::ptr_type> >
resize_(const SrcPtr& src, float fx, float fy)
{
    return makeExpr(resizePtr(src, fx, fy));
}

// remap

template <class SrcPtr, class MapPtr>
__host__ Expr<RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<MapPtr>::ptr_type> >
remap_(const SrcPtr& src, const MapPtr& map)
{
    return makeExpr(remapPtr(src, map));
}

template <class SrcPtr, class MapXPtr, class MapYPtr>
__host__ Expr<RemapPtr2Sz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<MapXPtr>::ptr_type, typename PtrTraits<MapYPtr>::ptr_type> >
remap_(const SrcPtr& src, const MapXPtr& mapx, const MapYPtr& mapy)
{
    return makeExpr(remapPtr(src, mapx, mapy));
}

// warpAffine

template <class SrcPtr>
__host__ Expr<RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, AffineMapPtr> >
warpAffine_(const SrcPtr& src, Size dstSize, const GpuMat_<float>& warpMat)
{
    return makeExpr(warpAffinePtr(src, dstSize, warpMat));
}

// warpPerspective

template <class SrcPtr>
__host__ Expr<RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, PerspectiveMapPtr> >
warpPerspective_(const SrcPtr& src, Size dstSize, const GpuMat_<float>& warpMat)
{
    return makeExpr(warpPerspectivePtr(src, dstSize, warpMat));
}

// pyrDown

template <class SrcPtr> struct PyrDownBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridPyrDown(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<PyrDownBody<SrcPtr> >
pyrDown_(const SrcPtr& src)
{
    PyrDownBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// pyrUp

template <class SrcPtr> struct PyrUpBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridPyrUp(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<PyrUpBody<SrcPtr> >
pyrUp_(const SrcPtr& src)
{
    PyrUpBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

// transpose

template <class SrcPtr> struct TransposeBody
{
    SrcPtr src;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridTranspose(src, dst, stream);
    }
};

template <class SrcPtr>
__host__ Expr<TransposeBody<SrcPtr> >
transpose_(const SrcPtr& src)
{
    TransposeBody<SrcPtr> body;
    body.src = src;
    return makeExpr(body);
}

//! @}

}}

#endif
