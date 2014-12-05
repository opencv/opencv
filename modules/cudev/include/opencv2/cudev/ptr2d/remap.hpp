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

#ifndef __OPENCV_CUDEV_PTR2D_REMAP_HPP__
#define __OPENCV_CUDEV_PTR2D_REMAP_HPP__

#include "opencv2/core/base.hpp"
#include "../common.hpp"
#include "../grid/copy.hpp"
#include "traits.hpp"
#include "gpumat.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

template <class SrcPtr, class MapPtr> struct RemapPtr1
{
    typedef typename PtrTraits<SrcPtr>::value_type value_type;
    typedef typename PtrTraits<MapPtr>::index_type index_type;

    SrcPtr src;
    MapPtr map;

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(typename PtrTraits<MapPtr>::index_type y, typename PtrTraits<MapPtr>::index_type x) const
    {
        const typename PtrTraits<MapPtr>::value_type coord = map(y, x);
        return src(coord.y, coord.x);
    }
};

template <class SrcPtr, class MapXPtr, class MapYPtr> struct RemapPtr2
{
    typedef typename PtrTraits<SrcPtr>::value_type  value_type;
    typedef typename PtrTraits<MapXPtr>::index_type index_type;

    SrcPtr src;
    MapXPtr mapx;
    MapYPtr mapy;

    __device__ __forceinline__ typename PtrTraits<SrcPtr>::value_type operator ()(typename PtrTraits<MapXPtr>::index_type y, typename PtrTraits<MapXPtr>::index_type x) const
    {
        const typename PtrTraits<MapXPtr>::value_type nx = mapx(y, x);
        const typename PtrTraits<MapYPtr>::value_type ny = mapy(y, x);
        return src(ny, nx);
    }
};

template <class SrcPtr, class MapPtr> struct RemapPtr1Sz : RemapPtr1<SrcPtr, MapPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr, class MapXPtr, class MapYPtr> struct RemapPtr2Sz : RemapPtr2<SrcPtr, MapXPtr, MapYPtr>
{
    int rows, cols;

    template <typename T>
    __host__ void assignTo(GpuMat_<T>& dst, Stream& stream = Stream::Null()) const
    {
        gridCopy(*this, dst, stream);
    }
};

template <class SrcPtr, class MapPtr>
__host__ RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<MapPtr>::ptr_type>
remapPtr(const SrcPtr& src, const MapPtr& map)
{
    const int rows = getRows(map);
    const int cols = getCols(map);

    RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<MapPtr>::ptr_type> r;
    r.src = shrinkPtr(src);
    r.map = shrinkPtr(map);
    r.rows = rows;
    r.cols = cols;
    return r;
}

template <class SrcPtr, class MapXPtr, class MapYPtr>
__host__ RemapPtr2Sz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<MapXPtr>::ptr_type, typename PtrTraits<MapYPtr>::ptr_type>
remapPtr(const SrcPtr& src, const MapXPtr& mapx, const MapYPtr& mapy)
{
    const int rows = getRows(mapx);
    const int cols = getCols(mapx);

    CV_Assert( getRows(mapy) == rows && getCols(mapy) == cols );

    RemapPtr2Sz<typename PtrTraits<SrcPtr>::ptr_type, typename PtrTraits<MapXPtr>::ptr_type, typename PtrTraits<MapYPtr>::ptr_type> r;
    r.src = shrinkPtr(src);
    r.mapx = shrinkPtr(mapx);
    r.mapy = shrinkPtr(mapy);
    r.rows = rows;
    r.cols = cols;
    return r;
}

template <class SrcPtr, class MapPtr> struct PtrTraits< RemapPtr1Sz<SrcPtr, MapPtr> > : PtrTraitsBase<RemapPtr1Sz<SrcPtr, MapPtr>, RemapPtr1<SrcPtr, MapPtr> >
{
};

template <class SrcPtr, class MapXPtr, class MapYPtr> struct PtrTraits< RemapPtr2Sz<SrcPtr, MapXPtr, MapYPtr> > : PtrTraitsBase<RemapPtr2Sz<SrcPtr, MapXPtr, MapYPtr>, RemapPtr2<SrcPtr, MapXPtr, MapYPtr> >
{
};

//! @}

}}

#endif
