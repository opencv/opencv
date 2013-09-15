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

#ifndef __OPENCV_CUDEV_PTR2D_WARPING_HPP__
#define __OPENCV_CUDEV_PTR2D_WARPING_HPP__

#include "../common.hpp"
#include "traits.hpp"
#include "remap.hpp"
#include "gpumat.hpp"

namespace cv { namespace cudev {

// affine

struct AffineMapPtr
{
    typedef float2 value_type;
    typedef float  index_type;

    const float* warpMat;

    __device__ __forceinline__ float2 operator ()(float y, float x) const
    {
        const float xcoo = warpMat[0] * x + warpMat[1] * y + warpMat[2];
        const float ycoo = warpMat[3] * x + warpMat[4] * y + warpMat[5];

        return make_float2(xcoo, ycoo);
    }
};

struct AffineMapPtrSz : AffineMapPtr
{
    int rows, cols;
};

template <> struct PtrTraits<AffineMapPtrSz> : PtrTraitsBase<AffineMapPtrSz, AffineMapPtr>
{
};

__host__ static AffineMapPtrSz affineMap(Size dstSize, const GpuMat_<float>& warpMat)
{
    CV_Assert( warpMat.rows == 2 && warpMat.cols == 3 );
    CV_Assert( warpMat.isContinuous() );

    AffineMapPtrSz map;
    map.warpMat = warpMat[0];
    map.rows = dstSize.height;
    map.cols = dstSize.width;
    return map;
}

template <class SrcPtr>
__host__ RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, AffineMapPtr>
warpAffinePtr(const SrcPtr& src, Size dstSize, const GpuMat_<float>& warpMat)
{
    return remapPtr(src, affineMap(dstSize, warpMat));
}

// perspective

struct PerspectiveMapPtr
{
    typedef float2 value_type;
    typedef float  index_type;

    const float* warpMat;

    __device__ __forceinline__ float2 operator ()(float y, float x) const
    {
        const float coeff = 1.0f / (warpMat[6] * x + warpMat[7] * y + warpMat[8]);

        const float xcoo = coeff * (warpMat[0] * x + warpMat[1] * y + warpMat[2]);
        const float ycoo = coeff * (warpMat[3] * x + warpMat[4] * y + warpMat[5]);

        return make_float2(xcoo, ycoo);
    }
};

struct PerspectiveMapPtrSz : PerspectiveMapPtr
{
    int rows, cols;
};

template <> struct PtrTraits<PerspectiveMapPtrSz> : PtrTraitsBase<PerspectiveMapPtrSz, PerspectiveMapPtr>
{
};

__host__ static PerspectiveMapPtrSz perspectiveMap(Size dstSize, const GpuMat_<float>& warpMat)
{
    CV_Assert( warpMat.rows == 3 && warpMat.cols == 3 );
    CV_Assert( warpMat.isContinuous() );

    PerspectiveMapPtrSz map;
    map.warpMat = warpMat[0];
    map.rows = dstSize.height;
    map.cols = dstSize.width;
    return map;
}

template <class SrcPtr>
__host__ RemapPtr1Sz<typename PtrTraits<SrcPtr>::ptr_type, PerspectiveMapPtr>
warpPerspectivePtr(const SrcPtr& src, Size dstSize, const GpuMat_<float>& warpMat)
{
    return remapPtr(src, perspectiveMap(dstSize, warpMat));
}

}}

#endif
