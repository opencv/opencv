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

#ifndef OPENCV_CUDEV_WARP_SCAN_HPP
#define OPENCV_CUDEV_WARP_SCAN_HPP

#include "../common.hpp"
#include "warp.hpp"
#include "shuffle.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

#if __CUDACC_VER_MAJOR__ >= 9

// Starting from CUDA 9.0, support for Fermi is dropped.
// So CV_CUDEV_ARCH >= 300 is implied.

template <typename T>
__device__ T warpScanInclusive(uint mask, T data)
{
    const uint laneId = Warp::laneId();

    // scan on shufl functions
    #pragma unroll
    for (int i = 1; i <= (WARP_SIZE / 2); i *= 2)
    {
        const T val = shfl_up_sync(mask, data, i);
        if (laneId >= i)
              data += val;
    }

    return data;
}

template <typename T>
__device__ __forceinline__ T warpScanExclusive(uint mask, T data)
{
    return warpScanInclusive(mask, data) - data;
}

#else // __CUDACC_VER_MAJOR__ >= 9

template <typename T>
__device__ T warpScanInclusive(T data, volatile T* smem, uint tid)
{
#if CV_CUDEV_ARCH >= 300
    CV_UNUSED(smem);
    CV_UNUSED(tid);

    const uint laneId = Warp::laneId();

    // scan on shufl functions
    #pragma unroll
    for (int i = 1; i <= (WARP_SIZE / 2); i *= 2)
    {
        const T val = shfl_up(data, i);
        if (laneId >= i)
              data += val;
    }

    return data;
#else
    const uint laneId = Warp::laneId();

    smem[tid] = data;

    #pragma unroll
    for (int i = 1; i <= (WARP_SIZE / 2); i *= 2)
        if (laneId >= i)
            smem[tid] += smem[tid - i];

    return smem[tid];
#endif
}

template <typename T>
__device__ __forceinline__ T warpScanExclusive(T data, volatile T* smem, uint tid)
{
    return warpScanInclusive(data, smem, tid) - data;
}

#endif // __CUDACC_VER_MAJOR__ >= 9

//! @}

}}

#endif
