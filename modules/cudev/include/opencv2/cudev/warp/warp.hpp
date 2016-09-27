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

#ifndef OPENCV_CUDEV_WARP_WARP_HPP
#define OPENCV_CUDEV_WARP_WARP_HPP

#include "../common.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

enum
{
    LOG_WARP_SIZE = 5,
    WARP_SIZE     = 1 << LOG_WARP_SIZE
};

struct Warp
{
    __device__ __forceinline__ static uint laneId()
    {
        uint ret;
        asm("mov.u32 %0, %laneid;" : "=r"(ret));
        return ret;
    }

    __device__ __forceinline__ static uint warpId()
    {
        const uint tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        return tid / WARP_SIZE;
    }
};

template <class It, typename T>
__device__ __forceinline__ void warpFill(It beg, It end, const T& value)
{
    for(It t = beg + Warp::laneId(); t < end; t += WARP_SIZE)
        *t = value;
}

template <class InIt, class OutIt>
__device__ __forceinline__ OutIt warpCopy(InIt beg, InIt end, OutIt out)
{
    for(InIt t = beg + Warp::laneId(); t < end; t += WARP_SIZE, out += WARP_SIZE)
        *out = *t;
    return out;
}

template <class InIt, class OutIt, class UnOp>
__device__ __forceinline__ OutIt warpTransform(InIt beg, InIt end, OutIt out, const UnOp& op)
{
    for(InIt t = beg + Warp::laneId(); t < end; t += WARP_SIZE, out += WARP_SIZE)
        *out = op(*t);
    return out;
}

template <class InIt1, class InIt2, class OutIt, class BinOp>
__device__ __forceinline__ OutIt warpTransform(InIt1 beg1, InIt1 end1, InIt2 beg2, OutIt out, const BinOp& op)
{
    uint lane = Warp::laneId();

    InIt1 t1 = beg1 + lane;
    InIt2 t2 = beg2 + lane;
    for(; t1 < end1; t1 += WARP_SIZE, t2 += WARP_SIZE, out += WARP_SIZE)
        *out = op(*t1, *t2);
    return out;
}

template<typename OutIt, typename T>
__device__ __forceinline__ void warpYota(OutIt beg, OutIt end, T value)
{
    uint lane = Warp::laneId();
    value += lane;

    for(OutIt t = beg + lane; t < end; t += WARP_SIZE, value += WARP_SIZE)
        *t = value;
}

//! @}

}}

#endif
