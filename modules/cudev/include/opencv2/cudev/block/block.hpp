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

#ifndef __OPENCV_CUDEV_BLOCK_BLOCK_HPP__
#define __OPENCV_CUDEV_BLOCK_BLOCK_HPP__

#include "../common.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

struct Block
{
    __device__ __forceinline__ static uint blockId()
    {
        return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    }

    __device__ __forceinline__ static uint blockSize()
    {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    __device__ __forceinline__ static uint threadLineId()
    {
        return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    }
};

template <class It, typename T>
__device__ __forceinline__ static void blockFill(It beg, It end, const T& value)
{
    uint STRIDE = Block::blockSize();
    It t = beg + Block::threadLineId();

    for(; t < end; t += STRIDE)
        *t = value;
}

template <class OutIt, typename T>
__device__ __forceinline__ static void blockYota(OutIt beg, OutIt end, T value)
{
    uint STRIDE = Block::blockSize();
    uint tid = Block::threadLineId();
    value += tid;

    for(OutIt t = beg + tid; t < end; t += STRIDE, value += STRIDE)
        *t = value;
}

template <class InIt, class OutIt>
__device__ __forceinline__ static void blockCopy(InIt beg, InIt end, OutIt out)
{
    uint STRIDE = Block::blockSize();
    InIt  t = beg + Block::threadLineId();
    OutIt o = out + (t - beg);

    for(; t < end; t += STRIDE, o += STRIDE)
        *o = *t;
}

template <class InIt, class OutIt, class UnOp>
__device__ __forceinline__ static void blockTransfrom(InIt beg, InIt end, OutIt out, const UnOp& op)
{
    uint STRIDE = Block::blockSize();
    InIt  t = beg + Block::threadLineId();
    OutIt o = out + (t - beg);

    for(; t < end; t += STRIDE, o += STRIDE)
        *o = op(*t);
}

template <class InIt1, class InIt2, class OutIt, class BinOp>
__device__ __forceinline__ static void blockTransfrom(InIt1 beg1, InIt1 end1, InIt2 beg2, OutIt out, const BinOp& op)
{
    uint STRIDE = Block::blockSize();
    InIt1 t1 = beg1 + Block::threadLineId();
    InIt2 t2 = beg2 + Block::threadLineId();
    OutIt o  = out + (t1 - beg1);

    for(; t1 < end1; t1 += STRIDE, t2 += STRIDE, o += STRIDE)
        *o = op(*t1, *t2);
}

//! @}

}}

#endif
