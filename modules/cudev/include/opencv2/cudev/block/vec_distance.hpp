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

#ifndef __OPENCV_CUDEV_BLOCK_VEC_DISTANCE_HPP__
#define __OPENCV_CUDEV_BLOCK_VEC_DISTANCE_HPP__

#include "../common.hpp"
#include "../functional/functional.hpp"
#include "../warp/reduce.hpp"
#include "reduce.hpp"

namespace cv { namespace cudev {

// NormL1

template <typename T> struct NormL1
{
    typedef int value_type;
    typedef uint result_type;

    result_type mySum;

    __device__ __forceinline__ NormL1() : mySum(0) {}

    __device__ __forceinline__ void reduceThread(value_type val1, value_type val2)
    {
        mySum = __sad(val1, val2, mySum);
    }

    __device__ __forceinline__ void reduceWarp(result_type* smem, uint tid)
    {
        warpReduce(smem, mySum, tid, plus<result_type>());
    }

    template <int THREAD_DIM> __device__ __forceinline__ void reduceBlock(result_type* smem, uint tid)
    {
        blockReduce<THREAD_DIM>(smem, mySum, tid, plus<result_type>());
    }

    __device__ __forceinline__ operator result_type() const
    {
        return mySum;
    }
};
template <> struct NormL1<float>
{
    typedef float value_type;
    typedef float result_type;

    result_type mySum;

    __device__ __forceinline__ NormL1() : mySum(0.0f) {}

    __device__ __forceinline__ void reduceThread(value_type val1, value_type val2)
    {
        mySum += ::fabsf(val1 - val2);
    }

    __device__ __forceinline__ void reduceWarp(result_type* smem, uint tid)
    {
        warpReduce(smem, mySum, tid, plus<result_type>());
    }

    template <int THREAD_DIM> __device__ __forceinline__ void reduceBlock(result_type* smem, uint tid)
    {
        blockReduce<THREAD_DIM>(smem, mySum, tid, plus<result_type>());
    }

    __device__ __forceinline__ operator result_type() const
    {
        return mySum;
    }
};

// NormL2

struct NormL2
{
    typedef float value_type;
    typedef float result_type;

    result_type mySum;

    __device__ __forceinline__ NormL2() : mySum(0.0f) {}

    __device__ __forceinline__ void reduceThread(value_type val1, value_type val2)
    {
        const float diff = val1 - val2;
        mySum += diff * diff;
    }

    __device__ __forceinline__ void reduceWarp(result_type* smem, uint tid)
    {
        warpReduce(smem, mySum, tid, plus<result_type>());
    }

    template <int THREAD_DIM> __device__ __forceinline__ void reduceBlock(result_type* smem, uint tid)
    {
        blockReduce<THREAD_DIM>(smem, mySum, tid, plus<result_type>());
    }

    __device__ __forceinline__ operator result_type() const
    {
        return ::sqrtf(mySum);
    }
};

// NormHamming

struct NormHamming
{
    typedef int value_type;
    typedef int result_type;

    result_type mySum;

    __device__ __forceinline__ NormHamming() : mySum(0) {}

    __device__ __forceinline__ void reduceThread(value_type val1, value_type val2)
    {
        mySum += __popc(val1 ^ val2);
    }

    __device__ __forceinline__ void reduceWarp(result_type* smem, uint tid)
    {
        warpReduce(smem, mySum, tid, plus<result_type>());
    }

    template <int THREAD_DIM> __device__ __forceinline__ void reduceBlock(result_type* smem, uint tid)
    {
        blockReduce<THREAD_DIM>(smem, mySum, tid, plus<result_type>());
    }

    __device__ __forceinline__ operator result_type() const
    {
        return mySum;
    }
};

}}

#endif
