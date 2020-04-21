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

#ifndef OPENCV_CUDEV_BLOCK_SCAN_HPP
#define OPENCV_CUDEV_BLOCK_SCAN_HPP

#include "../common.hpp"
#include "../warp/scan.hpp"
#include "../warp/warp.hpp"

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

#if __CUDACC_VER_MAJOR__ >= 9

// Usage Note
// - THREADS_NUM should be equal to the number of threads in this block.
// - smem must be able to contain at least n elements of type T, where n is equal to the number
//   of warps in this block. The number can be calculated by divUp(THREADS_NUM, WARP_SIZE).
//
// Dev Note
// - Starting from CUDA 9.0, support for Fermi is dropped. So CV_CUDEV_ARCH >= 300 is implied.
// - "For Pascal and earlier architectures (CV_CUDEV_ARCH < 700), all threads in mask must execute
//    the same warp intrinsic instruction in convergence, and the union of all values in mask must
//    be equal to the warp's active mask."
//   (https://docs.nvidia.com/cuda/archive/10.0/cuda-c-programming-guide#independent-thread-scheduling-7-x)
// - Above restriction does not apply starting from Volta (CV_CUDEV_ARCH >= 700). We just need to
//   take care so that "all non-exited threads named in mask must execute the same intrinsic with
//   the same mask."
//   (https://docs.nvidia.com/cuda/archive/10.0/cuda-c-programming-guide#warp-description)

template <int THREADS_NUM, typename T>
__device__ T blockScanInclusive(T data, volatile T* smem, uint tid)
{
    const int residual = THREADS_NUM & (WARP_SIZE - 1);

#if CV_CUDEV_ARCH < 700
    const uint residual_mask = (1U << residual) - 1;
#endif

    if (THREADS_NUM > WARP_SIZE)
    {
        // bottom-level inclusive warp scan
    #if CV_CUDEV_ARCH >= 700
        T warpResult = warpScanInclusive(0xFFFFFFFFU, data);
    #else
        T warpResult;

        if (0 == residual)
            warpResult = warpScanInclusive(0xFFFFFFFFU, data);
        else
        {
            const int n_warps = divUp(THREADS_NUM, WARP_SIZE);
            const int warp_num = Warp::warpId();

            if (warp_num < n_warps - 1)
                warpResult = warpScanInclusive(0xFFFFFFFFU, data);
            else
            {
                // We are at the last threads of a block whose number of threads
                // is not a multiple of the warp size
                warpResult = warpScanInclusive(residual_mask, data);
            }
        }
    #endif

        __syncthreads();

        // save top elements of each warp for exclusive warp scan
        // sync to wait for warp scans to complete (because smem is being overwritten)
        if ((tid & (WARP_SIZE - 1)) == (WARP_SIZE - 1))
        {
            smem[tid >> LOG_WARP_SIZE] = warpResult;
        }

        __syncthreads();

        int quot = THREADS_NUM / WARP_SIZE;

        if (tid < quot)
        {
            // grab top warp elements
            T val = smem[tid];

            uint mask = (1LLU << quot) - 1;

            if (0 == residual)
            {
                // calculate exclusive scan and write back to shared memory
                smem[tid] = warpScanExclusive(mask, val);
            }
            else
            {
                // Read from smem[tid]              (T val = smem[tid])
                // and write to smem[tid + 1]       (smem[tid + 1] = warpScanInclusive(mask, val))
                // should be explicitly fenced by "__syncwarp" to get rid of
                // "cuda-memcheck --tool racecheck" warnings.
                __syncwarp(mask);

                // calculate inclusive scan and write back to shared memory with offset 1
                smem[tid + 1] = warpScanInclusive(mask, val);

                if (tid == 0)
                    smem[0] = 0;
            }
        }

        __syncthreads();

        // return updated warp scans
        return warpResult + smem[tid >> LOG_WARP_SIZE];
    }
    else
    {
    #if CV_CUDEV_ARCH >= 700
        return warpScanInclusive(0xFFFFFFFFU, data);
    #else
        if (THREADS_NUM == WARP_SIZE)
            return warpScanInclusive(0xFFFFFFFFU, data);
        else
            return warpScanInclusive(residual_mask, data);
    #endif
    }
}

template <int THREADS_NUM, typename T>
__device__ __forceinline__ T blockScanExclusive(T data, volatile T* smem, uint tid)
{
    return blockScanInclusive<THREADS_NUM>(data, smem, tid) - data;
}

#else // __CUDACC_VER_MAJOR__ >= 9

// Usage Note
// - THREADS_NUM should be equal to the number of threads in this block.
// - (>= Kepler) smem must be able to contain at least n elements of type T, where n is equal to the number
//   of warps in this block. The number can be calculated by divUp(THREADS_NUM, WARP_SIZE).
// - (Fermi) smem must be able to contain at least n elements of type T, where n is equal to the number
//   of threads in this block (= THREADS_NUM).

template <int THREADS_NUM, typename T>
__device__ T blockScanInclusive(T data, volatile T* smem, uint tid)
{
    if (THREADS_NUM > WARP_SIZE)
    {
        // bottom-level inclusive warp scan
        T warpResult = warpScanInclusive(data, smem, tid);

        __syncthreads();

        // save top elements of each warp for exclusive warp scan
        // sync to wait for warp scans to complete (because s_Data is being overwritten)
        if ((tid & (WARP_SIZE - 1)) == (WARP_SIZE - 1))
        {
            smem[tid >> LOG_WARP_SIZE] = warpResult;
        }

        __syncthreads();

        int quot = THREADS_NUM / WARP_SIZE;

        T val;

        if (tid < quot)
        {
            // grab top warp elements
            val = smem[tid];
        }

        __syncthreads();

        if (tid < quot)
        {

            if (0 == (THREADS_NUM & (WARP_SIZE - 1)))
            {
                // calculate exclusive scan and write back to shared memory
                smem[tid] = warpScanExclusive(val, smem, tid);
            }
            else
            {
                // calculate inclusive scan and write back to shared memory with offset 1
                smem[tid + 1] = warpScanInclusive(val, smem, tid);

                if (tid == 0)
                    smem[0] = 0;
            }
        }

        __syncthreads();

        // return updated warp scans
        return warpResult + smem[tid >> LOG_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(data, smem, tid);
    }
}

template <int THREADS_NUM, typename T>
__device__ __forceinline__ T blockScanExclusive(T data, volatile T* smem, uint tid)
{
    return blockScanInclusive<THREADS_NUM>(data, smem, tid) - data;
}

#endif // __CUDACC_VER_MAJOR__ >= 9

//! @}

}}

#endif
