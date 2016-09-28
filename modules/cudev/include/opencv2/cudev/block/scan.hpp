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

namespace cv { namespace cudev {

//! @addtogroup cudev
//! @{

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

        if (tid < (THREADS_NUM / WARP_SIZE))
        {
            // grab top warp elements
            T val = smem[tid];

            // calculate exclusive scan and write back to shared memory
            smem[tid] = warpScanExclusive(val, smem, tid);
        }

        __syncthreads();

        // return updated warp scans with exclusive scan results
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

//! @}

}}

#endif
