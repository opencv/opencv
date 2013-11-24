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

#ifndef __OPENCV_CUDEV_GRID_HISTOGRAM_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_HISTOGRAM_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/atomic.hpp"

namespace cv { namespace cudev {

namespace grid_histogram_detail
{
    template <int BIN_COUNT, int BLOCK_SIZE, class SrcPtr, typename ResType, class MaskPtr>
    __global__ void histogram(const SrcPtr src, ResType* hist, const MaskPtr mask, const int rows, const int cols)
    {
    #if CV_CUDEV_ARCH >= 120
        __shared__ ResType smem[BIN_COUNT];

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        for (int i = tid; i < BIN_COUNT; i += BLOCK_SIZE)
            smem[i] = 0;

        __syncthreads();

        if (y < rows)
        {
            for (int x = threadIdx.x; x < cols; x += blockDim.x)
            {
                if (mask(y, x))
                {
                    const uint data = src(y, x);
                    atomicAdd(&smem[data % BIN_COUNT], 1);
                }
            }
        }

        __syncthreads();

        for (int i = tid; i < BIN_COUNT; i += BLOCK_SIZE)
        {
            const ResType histVal = smem[i];
            if (histVal > 0)
                atomicAdd(hist + i, histVal);
        }
    #endif
    }

    template <int BIN_COUNT, class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void histogram(const SrcPtr& src, ResType* hist, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(rows, block.y));

        const int BLOCK_SIZE = Policy::block_size_x * Policy::block_size_y;

        histogram<BIN_COUNT, BLOCK_SIZE><<<grid, block, 0, stream>>>(src, hist, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
