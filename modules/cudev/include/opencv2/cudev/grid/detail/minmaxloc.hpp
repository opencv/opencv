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

#ifndef OPENCV_CUDEV_GRID_MINMAXLOC_DETAIL_HPP
#define OPENCV_CUDEV_GRID_MINMAXLOC_DETAIL_HPP

#include "../../common.hpp"
#include "../../util/vec_traits.hpp"
#include "../../util/type_traits.hpp"
#include "../../util/limits.hpp"
#include "../../block/reduce.hpp"

namespace cv { namespace cudev {

namespace grid_minmaxloc_detail
{
    template <int BLOCK_SIZE, class SrcPtr, typename ResType, class MaskPtr>
    __global__ void minMaxLoc_pass_1(const SrcPtr src, ResType* minVal, ResType* maxVal, int* minLoc, int* maxLoc, const MaskPtr mask, const int rows, const int cols, const int patch_y, const int patch_x)
    {
        __shared__ ResType sMinVal[BLOCK_SIZE];
        __shared__ ResType sMaxVal[BLOCK_SIZE];
        __shared__ uint sMinLoc[BLOCK_SIZE];
        __shared__ uint sMaxLoc[BLOCK_SIZE];

        const int x0 = blockIdx.x * blockDim.x * patch_x + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * patch_y + threadIdx.y;

        ResType myMin = numeric_limits<ResType>::max();
        ResType myMax = -numeric_limits<ResType>::max();
        int myMinLoc = -1;
        int myMaxLoc = -1;

        for (int i = 0, y = y0; i < patch_y && y < rows; ++i, y += blockDim.y)
        {
            for (int j = 0, x = x0; j < patch_x && x < cols; ++j, x += blockDim.x)
            {
                if (mask(y, x))
                {
                    const ResType srcVal = src(y, x);

                    if (srcVal < myMin)
                    {
                        myMin = srcVal;
                        myMinLoc = y * cols + x;
                    }

                    if (srcVal > myMax)
                    {
                        myMax = srcVal;
                        myMaxLoc = y * cols + x;
                    }
                }
            }
        }

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        blockReduceKeyVal<BLOCK_SIZE>(smem_tuple(sMinVal, sMaxVal), tie(myMin, myMax),
                                      smem_tuple(sMinLoc, sMaxLoc), tie(myMinLoc, myMaxLoc),
                                      tid,
                                      make_tuple(less<ResType>(), greater<ResType>()));

        const int bid = blockIdx.y * gridDim.x + blockIdx.x;

        if (tid == 0)
        {
            minVal[bid] = myMin;
            maxVal[bid] = myMax;
            minLoc[bid] = myMinLoc;
            maxLoc[bid] = myMaxLoc;
        }
    }

    template <int BLOCK_SIZE, typename T>
    __global__ void minMaxLoc_pass_2(T* minMal, T* maxVal, int* minLoc, int* maxLoc, int count)
    {
        __shared__ T sMinVal[BLOCK_SIZE];
        __shared__ T sMaxVal[BLOCK_SIZE];
        __shared__ int sMinLoc[BLOCK_SIZE];
        __shared__ int sMaxLoc[BLOCK_SIZE];

        const int idx = ::min(threadIdx.x, count - 1);

        T myMin = minMal[idx];
        T myMax = maxVal[idx];
        int myMinLoc = minLoc[idx];
        int myMaxLoc = maxLoc[idx];

        blockReduceKeyVal<BLOCK_SIZE>(smem_tuple(sMinVal, sMaxVal), tie(myMin, myMax),
                                      smem_tuple(sMinLoc, sMaxLoc), tie(myMinLoc, myMaxLoc),
                                      threadIdx.x,
                                      make_tuple(less<T>(), greater<T>()));

        if (threadIdx.x == 0)
        {
            minMal[0] = myMin;
            maxVal[0] = myMax;
            minLoc[0] = myMinLoc;
            maxLoc[0] = myMaxLoc;
        }
    }

    template <class Policy>
    void getLaunchCfg(int rows, int cols, dim3& block, dim3& grid)
    {
        block = dim3(Policy::block_size_x, Policy::block_size_y);
        grid = dim3(divUp(cols, block.x * Policy::patch_size_x), divUp(rows, block.y * Policy::patch_size_y));

        grid.x = ::min(grid.x, block.x);
        grid.y = ::min(grid.y, block.y);
    }

    template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void minMaxLoc(const SrcPtr& src, ResType* minVal, ResType* maxVal, int* minLoc, int* maxLoc, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        dim3 block, grid;
        getLaunchCfg<Policy>(rows, cols, block, grid);

        const int patch_x = divUp(divUp(cols, grid.x), block.x);
        const int patch_y = divUp(divUp(rows, grid.y), block.y);

        minMaxLoc_pass_1<Policy::block_size_x * Policy::block_size_y><<<grid, block, 0, stream>>>(src, minVal, maxVal, minLoc, maxLoc, mask, rows, cols, patch_y, patch_x);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        minMaxLoc_pass_2<Policy::block_size_x * Policy::block_size_y><<<1, Policy::block_size_x * Policy::block_size_y, 0, stream>>>(minVal, maxVal, minLoc, maxLoc, grid.x * grid.y);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
