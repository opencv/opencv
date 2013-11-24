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

#ifndef __OPENCV_CUDEV_GRID_TRANSPOSE_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_TRANSPOSE_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace transpose_detail
{
    template <int TILE_DIM, int BLOCK_DIM_Y, class SrcPtr, typename DstType>
    __global__ void transpose(const SrcPtr src, GlobPtr<DstType> dst, const int rows, const int cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        __shared__ src_type tile[TILE_DIM][TILE_DIM + 1];

        int blockIdx_x, blockIdx_y;

        // do diagonal reordering
        if (gridDim.x == gridDim.y)
        {
            blockIdx_y = blockIdx.x;
            blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
        }
        else
        {
            int bid = blockIdx.x + gridDim.x * blockIdx.y;
            blockIdx_y = bid % gridDim.y;
            blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
        }

        int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
        int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;

        if (xIndex < cols)
        {
            for (int i = 0; i < TILE_DIM; i += BLOCK_DIM_Y)
            {
                if (yIndex + i < rows)
                {
                    tile[threadIdx.y + i][threadIdx.x] = src(yIndex + i, xIndex);
                }
            }
        }

        __syncthreads();

        xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
        yIndex = blockIdx_x * TILE_DIM + threadIdx.y;

        if (xIndex < rows)
        {
            for (int i = 0; i < TILE_DIM; i += BLOCK_DIM_Y)
            {
                if (yIndex + i < cols)
                {
                    dst(yIndex + i, xIndex) = saturate_cast<DstType>(tile[threadIdx.x][threadIdx.y + i]);
                }
            }
        }
    }

    template <class Policy, class SrcPtr, typename DstType>
    __host__ void transpose(const SrcPtr& src, const GlobPtr<DstType>& dst, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::tile_dim, Policy::block_dim_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        transpose<Policy::tile_dim, Policy::block_dim_y><<<grid, block, 0, stream>>>(src, dst, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
