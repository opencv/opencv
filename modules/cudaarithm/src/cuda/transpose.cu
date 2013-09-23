/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace arithm
{
    const int TRANSPOSE_TILE_DIM   = 16;
    const int TRANSPOSE_BLOCK_ROWS = 16;

    template <typename T>
    __global__ void transposeKernel(const PtrStepSz<T> src, PtrStep<T> dst)
    {
        __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

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

        int xIndex = blockIdx_x * TRANSPOSE_TILE_DIM + threadIdx.x;
        int yIndex = blockIdx_y * TRANSPOSE_TILE_DIM + threadIdx.y;

        if (xIndex < src.cols)
        {
            for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
            {
                if (yIndex + i < src.rows)
                {
                    tile[threadIdx.y + i][threadIdx.x] = src(yIndex + i, xIndex);
                }
            }
        }

        __syncthreads();

        xIndex = blockIdx_y * TRANSPOSE_TILE_DIM + threadIdx.x;
        yIndex = blockIdx_x * TRANSPOSE_TILE_DIM + threadIdx.y;

        if (xIndex < src.rows)
        {
            for (int i = 0; i < TRANSPOSE_TILE_DIM; i += TRANSPOSE_BLOCK_ROWS)
            {
                if (yIndex + i < src.cols)
                {
                    dst(yIndex + i, xIndex) = tile[threadIdx.x][threadIdx.y + i];
                }
            }
        }
    }

    template <typename T> void transpose(PtrStepSz<T> src, PtrStepSz<T> dst, cudaStream_t stream)
    {
        const dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        transposeKernel<<<grid, block, 0, stream>>>(src, dst);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void transpose<int>(PtrStepSz<int> src, PtrStepSz<int> dst, cudaStream_t stream);
    template void transpose<double>(PtrStepSz<double> src, PtrStepSz<double> dst, cudaStream_t stream);
}

#endif // CUDA_DISABLER
