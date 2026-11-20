// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

const int TILE = 32, TILE_ROWS = 8;

template<int N>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    // +1 column so columns of the tile land in different LDS banks
    __shared__ Bytes<N> tile[TILE][TILE + 1];

    // gridDim.y is capped, so one block walks several tile rows. The condition depends only
    // on blockIdx.y and gridDim.y, so every thread of the block runs the same iterations and
    // reaches both __syncthreads().
    for (int ty = blockIdx.y; (size_t)ty * TILE < (size_t)rows; ty += gridDim.y)
    {
        const int xs = blockIdx.x * TILE + threadIdx.x;
        const int ys = ty * TILE + threadIdx.y;

        __syncthreads();    // the previous iteration has finished reading the tile
        for (int j = 0; j < TILE; j += TILE_ROWS)
            if (xs < cols && ys + j < rows)
                hipCopyN<N>(tile[threadIdx.y + j][threadIdx.x].b,
                            src + (size_t)(ys + j) * sstep + (size_t)xs * N);
        __syncthreads();

        const int xd = ty * TILE + threadIdx.x;
        const int yd = blockIdx.x * TILE + threadIdx.y;
        for (int j = 0; j < TILE; j += TILE_ROWS)
            if (xd < rows && yd + j < cols)
                hipCopyN<N>(dst + (size_t)(yd + j) * dstep + (size_t)xd * N,
                            tile[threadIdx.x][threadIdx.y + j].b);
    }
}

template<int N>
hipError_t transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, hipStream_t s)
{
    const unsigned tilesY = (unsigned)((rows + TILE - 1) / TILE);
    const dim3 g((unsigned)((cols + TILE - 1) / TILE), tilesY < MAX_GRID_Y ? tilesY : MAX_GRID_Y);
    hipLaunchKernelGGL(transposeKernel<N>, g, dim3(TILE, TILE_ROWS), 0, s,
                       src, sstep, dst, dstep, rows, cols);
    return hipGetLastError();
}

} // namespace

hipError_t transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, hipStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return hipSuccess;
    CV_ACCEL_HIP_BY_SIZE(transposeN, src, sstep, dst, dstep, rows, cols, s)
}

}}}} // namespace cv::accel::hip_impl::kernels
