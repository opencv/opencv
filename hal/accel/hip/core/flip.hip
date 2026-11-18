// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Elements are moved as opaque groups of bytes, so one kernel serves every type of the
// same element size.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

template<int N>
__global__ void flipKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                           int rows, int cols, bool flipRows, bool flipCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    const int sx = flipCols ? cols - 1 - x : x;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        const unsigned sy = flipRows ? (unsigned)rows - 1 - y : y;
        hipCopyN<N>(dst + (size_t)y * dstep + (size_t)x * N,
                    src + (size_t)sy * sstep + (size_t)sx * N);
    }
}

template<int N>
hipError_t flipN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                 int rows, int cols, int mode, hipStream_t s)
{
    // OpenCV's codes: 0 around the x axis (rows), > 0 around the y axis (columns), < 0 both
    const bool flipRows = mode <= 0, flipCols = mode != 0;
    hipLaunchKernelGGL(flipKernel<N>, grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), 0, s,
                       src, sstep, dst, dstep, rows, cols, flipRows, flipCols);
    return hipGetLastError();
}

} // namespace

bool supportedElementSize(int esz)
{
    switch (esz)
    {
    case 1: case 2: case 3: case 4: case 6: case 8: case 12: case 16: case 24: case 32:
        return true;
    default:
        return false;
    }
}

hipError_t flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                int rows, int cols, int esz, int mode, hipStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return hipSuccess;
    CV_ACCEL_HIP_BY_SIZE(flipN, src, sstep, dst, dstep, rows, cols, mode, s)
}

}}}} // namespace cv::accel::hip_impl::kernels
