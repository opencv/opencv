// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: kernels of the CUDA backend (core operations). Elements are moved as opaque
// groups of bytes, so one kernel serves every type of the same element size.

#include "accel_kernels.hpp"

namespace cv { namespace accel { namespace impl { namespace cuda_kernels {

namespace {

template<int N> struct Bytes { unsigned char b[N]; };

const int BLOCK_X = 32, BLOCK_Y = 8;

// gridDim.y is at most 65535: taller images are covered by a loop over y, so the launch never
// fails on a size that QUERY accepted
const unsigned MAX_GRID_Y = 65535;

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
        *(Bytes<N>*)(dst + (size_t)y * dstep + (size_t)x * N) =
            *(const Bytes<N>*)(src + (size_t)sy * sstep + (size_t)sx * N);
    }
}

template<int N>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    // rows x cols in, cols x rows out
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
        *(Bytes<N>*)(dst + (size_t)x * dstep + (size_t)y * N) =
            *(const Bytes<N>*)(src + (size_t)y * sstep + (size_t)x * N);
}

static inline dim3 grid(int rows, int cols)
{
    const unsigned gy = (unsigned)((rows + BLOCK_Y - 1) / BLOCK_Y);
    return dim3((unsigned)((cols + BLOCK_X - 1) / BLOCK_X), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

// cudaLaunchKernel returns the status of this launch, where cudaGetLastError() after <<<>>>
// would return the oldest error left on the thread by any earlier call

template<int N>
cudaError_t flipN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                  int rows, int cols, int mode, cudaStream_t s)
{
    // OpenCV's codes: 0 around the x axis (rows), > 0 around the y axis (columns), < 0 both
    bool flipRows = mode <= 0, flipCols = mode != 0;
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols, &flipRows, &flipCols };
    return cudaLaunchKernel((const void*)flipKernel<N>, grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), args, 0, s);
}

template<int N>
cudaError_t transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                       int rows, int cols, cudaStream_t s)
{
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols };
    return cudaLaunchKernel((const void*)transposeKernel<N>, grid(rows, cols), dim3(BLOCK_X, BLOCK_Y), args, 0, s);
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

#define CV_ACCEL_CUDA_BY_SIZE(fn, ...) \
    switch (esz) \
    { \
    case 1: return fn<1>(__VA_ARGS__); \
    case 2: return fn<2>(__VA_ARGS__); \
    case 3: return fn<3>(__VA_ARGS__); \
    case 4: return fn<4>(__VA_ARGS__); \
    case 6: return fn<6>(__VA_ARGS__); \
    case 8: return fn<8>(__VA_ARGS__); \
    case 12: return fn<12>(__VA_ARGS__); \
    case 16: return fn<16>(__VA_ARGS__); \
    case 24: return fn<24>(__VA_ARGS__); \
    case 32: return fn<32>(__VA_ARGS__); \
    default: return cudaErrorInvalidValue; \
    }

cudaError_t flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                 int rows, int cols, int esz, int mode, cudaStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return cudaSuccess;
    CV_ACCEL_CUDA_BY_SIZE(flipN, src, sstep, dst, dstep, rows, cols, mode, s)
}

cudaError_t transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, int esz, cudaStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return cudaSuccess;
    CV_ACCEL_CUDA_BY_SIZE(transposeN, src, sstep, dst, dstep, rows, cols, s)
}

#undef CV_ACCEL_CUDA_BY_SIZE

}}}} // namespace cv::accel::impl::cuda_kernels
