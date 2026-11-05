// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: kernels of the GPU backends (core operations), compiled once per runtime, for
// CUDA by src/cuda/accel_kernels.cu. Elements are moved as opaque
// groups of bytes, so one kernel serves every type of the same element size.

#include "gpu_kernels.hpp"

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

//! the largest power of two dividing @p n, up to 16
constexpr int elemAlign(int n)
{
    return n % 16 == 0 ? 16 : n % 8 == 0 ? 8 : n % 4 == 0 ? 4 : n % 2 == 0 ? 2 : 1;
}
// one element of N bytes, aligned as far as N allows, so a copy is one wide load and store
template<int N> struct alignas(elemAlign(N)) Elem { unsigned char b[N]; };
// the same bytes at any alignment, for rows whose start or step is not a multiple of that
template<int N> struct Bytes { unsigned char b[N]; };

const int BLOCK_X = 32, BLOCK_Y = 8;
const int TILE = 32;

// gridDim.y is at most 65535: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

inline bool aligned(const void* src, size_t sstep, const void* dst, size_t dstep, size_t a)
{
    return (((size_t)src | sstep | (size_t)dst | dstep) & (a - 1)) == 0;
}

template<typename T>
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
        ((T*)(dst + (size_t)y * dstep))[x] = ((const T*)(src + (size_t)sy * sstep))[sx];
    }
}

// a TILE x TILE block goes through shared memory, so both the reads and the writes walk along rows
template<typename T>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    __shared__ T tile[TILE][TILE + 1];
    const int x0 = blockIdx.x * TILE;
    for (int y0 = blockIdx.y * TILE; y0 < rows; y0 += gridDim.y * TILE)
    {
        const int x = x0 + threadIdx.x;
        for (int j = threadIdx.y; j < TILE; j += BLOCK_Y)
            if (x < cols && y0 + j < rows)
                tile[j][threadIdx.x] = ((const T*)(src + (size_t)(y0 + j) * sstep))[x];
        __syncthreads();
        const int ox = y0 + threadIdx.x;
        for (int j = threadIdx.y; j < TILE; j += BLOCK_Y)
            if (ox < rows && x0 + j < cols)
                ((T*)(dst + (size_t)(x0 + j) * dstep))[ox] = tile[threadIdx.x][j];
        __syncthreads();
    }
}

static inline dim3 grid(int rows, int cols, int bx, int by)
{
    const unsigned gy = (unsigned)((rows + by - 1) / by);
    return dim3((unsigned)((cols + bx - 1) / bx), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

// LaunchKernel returns the status of this launch, where GetLastError() after <<<>>> would
// return the oldest error left on the thread by any earlier call

template<int N>
CV_GPU(Error_t) flipN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, int mode, CV_GPU(Stream_t) s)
{
    // OpenCV's codes: 0 around the x axis (rows), > 0 around the y axis (columns), < 0 both
    bool flipRows = mode <= 0, flipCols = mode != 0;
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols, &flipRows, &flipCols };
    const bool wide = aligned(src, sstep, dst, dstep, alignof(Elem<N>));
    const void* k = wide ? (const void*)flipKernel<Elem<N> > : (const void*)flipKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y),
                                args, 0, s);
}

template<int N>
CV_GPU(Error_t) transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                           int rows, int cols, CV_GPU(Stream_t) s)
{
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols };
    const bool wide = aligned(src, sstep, dst, dstep, alignof(Elem<N>));
    const void* k = wide ? (const void*)transposeKernel<Elem<N> >
                         : (const void*)transposeKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, TILE, TILE), dim3(TILE, BLOCK_Y), args, 0, s);
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

#define CV_ACCEL_GPU_BY_SIZE(fn, ...) \
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
    default: return CV_GPU(ErrorInvalidValue); \
    }

CV_GPU(Error_t) flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, int mode, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(flipN, src, sstep, dst, dstep, rows, cols, mode, s)
}

CV_GPU(Error_t) transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                          int rows, int cols, int esz, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(transposeN, src, sstep, dst, dstep, rows, cols, s)
}

#undef CV_ACCEL_GPU_BY_SIZE

}}}} // namespace cv::accel::CV_GPU_NS::kernels
