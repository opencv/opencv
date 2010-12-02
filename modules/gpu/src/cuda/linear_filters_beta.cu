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

#include "opencv2/gpu/devmem2d.hpp"
#include "safe_call.hpp"
#include "cuda_shared.hpp"
#include "border_interpolate.hpp"

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define MAX_KERNEL_SIZE 16

using namespace cv::gpu;


namespace cv { namespace gpu { namespace linear_filters {


// Global linear kernel data storage
__constant__ float ckernel[MAX_KERNEL_SIZE];


void loadKernel(const float* kernel, int ksize) 
{
    cudaSafeCall(cudaMemcpyToSymbol(ckernel, kernel, ksize * sizeof(float)));
}


template <typename T, typename B, int ksize>
__global__ void rowFilterKernel(const DevMem2D_<T> src, PtrStepf dst, 
                                int anchor, B border)
{
    __shared__ float smem[BLOCK_DIM_X * BLOCK_DIM_Y * 3];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float* srow = smem + threadIdx.y * blockDim.x * 3;

    if (y < src.rows)
    {
        const T* src_row = src.ptr(y);

        srow[threadIdx.x + blockDim.x] = border.at_high(x, src_row);

        srow[threadIdx.x] = border.at_low(x - blockDim.x, src_row);

        srow[threadIdx.x + (blockDim.x << 1)] = border.at_high(x + blockDim.x, src_row);

        __syncthreads();

        if (x < src.cols)
        {
            srow += threadIdx.x + blockDim.x - anchor;

            float sum = 0.f;
            for (int i = 0; i < ksize; ++i)
                sum += srow[i] * ckernel[i];

            dst.ptr(y)[x] = sum;
        }
    }
}


template <typename T, typename B, int ksize>
void rowFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor)
{
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(divUp(src.cols, threads.x), divUp(src.rows, threads.y));

    B border(src.cols);

    if (!border.is_range_safe(-BLOCK_DIM_X, (grid.x + 1) * BLOCK_DIM_X - 1))
        cv::gpu::error("rowFilterCaller: can't use specified border extrapolation, image is too small, "
                       "try bigger image or another border extrapolation mode", __FILE__, __LINE__);

    rowFilterKernel<T, B, ksize><<<grid, threads>>>(src, dst, anchor, border);
    cudaSafeCall(cudaThreadSynchronize());
}


template <typename T, typename B>
void rowFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor, 
                     const float* kernel, int ksize)
{
    typedef void (*Caller)(const DevMem2D_<T>, PtrStepf, int);

    static const Caller callers[] = 
    { 
        0, rowFilterCaller<T, B, 1>, 
        rowFilterCaller<T, B, 2>, rowFilterCaller<T, B, 3>, 
        rowFilterCaller<T, B, 4>, rowFilterCaller<T, B, 5>, 
        rowFilterCaller<T, B, 6>, rowFilterCaller<T, B, 7>, 
        rowFilterCaller<T, B, 8>, rowFilterCaller<T, B, 9>, 
        rowFilterCaller<T, B, 10>, rowFilterCaller<T, B, 11>, 
        rowFilterCaller<T, B, 12>, rowFilterCaller<T, B, 13>, 
        rowFilterCaller<T, B, 14>, rowFilterCaller<T, B, 15> 
    };

    loadKernel(kernel, ksize);
    callers[ksize](src, dst, anchor);
}


template <typename T>
void rowFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor, 
                     const float* kernel, int ksize, int brd_interp)
{
    typedef void (*Caller)(const DevMem2D_<T>, PtrStepf, int, const float*, int);

    static const Caller callers[] = 
    { 
        rowFilterCaller<T, BrdRowReflect101<T> >
    };

    callers[brd_interp](src, dst, anchor, kernel, ksize);
}


template void rowFilterCaller<unsigned char>(const DevMem2D_<unsigned char>, PtrStepf, int, const float*, int, int);
template void rowFilterCaller<float>(const DevMem2D_<float>, PtrStepf, int, const float*, int, int);


template <typename T, typename B, int ksize>
__global__ void colFilterKernel(const DevMem2D_<T> src, PtrStepf dst, int anchor, B border)
{
    __shared__ float smem[BLOCK_DIM_X * BLOCK_DIM_Y * 3];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int smem_step = blockDim.x;

    float* scol = smem + threadIdx.x;

    if (x < src.cols)
    {
        const T* src_col = src.data + x;

        scol[(threadIdx.y + blockDim.y) * smem_step] = border.at_high(y, src_col);

        scol[threadIdx.y * smem_step] = border.at_low(y - blockDim.y, src_col);

        scol[(threadIdx.y + (blockDim.y << 1)) * smem_step] = border.at_high(y + blockDim.y, src_col);

        __syncthreads();

        if (y < src.rows)
        {
            scol += (threadIdx.y + blockDim.y - anchor)* smem_step;

            float sum = 0.f;
            for(int i = 0; i < ksize; ++i)
                sum += scol[i * smem_step] * ckernel[i];

            dst.ptr(y)[x] = sum;
        }
    }
}


template <typename T, typename B, int ksize>
void colFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor)
{
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid(divUp(src.cols, threads.x), divUp(src.rows, threads.y));

    B border(src.rows, src.step / src.elem_size);

    if (src.step - border.step * src.elem_size != 0)
        cv::gpu::error("colFilterCaller: src step must be multiple of its element size", 
                       __FILE__, __LINE__);

    if (!border.is_range_safe(-BLOCK_DIM_Y, (grid.y + 1) * BLOCK_DIM_Y - 1))
        cv::gpu::error("colFilterCaller: can't use specified border extrapolation, image is too small, "
                       "try bigger image or another border extrapolation mode", __FILE__, __LINE__);

    colFilterKernel<T, B, ksize><<<grid, threads>>>(src, dst, anchor, border);
    cudaSafeCall(cudaThreadSynchronize());
}


template <typename T, typename B>
void colFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor, 
                     const float* kernel, int ksize)
{
    typedef void (*Caller)(const DevMem2D_<T>, PtrStepf, int);

    static const Caller callers[] = 
    { 
        0, colFilterCaller<T, B, 1>, 
        colFilterCaller<T, B, 2>, colFilterCaller<T, B, 3>, 
        colFilterCaller<T, B, 4>, colFilterCaller<T, B, 5>, 
        colFilterCaller<T, B, 6>, colFilterCaller<T, B, 7>, 
        colFilterCaller<T, B, 8>, colFilterCaller<T, B, 9>, 
        colFilterCaller<T, B, 10>, colFilterCaller<T, B, 11>, 
        colFilterCaller<T, B, 12>, colFilterCaller<T, B, 13>, 
        colFilterCaller<T, B, 14>, colFilterCaller<T, B, 15> 
    };

    loadKernel(kernel, ksize);
    callers[ksize](src, dst, anchor);
}


template <typename T>
void colFilterCaller(const DevMem2D_<T> src, PtrStepf dst, int anchor, 
                     const float* kernel, int ksize, int brd_interp)
{
    typedef void (*Caller)(const DevMem2D_<T>, PtrStepf, int, const float*, int);

    static const Caller callers[] = 
    { 
        colFilterCaller<T, BrdColReflect101<T> >
    };

    callers[brd_interp](src, dst, anchor, kernel, ksize);
}


template void colFilterCaller<unsigned char>(const DevMem2D_<unsigned char>, PtrStepf, int, const float*, int, int);
template void colFilterCaller<float>(const DevMem2D_<float>, PtrStepf, int, const float*, int, int);

}}} 