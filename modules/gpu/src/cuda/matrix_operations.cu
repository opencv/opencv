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

#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include "cuda_shared.hpp"
#include "cuda_runtime.h"

__constant__ __align__(16) float scalar_d[4];

namespace mat_operators
{
    template <typename T, int channels, int count = channels>
    struct unroll
    {
        __device__ static void unroll_set(T * mat, size_t i)
        {
            mat[i] = static_cast<T>(scalar_d[channels - count]);
            unroll<T, channels, count - 1>::unroll_set(mat, i+1);
        }

        __device__ static void unroll_set_with_mask(T * mat, unsigned char mask, size_t i)
        {
            if ( mask != 0 )
                mat[i] = static_cast<T>(scalar_d[channels - count]);

            unroll<T, channels, count - 1>::unroll_set_with_mask(mat, mask, i+1);
        }
    };

    template <typename T, int channels>
    struct unroll<T, channels, 0>
    {
        __device__ static void unroll_set(T * , size_t){}
        __device__ static void unroll_set_with_mask(T * , unsigned char, size_t){}
    };

    template <typename T, int channels>
    __device__ size_t GetIndex(size_t i, int cols, int rows, int step)
    {
        return ((i / static_cast<size_t>(cols))*static_cast<size_t>(step) / static_cast<size_t>(sizeof(T))) +
                (i % static_cast<size_t>(rows))*static_cast<size_t>(channels) ;
    }

    template <typename T, int channels>
    __global__ void kernel_set_to_without_mask(T * mat, int cols, int rows, int step)
    {
        size_t i = (blockIdx.x * blockDim.x + threadIdx.x);
        if (i < cols * rows)
        {
            unroll<T, channels>::unroll_set(mat, GetIndex<T,channels>(i, cols, rows, step));
        }
    }

    template <typename T, int channels>
    __global__ void kernel_set_to_with_mask(T * mat, const unsigned char * mask, int cols, int rows, int step)
    {
        size_t i = (blockIdx.x * blockDim.x + threadIdx.x);
        if (i < cols * rows)
            unroll<T, channels>::unroll_set_with_mask(mat, mask[i], GetIndex<T,channels>(i, cols, rows, step));
    }
}

extern "C" void cv::gpu::impl::set_to_with_mask(const DevMem2D& mat, const double * scalar, const DevMem2D& mask, int elemSize1, int channels)
{
    // download scalar to constant memory
    float data[4];
    data[0] = scalar[0];
    data[1] = scalar[1];
    data[2] = scalar[2];
    data[3] = scalar[3];
    cudaSafeCall( cudaMemcpyToSymbol(scalar_d, &data, sizeof(data)));

    dim3 threadsPerBlock(256,1,1);
    dim3 numBlocks (mat.rows * mat.cols / threadsPerBlock.x + 1, 1, 1);

    if (channels == 1)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  1><<<numBlocks,threadsPerBlock>>>(mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 1><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float ,   1><<<numBlocks,threadsPerBlock>>>((float  *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 2)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  2><<<numBlocks,threadsPerBlock>>>(mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 2><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float ,   2><<<numBlocks,threadsPerBlock>>>((float  *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 3)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  3><<<numBlocks,threadsPerBlock>>>(mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 3><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float ,   3><<<numBlocks,threadsPerBlock>>>((float  *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 4)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  4><<<numBlocks,threadsPerBlock>>>(mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 4><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float ,   4><<<numBlocks,threadsPerBlock>>>((float  *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step);
    }
    cudaSafeCall( cudaThreadSynchronize() );
}

extern "C" void cv::gpu::impl::set_to_without_mask(const DevMem2D& mat, const double * scalar, int elemSize1, int channels)
{
    float data[4];
    data[0] = scalar[0];
    data[1] = scalar[1];
    data[2] = scalar[2];
    data[3] = scalar[3];
    cudaSafeCall( cudaMemcpyToSymbol(scalar_d, &data, sizeof(data)));

    dim3 threadsPerBlock(256, 1, 1);
    dim3 numBlocks (mat.rows * mat.cols / threadsPerBlock.x + 1, 1, 1);

    if (channels == 1)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  1><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 1><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask< float,   1><<<numBlocks,threadsPerBlock>>>(( float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 2)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  2><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 2><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask< float ,   2><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 3)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  3><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 3><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask< float,   3><<<numBlocks,threadsPerBlock>>>(( float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 4)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  4><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 4><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask<float,   4><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }

    cudaSafeCall( cudaThreadSynchronize() );
}
