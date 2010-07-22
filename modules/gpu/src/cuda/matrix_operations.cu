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
    template<typename T, int channels>
    __global__ void kernel_set_to_without_mask(T * mat, int cols, int rows, int step)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < cols * channels ) && (y < rows))
        {
            size_t idx = y * (step / sizeof(T)) + x;
            mat[idx] = scalar_d[ x % channels ];
        }
    }

    template<typename T, int channels>
    __global__ void kernel_set_to_with_mask(T * mat, const unsigned char * mask, int cols, int rows, int step, int step_mask)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (mask[y * step_mask + x] != 0)
            if ((x < cols * channels ) && (y < rows))
            {
                size_t idx = y * (step / sizeof(T)) + x;
                mat[idx] = scalar_d[ x % channels ];
            }
    }
}

extern "C" void cv::gpu::impl::set_to_without_mask(const DevMem2D& mat, const double * scalar, int elemSize1, int channels)
{
    float data[4];
    data[0] = static_cast<float>(scalar[0]);
    data[1] = static_cast<float>(scalar[1]);
    data[2] = static_cast<float>(scalar[2]);
    data[3] = static_cast<float>(scalar[3]);
    cudaSafeCall( cudaMemcpyToSymbol(scalar_d, &data, sizeof(data)));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

    if (channels == 1)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  1><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 1><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask<float,          1><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 2)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  2><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 2><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask<float,          2><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 3)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  3><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 3><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask<float,          3><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }
    if (channels == 4)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_without_mask<unsigned char,  4><<<numBlocks,threadsPerBlock>>>(mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_without_mask<unsigned short, 4><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, mat.cols, mat.rows, mat.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_without_mask<float,          4><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr, mat.cols, mat.rows, mat.step);
    }

    cudaSafeCall ( cudaThreadSynchronize() );
}

extern "C" void cv::gpu::impl::set_to_with_mask(const DevMem2D& mat, const double * scalar, const DevMem2D& mask, int elemSize1, int channels)
{
    float data[4];
    data[0] = static_cast<float>(scalar[0]);
    data[1] = static_cast<float>(scalar[1]);
    data[2] = static_cast<float>(scalar[2]);
    data[3] = static_cast<float>(scalar[3]);
    cudaSafeCall( cudaMemcpyToSymbol(scalar_d, &data, sizeof(data)));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks (mat.cols * channels / threadsPerBlock.x + 1, mat.rows / threadsPerBlock.y + 1, 1);

    if (channels == 1)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  1><<<numBlocks,threadsPerBlock>>>(mat.ptr,                   (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 1><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float,          1><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr,          (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
    }
    if (channels == 2)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  2><<<numBlocks,threadsPerBlock>>>(mat.ptr,                   (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 2><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float,          2><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr,          (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
    }
    if (channels == 3)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  3><<<numBlocks,threadsPerBlock>>>(mat.ptr,                   (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 3><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float,          3><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr,          (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
    }
    if (channels == 4)
    {
        if (elemSize1 == 1) ::mat_operators::kernel_set_to_with_mask<unsigned char,  4><<<numBlocks,threadsPerBlock>>>(mat.ptr,                   (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 2) ::mat_operators::kernel_set_to_with_mask<unsigned short, 4><<<numBlocks,threadsPerBlock>>>((unsigned short *)mat.ptr, (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
        if (elemSize1 == 4) ::mat_operators::kernel_set_to_with_mask<float,          4><<<numBlocks,threadsPerBlock>>>((float *)mat.ptr,          (unsigned char *)mask.ptr, mat.cols, mat.rows, mat.step, mask.step);
    }

    cudaSafeCall ( cudaThreadSynchronize() );
}

