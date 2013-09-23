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
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/emulation.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace countNonZero
{
    __device__ unsigned int blocks_finished = 0;

    template <int BLOCK_SIZE, typename T>
    __global__ void kernel(const PtrStepSz<T> src, unsigned int* count, const int twidth, const int theight)
    {
        __shared__ unsigned int scount[BLOCK_SIZE];

        const int x0 = blockIdx.x * blockDim.x * twidth + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * theight + threadIdx.y;

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        unsigned int mycount = 0;

        for (int i = 0, y = y0; i < theight && y < src.rows; ++i, y += blockDim.y)
        {
            const T* ptr = src.ptr(y);

            for (int j = 0, x = x0; j < twidth && x < src.cols; ++j, x += blockDim.x)
            {
                const T srcVal = ptr[x];

                mycount += (srcVal != 0);
            }
        }

        device::reduce<BLOCK_SIZE>(scount, mycount, tid, plus<unsigned int>());

    #if __CUDA_ARCH__ >= 200
        if (tid == 0)
            ::atomicAdd(count, mycount);
    #else
        __shared__ bool is_last;
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;

        if (tid == 0)
        {
            count[bid] = mycount;

            __threadfence();

            unsigned int ticket = ::atomicInc(&blocks_finished, gridDim.x * gridDim.y);
            is_last = (ticket == gridDim.x * gridDim.y - 1);
        }

        __syncthreads();

        if (is_last)
        {
            mycount = tid < gridDim.x * gridDim.y ? count[tid] : 0;

            device::reduce<BLOCK_SIZE>(scount, mycount, tid, plus<unsigned int>());

            if (tid == 0)
            {
                count[0] = mycount;

                blocks_finished = 0;
            }
        }
    #endif
    }

    const int threads_x = 32;
    const int threads_y = 8;

    void getLaunchCfg(int cols, int rows, dim3& block, dim3& grid)
    {
        block = dim3(threads_x, threads_y);

        grid = dim3(divUp(cols, block.x * block.y),
                    divUp(rows, block.y * block.x));

        grid.x = ::min(grid.x, block.x);
        grid.y = ::min(grid.y, block.y);
    }

    void getBufSize(int cols, int rows, int& bufcols, int& bufrows)
    {
        dim3 block, grid;
        getLaunchCfg(cols, rows, block, grid);

        bufcols = grid.x * grid.y * sizeof(int);
        bufrows = 1;
    }

    template <typename T>
    int run(const PtrStepSzb src, PtrStep<unsigned int> buf)
    {
        dim3 block, grid;
        getLaunchCfg(src.cols, src.rows, block, grid);

        const int twidth = divUp(divUp(src.cols, grid.x), block.x);
        const int theight = divUp(divUp(src.rows, grid.y), block.y);

        unsigned int* count_buf = buf.ptr(0);

        cudaSafeCall( cudaMemset(count_buf, 0, sizeof(unsigned int)) );

        kernel<threads_x * threads_y><<<grid, block>>>((PtrStepSz<T>) src, count_buf, twidth, theight);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );

        unsigned int count;
        cudaSafeCall(cudaMemcpy(&count, count_buf, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        return count;
    }

    template int run<uchar >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<schar >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<ushort>(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<short >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<int   >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<float >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<double>(const PtrStepSzb src, PtrStep<unsigned int> buf);
}

#endif // CUDA_DISABLER
