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
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/utility.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace minMax
{
    __device__ unsigned int blocks_finished = 0;

    // To avoid shared bank conflicts we convert each value into value of
    // appropriate type (32 bits minimum)
    template <typename T> struct MinMaxTypeTraits;
    template <> struct MinMaxTypeTraits<uchar> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<schar> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<ushort> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<int> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<float> { typedef float best_type; };
    template <> struct MinMaxTypeTraits<double> { typedef double best_type; };

    template <int BLOCK_SIZE, typename R>
    struct GlobalReduce
    {
        static __device__ void run(R& mymin, R& mymax, R* minval, R* maxval, int tid, int bid, R* sminval, R* smaxval)
        {
        #if __CUDA_ARCH__ >= 200
            if (tid == 0)
            {
                Emulation::glob::atomicMin(minval, mymin);
                Emulation::glob::atomicMax(maxval, mymax);
            }
        #else
            __shared__ bool is_last;

            if (tid == 0)
            {
                minval[bid] = mymin;
                maxval[bid] = mymax;

                __threadfence();

                unsigned int ticket = ::atomicAdd(&blocks_finished, 1);
                is_last = (ticket == gridDim.x * gridDim.y - 1);
            }

            __syncthreads();

            if (is_last)
            {
                int idx = ::min(tid, gridDim.x * gridDim.y - 1);

                mymin = minval[idx];
                mymax = maxval[idx];

                const minimum<R> minOp;
                const maximum<R> maxOp;
                device::reduce<BLOCK_SIZE>(smem_tuple(sminval, smaxval), thrust::tie(mymin, mymax), tid, thrust::make_tuple(minOp, maxOp));

                if (tid == 0)
                {
                    minval[0] = mymin;
                    maxval[0] = mymax;

                    blocks_finished = 0;
                }
            }
        #endif
        }
    };

    template <int BLOCK_SIZE, typename T, typename R, class Mask>
    __global__ void kernel(const PtrStepSz<T> src, const Mask mask, R* minval, R* maxval, const int twidth, const int theight)
    {
        __shared__ R sminval[BLOCK_SIZE];
        __shared__ R smaxval[BLOCK_SIZE];

        const int x0 = blockIdx.x * blockDim.x * twidth + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * theight + threadIdx.y;

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;

        R mymin = numeric_limits<R>::max();
        R mymax = -numeric_limits<R>::max();

        const minimum<R> minOp;
        const maximum<R> maxOp;

        for (int i = 0, y = y0; i < theight && y < src.rows; ++i, y += blockDim.y)
        {
            const T* ptr = src.ptr(y);

            for (int j = 0, x = x0; j < twidth && x < src.cols; ++j, x += blockDim.x)
            {
                if (mask(y, x))
                {
                    const R srcVal = ptr[x];

                    mymin = minOp(mymin, srcVal);
                    mymax = maxOp(mymax, srcVal);
                }
            }
        }

        device::reduce<BLOCK_SIZE>(smem_tuple(sminval, smaxval), thrust::tie(mymin, mymax), tid, thrust::make_tuple(minOp, maxOp));

        GlobalReduce<BLOCK_SIZE, R>::run(mymin, mymax, minval, maxval, tid, bid, sminval, smaxval);
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

        bufcols = grid.x * grid.y * sizeof(double);
        bufrows = 2;
    }

    __global__ void setDefaultKernel(int* minval_buf, int* maxval_buf)
    {
        *minval_buf = numeric_limits<int>::max();
        *maxval_buf = numeric_limits<int>::min();
    }
    __global__ void setDefaultKernel(float* minval_buf, float* maxval_buf)
    {
        *minval_buf = numeric_limits<float>::max();
        *maxval_buf = -numeric_limits<float>::max();
    }
    __global__ void setDefaultKernel(double* minval_buf, double* maxval_buf)
    {
        *minval_buf = numeric_limits<double>::max();
        *maxval_buf = -numeric_limits<double>::max();
    }

    template <typename R>
    void setDefault(R* minval_buf, R* maxval_buf)
    {
        setDefaultKernel<<<1, 1>>>(minval_buf, maxval_buf);
    }

    template <typename T>
    void run(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf)
    {
        typedef typename MinMaxTypeTraits<T>::best_type R;

        dim3 block, grid;
        getLaunchCfg(src.cols, src.rows, block, grid);

        const int twidth = divUp(divUp(src.cols, grid.x), block.x);
        const int theight = divUp(divUp(src.rows, grid.y), block.y);

        R* minval_buf = (R*) buf.ptr(0);
        R* maxval_buf = (R*) buf.ptr(1);

        setDefault(minval_buf, maxval_buf);

        if (mask.data)
            kernel<threads_x * threads_y><<<grid, block>>>((PtrStepSz<T>) src, SingleMask(mask), minval_buf, maxval_buf, twidth, theight);
        else
            kernel<threads_x * threads_y><<<grid, block>>>((PtrStepSz<T>) src, WithOutMask(), minval_buf, maxval_buf, twidth, theight);

        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );

        R minval_, maxval_;
        cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(R), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(R), cudaMemcpyDeviceToHost) );
        *minval = minval_;
        *maxval = maxval_;
    }

    template void run<uchar >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<schar >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<ushort>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<short >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<int   >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<float >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<double>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
}

#endif // CUDA_DISABLER
