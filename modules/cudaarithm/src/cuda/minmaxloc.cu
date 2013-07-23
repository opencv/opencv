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

namespace minMaxLoc
{
    // To avoid shared bank conflicts we convert each value into value of
    // appropriate type (32 bits minimum)
    template <typename T> struct MinMaxTypeTraits;
    template <> struct MinMaxTypeTraits<unsigned char> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<signed char> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<unsigned short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<short> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<int> { typedef int best_type; };
    template <> struct MinMaxTypeTraits<float> { typedef float best_type; };
    template <> struct MinMaxTypeTraits<double> { typedef double best_type; };

    template <int BLOCK_SIZE, typename T, class Mask>
    __global__ void kernel_pass_1(const PtrStepSz<T> src, const Mask mask, T* minval, T* maxval, unsigned int* minloc, unsigned int* maxloc, const int twidth, const int theight)
    {
        typedef typename MinMaxTypeTraits<T>::best_type work_type;

        __shared__ work_type sminval[BLOCK_SIZE];
        __shared__ work_type smaxval[BLOCK_SIZE];
        __shared__ unsigned int sminloc[BLOCK_SIZE];
        __shared__ unsigned int smaxloc[BLOCK_SIZE];

        const int x0 = blockIdx.x * blockDim.x * twidth + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * theight + threadIdx.y;

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;

        work_type mymin = numeric_limits<work_type>::max();
        work_type mymax = -numeric_limits<work_type>::max();
        unsigned int myminloc = 0;
        unsigned int mymaxloc = 0;

        for (int i = 0, y = y0; i < theight && y < src.rows; ++i, y += blockDim.y)
        {
            const T* ptr = src.ptr(y);

            for (int j = 0, x = x0; j < twidth && x < src.cols; ++j, x += blockDim.x)
            {
                if (mask(y, x))
                {
                    const work_type srcVal = ptr[x];

                    if (srcVal < mymin)
                    {
                        mymin = srcVal;
                        myminloc = y * src.cols + x;
                    }

                    if (srcVal > mymax)
                    {
                        mymax = srcVal;
                        mymaxloc = y * src.cols + x;
                    }
                }
            }
        }

        reduceKeyVal<BLOCK_SIZE>(smem_tuple(sminval, smaxval), thrust::tie(mymin, mymax),
                                 smem_tuple(sminloc, smaxloc), thrust::tie(myminloc, mymaxloc),
                                 tid,
                                 thrust::make_tuple(less<work_type>(), greater<work_type>()));

        if (tid == 0)
        {
            minval[bid] = (T) mymin;
            maxval[bid] = (T) mymax;
            minloc[bid] = myminloc;
            maxloc[bid] = mymaxloc;
        }
    }
    template <int BLOCK_SIZE, typename T>
    __global__ void kernel_pass_2(T* minval, T* maxval, unsigned int* minloc, unsigned int* maxloc, int count)
    {
        typedef typename MinMaxTypeTraits<T>::best_type work_type;

        __shared__ work_type sminval[BLOCK_SIZE];
        __shared__ work_type smaxval[BLOCK_SIZE];
        __shared__ unsigned int sminloc[BLOCK_SIZE];
        __shared__ unsigned int smaxloc[BLOCK_SIZE];

        unsigned int idx = ::min(threadIdx.x, count - 1);

        work_type mymin = minval[idx];
        work_type mymax = maxval[idx];
        unsigned int myminloc = minloc[idx];
        unsigned int mymaxloc = maxloc[idx];

        reduceKeyVal<BLOCK_SIZE>(smem_tuple(sminval, smaxval), thrust::tie(mymin, mymax),
                                 smem_tuple(sminloc, smaxloc), thrust::tie(myminloc, mymaxloc),
                                 threadIdx.x,
                                 thrust::make_tuple(less<work_type>(), greater<work_type>()));

        if (threadIdx.x == 0)
        {
            minval[0] = (T) mymin;
            maxval[0] = (T) mymax;
            minloc[0] = myminloc;
            maxloc[0] = mymaxloc;
        }
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

    void getBufSize(int cols, int rows, size_t elem_size, int& b1cols, int& b1rows, int& b2cols, int& b2rows)
    {
        dim3 block, grid;
        getLaunchCfg(cols, rows, block, grid);

        // For values
        b1cols = (int)(grid.x * grid.y * elem_size);
        b1rows = 2;

        // For locations
        b2cols = grid.x * grid.y * sizeof(int);
        b2rows = 2;
    }

    template <typename T>
    void run(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf)
    {
        dim3 block, grid;
        getLaunchCfg(src.cols, src.rows, block, grid);

        const int twidth = divUp(divUp(src.cols, grid.x), block.x);
        const int theight = divUp(divUp(src.rows, grid.y), block.y);

        T* minval_buf = (T*) valbuf.ptr(0);
        T* maxval_buf = (T*) valbuf.ptr(1);
        unsigned int* minloc_buf = locbuf.ptr(0);
        unsigned int* maxloc_buf = locbuf.ptr(1);

        if (mask.data)
            kernel_pass_1<threads_x * threads_y><<<grid, block>>>((PtrStepSz<T>) src, SingleMask(mask), minval_buf, maxval_buf, minloc_buf, maxloc_buf, twidth, theight);
        else
            kernel_pass_1<threads_x * threads_y><<<grid, block>>>((PtrStepSz<T>) src, WithOutMask(), minval_buf, maxval_buf, minloc_buf, maxloc_buf, twidth, theight);

        cudaSafeCall( cudaGetLastError() );

        kernel_pass_2<threads_x * threads_y><<<1, threads_x * threads_y>>>(minval_buf, maxval_buf, minloc_buf, maxloc_buf, grid.x * grid.y);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );

        T minval_, maxval_;
        cudaSafeCall( cudaMemcpy(&minval_, minval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(&maxval_, maxval_buf, sizeof(T), cudaMemcpyDeviceToHost) );
        *minval = minval_;
        *maxval = maxval_;

        unsigned int minloc_, maxloc_;
        cudaSafeCall( cudaMemcpy(&minloc_, minloc_buf, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(&maxloc_, maxloc_buf, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        minloc[1] = minloc_ / src.cols; minloc[0] = minloc_ - minloc[1] * src.cols;
        maxloc[1] = maxloc_ / src.cols; maxloc[0] = maxloc_ - maxloc[1] * src.cols;
    }

    template void run<unsigned char >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<signed char >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<unsigned short>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<short >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<int   >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<float >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<double>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
}

#endif // CUDA_DISABLER
