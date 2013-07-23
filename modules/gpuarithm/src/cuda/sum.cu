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
#include "opencv2/core/cuda/utility.hpp"

#include "unroll_detail.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace sum
{
    __device__ unsigned int blocks_finished = 0;

    template <typename R, int cn> struct AtomicAdd;
    template <typename R> struct AtomicAdd<R, 1>
    {
        static __device__ void run(R* ptr, R val)
        {
            Emulation::glob::atomicAdd(ptr, val);
        }
    };
    template <typename R> struct AtomicAdd<R, 2>
    {
        typedef typename TypeVec<R, 2>::vec_type val_type;

        static __device__ void run(R* ptr, val_type val)
        {
            Emulation::glob::atomicAdd(ptr, val.x);
            Emulation::glob::atomicAdd(ptr + 1, val.y);
        }
    };
    template <typename R> struct AtomicAdd<R, 3>
    {
        typedef typename TypeVec<R, 3>::vec_type val_type;

        static __device__ void run(R* ptr, val_type val)
        {
            Emulation::glob::atomicAdd(ptr, val.x);
            Emulation::glob::atomicAdd(ptr + 1, val.y);
            Emulation::glob::atomicAdd(ptr + 2, val.z);
        }
    };
    template <typename R> struct AtomicAdd<R, 4>
    {
        typedef typename TypeVec<R, 4>::vec_type val_type;

        static __device__ void run(R* ptr, val_type val)
        {
            Emulation::glob::atomicAdd(ptr, val.x);
            Emulation::glob::atomicAdd(ptr + 1, val.y);
            Emulation::glob::atomicAdd(ptr + 2, val.z);
            Emulation::glob::atomicAdd(ptr + 3, val.w);
        }
    };

    template <int BLOCK_SIZE, typename R, int cn>
    struct GlobalReduce
    {
        typedef typename TypeVec<R, cn>::vec_type result_type;

        static __device__ void run(result_type& sum, result_type* result, int tid, int bid, R* smem)
        {
        #if __CUDA_ARCH__ >= 200
            if (tid == 0)
                AtomicAdd<R, cn>::run((R*) result, sum);
        #else
            __shared__ bool is_last;

            if (tid == 0)
            {
                result[bid] = sum;

                __threadfence();

                unsigned int ticket = ::atomicAdd(&blocks_finished, 1);
                is_last = (ticket == gridDim.x * gridDim.y - 1);
            }

            __syncthreads();

            if (is_last)
            {
                sum = tid < gridDim.x * gridDim.y ? result[tid] : VecTraits<result_type>::all(0);

                device::reduce<BLOCK_SIZE>(detail::Unroll<cn>::template smem_tuple<BLOCK_SIZE>(smem), detail::Unroll<cn>::tie(sum), tid, detail::Unroll<cn>::op(plus<R>()));

                if (tid == 0)
                {
                    result[0] = sum;
                    blocks_finished = 0;
                }
            }
        #endif
        }
    };

    template <int BLOCK_SIZE, typename src_type, typename result_type, class Mask, class Op>
    __global__ void kernel(const PtrStepSz<src_type> src, result_type* result, const Mask mask, const Op op, const int twidth, const int theight)
    {
        typedef typename VecTraits<src_type>::elem_type T;
        typedef typename VecTraits<result_type>::elem_type R;
        const int cn = VecTraits<src_type>::cn;

        __shared__ R smem[BLOCK_SIZE * cn];

        const int x0 = blockIdx.x * blockDim.x * twidth + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * theight + threadIdx.y;

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;

        result_type sum = VecTraits<result_type>::all(0);

        for (int i = 0, y = y0; i < theight && y < src.rows; ++i, y += blockDim.y)
        {
            const src_type* ptr = src.ptr(y);

            for (int j = 0, x = x0; j < twidth && x < src.cols; ++j, x += blockDim.x)
            {
                if (mask(y, x))
                {
                    const src_type srcVal = ptr[x];
                    sum = sum + op(saturate_cast<result_type>(srcVal));
                }
            }
        }

        device::reduce<BLOCK_SIZE>(detail::Unroll<cn>::template smem_tuple<BLOCK_SIZE>(smem), detail::Unroll<cn>::tie(sum), tid, detail::Unroll<cn>::op(plus<R>()));

        GlobalReduce<BLOCK_SIZE, R, cn>::run(sum, result, tid, bid, smem);
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

    void getBufSize(int cols, int rows, int cn, int& bufcols, int& bufrows)
    {
        dim3 block, grid;
        getLaunchCfg(cols, rows, block, grid);

        bufcols = grid.x * grid.y * sizeof(double) * cn;
        bufrows = 1;
    }

    template <typename T, typename R, int cn, template <typename> class Op>
    void caller(PtrStepSzb src_, void* buf_, double* out, PtrStepSzb mask)
    {
        typedef typename TypeVec<T, cn>::vec_type src_type;
        typedef typename TypeVec<R, cn>::vec_type result_type;

        PtrStepSz<src_type> src(src_);
        result_type* buf = (result_type*) buf_;

        dim3 block, grid;
        getLaunchCfg(src.cols, src.rows, block, grid);

        const int twidth = divUp(divUp(src.cols, grid.x), block.x);
        const int theight = divUp(divUp(src.rows, grid.y), block.y);

        Op<result_type> op;

        if (mask.data)
            kernel<threads_x * threads_y><<<grid, block>>>(src, buf, SingleMask(mask), op, twidth, theight);
        else
            kernel<threads_x * threads_y><<<grid, block>>>(src, buf, WithOutMask(), op, twidth, theight);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );

        R result[4] = {0, 0, 0, 0};
        cudaSafeCall( cudaMemcpy(&result, buf, sizeof(result_type), cudaMemcpyDeviceToHost) );

        out[0] = result[0];
        out[1] = result[1];
        out[2] = result[2];
        out[3] = result[3];
    }

    template <typename T> struct SumType;
    template <> struct SumType<uchar> { typedef unsigned int R; };
    template <> struct SumType<schar> { typedef int R; };
    template <> struct SumType<ushort> { typedef unsigned int R; };
    template <> struct SumType<short> { typedef int R; };
    template <> struct SumType<int> { typedef int R; };
    template <> struct SumType<float> { typedef float R; };
    template <> struct SumType<double> { typedef double R; };

    template <typename T, int cn>
    void run(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask)
    {
        typedef typename SumType<T>::R R;
        caller<T, R, cn, identity>(src, buf, out, mask);
    }

    template void run<uchar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<uchar, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<uchar, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<uchar, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<schar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<schar, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<schar, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<schar, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<ushort, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<ushort, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<ushort, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<ushort, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<short, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<short, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<short, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<short, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<int, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<int, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<int, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<int, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<float, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<float, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<float, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<float, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<double, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<double, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<double, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<double, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template <typename T, int cn>
    void runAbs(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask)
    {
        typedef typename SumType<T>::R R;
        caller<T, R, cn, abs_func>(src, buf, out, mask);
    }

    template void runAbs<uchar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<uchar, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<uchar, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<uchar, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<schar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<schar, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<schar, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<schar, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<ushort, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<ushort, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<ushort, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<ushort, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<short, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<short, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<short, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<short, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<int, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<int, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<int, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<int, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<float, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<float, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<float, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<float, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<double, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<double, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<double, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<double, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template <typename T> struct Sqr : unary_function<T, T>
    {
        __device__ __forceinline__ T operator ()(T x) const
        {
            return x * x;
        }
    };

    template <typename T, int cn>
    void runSqr(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask)
    {
        caller<T, double, cn, Sqr>(src, buf, out, mask);
    }

    template void runSqr<uchar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<uchar, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<uchar, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<uchar, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<schar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<schar, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<schar, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<schar, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<ushort, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<ushort, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<ushort, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<ushort, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<short, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<short, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<short, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<short, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<int, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<int, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<int, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<int, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<float, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<float, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<float, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<float, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<double, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<double, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<double, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<double, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
}

#endif // CUDA_DISABLER
