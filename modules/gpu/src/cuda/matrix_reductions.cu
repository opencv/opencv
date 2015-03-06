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

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/reduce.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/type_traits.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace detail
{
    __device__ __forceinline__ int cvAtomicAdd(int* address, int val)
    {
        return ::atomicAdd(address, val);
    }
    __device__ __forceinline__ unsigned int cvAtomicAdd(unsigned int* address, unsigned int val)
    {
        return ::atomicAdd(address, val);
    }
    __device__ __forceinline__ float cvAtomicAdd(float* address, float val)
    {
    #if __CUDA_ARCH__ >= 200
        return ::atomicAdd(address, val);
    #else
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(val + __int_as_float(assumed)));
        } while (assumed != old);
        return __int_as_float(old);
    #endif
    }
    __device__ __forceinline__ double cvAtomicAdd(double* address, double val)
    {
    #if __CUDA_ARCH__ >= 130
        unsigned long long int* address_as_ull = (unsigned long long int*) address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    #else
        (void) address;
        (void) val;
        return 0.0;
    #endif
    }

    __device__ __forceinline__ int cvAtomicMin(int* address, int val)
    {
        return ::atomicMin(address, val);
    }
    __device__ __forceinline__ float cvAtomicMin(float* address, float val)
    {
    #if __CUDA_ARCH__ >= 120
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    #else
        (void) address;
        (void) val;
        return 0.0f;
    #endif
    }
    __device__ __forceinline__ double cvAtomicMin(double* address, double val)
    {
    #if __CUDA_ARCH__ >= 130
        unsigned long long int* address_as_ull = (unsigned long long int*) address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed,
                __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    #else
        (void) address;
        (void) val;
        return 0.0;
    #endif
    }

    __device__ __forceinline__ int cvAtomicMax(int* address, int val)
    {
        return ::atomicMax(address, val);
    }
    __device__ __forceinline__ float cvAtomicMax(float* address, float val)
    {
    #if __CUDA_ARCH__ >= 120
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    #else
        (void) address;
        (void) val;
        return 0.0f;
    #endif
    }
    __device__ __forceinline__ double cvAtomicMax(double* address, double val)
    {
    #if __CUDA_ARCH__ >= 130
        unsigned long long int* address_as_ull = (unsigned long long int*) address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed,
                __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    #else
        (void) address;
        (void) val;
        return 0.0;
    #endif
    }
}

namespace detail
{
    template <int cn> struct Unroll;
    template <> struct Unroll<1>
    {
        template <int BLOCK_SIZE, typename R>
        static __device__ __forceinline__ volatile R* smem_tuple(R* smem)
        {
            return smem;
        }

        template <typename R>
        static __device__ __forceinline__ R& tie(R& val)
        {
            return val;
        }

        template <class Op>
        static __device__ __forceinline__ const Op& op(const Op& op)
        {
            return op;
        }
    };
    template <> struct Unroll<2>
    {
        template <int BLOCK_SIZE, typename R>
        static __device__ __forceinline__ thrust::tuple<volatile R*, volatile R*> smem_tuple(R* smem)
        {
            return cv::gpu::device::smem_tuple(smem, smem + BLOCK_SIZE);
        }

        template <typename R>
        static __device__ __forceinline__ thrust::tuple<typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&> tie(R& val)
        {
            return thrust::tie(val.x, val.y);
        }

        template <class Op>
        static __device__ __forceinline__ const thrust::tuple<Op, Op> op(const Op& op)
        {
            return thrust::make_tuple(op, op);
        }
    };
    template <> struct Unroll<3>
    {
        template <int BLOCK_SIZE, typename R>
        static __device__ __forceinline__ thrust::tuple<volatile R*, volatile R*, volatile R*> smem_tuple(R* smem)
        {
            return cv::gpu::device::smem_tuple(smem, smem + BLOCK_SIZE, smem + 2 * BLOCK_SIZE);
        }

        template <typename R>
        static __device__ __forceinline__ thrust::tuple<typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&> tie(R& val)
        {
            return thrust::tie(val.x, val.y, val.z);
        }

        template <class Op>
        static __device__ __forceinline__ const thrust::tuple<Op, Op, Op> op(const Op& op)
        {
            return thrust::make_tuple(op, op, op);
        }
    };
    template <> struct Unroll<4>
    {
        template <int BLOCK_SIZE, typename R>
        static __device__ __forceinline__ thrust::tuple<volatile R*, volatile R*, volatile R*, volatile R*> smem_tuple(R* smem)
        {
            return cv::gpu::device::smem_tuple(smem, smem + BLOCK_SIZE, smem + 2 * BLOCK_SIZE, smem + 3 * BLOCK_SIZE);
        }

        template <typename R>
        static __device__ __forceinline__ thrust::tuple<typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&> tie(R& val)
        {
            return thrust::tie(val.x, val.y, val.z, val.w);
        }

        template <class Op>
        static __device__ __forceinline__ const thrust::tuple<Op, Op, Op, Op> op(const Op& op)
        {
            return thrust::make_tuple(op, op, op, op);
        }
    };
}

/////////////////////////////////////////////////////////////
// sum

namespace sum
{
    __device__ unsigned int blocks_finished = 0;

    template <typename R, int cn> struct AtomicAdd;
    template <typename R> struct AtomicAdd<R, 1>
    {
        static __device__ void run(R* ptr, R val)
        {
            detail::cvAtomicAdd(ptr, val);
        }
    };
    template <typename R> struct AtomicAdd<R, 2>
    {
        typedef typename TypeVec<R, 2>::vec_type val_type;

        static __device__ void run(R* ptr, val_type val)
        {
            detail::cvAtomicAdd(ptr, val.x);
            detail::cvAtomicAdd(ptr + 1, val.y);
        }
    };
    template <typename R> struct AtomicAdd<R, 3>
    {
        typedef typename TypeVec<R, 3>::vec_type val_type;

        static __device__ void run(R* ptr, val_type val)
        {
            detail::cvAtomicAdd(ptr, val.x);
            detail::cvAtomicAdd(ptr + 1, val.y);
            detail::cvAtomicAdd(ptr + 2, val.z);
        }
    };
    template <typename R> struct AtomicAdd<R, 4>
    {
        typedef typename TypeVec<R, 4>::vec_type val_type;

        static __device__ void run(R* ptr, val_type val)
        {
            detail::cvAtomicAdd(ptr, val.x);
            detail::cvAtomicAdd(ptr + 1, val.y);
            detail::cvAtomicAdd(ptr + 2, val.z);
            detail::cvAtomicAdd(ptr + 3, val.w);
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
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif

    template void run<float, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#ifndef OPENCV_TINY_GPU_MODULE
    template void run<float, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<float, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<float, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void run<double, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<double, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<double, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void run<double, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#endif

    template <typename T, int cn>
    void runAbs(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask)
    {
        typedef typename SumType<T>::R R;
        caller<T, R, cn, abs_func>(src, buf, out, mask);
    }

    template void runAbs<uchar, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif

    template void runAbs<float, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#ifndef OPENCV_TINY_GPU_MODULE
    template void runAbs<float, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<float, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<float, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runAbs<double, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<double, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<double, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runAbs<double, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#endif

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
#ifndef OPENCV_TINY_GPU_MODULE
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
#endif

    template void runSqr<float, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#ifndef OPENCV_TINY_GPU_MODULE
    template void runSqr<float, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<float, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<float, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);

    template void runSqr<double, 1>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<double, 2>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<double, 3>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
    template void runSqr<double, 4>(PtrStepSzb src, void* buf, double* out, PtrStepSzb mask);
#endif
}

/////////////////////////////////////////////////////////////
// minMax

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
                detail::cvAtomicMin(minval, mymin);
                detail::cvAtomicMax(maxval, mymax);
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
#ifndef OPENCV_TINY_GPU_MODULE
    template void run<schar >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<ushort>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<short >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    template void run<int   >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
#endif
    template void run<float >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
#ifndef OPENCV_TINY_GPU_MODULE
    template void run<double>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
#endif
}

/////////////////////////////////////////////////////////////
// minMaxLoc

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
#ifndef OPENCV_TINY_GPU_MODULE
    template void run<signed char >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<unsigned short>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<short >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
#endif
    template void run<int   >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    template void run<float >(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
#ifndef OPENCV_TINY_GPU_MODULE
    template void run<double>(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
#endif
}

/////////////////////////////////////////////////////////////
// countNonZero

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
#ifndef OPENCV_TINY_GPU_MODULE
    template int run<schar >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<ushort>(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<short >(const PtrStepSzb src, PtrStep<unsigned int> buf);
    template int run<int   >(const PtrStepSzb src, PtrStep<unsigned int> buf);
#endif
    template int run<float >(const PtrStepSzb src, PtrStep<unsigned int> buf);
#ifndef OPENCV_TINY_GPU_MODULE
    template int run<double>(const PtrStepSzb src, PtrStep<unsigned int> buf);
#endif
}

//////////////////////////////////////////////////////////////////////////////
// reduce

namespace reduce
{
    struct Sum
    {
        template <typename T>
        __device__ __forceinline__ T startValue() const
        {
            return VecTraits<T>::all(0);
        }

        template <typename T>
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            return a + b;
        }

        template <typename T>
        __device__ __forceinline__ T result(T r, double) const
        {
            return r;
        }

        __device__ __forceinline__ Sum() {}
        __device__ __forceinline__ Sum(const Sum&) {}
    };

    struct Avg
    {
        template <typename T>
        __device__ __forceinline__ T startValue() const
        {
            return VecTraits<T>::all(0);
        }

        template <typename T>
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            return a + b;
        }

        template <typename T>
        __device__ __forceinline__ typename TypeVec<double, VecTraits<T>::cn>::vec_type result(T r, double sz) const
        {
            return r / sz;
        }

        __device__ __forceinline__ Avg() {}
        __device__ __forceinline__ Avg(const Avg&) {}
    };

    struct Min
    {
        template <typename T>
        __device__ __forceinline__ T startValue() const
        {
            return VecTraits<T>::all(numeric_limits<typename VecTraits<T>::elem_type>::max());
        }

        template <typename T>
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            minimum<T> minOp;
            return minOp(a, b);
        }

        template <typename T>
        __device__ __forceinline__ T result(T r, double) const
        {
            return r;
        }

        __device__ __forceinline__ Min() {}
        __device__ __forceinline__ Min(const Min&) {}
    };

    struct Max
    {
        template <typename T>
        __device__ __forceinline__ T startValue() const
        {
            return VecTraits<T>::all(-numeric_limits<typename VecTraits<T>::elem_type>::max());
        }

        template <typename T>
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            maximum<T> maxOp;
            return maxOp(a, b);
        }

        template <typename T>
        __device__ __forceinline__ T result(T r, double) const
        {
            return r;
        }

        __device__ __forceinline__ Max() {}
        __device__ __forceinline__ Max(const Max&) {}
    };

    ///////////////////////////////////////////////////////////

    template <typename T, typename S, typename D, class Op>
    __global__ void rowsKernel(const PtrStepSz<T> src, D* dst, const Op op)
    {
        __shared__ S smem[16 * 16];

        const int x = blockIdx.x * 16 + threadIdx.x;

        S myVal = op.template startValue<S>();

        if (x < src.cols)
        {
            for (int y = threadIdx.y; y < src.rows; y += 16)
            {
                S srcVal = src(y, x);
                myVal = op(myVal, srcVal);
            }
        }

        smem[threadIdx.x * 16 + threadIdx.y] = myVal;

        __syncthreads();

        volatile S* srow = smem + threadIdx.y * 16;

        myVal = srow[threadIdx.x];
        device::reduce<16>(srow, myVal, threadIdx.x, op);

        if (threadIdx.x == 0)
            srow[0] = myVal;

        __syncthreads();

        if (threadIdx.y == 0 && x < src.cols)
            dst[x] = (D) op.result(smem[threadIdx.x * 16], src.rows);
    }

    template <typename T, typename S, typename D, class Op>
    void rowsCaller(PtrStepSz<T> src, D* dst, cudaStream_t stream)
    {
        const dim3 block(16, 16);
        const dim3 grid(divUp(src.cols, block.x));

        Op op;
        rowsKernel<T, S, D, Op><<<grid, block, 0, stream>>>(src, dst, op);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename S, typename D>
    void rows(PtrStepSzb src, void* dst, int op, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSz<T> src, D* dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            rowsCaller<T, S, D, Sum>,
            rowsCaller<T, S, D, Avg>,
            rowsCaller<T, S, D, Max>,
            rowsCaller<T, S, D, Min>
        };

        funcs[op]((PtrStepSz<T>) src, (D*) dst, stream);
    }

#ifdef OPENCV_TINY_GPU_MODULE
    template void rows<unsigned char, int, unsigned char>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned char, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<float, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
#else
    template void rows<unsigned char, int, unsigned char>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned char, int, int>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned char, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned char, double, double>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);

    template void rows<unsigned short, int, unsigned short>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned short, int, int>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned short, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<unsigned short, double, double>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);

    template void rows<short, int, short>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<short, int, int>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<short, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<short, double, double>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);

    template void rows<int, int, int>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<int, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<int, double, double>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);

    template void rows<float, float, float>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
    template void rows<float, double, double>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);

    template void rows<double, double, double>(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
#endif

    ///////////////////////////////////////////////////////////

    template <int BLOCK_SIZE, typename T, typename S, typename D, int cn, class Op>
    __global__ void colsKernel(const PtrStepSz<typename TypeVec<T, cn>::vec_type> src, typename TypeVec<D, cn>::vec_type* dst, const Op op)
    {
        typedef typename TypeVec<T, cn>::vec_type src_type;
        typedef typename TypeVec<S, cn>::vec_type work_type;
        typedef typename TypeVec<D, cn>::vec_type dst_type;

        __shared__ S smem[BLOCK_SIZE * cn];

        const int y = blockIdx.x;

        const src_type* srcRow = src.ptr(y);

        work_type myVal = op.template startValue<work_type>();

        for (int x = threadIdx.x; x < src.cols; x += BLOCK_SIZE)
            myVal = op(myVal, saturate_cast<work_type>(srcRow[x]));

        device::reduce<BLOCK_SIZE>(detail::Unroll<cn>::template smem_tuple<BLOCK_SIZE>(smem), detail::Unroll<cn>::tie(myVal), threadIdx.x, detail::Unroll<cn>::op(op));

        if (threadIdx.x == 0)
            dst[y] = saturate_cast<dst_type>(op.result(myVal, src.cols));
    }

    template <typename T, typename S, typename D, int cn, class Op> void colsCaller(PtrStepSzb src, void* dst, cudaStream_t stream)
    {
        const int BLOCK_SIZE = 256;

        const dim3 block(BLOCK_SIZE);
        const dim3 grid(src.rows);

        Op op;
        colsKernel<BLOCK_SIZE, T, S, D, cn, Op><<<grid, block, 0, stream>>>((PtrStepSz<typename TypeVec<T, cn>::vec_type>) src, (typename TypeVec<D, cn>::vec_type*) dst, op);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );

    }

    template <typename T, typename S, typename D> void cols(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, void* dst, cudaStream_t stream);
        static const func_t funcs[5][4] =
        {
            {0,0,0,0},
            {colsCaller<T, S, D, 1, Sum>, colsCaller<T, S, D, 1, Avg>, colsCaller<T, S, D, 1, Max>, colsCaller<T, S, D, 1, Min>},
            {colsCaller<T, S, D, 2, Sum>, colsCaller<T, S, D, 2, Avg>, colsCaller<T, S, D, 2, Max>, colsCaller<T, S, D, 2, Min>},
            {colsCaller<T, S, D, 3, Sum>, colsCaller<T, S, D, 3, Avg>, colsCaller<T, S, D, 3, Max>, colsCaller<T, S, D, 3, Min>},
            {colsCaller<T, S, D, 4, Sum>, colsCaller<T, S, D, 4, Avg>, colsCaller<T, S, D, 4, Max>, colsCaller<T, S, D, 4, Min>},
        };

        funcs[cn][op](src, dst, stream);
    }

#ifdef OPENCV_TINY_GPU_MODULE
    template void cols<unsigned char, int, unsigned char>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned char, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<float, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
#else
    template void cols<unsigned char, int, unsigned char>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned char, int, int>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned char, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned char, double, double>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);

    template void cols<unsigned short, int, unsigned short>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned short, int, int>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned short, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<unsigned short, double, double>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);

    template void cols<short, int, short>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<short, int, int>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<short, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<short, double, double>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);

    template void cols<int, int, int>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<int, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<int, double, double>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);

    template void cols<float, float, float>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
    template void cols<float, double, double>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);

    template void cols<double, double, double>(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
#endif
}

#endif /* CUDA_DISABLER */
