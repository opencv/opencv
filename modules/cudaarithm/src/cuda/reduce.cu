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
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/limits.hpp"

#include "unroll_detail.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

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
        __device__ __forceinline__ T result(T r, int) const
        {
            return r;
        }

        __host__ __device__ __forceinline__ Sum() {}
        __host__ __device__ __forceinline__ Sum(const Sum&) {}
    };

    template <typename T> struct OutputType
    {
        typedef float type;
    };
    template <> struct OutputType<double>
    {
        typedef double type;
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
        __device__ __forceinline__ typename TypeVec<typename OutputType<typename VecTraits<T>::elem_type>::type, VecTraits<T>::cn>::vec_type result(T r, float sz) const
        {
            return r / sz;
        }

        __host__ __device__ __forceinline__ Avg() {}
        __host__ __device__ __forceinline__ Avg(const Avg&) {}
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
        __device__ __forceinline__ T result(T r, int) const
        {
            return r;
        }

        __host__ __device__ __forceinline__ Min() {}
        __host__ __device__ __forceinline__ Min(const Min&) {}
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
        __device__ __forceinline__ T result(T r, int) const
        {
            return r;
        }

        __host__ __device__ __forceinline__ Max() {}
        __host__ __device__ __forceinline__ Max(const Max&) {}
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
}

#endif /* CUDA_DISABLER */
