/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef OPENCV_CUDEV_GRID_REDUCE_DETAIL_HPP
#define OPENCV_CUDEV_GRID_REDUCE_DETAIL_HPP

#include "../../common.hpp"
#include "../../util/tuple.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../util/atomic.hpp"
#include "../../util/vec_traits.hpp"
#include "../../util/type_traits.hpp"
#include "../../util/limits.hpp"
#include "../../block/reduce.hpp"
#include "../../functional/functional.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace grid_reduce_detail
{
    // Unroll

    template <int cn> struct Unroll;

    template <> struct Unroll<1>
    {
        template <int BLOCK_SIZE, typename R>
        __device__ __forceinline__ static volatile R* smem(R* ptr)
        {
            return ptr;
        }

        template <typename R>
        __device__ __forceinline__ static R& res(R& val)
        {
            return val;
        }

        template <class Op>
        __device__ __forceinline__ static const Op& op(const Op& aop)
        {
            return aop;
        }
    };

    template <> struct Unroll<2>
    {
        template <int BLOCK_SIZE, typename R>
        __device__ __forceinline__ static tuple<volatile R*, volatile R*> smem(R* ptr)
        {
            return smem_tuple(ptr, ptr + BLOCK_SIZE);
        }

        template <typename R>
        __device__ __forceinline__ static tuple<typename VecTraits<R>::elem_type&, typename VecTraits<R>::elem_type&> res(R& val)
        {
            return tie(val.x, val.y);
        }

        template <class Op>
        __device__ __forceinline__ static tuple<Op, Op> op(const Op& aop)
        {
            return make_tuple(aop, aop);
        }
    };

    template <> struct Unroll<3>
    {
        template <int BLOCK_SIZE, typename R>
        __device__ __forceinline__ static tuple<volatile R*, volatile R*, volatile R*> smem(R* ptr)
        {
            return smem_tuple(ptr, ptr + BLOCK_SIZE, ptr + 2 * BLOCK_SIZE);
        }

        template <typename R>
        __device__ __forceinline__ static tuple<typename VecTraits<R>::elem_type&,
                                                typename VecTraits<R>::elem_type&,
                                                typename VecTraits<R>::elem_type&> res(R& val)
        {
            return tie(val.x, val.y, val.z);
        }

        template <class Op>
        __device__ __forceinline__ static tuple<Op, Op, Op> op(const Op& aop)
        {
            return make_tuple(aop, aop, aop);
        }
    };

    template <> struct Unroll<4>
    {
        template <int BLOCK_SIZE, typename R>
        __device__ __forceinline__ static tuple<volatile R*, volatile R*, volatile R*, volatile R*> smem(R* ptr)
        {
            return smem_tuple(ptr, ptr + BLOCK_SIZE, ptr + 2 * BLOCK_SIZE, ptr + 3 * BLOCK_SIZE);
        }

        template <typename R>
        __device__ __forceinline__ static tuple<typename VecTraits<R>::elem_type&,
                                                typename VecTraits<R>::elem_type&,
                                                typename VecTraits<R>::elem_type&,
                                                typename VecTraits<R>::elem_type&> res(R& val)
        {
            return tie(val.x, val.y, val.z, val.w);
        }

        template <class Op>
        __device__ __forceinline__ static tuple<Op, Op, Op, Op> op(const Op& aop)
        {
            return make_tuple(aop, aop, aop, aop);
        }
    };

    // AtomicUnroll

    template <typename R, int cn> struct AtomicUnroll;

    template <typename R> struct AtomicUnroll<R, 1>
    {
        __device__ __forceinline__ static void add(R* ptr, R val)
        {
            atomicAdd(ptr, val);
        }

        __device__ __forceinline__ static void min(R* ptr, R val)
        {
            atomicMin(ptr, val);
        }

        __device__ __forceinline__ static void max(R* ptr, R val)
        {
            atomicMax(ptr, val);
        }
    };

    template <typename R> struct AtomicUnroll<R, 2>
    {
        typedef typename MakeVec<R, 2>::type val_type;

        __device__ __forceinline__ static void add(R* ptr, val_type val)
        {
            atomicAdd(ptr, val.x);
            atomicAdd(ptr + 1, val.y);
        }

        __device__ __forceinline__ static void min(R* ptr, val_type val)
        {
            atomicMin(ptr, val.x);
            atomicMin(ptr + 1, val.y);
        }

        __device__ __forceinline__ static void max(R* ptr, val_type val)
        {
            atomicMax(ptr, val.x);
            atomicMax(ptr + 1, val.y);
        }
    };

    template <typename R> struct AtomicUnroll<R, 3>
    {
        typedef typename MakeVec<R, 3>::type val_type;

        __device__ __forceinline__ static void add(R* ptr, val_type val)
        {
            atomicAdd(ptr, val.x);
            atomicAdd(ptr + 1, val.y);
            atomicAdd(ptr + 2, val.z);
        }

        __device__ __forceinline__ static void min(R* ptr, val_type val)
        {
            atomicMin(ptr, val.x);
            atomicMin(ptr + 1, val.y);
            atomicMin(ptr + 2, val.z);
        }

        __device__ __forceinline__ static void max(R* ptr, val_type val)
        {
            atomicMax(ptr, val.x);
            atomicMax(ptr + 1, val.y);
            atomicMax(ptr + 2, val.z);
        }
    };

    template <typename R> struct AtomicUnroll<R, 4>
    {
        typedef typename MakeVec<R, 4>::type val_type;

        __device__ __forceinline__ static void add(R* ptr, val_type val)
        {
            atomicAdd(ptr, val.x);
            atomicAdd(ptr + 1, val.y);
            atomicAdd(ptr + 2, val.z);
            atomicAdd(ptr + 3, val.w);
        }

        __device__ __forceinline__ static void min(R* ptr, val_type val)
        {
            atomicMin(ptr, val.x);
            atomicMin(ptr + 1, val.y);
            atomicMin(ptr + 2, val.z);
            atomicMin(ptr + 3, val.w);
        }

        __device__ __forceinline__ static void max(R* ptr, val_type val)
        {
            atomicMax(ptr, val.x);
            atomicMax(ptr + 1, val.y);
            atomicMax(ptr + 2, val.z);
            atomicMax(ptr + 3, val.w);
        }
    };

    // SumReductor

    template <typename src_type, typename work_type> struct SumReductor
    {
        typedef typename VecTraits<work_type>::elem_type work_elem_type;
        enum { cn = VecTraits<src_type>::cn };

        work_type sum;

        __device__ __forceinline__ SumReductor()
        {
            sum = VecTraits<work_type>::all(0);
        }

        __device__ __forceinline__ void reduceVal(typename TypeTraits<src_type>::parameter_type srcVal)
        {
            sum = sum + saturate_cast<work_type>(srcVal);
        }

        template <int BLOCK_SIZE>
        __device__ void reduceGrid(work_elem_type* result, int tid)
        {
            __shared__ work_elem_type smem[BLOCK_SIZE * cn];

            blockReduce<BLOCK_SIZE>(Unroll<cn>::template smem<BLOCK_SIZE>(smem), Unroll<cn>::res(sum), tid, Unroll<cn>::op(plus<work_elem_type>()));

            if (tid == 0)
                AtomicUnroll<work_elem_type, cn>::add(result, sum);
        }
    };

    // MinMaxReductor

    template <typename T> struct minop : minimum<T>
    {
        __device__ __forceinline__ static T initial()
        {
            return numeric_limits<T>::max();
        }

        __device__ __forceinline__ static void atomic(T* result, T myval)
        {
            atomicMin(result, myval);
        }
    };

    template <typename T> struct maxop : maximum<T>
    {
        __device__ __forceinline__ static T initial()
        {
            return -numeric_limits<T>::max();
        }

        __device__ __forceinline__ static void atomic(T* result, T myval)
        {
            atomicMax(result, myval);
        }
    };

    struct both
    {
    };

    template <class Op, typename src_type, typename work_type> struct MinMaxReductor
    {
        work_type myval;

        __device__ __forceinline__ MinMaxReductor()
        {
            myval = Op::initial();
        }

        __device__ __forceinline__ void reduceVal(typename TypeTraits<src_type>::parameter_type srcVal)
        {
            Op op;

            myval = op(myval, srcVal);
        }

        template <int BLOCK_SIZE>
        __device__ void reduceGrid(work_type* result, int tid)
        {
            __shared__ work_type smem[BLOCK_SIZE];

            Op op;

            blockReduce<BLOCK_SIZE>(smem, myval, tid, op);

            if (tid == 0)
                Op::atomic(result, myval);
        }
    };

    template <typename src_type, typename work_type> struct MinMaxReductor<both, src_type, work_type>
    {
        work_type mymin;
        work_type mymax;

        __device__ __forceinline__ MinMaxReductor()
        {
            mymin = numeric_limits<work_type>::max();
            mymax = -numeric_limits<work_type>::max();
        }

        __device__ __forceinline__ void reduceVal(typename TypeTraits<src_type>::parameter_type srcVal)
        {
            minimum<work_type> minOp;
            maximum<work_type> maxOp;

            mymin = minOp(mymin, srcVal);
            mymax = maxOp(mymax, srcVal);
        }

        template <int BLOCK_SIZE>
        __device__ void reduceGrid(work_type* result, int tid)
        {
            __shared__ work_type sminval[BLOCK_SIZE];
            __shared__ work_type smaxval[BLOCK_SIZE];

            minimum<work_type> minOp;
            maximum<work_type> maxOp;

            blockReduce<BLOCK_SIZE>(smem_tuple(sminval, smaxval), tie(mymin, mymax), tid, make_tuple(minOp, maxOp));

            if (tid == 0)
            {
                atomicMin(result, mymin);
                atomicMax(result + 1, mymax);
            }
        }
    };

    // glob_reduce

    template <class Reductor, int BLOCK_SIZE, int PATCH_X, int PATCH_Y, class SrcPtr, typename ResType, class MaskPtr>
    __global__ void reduce(const SrcPtr src, ResType* result, const MaskPtr mask, const int rows, const int cols)
    {
        const int x0 = blockIdx.x * blockDim.x * PATCH_X + threadIdx.x;
        const int y0 = blockIdx.y * blockDim.y * PATCH_Y + threadIdx.y;

        Reductor reductor;

        for (int i = 0, y = y0; i < PATCH_Y && y < rows; ++i, y += blockDim.y)
        {
            for (int j = 0, x = x0; j < PATCH_X && x < cols; ++j, x += blockDim.x)
            {
                if (mask(y, x))
                {
                    reductor.reduceVal(src(y, x));
                }
            }
        }

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        reductor.template reduceGrid<BLOCK_SIZE>(result, tid);
    }

    template <class Reductor, class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void reduce(const SrcPtr& src, ResType* result, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x * Policy::patch_size_x), divUp(rows, block.y * Policy::patch_size_y));

        reduce<Reductor, Policy::block_size_x * Policy::block_size_y, Policy::patch_size_x, Policy::patch_size_y><<<grid, block, 0, stream>>>(src, result, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    // callers

    template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void sum(const SrcPtr& src, ResType* result, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<ResType>::elem_type res_elem_type;

        reduce<SumReductor<src_type, ResType>, Policy>(src, (res_elem_type*) result, mask, rows, cols, stream);
    }

    template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void minVal(const SrcPtr& src, ResType* result, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        reduce<MinMaxReductor<minop<ResType>, src_type, ResType>, Policy>(src, result, mask, rows, cols, stream);
    }

    template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void maxVal(const SrcPtr& src, ResType* result, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        reduce<MinMaxReductor<maxop<ResType>, src_type, ResType>, Policy>(src, result, mask, rows, cols, stream);
    }

    template <class Policy, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void minMaxVal(const SrcPtr& src, ResType* result, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        reduce<MinMaxReductor<both, src_type, ResType>, Policy>(src, result, mask, rows, cols, stream);
    }
}

}}

#endif
