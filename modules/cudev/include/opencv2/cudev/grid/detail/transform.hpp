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

#ifndef __OPENCV_CUDEV_GRID_TRANSFORM_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_TRANSFORM_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/tuple.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../util/vec_traits.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace grid_transform_detail
{
    // OpUnroller

    template <int cn> struct OpUnroller;

    template <> struct OpUnroller<1>
    {
        template <typename T, typename D, class UnOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T& src, D& dst, const UnOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
        }

        template <typename T1, typename T2, typename D, class BinOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2, D& dst, const BinOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
        }
    };

    template <> struct OpUnroller<2>
    {
        template <typename T, typename D, class UnOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T& src, D& dst, const UnOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src.y);
        }

        template <typename T1, typename T2, typename D, class BinOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2, D& dst, const BinOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src1.y, src2.y);
        }
    };

    template <> struct OpUnroller<3>
    {
        template <typename T, typename D, class UnOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T& src, D& dst, const UnOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src.z);
        }

        template <typename T1, typename T2, typename D, class BinOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2, D& dst, const BinOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src1.y, src2.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src1.z, src2.z);
        }
    };

    template <> struct OpUnroller<4>
    {
        template <typename T, typename D, class UnOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T& src, D& dst, const UnOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src.z);
            if (mask(y, x_shifted + 3))
                dst.w = op(src.w);
        }

        template <typename T1, typename T2, typename D, class BinOp, class MaskPtr>
        __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2, D& dst, const BinOp& op, const MaskPtr& mask, int x_shifted, int y)
        {
            if (mask(y, x_shifted))
                dst.x = op(src1.x, src2.x);
            if (mask(y, x_shifted + 1))
                dst.y = op(src1.y, src2.y);
            if (mask(y, x_shifted + 2))
                dst.z = op(src1.z, src2.z);
            if (mask(y, x_shifted + 3))
                dst.w = op(src1.w, src2.w);
        }
    };

    // transformSimple

    template <class SrcPtr, typename DstType, class UnOp, class MaskPtr>
    __global__ void transformSimple(const SrcPtr src, GlobPtr<DstType> dst, const UnOp op, const MaskPtr mask, const int rows, const int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = saturate_cast<DstType>(op(src(y, x)));
    }

    template <class SrcPtr1, class SrcPtr2, typename DstType, class BinOp, class MaskPtr>
    __global__ void transformSimple(const SrcPtr1 src1, const SrcPtr2 src2, GlobPtr<DstType> dst, const BinOp op, const MaskPtr mask, const int rows, const int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = saturate_cast<DstType>(op(src1(y, x), src2(y, x)));
    }

    // transformSmart

    template <int SHIFT, typename SrcType, typename DstType, class UnOp, class MaskPtr>
    __global__ void transformSmart(const GlobPtr<SrcType> src_, GlobPtr<DstType> dst_, const UnOp op, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename MakeVec<SrcType, SHIFT>::type read_type;
        typedef typename MakeVec<DstType, SHIFT>::type write_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x_shifted = x * SHIFT;

        if (y < rows)
        {
            const SrcType* src = src_.row(y);
            DstType* dst = dst_.row(y);

            if (x_shifted + SHIFT - 1 < cols)
            {
                const read_type src_n_el = ((const read_type*)src)[x];
                write_type dst_n_el = ((const write_type*)dst)[x];

                OpUnroller<SHIFT>::unroll(src_n_el, dst_n_el, op, mask, x_shifted, y);

                ((write_type*)dst)[x] = dst_n_el;
            }
            else
            {
                for (int real_x = x_shifted; real_x < cols; ++real_x)
                {
                    if (mask(y, real_x))
                        dst[real_x] = op(src[real_x]);
                }
            }
        }
    }

    template <int SHIFT, typename SrcType1, typename SrcType2, typename DstType, class BinOp, class MaskPtr>
    __global__ void transformSmart(const GlobPtr<SrcType1> src1_, const GlobPtr<SrcType2> src2_, GlobPtr<DstType> dst_, const BinOp op, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename MakeVec<SrcType1, SHIFT>::type read_type1;
        typedef typename MakeVec<SrcType2, SHIFT>::type read_type2;
        typedef typename MakeVec<DstType, SHIFT>::type write_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x_shifted = x * SHIFT;

        if (y < rows)
        {
            const SrcType1* src1 = src1_.row(y);
            const SrcType2* src2 = src2_.row(y);
            DstType* dst = dst_.row(y);

            if (x_shifted + SHIFT - 1 < cols)
            {
                const read_type1 src1_n_el = ((const read_type1*)src1)[x];
                const read_type2 src2_n_el = ((const read_type2*)src2)[x];
                write_type dst_n_el = ((const write_type*)dst)[x];

                OpUnroller<SHIFT>::unroll(src1_n_el, src2_n_el, dst_n_el, op, mask, x_shifted, y);

                ((write_type*)dst)[x] = dst_n_el;
            }
            else
            {
                for (int real_x = x_shifted; real_x < cols; ++real_x)
                {
                    if (mask(y, real_x))
                        dst[real_x] = op(src1[real_x], src2[real_x]);
                }
            }
        }
    }

    // TransformDispatcher

    template <bool UseSmart, class Policy> struct TransformDispatcher;

    template <class Policy> struct TransformDispatcher<false, Policy>
    {
        template <class SrcPtr, typename DstType, class UnOp, class MaskPtr>
        __host__ static void call(const SrcPtr& src, const GlobPtr<DstType>& dst, const UnOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            const dim3 block(Policy::block_size_x, Policy::block_size_y);
            const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

            transformSimple<<<grid, block, 0, stream>>>(src, dst, op, mask, rows, cols);
            CV_CUDEV_SAFE_CALL( cudaGetLastError() );

            if (stream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }

        template <class SrcPtr1, class SrcPtr2, typename DstType, class BinOp, class MaskPtr>
        __host__ static void call(const SrcPtr1& src1, const SrcPtr2& src2, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            const dim3 block(Policy::block_size_x, Policy::block_size_y);
            const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

            transformSimple<<<grid, block, 0, stream>>>(src1, src2, dst, op, mask, rows, cols);
            CV_CUDEV_SAFE_CALL( cudaGetLastError() );

            if (stream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }
    };

    template <class Policy> struct TransformDispatcher<true, Policy>
    {
        template <typename T>
        __host__ static bool isAligned(const T* ptr, size_t size)
        {
            return reinterpret_cast<size_t>(ptr) % size == 0;
        }

        __host__ static bool isAligned(size_t step, size_t size)
        {
            return step % size == 0;
        }

        template <typename SrcType, typename DstType, class UnOp, class MaskPtr>
        __host__ static void call(const GlobPtr<SrcType>& src, const GlobPtr<DstType>& dst, const UnOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            if (Policy::shift == 1 ||
                !isAligned(src.data, Policy::shift * sizeof(SrcType)) || !isAligned(src.step, Policy::shift * sizeof(SrcType)) ||
                !isAligned(dst.data, Policy::shift * sizeof(DstType)) || !isAligned(dst.step, Policy::shift * sizeof(DstType)))
            {
                TransformDispatcher<false, Policy>::call(src, dst, op, mask, rows, cols, stream);
                return;
            }

            const dim3 block(Policy::block_size_x, Policy::block_size_y);
            const dim3 grid(divUp(cols, block.x * Policy::shift), divUp(rows, block.y));

            transformSmart<Policy::shift><<<grid, block, 0, stream>>>(src, dst, op, mask, rows, cols);
            CV_CUDEV_SAFE_CALL( cudaGetLastError() );

            if (stream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }

        template <typename SrcType1, typename SrcType2, typename DstType, class BinOp, class MaskPtr>
        __host__ static void call(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            if (Policy::shift == 1 ||
                !isAligned(src1.data, Policy::shift * sizeof(SrcType1)) || !isAligned(src1.step, Policy::shift * sizeof(SrcType1)) ||
                !isAligned(src2.data, Policy::shift * sizeof(SrcType2)) || !isAligned(src2.step, Policy::shift * sizeof(SrcType2)) ||
                !isAligned(dst.data,  Policy::shift * sizeof(DstType))  || !isAligned(dst.step,  Policy::shift * sizeof(DstType)))
            {
                TransformDispatcher<false, Policy>::call(src1, src2, dst, op, mask, rows, cols, stream);
                return;
            }

            const dim3 block(Policy::block_size_x, Policy::block_size_y);
            const dim3 grid(divUp(cols, block.x * Policy::shift), divUp(rows, block.y));

            transformSmart<Policy::shift><<<grid, block, 0, stream>>>(src1, src2, dst, op, mask, rows, cols);
            CV_CUDEV_SAFE_CALL( cudaGetLastError() );

            if (stream == 0)
                CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
        }
    };

    template <class Policy, class SrcPtr, typename DstType, class UnOp, class MaskPtr>
    __host__ void transform_unary(const SrcPtr& src, const GlobPtr<DstType>& dst, const UnOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        TransformDispatcher<false, Policy>::call(src, dst, op, mask, rows, cols, stream);
    }

    template <class Policy, class SrcPtr1, class SrcPtr2, typename DstType, class BinOp, class MaskPtr>
    __host__ void transform_binary(const SrcPtr1& src1, const SrcPtr2& src2, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        TransformDispatcher<false, Policy>::call(src1, src2, dst, op, mask, rows, cols, stream);
    }

    template <class Policy, typename SrcType, typename DstType, class UnOp, class MaskPtr>
    __host__ void transform_unary(const GlobPtr<SrcType>& src, const GlobPtr<DstType>& dst, const UnOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        TransformDispatcher<VecTraits<SrcType>::cn == 1 && VecTraits<DstType>::cn == 1 && Policy::shift != 1, Policy>::call(src, dst, op, mask, rows, cols, stream);
    }

    template <class Policy, typename SrcType1, typename SrcType2, typename DstType, class BinOp, class MaskPtr>
    __host__ void transform_binary(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        TransformDispatcher<VecTraits<SrcType1>::cn == 1 && VecTraits<SrcType2>::cn == 1 && VecTraits<DstType>::cn == 1 && Policy::shift != 1, Policy>::call(src1, src2, dst, op, mask, rows, cols, stream);
    }

    // transform_tuple

    template <int count> struct Unroll
    {
        template <class SrcVal, class DstPtrTuple, class OpTuple>
        __device__ static void transform(const SrcVal& srcVal, DstPtrTuple& dst, const OpTuple& op, int y, int x)
        {
            typedef typename tuple_element<count - 1, DstPtrTuple>::type dst_ptr_type;
            typedef typename PtrTraits<dst_ptr_type>::value_type dst_type;

            get<count - 1>(dst)(y, x) = saturate_cast<dst_type>(get<count - 1>(op)(srcVal));
            Unroll<count - 1>::transform(srcVal, dst, op, y, x);
        }
    };
    template <> struct Unroll<0>
    {
        template <class SrcVal, class DstPtrTuple, class OpTuple>
        __device__ __forceinline__ static void transform(const SrcVal&, DstPtrTuple&, const OpTuple&, int, int)
        {
        }
    };

    template <class SrcPtr, class DstPtrTuple, class OpTuple, class MaskPtr>
    __global__ void transform_tuple(const SrcPtr src, DstPtrTuple dst, const OpTuple op, const MaskPtr mask, const int rows, const int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        typename PtrTraits<SrcPtr>::value_type srcVal = src(y, x);

        Unroll<tuple_size<DstPtrTuple>::value>::transform(srcVal, dst, op, y, x);
    }

    template <class Policy, class SrcPtrTuple, class DstPtrTuple, class OpTuple, class MaskPtr>
    __host__ void transform_tuple(const SrcPtrTuple& src, const DstPtrTuple& dst, const OpTuple& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        transform_tuple<<<grid, block, 0, stream>>>(src, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
