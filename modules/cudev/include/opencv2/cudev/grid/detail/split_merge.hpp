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

#ifndef __OPENCV_CUDEV_GRID_SPLIT_MERGE_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_SPLIT_MERGE_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../util/tuple.hpp"
#include "../../util/vec_traits.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace grid_split_merge_detail
{
    // merge

    template <class Src1Ptr, class Src2Ptr, typename DstType, class MaskPtr>
    __global__ void mergeC2(const Src1Ptr src1, const Src2Ptr src2, GlobPtr<DstType> dst, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename VecTraits<DstType>::elem_type dst_elem_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = VecTraits<DstType>::make(
                    saturate_cast<dst_elem_type>(src1(y, x)),
                    saturate_cast<dst_elem_type>(src2(y, x))
                    );
    }

    template <class Policy, class Src1Ptr, class Src2Ptr, typename DstType, class MaskPtr>
    __host__ void mergeC2(const Src1Ptr& src1, const Src2Ptr& src2, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        mergeC2<<<grid, block, 0, stream>>>(src1, src2, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    template <class Src1Ptr, class Src2Ptr, class Src3Ptr, typename DstType, class MaskPtr>
    __global__ void mergeC3(const Src1Ptr src1, const Src2Ptr src2, const Src3Ptr src3, GlobPtr<DstType> dst, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename VecTraits<DstType>::elem_type dst_elem_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = VecTraits<DstType>::make(
                    saturate_cast<dst_elem_type>(src1(y, x)),
                    saturate_cast<dst_elem_type>(src2(y, x)),
                    saturate_cast<dst_elem_type>(src3(y, x))
                    );
    }

    template <class Policy, class Src1Ptr, class Src2Ptr, class Src3Ptr, typename DstType, class MaskPtr>
    __host__ void mergeC3(const Src1Ptr& src1, const Src2Ptr& src2, const Src3Ptr& src3, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        mergeC3<<<grid, block, 0, stream>>>(src1, src2, src3, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    template <class Src1Ptr, class Src2Ptr, class Src3Ptr, class Src4Ptr, typename DstType, class MaskPtr>
    __global__ void mergeC4(const Src1Ptr src1, const Src2Ptr src2, const Src3Ptr src3, const Src4Ptr src4, GlobPtr<DstType> dst, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename VecTraits<DstType>::elem_type dst_elem_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = VecTraits<DstType>::make(
                    saturate_cast<dst_elem_type>(src1(y, x)),
                    saturate_cast<dst_elem_type>(src2(y, x)),
                    saturate_cast<dst_elem_type>(src3(y, x)),
                    saturate_cast<dst_elem_type>(src4(y, x))
                    );
    }

    template <class Policy, class Src1Ptr, class Src2Ptr, class Src3Ptr, class Src4Ptr, typename DstType, class MaskPtr>
    __host__ void mergeC4(const Src1Ptr& src1, const Src2Ptr& src2, const Src3Ptr& src3, const Src4Ptr& src4, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        mergeC4<<<grid, block, 0, stream>>>(src1, src2, src3, src4, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    template <int cn, class Policy> struct MergeImpl;

    template <class Policy> struct MergeImpl<2, Policy>
    {
        template <class SrcPtrTuple, typename DstType, class MaskPtr>
        __host__ static void merge(const SrcPtrTuple& src, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            mergeC2<Policy>(get<0>(src), get<1>(src), dst, mask, rows, cols, stream);
        }
    };

    template <class Policy> struct MergeImpl<3, Policy>
    {
        template <class SrcPtrTuple, typename DstType, class MaskPtr>
        __host__ static void merge(const SrcPtrTuple& src, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            mergeC3<Policy>(get<0>(src), get<1>(src), get<2>(src), dst, mask, rows, cols, stream);
        }
    };

    template <class Policy> struct MergeImpl<4, Policy>
    {
        template <class SrcPtrTuple, typename DstType, class MaskPtr>
        __host__ static void merge(const SrcPtrTuple& src, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
        {
            mergeC4<Policy>(get<0>(src), get<1>(src), get<2>(src), get<3>(src), dst, mask, rows, cols, stream);
        }
    };

    // split

    template <class SrcPtr, typename DstType, class MaskPtr>
    __global__ void split(const SrcPtr src, GlobPtr<DstType> dst1, GlobPtr<DstType> dst2, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        const src_type src_value = src(y, x);

        dst1(y, x) = src_value.x;
        dst2(y, x) = src_value.y;
    }

    template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
    __host__ void split(const SrcPtr& src, const GlobPtr<DstType>& dst1, const GlobPtr<DstType>& dst2, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        split<<<grid, block, 0, stream>>>(src, dst1, dst2, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    template <class SrcPtr, typename DstType, class MaskPtr>
    __global__ void split(const SrcPtr src, GlobPtr<DstType> dst1, GlobPtr<DstType> dst2, GlobPtr<DstType> dst3, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        const src_type src_value = src(y, x);

        dst1(y, x) = src_value.x;
        dst2(y, x) = src_value.y;
        dst3(y, x) = src_value.z;
    }

    template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
    __host__ void split(const SrcPtr& src, const GlobPtr<DstType>& dst1, const GlobPtr<DstType>& dst2, const GlobPtr<DstType>& dst3, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        split<<<grid, block, 0, stream>>>(src, dst1, dst2, dst3, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }

    template <class SrcPtr, typename DstType, class MaskPtr>
    __global__ void split(const SrcPtr src, GlobPtr<DstType> dst1, GlobPtr<DstType> dst2, GlobPtr<DstType> dst3, GlobPtr<DstType> dst4, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        const src_type src_value = src(y, x);

        dst1(y, x) = src_value.x;
        dst2(y, x) = src_value.y;
        dst3(y, x) = src_value.z;
        dst4(y, x) = src_value.w;
    }

    template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
    __host__ void split(const SrcPtr& src, const GlobPtr<DstType>& dst1, const GlobPtr<DstType>& dst2, const GlobPtr<DstType>& dst3, const GlobPtr<DstType>& dst4, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        split<<<grid, block, 0, stream>>>(src, dst1, dst2, dst3, dst4, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    }
}

}}

#endif
