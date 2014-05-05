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

#ifndef __OPENCV_CUDEV_GRID_COPY_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_COPY_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/tuple.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace grid_copy_detail
{
    template <class SrcPtr, typename DstType, class MaskPtr>
    __global__ void copy(const SrcPtr src, GlobPtr<DstType> dst, const MaskPtr mask, const int rows, const int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        dst(y, x) = saturate_cast<DstType>(src(y, x));
    }

    template <class Policy, class SrcPtr, typename DstType, class MaskPtr>
    __host__ void copy(const SrcPtr& src, const GlobPtr<DstType>& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        copy<<<grid, block, 0, stream>>>(src, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <int count> struct Unroll
    {
        template <class SrcPtrTuple, class DstPtrTuple>
        __device__ static void copy(const SrcPtrTuple& src, DstPtrTuple& dst, const int y, const int x)
        {
            typedef typename tuple_element<count - 1, DstPtrTuple>::type dst_ptr_type;
            typedef typename PtrTraits<dst_ptr_type>::value_type dst_type;

            get<count - 1>(dst)(y, x) = saturate_cast<dst_type>(get<count - 1>(src)(y, x));
            Unroll<count - 1>::copy(src, dst, y, x);
        }
    };
    template <> struct Unroll<0>
    {
        template <class SrcPtrTuple, class DstPtrTuple>
        __device__ __forceinline__ static void copy(const SrcPtrTuple&, DstPtrTuple&, const int, const int)
        {
        }
    };

    template <class SrcPtrTuple, class DstPtrTuple, class MaskPtr>
    __global__ void copy_tuple(const SrcPtrTuple src, DstPtrTuple dst, const MaskPtr mask, const int rows, const int cols)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= cols || y >= rows || !mask(y, x))
            return;

        Unroll<tuple_size<SrcPtrTuple>::value>::copy(src, dst, y, x);
    }

    template <class Policy, class SrcPtrTuple, class DstPtrTuple, class MaskPtr>
    __host__ void copy_tuple(const SrcPtrTuple& src, const DstPtrTuple& dst, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        copy_tuple<<<grid, block, 0, stream>>>(src, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
