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

#ifndef __OPENCV_CUDEV_GRID_PYR_UP_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_PYR_UP_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/vec_traits.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../util/type_traits.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace pyramids_detail
{
    template <class SrcPtr, typename DstType>
    __global__ void pyrUp(const SrcPtr src, GlobPtr<DstType> dst, const int src_rows, const int src_cols, const int dst_rows, const int dst_cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<src_type>::elem_type src_elem_type;
        typedef typename LargerType<float, src_elem_type>::type work_elem_type;
        typedef typename MakeVec<work_elem_type, VecTraits<src_type>::cn>::type work_type;

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ work_type s_srcPatch[10][10];
        __shared__ work_type s_dstPatch[20][16];

        if (threadIdx.x < 10 && threadIdx.y < 10)
        {
            int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
            int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

            srcx = ::abs(srcx);
            srcx = ::min(src_cols - 1, srcx);

            srcy = ::abs(srcy);
            srcy = ::min(src_rows - 1, srcy);

            s_srcPatch[threadIdx.y][threadIdx.x] = saturate_cast<work_type>(src(srcy, srcx));
        }

        __syncthreads();

        work_type sum = VecTraits<work_type>::all(0);

        const int evenFlag = static_cast<int>((threadIdx.x & 1) == 0);
        const int oddFlag  = static_cast<int>((threadIdx.x & 1) != 0);
        const bool eveny = ((threadIdx.y & 1) == 0);
        const int tidx = threadIdx.x;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (threadIdx.y >> 1)][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[2 + threadIdx.y][threadIdx.x] = sum;

        if (threadIdx.y < 2)
        {
            sum = VecTraits<work_type>::all(0);

            if (eveny)
            {
                sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
                sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
                sum = sum + (evenFlag * 0.375f ) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
                sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
                sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
            }

            s_dstPatch[threadIdx.y][threadIdx.x] = sum;
        }

        if (threadIdx.y > 13)
        {
            sum = VecTraits<work_type>::all(0);

            if (eveny)
            {
                sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
                sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
                sum = sum + (evenFlag * 0.375f ) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
                sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
                sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
            }

            s_dstPatch[4 + threadIdx.y][threadIdx.x] = sum;
        }

        __syncthreads();

        sum = VecTraits<work_type>::all(0);

        const int tidy = threadIdx.y;

        sum = sum + 0.0625f * s_dstPatch[2 + tidy - 2][threadIdx.x];
        sum = sum + 0.25f   * s_dstPatch[2 + tidy - 1][threadIdx.x];
        sum = sum + 0.375f  * s_dstPatch[2 + tidy    ][threadIdx.x];
        sum = sum + 0.25f   * s_dstPatch[2 + tidy + 1][threadIdx.x];
        sum = sum + 0.0625f * s_dstPatch[2 + tidy + 2][threadIdx.x];

        if (x < dst_cols && y < dst_rows)
            dst(y, x) = saturate_cast<DstType>(4.0f * sum);
    }

    template <class SrcPtr, typename DstType>
    __host__ void pyrUp(const SrcPtr& src, const GlobPtr<DstType>& dst, int src_rows, int src_cols, int dst_rows, int dst_cols, cudaStream_t stream)
    {
        const dim3 block(16, 16);
        const dim3 grid(divUp(dst_cols, block.x), divUp(dst_rows, block.y));

        pyrUp<<<grid, block, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
