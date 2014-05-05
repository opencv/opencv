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

#ifndef __OPENCV_CUDEV_GRID_PYR_DOWN_DETAIL_HPP__
#define __OPENCV_CUDEV_GRID_PYR_DOWN_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/vec_traits.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../util/type_traits.hpp"
#include "../../ptr2d/glob.hpp"
#include "../../ptr2d/traits.hpp"

namespace cv { namespace cudev {

namespace pyramids_detail
{
    template <class Brd, class SrcPtr, typename DstType>
    __global__ void pyrDown(const SrcPtr src, GlobPtr<DstType> dst, const int src_rows, const int src_cols, const int dst_cols)
    {
        typedef typename PtrTraits<SrcPtr>::value_type src_type;
        typedef typename VecTraits<src_type>::elem_type src_elem_type;
        typedef typename LargerType<float, src_elem_type>::type work_elem_type;
        typedef typename MakeVec<work_elem_type, VecTraits<src_type>::cn>::type work_type;

        __shared__ work_type smem[256 + 4];

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y;

        const int src_y = 2 * y;

        if (src_y >= 2 && src_y < src_rows - 2 && x >= 2 && x < src_cols - 2)
        {
            {
                work_type sum;

                sum =       0.0625f * src(src_y - 2, x);
                sum = sum + 0.25f   * src(src_y - 1, x);
                sum = sum + 0.375f  * src(src_y    , x);
                sum = sum + 0.25f   * src(src_y + 1, x);
                sum = sum + 0.0625f * src(src_y + 2, x);

                smem[2 + threadIdx.x] = sum;
            }

            if (threadIdx.x < 2)
            {
                const int left_x = x - 2;

                work_type sum;

                sum =       0.0625f * src(src_y - 2, left_x);
                sum = sum + 0.25f   * src(src_y - 1, left_x);
                sum = sum + 0.375f  * src(src_y    , left_x);
                sum = sum + 0.25f   * src(src_y + 1, left_x);
                sum = sum + 0.0625f * src(src_y + 2, left_x);

                smem[threadIdx.x] = sum;
            }

            if (threadIdx.x > 253)
            {
                const int right_x = x + 2;

                work_type sum;

                sum =       0.0625f * src(src_y - 2, right_x);
                sum = sum + 0.25f   * src(src_y - 1, right_x);
                sum = sum + 0.375f  * src(src_y    , right_x);
                sum = sum + 0.25f   * src(src_y + 1, right_x);
                sum = sum + 0.0625f * src(src_y + 2, right_x);

                smem[4 + threadIdx.x] = sum;
            }
        }
        else
        {
            {
                work_type sum;

                sum =       0.0625f * src(Brd::idx_low(src_y - 2, src_rows) , Brd::idx_high(x, src_cols));
                sum = sum + 0.25f   * src(Brd::idx_low(src_y - 1, src_rows) , Brd::idx_high(x, src_cols));
                sum = sum + 0.375f  * src(src_y                             , Brd::idx_high(x, src_cols));
                sum = sum + 0.25f   * src(Brd::idx_high(src_y + 1, src_rows), Brd::idx_high(x, src_cols));
                sum = sum + 0.0625f * src(Brd::idx_high(src_y + 2, src_rows), Brd::idx_high(x, src_cols));

                smem[2 + threadIdx.x] = sum;
            }

            if (threadIdx.x < 2)
            {
                const int left_x = x - 2;

                work_type sum;

                sum =       0.0625f * src(Brd::idx_low(src_y - 2, src_rows) , Brd::idx_low(Brd::idx_high(left_x, src_cols), src_cols));
                sum = sum + 0.25f   * src(Brd::idx_low(src_y - 1, src_rows) , Brd::idx_low(Brd::idx_high(left_x, src_cols), src_cols));
                sum = sum + 0.375f  * src(src_y                             , Brd::idx_low(Brd::idx_high(left_x, src_cols), src_cols));
                sum = sum + 0.25f   * src(Brd::idx_high(src_y + 1, src_rows), Brd::idx_low(Brd::idx_high(left_x, src_cols), src_cols));
                sum = sum + 0.0625f * src(Brd::idx_high(src_y + 2, src_rows), Brd::idx_low(Brd::idx_high(left_x, src_cols), src_cols));

                smem[threadIdx.x] = sum;
            }

            if (threadIdx.x > 253)
            {
                const int right_x = x + 2;

                work_type sum;

                sum =       0.0625f * src(Brd::idx_low(src_y - 2, src_rows) , Brd::idx_high(right_x, src_cols));
                sum = sum + 0.25f   * src(Brd::idx_low(src_y - 1, src_rows) , Brd::idx_high(right_x, src_cols));
                sum = sum + 0.375f  * src(src_y                             , Brd::idx_high(right_x, src_cols));
                sum = sum + 0.25f   * src(Brd::idx_high(src_y + 1, src_rows), Brd::idx_high(right_x, src_cols));
                sum = sum + 0.0625f * src(Brd::idx_high(src_y + 2, src_rows), Brd::idx_high(right_x, src_cols));

                smem[4 + threadIdx.x] = sum;
            }
        }

        __syncthreads();

        if (threadIdx.x < 128)
        {
            const int tid2 = threadIdx.x * 2;

            work_type sum;

            sum =       0.0625f * smem[2 + tid2 - 2];
            sum = sum + 0.25f   * smem[2 + tid2 - 1];
            sum = sum + 0.375f  * smem[2 + tid2    ];
            sum = sum + 0.25f   * smem[2 + tid2 + 1];
            sum = sum + 0.0625f * smem[2 + tid2 + 2];

            const int dst_x = (blockIdx.x * blockDim.x + tid2) / 2;

            if (dst_x < dst_cols)
                dst(y, dst_x) = saturate_cast<DstType>(sum);
        }
    }

    template <class Brd, class SrcPtr, typename DstType>
    __host__ void pyrDown(const SrcPtr& src, const GlobPtr<DstType>& dst, int src_rows, int src_cols, int dst_rows, int dst_cols, cudaStream_t stream)
    {
        const dim3 block(256);
        const dim3 grid(divUp(src_cols, block.x), dst_rows);

        pyrDown<Brd><<<grid, block, 0, stream>>>(src, dst, src_rows, src_cols, dst_cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
}

}}

#endif
