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
#include "opencv2/gpu/device/border_interpolate.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T, typename B> __global__ void pyrDown(const PtrStepSz<T> src, PtrStep<T> dst, const B b, int dst_cols)
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_t;

            __shared__ work_t smem[256 + 4];

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;

            const int src_y = 2 * y;

            if (src_y >= 2 && src_y < src.rows - 2 && x >= 2 && x < src.cols - 2)
            {
                {
                    work_t sum;

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

                    work_t sum;

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

                    work_t sum;

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
                    work_t sum;

                    sum =       0.0625f * src(b.idx_row_low (src_y - 2), b.idx_col_high(x));
                    sum = sum + 0.25f   * src(b.idx_row_low (src_y - 1), b.idx_col_high(x));
                    sum = sum + 0.375f  * src(src_y                    , b.idx_col_high(x));
                    sum = sum + 0.25f   * src(b.idx_row_high(src_y + 1), b.idx_col_high(x));
                    sum = sum + 0.0625f * src(b.idx_row_high(src_y + 2), b.idx_col_high(x));

                    smem[2 + threadIdx.x] = sum;
                }

                if (threadIdx.x < 2)
                {
                    const int left_x = x - 2;

                    work_t sum;

                    sum =       0.0625f * src(b.idx_row_low (src_y - 2), b.idx_col(left_x));
                    sum = sum + 0.25f   * src(b.idx_row_low (src_y - 1), b.idx_col(left_x));
                    sum = sum + 0.375f  * src(src_y                    , b.idx_col(left_x));
                    sum = sum + 0.25f   * src(b.idx_row_high(src_y + 1), b.idx_col(left_x));
                    sum = sum + 0.0625f * src(b.idx_row_high(src_y + 2), b.idx_col(left_x));

                    smem[threadIdx.x] = sum;
                }

                if (threadIdx.x > 253)
                {
                    const int right_x = x + 2;

                    work_t sum;

                    sum =       0.0625f * src(b.idx_row_low (src_y - 2), b.idx_col_high(right_x));
                    sum = sum + 0.25f   * src(b.idx_row_low (src_y - 1), b.idx_col_high(right_x));
                    sum = sum + 0.375f  * src(src_y                    , b.idx_col_high(right_x));
                    sum = sum + 0.25f   * src(b.idx_row_high(src_y + 1), b.idx_col_high(right_x));
                    sum = sum + 0.0625f * src(b.idx_row_high(src_y + 2), b.idx_col_high(right_x));

                    smem[4 + threadIdx.x] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 128)
            {
                const int tid2 = threadIdx.x * 2;

                work_t sum;

                sum =       0.0625f * smem[2 + tid2 - 2];
                sum = sum + 0.25f   * smem[2 + tid2 - 1];
                sum = sum + 0.375f  * smem[2 + tid2    ];
                sum = sum + 0.25f   * smem[2 + tid2 + 1];
                sum = sum + 0.0625f * smem[2 + tid2 + 2];

                const int dst_x = (blockIdx.x * blockDim.x + tid2) / 2;

                if (dst_x < dst_cols)
                    dst.ptr(y)[dst_x] = saturate_cast<T>(sum);
            }
        }

        template <typename T, template <typename> class B> void pyrDown_caller(PtrStepSz<T> src, PtrStepSz<T> dst, cudaStream_t stream)
        {
            const dim3 block(256);
            const dim3 grid(divUp(src.cols, block.x), dst.rows);

            B<T> b(src.rows, src.cols);

            pyrDown<T><<<grid, block, 0, stream>>>(src, dst, b, dst.cols);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <typename T> void pyrDown_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
        {
            pyrDown_caller<T, BrdReflect101>(static_cast< PtrStepSz<T> >(src), static_cast< PtrStepSz<T> >(dst), stream);
        }

        template void pyrDown_gpu<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<uchar2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<uchar3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<uchar4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        //template void pyrDown_gpu<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<char2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<char3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<char4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        template void pyrDown_gpu<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<ushort2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<ushort3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<ushort4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        template void pyrDown_gpu<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<short2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<short3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<short4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        //template void pyrDown_gpu<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<int2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<int3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<int4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        template void pyrDown_gpu<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrDown_gpu<float2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<float3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrDown_gpu<float4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device


#endif /* CUDA_DISABLER */
