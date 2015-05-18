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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T> __global__ void pyrUp(const PtrStepSz<T> src, PtrStepSz<T> dst)
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            __shared__ sum_t s_srcPatch[10][10];
            __shared__ sum_t s_dstPatch[20][16];

            if (threadIdx.x < 10 && threadIdx.y < 10)
            {
                int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
                int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

                srcx = ::abs(srcx);
                srcx = ::min(src.cols - 1, srcx);

                srcy = ::abs(srcy);
                srcy = ::min(src.rows - 1, srcy);

                s_srcPatch[threadIdx.y][threadIdx.x] = saturate_cast<sum_t>(src(srcy, srcx));
            }

            __syncthreads();

            sum_t sum = VecTraits<sum_t>::all(0);

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
                sum = VecTraits<sum_t>::all(0);

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
                sum = VecTraits<sum_t>::all(0);

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

            sum = VecTraits<sum_t>::all(0);

            const int tidy = threadIdx.y;

            sum = sum + 0.0625f * s_dstPatch[2 + tidy - 2][threadIdx.x];
            sum = sum + 0.25f   * s_dstPatch[2 + tidy - 1][threadIdx.x];
            sum = sum + 0.375f  * s_dstPatch[2 + tidy    ][threadIdx.x];
            sum = sum + 0.25f   * s_dstPatch[2 + tidy + 1][threadIdx.x];
            sum = sum + 0.0625f * s_dstPatch[2 + tidy + 2][threadIdx.x];

            if (x < dst.cols && y < dst.rows)
                dst(y, x) = saturate_cast<T>(4.0f * sum);
        }

        template <typename T> void pyrUp_caller(PtrStepSz<T> src, PtrStepSz<T> dst, cudaStream_t stream)
        {
            const dim3 block(16, 16);
            const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            pyrUp<<<grid, block, 0, stream>>>(src, dst);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <typename T> void pyrUp_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream)
        {
            pyrUp_caller<T>(static_cast< PtrStepSz<T> >(src), static_cast< PtrStepSz<T> >(dst), stream);
        }

        template void pyrUp_gpu<uchar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<uchar2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<uchar3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<uchar4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

#ifndef OPENCV_TINY_GPU_MODULE
        //template void pyrUp_gpu<schar>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<char2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<char3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<char4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        template void pyrUp_gpu<ushort>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<ushort2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<ushort3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<ushort4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        template void pyrUp_gpu<short>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<short2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<short3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<short4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

        //template void pyrUp_gpu<int>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<int2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<int3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<int4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
#endif

        template void pyrUp_gpu<float>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        //template void pyrUp_gpu<float2>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<float3>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template void pyrUp_gpu<float4>(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device

#endif /* CUDA_DISABLER */
