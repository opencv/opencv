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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace imgproc 
    {
        template <class SrcPtr, typename D> __global__ void pyrUp(const SrcPtr src, DevMem2D_<D> dst)
        {
            typedef typename SrcPtr::elem_type src_t;
            typedef typename TypeVec<float, VecTraits<D>::cn>::vec_type sum_t;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            __shared__ sum_t s_srcPatch[10][10];
            __shared__ sum_t s_dstPatch[20][16];

            if (threadIdx.x < 10 && threadIdx.y < 10)
            {
                const int srcx = static_cast<int>((blockIdx.x * blockDim.x) / 2 + threadIdx.x) - 1;
                const int srcy = static_cast<int>((blockIdx.y * blockDim.y) / 2 + threadIdx.y) - 1;

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
                dst(y, x) = saturate_cast<D>(4.0f * sum);
        }

        template <typename T, template <typename> class B> void pyrUp_caller(const DevMem2D_<T>& src, const DevMem2D_<T>& dst, cudaStream_t stream)
        {
            const dim3 block(16, 16);
            const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            B<T> b(src.rows, src.cols);
            BorderReader< PtrStep<T>, B<T> > srcReader(src, b);

            pyrUp<<<grid, block, 0, stream>>>(srcReader, dst);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <typename T, int cn> void pyrUp_gpu(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream)
        {
            typedef typename TypeVec<T, cn>::vec_type type;

            typedef void (*caller_t)(const DevMem2D_<type>& src, const DevMem2D_<type>& dst, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                pyrUp_caller<type, BrdReflect101>, pyrUp_caller<type, BrdReplicate>, pyrUp_caller<type, BrdConstant>, pyrUp_caller<type, BrdReflect>, pyrUp_caller<type, BrdWrap>
            };

            callers[borderType](static_cast< DevMem2D_<type> >(src), static_cast< DevMem2D_<type> >(dst), stream);
        }

        template void pyrUp_gpu<uchar, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<uchar, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<uchar, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<uchar, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

        template void pyrUp_gpu<schar, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<schar, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<schar, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<schar, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

        template void pyrUp_gpu<ushort, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<ushort, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<ushort, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<ushort, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

        template void pyrUp_gpu<short, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<short, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<short, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<short, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

        template void pyrUp_gpu<int, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<int, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<int, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<int, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);

        template void pyrUp_gpu<float, 1>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<float, 2>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<float, 3>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
        template void pyrUp_gpu<float, 4>(const DevMem2Db& src, const DevMem2Db& dst, int borderType, cudaStream_t stream);
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device
