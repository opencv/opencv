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

#include <opencv2/cudevice/common.hpp>
#include <opencv2/cudevice/vec_traits.hpp>
#include <opencv2/cudevice/vec_math.hpp>
#include <opencv2/cudevice/limits.hpp>

namespace cv { namespace gpu {
    namespace device
    {
        template <typename D>
        __global__ void Bayer2BGR_8u(const PtrStepb src, PtrStepSz<D> dst, const bool blue_last, const bool start_with_green)
        {
            const int s_x = blockIdx.x * blockDim.x + threadIdx.x;
            int s_y = blockIdx.y * blockDim.y + threadIdx.y;

            if (s_y >= dst.rows || (s_x << 2) >= dst.cols)
                return;

            s_y = ::min(::max(s_y, 1), dst.rows - 2);

            uchar4 patch[3][3];
            patch[0][1] = ((const uchar4*) src.ptr(s_y - 1))[s_x];
            patch[0][0] = ((const uchar4*) src.ptr(s_y - 1))[::max(s_x - 1, 0)];
            patch[0][2] = ((const uchar4*) src.ptr(s_y - 1))[::min(s_x + 1, ((dst.cols + 3) >> 2) - 1)];

            patch[1][1] = ((const uchar4*) src.ptr(s_y))[s_x];
            patch[1][0] = ((const uchar4*) src.ptr(s_y))[::max(s_x - 1, 0)];
            patch[1][2] = ((const uchar4*) src.ptr(s_y))[::min(s_x + 1, ((dst.cols + 3) >> 2) - 1)];

            patch[2][1] = ((const uchar4*) src.ptr(s_y + 1))[s_x];
            patch[2][0] = ((const uchar4*) src.ptr(s_y + 1))[::max(s_x - 1, 0)];
            patch[2][2] = ((const uchar4*) src.ptr(s_y + 1))[::min(s_x + 1, ((dst.cols + 3) >> 2) - 1)];

            D res0 = VecTraits<D>::all(numeric_limits<uchar>::max());
            D res1 = VecTraits<D>::all(numeric_limits<uchar>::max());
            D res2 = VecTraits<D>::all(numeric_limits<uchar>::max());
            D res3 = VecTraits<D>::all(numeric_limits<uchar>::max());

            if ((s_y & 1) ^ start_with_green)
            {
                const int t0 = (patch[0][1].x + patch[2][1].x + 1) >> 1;
                const int t1 = (patch[1][0].w + patch[1][1].y + 1) >> 1;

                const int t2 = (patch[0][1].x + patch[0][1].z + patch[2][1].x + patch[2][1].z + 2) >> 2;
                const int t3 = (patch[0][1].y + patch[1][1].x + patch[1][1].z + patch[2][1].y + 2) >> 2;

                const int t4 = (patch[0][1].z + patch[2][1].z + 1) >> 1;
                const int t5 = (patch[1][1].y + patch[1][1].w + 1) >> 1;

                const int t6 = (patch[0][1].z + patch[0][2].x + patch[2][1].z + patch[2][2].x + 2) >> 2;
                const int t7 = (patch[0][1].w + patch[1][1].z + patch[1][2].x + patch[2][1].w + 2) >> 2;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = t1;
                    res0.y = patch[1][1].x;
                    res0.z = t0;

                    res1.x = patch[1][1].y;
                    res1.y = t3;
                    res1.z = t2;

                    res2.x = t5;
                    res2.y = patch[1][1].z;
                    res2.z = t4;

                    res3.x = patch[1][1].w;
                    res3.y = t7;
                    res3.z = t6;
                }
                else
                {
                    res0.x = t0;
                    res0.y = patch[1][1].x;
                    res0.z = t1;

                    res1.x = t2;
                    res1.y = t3;
                    res1.z = patch[1][1].y;

                    res2.x = t4;
                    res2.y = patch[1][1].z;
                    res2.z = t5;

                    res3.x = t6;
                    res3.y = t7;
                    res3.z = patch[1][1].w;
                }
            }
            else
            {
                const int t0 = (patch[0][0].w + patch[0][1].y + patch[2][0].w + patch[2][1].y + 2) >> 2;
                const int t1 = (patch[0][1].x + patch[1][0].w + patch[1][1].y + patch[2][1].x + 2) >> 2;

                const int t2 = (patch[0][1].y + patch[2][1].y + 1) >> 1;
                const int t3 = (patch[1][1].x + patch[1][1].z + 1) >> 1;

                const int t4 = (patch[0][1].y + patch[0][1].w + patch[2][1].y + patch[2][1].w + 2) >> 2;
                const int t5 = (patch[0][1].z + patch[1][1].y + patch[1][1].w + patch[2][1].z + 2) >> 2;

                const int t6 = (patch[0][1].w + patch[2][1].w + 1) >> 1;
                const int t7 = (patch[1][1].z + patch[1][2].x + 1) >> 1;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = patch[1][1].x;
                    res0.y = t1;
                    res0.z = t0;

                    res1.x = t3;
                    res1.y = patch[1][1].y;
                    res1.z = t2;

                    res2.x = patch[1][1].z;
                    res2.y = t5;
                    res2.z = t4;

                    res3.x = t7;
                    res3.y = patch[1][1].w;
                    res3.z = t6;
                }
                else
                {
                    res0.x = t0;
                    res0.y = t1;
                    res0.z = patch[1][1].x;

                    res1.x = t2;
                    res1.y = patch[1][1].y;
                    res1.z = t3;

                    res2.x = t4;
                    res2.y = t5;
                    res2.z = patch[1][1].z;

                    res3.x = t6;
                    res3.y = patch[1][1].w;
                    res3.z = t7;
                }
            }

            const int d_x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
            const int d_y = blockIdx.y * blockDim.y + threadIdx.y;

            dst(d_y, d_x) = res0;
            if (d_x + 1 < dst.cols)
                dst(d_y, d_x + 1) = res1;
            if (d_x + 2 < dst.cols)
                dst(d_y, d_x + 2) = res2;
            if (d_x + 3 < dst.cols)
                dst(d_y, d_x + 3) = res3;
        }

        template <typename D>
        __global__ void Bayer2BGR_16u(const PtrStepb src, PtrStepSz<D> dst, const bool blue_last, const bool start_with_green)
        {
            const int s_x = blockIdx.x * blockDim.x + threadIdx.x;
            int s_y = blockIdx.y * blockDim.y + threadIdx.y;

            if (s_y >= dst.rows || (s_x << 1) >= dst.cols)
                return;

            s_y = ::min(::max(s_y, 1), dst.rows - 2);

            ushort2 patch[3][3];
            patch[0][1] = ((const ushort2*) src.ptr(s_y - 1))[s_x];
            patch[0][0] = ((const ushort2*) src.ptr(s_y - 1))[::max(s_x - 1, 0)];
            patch[0][2] = ((const ushort2*) src.ptr(s_y - 1))[::min(s_x + 1, ((dst.cols + 1) >> 1) - 1)];

            patch[1][1] = ((const ushort2*) src.ptr(s_y))[s_x];
            patch[1][0] = ((const ushort2*) src.ptr(s_y))[::max(s_x - 1, 0)];
            patch[1][2] = ((const ushort2*) src.ptr(s_y))[::min(s_x + 1, ((dst.cols + 1) >> 1) - 1)];

            patch[2][1] = ((const ushort2*) src.ptr(s_y + 1))[s_x];
            patch[2][0] = ((const ushort2*) src.ptr(s_y + 1))[::max(s_x - 1, 0)];
            patch[2][2] = ((const ushort2*) src.ptr(s_y + 1))[::min(s_x + 1, ((dst.cols + 1) >> 1) - 1)];

            D res0 = VecTraits<D>::all(numeric_limits<ushort>::max());
            D res1 = VecTraits<D>::all(numeric_limits<ushort>::max());

            if ((s_y & 1) ^ start_with_green)
            {
                const int t0 = (patch[0][1].x + patch[2][1].x + 1) >> 1;
                const int t1 = (patch[1][0].y + patch[1][1].y + 1) >> 1;

                const int t2 = (patch[0][1].x + patch[0][2].x + patch[2][1].x + patch[2][2].x + 2) >> 2;
                const int t3 = (patch[0][1].y + patch[1][1].x + patch[1][2].x + patch[2][1].y + 2) >> 2;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = t1;
                    res0.y = patch[1][1].x;
                    res0.z = t0;

                    res1.x = patch[1][1].y;
                    res1.y = t3;
                    res1.z = t2;
                }
                else
                {
                    res0.x = t0;
                    res0.y = patch[1][1].x;
                    res0.z = t1;

                    res1.x = t2;
                    res1.y = t3;
                    res1.z = patch[1][1].y;
                }
            }
            else
            {
                const int t0 = (patch[0][0].y + patch[0][1].y + patch[2][0].y + patch[2][1].y + 2) >> 2;
                const int t1 = (patch[0][1].x + patch[1][0].y + patch[1][1].y + patch[2][1].x + 2) >> 2;

                const int t2 = (patch[0][1].y + patch[2][1].y + 1) >> 1;
                const int t3 = (patch[1][1].x + patch[1][2].x + 1) >> 1;

                if ((s_y & 1) ^ blue_last)
                {
                    res0.x = patch[1][1].x;
                    res0.y = t1;
                    res0.z = t0;

                    res1.x = t3;
                    res1.y = patch[1][1].y;
                    res1.z = t2;
                }
                else
                {
                    res0.x = t0;
                    res0.y = t1;
                    res0.z = patch[1][1].x;

                    res1.x = t2;
                    res1.y = patch[1][1].y;
                    res1.z = t3;
                }
            }

            const int d_x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
            const int d_y = blockIdx.y * blockDim.y + threadIdx.y;

            dst(d_y, d_x) = res0;
            if (d_x + 1 < dst.cols)
                dst(d_y, d_x + 1) = res1;
        }

        template <int cn>
        void Bayer2BGR_8u_gpu(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream)
        {
            typedef typename TypeVec<uchar, cn>::vec_type dst_t;

            const dim3 block(32, 8);
            const dim3 grid(divUp(dst.cols, 4 * block.x), divUp(dst.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(Bayer2BGR_8u<dst_t>, cudaFuncCachePreferL1) );

            Bayer2BGR_8u<dst_t><<<grid, block, 0, stream>>>(src, (PtrStepSz<dst_t>)dst, blue_last, start_with_green);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
        template <int cn>
        void Bayer2BGR_16u_gpu(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream)
        {
            typedef typename TypeVec<ushort, cn>::vec_type dst_t;

            const dim3 block(32, 8);
            const dim3 grid(divUp(dst.cols, 2 * block.x), divUp(dst.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(Bayer2BGR_16u<dst_t>, cudaFuncCachePreferL1) );

            Bayer2BGR_16u<dst_t><<<grid, block, 0, stream>>>(src, (PtrStepSz<dst_t>)dst, blue_last, start_with_green);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void Bayer2BGR_8u_gpu<3>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template void Bayer2BGR_8u_gpu<4>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template void Bayer2BGR_16u_gpu<3>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template void Bayer2BGR_16u_gpu<4>(PtrStepSzb src, PtrStepSzb dst, bool blue_last, bool start_with_green, cudaStream_t stream);
    }
}}

#endif /* CUDA_DISABLER */