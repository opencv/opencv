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

#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/vec_traits.hpp>

namespace cv { namespace gpu {
    namespace device
    {
        template <class SrcPtr, typename T>
        __global__ void Bayer2BGR(const SrcPtr src, PtrStep_<T> dst, const int width, const int height, const bool glob_blue_last, const bool glob_start_with_green)
        {
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y >= height)
                return;

            const bool blue_last = (y & 1) ? !glob_blue_last : glob_blue_last;
            const bool start_with_green = (y & 1) ? !glob_start_with_green : glob_start_with_green;

            int x = tx * 2;

            if (start_with_green)
            {
                --x;

                if (tx == 0)
                {
                    const int t0 = (src(y, 1) + src(y + 2, 1) + 1) >> 1;
                    const int t1 = (src(y + 1, 0) + src(y + 1, 2) + 1) >> 1;

                    T res;
                    res.x = blue_last ? t0 : t1;
                    res.y = src(y + 1, 1);
                    res.z = blue_last ? t1 : t0;

                    dst(y + 1, 0) = dst(y + 1, 1) = res;
                    if (y == 0)
                    {
                        dst(0, 0) = dst(0, 1) = res;
                    }
                    else if (y == height - 1)
                    {
                        dst(height + 1, 0) = dst(height + 1, 1) = res;
                    }
                }
            }

            if (x >= 0 && x <= width - 2)
            {
                const int t0 = (src(y, x) + src(y, x + 2) + src(y + 2, x) + src(y + 2, x + 2) + 2) >> 2;
                const int t1 = (src(y, x + 1) + src(y + 1, x) + src(y + 1, x + 2) + src(y + 2, x + 1) + 2) >> 2;

                const int t2 = (src(y, x + 2) + src(y + 2, x + 2) + 1) >> 1;
                const int t3 = (src(y + 1, x + 1) + src(y + 1, x + 3) + 1) >> 1;

                T res1, res2;

                if (blue_last)
                {
                    res1.x = t0;
                    res1.y = t1;
                    res1.z = src(y + 1, x + 1);

                    res2.x = t2;
                    res2.y = src(y + 1, x + 2);
                    res2.z = t3;
                }
                else
                {
                    res1.x = src(y + 1, x + 1);
                    res1.y = t1;
                    res1.z = t0;

                    res2.x = t3;
                    res2.y = src(y + 1, x + 2);
                    res2.z = t2;
                }

                dst(y + 1, x + 1) = res1;
                dst(y + 1, x + 2) = res2;

                if (y == 0)
                {
                    dst(0, x + 1) = res1;
                    dst(0, x + 2) = res2;

                    if (x == 0)
                    {
                        dst(0, 0) = res1;
                    }
                    else if (x == width - 2)
                    {
                        dst(0, width + 1) = res2;
                    }
                }
                else if (y == height - 1)
                {
                    dst(height + 1, x + 1) = res1;
                    dst(height + 1, x + 2) = res2;

                    if (x == 0)
                    {
                        dst(height + 1, 0) = res1;
                    }
                    else if (x == width - 2)
                    {
                        dst(height + 1, width + 1) = res2;
                    }
                }

                if (x == 0)
                {
                    dst(y + 1, 0) = res1;
                }
                else if (x == width - 2)
                {
                    dst(y + 1, width + 1) = res2;
                }
            }
            else if (x == width - 1)
            {
                const int t0 = (src(y, x) + src(y, x + 2) + src(y + 2, x) + src(y + 2, x + 2) + 2) >> 2;
                const int t1 = (src(y, x + 1) + src(y + 1, x) + src(y + 1, x + 2) + src(y + 2, x + 1) + 2) >> 2;

                T res;
                res.x = blue_last ? t0 : src(y + 1, x + 1);
                res.y = t1;
                res.z = blue_last ? src(y + 1, x + 1) : t0;

                dst(y + 1, x + 1) = dst(y + 1, x + 2) = res;
                if (y == 0)
                {
                    dst(0, x + 1) = dst(0, x + 2) = res;
                }
                else if (y == height - 1)
                {
                    dst(height + 1, x + 1) = dst(height + 1, x + 2) = res;
                }
            }
        }

        template <typename T, int cn>
        void Bayer2BGR_gpu(DevMem2Db src, DevMem2Db dst, bool blue_last, bool start_with_green, cudaStream_t stream)
        {
            typedef typename TypeVec<T, cn>::vec_type dst_t;

            const int width = src.cols - 2;
            const int height = src.rows - 2;

            const dim3 total(divUp(width, 2), height);

            const dim3 block(32, 8);
            const dim3 grid(divUp(total.x, block.x), divUp(total.y, block.y));

            Bayer2BGR<PtrStep_<T>, dst_t><<<grid, block, 0, stream>>>((DevMem2D_<T>)src, (DevMem2D_<dst_t>)dst, width, height, blue_last, start_with_green);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void Bayer2BGR_gpu<uchar, 3>(DevMem2Db src, DevMem2Db dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template void Bayer2BGR_gpu<uchar, 4>(DevMem2Db src, DevMem2Db dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template void Bayer2BGR_gpu<ushort, 3>(DevMem2Db src, DevMem2Db dst, bool blue_last, bool start_with_green, cudaStream_t stream);
        template void Bayer2BGR_gpu<ushort, 4>(DevMem2Db src, DevMem2Db dst, bool blue_last, bool start_with_green, cudaStream_t stream);
    }
}}
