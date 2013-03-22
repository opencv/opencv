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
#include "opencv2/gpu/device/vec_traits.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace video_encoding
    {
        __device__ __forceinline__ void rgbtoy(const uchar b, const uchar g, const uchar r, uchar& y)
        {
            y = static_cast<uchar>(((int)(30 * r) + (int)(59 * g) + (int)(11 * b)) / 100);
        }

        __device__ __forceinline__ void rgbtoyuv(const uchar b, const uchar g, const uchar r, uchar& y, uchar& u, uchar& v)
        {
            rgbtoy(b, g, r, y);
            u = static_cast<uchar>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
            v = static_cast<uchar>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
        }

        __global__ void Gray_to_YV12(const PtrStepSzb src, PtrStepb dst)
        {
            const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
            const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

            if (x + 1 >= src.cols || y + 1 >= src.rows)
                return;

            // get pointers to the data
            const size_t planeSize = src.rows * dst.step;
            PtrStepb y_plane(dst.data, dst.step);
            PtrStepb u_plane(y_plane.data + planeSize, dst.step / 2);
            PtrStepb v_plane(u_plane.data + (planeSize / 4), dst.step / 2);

            uchar pix;
            uchar y_val, u_val, v_val;

            pix = src(y, x);
            rgbtoy(pix, pix, pix, y_val);
            y_plane(y, x) = y_val;

            pix = src(y, x + 1);
            rgbtoy(pix, pix, pix, y_val);
            y_plane(y, x + 1) = y_val;

            pix = src(y + 1, x);
            rgbtoy(pix, pix, pix, y_val);
            y_plane(y + 1, x) = y_val;

            pix = src(y + 1, x + 1);
            rgbtoyuv(pix, pix, pix, y_val, u_val, v_val);
            y_plane(y + 1, x + 1) = y_val;
            u_plane(y / 2, x / 2) = u_val;
            v_plane(y / 2, x / 2) = v_val;
        }

        template <typename T>
        __global__ void BGR_to_YV12(const PtrStepSz<T> src, PtrStepb dst)
        {
            const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
            const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

            if (x + 1 >= src.cols || y + 1 >= src.rows)
                return;

            // get pointers to the data
            const size_t planeSize = src.rows * dst.step;
            PtrStepb y_plane(dst.data, dst.step);
            PtrStepb u_plane(y_plane.data + planeSize, dst.step / 2);
            PtrStepb v_plane(u_plane.data + (planeSize / 4), dst.step / 2);

            T pix;
            uchar y_val, u_val, v_val;

            pix = src(y, x);
            rgbtoy(pix.z, pix.y, pix.x, y_val);
            y_plane(y, x) = y_val;

            pix = src(y, x + 1);
            rgbtoy(pix.z, pix.y, pix.x, y_val);
            y_plane(y, x + 1) = y_val;

            pix = src(y + 1, x);
            rgbtoy(pix.z, pix.y, pix.x, y_val);
            y_plane(y + 1, x) = y_val;

            pix = src(y + 1, x + 1);
            rgbtoyuv(pix.z, pix.y, pix.x, y_val, u_val, v_val);
            y_plane(y + 1, x + 1) = y_val;
            u_plane(y / 2, x / 2) = u_val;
            v_plane(y / 2, x / 2) = v_val;
        }

        void Gray_to_YV12_caller(const PtrStepSzb src, PtrStepb dst)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));

            Gray_to_YV12<<<grid, block>>>(src, dst);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
        template <int cn>
        void BGR_to_YV12_caller(const PtrStepSzb src, PtrStepb dst)
        {
            typedef typename TypeVec<uchar, cn>::vec_type src_t;

            dim3 block(32, 8);
            dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));

            BGR_to_YV12<<<grid, block>>>(static_cast< PtrStepSz<src_t> >(src), dst);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        void YV12_gpu(const PtrStepSzb src, int cn, PtrStepSzb dst)
        {
            typedef void (*func_t)(const PtrStepSzb src, PtrStepb dst);

            static const func_t funcs[] =
            {
                0, Gray_to_YV12_caller, 0, BGR_to_YV12_caller<3>, BGR_to_YV12_caller<4>
            };

            funcs[cn](src, dst);
        }
    }
}}}

#endif /* CUDA_DISABLER */
