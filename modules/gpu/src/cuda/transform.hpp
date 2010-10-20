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

#ifndef __OPENCV_GPU_TRANSFORM_HPP__
#define __OPENCV_GPU_TRANSFORM_HPP__

#include "cuda_shared.hpp"
#include "saturate_cast.hpp"
#include "vecmath.hpp"

namespace cv { namespace gpu { namespace algo_krnls
{
    template <typename T, typename D, typename UnOp>
    static __global__ void transform(const T* src, size_t src_step, 
                                     D* dst, size_t dst_step, int width, int height, UnOp op)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < width && y < height)
        {
            T src_data = src[y * src_step + x];
            dst[y * dst_step + x] = op(src_data, x, y);
        }
    }
    template <typename T1, typename T2, typename D, typename BinOp>
    static __global__ void transform(const T1* src1, size_t src1_step, const T2* src2, size_t src2_step, 
                                     D* dst, size_t dst_step, int width, int height, BinOp op)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < width && y < height)
        {
            T1 src1_data = src1[y * src1_step + x];
            T2 src2_data = src2[y * src2_step + x];
            dst[y * dst_step + x] = op(src1_data, src2_data, x, y);
        }
    }
}}}

namespace cv 
{ 
    namespace gpu 
    {
        template <typename T, typename D, typename UnOp>
        static void transform(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, UnOp op, cudaStream_t stream)
        {
            dim3 threads(16, 16, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(src.cols, threads.x);
            grid.y = divUp(src.rows, threads.y);        

            algo_krnls::transform<<<grid, threads, 0, stream>>>(src.ptr, src.elem_step, 
                dst.ptr, dst.elem_step, src.cols, src.rows, op);

            if (stream == 0)
                cudaSafeCall( cudaThreadSynchronize() );
        }
        template <typename T1, typename T2, typename D, typename BinOp>
        static void transform(const DevMem2D_<T1>& src1, const DevMem2D_<T2>& src2, const DevMem2D_<D>& dst, BinOp op, cudaStream_t stream)
        {
            dim3 threads(16, 16, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(src1.cols, threads.x);
            grid.y = divUp(src1.rows, threads.y);        

            algo_krnls::transform<<<grid, threads, 0, stream>>>(src1.ptr, src1.elem_step, 
                src2.ptr, src2.elem_step, dst.ptr, dst.elem_step, src1.cols, src1.rows, op);

            if (stream == 0)
                cudaSafeCall( cudaThreadSynchronize() );
        }
    }
}

#endif // __OPENCV_GPU_TRANSFORM_HPP__
