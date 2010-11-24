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

namespace cv { namespace gpu { namespace device
{
    //! Mask accessor
    template<class T> struct MaskReader_
    {
        PtrStep_<T> mask;
        explicit MaskReader_(PtrStep_<T> mask): mask(mask) {}                

        __device__ bool operator()(int y, int x) const { return mask.ptr(y)[x]; }
    };

    //! Stub mask accessor
    struct NoMask 
    {
        __device__ bool operator()(int y, int x) const { return true; } 
    };

    //! Transform kernels

    template <typename T, typename D, typename Mask, typename UnOp>
    static __global__ void transform(const DevMem2D_<T> src, PtrStep_<D> dst, const Mask mask, UnOp op)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < src.cols && y < src.rows && mask(y, x))
        {
            T src_data = src.ptr(y)[x];
            dst.ptr(y)[x] = op(src_data);
        }
    }

    template <typename T1, typename T2, typename D, typename Mask, typename BinOp>
    static __global__ void transform(const DevMem2D_<T1> src1, const PtrStep_<T2> src2, PtrStep_<D> dst, const Mask mask, BinOp op)
    {
		const int x = blockDim.x * blockIdx.x + threadIdx.x;
		const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < src1.cols && y < src1.rows && mask(y, x))
        {
            T1 src1_data = src1.ptr(y)[x];
            T2 src2_data = src2.ptr(y)[x];
            dst.ptr(y)[x] = op(src1_data, src2_data);
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

            device::transform<T, D, UnOp><<<grid, threads, 0, stream>>>(src, dst, device::NoMask(), op);

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

            device::transform<T1, T2, D><<<grid, threads, 0, stream>>>(src1, src2, dst, device::NoMask(), op);

            if (stream == 0)
                cudaSafeCall( cudaThreadSynchronize() );            
        }
    }
}

#endif // __OPENCV_GPU_TRANSFORM_HPP__
