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

#ifndef __OPENCV_GPU_KENELRS_HPP__
#define __OPENCV_GPU_KENELRS_HPP__

namespace cv
{
	namespace gpu
	{
		namespace device
		{
			struct Grid1D
			{
				static __forceinline__ __device__ int STRIDE() { return  gridDim.x * blockDim.x; }
				static __forceinline__ __device__ int SHIFT()  { return blockIdx.x * blockDim.x + threadIdx.x; }
			};

			struct Block1D
			{
				static __forceinline__ __device__ int STRIDE() { return blockDim.x; }
				static __forceinline__ __device__ int SHIFT()  { return threadIdx.x; }			
			};

			struct Warp
			{
				static __forceinline__ __device__ int STRIDE() { return warpSize };            
				static __forceinline__ __device__ int SHIFT()  { return threadIdx.x & (warpSize - 1); }			
			};

			template <class Worker, typename T>
			__forceinline__ __device__ void Copy(const T* in, T *out, int length)
			{
				int STRIDE = Worker::STRIDE();
				int idx    = Worker::SHIFT();				
				
				for (; idx < length; idx += STRIDE) 
	                out[idx] = in[idx];
			}

			template <class Worker, typename InIter, typename OutIter>
			__forceinline__ __device__ void Copy(InIter beg, InIter end, OutIter out)
			{
				int STRIDE = Worker::STRIDE();
				int SHIFT  = Worker::SHIFT();
				
				beg += SHIFT;
				out += SHIFT;

				for (; beg < end; beg += STRIDE, out += STRIDE) 
					*out = *beg;				
			}		

			 template <class Worker, typename T>
			__forceinline__ __device__ void Yota(T* out, int beg, int end)
			{	            				
				int STRIDE = Worker::STRIDE();
				int SHIFT  = Worker::SHIFT();

				int idx    = SHIFT;
				int cur    = beg + SHIFT;
				int length = end - beg;
				
				for (; idx < length; idx += STRIDE, cur += STRIDE)
					out[idx] = cur;				
			}

			template <class Worker, typename OutIter>
			__forceinline__ __device__ void Yota(OutIter beg, OutIter end, int val)
			{
				int STRIDE = Worker::STRIDE();
				int SHIFT  = Worker::SHIFT();

				beg += SHIFT;
				val += SHIFT;
				
				for (; beg < end; beg += STRIDE, val += STRIDE)
					*beg = val;
			}
		}
	}
}

#endif /* __OPENCV_GPU_KENELRS_HPP__ */