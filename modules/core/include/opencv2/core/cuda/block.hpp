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

#ifndef OPENCV_CUDA_DEVICE_BLOCK_HPP
#define OPENCV_CUDA_DEVICE_BLOCK_HPP

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    struct Block
    {
        static __device__ __forceinline__ unsigned int id()
        {
            return blockIdx.x;
        }

        static __device__ __forceinline__ unsigned int stride()
        {
            return blockDim.x * blockDim.y * blockDim.z;
        }

        static __device__ __forceinline__ void sync()
        {
            __syncthreads();
        }

        static __device__ __forceinline__ int flattenedThreadId()
        {
            return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        }

        template<typename It, typename T>
        static __device__ __forceinline__ void fill(It beg, It end, const T& value)
        {
            int STRIDE = stride();
            It t = beg + flattenedThreadId();

            for(; t < end; t += STRIDE)
                *t = value;
        }

        template<typename OutIt, typename T>
        static __device__ __forceinline__ void yota(OutIt beg, OutIt end, T value)
        {
            int STRIDE = stride();
            int tid = flattenedThreadId();
            value += tid;

            for(OutIt t = beg + tid; t < end; t += STRIDE, value += STRIDE)
                *t = value;
        }

        template<typename InIt, typename OutIt>
        static __device__ __forceinline__ void copy(InIt beg, InIt end, OutIt out)
        {
            int STRIDE = stride();
            InIt  t = beg + flattenedThreadId();
            OutIt o = out + (t - beg);

            for(; t < end; t += STRIDE, o += STRIDE)
                *o = *t;
        }

        template<typename InIt, typename OutIt, class UnOp>
        static __device__ __forceinline__ void transform(InIt beg, InIt end, OutIt out, UnOp op)
        {
            int STRIDE = stride();
            InIt  t = beg + flattenedThreadId();
            OutIt o = out + (t - beg);

            for(; t < end; t += STRIDE, o += STRIDE)
                *o = op(*t);
        }

        template<typename InIt1, typename InIt2, typename OutIt, class BinOp>
        static __device__ __forceinline__ void transform(InIt1 beg1, InIt1 end1, InIt2 beg2, OutIt out, BinOp op)
        {
            int STRIDE = stride();
            InIt1 t1 = beg1 + flattenedThreadId();
            InIt2 t2 = beg2 + flattenedThreadId();
            OutIt o  = out + (t1 - beg1);

            for(; t1 < end1; t1 += STRIDE, t2 += STRIDE, o += STRIDE)
                *o = op(*t1, *t2);
        }

        template<int CTA_SIZE, typename T, class BinOp>
        static __device__ __forceinline__ void reduce(volatile T* buffer, BinOp op)
        {
            int tid = flattenedThreadId();
            T val =  buffer[tid];

            if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]); __syncthreads(); }
            if (CTA_SIZE >=  512) { if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]); __syncthreads(); }
            if (CTA_SIZE >=  256) { if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]); __syncthreads(); }
            if (CTA_SIZE >=  128) { if (tid <  64) buffer[tid] = val = op(val, buffer[tid +  64]); __syncthreads(); }

            if (tid < 32)
            {
                if (CTA_SIZE >=   64) { buffer[tid] = val = op(val, buffer[tid +  32]); }
                if (CTA_SIZE >=   32) { buffer[tid] = val = op(val, buffer[tid +  16]); }
                if (CTA_SIZE >=   16) { buffer[tid] = val = op(val, buffer[tid +   8]); }
                if (CTA_SIZE >=    8) { buffer[tid] = val = op(val, buffer[tid +   4]); }
                if (CTA_SIZE >=    4) { buffer[tid] = val = op(val, buffer[tid +   2]); }
                if (CTA_SIZE >=    2) { buffer[tid] = val = op(val, buffer[tid +   1]); }
            }
        }

        template<int CTA_SIZE, typename T, class BinOp>
        static __device__ __forceinline__ T reduce(volatile T* buffer, T init, BinOp op)
        {
            int tid = flattenedThreadId();
            T val =  buffer[tid] = init;
            __syncthreads();

            if (CTA_SIZE >= 1024) { if (tid < 512) buffer[tid] = val = op(val, buffer[tid + 512]); __syncthreads(); }
            if (CTA_SIZE >=  512) { if (tid < 256) buffer[tid] = val = op(val, buffer[tid + 256]); __syncthreads(); }
            if (CTA_SIZE >=  256) { if (tid < 128) buffer[tid] = val = op(val, buffer[tid + 128]); __syncthreads(); }
            if (CTA_SIZE >=  128) { if (tid <  64) buffer[tid] = val = op(val, buffer[tid +  64]); __syncthreads(); }

            if (tid < 32)
            {
                if (CTA_SIZE >=   64) { buffer[tid] = val = op(val, buffer[tid +  32]); }
                if (CTA_SIZE >=   32) { buffer[tid] = val = op(val, buffer[tid +  16]); }
                if (CTA_SIZE >=   16) { buffer[tid] = val = op(val, buffer[tid +   8]); }
                if (CTA_SIZE >=    8) { buffer[tid] = val = op(val, buffer[tid +   4]); }
                if (CTA_SIZE >=    4) { buffer[tid] = val = op(val, buffer[tid +   2]); }
                if (CTA_SIZE >=    2) { buffer[tid] = val = op(val, buffer[tid +   1]); }
            }
            __syncthreads();
            return buffer[0];
        }

        template <typename T, class BinOp>
        static __device__ __forceinline__ void reduce_n(T* data, unsigned int n, BinOp op)
        {
            int ftid = flattenedThreadId();
            int sft = stride();

            if (sft < n)
            {
                for (unsigned int i = sft + ftid; i < n; i += sft)
                    data[ftid] = op(data[ftid], data[i]);

                __syncthreads();

                n = sft;
            }

            while (n > 1)
            {
                unsigned int half = n/2;

                if (ftid < half)
                    data[ftid] = op(data[ftid], data[n - ftid - 1]);

                __syncthreads();

                n = n - half;
            }
        }
    };
}}}

//! @endcond

#endif /* OPENCV_CUDA_DEVICE_BLOCK_HPP */
