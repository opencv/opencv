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

#ifndef OPENCV_CUDA_EMULATION_HPP_
#define OPENCV_CUDA_EMULATION_HPP_

#include "common.hpp"
#include "warp_reduce.hpp"

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    struct Emulation
    {

        static __device__ __forceinline__ int syncthreadsOr(int pred)
        {
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
                // just campilation stab
                return 0;
#else
                return __syncthreads_or(pred);
#endif
        }

        template<int CTA_SIZE>
        static __forceinline__ __device__ int Ballot(int predicate)
        {
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
            return __ballot(predicate);
#else
            __shared__ volatile int cta_buffer[CTA_SIZE];

            int tid = threadIdx.x;
            cta_buffer[tid] = predicate ? (1 << (tid & 31)) : 0;
            return warp_reduce(cta_buffer);
#endif
        }

        struct smem
        {
            enum { TAG_MASK = (1U << ( (sizeof(unsigned int) << 3) - 5U)) - 1U };

            template<typename T>
            static __device__ __forceinline__ T atomicInc(T* address, T val)
            {
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 120)
                T count;
                unsigned int tag = threadIdx.x << ( (sizeof(unsigned int) << 3) - 5U);
                do
                {
                    count = *address & TAG_MASK;
                    count = tag | (count + 1);
                    *address = count;
                } while (*address != count);

                return (count & TAG_MASK) - 1;
#else
                return ::atomicInc(address, val);
#endif
            }

            template<typename T>
            static __device__ __forceinline__ T atomicAdd(T* address, T val)
            {
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 120)
                T count;
                unsigned int tag = threadIdx.x << ( (sizeof(unsigned int) << 3) - 5U);
                do
                {
                    count = *address & TAG_MASK;
                    count = tag | (count + val);
                    *address = count;
                } while (*address != count);

                return (count & TAG_MASK) - val;
#else
                return ::atomicAdd(address, val);
#endif
            }

            template<typename T>
            static __device__ __forceinline__ T atomicMin(T* address, T val)
            {
#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 120)
                T count = ::min(*address, val);
                do
                {
                    *address = count;
                } while (*address > count);

                return count;
#else
                return ::atomicMin(address, val);
#endif
            }
        }; // struct cmem

        struct glob
        {
            static __device__ __forceinline__ int atomicAdd(int* address, int val)
            {
                return ::atomicAdd(address, val);
            }
            static __device__ __forceinline__ unsigned int atomicAdd(unsigned int* address, unsigned int val)
            {
                return ::atomicAdd(address, val);
            }
            static __device__ __forceinline__ float atomicAdd(float* address, float val)
            {
            #if __CUDA_ARCH__ >= 200
                return ::atomicAdd(address, val);
            #else
                int* address_as_i = (int*) address;
                int old = *address_as_i, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(val + __int_as_float(assumed)));
                } while (assumed != old);
                return __int_as_float(old);
            #endif
            }
            static __device__ __forceinline__ double atomicAdd(double* address, double val)
            {
            #if __CUDA_ARCH__ >= 130
                unsigned long long int* address_as_ull = (unsigned long long int*) address;
                unsigned long long int old = *address_as_ull, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
                } while (assumed != old);
                return __longlong_as_double(old);
            #else
                (void) address;
                (void) val;
                return 0.0;
            #endif
            }

            static __device__ __forceinline__ int atomicMin(int* address, int val)
            {
                return ::atomicMin(address, val);
            }
            static __device__ __forceinline__ float atomicMin(float* address, float val)
            {
            #if __CUDA_ARCH__ >= 120
                int* address_as_i = (int*) address;
                int old = *address_as_i, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fminf(val, __int_as_float(assumed))));
                } while (assumed != old);
                return __int_as_float(old);
            #else
                (void) address;
                (void) val;
                return 0.0f;
            #endif
            }
            static __device__ __forceinline__ double atomicMin(double* address, double val)
            {
            #if __CUDA_ARCH__ >= 130
                unsigned long long int* address_as_ull = (unsigned long long int*) address;
                unsigned long long int old = *address_as_ull, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
                } while (assumed != old);
                return __longlong_as_double(old);
            #else
                (void) address;
                (void) val;
                return 0.0;
            #endif
            }

            static __device__ __forceinline__ int atomicMax(int* address, int val)
            {
                return ::atomicMax(address, val);
            }
            static __device__ __forceinline__ float atomicMax(float* address, float val)
            {
            #if __CUDA_ARCH__ >= 120
                int* address_as_i = (int*) address;
                int old = *address_as_i, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
                } while (assumed != old);
                return __int_as_float(old);
            #else
                (void) address;
                (void) val;
                return 0.0f;
            #endif
            }
            static __device__ __forceinline__ double atomicMax(double* address, double val)
            {
            #if __CUDA_ARCH__ >= 130
                unsigned long long int* address_as_ull = (unsigned long long int*) address;
                unsigned long long int old = *address_as_ull, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
                } while (assumed != old);
                return __longlong_as_double(old);
            #else
                (void) address;
                (void) val;
                return 0.0;
            #endif
            }
        };
    }; //struct Emulation
}}} // namespace cv { namespace cuda { namespace cudev

//! @endcond

#endif /* OPENCV_CUDA_EMULATION_HPP_ */
