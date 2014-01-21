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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/emulation.hpp"
#include "opencv2/gpu/device/dynamic_smem.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        __device__ static int g_counter;

        ////////////////////////////////////////////////////////////////////////
        // linesAccum

        __global__ void linesAccumGlobal(const unsigned int* list, const int count, PtrStepi accum, const float irho, const float theta, const int numrho)
        {
            const int n = blockIdx.x;
            const float ang = n * theta;

            float sinVal;
            float cosVal;
            sincosf(ang, &sinVal, &cosVal);
            sinVal *= irho;
            cosVal *= irho;

            const int shift = (numrho - 1) / 2;

            int* accumRow = accum.ptr(n + 1);
            for (int i = threadIdx.x; i < count; i += blockDim.x)
            {
                const unsigned int val = list[i];

                const int x = (val & 0xFFFF);
                const int y = (val >> 16) & 0xFFFF;

                int r = __float2int_rn(x * cosVal + y * sinVal);
                r += shift;

                ::atomicAdd(accumRow + r + 1, 1);
            }
        }

        __global__ void linesAccumShared(const unsigned int* list, const int count, PtrStepi accum, const float irho, const float theta, const int numrho)
        {
            int* smem = DynamicSharedMem<int>();

            for (int i = threadIdx.x; i < numrho + 1; i += blockDim.x)
                smem[i] = 0;

            __syncthreads();

            const int n = blockIdx.x;
            const float ang = n * theta;

            float sinVal;
            float cosVal;
            sincosf(ang, &sinVal, &cosVal);
            sinVal *= irho;
            cosVal *= irho;

            const int shift = (numrho - 1) / 2;

            for (int i = threadIdx.x; i < count; i += blockDim.x)
            {
                const unsigned int val = list[i];

                const int x = (val & 0xFFFF);
                const int y = (val >> 16) & 0xFFFF;

                int r = __float2int_rn(x * cosVal + y * sinVal);
                r += shift;

                Emulation::smem::atomicAdd(&smem[r + 1], 1);
            }

            __syncthreads();

            int* accumRow = accum.ptr(n + 1);
            for (int i = threadIdx.x; i < numrho + 1; i += blockDim.x)
                accumRow[i] = smem[i];
        }

        void linesAccum_gpu(const unsigned int* list, int count, PtrStepSzi accum, float rho, float theta, size_t sharedMemPerBlock, bool has20)
        {
            const dim3 block(has20 ? 1024 : 512);
            const dim3 grid(accum.rows - 2);

            size_t smemSize = (accum.cols - 1) * sizeof(int);

            if (smemSize < sharedMemPerBlock - 1000)
                linesAccumShared<<<grid, block, smemSize>>>(list, count, accum, 1.0f / rho, theta, accum.cols - 2);
            else
                linesAccumGlobal<<<grid, block>>>(list, count, accum, 1.0f / rho, theta, accum.cols - 2);

            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // linesGetResult

        __global__ void linesGetResult(const PtrStepSzi accum, float2* out, int* votes, const int maxSize, const float rho, const float theta, const int threshold, const int numrho)
        {
            const int r = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = blockIdx.y * blockDim.y + threadIdx.y;

            if (r >= accum.cols - 2 || n >= accum.rows - 2)
                return;

            const int curVotes = accum(n + 1, r + 1);

            if (curVotes > threshold &&
                curVotes >  accum(n + 1, r) &&
                curVotes >= accum(n + 1, r + 2) &&
                curVotes >  accum(n, r + 1) &&
                curVotes >= accum(n + 2, r + 1))
            {
                const float radius = (r - (numrho - 1) * 0.5f) * rho;
                const float angle = n * theta;

                const int ind = ::atomicAdd(&g_counter, 1);
                if (ind < maxSize)
                {
                    out[ind] = make_float2(radius, angle);
                    votes[ind] = curVotes;
                }
            }
        }

        int linesGetResult_gpu(PtrStepSzi accum, float2* out, int* votes, int maxSize, float rho, float theta, int threshold, bool doSort)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols - 2, block.x), divUp(accum.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(linesGetResult, cudaFuncCachePreferL1) );

            linesGetResult<<<grid, block>>>(accum, out, votes, maxSize, rho, theta, threshold, accum.cols - 2);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            if (doSort && totalCount > 0)
            {
                thrust::device_ptr<float2> outPtr(out);
                thrust::device_ptr<int> votesPtr(votes);
                thrust::sort_by_key(votesPtr, votesPtr + totalCount, outPtr, thrust::greater<int>());
            }

            return totalCount;
        }
    }
}}}


#endif /* CUDA_DISABLER */
