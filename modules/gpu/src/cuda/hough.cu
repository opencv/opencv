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
// any express or bpied warranties, including, but not limited to, the bpied
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

#include <thrust/sort.h>
#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/emulation.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        __device__ unsigned int g_counter;

        const int PIXELS_PER_THREAD = 16;

        __global__ void buildPointList(const DevMem2Db src, unsigned int* list)
        {
            __shared__ unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
            __shared__ unsigned int s_qsize[4];
            __shared__ unsigned int s_start[4];

            const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y >= src.rows)
                return;

            if (threadIdx.x == 0)
                s_qsize[threadIdx.y] = 0;

            __syncthreads();

            // fill the queue
            for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < src.cols; ++i, xx += blockDim.x)
            {
                if (src(y, xx))
                {
                    const unsigned int val = (y << 16) | xx;
                    int qidx = Emulation::smem::atomicInc(&s_qsize[threadIdx.y], (unsigned int)(-1));
                    s_queues[threadIdx.y][qidx] = val;
                }
            }

            __syncthreads();

            // let one thread reserve the space required in the global list
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                // find how many items are stored in each list
                unsigned int total_size = 0;
                for (int i = 0; i < blockDim.y; ++i)
                {
                    s_start[i] = total_size;
                    total_size += s_qsize[i];
                }

                //calculate the offset in the global list
                const unsigned int global_offset = atomicAdd(&g_counter, total_size);
                for (int i = 0; i < blockDim.y; ++i)
                    s_start[i] += global_offset;
            }

            __syncthreads();

            // copy local queues to global queue
            const unsigned int qsize = s_qsize[threadIdx.y];
            for(int i = threadIdx.x; i < qsize; i += blockDim.x)
            {
                unsigned int val = s_queues[threadIdx.y][i];
                list[s_start[threadIdx.y] + i] = val;
            }
        }

        unsigned int buildPointList_gpu(DevMem2Db src, unsigned int* list)
        {
            void* counter_ptr;
            cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, g_counter) );

            cudaSafeCall( cudaMemset(counter_ptr, 0, sizeof(unsigned int)) );

            const dim3 block(32, 4);
            const dim3 grid(divUp(src.cols, block.x * PIXELS_PER_THREAD), divUp(src.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(buildPointList, cudaFuncCachePreferShared) );

            buildPointList<<<grid, block>>>(src, list);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            unsigned int total_count;
            cudaSafeCall( cudaMemcpy(&total_count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

            return total_count;
        }

        __global__ void linesAccum(const unsigned int* list, const unsigned int count, PtrStep_<unsigned int> accum,
                                   const float irho, const float theta, const int numrho)
        {
            extern __shared__ unsigned int smem[];

            for (int i = threadIdx.x; i < numrho; i += blockDim.x)
                smem[i] = 0;
            __syncthreads();

            const int n = blockIdx.x;
            const float ang = n * theta;

            float sin_ang;
            float cos_ang;
            sincosf(ang, &sin_ang, &cos_ang);

            const float tabSin = sin_ang * irho;
            const float tabCos = cos_ang * irho;

            for (int i = threadIdx.x; i < count; i += blockDim.x)
            {
                // read one element from global memory
                const unsigned int qvalue = list[i];
                const unsigned int x = (qvalue & 0x0000FFFF);
                const unsigned int y = (qvalue >> 16) & 0x0000FFFF;

                int r = __float2int_rn(x * tabCos + y * tabSin);
                r += (numrho - 1) / 2;

                Emulation::smem::atomicInc(&smem[r], (unsigned int)(-1));
            }
            __syncthreads();

            for (int i = threadIdx.x; i < numrho; i += blockDim.x)
                accum(n + 1, i + 1) = smem[i];
        }

        void linesAccum_gpu(const unsigned int* list, unsigned int count, DevMem2D_<unsigned int> accum, float rho, float theta)
        {
            const dim3 block(1024);
            const dim3 grid(accum.rows - 2);

            cudaSafeCall( cudaFuncSetCacheConfig(linesAccum, cudaFuncCachePreferShared) );

            size_t smem_size = (accum.cols - 2) * sizeof(unsigned int);

            linesAccum<<<grid, block, smem_size>>>(list, count, accum, 1.0f / rho, theta, accum.cols - 2);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void linesGetResult(const DevMem2D_<unsigned int> accum, float2* out, int* voices, const int maxSize,
                                       const float threshold, const float theta, const float rho, const int numrho)
        {
            __shared__ unsigned int smem[8][32];

            int r = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
            int n = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

            if (r >= accum.cols || n >= accum.rows)
                return;

            smem[threadIdx.y][threadIdx.x] = accum(n, r);
            __syncthreads();

            r -= 1;
            n -= 1;

            if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1 || threadIdx.y == 0 || threadIdx.y == blockDim.y - 1 || r >= accum.cols - 2 || n >= accum.rows - 2)
                return;

            if (smem[threadIdx.y][threadIdx.x] > threshold &&
                smem[threadIdx.y][threadIdx.x] >  smem[threadIdx.y - 1][threadIdx.x] &&
                smem[threadIdx.y][threadIdx.x] >= smem[threadIdx.y + 1][threadIdx.x] &&
                smem[threadIdx.y][threadIdx.x] >  smem[threadIdx.y][threadIdx.x - 1] &&
                smem[threadIdx.y][threadIdx.x] >= smem[threadIdx.y][threadIdx.x + 1])
            {
                float radius = (r - (numrho - 1) * 0.5f) * rho;
                float angle = n * theta;

                const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));
                if (ind < maxSize)
                {
                    out[ind] = make_float2(radius, angle);
                    voices[ind] = smem[threadIdx.y][threadIdx.x];
                }
            }
        }

        unsigned int linesGetResult_gpu(DevMem2D_<unsigned int> accum, float2* out, int* voices, unsigned int maxSize,
                                        float rho, float theta, float threshold, bool doSort)
        {
            void* counter_ptr;
            cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, g_counter) );

            cudaSafeCall( cudaMemset(counter_ptr, 0, sizeof(unsigned int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols, block.x - 2), divUp(accum.rows, block.y - 2));

            linesGetResult<<<grid, block>>>(accum, out, voices, maxSize, threshold, theta, rho, accum.cols - 2);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            unsigned int total_count;
            cudaSafeCall( cudaMemcpy(&total_count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

            total_count = ::min(total_count, maxSize);

            if (doSort && total_count > 0)
            {
                thrust::device_ptr<float2> out_ptr(out);
                thrust::device_ptr<int> voices_ptr(voices);
                thrust::sort_by_key(voices_ptr, voices_ptr + total_count, out_ptr, thrust::greater<int>());
            }

            return total_count;
        }
    }
}}}
