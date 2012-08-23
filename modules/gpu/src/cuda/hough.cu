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
        __device__ int g_counter;

        ////////////////////////////////////////////////////////////////////////
        // buildPointList

        const int PIXELS_PER_THREAD = 16;

        __global__ void buildPointList(const DevMem2Db src, unsigned int* list)
        {
            __shared__ unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
            __shared__ int s_qsize[4];
            __shared__ int s_globStart[4];

            const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y >= src.rows)
                return;

            if (threadIdx.x == 0)
                s_qsize[threadIdx.y] = 0;

            __syncthreads();

            // fill the queue
            const uchar* srcRow = src.ptr(y);
            for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < src.cols; ++i, xx += blockDim.x)
            {
                if (srcRow[xx])
                {
                    const unsigned int val = (y << 16) | xx;
                    const int qidx = Emulation::smem::atomicAdd(&s_qsize[threadIdx.y], 1);
                    s_queues[threadIdx.y][qidx] = val;
                }
            }

            __syncthreads();

            // let one thread reserve the space required in the global list
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                // find how many items are stored in each list
                int totalSize = 0;
                for (int i = 0; i < blockDim.y; ++i)
                {
                    s_globStart[i] = totalSize;
                    totalSize += s_qsize[i];
                }

                // calculate the offset in the global list
                const int globalOffset = atomicAdd(&g_counter, totalSize);
                for (int i = 0; i < blockDim.y; ++i)
                    s_globStart[i] += globalOffset;
            }

            __syncthreads();

            // copy local queues to global queue
            const int qsize = s_qsize[threadIdx.y];
            int gidx = s_globStart[threadIdx.y] + threadIdx.x;
            for(int i = threadIdx.x; i < qsize; i += blockDim.x, gidx += blockDim.x)
                list[gidx] = s_queues[threadIdx.y][i];
        }

        int buildPointList_gpu(DevMem2Db src, unsigned int* list)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 4);
            const dim3 grid(divUp(src.cols, block.x * PIXELS_PER_THREAD), divUp(src.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(buildPointList, cudaFuncCachePreferShared) );

            buildPointList<<<grid, block>>>(src, list);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            return totalCount;
        }

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
            extern __shared__ int smem[];

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

        void linesAccum_gpu(const unsigned int* list, int count, DevMem2Di accum, float rho, float theta, size_t sharedMemPerBlock, bool has20)
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

        __global__ void linesGetResult(const DevMem2Di accum, float2* out, int* votes, const int maxSize, const float rho, const float theta, const float threshold, const int numrho)
        {
            const int r = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = blockIdx.y * blockDim.y + threadIdx.y;

            if (r >= accum.cols - 2 && n >= accum.rows - 2)
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

        int linesGetResult_gpu(DevMem2Di accum, float2* out, int* votes, int maxSize, float rho, float theta, float threshold, bool doSort)
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

        ////////////////////////////////////////////////////////////////////////
        // circlesAccumCenters

        __global__ void circlesAccumCenters(const unsigned int* list, const int count, const PtrStepi dx, const PtrStepi dy,
                                            PtrStepi accum, const int width, const int height, const int minRadius, const int maxRadius, const float idp)
        {
            const int SHIFT = 10;
            const int ONE = 1 << SHIFT;

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= count)
                return;

            const unsigned int val = list[tid];

            const int x = (val & 0xFFFF);
            const int y = (val >> 16) & 0xFFFF;

            const int vx = dx(y, x);
            const int vy = dy(y, x);

            if (vx == 0 && vy == 0)
                return;

            const float mag = ::sqrtf(vx * vx + vy * vy);

            const int x0 = __float2int_rn((x * idp) * ONE);
            const int y0 = __float2int_rn((y * idp) * ONE);

            int sx = __float2int_rn((vx * idp) * ONE / mag);
            int sy = __float2int_rn((vy * idp) * ONE / mag);

            // Step from minRadius to maxRadius in both directions of the gradient
            for (int k1 = 0; k1 < 2; ++k1)
            {
                int x1 = x0 + minRadius * sx;
                int y1 = y0 + minRadius * sy;

                for (int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, ++r)
                {
                    const int x2 = x1 >> SHIFT;
                    const int y2 = y1 >> SHIFT;

                    if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height)
                        break;

                    ::atomicAdd(accum.ptr(y2 + 1) + x2 + 1, 1);
                }

                sx = -sx;
                sy = -sy;
            }
        }

        void circlesAccumCenters_gpu(const unsigned int* list, int count, PtrStepi dx, PtrStepi dy, DevMem2Di accum, int minRadius, int maxRadius, float idp)
        {
            const dim3 block(256);
            const dim3 grid(divUp(count, block.x));

            cudaSafeCall( cudaFuncSetCacheConfig(circlesAccumCenters, cudaFuncCachePreferL1) );

            circlesAccumCenters<<<grid, block>>>(list, count, dx, dy, accum, accum.cols - 2, accum.rows - 2, minRadius, maxRadius, idp);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // buildCentersList

        __global__ void buildCentersList(const DevMem2Di accum, unsigned int* centers, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < accum.cols - 2 && y < accum.rows - 2)
            {
                const int top = accum(y, x + 1);

                const int left = accum(y + 1, x);
                const int cur = accum(y + 1, x + 1);
                const int right = accum(y + 1, x + 2);

                const int bottom = accum(y + 2, x + 1);

                if (cur > threshold && cur > top && cur >= bottom && cur >  left && cur >= right)
                {
                    const unsigned int val = (y << 16) | x;
                    const int idx = ::atomicAdd(&g_counter, 1);
                    centers[idx] = val;
                }
            }
        }

        int buildCentersList_gpu(DevMem2Di accum, unsigned int* centers, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols - 2, block.x), divUp(accum.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(buildCentersList, cudaFuncCachePreferL1) );

            buildCentersList<<<grid, block>>>(accum, centers, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            return totalCount;
        }

        ////////////////////////////////////////////////////////////////////////
        // circlesAccumRadius

        __global__ void circlesAccumRadius(const unsigned int* centers, const unsigned int* list, const int count,
                                           float3* circles, const int maxCircles, const float dp,
                                           const int minRadius, const int maxRadius, const int histSize, const int threshold)
        {
            extern __shared__ int smem[];

            for (int i = threadIdx.x; i < histSize + 2; i += blockDim.x)
                smem[i] = 0;
            __syncthreads();

            unsigned int val = centers[blockIdx.x];

            float cx = (val & 0xFFFF);
            float cy = (val >> 16) & 0xFFFF;

            cx = (cx + 0.5f) * dp;
            cy = (cy + 0.5f) * dp;

            for (int i = threadIdx.x; i < count; i += blockDim.x)
            {
                val = list[i];

                const int x = (val & 0xFFFF);
                const int y = (val >> 16) & 0xFFFF;

                const float rad = ::sqrtf((cx - x) * (cx - x) + (cy - y) * (cy - y));
                if (rad >= minRadius && rad <= maxRadius)
                {
                    const int r = __float2int_rn(rad - minRadius);

                    Emulation::smem::atomicAdd(&smem[r + 1], 1);
                }
            }

            __syncthreads();

            for (int i = threadIdx.x; i < histSize; i += blockDim.x)
            {
                const int curVotes = smem[i + 1];

                if (curVotes >= threshold && curVotes > smem[i] && curVotes >= smem[i + 2])
                {
                    const int ind = ::atomicAdd(&g_counter, 1);
                    if (ind < maxCircles)
                        circles[ind] = make_float3(cx, cy, i + minRadius);
                }
            }
        }

        int circlesAccumRadius_gpu(const unsigned int* centers, int centersCount, const unsigned int* list, int count,
                                   float3* circles, int maxCircles, float dp, int minRadius, int maxRadius, int threshold, bool has20)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(has20 ? 1024 : 512);
            const dim3 grid(centersCount);

            const int histSize = ::ceil(maxRadius - minRadius + 1);
            size_t smemSize = (histSize + 2) * sizeof(int);

            circlesAccumRadius<<<grid, block, smemSize>>>(centers, list, count, circles, maxCircles, dp, minRadius, maxRadius, histSize, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxCircles);

            return totalCount;
        }
    }
}}}
