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

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

////////////////////////////////////////////////////////////////////////
// buildPointList

#define PIXELS_PER_THREAD 16

__kernel void buildPointList(__global const uchar* src,
                             int cols,
                             int rows,
                             int step,
                             __global unsigned int* list,
                             __global int* counter)
{
    __local unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
    __local int s_qsize[4];
    __local int s_globStart[4];

    const int x = get_group_id(0) * get_local_size(0) * PIXELS_PER_THREAD + get_local_id(0);
    const int y = get_global_id(1);

    if (get_local_id(0) == 0)
        s_qsize[get_local_id(1)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
        
    if (y < rows)
    {
        // fill the queue
        __global const uchar* srcRow = &src[y * step];
        for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < cols; ++i, xx += get_local_size(0))
        {
            if (srcRow[xx])
            {
                const unsigned int val = (y << 16) | xx;
                const int qidx = atomic_add(&s_qsize[get_local_id(1)], 1);
                s_queues[get_local_id(1)][qidx] = val;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // let one work-item reserve the space required in the global list
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        // find how many items are stored in each list
        int totalSize = 0;
        for (int i = 0; i < get_local_size(1); ++i)
        {
            s_globStart[i] = totalSize;
            totalSize += s_qsize[i];
        }

        // calculate the offset in the global list
        const int globalOffset = atomic_add(counter, totalSize);
        for (int i = 0; i < get_local_size(1); ++i)
            s_globStart[i] += globalOffset;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // copy local queues to global queue
    const int qsize = s_qsize[get_local_id(1)];
    int gidx = s_globStart[get_local_id(1)] + get_local_id(0);
    for(int i = get_local_id(0); i < qsize; i += get_local_size(0), gidx += get_local_size(0))
        list[gidx] = s_queues[get_local_id(1)][i];
}

////////////////////////////////////////////////////////////////////////
// circlesAccumCenters

// __global__ void circlesAccumCenters(const unsigned int* list, const int count, const PtrStepi dx, const PtrStepi dy,
//                                     PtrStepi accum, const int width, const int height, const int minRadius, const int maxRadius, const float idp)
// {
//     const int SHIFT = 10;
//     const int ONE = 1 << SHIFT;

//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tid >= count)
//         return;

//     const unsigned int val = list[tid];

//     const int x = (val & 0xFFFF);
//     const int y = (val >> 16) & 0xFFFF;

//     const int vx = dx(y, x);
//     const int vy = dy(y, x);

//     if (vx == 0 && vy == 0)
//         return;

//     const float mag = ::sqrtf(vx * vx + vy * vy);

//     const int x0 = __float2int_rn((x * idp) * ONE);
//     const int y0 = __float2int_rn((y * idp) * ONE);

//     int sx = __float2int_rn((vx * idp) * ONE / mag);
//     int sy = __float2int_rn((vy * idp) * ONE / mag);

//     // Step from minRadius to maxRadius in both directions of the gradient
//     for (int k1 = 0; k1 < 2; ++k1)
//     {
//         int x1 = x0 + minRadius * sx;
//         int y1 = y0 + minRadius * sy;

//         for (int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, ++r)
//         {
//             const int x2 = x1 >> SHIFT;
//             const int y2 = y1 >> SHIFT;

//             if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height)
//                 break;

//             ::atomicAdd(accum.ptr(y2 + 1) + x2 + 1, 1);
//         }

//         sx = -sx;
//         sy = -sy;
//     }
// }

// void circlesAccumCenters_gpu(const unsigned int* list, int count, PtrStepi dx, PtrStepi dy, PtrStepSzi accum, int minRadius, int maxRadius, float idp)
// {
//     const dim3 block(256);
//     const dim3 grid(divUp(count, block.x));

//     cudaSafeCall( cudaFuncSetCacheConfig(circlesAccumCenters, cudaFuncCachePreferL1) );

//     circlesAccumCenters<<<grid, block>>>(list, count, dx, dy, accum, accum.cols - 2, accum.rows - 2, minRadius, maxRadius, idp);
//     cudaSafeCall( cudaGetLastError() );

//     cudaSafeCall( cudaDeviceSynchronize() );
// }

// ////////////////////////////////////////////////////////////////////////
// // buildCentersList

// __global__ void buildCentersList(const PtrStepSzi accum, unsigned int* centers, const int threshold)
// {
//     const int x = blockIdx.x * blockDim.x + threadIdx.x;
//     const int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < accum.cols - 2 && y < accum.rows - 2)
//     {
//         const int top = accum(y, x + 1);

//         const int left = accum(y + 1, x);
//         const int cur = accum(y + 1, x + 1);
//         const int right = accum(y + 1, x + 2);

//         const int bottom = accum(y + 2, x + 1);

//         if (cur > threshold && cur > top && cur >= bottom && cur >  left && cur >= right)
//         {
//             const unsigned int val = (y << 16) | x;
//             const int idx = ::atomicAdd(&g_counter, 1);
//             centers[idx] = val;
//         }
//     }
// }

// int buildCentersList_gpu(PtrStepSzi accum, unsigned int* centers, int threshold)
// {
//     void* counterPtr;
//     cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

//     cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

//     const dim3 block(32, 8);
//     const dim3 grid(divUp(accum.cols - 2, block.x), divUp(accum.rows - 2, block.y));

//     cudaSafeCall( cudaFuncSetCacheConfig(buildCentersList, cudaFuncCachePreferL1) );

//     buildCentersList<<<grid, block>>>(accum, centers, threshold);
//     cudaSafeCall( cudaGetLastError() );

//     cudaSafeCall( cudaDeviceSynchronize() );

//     int totalCount;
//     cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

//     return totalCount;
// }

// ////////////////////////////////////////////////////////////////////////
// // circlesAccumRadius

// __global__ void circlesAccumRadius(const unsigned int* centers, const unsigned int* list, const int count,
//                                    float3* circles, const int maxCircles, const float dp,
//                                    const int minRadius, const int maxRadius, const int histSize, const int threshold)
// {
//     int* smem = DynamicSharedMem<int>();

//     for (int i = threadIdx.x; i < histSize + 2; i += blockDim.x)
//         smem[i] = 0;
//     __syncthreads();

//     unsigned int val = centers[blockIdx.x];

//     float cx = (val & 0xFFFF);
//     float cy = (val >> 16) & 0xFFFF;

//     cx = (cx + 0.5f) * dp;
//     cy = (cy + 0.5f) * dp;

//     for (int i = threadIdx.x; i < count; i += blockDim.x)
//     {
//         val = list[i];

//         const int x = (val & 0xFFFF);
//         const int y = (val >> 16) & 0xFFFF;

//         const float rad = ::sqrtf((cx - x) * (cx - x) + (cy - y) * (cy - y));
//         if (rad >= minRadius && rad <= maxRadius)
//         {
//             const int r = __float2int_rn(rad - minRadius);

//             Emulation::smem::atomicAdd(&smem[r + 1], 1);
//         }
//     }

//     __syncthreads();

//     for (int i = threadIdx.x; i < histSize; i += blockDim.x)
//     {
//         const int curVotes = smem[i + 1];

//         if (curVotes >= threshold && curVotes > smem[i] && curVotes >= smem[i + 2])
//         {
//             const int ind = ::atomicAdd(&g_counter, 1);
//             if (ind < maxCircles)
//                 circles[ind] = make_float3(cx, cy, i + minRadius);
//         }
//     }
// }

// int circlesAccumRadius_gpu(const unsigned int* centers, int centersCount, const unsigned int* list, int count,
//                            float3* circles, int maxCircles, float dp, int minRadius, int maxRadius, int threshold, bool has20)
// {
//     void* counterPtr;
//     cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

//     cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

//     const dim3 block(has20 ? 1024 : 512);
//     const dim3 grid(centersCount);

//     const int histSize = maxRadius - minRadius + 1;
//     size_t smemSize = (histSize + 2) * sizeof(int);

//     circlesAccumRadius<<<grid, block, smemSize>>>(centers, list, count, circles, maxCircles, dp, minRadius, maxRadius, histSize, threshold);
//     cudaSafeCall( cudaGetLastError() );

//     cudaSafeCall( cudaDeviceSynchronize() );

//     int totalCount;
//     cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

//     totalCount = ::min(totalCount, maxCircles);

//     return totalCount;
// }
