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
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/dynamic_smem.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        __device__ int g_counter;

        ////////////////////////////////////////////////////////////////////////
        // buildPointList

        template <int PIXELS_PER_THREAD>
        __global__ void buildPointList(const PtrStepSzb src, unsigned int* list)
        {
            __shared__ unsigned int s_queues[4][32 * PIXELS_PER_THREAD];
            __shared__ int s_qsize[4];
            __shared__ int s_globStart[4];

            const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (threadIdx.x == 0)
                s_qsize[threadIdx.y] = 0;
            __syncthreads();

            if (y < src.rows)
            {
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

        int buildPointList_gpu(PtrStepSzb src, unsigned int* list)
        {
            const int PIXELS_PER_THREAD = 16;

            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 4);
            const dim3 grid(divUp(src.cols, block.x * PIXELS_PER_THREAD), divUp(src.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(buildPointList<PIXELS_PER_THREAD>, cudaFuncCachePreferShared) );

            buildPointList<PIXELS_PER_THREAD><<<grid, block>>>(src, list);
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

        ////////////////////////////////////////////////////////////////////////
        // houghLinesProbabilistic

        texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_mask(false, cudaFilterModePoint, cudaAddressModeClamp);

        __global__ void houghLinesProbabilistic(const PtrStepSzi accum,
                                                int4* out, const int maxSize,
                                                const float rho, const float theta,
                                                const int lineGap, const int lineLength,
                                                const int rows, const int cols)
        {
            const int r = blockIdx.x * blockDim.x + threadIdx.x;
            const int n = blockIdx.y * blockDim.y + threadIdx.y;

            if (r >= accum.cols - 2 || n >= accum.rows - 2)
                return;

            const int curVotes = accum(n + 1, r + 1);

            if (curVotes >= lineLength &&
                curVotes > accum(n, r) &&
                curVotes > accum(n, r + 1) &&
                curVotes > accum(n, r + 2) &&
                curVotes > accum(n + 1, r) &&
                curVotes > accum(n + 1, r + 2) &&
                curVotes > accum(n + 2, r) &&
                curVotes > accum(n + 2, r + 1) &&
                curVotes > accum(n + 2, r + 2))
            {
                const float radius = (r - (accum.cols - 2 - 1) * 0.5f) * rho;
                const float angle = n * theta;

                float cosa;
                float sina;
                sincosf(angle, &sina, &cosa);

                float2 p0 = make_float2(cosa * radius, sina * radius);
                float2 dir = make_float2(-sina, cosa);

                float2 pb[4] = {make_float2(-1, -1), make_float2(-1, -1), make_float2(-1, -1), make_float2(-1, -1)};
                float a;

                if (dir.x != 0)
                {
                    a = -p0.x / dir.x;
                    pb[0].x = 0;
                    pb[0].y = p0.y + a * dir.y;

                    a = (cols - 1 - p0.x) / dir.x;
                    pb[1].x = cols - 1;
                    pb[1].y = p0.y + a * dir.y;
                }
                if (dir.y != 0)
                {
                    a = -p0.y / dir.y;
                    pb[2].x = p0.x + a * dir.x;
                    pb[2].y = 0;

                    a = (rows - 1 - p0.y) / dir.y;
                    pb[3].x = p0.x + a * dir.x;
                    pb[3].y = rows - 1;
                }

                if (pb[0].x == 0 && (pb[0].y >= 0 && pb[0].y < rows))
                {
                    p0 = pb[0];
                    if (dir.x < 0)
                        dir = -dir;
                }
                else if (pb[1].x == cols - 1 && (pb[0].y >= 0 && pb[0].y < rows))
                {
                    p0 = pb[1];
                    if (dir.x > 0)
                        dir = -dir;
                }
                else if (pb[2].y == 0 && (pb[2].x >= 0 && pb[2].x < cols))
                {
                    p0 = pb[2];
                    if (dir.y < 0)
                        dir = -dir;
                }
                else if (pb[3].y == rows - 1 && (pb[3].x >= 0 && pb[3].x < cols))
                {
                    p0 = pb[3];
                    if (dir.y > 0)
                        dir = -dir;
                }

                float2 d;
                if (::fabsf(dir.x) > ::fabsf(dir.y))
                {
                    d.x = dir.x > 0 ? 1 : -1;
                    d.y = dir.y / ::fabsf(dir.x);
                }
                else
                {
                    d.x = dir.x / ::fabsf(dir.y);
                    d.y = dir.y > 0 ? 1 : -1;
                }

                float2 line_end[2];
                int gap;
                bool inLine = false;

                float2 p1 = p0;
                if (p1.x < 0 || p1.x >= cols || p1.y < 0 || p1.y >= rows)
                    return;

                for (;;)
                {
                    if (tex2D(tex_mask, p1.x, p1.y))
                    {
                        gap = 0;

                        if (!inLine)
                        {
                            line_end[0] = p1;
                            line_end[1] = p1;
                            inLine = true;
                        }
                        else
                        {
                            line_end[1] = p1;
                        }
                    }
                    else if (inLine)
                    {
                        if (++gap > lineGap)
                        {
                            bool good_line = ::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                                             ::abs(line_end[1].y - line_end[0].y) >= lineLength;

                            if (good_line)
                            {
                                const int ind = ::atomicAdd(&g_counter, 1);
                                if (ind < maxSize)
                                    out[ind] = make_int4(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
                            }

                            gap = 0;
                            inLine = false;
                        }
                    }

                    p1 = p1 + d;
                    if (p1.x < 0 || p1.x >= cols || p1.y < 0 || p1.y >= rows)
                    {
                        if (inLine)
                        {
                            bool good_line = ::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                                             ::abs(line_end[1].y - line_end[0].y) >= lineLength;

                            if (good_line)
                            {
                                const int ind = ::atomicAdd(&g_counter, 1);
                                if (ind < maxSize)
                                    out[ind] = make_int4(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
                            }

                        }
                        break;
                    }
                }
            }
        }

        int houghLinesProbabilistic_gpu(PtrStepSzb mask, PtrStepSzi accum, int4* out, int maxSize, float rho, float theta, int lineGap, int lineLength)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols - 2, block.x), divUp(accum.rows - 2, block.y));

            bindTexture(&tex_mask, mask);

            houghLinesProbabilistic<<<grid, block>>>(accum,
                                                     out, maxSize,
                                                     rho, theta,
                                                     lineGap, lineLength,
                                                     mask.rows, mask.cols);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

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

        void circlesAccumCenters_gpu(const unsigned int* list, int count, PtrStepi dx, PtrStepi dy, PtrStepSzi accum, int minRadius, int maxRadius, float idp)
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

        __global__ void buildCentersList(const PtrStepSzi accum, unsigned int* centers, const int threshold)
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

        int buildCentersList_gpu(PtrStepSzi accum, unsigned int* centers, int threshold)
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
            int* smem = DynamicSharedMem<int>();

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

            const int histSize = maxRadius - minRadius + 1;
            size_t smemSize = (histSize + 2) * sizeof(int);

            circlesAccumRadius<<<grid, block, smemSize>>>(centers, list, count, circles, maxCircles, dp, minRadius, maxRadius, histSize, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxCircles);

            return totalCount;
        }

        ////////////////////////////////////////////////////////////////////////
        // Generalized Hough

        template <typename T, int PIXELS_PER_THREAD>
        __global__ void buildEdgePointList(const PtrStepSzb edges, const PtrStep<T> dx, const PtrStep<T> dy, unsigned int* coordList, float* thetaList)
        {
            __shared__ unsigned int s_coordLists[4][32 * PIXELS_PER_THREAD];
            __shared__ float s_thetaLists[4][32 * PIXELS_PER_THREAD];
            __shared__ int s_sizes[4];
            __shared__ int s_globStart[4];

            const int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (threadIdx.x == 0)
                s_sizes[threadIdx.y] = 0;
            __syncthreads();

            if (y < edges.rows)
            {
                // fill the queue
                const uchar* edgesRow = edges.ptr(y);
                const T* dxRow = dx.ptr(y);
                const T* dyRow = dy.ptr(y);

                for (int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < edges.cols; ++i, xx += blockDim.x)
                {
                    const T dxVal = dxRow[xx];
                    const T dyVal = dyRow[xx];

                    if (edgesRow[xx] && (dxVal != 0 || dyVal != 0))
                    {
                        const unsigned int coord = (y << 16) | xx;

                        float theta = ::atan2f(dyVal, dxVal);
                        if (theta < 0)
                            theta += 2.0f * CV_PI_F;

                        const int qidx = Emulation::smem::atomicAdd(&s_sizes[threadIdx.y], 1);

                        s_coordLists[threadIdx.y][qidx] = coord;
                        s_thetaLists[threadIdx.y][qidx] = theta;
                    }
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
                    totalSize += s_sizes[i];
                }

                // calculate the offset in the global list
                const int globalOffset = atomicAdd(&g_counter, totalSize);
                for (int i = 0; i < blockDim.y; ++i)
                    s_globStart[i] += globalOffset;
            }

            __syncthreads();

            // copy local queues to global queue
            const int qsize = s_sizes[threadIdx.y];
            int gidx = s_globStart[threadIdx.y] + threadIdx.x;
            for(int i = threadIdx.x; i < qsize; i += blockDim.x, gidx += blockDim.x)
            {
                coordList[gidx] = s_coordLists[threadIdx.y][i];
                thetaList[gidx] = s_thetaLists[threadIdx.y][i];
            }
        }

        template <typename T>
        int buildEdgePointList_gpu(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList)
        {
            const int PIXELS_PER_THREAD = 8;

            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 4);
            const dim3 grid(divUp(edges.cols, block.x * PIXELS_PER_THREAD), divUp(edges.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(buildEdgePointList<T, PIXELS_PER_THREAD>, cudaFuncCachePreferShared) );

            buildEdgePointList<T, PIXELS_PER_THREAD><<<grid, block>>>(edges, (PtrStepSz<T>) dx, (PtrStepSz<T>) dy, coordList, thetaList);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            return totalCount;
        }

        template int buildEdgePointList_gpu<short>(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);
        template int buildEdgePointList_gpu<int>(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);
        template int buildEdgePointList_gpu<float>(PtrStepSzb edges, PtrStepSzb dx, PtrStepSzb dy, unsigned int* coordList, float* thetaList);

        __global__ void buildRTable(const unsigned int* coordList, const float* thetaList, const int pointsCount,
                                    PtrStep<short2> r_table, int* r_sizes, int maxSize,
                                    const short2 templCenter, const float thetaScale)
        {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= pointsCount)
                return;

            const unsigned int coord = coordList[tid];
            short2 p;
            p.x = (coord & 0xFFFF);
            p.y = (coord >> 16) & 0xFFFF;

            const float theta = thetaList[tid];
            const int n = __float2int_rn(theta * thetaScale);

            const int ind = ::atomicAdd(r_sizes + n, 1);
            if (ind < maxSize)
                r_table(n, ind) = saturate_cast<short2>(p - templCenter);
        }

        void buildRTable_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                             PtrStepSz<short2> r_table, int* r_sizes,
                             short2 templCenter, int levels)
        {
            const dim3 block(256);
            const dim3 grid(divUp(pointsCount, block.x));

            const float thetaScale = levels / (2.0f * CV_PI_F);

            buildRTable<<<grid, block>>>(coordList, thetaList, pointsCount, r_table, r_sizes, r_table.cols, templCenter, thetaScale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        ////////////////////////////////////////////////////////////////////////
        // GHT_Ballard_Pos

        __global__ void GHT_Ballard_Pos_calcHist(const unsigned int* coordList, const float* thetaList, const int pointsCount,
                                                 const PtrStep<short2> r_table, const int* r_sizes,
                                                 PtrStepSzi hist,
                                                 const float idp, const float thetaScale)
        {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= pointsCount)
                return;

            const unsigned int coord = coordList[tid];
            short2 p;
            p.x = (coord & 0xFFFF);
            p.y = (coord >> 16) & 0xFFFF;

            const float theta = thetaList[tid];
            const int n = __float2int_rn(theta * thetaScale);

            const short2* r_row = r_table.ptr(n);
            const int r_row_size = r_sizes[n];

            for (int j = 0; j < r_row_size; ++j)
            {
                int2 c = p - r_row[j];

                c.x = __float2int_rn(c.x * idp);
                c.y = __float2int_rn(c.y * idp);

                if (c.x >= 0 && c.x < hist.cols - 2 && c.y >= 0 && c.y < hist.rows - 2)
                    ::atomicAdd(hist.ptr(c.y + 1) + c.x + 1, 1);
            }
        }

        void GHT_Ballard_Pos_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                          PtrStepSz<short2> r_table, const int* r_sizes,
                                          PtrStepSzi hist,
                                          float dp, int levels)
        {
            const dim3 block(256);
            const dim3 grid(divUp(pointsCount, block.x));

            const float idp = 1.0f / dp;
            const float thetaScale = levels / (2.0f * CV_PI_F);

            GHT_Ballard_Pos_calcHist<<<grid, block>>>(coordList, thetaList, pointsCount, r_table, r_sizes, hist, idp, thetaScale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void GHT_Ballard_Pos_findPosInHist(const PtrStepSzi hist, float4* out, int3* votes, const int maxSize, const float dp, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= hist.cols - 2 || y >= hist.rows - 2)
                return;

            const int curVotes = hist(y + 1, x + 1);

            if (curVotes > threshold &&
                curVotes >  hist(y + 1, x) &&
                curVotes >= hist(y + 1, x + 2) &&
                curVotes >  hist(y, x + 1) &&
                curVotes >= hist(y + 2, x + 1))
            {
                const int ind = ::atomicAdd(&g_counter, 1);

                if (ind < maxSize)
                {
                    out[ind] = make_float4(x * dp, y * dp, 1.0f, 0.0f);
                    votes[ind] = make_int3(curVotes, 0, 0);
                }
            }
        }

        int GHT_Ballard_Pos_findPosInHist_gpu(PtrStepSzi hist, float4* out, int3* votes, int maxSize, float dp, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(hist.cols - 2, block.x), divUp(hist.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(GHT_Ballard_Pos_findPosInHist, cudaFuncCachePreferL1) );

            GHT_Ballard_Pos_findPosInHist<<<grid, block>>>(hist, out, votes, maxSize, dp, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }

        ////////////////////////////////////////////////////////////////////////
        // GHT_Ballard_PosScale

        __global__ void GHT_Ballard_PosScale_calcHist(const unsigned int* coordList, const float* thetaList,
                                                      PtrStep<short2> r_table, const int* r_sizes,
                                                      PtrStepi hist, const int rows, const int cols,
                                                      const float minScale, const float scaleStep, const int scaleRange,
                                                      const float idp, const float thetaScale)
        {
            const unsigned int coord = coordList[blockIdx.x];
            float2 p;
            p.x = (coord & 0xFFFF);
            p.y = (coord >> 16) & 0xFFFF;

            const float theta = thetaList[blockIdx.x];
            const int n = __float2int_rn(theta * thetaScale);

            const short2* r_row = r_table.ptr(n);
            const int r_row_size = r_sizes[n];

            for (int j = 0; j < r_row_size; ++j)
            {
                const float2 d = saturate_cast<float2>(r_row[j]);

                for (int s = threadIdx.x; s < scaleRange; s += blockDim.x)
                {
                    const float scale = minScale + s * scaleStep;

                    float2 c = p - scale * d;

                    c.x *= idp;
                    c.y *= idp;

                    if (c.x >= 0 && c.x < cols && c.y >= 0 && c.y < rows)
                        ::atomicAdd(hist.ptr((s + 1) * (rows + 2) + __float2int_rn(c.y + 1)) + __float2int_rn(c.x + 1), 1);
                }
            }
        }

        void GHT_Ballard_PosScale_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                               PtrStepSz<short2> r_table, const int* r_sizes,
                                               PtrStepi hist, int rows, int cols,
                                               float minScale, float scaleStep, int scaleRange,
                                               float dp, int levels)
        {
            const dim3 block(256);
            const dim3 grid(pointsCount);

            const float idp = 1.0f / dp;
            const float thetaScale = levels / (2.0f * CV_PI_F);

            GHT_Ballard_PosScale_calcHist<<<grid, block>>>(coordList, thetaList,
                                                           r_table, r_sizes,
                                                           hist, rows, cols,
                                                           minScale, scaleStep, scaleRange,
                                                           idp, thetaScale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void GHT_Ballard_PosScale_findPosInHist(const PtrStepi hist, const int rows, const int cols, const int scaleRange,
                                                           float4* out, int3* votes, const int maxSize,
                                                           const float minScale, const float scaleStep, const float dp, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= cols || y >= rows)
                return;

            for (int s = 0; s < scaleRange; ++s)
            {
                const float scale = minScale + s * scaleStep;

                const int prevScaleIdx = (s) * (rows + 2);
                const int curScaleIdx = (s + 1) * (rows + 2);
                const int nextScaleIdx = (s + 2) * (rows + 2);

                const int curVotes = hist(curScaleIdx + y + 1, x + 1);

                if (curVotes > threshold &&
                    curVotes >  hist(curScaleIdx + y + 1, x) &&
                    curVotes >= hist(curScaleIdx + y + 1, x + 2) &&
                    curVotes >  hist(curScaleIdx + y, x + 1) &&
                    curVotes >= hist(curScaleIdx + y + 2, x + 1) &&
                    curVotes >  hist(prevScaleIdx + y + 1, x + 1) &&
                    curVotes >= hist(nextScaleIdx + y + 1, x + 1))
                {
                    const int ind = ::atomicAdd(&g_counter, 1);

                    if (ind < maxSize)
                    {
                        out[ind] = make_float4(x * dp, y * dp, scale, 0.0f);
                        votes[ind] = make_int3(curVotes, curVotes, 0);
                    }
                }
            }
        }

        int GHT_Ballard_PosScale_findPosInHist_gpu(PtrStepi hist, int rows, int cols, int scaleRange, float4* out, int3* votes, int maxSize,
                                                   float minScale, float scaleStep, float dp, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(GHT_Ballard_PosScale_findPosInHist, cudaFuncCachePreferL1) );

            GHT_Ballard_PosScale_findPosInHist<<<grid, block>>>(hist, rows, cols, scaleRange, out, votes, maxSize, minScale, scaleStep, dp, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }

        ////////////////////////////////////////////////////////////////////////
        // GHT_Ballard_PosRotation

        __global__ void GHT_Ballard_PosRotation_calcHist(const unsigned int* coordList, const float* thetaList,
                                                         PtrStep<short2> r_table, const int* r_sizes,
                                                         PtrStepi hist, const int rows, const int cols,
                                                         const float minAngle, const float angleStep, const int angleRange,
                                                         const float idp, const float thetaScale)
        {
            const unsigned int coord = coordList[blockIdx.x];
            float2 p;
            p.x = (coord & 0xFFFF);
            p.y = (coord >> 16) & 0xFFFF;

            const float thetaVal = thetaList[blockIdx.x];

            for (int a = threadIdx.x; a < angleRange; a += blockDim.x)
            {
                const float angle = (minAngle + a * angleStep) * (CV_PI_F / 180.0f);
                float sinA, cosA;
                sincosf(angle, &sinA, &cosA);

                float theta = thetaVal - angle;
                if (theta < 0)
                    theta += 2.0f * CV_PI_F;

                const int n = __float2int_rn(theta * thetaScale);

                const short2* r_row = r_table.ptr(n);
                const int r_row_size = r_sizes[n];

                for (int j = 0; j < r_row_size; ++j)
                {
                    const float2 d = saturate_cast<float2>(r_row[j]);

                    const float2 dr = make_float2(d.x * cosA - d.y * sinA, d.x * sinA + d.y * cosA);

                    float2 c = make_float2(p.x - dr.x, p.y - dr.y);
                    c.x *= idp;
                    c.y *= idp;

                    if (c.x >= 0 && c.x < cols && c.y >= 0 && c.y < rows)
                        ::atomicAdd(hist.ptr((a + 1) * (rows + 2) + __float2int_rn(c.y + 1)) + __float2int_rn(c.x + 1), 1);
                }
            }
        }

        void GHT_Ballard_PosRotation_calcHist_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                  PtrStepSz<short2> r_table, const int* r_sizes,
                                                  PtrStepi hist, int rows, int cols,
                                                  float minAngle, float angleStep, int angleRange,
                                                  float dp, int levels)
        {
            const dim3 block(256);
            const dim3 grid(pointsCount);

            const float idp = 1.0f / dp;
            const float thetaScale = levels / (2.0f * CV_PI_F);

            GHT_Ballard_PosRotation_calcHist<<<grid, block>>>(coordList, thetaList,
                                                              r_table, r_sizes,
                                                              hist, rows, cols,
                                                              minAngle, angleStep, angleRange,
                                                              idp, thetaScale);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void GHT_Ballard_PosRotation_findPosInHist(const PtrStepi hist, const int rows, const int cols, const int angleRange,
                                                              float4* out, int3* votes, const int maxSize,
                                                              const float minAngle, const float angleStep, const float dp, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= cols || y >= rows)
                return;

            for (int a = 0; a < angleRange; ++a)
            {
                const float angle = minAngle + a * angleStep;

                const int prevAngleIdx = (a) * (rows + 2);
                const int curAngleIdx = (a + 1) * (rows + 2);
                const int nextAngleIdx = (a + 2) * (rows + 2);

                const int curVotes = hist(curAngleIdx + y + 1, x + 1);

                if (curVotes > threshold &&
                    curVotes >  hist(curAngleIdx + y + 1, x) &&
                    curVotes >= hist(curAngleIdx + y + 1, x + 2) &&
                    curVotes >  hist(curAngleIdx + y, x + 1) &&
                    curVotes >= hist(curAngleIdx + y + 2, x + 1) &&
                    curVotes >  hist(prevAngleIdx + y + 1, x + 1) &&
                    curVotes >= hist(nextAngleIdx + y + 1, x + 1))
                {
                    const int ind = ::atomicAdd(&g_counter, 1);

                    if (ind < maxSize)
                    {
                        out[ind] = make_float4(x * dp, y * dp, 1.0f, angle);
                        votes[ind] = make_int3(curVotes, 0, curVotes);
                    }
                }
            }
        }

        int GHT_Ballard_PosRotation_findPosInHist_gpu(PtrStepi hist, int rows, int cols, int angleRange, float4* out, int3* votes, int maxSize,
                                                      float minAngle, float angleStep, float dp, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemset(counterPtr, 0, sizeof(int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(GHT_Ballard_PosRotation_findPosInHist, cudaFuncCachePreferL1) );

            GHT_Ballard_PosRotation_findPosInHist<<<grid, block>>>(hist, rows, cols, angleRange, out, votes, maxSize, minAngle, angleStep, dp, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }

        ////////////////////////////////////////////////////////////////////////
        // GHT_Guil_Full

        struct FeatureTable
        {
            uchar* p1_pos_data;
            size_t p1_pos_step;

            uchar* p1_theta_data;
            size_t p1_theta_step;

            uchar* p2_pos_data;
            size_t p2_pos_step;

            uchar* d12_data;
            size_t d12_step;

            uchar* r1_data;
            size_t r1_step;

            uchar* r2_data;
            size_t r2_step;
        };

        __constant__ FeatureTable c_templFeatures;
        __constant__ FeatureTable c_imageFeatures;

        void GHT_Guil_Full_setTemplFeatures(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2)
        {
            FeatureTable tbl;

            tbl.p1_pos_data = p1_pos.data;
            tbl.p1_pos_step = p1_pos.step;

            tbl.p1_theta_data = p1_theta.data;
            tbl.p1_theta_step = p1_theta.step;

            tbl.p2_pos_data = p2_pos.data;
            tbl.p2_pos_step = p2_pos.step;

            tbl.d12_data = d12.data;
            tbl.d12_step = d12.step;

            tbl.r1_data = r1.data;
            tbl.r1_step = r1.step;

            tbl.r2_data = r2.data;
            tbl.r2_step = r2.step;

            cudaSafeCall( cudaMemcpyToSymbol(c_templFeatures, &tbl, sizeof(FeatureTable)) );
        }
        void GHT_Guil_Full_setImageFeatures(PtrStepb p1_pos, PtrStepb p1_theta, PtrStepb p2_pos, PtrStepb d12, PtrStepb r1, PtrStepb r2)
        {
            FeatureTable tbl;

            tbl.p1_pos_data = p1_pos.data;
            tbl.p1_pos_step = p1_pos.step;

            tbl.p1_theta_data = p1_theta.data;
            tbl.p1_theta_step = p1_theta.step;

            tbl.p2_pos_data = p2_pos.data;
            tbl.p2_pos_step = p2_pos.step;

            tbl.d12_data = d12.data;
            tbl.d12_step = d12.step;

            tbl.r1_data = r1.data;
            tbl.r1_step = r1.step;

            tbl.r2_data = r2.data;
            tbl.r2_step = r2.step;

            cudaSafeCall( cudaMemcpyToSymbol(c_imageFeatures, &tbl, sizeof(FeatureTable)) );
        }

        struct TemplFeatureTable
        {
            static __device__ float2* p1_pos(int n)
            {
                return (float2*)(c_templFeatures.p1_pos_data + n * c_templFeatures.p1_pos_step);
            }
            static __device__ float* p1_theta(int n)
            {
                return (float*)(c_templFeatures.p1_theta_data + n * c_templFeatures.p1_theta_step);
            }
            static __device__ float2* p2_pos(int n)
            {
                return (float2*)(c_templFeatures.p2_pos_data + n * c_templFeatures.p2_pos_step);
            }

            static __device__ float* d12(int n)
            {
                return (float*)(c_templFeatures.d12_data + n * c_templFeatures.d12_step);
            }

            static __device__ float2* r1(int n)
            {
                return (float2*)(c_templFeatures.r1_data + n * c_templFeatures.r1_step);
            }
            static __device__ float2* r2(int n)
            {
                return (float2*)(c_templFeatures.r2_data + n * c_templFeatures.r2_step);
            }
        };
        struct ImageFeatureTable
        {
            static __device__ float2* p1_pos(int n)
            {
                return (float2*)(c_imageFeatures.p1_pos_data + n * c_imageFeatures.p1_pos_step);
            }
            static __device__ float* p1_theta(int n)
            {
                return (float*)(c_imageFeatures.p1_theta_data + n * c_imageFeatures.p1_theta_step);
            }
            static __device__ float2* p2_pos(int n)
            {
                return (float2*)(c_imageFeatures.p2_pos_data + n * c_imageFeatures.p2_pos_step);
            }

            static __device__ float* d12(int n)
            {
                return (float*)(c_imageFeatures.d12_data + n * c_imageFeatures.d12_step);
            }

            static __device__ float2* r1(int n)
            {
                return (float2*)(c_imageFeatures.r1_data + n * c_imageFeatures.r1_step);
            }
            static __device__ float2* r2(int n)
            {
                return (float2*)(c_imageFeatures.r2_data + n * c_imageFeatures.r2_step);
            }
        };

        __device__ float clampAngle(float a)
        {
            float res = a;

            while (res > 2.0f * CV_PI_F)
                res -= 2.0f * CV_PI_F;
            while (res < 0.0f)
                res += 2.0f * CV_PI_F;

            return res;
        }

        __device__ bool angleEq(float a, float b, float eps)
        {
            return (::fabs(clampAngle(a - b)) <= eps);
        }

        template <class FT, bool isTempl>
        __global__ void GHT_Guil_Full_buildFeatureList(const unsigned int* coordList, const float* thetaList, const int pointsCount,
                                                       int* sizes, const int maxSize,
                                                       const float xi, const float angleEpsilon, const float alphaScale,
                                                       const float2 center, const float maxDist)
        {
            const float p1_theta = thetaList[blockIdx.x];
            const unsigned int coord1 = coordList[blockIdx.x];
            float2 p1_pos;
            p1_pos.x = (coord1 & 0xFFFF);
            p1_pos.y = (coord1 >> 16) & 0xFFFF;

            for (int i = threadIdx.x; i < pointsCount; i += blockDim.x)
            {
                const float p2_theta = thetaList[i];
                const unsigned int coord2 = coordList[i];
                float2 p2_pos;
                p2_pos.x = (coord2 & 0xFFFF);
                p2_pos.y = (coord2 >> 16) & 0xFFFF;

                if (angleEq(p1_theta - p2_theta, xi, angleEpsilon))
                {
                    const float2 d = p1_pos - p2_pos;

                    float alpha12 = clampAngle(::atan2(d.y, d.x) - p1_theta);
                    float d12 = ::sqrtf(d.x * d.x + d.y * d.y);

                    if (d12 > maxDist)
                        continue;

                    float2 r1 = p1_pos - center;
                    float2 r2 = p2_pos - center;

                    const int n = __float2int_rn(alpha12 * alphaScale);

                    const int ind = ::atomicAdd(sizes + n, 1);

                    if (ind < maxSize)
                    {
                        if (!isTempl)
                        {
                            FT::p1_pos(n)[ind] = p1_pos;
                            FT::p2_pos(n)[ind] = p2_pos;
                        }

                        FT::p1_theta(n)[ind] = p1_theta;

                        FT::d12(n)[ind] = d12;

                        if (isTempl)
                        {
                            FT::r1(n)[ind] = r1;
                            FT::r2(n)[ind] = r2;
                        }
                    }
                }
            }
        }

        template <class FT, bool isTempl>
        void GHT_Guil_Full_buildFeatureList_caller(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                   int* sizes, int maxSize,
                                                   float xi, float angleEpsilon, int levels,
                                                   float2 center, float maxDist)
        {
            const dim3 block(256);
            const dim3 grid(pointsCount);

            const float alphaScale = levels / (2.0f * CV_PI_F);

            GHT_Guil_Full_buildFeatureList<FT, isTempl><<<grid, block>>>(coordList, thetaList, pointsCount,
                                                                         sizes, maxSize,
                                                                         xi * (CV_PI_F / 180.0f), angleEpsilon * (CV_PI_F / 180.0f), alphaScale,
                                                                         center, maxDist);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            thrust::device_ptr<int> sizesPtr(sizes);
            thrust::transform(sizesPtr, sizesPtr + levels + 1, sizesPtr, device::bind2nd(device::minimum<int>(), maxSize));
        }

        void GHT_Guil_Full_buildTemplFeatureList_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                     int* sizes, int maxSize,
                                                     float xi, float angleEpsilon, int levels,
                                                     float2 center, float maxDist)
        {
            GHT_Guil_Full_buildFeatureList_caller<TemplFeatureTable, true>(coordList, thetaList, pointsCount,
                                                                           sizes, maxSize,
                                                                           xi, angleEpsilon, levels,
                                                                           center, maxDist);
        }
        void GHT_Guil_Full_buildImageFeatureList_gpu(const unsigned int* coordList, const float* thetaList, int pointsCount,
                                                     int* sizes, int maxSize,
                                                     float xi, float angleEpsilon, int levels,
                                                     float2 center, float maxDist)
        {
            GHT_Guil_Full_buildFeatureList_caller<ImageFeatureTable, false>(coordList, thetaList, pointsCount,
                                                                            sizes, maxSize,
                                                                            xi, angleEpsilon, levels,
                                                                            center, maxDist);
        }

        __global__ void GHT_Guil_Full_calcOHist(const int* templSizes, const int* imageSizes, int* OHist,
                                                const float minAngle, const float maxAngle, const float iAngleStep, const int angleRange)
        {
            extern __shared__ int s_OHist[];
            for (int i = threadIdx.x; i <= angleRange; i += blockDim.x)
                s_OHist[i] = 0;
            __syncthreads();

            const int tIdx = blockIdx.x;
            const int level = blockIdx.y;

            const int tSize = templSizes[level];

            if (tIdx < tSize)
            {
                const int imSize = imageSizes[level];

                const float t_p1_theta = TemplFeatureTable::p1_theta(level)[tIdx];

                for (int i = threadIdx.x; i < imSize; i += blockDim.x)
                {
                    const float im_p1_theta = ImageFeatureTable::p1_theta(level)[i];

                    const float angle = clampAngle(im_p1_theta - t_p1_theta);

                    if (angle >= minAngle && angle <= maxAngle)
                    {
                        const int n = __float2int_rn((angle - minAngle) * iAngleStep);
                        Emulation::smem::atomicAdd(&s_OHist[n], 1);
                    }
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i <= angleRange; i += blockDim.x)
                ::atomicAdd(OHist + i, s_OHist[i]);
        }

        void GHT_Guil_Full_calcOHist_gpu(const int* templSizes, const int* imageSizes, int* OHist,
                                         float minAngle, float maxAngle, float angleStep, int angleRange,
                                         int levels, int tMaxSize)
        {
            const dim3 block(256);
            const dim3 grid(tMaxSize, levels + 1);

            minAngle *= (CV_PI_F / 180.0f);
            maxAngle *= (CV_PI_F / 180.0f);
            angleStep *= (CV_PI_F / 180.0f);

            const size_t smemSize = (angleRange + 1) * sizeof(float);

            GHT_Guil_Full_calcOHist<<<grid, block, smemSize>>>(templSizes, imageSizes, OHist,
                                                               minAngle, maxAngle, 1.0f / angleStep, angleRange);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void GHT_Guil_Full_calcSHist(const int* templSizes, const int* imageSizes, int* SHist,
                                                const float angle, const float angleEpsilon,
                                                const float minScale, const float maxScale, const float iScaleStep, const int scaleRange)
        {
            extern __shared__ int s_SHist[];
            for (int i = threadIdx.x; i <= scaleRange; i += blockDim.x)
                s_SHist[i] = 0;
            __syncthreads();

            const int tIdx = blockIdx.x;
            const int level = blockIdx.y;

            const int tSize = templSizes[level];

            if (tIdx < tSize)
            {
                const int imSize = imageSizes[level];

                const float t_p1_theta = TemplFeatureTable::p1_theta(level)[tIdx] + angle;
                const float t_d12 = TemplFeatureTable::d12(level)[tIdx] + angle;

                for (int i = threadIdx.x; i < imSize; i += blockDim.x)
                {
                    const float im_p1_theta = ImageFeatureTable::p1_theta(level)[i];
                    const float im_d12 = ImageFeatureTable::d12(level)[i];

                    if (angleEq(im_p1_theta, t_p1_theta, angleEpsilon))
                    {
                        const float scale = im_d12 / t_d12;

                        if (scale >= minScale && scale <= maxScale)
                        {
                            const int s = __float2int_rn((scale - minScale) * iScaleStep);
                            Emulation::smem::atomicAdd(&s_SHist[s], 1);
                        }
                    }
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i <= scaleRange; i += blockDim.x)
                ::atomicAdd(SHist + i, s_SHist[i]);
        }

        void GHT_Guil_Full_calcSHist_gpu(const int* templSizes, const int* imageSizes, int* SHist,
                                         float angle, float angleEpsilon,
                                         float minScale, float maxScale, float iScaleStep, int scaleRange,
                                         int levels, int tMaxSize)
        {
            const dim3 block(256);
            const dim3 grid(tMaxSize, levels + 1);

            angle *= (CV_PI_F / 180.0f);
            angleEpsilon *= (CV_PI_F / 180.0f);

            const size_t smemSize = (scaleRange + 1) * sizeof(float);

            GHT_Guil_Full_calcSHist<<<grid, block, smemSize>>>(templSizes, imageSizes, SHist,
                                                               angle, angleEpsilon,
                                                               minScale, maxScale, iScaleStep, scaleRange);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void GHT_Guil_Full_calcPHist(const int* templSizes, const int* imageSizes, PtrStepSzi PHist,
                                                const float angle, const float sinVal, const float cosVal, const float angleEpsilon, const float scale,
                                                const float idp)
        {
            const int tIdx = blockIdx.x;
            const int level = blockIdx.y;

            const int tSize = templSizes[level];

            if (tIdx < tSize)
            {
                const int imSize = imageSizes[level];

                const float t_p1_theta = TemplFeatureTable::p1_theta(level)[tIdx] + angle;

                float2 r1 = TemplFeatureTable::r1(level)[tIdx];
                float2 r2 = TemplFeatureTable::r2(level)[tIdx];

                r1 = r1 * scale;
                r2 = r2 * scale;

                r1 = make_float2(cosVal * r1.x - sinVal * r1.y, sinVal * r1.x + cosVal * r1.y);
                r2 = make_float2(cosVal * r2.x - sinVal * r2.y, sinVal * r2.x + cosVal * r2.y);

                for (int i = threadIdx.x; i < imSize; i += blockDim.x)
                {
                    const float im_p1_theta = ImageFeatureTable::p1_theta(level)[i];

                    const float2 im_p1_pos = ImageFeatureTable::p1_pos(level)[i];
                    const float2 im_p2_pos = ImageFeatureTable::p2_pos(level)[i];

                    if (angleEq(im_p1_theta, t_p1_theta, angleEpsilon))
                    {
                        float2 c1, c2;

                        c1 = im_p1_pos - r1;
                        c1 = c1 * idp;

                        c2 = im_p2_pos - r2;
                        c2 = c2 * idp;

                        if (::fabs(c1.x - c2.x) > 1 || ::fabs(c1.y - c2.y) > 1)
                            continue;

                        if (c1.y >= 0 && c1.y < PHist.rows - 2 && c1.x >= 0 && c1.x < PHist.cols - 2)
                            ::atomicAdd(PHist.ptr(__float2int_rn(c1.y) + 1) + __float2int_rn(c1.x) + 1, 1);
                    }
                }
            }
        }

        void GHT_Guil_Full_calcPHist_gpu(const int* templSizes, const int* imageSizes, PtrStepSzi PHist,
                                         float angle, float angleEpsilon, float scale,
                                         float dp,
                                         int levels, int tMaxSize)
        {
            const dim3 block(256);
            const dim3 grid(tMaxSize, levels + 1);

            angle *= (CV_PI_F / 180.0f);
            angleEpsilon *= (CV_PI_F / 180.0f);

            const float sinVal = ::sinf(angle);
            const float cosVal = ::cosf(angle);

            cudaSafeCall( cudaFuncSetCacheConfig(GHT_Guil_Full_calcPHist, cudaFuncCachePreferL1) );

            GHT_Guil_Full_calcPHist<<<grid, block>>>(templSizes, imageSizes, PHist,
                                                     angle, sinVal, cosVal, angleEpsilon, scale,
                                                     1.0f / dp);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __global__ void GHT_Guil_Full_findPosInHist(const PtrStepSzi hist, float4* out, int3* votes, const int maxSize,
                                                    const float angle, const int angleVotes, const float scale, const int scaleVotes,
                                                    const float dp, const int threshold)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= hist.cols - 2 || y >= hist.rows - 2)
                return;

            const int curVotes = hist(y + 1, x + 1);

            if (curVotes > threshold &&
                curVotes >  hist(y + 1, x) &&
                curVotes >= hist(y + 1, x + 2) &&
                curVotes >  hist(y, x + 1) &&
                curVotes >= hist(y + 2, x + 1))
            {
                const int ind = ::atomicAdd(&g_counter, 1);

                if (ind < maxSize)
                {
                    out[ind] = make_float4(x * dp, y * dp, scale, angle);
                    votes[ind] = make_int3(curVotes, scaleVotes, angleVotes);
                }
            }
        }

        int GHT_Guil_Full_findPosInHist_gpu(PtrStepSzi hist, float4* out, int3* votes, int curSize, int maxSize,
                                             float angle, int angleVotes, float scale, int scaleVotes,
                                             float dp, int threshold)
        {
            void* counterPtr;
            cudaSafeCall( cudaGetSymbolAddress(&counterPtr, g_counter) );

            cudaSafeCall( cudaMemcpy(counterPtr, &curSize, sizeof(int), cudaMemcpyHostToDevice) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(hist.cols - 2, block.x), divUp(hist.rows - 2, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(GHT_Guil_Full_findPosInHist, cudaFuncCachePreferL1) );

            GHT_Guil_Full_findPosInHist<<<grid, block>>>(hist, out, votes, maxSize,
                                                         angle, angleVotes, scale, scaleVotes,
                                                         dp, threshold);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            int totalCount;
            cudaSafeCall( cudaMemcpy(&totalCount, counterPtr, sizeof(int), cudaMemcpyDeviceToHost) );

            totalCount = ::min(totalCount, maxSize);

            return totalCount;
        }
    }
}}}


#endif /* CUDA_DISABLER */
