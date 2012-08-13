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

namespace cv { namespace gpu { namespace device
{
    namespace hough
    {
        __global__ void linesAccum(const DevMem2Db src, PtrStep_<uint> accum, const float theta, const int numangle, const int numrho, const float irho)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= src.cols || y >= src.rows)
                return;

            if (src(y, x))
            {
                float ang = 0.0f;
                for(int n = 0; n < numangle; ++n, ang += theta)
                {
                    float sin_ang;
                    float cos_ang;
                    sincosf(ang, &sin_ang, &cos_ang);

                    const float tabSin = sin_ang * irho;
                    const float tabCos = cos_ang * irho;

                    int r = __float2int_rn(x * tabCos + y * tabSin);
                    r += (numrho - 1) / 2;

                    atomicInc(accum.ptr(n + 1) + r + 1, (unsigned int)-1);
                }
            }
        }

        void linesAccum_gpu(DevMem2Db src, PtrStep_<uint> accum, float theta, int numangle, int numrho, float irho)
        {
            const dim3 block(32, 8);
            const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

            linesAccum<<<grid, block>>>(src, accum, theta, numangle, numrho, irho);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }

        __device__ unsigned int g_counter;

        __global__ void linesGetResult(const DevMem2D_<uint> accum, float2* out, int* voices, const int maxSize, const float threshold, const float theta, const float rho, const int numrho)
        {
            __shared__ uint smem[8][32];

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

        int linesGetResult_gpu(DevMem2D_<uint> accum, float2* out, int* voices, int maxSize, float threshold, float theta, float rho, bool doSort)
        {
            void* counter_ptr;
            cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, g_counter) );

            cudaSafeCall( cudaMemset(counter_ptr, 0, sizeof(unsigned int)) );

            const dim3 block(32, 8);
            const dim3 grid(divUp(accum.cols, block.x - 2), divUp(accum.rows, block.y - 2));

            linesGetResult<<<grid, block>>>(accum, out, voices, maxSize, threshold, theta, rho, accum.cols - 2);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );

            uint total_count;
            cudaSafeCall( cudaMemcpy(&total_count, counter_ptr, sizeof(uint), cudaMemcpyDeviceToHost) );

            if (doSort)
            {
                thrust::device_ptr<float2> out_ptr(out);
                thrust::device_ptr<int> voices_ptr(voices);
                thrust::sort_by_key(voices_ptr, voices_ptr + total_count, out_ptr, thrust::greater<int>());
            }

            return total_count;
        }
    }
}}}
