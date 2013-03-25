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

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/emulation.hpp"
#include "opencv2/gpu/device/scan.hpp"
#include "opencv2/gpu/device/reduce.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace clahe
{
    __global__ void calcLutKernel(const PtrStepb src, PtrStepb lut,
                                  const int2 tileSize, const int tilesX,
                                  const int clipLimit, const float lutScale)
    {
        __shared__ int smem[512];

        const int tx = blockIdx.x;
        const int ty = blockIdx.y;
        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        smem[tid] = 0;
        __syncthreads();

        for (int i = threadIdx.y; i < tileSize.y; i += blockDim.y)
        {
            const uchar* srcPtr = src.ptr(ty * tileSize.y + i) + tx * tileSize.x;
            for (int j = threadIdx.x; j < tileSize.x; j += blockDim.x)
            {
                const int data = srcPtr[j];
                Emulation::smem::atomicAdd(&smem[data], 1);
            }
        }

        __syncthreads();

        int tHistVal = smem[tid];

        __syncthreads();

        if (clipLimit > 0)
        {
            // clip histogram bar

            int clipped = 0;
            if (tHistVal > clipLimit)
            {
                clipped = tHistVal - clipLimit;
                tHistVal = clipLimit;
            }

            // find number of overall clipped samples

            reduce<256>(smem, clipped, tid, plus<int>());

            // broadcast evaluated value

            __shared__ int totalClipped;

            if (tid == 0)
                totalClipped = clipped;
            __syncthreads();

            // redistribute clipped samples evenly

            int redistBatch = totalClipped / 256;
            tHistVal += redistBatch;

            int residual = totalClipped - redistBatch * 256;
            if (tid < residual)
                ++tHistVal;
        }

        const int lutVal = blockScanInclusive<256>(tHistVal, smem, tid);

        lut(ty * tilesX + tx, tid) = saturate_cast<uchar>(__float2int_rn(lutScale * lutVal));
    }

    void calcLut(PtrStepSzb src, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(tilesX, tilesY);

        calcLutKernel<<<grid, block, 0, stream>>>(src, lut, tileSize, tilesX, clipLimit, lutScale);

        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    __global__ void tranformKernel(const PtrStepSzb src, PtrStepb dst, const PtrStepb lut, const int2 tileSize, const int tilesX, const int tilesY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src.cols || y >= src.rows)
            return;

        const float tyf = (static_cast<float>(y) / tileSize.y) - 0.5f;
        int ty1 = __float2int_rd(tyf);
        int ty2 = ty1 + 1;
        const float ya = tyf - ty1;
        ty1 = ::max(ty1, 0);
        ty2 = ::min(ty2, tilesY - 1);

        const float txf = (static_cast<float>(x) / tileSize.x) - 0.5f;
        int tx1 = __float2int_rd(txf);
        int tx2 = tx1 + 1;
        const float xa = txf - tx1;
        tx1 = ::max(tx1, 0);
        tx2 = ::min(tx2, tilesX - 1);

        const int srcVal = src(y, x);

        float res = 0;

        res += lut(ty1 * tilesX + tx1, srcVal) * ((1.0f - xa) * (1.0f - ya));
        res += lut(ty1 * tilesX + tx2, srcVal) * ((xa) * (1.0f - ya));
        res += lut(ty2 * tilesX + tx1, srcVal) * ((1.0f - xa) * (ya));
        res += lut(ty2 * tilesX + tx2, srcVal) * ((xa) * (ya));

        dst(y, x) = saturate_cast<uchar>(res);
    }

    void transform(PtrStepSzb src, PtrStepSzb dst, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        cudaSafeCall( cudaFuncSetCacheConfig(tranformKernel, cudaFuncCachePreferL1) );

        tranformKernel<<<grid, block, 0, stream>>>(src, dst, lut, tileSize, tilesX, tilesY);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

#endif // CUDA_DISABLER
