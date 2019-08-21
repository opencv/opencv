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

#include "opencv2/cudev.hpp"

using namespace cv::cudev;

namespace clahe
{
    __global__ void calcLutKernel_8U(const PtrStepb src, PtrStepb lut,
                                     const int2 tileSize, const int tilesX,
                                     const int clipLimit, const float lutScale)
    {
        __shared__ int smem[256];

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
                ::atomicAdd(&smem[data], 1);
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

            blockReduce<256>(smem, clipped, tid, plus<int>());

            // broadcast evaluated value

            __shared__ int totalClipped;
            __shared__ int redistBatch;
            __shared__ int residual;
            __shared__ int rStep;

            if (tid == 0)
            {
                totalClipped = clipped;
                redistBatch = totalClipped / 256;
                residual = totalClipped - redistBatch * 256;

                rStep = 1;
                if (residual != 0)
                    rStep = 256 / residual;
            }

            __syncthreads();

            // redistribute clipped samples evenly

            tHistVal += redistBatch;

            if (residual && tid % rStep == 0 && tid / rStep < residual)
                ++tHistVal;
        }

        const int lutVal = blockScanInclusive<256>(tHistVal, smem, tid);

        lut(ty * tilesX + tx, tid) = saturate_cast<uchar>(__float2int_rn(lutScale * lutVal));
    }

    __global__ void calcLutKernel_16U(const PtrStepus src, PtrStepus lut,
                                      const int2 tileSize, const int tilesX,
                                      const int clipLimit, const float lutScale,
                                      PtrStepSzi hist)
    {
        #define histSize 65536
        #define blockSize 256

        __shared__ int smem[blockSize];

        const int tx = blockIdx.x;
        const int ty = blockIdx.y;
        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        const int histRow = ty * tilesX + tx;

        // build histogram

        for (int i = tid; i < histSize; i += blockSize)
            hist(histRow, i) = 0;

        __syncthreads();

        for (int i = threadIdx.y; i < tileSize.y; i += blockDim.y)
        {
            const ushort* srcPtr = src.ptr(ty * tileSize.y + i) + tx * tileSize.x;
            for (int j = threadIdx.x; j < tileSize.x; j += blockDim.x)
            {
                const int data = srcPtr[j];
                ::atomicAdd(&hist(histRow, data), 1);
            }
        }

        __syncthreads();

        if (clipLimit > 0)
        {
            // clip histogram bar &&
            // find number of overall clipped samples

            __shared__ int partialSum[blockSize];

            for (int i = tid; i < histSize; i += blockSize)
            {
                int histVal = hist(histRow, i);

                int clipped = 0;
                if (histVal > clipLimit)
                {
                    clipped = histVal - clipLimit;
                    hist(histRow, i) = clipLimit;
                }

                // Following code block is in effect equivalent to:
                //
                //      blockReduce<blockSize>(smem, clipped, tid, plus<int>());
                //
                {
                    for (int j = 16; j >= 1; j /= 2)
                    {
                    #if __CUDACC_VER_MAJOR__ >= 9
                        int val = __shfl_down_sync(0xFFFFFFFFU, clipped, j);
                    #else
                        int val = __shfl_down(clipped, j);
                    #endif
                        clipped += val;
                    }

                    if (tid % 32 == 0)
                        smem[tid / 32] = clipped;

                    __syncthreads();

                    if (tid < 8)
                    {
                        clipped = smem[tid];

                        for (int j = 4; j >= 1; j /= 2)
                        {
                        #if __CUDACC_VER_MAJOR__ >= 9
                            int val = __shfl_down_sync(0x000000FFU, clipped, j);
                        #else
                            int val = __shfl_down(clipped, j);
                        #endif
                            clipped += val;
                        }
                    }
                }
                // end of code block

                if (tid == 0)
                    partialSum[i / blockSize] = clipped;

                __syncthreads();
            }

            int partialSum_ = partialSum[tid];

            // Following code block is in effect equivalent to:
            //
            //      blockReduce<blockSize>(smem, partialSum_, tid, plus<int>());
            //
            {
                for (int j = 16; j >= 1; j /= 2)
                {
                #if __CUDACC_VER_MAJOR__ >= 9
                    int val = __shfl_down_sync(0xFFFFFFFFU, partialSum_, j);
                #else
                    int val = __shfl_down(partialSum_, j);
                #endif
                    partialSum_ += val;
                }

                if (tid % 32 == 0)
                    smem[tid / 32] = partialSum_;

                __syncthreads();

                if (tid < 8)
                {
                    partialSum_ = smem[tid];

                    for (int j = 4; j >= 1; j /= 2)
                    {
                    #if __CUDACC_VER_MAJOR__ >= 9
                        int val = __shfl_down_sync(0x000000FFU, partialSum_, j);
                    #else
                        int val = __shfl_down(partialSum_, j);
                    #endif
                        partialSum_ += val;
                    }
                }
            }
            // end of code block

            // broadcast evaluated value &&
            // redistribute clipped samples evenly

            __shared__ int totalClipped;
            __shared__ int redistBatch;
            __shared__ int residual;
            __shared__ int rStep;

            if (tid == 0)
            {
                totalClipped = partialSum_;
                redistBatch = totalClipped / histSize;
                residual = totalClipped - redistBatch * histSize;

                rStep = 1;
                if (residual != 0)
                    rStep = histSize / residual;
            }

            __syncthreads();

            for (int i = tid; i < histSize; i += blockSize)
            {
                int histVal = hist(histRow, i);

                int equalized = histVal + redistBatch;

                if (residual && i % rStep == 0 && i / rStep < residual)
                    ++equalized;

                hist(histRow, i) = equalized;
            }
        }

        __shared__ int partialScan[blockSize];

        for (int i = tid; i < histSize; i += blockSize)
        {
            int equalized = hist(histRow, i);
            equalized = blockScanInclusive<blockSize>(equalized, smem, tid);

            if (tid == blockSize - 1)
                partialScan[i / blockSize] = equalized;

            hist(histRow, i) = equalized;
        }

        __syncthreads();

        int partialScan_ = partialScan[tid];
        partialScan[tid] = blockScanExclusive<blockSize>(partialScan_, smem, tid);

        __syncthreads();

        for (int i = tid; i < histSize; i += blockSize)
        {
            const int lutVal = hist(histRow, i) + partialScan[i / blockSize];

            lut(histRow, i) = saturate_cast<ushort>(__float2int_rn(lutScale * lutVal));
        }

        #undef histSize
        #undef blockSize
    }

    void calcLut_8U(PtrStepSzb src, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(tilesX, tilesY);

        calcLutKernel_8U<<<grid, block, 0, stream>>>(src, lut, tileSize, tilesX, clipLimit, lutScale);

        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    void calcLut_16U(PtrStepSzus src, PtrStepus lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, PtrStepSzi hist, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(tilesX, tilesY);

        calcLutKernel_16U<<<grid, block, 0, stream>>>(src, lut, tileSize, tilesX, clipLimit, lutScale, hist);

        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <typename T>
    __global__ void transformKernel(const PtrStepSz<T> src, PtrStep<T> dst, const PtrStep<T> lut, const int2 tileSize, const int tilesX, const int tilesY)
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

        dst(y, x) = saturate_cast<T>(res);
    }

    template <typename T>
    void transform(PtrStepSz<T> src, PtrStepSz<T> dst, PtrStep<T> lut, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        CV_CUDEV_SAFE_CALL( cudaFuncSetCacheConfig(transformKernel<T>, cudaFuncCachePreferL1) );

        transformKernel<T><<<grid, block, 0, stream>>>(src, dst, lut, tileSize, tilesX, tilesY);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template void transform<uchar>(PtrStepSz<uchar> src, PtrStepSz<uchar> dst, PtrStep<uchar> lut, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream);
    template void transform<ushort>(PtrStepSz<ushort> src, PtrStepSz<ushort> dst, PtrStep<ushort> lut, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream);
}

#endif // CUDA_DISABLER
