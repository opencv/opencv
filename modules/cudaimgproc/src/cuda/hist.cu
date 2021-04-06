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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace hist
{
    __global__ void histogram256Kernel(const uchar* src, int cols, int rows, size_t step, int* hist)
    {
        __shared__ int shist[256];

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        shist[tid] = 0;
        __syncthreads();

        if (y < rows)
        {
            const unsigned int* rowPtr = (const unsigned int*) (src + y * step);

            const int cols_4 = cols / 4;
            for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
            {
                unsigned int data = rowPtr[x];

                Emulation::smem::atomicAdd(&shist[(data >>  0) & 0xFFU], 1);
                Emulation::smem::atomicAdd(&shist[(data >>  8) & 0xFFU], 1);
                Emulation::smem::atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
                Emulation::smem::atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
            }

            if (cols % 4 != 0 && threadIdx.x == 0)
            {
                for (int x = cols_4 * 4; x < cols; ++x)
                {
                    unsigned int data = ((const uchar*)rowPtr)[x];
                    Emulation::smem::atomicAdd(&shist[data], 1);
                }
            }
        }

        __syncthreads();

        const int histVal = shist[tid];
        if (histVal > 0)
            ::atomicAdd(hist + tid, histVal);
    }

    void histogram256(PtrStepSzb src, int* hist, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.rows, block.y));

        histogram256Kernel<<<grid, block, 0, stream>>>(src.data, src.cols, src.rows, src.step, hist);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    __global__ void histogram256Kernel(const uchar* src, int cols, int rows, size_t srcStep, const uchar* mask, size_t maskStep, int* hist)
    {
        __shared__ int shist[256];

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        shist[tid] = 0;
        __syncthreads();

        if (y < rows)
        {
            const unsigned int* rowPtr = (const unsigned int*) (src + y * srcStep);
            const unsigned int* maskRowPtr = (const unsigned int*) (mask + y * maskStep);

            const int cols_4 = cols / 4;
            for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
            {
                unsigned int data = rowPtr[x];
                unsigned int m = maskRowPtr[x];

                if ((m >>  0) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >>  0) & 0xFFU], 1);

                if ((m >>  8) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >>  8) & 0xFFU], 1);

                if ((m >>  16) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >> 16) & 0xFFU], 1);

                if ((m >>  24) & 0xFFU)
                    Emulation::smem::atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
            }

            if (cols % 4 != 0 && threadIdx.x == 0)
            {
                for (int x = cols_4 * 4; x < cols; ++x)
                {
                    unsigned int data = ((const uchar*)rowPtr)[x];
                    unsigned int m = ((const uchar*)maskRowPtr)[x];

                    if (m)
                        Emulation::smem::atomicAdd(&shist[data], 1);
                }
            }
        }

        __syncthreads();

        const int histVal = shist[tid];
        if (histVal > 0)
            ::atomicAdd(hist + tid, histVal);
    }

    void histogram256(PtrStepSzb src, PtrStepSzb mask, int* hist, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.rows, block.y));

        histogram256Kernel<<<grid, block, 0, stream>>>(src.data, src.cols, src.rows, src.step, mask.data, mask.step, hist);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

/////////////////////////////////////////////////////////////////////////

namespace hist
{
    __device__ __forceinline__ void histEvenInc(int* shist, uint data, int binSize, int lowerLevel, int upperLevel)
    {
        if (data >= lowerLevel && data <= upperLevel)
        {
            const uint ind = (data - lowerLevel) / binSize;
            Emulation::smem::atomicAdd(shist + ind, 1);
        }
    }

    __global__ void histEven8u(const uchar* src, const size_t step, const int rows, const int cols,
                               int* hist, const int binCount, const int binSize, const int lowerLevel, const int upperLevel)
    {
        extern __shared__ int shist[];

        const int y = blockIdx.x * blockDim.y + threadIdx.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (tid < binCount)
            shist[tid] = 0;

        __syncthreads();

        if (y < rows)
        {
            const uchar* rowPtr = src + y * step;
            const uint* rowPtr4 = (uint*) rowPtr;

            const int cols_4 = cols / 4;
            for (int x = threadIdx.x; x < cols_4; x += blockDim.x)
            {
                const uint data = rowPtr4[x];

                histEvenInc(shist, (data >>  0) & 0xFFU, binSize, lowerLevel, upperLevel);
                histEvenInc(shist, (data >>  8) & 0xFFU, binSize, lowerLevel, upperLevel);
                histEvenInc(shist, (data >> 16) & 0xFFU, binSize, lowerLevel, upperLevel);
                histEvenInc(shist, (data >> 24) & 0xFFU, binSize, lowerLevel, upperLevel);
            }

            if (cols % 4 != 0 && threadIdx.x == 0)
            {
                for (int x = cols_4 * 4; x < cols; ++x)
                {
                    const uchar data = rowPtr[x];
                    histEvenInc(shist, data, binSize, lowerLevel, upperLevel);
                }
            }
        }

        __syncthreads();

        if (tid < binCount)
        {
            const int histVal = shist[tid];

            if (histVal > 0)
                ::atomicAdd(hist + tid, histVal);
        }
    }

    void histEven8u(PtrStepSzb src, int* hist, int binCount, int lowerLevel, int upperLevel, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.rows, block.y));

        const int binSize = divUp(upperLevel - lowerLevel, binCount);

        const size_t smem_size = binCount * sizeof(int);

        histEven8u<<<grid, block, smem_size, stream>>>(src.data, src.step, src.rows, src.cols, hist, binCount, binSize, lowerLevel, upperLevel);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

/////////////////////////////////////////////////////////////////////////

namespace hist
{
    struct EqualizeHist : unary_function<uchar, uchar>
    {
        const uchar* lut;

        __host__ EqualizeHist(const uchar* _lut) : lut(_lut) {}

        __device__ __forceinline__ uchar operator ()(uchar val) const
        {
            return lut[val];
        }
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<hist::EqualizeHist> : DefaultTransformFunctorTraits<hist::EqualizeHist>
    {
        enum { smart_shift = 4 };
    };
}}}

namespace hist
{
    void equalizeHist(PtrStepSzb src, PtrStepSzb dst, const uchar* lut, cudaStream_t stream)
    {
        device::transform(src, dst, EqualizeHist(lut), WithOutMask(), stream);
    }

    __global__ void buildLutKernel(int* hist, unsigned char* lut, int size)
    {
        __shared__ int warp_smem[8];
        __shared__ int hist_smem[8][33];

#define HIST_SMEM_NO_BANK_CONFLICT(idx) hist_smem[(idx) >> 5][(idx) & 31]

        const int tId = threadIdx.x;
        const int warpId = threadIdx.x / 32;
        const int laneId = threadIdx.x % 32;

        // Step1 - Find minimum non-zero value in hist and make it zero
        HIST_SMEM_NO_BANK_CONFLICT(tId) = hist[tId];
        int nonZeroIdx = HIST_SMEM_NO_BANK_CONFLICT(tId) > 0 ? tId : 256;

        __syncthreads();

        for (int delta = 16; delta > 0; delta /= 2)
        {
#if __CUDACC_VER_MAJOR__ >= 9
            int shflVal = __shfl_down_sync(0xFFFFFFFF, nonZeroIdx, delta);
#else
            int shflVal = __shfl_down(nonZeroIdx, delta);
#endif
            if (laneId < delta)
                nonZeroIdx = min(nonZeroIdx, shflVal);
        }

        if (laneId == 0)
            warp_smem[warpId] = nonZeroIdx;

        __syncthreads();

        if (tId < 8)
        {
            int warpVal = warp_smem[tId];
            for (int delta = 4; delta > 0; delta /= 2)
            {
#if __CUDACC_VER_MAJOR__ >= 9
                int shflVal = __shfl_down_sync(0x000000FF, warpVal, delta);
#else
                int shflVal = __shfl_down(warpVal, delta);
#endif
                if (tId < delta)
                    warpVal = min(warpVal, shflVal);
            }
            if (tId == 0)
            {
                warp_smem[0] = warpVal; // warpVal - minimum index
            }
        }

        __syncthreads();

        const int minNonZeroIdx = warp_smem[0];
        const int minNonZeroVal = HIST_SMEM_NO_BANK_CONFLICT(minNonZeroIdx);
        if (minNonZeroVal == size)
        {
            // This is a special case: the whole image has the same color

            lut[tId] = 0;
            if (tId == minNonZeroIdx)
                lut[tId] = minNonZeroIdx;
            return;
        }

        if (tId == 0)
            HIST_SMEM_NO_BANK_CONFLICT(minNonZeroIdx) = 0;

        __syncthreads();

        // Step2 - Inclusive sum
        // Algorithm from GPU Gems 3 (A Work-Efficient Parallel Scan)
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

        // Step2 Phase1 - The Up-Sweep Phase
        for (int delta = 1; delta < 256; delta *= 2)
        {
            if (tId < 128 / delta)
            {
                int idx = 255 - 2 * tId * delta;
                HIST_SMEM_NO_BANK_CONFLICT(idx) += HIST_SMEM_NO_BANK_CONFLICT(idx - delta);
            }
            __syncthreads();
        }

        // Step2 Phase2 - The Down-Sweep Phase
        if (tId == 0)
            HIST_SMEM_NO_BANK_CONFLICT(255) = 0;

        for (int delta = 128; delta >= 1; delta /= 2)
        {
            if (tId < 128 / delta)
            {
                int rootIdx = 255 - tId * delta * 2;
                int leftIdx = rootIdx - delta;
                int tmp = HIST_SMEM_NO_BANK_CONFLICT(leftIdx);
                HIST_SMEM_NO_BANK_CONFLICT(leftIdx) = HIST_SMEM_NO_BANK_CONFLICT(rootIdx);
                HIST_SMEM_NO_BANK_CONFLICT(rootIdx) += tmp;
            }
            __syncthreads();
        }

        // Step2 Phase3 - Convert exclusive sum to inclusive sum
        int tmp = HIST_SMEM_NO_BANK_CONFLICT(tId);
        __syncthreads();
        if (tId >= 1)
            HIST_SMEM_NO_BANK_CONFLICT(tId - 1) = tmp;
        if (tId == 255)
            HIST_SMEM_NO_BANK_CONFLICT(tId) = tmp + hist[tId];
        __syncthreads();

        // Step3 - Scale values to build lut

        lut[tId] = saturate_cast<unsigned char>(HIST_SMEM_NO_BANK_CONFLICT(tId) * (255.0f / (size - minNonZeroVal)));

#undef HIST_SMEM_NO_BANK_CONFLICT
    }

    void buildLut(PtrStepSzi hist, PtrStepSzb lut, int size, cudaStream_t stream)
    {
        buildLutKernel<<<1, 256, 0, stream>>>(hist.data, lut.data, size);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

#endif /* CUDA_DISABLER */
