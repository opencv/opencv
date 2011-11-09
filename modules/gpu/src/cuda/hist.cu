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
// Copyright (C) 1993-2011, NVIDIA Corporation, all rights reserved.
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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"

BEGIN_OPENCV_DEVICE_NAMESPACE

#define UINT_BITS 32U

//Warps == subhistograms per threadblock
#define WARP_COUNT 6

//Threadblock size
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * OPENCV_GPU_WARP_SIZE)
#define HISTOGRAM256_BIN_COUNT 256

//Shared memory per threadblock
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)

#define PARTIAL_HISTOGRAM256_COUNT 240

#define MERGE_THREADBLOCK_SIZE 256

#define USE_SMEM_ATOMICS (__CUDA_ARCH__ >= 120)

namespace hist {

#if (!USE_SMEM_ATOMICS)

    #define TAG_MASK ( (1U << (UINT_BITS - OPENCV_GPU_LOG_WARP_SIZE)) - 1U )

    __forceinline__ __device__ void addByte(volatile uint* s_WarpHist, uint data, uint threadTag)
    {
        uint count;
        do
        {
            count = s_WarpHist[data] & TAG_MASK;
            count = threadTag | (count + 1);
            s_WarpHist[data] = count;
        } while (s_WarpHist[data] != count);
    }

#else

    #define TAG_MASK 0xFFFFFFFFU

    __forceinline__ __device__ void addByte(uint* s_WarpHist, uint data, uint threadTag)
    {
        atomicAdd(s_WarpHist + data, 1);
    }

#endif

__forceinline__ __device__ void addWord(uint* s_WarpHist, uint data, uint tag, uint pos_x, uint cols)
{
    uint x = pos_x << 2;

    if (x + 0 < cols) addByte(s_WarpHist, (data >>  0) & 0xFFU, tag);
    if (x + 1 < cols) addByte(s_WarpHist, (data >>  8) & 0xFFU, tag);
    if (x + 2 < cols) addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    if (x + 3 < cols) addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram256(const PtrStep<uint> d_Data, uint* d_PartialHistograms, uint dataCount, uint cols)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint* s_WarpHist= s_Hist + (threadIdx.x >> OPENCV_GPU_LOG_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
    #pragma unroll
    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
       s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - OPENCV_GPU_LOG_WARP_SIZE);

    __syncthreads();
    const uint colsui = d_Data.step / sizeof(uint);
    for(uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < dataCount; pos += blockDim.x * gridDim.x)
    {
        uint pos_y = pos / colsui;
        uint pos_x = pos % colsui;
        uint data = d_Data.ptr(pos_y)[pos_x];
        addWord(s_WarpHist, data, tag, pos_x, cols);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();
    for(uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////

__global__ void mergeHistogram256(const uint* d_PartialHistograms, int* d_Histogram)
{
    uint sum = 0;

    #pragma unroll
    for (uint i = threadIdx.x; i < PARTIAL_HISTOGRAM256_COUNT; i += MERGE_THREADBLOCK_SIZE)
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];

    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if(threadIdx.x < stride)
            data[threadIdx.x] += data[threadIdx.x + stride];
    }

    if(threadIdx.x == 0)
        d_Histogram[blockIdx.x] = saturate_cast<int>(data[0]);
}

void histogram256_gpu(DevMem2Db src, int* hist, uint* buf, cudaStream_t stream)
{
    histogram256<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE, 0, stream>>>(
        DevMem2D_<uint>(src),
        buf, 
        static_cast<uint>(src.rows * src.step / sizeof(uint)),
        src.cols);

    cudaSafeCall( cudaGetLastError() );

    mergeHistogram256<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE, 0, stream>>>(buf, hist);

    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

__constant__ int c_lut[256];

__global__ void equalizeHist(const DevMem2Db src, PtrStepb dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < src.cols && y < src.rows)
    {
        const uchar val = src.ptr(y)[x];
        const int lut = c_lut[val];
        dst.ptr(y)[x] = __float2int_rn(255.0f / (src.cols * src.rows) * lut);
    }
}

void equalizeHist_gpu(DevMem2Db src, DevMem2Db dst, const int* lut, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    cudaSafeCall( cudaMemcpyToSymbol(c_lut, lut, 256 * sizeof(int), 0, cudaMemcpyDeviceToDevice) );

    equalizeHist<<<grid, block, 0, stream>>>(src, dst);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

} // namespace hist

END_OPENCV_DEVICE_NAMESPACE
