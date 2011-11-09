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

#include <utility>
#include <algorithm>
#include "internal_shared.hpp"

BEGIN_OPENCV_DEVICE_NAMESPACE

namespace canny {

__global__ void calcSobelRowPass(const PtrStepb src, PtrStepi dx_buf, PtrStepi dy_buf, int rows, int cols)
{
    __shared__ int smem[16][18];

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows)
    {
        smem[threadIdx.y][threadIdx.x + 1] = src.ptr(i)[j];
        if (threadIdx.x == 0)
        {
            smem[threadIdx.y][0] = src.ptr(i)[::max(j - 1, 0)];
            smem[threadIdx.y][17] = src.ptr(i)[::min(j + 16, cols - 1)];
        }
        __syncthreads();

        if (j < cols)
        {
            dx_buf.ptr(i)[j] = -smem[threadIdx.y][threadIdx.x] + smem[threadIdx.y][threadIdx.x + 2];
            dy_buf.ptr(i)[j] = smem[threadIdx.y][threadIdx.x] + 2 * smem[threadIdx.y][threadIdx.x + 1] + smem[threadIdx.y][threadIdx.x + 2];
        }
    }
}

void calcSobelRowPass_gpu(PtrStepb src, PtrStepi dx_buf, PtrStepi dy_buf, int rows, int cols)
{
    dim3 block(16, 16, 1);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), 1);

    calcSobelRowPass<<<grid, block>>>(src, dx_buf, dy_buf, rows, cols);
    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall(cudaThreadSynchronize());
}

struct L1
{
    static __device__ __forceinline__ float calc(int x, int y)
    {
        return ::abs(x) + ::abs(y);
    }
};
struct L2
{
    static __device__ __forceinline__ float calc(int x, int y)
    {
        return ::sqrtf(x * x + y * y);
    }
};

template <typename Norm> __global__ void calcMagnitude(const PtrStepi dx_buf, const PtrStepi dy_buf, 
    PtrStepi dx, PtrStepi dy, PtrStepf mag, int rows, int cols)
{
    __shared__ int sdx[18][16];
    __shared__ int sdy[18][16];

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < cols)
    {
        sdx[threadIdx.y + 1][threadIdx.x] = dx_buf.ptr(i)[j];
        sdy[threadIdx.y + 1][threadIdx.x] = dy_buf.ptr(i)[j];
        if (threadIdx.y == 0)
        {
            sdx[0][threadIdx.x] = dx_buf.ptr(::max(i - 1, 0))[j];
            sdx[17][threadIdx.x] = dx_buf.ptr(::min(i + 16, rows - 1))[j];

            sdy[0][threadIdx.x] = dy_buf.ptr(::max(i - 1, 0))[j];
            sdy[17][threadIdx.x] = dy_buf.ptr(::min(i + 16, rows - 1))[j];
        }
        __syncthreads();

        if (i < rows)
        {
            int x = sdx[threadIdx.y][threadIdx.x] + 2 * sdx[threadIdx.y + 1][threadIdx.x] + sdx[threadIdx.y + 2][threadIdx.x];
            int y = -sdy[threadIdx.y][threadIdx.x] + sdy[threadIdx.y + 2][threadIdx.x];

            dx.ptr(i)[j] = x;
            dy.ptr(i)[j] = y;

            mag.ptr(i + 1)[j + 1] = Norm::calc(x, y);
        }
    }
}

void calcMagnitude_gpu(PtrStepi dx_buf, PtrStepi dy_buf, PtrStepi dx, PtrStepi dy, PtrStepf mag, int rows, int cols, bool L2Grad)
{
    dim3 block(16, 16, 1);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), 1);

    if (L2Grad)
        calcMagnitude<L2><<<grid, block>>>(dx_buf, dy_buf, dx, dy, mag, rows, cols);
    else
        calcMagnitude<L1><<<grid, block>>>(dx_buf, dy_buf, dx, dy, mag, rows, cols);

    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall(cudaThreadSynchronize());
}

template <typename Norm> __global__ void calcMagnitude(PtrStepi dx, PtrStepi dy, PtrStepf mag, int rows, int cols)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
        mag.ptr(i + 1)[j + 1] = Norm::calc(dx.ptr(i)[j], dy.ptr(i)[j]);
}

void calcMagnitude_gpu(PtrStepi dx, PtrStepi dy, PtrStepf mag, int rows, int cols, bool L2Grad)
{
    dim3 block(16, 16, 1);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), 1);

    if (L2Grad)
        calcMagnitude<L2><<<grid, block>>>(dx, dy, mag, rows, cols);
    else
        calcMagnitude<L1><<<grid, block>>>(dx, dy, mag, rows, cols);

    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall(cudaThreadSynchronize());
}

//////////////////////////////////////////////////////////////////////////////////////////
    
#define CANNY_SHIFT 15
#define TG22        (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

__global__ void calcMap(const PtrStepi dx, const PtrStepi dy, const PtrStepf mag, PtrStepi map, int rows, int cols, float low_thresh, float high_thresh)
{
    __shared__ float smem[18][18];

    const int j = blockIdx.x * 16 + threadIdx.x;
    const int i = blockIdx.y * 16 + threadIdx.y;

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int lx = tid % 18;
    const int ly = tid / 18;

    if (ly < 14)
        smem[ly][lx] = mag.ptr(blockIdx.y * 16 + ly)[blockIdx.x * 16 + lx];

    if (ly < 4 && blockIdx.y * 16 + ly + 14 <= rows && blockIdx.x * 16 + lx <= cols)
        smem[ly + 14][lx] = mag.ptr(blockIdx.y * 16 + ly + 14)[blockIdx.x * 16 + lx];

    __syncthreads();

    if (i < rows && j < cols)
    {
        int x = dx.ptr(i)[j];
        int y = dy.ptr(i)[j];
        const int s = (x ^ y) < 0 ? -1 : 1;
        const float m = smem[threadIdx.y + 1][threadIdx.x + 1];

        x = ::abs(x);
        y = ::abs(y);

        // 0 - the pixel can not belong to an edge
        // 1 - the pixel might belong to an edge
        // 2 - the pixel does belong to an edge
        int edge_type = 0;

        if (m > low_thresh)
        {
            const int tg22x = x * TG22;
            const int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

            y <<= CANNY_SHIFT;

            if (y < tg22x)
            {
                if (m > smem[threadIdx.y + 1][threadIdx.x] && m >= smem[threadIdx.y + 1][threadIdx.x + 2])
                    edge_type = 1 + (int)(m > high_thresh);
            }
            else if( y > tg67x )
            {
                if (m > smem[threadIdx.y][threadIdx.x + 1] && m >= smem[threadIdx.y + 2][threadIdx.x + 1])
                    edge_type = 1 + (int)(m > high_thresh);
            }
            else
            {
                if (m > smem[threadIdx.y][threadIdx.x + 1 - s] && m > smem[threadIdx.y + 2][threadIdx.x + 1 + s])
                    edge_type = 1 + (int)(m > high_thresh);
            }
        }
        
        map.ptr(i + 1)[j + 1] = edge_type;
    }
}

#undef CANNY_SHIFT
#undef TG22

void calcMap_gpu(PtrStepi dx, PtrStepi dy, PtrStepf mag, PtrStepi map, int rows, int cols, float low_thresh, float high_thresh)
{
    dim3 block(16, 16, 1);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), 1);

    calcMap<<<grid, block>>>(dx, dy, mag, map, rows, cols, low_thresh, high_thresh);
    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall(cudaThreadSynchronize());
}

//////////////////////////////////////////////////////////////////////////////////////////

__device__ unsigned int counter = 0;

__global__ void edgesHysteresisLocal(PtrStepi map, ushort2* st, int rows, int cols)
{
    #if __CUDA_ARCH__ >= 120

    __shared__ int smem[18][18];

    const int j = blockIdx.x * 16 + threadIdx.x;
    const int i = blockIdx.y * 16 + threadIdx.y;

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int lx = tid % 18;
    const int ly = tid / 18; 

    if (ly < 14)
        smem[ly][lx] = map.ptr(blockIdx.y * 16 + ly)[blockIdx.x * 16 + lx];

    if (ly < 4 && blockIdx.y * 16 + ly + 14 <= rows && blockIdx.x * 16 + lx <= cols)
        smem[ly + 14][lx] = map.ptr(blockIdx.y * 16 + ly + 14)[blockIdx.x * 16 + lx];

    __syncthreads();

    if (i < rows && j < cols)
    {
        int n;

        #pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            n = 0;

            if (smem[threadIdx.y + 1][threadIdx.x + 1] == 1)
            {
                n += smem[threadIdx.y    ][threadIdx.x    ] == 2;
                n += smem[threadIdx.y    ][threadIdx.x + 1] == 2;
                n += smem[threadIdx.y    ][threadIdx.x + 2] == 2;
                
                n += smem[threadIdx.y + 1][threadIdx.x    ] == 2;
                n += smem[threadIdx.y + 1][threadIdx.x + 2] == 2;
                
                n += smem[threadIdx.y + 2][threadIdx.x    ] == 2;
                n += smem[threadIdx.y + 2][threadIdx.x + 1] == 2;
                n += smem[threadIdx.y + 2][threadIdx.x + 2] == 2;
            }

            if (n > 0)
                smem[threadIdx.y + 1][threadIdx.x + 1] = 2;
        }

        const int e = smem[threadIdx.y + 1][threadIdx.x + 1];

        map.ptr(i + 1)[j + 1] = e;

        n = 0;

        if (e == 2)
        {
            n += smem[threadIdx.y    ][threadIdx.x    ] == 1;
            n += smem[threadIdx.y    ][threadIdx.x + 1] == 1;
            n += smem[threadIdx.y    ][threadIdx.x + 2] == 1;
            
            n += smem[threadIdx.y + 1][threadIdx.x    ] == 1;
            n += smem[threadIdx.y + 1][threadIdx.x + 2] == 1;
            
            n += smem[threadIdx.y + 2][threadIdx.x    ] == 1;
            n += smem[threadIdx.y + 2][threadIdx.x + 1] == 1;
            n += smem[threadIdx.y + 2][threadIdx.x + 2] == 1;
        }

        if (n > 0)
        {
            const unsigned int ind = atomicInc(&counter, (unsigned int)(-1));
            st[ind] = make_ushort2(j + 1, i + 1);
        }
    }

    #endif
}

void edgesHysteresisLocal_gpu(PtrStepi map, ushort2* st1, int rows, int cols)
{
    dim3 block(16, 16, 1);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), 1);

    edgesHysteresisLocal<<<grid, block>>>(map, st1, rows, cols);
    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall(cudaThreadSynchronize());
}

__constant__ int c_dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
__constant__ int c_dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

__global__ void edgesHysteresisGlobal(PtrStepi map, ushort2* st1, ushort2* st2, int rows, int cols, int count)
{
    #if __CUDA_ARCH__ >= 120

    const int stack_size = 512;
    
    __shared__ unsigned int s_counter;
    __shared__ unsigned int s_ind;
    __shared__ ushort2 s_st[stack_size];

    if (threadIdx.x == 0)
        s_counter = 0;
    __syncthreads();

    int ind = blockIdx.y * gridDim.x + blockIdx.x;

    if (ind < count)
    {
        ushort2 pos = st1[ind];

        if (pos.x > 0 && pos.x <= cols && pos.y > 0 && pos.y <= rows)
        {
            if (threadIdx.x < 8)
            {
                pos.x += c_dx[threadIdx.x];
                pos.y += c_dy[threadIdx.x];

                if (map.ptr(pos.y)[pos.x] == 1)
                {
                    map.ptr(pos.y)[pos.x] = 2;

                    ind = atomicInc(&s_counter, (unsigned int)(-1));

                    s_st[ind] = pos;
                }
            }
            __syncthreads();

            while (s_counter > 0 && s_counter <= stack_size - blockDim.x)
            {
                const int subTaskIdx = threadIdx.x >> 3;
                const int portion = ::min(s_counter, blockDim.x >> 3);

                pos.x = pos.y = 0;

                if (subTaskIdx < portion)
                    pos = s_st[s_counter - 1 - subTaskIdx];
                __syncthreads();
                    
                if (threadIdx.x == 0)
                    s_counter -= portion;
                __syncthreads();
                 
                if (pos.x > 0 && pos.x <= cols && pos.y > 0 && pos.y <= rows)
                {
                    pos.x += c_dx[threadIdx.x & 7];
                    pos.y += c_dy[threadIdx.x & 7];

                    if (map.ptr(pos.y)[pos.x] == 1)
                    {
                        map.ptr(pos.y)[pos.x] = 2;

                        ind = atomicInc(&s_counter, (unsigned int)(-1));

                        s_st[ind] = pos;
                    }
                }
                __syncthreads();
            }

            if (s_counter > 0)
            {
                if (threadIdx.x == 0)
                {
                    ind = atomicAdd(&counter, s_counter);
                    s_ind = ind - s_counter;
                }
                __syncthreads();

                ind = s_ind;

                for (int i = threadIdx.x; i < s_counter; i += blockDim.x)
                {
                    st2[ind + i] = s_st[i];
                }
            }
        }
    }

    #endif
}

void edgesHysteresisGlobal_gpu(PtrStepi map, ushort2* st1, ushort2* st2, int rows, int cols)
{
    void* counter_ptr;
    cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, counter) );
    
    unsigned int count;
    cudaSafeCall( cudaMemcpy(&count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    while (count > 0)
    {
        cudaSafeCall( cudaMemset(counter_ptr, 0, sizeof(unsigned int)) );

        dim3 block(128, 1, 1);
        dim3 grid(min(count, 65535u), divUp(count, 65535), 1);
        edgesHysteresisGlobal<<<grid, block>>>(map, st1, st2, rows, cols, count);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall(cudaThreadSynchronize());

        cudaSafeCall( cudaMemcpy(&count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

        std::swap(st1, st2);
    }
}

__global__ void getEdges(PtrStepi map, PtrStepb dst, int rows, int cols)
{
    const int j = blockIdx.x * 16 + threadIdx.x;
    const int i = blockIdx.y * 16 + threadIdx.y;

    if (i < rows && j < cols)
        dst.ptr(i)[j] = (uchar)(-(map.ptr(i + 1)[j + 1] >> 1));
}

void getEdges_gpu(PtrStepi map, PtrStepb dst, int rows, int cols)
{
    dim3 block(16, 16, 1);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), 1);

    getEdges<<<grid, block>>>(map, dst, rows, cols);
    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall(cudaThreadSynchronize());
}

} // namespace canny

END_OPENCV_DEVICE_NAMESPACE
