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
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/reduce.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace optflowbm
{
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_prev(false, cudaFilterModePoint, cudaAddressModeClamp);
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_curr(false, cudaFilterModePoint, cudaAddressModeClamp);

    __device__ int cmpBlocks(int X1, int Y1, int X2, int Y2, int2 blockSize)
    {
        int s = 0;

        for (int y = 0; y < blockSize.y; ++y)
        {
            for (int x = 0; x < blockSize.x; ++x)
                s += ::abs(tex2D(tex_prev, X1 + x, Y1 + y) - tex2D(tex_curr, X2 + x, Y2 + y));
        }

        return s;
    }

    __global__ void calcOptFlowBM(PtrStepSzf velx, PtrStepf vely, const int2 blockSize, const int2 shiftSize, const bool usePrevious,
                                  const int maxX, const int maxY, const int acceptLevel, const int escapeLevel,
                                  const short2* ss, const int ssCount)
    {
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        const int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= velx.rows || j >= velx.cols)
            return;

        const int X1 = j * shiftSize.x;
        const int Y1 = i * shiftSize.y;

        const int offX = usePrevious ? __float2int_rn(velx(i, j)) : 0;
        const int offY = usePrevious ? __float2int_rn(vely(i, j)) : 0;

        int X2 = X1 + offX;
        int Y2 = Y1 + offY;

        int dist = numeric_limits<int>::max();

        if (0 <= X2 && X2 <= maxX && 0 <= Y2 && Y2 <= maxY)
            dist = cmpBlocks(X1, Y1, X2, Y2, blockSize);

        int countMin = 1;
        int sumx = offX;
        int sumy = offY;

        if (dist > acceptLevel)
        {
            // do brute-force search
            for (int k = 0; k < ssCount; ++k)
            {
                const short2 ssVal = ss[k];

                const int dx = offX + ssVal.x;
                const int dy = offY + ssVal.y;

                X2 = X1 + dx;
                Y2 = Y1 + dy;

                if (0 <= X2 && X2 <= maxX && 0 <= Y2 && Y2 <= maxY)
                {
                    const int tmpDist = cmpBlocks(X1, Y1, X2, Y2, blockSize);
                    if (tmpDist < acceptLevel)
                    {
                        sumx = dx;
                        sumy = dy;
                        countMin = 1;
                        break;
                    }

                    if (tmpDist < dist)
                    {
                        dist = tmpDist;
                        sumx = dx;
                        sumy = dy;
                        countMin = 1;
                    }
                    else if (tmpDist == dist)
                    {
                        sumx += dx;
                        sumy += dy;
                        countMin++;
                    }
                }
            }

            if (dist > escapeLevel)
            {
                sumx = offX;
                sumy = offY;
                countMin = 1;
            }
        }

        velx(i, j) = static_cast<float>(sumx) / countMin;
        vely(i, j) = static_cast<float>(sumy) / countMin;
    }

    void calc(PtrStepSzb prev, PtrStepSzb curr, PtrStepSzf velx, PtrStepSzf vely, int2 blockSize, int2 shiftSize, bool usePrevious,
              int maxX, int maxY, int acceptLevel, int escapeLevel, const short2* ss, int ssCount, cudaStream_t stream)
    {
        bindTexture(&tex_prev, prev);
        bindTexture(&tex_curr, curr);

        const dim3 block(32, 8);
        const dim3 grid(divUp(velx.cols, block.x), divUp(vely.rows, block.y));

        calcOptFlowBM<<<grid, block, 0, stream>>>(velx, vely, blockSize, shiftSize, usePrevious,
                                                  maxX, maxY, acceptLevel,  escapeLevel, ss, ssCount);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

/////////////////////////////////////////////////////////
// Fast approximate version

namespace optflowbm_fast
{
    enum
    {
        CTA_SIZE = 128,

        TILE_COLS = 128,
        TILE_ROWS = 32,

        STRIDE = CTA_SIZE
    };

    template <typename T> __device__ __forceinline__ int calcDist(T a, T b)
    {
        return ::abs(a - b);
    }

    template <class T> struct FastOptFlowBM
    {

        int search_radius;
        int block_radius;

        int search_window;
        int block_window;

        PtrStepSz<T> I0;
        PtrStep<T> I1;

        mutable PtrStepi buffer;

        FastOptFlowBM(int search_window_, int block_window_,
                      PtrStepSz<T> I0_, PtrStepSz<T> I1_,
                      PtrStepi buffer_) :
            search_radius(search_window_ / 2), block_radius(block_window_ / 2),
            search_window(search_window_), block_window(block_window_),
            I0(I0_), I1(I1_),
            buffer(buffer_)
        {
        }

        __device__ __forceinline__ void initSums_BruteForce(int i, int j, int* dist_sums, PtrStepi& col_sums, PtrStepi& up_col_sums) const
        {
            for (int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
            {
                dist_sums[index] = 0;

                for (int tx = 0; tx < block_window; ++tx)
                    col_sums(tx, index) = 0;

                int y = index / search_window;
                int x = index - y * search_window;

                int ay = i;
                int ax = j;

                int by = i + y - search_radius;
                int bx = j + x - search_radius;

                for (int tx = -block_radius; tx <= block_radius; ++tx)
                {
                    int col_sum = 0;
                    for (int ty = -block_radius; ty <= block_radius; ++ty)
                    {
                        int dist = calcDist(I0(ay + ty, ax + tx), I1(by + ty, bx + tx));

                        dist_sums[index] += dist;
                        col_sum += dist;
                    }

                    col_sums(tx + block_radius, index) = col_sum;
                }

                up_col_sums(j, index) = col_sums(block_window - 1, index);
            }
        }

        __device__ __forceinline__ void shiftRight_FirstRow(int i, int j, int first, int* dist_sums, PtrStepi& col_sums, PtrStepi& up_col_sums) const
        {
            for (int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
            {
                int y = index / search_window;
                int x = index - y * search_window;

                int ay = i;
                int ax = j + block_radius;

                int by = i + y - search_radius;
                int bx = j + x - search_radius + block_radius;

                int col_sum = 0;

                for (int ty = -block_radius; ty <= block_radius; ++ty)
                    col_sum += calcDist(I0(ay + ty, ax), I1(by + ty, bx));

                dist_sums[index] += col_sum - col_sums(first, index);

                col_sums(first, index) = col_sum;
                up_col_sums(j, index) = col_sum;
            }
        }

        __device__ __forceinline__ void shiftRight_UpSums(int i, int j, int first, int* dist_sums, PtrStepi& col_sums, PtrStepi& up_col_sums) const
        {
            int ay = i;
            int ax = j + block_radius;

            T a_up   = I0(ay - block_radius - 1, ax);
            T a_down = I0(ay + block_radius, ax);

            for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
            {
                int y = index / search_window;
                int x = index - y * search_window;

                int by = i + y - search_radius;
                int bx = j + x - search_radius + block_radius;

                T b_up   = I1(by - block_radius - 1, bx);
                T b_down = I1(by + block_radius, bx);

                int col_sum = up_col_sums(j, index) + calcDist(a_down, b_down) - calcDist(a_up, b_up);

                dist_sums[index] += col_sum  - col_sums(first, index);
                col_sums(first, index) = col_sum;
                up_col_sums(j, index) = col_sum;
            }
        }

        __device__ __forceinline__ void convolve_window(int i, int j, const int* dist_sums, float& velx, float& vely) const
        {
            int bestDist = numeric_limits<int>::max();
            int bestInd = -1;

            for (int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
            {
                int curDist = dist_sums[index];
                if (curDist < bestDist)
                {
                    bestDist = curDist;
                    bestInd = index;
                }
            }

            __shared__ int cta_dist_buffer[CTA_SIZE];
            __shared__ int cta_ind_buffer[CTA_SIZE];

            reduceKeyVal<CTA_SIZE>(cta_dist_buffer, bestDist, cta_ind_buffer, bestInd, threadIdx.x, less<int>());

            if (threadIdx.x == 0)
            {
                int y = bestInd / search_window;
                int x = bestInd - y * search_window;

                velx = x - search_radius;
                vely = y - search_radius;
            }
        }

        __device__ __forceinline__ void operator()(PtrStepf velx, PtrStepf vely) const
        {
            int tbx = blockIdx.x * TILE_COLS;
            int tby = blockIdx.y * TILE_ROWS;

            int tex = ::min(tbx + TILE_COLS, I0.cols);
            int tey = ::min(tby + TILE_ROWS, I0.rows);

            PtrStepi col_sums;
            col_sums.data = buffer.ptr(I0.cols + blockIdx.x * block_window) + blockIdx.y * search_window * search_window;
            col_sums.step = buffer.step;

            PtrStepi up_col_sums;
            up_col_sums.data = buffer.data + blockIdx.y * search_window * search_window;
            up_col_sums.step = buffer.step;

            extern __shared__ int dist_sums[]; //search_window * search_window

            int first = 0;

            for (int i = tby; i < tey; ++i)
            {
                for (int j = tbx; j < tex; ++j)
                {
                    __syncthreads();

                    if (j == tbx)
                    {
                        initSums_BruteForce(i, j, dist_sums, col_sums, up_col_sums);
                        first = 0;
                    }
                    else
                    {
                        if (i == tby)
                          shiftRight_FirstRow(i, j, first, dist_sums, col_sums, up_col_sums);
                        else
                          shiftRight_UpSums(i, j, first, dist_sums, col_sums, up_col_sums);

                        first = (first + 1) % block_window;
                    }

                    __syncthreads();

                    convolve_window(i, j, dist_sums, velx(i, j), vely(i, j));
                }
            }
        }

    };

    template<typename T> __global__ void optflowbm_fast_kernel(const FastOptFlowBM<T> fbm, PtrStepf velx, PtrStepf vely)
    {
        fbm(velx, vely);
    }

    void get_buffer_size(int src_cols, int src_rows, int search_window, int block_window, int& buffer_cols, int& buffer_rows)
    {
        dim3 grid(divUp(src_cols, TILE_COLS), divUp(src_rows, TILE_ROWS));

        buffer_cols = search_window * search_window * grid.y;
        buffer_rows = src_cols + block_window * grid.x;
    }

    template <typename T>
    void calc(PtrStepSzb I0, PtrStepSzb I1, PtrStepSzf velx, PtrStepSzf vely, PtrStepi buffer, int search_window, int block_window, cudaStream_t stream)
    {
        FastOptFlowBM<T> fbm(search_window, block_window, I0, I1, buffer);

        dim3 block(CTA_SIZE, 1);
        dim3 grid(divUp(I0.cols, TILE_COLS), divUp(I0.rows, TILE_ROWS));

        size_t smem = search_window * search_window * sizeof(int);

        optflowbm_fast_kernel<<<grid, block, smem, stream>>>(fbm, velx, vely);
        cudaSafeCall ( cudaGetLastError () );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void calc<uchar>(PtrStepSzb I0, PtrStepSzb I1, PtrStepSzf velx, PtrStepSzf vely, PtrStepi buffer, int search_window, int block_window, cudaStream_t stream);
}

#endif // !defined CUDA_DISABLER
