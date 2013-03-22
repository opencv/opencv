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

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        // Utility function to extract unsigned chars from an unsigned integer
        __device__ uchar4 int_to_uchar4(unsigned int in)
        {
            uchar4 bytes;
            bytes.x = (in & 0x000000ff) >>  0;
            bytes.y = (in & 0x0000ff00) >>  8;
            bytes.z = (in & 0x00ff0000) >> 16;
            bytes.w = (in & 0xff000000) >> 24;
            return bytes;
        }

        __global__ void shfl_integral_horizontal(const PtrStep<uint4> img, PtrStep<uint4> integral)
        {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
            __shared__ int sums[128];

            const int id = threadIdx.x;
            const int lane_id = id % warpSize;
            const int warp_id = id / warpSize;

            const uint4 data = img(blockIdx.x, id);

            const uchar4 a = int_to_uchar4(data.x);
            const uchar4 b = int_to_uchar4(data.y);
            const uchar4 c = int_to_uchar4(data.z);
            const uchar4 d = int_to_uchar4(data.w);

            int result[16];

            result[0]  =              a.x;
            result[1]  = result[0]  + a.y;
            result[2]  = result[1]  + a.z;
            result[3]  = result[2]  + a.w;

            result[4]  = result[3]  + b.x;
            result[5]  = result[4]  + b.y;
            result[6]  = result[5]  + b.z;
            result[7]  = result[6]  + b.w;

            result[8]  = result[7]  + c.x;
            result[9]  = result[8]  + c.y;
            result[10] = result[9]  + c.z;
            result[11] = result[10] + c.w;

            result[12] = result[11] + d.x;
            result[13] = result[12] + d.y;
            result[14] = result[13] + d.z;
            result[15] = result[14] + d.w;

            int sum = result[15];

            // the prefix sum for each thread's 16 value is computed,
            // now the final sums (result[15]) need to be shared
            // with the other threads and add.  To do this,
            // the __shfl_up() instruction is used and a shuffle scan
            // operation is performed to distribute the sums to the correct
            // threads
            #pragma unroll
            for (int i = 1; i < 32; i *= 2)
            {
                const int n = __shfl_up(sum, i, 32);

                if (lane_id >= i)
                {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        result[i] += n;

                    sum += n;
                }
            }

            // Now the final sum for the warp must be shared
            // between warps.  This is done by each warp
            // having a thread store to shared memory, then
            // having some other warp load the values and
            // compute a prefix sum, again by using __shfl_up.
            // The results are uniformly added back to the warps.
            // last thread in the warp holding sum of the warp
            // places that in shared
            if (threadIdx.x % warpSize == warpSize - 1)
                sums[warp_id] = result[15];

            __syncthreads();

            if (warp_id == 0)
            {
                int warp_sum = sums[lane_id];

                #pragma unroll
                for (int i = 1; i <= 32; i *= 2)
                {
                    const int n = __shfl_up(warp_sum, i, 32);

                    if (lane_id >= i)
                        warp_sum += n;
                }

                sums[lane_id] = warp_sum;
            }

            __syncthreads();

            int blockSum = 0;

            // fold in unused warp
            if (warp_id > 0)
            {
                blockSum = sums[warp_id - 1];

                #pragma unroll
                for (int i = 0; i < 16; ++i)
                    result[i] += blockSum;
            }

            // assemble result
            // Each thread has 16 values to write, which are
            // now integer data (to avoid overflow).  Instead of
            // each thread writing consecutive uint4s, the
            // approach shown here experiments using
            // the shuffle command to reformat the data
            // inside the registers so that each thread holds
            // consecutive data to be written so larger contiguous
            // segments can be assembled for writing.

            /*
                For example data that needs to be written as

                GMEM[16] <- x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3 w0 w1 w2 w3
                but is stored in registers (r0..r3), in four threads (0..3) as:

                threadId   0  1  2  3
                  r0      x0 y0 z0 w0
                  r1      x1 y1 z1 w1
                  r2      x2 y2 z2 w2
                  r3      x3 y3 z3 w3

                  after apply __shfl_xor operations to move data between registers r1..r3:

                threadId  00 01 10 11
                          x0 y0 z0 w0
                 xor(01)->y1 x1 w1 z1
                 xor(10)->z2 w2 x2 y2
                 xor(11)->w3 z3 y3 x3

                 and now x0..x3, and z0..z3 can be written out in order by all threads.

                 In the current code, each register above is actually representing
                 four integers to be written as uint4's to GMEM.
            */

            result[4]  = __shfl_xor(result[4] , 1, 32);
            result[5]  = __shfl_xor(result[5] , 1, 32);
            result[6]  = __shfl_xor(result[6] , 1, 32);
            result[7]  = __shfl_xor(result[7] , 1, 32);

            result[8]  = __shfl_xor(result[8] , 2, 32);
            result[9]  = __shfl_xor(result[9] , 2, 32);
            result[10] = __shfl_xor(result[10], 2, 32);
            result[11] = __shfl_xor(result[11], 2, 32);

            result[12] = __shfl_xor(result[12], 3, 32);
            result[13] = __shfl_xor(result[13], 3, 32);
            result[14] = __shfl_xor(result[14], 3, 32);
            result[15] = __shfl_xor(result[15], 3, 32);

            uint4* integral_row = integral.ptr(blockIdx.x);
            uint4 output;

            ///////

            if (threadIdx.x % 4 == 0)
                output = make_uint4(result[0], result[1], result[2], result[3]);

            if (threadIdx.x % 4 == 1)
                output = make_uint4(result[4], result[5], result[6], result[7]);

            if (threadIdx.x % 4 == 2)
                output = make_uint4(result[8], result[9], result[10], result[11]);

            if (threadIdx.x % 4 == 3)
                output = make_uint4(result[12], result[13], result[14], result[15]);

            integral_row[threadIdx.x % 4 + (threadIdx.x / 4) * 16] = output;

            ///////

            if (threadIdx.x % 4 == 2)
                output = make_uint4(result[0], result[1], result[2], result[3]);

            if (threadIdx.x % 4 == 3)
                output = make_uint4(result[4], result[5], result[6], result[7]);

            if (threadIdx.x % 4 == 0)
                output = make_uint4(result[8], result[9], result[10], result[11]);

            if (threadIdx.x % 4 == 1)
                output = make_uint4(result[12], result[13], result[14], result[15]);

            integral_row[(threadIdx.x + 2) % 4 + (threadIdx.x / 4) * 16 + 8] = output;

            // continuning from the above example,
            // this use of __shfl_xor() places the y0..y3 and w0..w3 data
            // in order.

            #pragma unroll
            for (int i = 0; i < 16; ++i)
                result[i] = __shfl_xor(result[i], 1, 32);

            if (threadIdx.x % 4 == 0)
                output = make_uint4(result[0], result[1], result[2], result[3]);

            if (threadIdx.x % 4 == 1)
                output = make_uint4(result[4], result[5], result[6], result[7]);

            if (threadIdx.x % 4 == 2)
                output = make_uint4(result[8], result[9], result[10], result[11]);

            if (threadIdx.x % 4 == 3)
                output = make_uint4(result[12], result[13], result[14], result[15]);

            integral_row[threadIdx.x % 4 + (threadIdx.x / 4) * 16 + 4] = output;

            ///////

            if (threadIdx.x % 4 == 2)
                output = make_uint4(result[0], result[1], result[2], result[3]);

            if (threadIdx.x % 4 == 3)
                output = make_uint4(result[4], result[5], result[6], result[7]);

            if (threadIdx.x % 4 == 0)
                output = make_uint4(result[8], result[9], result[10], result[11]);

            if (threadIdx.x % 4 == 1)
                output = make_uint4(result[12], result[13], result[14], result[15]);

            integral_row[(threadIdx.x + 2) % 4 + (threadIdx.x / 4) * 16 + 12] = output;
        #endif
        }

        // This kernel computes columnwise prefix sums.  When the data input is
        // the row sums from above, this completes the integral image.
        // The approach here is to have each block compute a local set of sums.
        // First , the data covered by the block is loaded into shared memory,
        // then instead of performing a sum in shared memory using __syncthreads
        // between stages, the data is reformatted so that the necessary sums
        // occur inside warps and the shuffle scan operation is used.
        // The final set of sums from the block is then propgated, with the block
        // computing "down" the image and adding the running sum to the local
        // block sums.
        __global__ void shfl_integral_vertical(PtrStepSz<unsigned int> integral)
        {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
            __shared__ unsigned int sums[32][9];

            const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
            const int lane_id = tidx % 8;

            if (tidx >= integral.cols)
                return;

            sums[threadIdx.x][threadIdx.y] = 0;
            __syncthreads();

            unsigned int stepSum = 0;

            for (int y = threadIdx.y; y < integral.rows; y += blockDim.y)
            {
                unsigned int* p = integral.ptr(y) + tidx;

                unsigned int sum = *p;

                sums[threadIdx.x][threadIdx.y] = sum;
                __syncthreads();

                // place into SMEM
                // shfl scan reduce the SMEM, reformating so the column
                // sums are computed in a warp
                // then read out properly
                const int j = threadIdx.x % 8;
                const int k = threadIdx.x / 8 + threadIdx.y * 4;

                int partial_sum = sums[k][j];

                for (int i = 1; i <= 8; i *= 2)
                {
                    int n = __shfl_up(partial_sum, i, 32);

                    if (lane_id >= i)
                        partial_sum += n;
                }

                sums[k][j] = partial_sum;
                __syncthreads();

                if (threadIdx.y > 0)
                    sum += sums[threadIdx.x][threadIdx.y - 1];

                sum += stepSum;
                stepSum += sums[threadIdx.x][blockDim.y - 1];

                __syncthreads();

                *p = sum;
            }
        #endif
        }

        void shfl_integral_gpu(const PtrStepSzb& img, PtrStepSz<unsigned int> integral, cudaStream_t stream)
        {
            {
                // each thread handles 16 values, use 1 block/row
                // save, becouse step is actually can't be less 512 bytes
                int block = integral.cols / 16;

                // launch 1 block / row
                const int grid = img.rows;

                cudaSafeCall( cudaFuncSetCacheConfig(shfl_integral_horizontal, cudaFuncCachePreferL1) );

                shfl_integral_horizontal<<<grid, block, 0, stream>>>((const PtrStepSz<uint4>) img, (PtrStepSz<uint4>) integral);
                cudaSafeCall( cudaGetLastError() );
            }

            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(integral.cols, block.x), 1);

                shfl_integral_vertical<<<grid, block, 0, stream>>>(integral);
                cudaSafeCall( cudaGetLastError() );
            }

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    }
}}}

#endif /* CUDA_DISABLER */
