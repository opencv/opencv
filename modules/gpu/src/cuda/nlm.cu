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

#if !defined CUDA_DISABLER

#include "internal_shared.hpp"

#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/block.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

using namespace cv::gpu;

typedef unsigned char uchar;
typedef unsigned short ushort;

//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        __device__ __forceinline__ float norm2(const float& v) { return v*v; }
        __device__ __forceinline__ float norm2(const float2& v) { return v.x*v.x + v.y*v.y; }
        __device__ __forceinline__ float norm2(const float3& v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
        __device__ __forceinline__ float norm2(const float4& v) { return v.x*v.x + v.y*v.y + v.z*v.z  + v.w*v.w; }

        template<typename T, typename B>
        __global__ void nlm_kernel(const PtrStepSz<T> src, PtrStep<T> dst, const B b, int search_radius, int block_radius, float h2_inv_half)
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type value_type;

            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= src.cols || y >= src.rows)
                return;

            float block_radius2_inv = -1.f/(block_radius * block_radius);

            value_type sum1 = VecTraits<value_type>::all(0);
            float sum2 = 0.f;

            if (x - search_radius - block_radius >=0        && y - search_radius - block_radius >=0 &&
                x + search_radius + block_radius < src.cols && y + search_radius + block_radius < src.rows)
            {

                for(float cy = -search_radius; cy <= search_radius; ++cy)
                    for(float cx = -search_radius; cx <= search_radius; ++cx)
                    {
                        float color2 = 0;
                        for(float by = -block_radius; by <= block_radius; ++by)
                            for(float bx = -block_radius; bx <= block_radius; ++bx)
                            {
                                value_type v1 = saturate_cast<value_type>(src(y +      by, x +      bx));
                                value_type v2 = saturate_cast<value_type>(src(y + cy + by, x + cx + bx));
                                color2 += norm2(v1 - v2);
                            }

                        float dist2 = cx * cx + cy * cy;
                        float w = __expf(color2 * h2_inv_half + dist2 * block_radius2_inv);

                        sum1 = sum1 + saturate_cast<value_type>(src(y + cy, x + cy)) * w;
                        sum2 += w;
                    }
            }
            else
            {
                for(float cy = -search_radius; cy <= search_radius; ++cy)
                    for(float cx = -search_radius; cx <= search_radius; ++cx)
                    {
                        float color2 = 0;
                        for(float by = -block_radius; by <= block_radius; ++by)
                            for(float bx = -block_radius; bx <= block_radius; ++bx)
                            {
                                value_type v1 = saturate_cast<value_type>(b.at(y +      by, x +      bx, src.data, src.step));
                                value_type v2 = saturate_cast<value_type>(b.at(y + cy + by, x + cx + bx, src.data, src.step));
                                color2 += norm2(v1 - v2);
                            }

                        float dist2 = cx * cx + cy * cy;
                        float w = __expf(color2 * h2_inv_half + dist2 * block_radius2_inv);

                        sum1 = sum1 + saturate_cast<value_type>(b.at(y + cy, x + cy, src.data, src.step)) * w;
                        sum2 += w;
                    }

            }

            dst(y, x) = saturate_cast<T>(sum1 / sum2);

        }

        template<typename T, template <typename> class B>
        void nlm_caller(const PtrStepSzb src, PtrStepSzb dst, int search_radius, int block_radius, float h, cudaStream_t stream)
        {
            dim3 block (32, 8);
            dim3 grid (divUp (src.cols, block.x), divUp (src.rows, block.y));

            B<T> b(src.rows, src.cols);

            float h2_inv_half = -0.5f/(h * h * VecTraits<T>::cn);

            cudaSafeCall( cudaFuncSetCacheConfig (nlm_kernel<T, B<T> >, cudaFuncCachePreferL1) );
            nlm_kernel<<<grid, block>>>((PtrStepSz<T>)src, (PtrStepSz<T>)dst, b, search_radius, block_radius, h2_inv_half);
            cudaSafeCall ( cudaGetLastError () );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template<typename T>
        void nlm_bruteforce_gpu(const PtrStepSzb& src, PtrStepSzb dst, int search_radius, int block_radius, float h, int borderMode, cudaStream_t stream)
        {
            typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb dst, int search_radius, int block_radius, float h, cudaStream_t stream);

            static func_t funcs[] = 
            {
                nlm_caller<T, BrdReflect101>,
                nlm_caller<T, BrdReplicate>,
                nlm_caller<T, BrdConstant>,
                nlm_caller<T, BrdReflect>,
                nlm_caller<T, BrdWrap>,
            };
            funcs[borderMode](src, dst, search_radius, block_radius, h, stream);
        }

        template void nlm_bruteforce_gpu<uchar>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
        template void nlm_bruteforce_gpu<uchar2>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
        template void nlm_bruteforce_gpu<uchar3>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
    }
}}}

//////////////////////////////////////////////////////////////////////////////////
//// Non Local Means Denosing (fast approximate version)

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {  
        __device__ __forceinline__ int calcDist(const uchar&  a, const uchar&  b) { return (a-b)*(a-b); }
        __device__ __forceinline__ int calcDist(const uchar2& a, const uchar2& b) { return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y); }
        __device__ __forceinline__ int calcDist(const uchar3& a, const uchar3& b) { return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z); }



        template <class T> struct FastNonLocalMenas
        {
            enum
            {
                CTA_SIZE = 256,

                //TILE_COLS = 256,
                //TILE_ROWS = 32,

                TILE_COLS = 256,
                TILE_ROWS = 32,

                STRIDE = CTA_SIZE
            };

            struct plus
            {
                __device__ __forceinline float operator()(float v1, float v2) const { return v1 + v2; }
            };

            int search_radius;
            int block_radius;

            int search_window;
            int block_window;
            float minus_h2_inv;

            FastNonLocalMenas(int search_window_, int block_window_, float h) : search_radius(search_window_/2), block_radius(block_window_/2),
                search_window(search_window_), block_window(block_window_), minus_h2_inv(-1.f/(h * h * VecTraits<T>::cn)) {}

            PtrStep<T> src;
            mutable PtrStepi buffer;
            
            __device__ __forceinline__ void initSums_TileFistColumn(int i, int j, int* dist_sums, PtrStepi& col_dist_sums, PtrStepi& up_col_dist_sums) const
            {
                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    dist_sums[index] = 0;

                    for(int tx = 0; tx < block_window; ++tx)
                        col_dist_sums(tx, index) = 0;

                    int y = index / search_window;
                    int x = index - y * search_window;

                    int ay = i;
                    int ax = j;

                    int by = i + y - search_radius;
                    int bx = j + x - search_radius;

#if 1
                    for (int tx = -block_radius; tx <= block_radius; ++tx)
                    {
                        int col_dist_sums_tx_block_radius_index = 0;

                        for (int ty = -block_radius; ty <= block_radius; ++ty)
                        {
                            int dist = calcDist(src(ay + ty, ax + tx), src(by + ty, bx + tx));

                            dist_sums[index] += dist;
                            col_dist_sums_tx_block_radius_index += dist;
                        }

                        col_dist_sums(tx + block_radius, index) = col_dist_sums_tx_block_radius_index;
                    }
#else
                    for (int ty = -block_radius; ty <= block_radius; ++ty)
                        for (int tx = -block_radius; tx <= block_radius; ++tx)
                        {
                            int dist = calcDist(src(ay + ty, ax + tx), src(by + ty, bx + tx));

                            dist_sums[index] += dist;
                            col_dist_sums(tx + block_radius, index) += dist;                            
                        }
#endif

                    up_col_dist_sums(j, index) = col_dist_sums(block_window - 1, index);
                }
            }

            __device__ __forceinline__ void shiftLeftSums_TileFirstRow(int i, int j, int first_col, int* dist_sums, PtrStepi& col_dist_sums, PtrStepi& up_col_dist_sums) const
            {              
                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {                                        
                    int y = index / search_window;
                    int x = index - y * search_window;

                    int ay = i;
                    int ax = j + block_radius;

                    int by = i + y - search_radius;
                    int bx = j + x - search_radius + block_radius;

                    int col_dist_sum = 0;

                    for (int ty = -block_radius; ty <= block_radius; ++ty)
                        col_dist_sum += calcDist(src(ay + ty, ax), src(by + ty, bx));                  

                    int old_dist_sums = dist_sums[index];
                    int old_col_sum = col_dist_sums(first_col, index);
                    dist_sums[index] += col_dist_sum - old_col_sum;

                    
                    col_dist_sums(first_col, index) = col_dist_sum;
                    up_col_dist_sums(j, index) = col_dist_sum;
                }
            }

            __device__ __forceinline__ void shiftLeftSums_UsingUpSums(int i, int j, int first_col, int* dist_sums, PtrStepi& col_dist_sums, PtrStepi& up_col_dist_sums) const
            {
                int ay = i;
                int ax = j + block_radius;

                int start_by = i - search_radius;
                int start_bx = j - search_radius + block_radius;

                T a_up   = src(ay - block_radius - 1, ax);
                T a_down = src(ay + block_radius, ax);

                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    dist_sums[index] -= col_dist_sums(first_col, index);

                    int y = index / search_window;
                    int x = index - y * search_window;

                    int by = start_by + y;
                    int bx = start_bx + x;

                    T b_up   = src(by - block_radius - 1, bx);
                    T b_down = src(by + block_radius, bx);

                    int col_dist_sums_first_col_index = up_col_dist_sums(j, index) + calcDist(a_down, b_down) - calcDist(a_up, b_up);

                    col_dist_sums(first_col, index) = col_dist_sums_first_col_index;
                    dist_sums[index] += col_dist_sums_first_col_index;
                    up_col_dist_sums(j, index) = col_dist_sums_first_col_index;
                }
            }

            __device__ __forceinline__ void convolve_search_window(int i, int j, const int* dist_sums, PtrStepi& col_dist_sums, PtrStepi& up_col_dist_sums, T& dst) const
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_type;

                float weights_sum = 0;
                sum_type sum = VecTraits<sum_type>::all(0);

                float bw2_inv = 1.f/(block_window * block_window);

                int start_x = j - search_radius;
                int start_y = i - search_radius;

                for(int index = threadIdx.x; index < search_window * search_window; index += STRIDE)
                {
                    int y = index / search_window;
                    int x = index - y * search_window;

                    float avg_dist = dist_sums[index] * bw2_inv;
                    float weight = __expf(avg_dist * minus_h2_inv);
                    weights_sum += weight;

                    sum = sum + weight * saturate_cast<sum_type>(src(start_y + y, start_x + x));
                }
                
                volatile __shared__ float cta_buffer[CTA_SIZE];
                
                int tid = threadIdx.x;

                cta_buffer[tid] = weights_sum;
                __syncthreads();
                Block::reduce<CTA_SIZE>(cta_buffer, plus());

                if (tid == 0)
                    weights_sum = cta_buffer[0];

                __syncthreads();

                for(int n = 0; n < VecTraits<T>::cn; ++n)
                {
                    cta_buffer[tid] = reinterpret_cast<float*>(&sum)[n];
                    __syncthreads();
                    Block::reduce<CTA_SIZE>(cta_buffer, plus());
                    
                    if (tid == 0)
                      reinterpret_cast<float*>(&sum)[n] = cta_buffer[0];
                    __syncthreads();
                }

                if (tid == 0)
                    dst = saturate_cast<T>(sum/weights_sum);
            }

            __device__ __forceinline__ void operator()(PtrStepSz<T>& dst) const
            {
                int tbx = blockIdx.x * TILE_COLS;
                int tby = blockIdx.y * TILE_ROWS;

                int tex = ::min(tbx + TILE_COLS, dst.cols);
                int tey = ::min(tby + TILE_ROWS, dst.rows);

                PtrStepi col_dist_sums;
                col_dist_sums.data = buffer.ptr(dst.cols + blockIdx.x * block_window) + blockIdx.y * search_window * search_window;
                col_dist_sums.step = buffer.step;

                PtrStepi up_col_dist_sums;
                up_col_dist_sums.data = buffer.data + blockIdx.y * search_window * search_window;
                up_col_dist_sums.step = buffer.step;

                extern __shared__ int dist_sums[]; //search_window * search_window

                int first_col = -1;

                for (int i = tby; i < tey; ++i)
                    for (int j = tbx; j < tex; ++j)
                    {           
                        __syncthreads();

                        if (j == tbx)
                        {
                            initSums_TileFistColumn(i, j, dist_sums, col_dist_sums, up_col_dist_sums);
                            first_col = 0;
                        }
                        else
                        {                            
                            if (i == tby)
                              shiftLeftSums_TileFirstRow(i, j, first_col, dist_sums, col_dist_sums, up_col_dist_sums);
                            else
                              shiftLeftSums_UsingUpSums(i, j, first_col, dist_sums, col_dist_sums, up_col_dist_sums);

                            first_col = (first_col + 1) % block_window;
                        }

                        __syncthreads();
                        
                        convolve_search_window(i, j, dist_sums, col_dist_sums, up_col_dist_sums, dst(i, j));
                    }
            }

        };

        template<typename T>
        __global__ void fast_nlm_kernel(const FastNonLocalMenas<T> fnlm, PtrStepSz<T> dst) { fnlm(dst); }

        void nln_fast_get_buffer_size(const PtrStepSzb& src, int search_window, int block_window, int& buffer_cols, int& buffer_rows)
        {            
            typedef FastNonLocalMenas<uchar> FNLM;
            dim3 grid(divUp(src.cols, FNLM::TILE_COLS), divUp(src.rows, FNLM::TILE_ROWS));

            buffer_cols = search_window * search_window * grid.y;
            buffer_rows = src.cols + block_window * grid.x;
        }

        template<typename T>
        void nlm_fast_gpu(const PtrStepSzb& src, PtrStepSzb dst, PtrStepi buffer,
                          int search_window, int block_window, float h, cudaStream_t stream)
        {
            typedef FastNonLocalMenas<T> FNLM;
            FNLM fnlm(search_window, block_window, h);

            fnlm.src = (PtrStepSz<T>)src;
            fnlm.buffer = buffer;            

            dim3 block(FNLM::CTA_SIZE, 1);
            dim3 grid(divUp(src.cols, FNLM::TILE_COLS), divUp(src.rows, FNLM::TILE_ROWS));
            int smem = search_window * search_window * sizeof(int);

           
            fast_nlm_kernel<<<grid, block, smem>>>(fnlm, (PtrStepSz<T>)dst);
            cudaSafeCall ( cudaGetLastError () );
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void nlm_fast_gpu<uchar>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float,  cudaStream_t);
        template void nlm_fast_gpu<uchar2>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);
        template void nlm_fast_gpu<uchar3>(const PtrStepSzb&, PtrStepSzb, PtrStepi, int, int, float, cudaStream_t);
    }
}}}


#endif /* CUDA_DISABLER */