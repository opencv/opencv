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

#include "opencv2/gpu/devmem2d.hpp"
#include "saturate_cast.hpp"
#include "safe_call.hpp"
#include "cuda_shared.hpp"
#include "vecmath.hpp"

using namespace cv::gpu;

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+30F
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// Linear filters

#define MAX_KERNEL_SIZE 16

namespace filter_krnls
{
    __constant__ float cLinearKernel[MAX_KERNEL_SIZE];
}

namespace cv { namespace gpu { namespace filters
{
    void loadLinearKernel(const float kernel[], int ksize)
    {
        cudaSafeCall( cudaMemcpyToSymbol(filter_krnls::cLinearKernel, kernel, ksize * sizeof(float)) );
    }
}}}

namespace filter_krnls
{
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int KERNEL_SIZE, int CN, typename T, typename D>
    __global__ void linearRowFilter(const T* src, size_t src_step, D* dst, size_t dst_step, int anchor, int width, int height)
    {
        __shared__ T smem[BLOCK_DIM_Y * BLOCK_DIM_X * 3];
        
        const int blockStartX = blockDim.x * blockIdx.x;
        const int blockStartY = blockDim.y * blockIdx.y;

		const int threadX = blockStartX + threadIdx.x;
        const int prevThreadX = threadX - blockDim.x;
        const int nextThreadX = threadX + blockDim.x;

		const int threadY = blockStartY + threadIdx.y;

        T* sDataRow = smem + threadIdx.y * blockDim.x * 3;

        if (threadY < height)
        {
            const T* rowSrc = src + threadY * src_step;

            sDataRow[threadIdx.x + blockDim.x] = threadX < width ? rowSrc[threadX] : VecTraits<T>::all(0);

            sDataRow[threadIdx.x] = prevThreadX >= 0 ? rowSrc[prevThreadX] : VecTraits<T>::all(0);

            sDataRow[(blockDim.x << 1) + threadIdx.x] = nextThreadX < width ? rowSrc[nextThreadX] : VecTraits<T>::all(0);

            __syncthreads();

            if (threadX < width)
            {
                typedef typename TypeVec<float, CN>::vec_t sum_t;
                sum_t sum = VecTraits<sum_t>::all(0);

                sDataRow += threadIdx.x + blockDim.x - anchor;

                #pragma unroll
                for(int i = 0; i < KERNEL_SIZE; ++i)
                    sum = sum + sDataRow[i] * cLinearKernel[i];

                dst[threadY * dst_step + threadX] = saturate_cast<D>(sum);
            }
        }
    }
}

namespace cv { namespace gpu { namespace filters
{
    template <int KERNEL_SIZE, int CN, typename T, typename D>
    void linearRowFilter_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor)
    {
        const int BLOCK_DIM_X = 16;
        const int BLOCK_DIM_Y = 16;

        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 blocks(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y));

        filter_krnls::linearRowFilter<BLOCK_DIM_X, BLOCK_DIM_Y, KERNEL_SIZE, CN><<<blocks, threads>>>(src.data, src.step/src.elemSize(), 
            dst.data, dst.step/dst.elemSize(), anchor, src.cols, src.rows);

        cudaSafeCall( cudaThreadSynchronize() );
    }

    template <int CN, typename T, typename D>
    inline void linearRowFilter_gpu(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor);
        static const caller_t callers[] = 
        {linearRowFilter_caller<0 , CN, T, D>, linearRowFilter_caller<1 , CN, T, D>, 
         linearRowFilter_caller<2 , CN, T, D>, linearRowFilter_caller<3 , CN, T, D>, 
         linearRowFilter_caller<4 , CN, T, D>, linearRowFilter_caller<5 , CN, T, D>, 
         linearRowFilter_caller<6 , CN, T, D>, linearRowFilter_caller<7 , CN, T, D>, 
         linearRowFilter_caller<8 , CN, T, D>, linearRowFilter_caller<9 , CN, T, D>, 
         linearRowFilter_caller<10, CN, T, D>, linearRowFilter_caller<11, CN, T, D>, 
         linearRowFilter_caller<12, CN, T, D>, linearRowFilter_caller<13, CN, T, D>, 
         linearRowFilter_caller<14, CN, T, D>, linearRowFilter_caller<15, CN, T, D>};

        loadLinearKernel(kernel, ksize);
        callers[ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor);
    }

    template void linearRowFilter_gpu<4, uchar4, uchar4>(const DevMem2D&, const DevMem2D&, const float[], int , int);

  /*  void linearRowFilter_gpu_8u_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<4, uchar4, uchar4>(src, dst, kernel, ksize, anchor);
    }*/
    void linearRowFilter_gpu_8u_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<4, uchar4, char4>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_8s_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<4, char4, uchar4>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_8s_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<4, char4, char4>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_16u_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<2, ushort2, ushort2>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_16u_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<2, ushort2, short2>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_16s_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<2, short2, ushort2>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_16s_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<2, short2, short2>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_32s_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<1, int, int>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_32s_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<1, int, float>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_32f_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<1, float, int>(src, dst, kernel, ksize, anchor);
    }
    void linearRowFilter_gpu_32f_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearRowFilter_gpu<1 ,float, float>(src, dst, kernel, ksize, anchor);
    }
}}}

namespace filter_krnls
{
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int KERNEL_SIZE, int CN, typename T, typename D>
    __global__ void linearColumnFilter(const T* src, size_t src_step, D* dst, size_t dst_step, int anchor, int width, int height)
    {
        __shared__ T smem[BLOCK_DIM_Y * BLOCK_DIM_X * 3];
        
        const int blockStartX = blockDim.x * blockIdx.x;
        const int blockStartY = blockDim.y * blockIdx.y;

		const int threadX = blockStartX + threadIdx.x;

		const int threadY = blockStartY + threadIdx.y;
        const int prevThreadY = threadY - blockDim.y;
        const int nextThreadY = threadY + blockDim.y;

        const int smem_step = blockDim.x;

        T* sDataColumn = smem + threadIdx.x;

        if (threadX < width)
        {
            const T* colSrc = src + threadX;

            sDataColumn[(threadIdx.y + blockDim.y) * smem_step] = threadY < height ? colSrc[threadY * src_step] : VecTraits<T>::all(0);

            sDataColumn[threadIdx.y * smem_step] = prevThreadY >= 0 ? colSrc[prevThreadY * src_step] : VecTraits<T>::all(0);

            sDataColumn[(threadIdx.y + (blockDim.y << 1)) * smem_step] = nextThreadY < height ? colSrc[nextThreadY * src_step] : VecTraits<T>::all(0);

            __syncthreads();

            if (threadY < height)
            {
                typedef typename TypeVec<float, CN>::vec_t sum_t;
                sum_t sum = VecTraits<sum_t>::all(0);

                sDataColumn += (threadIdx.y + blockDim.y - anchor)* smem_step;

                #pragma unroll
                for(int i = 0; i < KERNEL_SIZE; ++i)
                    sum = sum + sDataColumn[i * smem_step] * cLinearKernel[i];

                dst[threadY * dst_step + threadX] = saturate_cast<D>(sum);
            }
        }
    }
}

namespace cv { namespace gpu { namespace filters
{
    template <int KERNEL_SIZE, int CN, typename T, typename D>
    void linearColumnFilter_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor)
    {
        const int BLOCK_DIM_X = 16;
        const int BLOCK_DIM_Y = 16;

        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 blocks(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y));

        filter_krnls::linearColumnFilter<BLOCK_DIM_X, BLOCK_DIM_Y, KERNEL_SIZE, CN><<<blocks, threads>>>(src.data, src.step/src.elemSize(), 
            dst.data, dst.step/dst.elemSize(), anchor, src.cols, src.rows);

        cudaSafeCall( cudaThreadSynchronize() );
    }

    template <int CN, typename T, typename D>
    inline void linearColumnFilter_gpu(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor);
        static const caller_t callers[] = 
        {linearColumnFilter_caller<0 , CN, T, D>, linearColumnFilter_caller<1 , CN, T, D>, 
         linearColumnFilter_caller<2 , CN, T, D>, linearColumnFilter_caller<3 , CN, T, D>, 
         linearColumnFilter_caller<4 , CN, T, D>, linearColumnFilter_caller<5 , CN, T, D>, 
         linearColumnFilter_caller<6 , CN, T, D>, linearColumnFilter_caller<7 , CN, T, D>, 
         linearColumnFilter_caller<8 , CN, T, D>, linearColumnFilter_caller<9 , CN, T, D>, 
         linearColumnFilter_caller<10, CN, T, D>, linearColumnFilter_caller<11, CN, T, D>, 
         linearColumnFilter_caller<12, CN, T, D>, linearColumnFilter_caller<13, CN, T, D>, 
         linearColumnFilter_caller<14, CN, T, D>, linearColumnFilter_caller<15, CN, T, D>};

        loadLinearKernel(kernel, ksize);
        callers[ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor);
    }

    void linearColumnFilter_gpu_8u_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<4, uchar4, uchar4>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_8u_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<4, uchar4, char4>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_8s_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<4, char4, uchar4>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_8s_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<4, char4, char4>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_16u_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<2, ushort2, ushort2>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_16u_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<2, ushort2, short2>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_16s_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<2, short2, ushort2>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_16s_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<2, short2, short2>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_32s_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<1, int, int>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_32s_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<1, int, float>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_32f_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<1, float, int>(src, dst, kernel, ksize, anchor);
    }
    void linearColumnFilter_gpu_32f_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor)
    {
        linearColumnFilter_gpu<1, float, float>(src, dst, kernel, ksize, anchor);
    }
}}}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral filters

namespace bf_krnls
{
    __constant__ float* ctable_color;
    __constant__ float* ctable_space;
    __constant__ size_t ctable_space_step;

    __constant__ int cndisp;
    __constant__ int cradius;

    __constant__ short cedge_disc;
    __constant__ short cmax_disc;
}

namespace cv { namespace gpu { namespace bf 
{
    void load_constants(float* table_color, const DevMem2Df& table_space, int ndisp, int radius, short edge_disc, short max_disc)
    {
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::ctable_color, &table_color, sizeof(table_color)) );
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::ctable_space, &table_space.data, sizeof(table_space.data)) );
        size_t table_space_step = table_space.step / sizeof(float);
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::ctable_space_step, &table_space_step, sizeof(size_t)) );
        
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::cndisp, &ndisp, sizeof(int)) );
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::cradius, &radius, sizeof(int)) );
        
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::cedge_disc, &edge_disc, sizeof(short)) );
        cudaSafeCall( cudaMemcpyToSymbol(bf_krnls::cmax_disc, &max_disc, sizeof(short)) );
    }
}}}

namespace bf_krnls
{
    template <int channels>
    struct DistRgbMax
    {
        static __device__ uchar calc(const uchar* a, const uchar* b)
        {
            uchar x = abs(a[0] - b[0]);
            uchar y = abs(a[1] - b[1]);
            uchar z = abs(a[2] - b[2]);
            return (max(max(x, y), z));
        }
    };

    template <>
    struct DistRgbMax<1>
    {
        static __device__ uchar calc(const uchar* a, const uchar* b)
        {
            return abs(a[0] - b[0]);
        }
    };

    template <int channels, typename T>
    __global__ void bilateral_filter(int t, T* disp, size_t disp_step, const uchar* img, size_t img_step, int h, int w)
    {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int x = ((blockIdx.x * blockDim.x + threadIdx.x) << 1) + ((y + t) & 1);

        T dp[5];

        if (y > 0 && y < h - 1 && x > 0 && x < w - 1)
        {
            dp[0] = *(disp + (y  ) * disp_step + x + 0);
            dp[1] = *(disp + (y-1) * disp_step + x + 0);
            dp[2] = *(disp + (y  ) * disp_step + x - 1);
            dp[3] = *(disp + (y+1) * disp_step + x + 0);
            dp[4] = *(disp + (y  ) * disp_step + x + 1);

            if(abs(dp[1] - dp[0]) >= cedge_disc || abs(dp[2] - dp[0]) >= cedge_disc || abs(dp[3] - dp[0]) >= cedge_disc || abs(dp[4] - dp[0]) >= cedge_disc)            
            {
                const int ymin = max(0, y - cradius);
                const int xmin = max(0, x - cradius);
                const int ymax = min(h - 1, y + cradius);
                const int xmax = min(w - 1, x + cradius);

                float cost[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                const uchar* ic = img + y * img_step + channels * x;

                for(int yi = ymin; yi <= ymax; yi++)
                {
                    const T* disp_y = disp + yi * disp_step;

                    for(int xi = xmin; xi <= xmax; xi++)
                    {
                        const uchar* in = img + yi * img_step + channels * xi;

                        uchar dist_rgb = DistRgbMax<channels>::calc(in, ic);

                        const float weight = ctable_color[dist_rgb] * (ctable_space + abs(y-yi)* ctable_space_step)[abs(x-xi)];

                        const T disp_reg = disp_y[xi];

                        cost[0] += min(cmax_disc, abs(disp_reg - dp[0])) * weight;
                        cost[1] += min(cmax_disc, abs(disp_reg - dp[1])) * weight;
                        cost[2] += min(cmax_disc, abs(disp_reg - dp[2])) * weight;
                        cost[3] += min(cmax_disc, abs(disp_reg - dp[3])) * weight;
                        cost[4] += min(cmax_disc, abs(disp_reg - dp[4])) * weight;
                    }
                }

                float minimum = FLT_MAX;
                int id = 0;

                if (cost[0] < minimum)
                {
                    minimum = cost[0];
                    id = 0;
                }
                if (cost[1] < minimum)
                {
                    minimum = cost[1];
                    id = 1;
                }
                if (cost[2] < minimum)
                {
                    minimum = cost[2];
                    id = 2;
                }
                if (cost[3] < minimum)
                {
                    minimum = cost[3];
                    id = 3;
                }
                if (cost[4] < minimum)
                {
                    minimum = cost[4];
                    id = 4;
                }

                *(disp + y * disp_step + x) = dp[id];
            }
        }
    }
}

namespace cv { namespace gpu { namespace bf 
{
    template <typename T>     
    void bilateral_filter_caller(const DevMem2D_<T>& disp, const DevMem2D& img, int channels, int iters, cudaStream_t stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(disp.cols, threads.x << 1);
        grid.y = divUp(disp.rows, threads.y);

        switch (channels)
        {
        case 1:
            for (int i = 0; i < iters; ++i)
            {
                bf_krnls::bilateral_filter<1><<<grid, threads, 0, stream>>>(0, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
                bf_krnls::bilateral_filter<1><<<grid, threads, 0, stream>>>(1, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
            }
            break;
        case 3:
            for (int i = 0; i < iters; ++i)
            {
                bf_krnls::bilateral_filter<3><<<grid, threads, 0, stream>>>(0, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
                bf_krnls::bilateral_filter<3><<<grid, threads, 0, stream>>>(1, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
            }
            break;
        default:
            cv::gpu::error("Unsupported channels count", __FILE__, __LINE__);
        }        

        if (stream != 0)
            cudaSafeCall( cudaThreadSynchronize() );
    }

    void bilateral_filter_gpu(const DevMem2D& disp, const DevMem2D& img, int channels, int iters, cudaStream_t stream)
    {
        bilateral_filter_caller(disp, img, channels, iters, stream);
    }

    void bilateral_filter_gpu(const DevMem2D_<short>& disp, const DevMem2D& img, int channels, int iters, cudaStream_t stream)
    {
        bilateral_filter_caller(disp, img, channels, iters, stream);
    }
}}}
