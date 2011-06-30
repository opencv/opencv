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
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/vecmath.hpp"
#include "opencv2/gpu/device/limits_gpu.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

#include "safe_call.hpp"
#include "internal_shared.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Linear filters

#define MAX_KERNEL_SIZE 16
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

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
    template <typename T, size_t size> struct SmemType_
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_t smem_t;
    };
    template <typename T> struct SmemType_<T, 4>
    {
        typedef T smem_t;
    };
    template <typename T> struct SmemType
    {
        typedef typename SmemType_<T, sizeof(T)>::smem_t smem_t;
    };

    template <int ksize, typename T, typename D, typename B>
    __global__ void linearRowFilter(const DevMem2D_<T> src, PtrStep_<D> dst, int anchor, const B b)
    {
        typedef typename SmemType<T>::smem_t smem_t;

        __shared__ smem_t smem[BLOCK_DIM_Y * BLOCK_DIM_X * 3];

        const int x = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
        const int y = BLOCK_DIM_Y * blockIdx.y + threadIdx.y;

        smem_t* sDataRow = smem + threadIdx.y * BLOCK_DIM_X * 3;

        if (y < src.rows)
        {
            const T* rowSrc = src.ptr(y);

            sDataRow[threadIdx.x                  ] = b.at_low(x - BLOCK_DIM_X, rowSrc);
            sDataRow[threadIdx.x + BLOCK_DIM_X    ] = b.at_high(x, rowSrc);
            sDataRow[threadIdx.x + BLOCK_DIM_X * 2] = b.at_high(x + BLOCK_DIM_X, rowSrc);

            __syncthreads();

            if (x < src.cols)
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_t sum_t;
                sum_t sum = VecTraits<sum_t>::all(0);

                sDataRow += threadIdx.x + BLOCK_DIM_X - anchor;

                #pragma unroll
                for(int i = 0; i < ksize; ++i)
                    sum = sum + sDataRow[i] * cLinearKernel[i];

                dst.ptr(y)[x] = saturate_cast<D>(sum);
            }
        }
    }
}

namespace cv { namespace gpu { namespace filters
{
    template <int ksize, typename T, typename D, template<typename> class B>
    void linearRowFilter_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream)
    {
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 grid(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y));

        typedef typename filter_krnls::SmemType<T>::smem_t smem_t;
        B<smem_t> b(src.cols);

        if (!b.is_range_safe(-BLOCK_DIM_X, (grid.x + 1) * BLOCK_DIM_X - 1))
        {
            cv::gpu::error("linearRowFilter: can't use specified border extrapolation, image is too small, "
                           "try bigger image or another border extrapolation mode", __FILE__, __LINE__);
        }

        filter_krnls::linearRowFilter<ksize, T, D><<<grid, threads, 0, stream>>>(src, dst, anchor, b);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename D>
    void linearRowFilter_gpu(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream);
        static const caller_t callers[3][17] = 
        {
            {
                0, 
                linearRowFilter_caller<1 , T, D, BrdRowReflect101>, 
                linearRowFilter_caller<2 , T, D, BrdRowReflect101>,
                linearRowFilter_caller<3 , T, D, BrdRowReflect101>, 
                linearRowFilter_caller<4 , T, D, BrdRowReflect101>, 
                linearRowFilter_caller<5 , T, D, BrdRowReflect101>, 
                linearRowFilter_caller<6 , T, D, BrdRowReflect101>, 
                linearRowFilter_caller<7 , T, D, BrdRowReflect101>,
                linearRowFilter_caller<8 , T, D, BrdRowReflect101>,
                linearRowFilter_caller<9 , T, D, BrdRowReflect101>, 
                linearRowFilter_caller<10, T, D, BrdRowReflect101>, 
                linearRowFilter_caller<11, T, D, BrdRowReflect101>, 
                linearRowFilter_caller<12, T, D, BrdRowReflect101>, 
                linearRowFilter_caller<13, T, D, BrdRowReflect101>, 
                linearRowFilter_caller<14, T, D, BrdRowReflect101>,
                linearRowFilter_caller<15, T, D, BrdRowReflect101>, 
                linearRowFilter_caller<16, T, D, BrdRowReflect101>,
            },
            {
                0, 
                linearRowFilter_caller<1 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<2 , T, D, BrdRowReplicate>,
                linearRowFilter_caller<3 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<4 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<5 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<6 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<7 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<8 , T, D, BrdRowReplicate>,
                linearRowFilter_caller<9 , T, D, BrdRowReplicate>, 
                linearRowFilter_caller<10, T, D, BrdRowReplicate>, 
                linearRowFilter_caller<11, T, D, BrdRowReplicate>, 
                linearRowFilter_caller<12, T, D, BrdRowReplicate>, 
                linearRowFilter_caller<13, T, D, BrdRowReplicate>, 
                linearRowFilter_caller<14, T, D, BrdRowReplicate>,
                linearRowFilter_caller<15, T, D, BrdRowReplicate>, 
                linearRowFilter_caller<16, T, D, BrdRowReplicate>,
            },
            {
                0, 
                linearRowFilter_caller<1 , T, D, BrdRowConstant>, 
                linearRowFilter_caller<2 , T, D, BrdRowConstant>,
                linearRowFilter_caller<3 , T, D, BrdRowConstant>, 
                linearRowFilter_caller<4 , T, D, BrdRowConstant>, 
                linearRowFilter_caller<5 , T, D, BrdRowConstant>, 
                linearRowFilter_caller<6 , T, D, BrdRowConstant>, 
                linearRowFilter_caller<7 , T, D, BrdRowConstant>, 
                linearRowFilter_caller<8 , T, D, BrdRowConstant>,
                linearRowFilter_caller<9 , T, D, BrdRowConstant>,
                linearRowFilter_caller<10, T, D, BrdRowConstant>, 
                linearRowFilter_caller<11, T, D, BrdRowConstant>, 
                linearRowFilter_caller<12, T, D, BrdRowConstant>, 
                linearRowFilter_caller<13, T, D, BrdRowConstant>,
                linearRowFilter_caller<14, T, D, BrdRowConstant>,
                linearRowFilter_caller<15, T, D, BrdRowConstant>, 
                linearRowFilter_caller<16, T, D, BrdRowConstant>,
            }
        };
        
        loadLinearKernel(kernel, ksize);

        callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
    }

    template void linearRowFilter_gpu<uchar , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<uchar4, float4>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<short , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<short2, float2>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<short3, float3>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<int   , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<float , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
}}}

namespace filter_krnls
{
    template <int ksize, typename T, typename D, typename B>
    __global__ void linearColumnFilter(const DevMem2D_<T> src, PtrStep_<D> dst, int anchor, const B b)
    {
        __shared__ T smem[BLOCK_DIM_Y * BLOCK_DIM_X * 3];

        const int x = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
        const int y = BLOCK_DIM_Y * blockIdx.y + threadIdx.y;

        T* sDataColumn = smem + threadIdx.x;

        if (x < src.cols)
        {
            const T* srcCol = src.ptr() + x;

            sDataColumn[ threadIdx.y                    * BLOCK_DIM_X] = b.at_low(y - BLOCK_DIM_Y, srcCol);
            sDataColumn[(threadIdx.y + BLOCK_DIM_Y)     * BLOCK_DIM_X] = b.at_high(y, srcCol);
            sDataColumn[(threadIdx.y + BLOCK_DIM_Y * 2) * BLOCK_DIM_X] = b.at_high(y + BLOCK_DIM_Y, srcCol);

            __syncthreads();

            if (y < src.rows)
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_t sum_t;
                sum_t sum = VecTraits<sum_t>::all(0);

                sDataColumn += (threadIdx.y + BLOCK_DIM_Y - anchor) * BLOCK_DIM_X;

                #pragma unroll
                for(int i = 0; i < ksize; ++i)
                    sum = sum + sDataColumn[i * BLOCK_DIM_X] * cLinearKernel[i];

                dst.ptr(y)[x] = saturate_cast<D>(sum);
            }
        }
    }
}

namespace cv { namespace gpu { namespace filters
{
    template <int ksize, typename T, typename D, template<typename> class B>
    void linearColumnFilter_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream)
    {
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 grid(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y));

        B<T> b(src.rows, src.step);

        if (!b.is_range_safe(-BLOCK_DIM_Y, (grid.y + 1) * BLOCK_DIM_Y - 1))
        {
            cv::gpu::error("linearColumnFilter: can't use specified border extrapolation, image is too small, "
                           "try bigger image or another border extrapolation mode", __FILE__, __LINE__);
        }

        filter_krnls::linearColumnFilter<ksize, T, D><<<grid, threads, 0, stream>>>(src, dst, anchor, b);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename D>
    void linearColumnFilter_gpu(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream);
        static const caller_t callers[3][17] = 
        {
            {
                0, 
                linearColumnFilter_caller<1 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<2 , T, D, BrdColReflect101>,
                linearColumnFilter_caller<3 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<4 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<5 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<6 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<7 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<8 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<9 , T, D, BrdColReflect101>, 
                linearColumnFilter_caller<10, T, D, BrdColReflect101>, 
                linearColumnFilter_caller<11, T, D, BrdColReflect101>, 
                linearColumnFilter_caller<12, T, D, BrdColReflect101>, 
                linearColumnFilter_caller<13, T, D, BrdColReflect101>, 
                linearColumnFilter_caller<14, T, D, BrdColReflect101>, 
                linearColumnFilter_caller<15, T, D, BrdColReflect101>, 
                linearColumnFilter_caller<16, T, D, BrdColReflect101>, 
            },
            {
                0, 
                linearColumnFilter_caller<1 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<2 , T, D, BrdColReplicate>,
                linearColumnFilter_caller<3 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<4 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<5 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<6 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<7 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<8 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<9 , T, D, BrdColReplicate>, 
                linearColumnFilter_caller<10, T, D, BrdColReplicate>, 
                linearColumnFilter_caller<11, T, D, BrdColReplicate>, 
                linearColumnFilter_caller<12, T, D, BrdColReplicate>, 
                linearColumnFilter_caller<13, T, D, BrdColReplicate>, 
                linearColumnFilter_caller<14, T, D, BrdColReplicate>, 
                linearColumnFilter_caller<15, T, D, BrdColReplicate>, 
                linearColumnFilter_caller<16, T, D, BrdColReplicate>, 
            },
            {
                0, 
                linearColumnFilter_caller<1 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<2 , T, D, BrdColConstant>,
                linearColumnFilter_caller<3 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<4 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<5 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<6 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<7 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<8 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<9 , T, D, BrdColConstant>, 
                linearColumnFilter_caller<10, T, D, BrdColConstant>, 
                linearColumnFilter_caller<11, T, D, BrdColConstant>, 
                linearColumnFilter_caller<12, T, D, BrdColConstant>, 
                linearColumnFilter_caller<13, T, D, BrdColConstant>, 
                linearColumnFilter_caller<14, T, D, BrdColConstant>, 
                linearColumnFilter_caller<15, T, D, BrdColConstant>, 
                linearColumnFilter_caller<16, T, D, BrdColConstant>, 
            }
        };
        
        loadLinearKernel(kernel, ksize);

        callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
    }

    template void linearColumnFilter_gpu<float , uchar >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float4, uchar4>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , short >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float2, short2>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float3, short3>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , int   >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
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
        static __device__ __forceinline__ uchar calc(const uchar* a, const uchar* b)
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
        static __device__ __forceinline__ uchar calc(const uchar* a, const uchar* b)
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

                float minimum = numeric_limits_gpu<float>::max();
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
                cudaSafeCall( cudaGetLastError() );
                bf_krnls::bilateral_filter<1><<<grid, threads, 0, stream>>>(1, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
                cudaSafeCall( cudaGetLastError() );
            }
            break;
        case 3:
            for (int i = 0; i < iters; ++i)
            {
                bf_krnls::bilateral_filter<3><<<grid, threads, 0, stream>>>(0, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
                cudaSafeCall( cudaGetLastError() );
                bf_krnls::bilateral_filter<3><<<grid, threads, 0, stream>>>(1, disp.data, disp.step/sizeof(T), img.data, img.step, disp.rows, disp.cols);
                cudaSafeCall( cudaGetLastError() );
            }
            break;
        default:
            cv::gpu::error("Unsupported channels count", __FILE__, __LINE__);
        }

        if (stream != 0)
            cudaSafeCall( cudaDeviceSynchronize() );
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
