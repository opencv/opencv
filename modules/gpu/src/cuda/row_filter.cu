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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

#define MAX_KERNEL_SIZE 16
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 4
#define RESULT_STEPS 8
#define HALO_STEPS 1

namespace filter_row
{
    __constant__ float c_kernel[MAX_KERNEL_SIZE];

    void loadKernel(const float kernel[], int ksize)
    {
        cudaSafeCall( cudaMemcpyToSymbol(c_kernel, kernel, ksize * sizeof(float)) );
    }

    namespace detail
    {
        template <typename T, size_t size> struct SmemType
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type smem_t;
        };

        template <typename T> struct SmemType<T, 4>
        {
            typedef T smem_t;
        };
    }

    template <typename T> struct SmemType
    {
        typedef typename detail::SmemType<T, sizeof(T)>::smem_t smem_t;
    };

    template <int KERNEL_SIZE, typename T, typename D, typename B>
    __global__ void linearRowFilter(const DevMem2D_<T> src, PtrStep<D> dst, int anchor, const B b)
    {
        typedef typename SmemType<T>::smem_t smem_t;
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

        __shared__ smem_t smem[BLOCK_DIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCK_DIM_X];

        //Offset to the left halo edge
        const int x = (blockIdx.x * RESULT_STEPS - HALO_STEPS) * BLOCK_DIM_X + threadIdx.x;
        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (y < src.rows)
        {
            const T* src_row = src.ptr(y);

            //Load main data
            #pragma unroll
            for(int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; ++i)
                smem[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = b.at_high(i * BLOCK_DIM_X + x, src_row);

            //Load left halo
            #pragma unroll
            for(int i = 0; i < HALO_STEPS; ++i)
                smem[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = b.at_low(i * BLOCK_DIM_X + x, src_row);

            //Load right halo
            #pragma unroll
            for(int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; ++i)
                smem[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = b.at_high(i * BLOCK_DIM_X + x, src_row);

            __syncthreads();

            D* dst_row = dst.ptr(y);

            #pragma unroll
            for(int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; ++i)
            {
                sum_t sum = VecTraits<sum_t>::all(0);

                #pragma unroll
                for (int j = 0; j < KERNEL_SIZE; ++j)
                    sum = sum + smem[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X + j - anchor] * c_kernel[j];

                int dstX = x + i * BLOCK_DIM_X;

                if (dstX < src.cols)
                    dst_row[dstX] = saturate_cast<D>(sum);
            }
        }
    }
}

namespace cv { namespace gpu { namespace filters
{
    template <int ksize, typename T, typename D, template<typename> class B>
    void linearRowFilter_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream)
    {
        typedef typename filter_row::SmemType<T>::smem_t smem_t;

        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid(divUp(src.cols, RESULT_STEPS * BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y));

        B<smem_t> b(src.cols);

        filter_row::linearRowFilter<ksize, T, D><<<grid, block, 0, stream>>>(src, dst, anchor, b);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename D>
    void linearRowFilter_gpu(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream);
        static const caller_t callers[5][17] = 
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
                linearRowFilter_caller<16, T, D, BrdRowReflect101>
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
                linearRowFilter_caller<16, T, D, BrdRowReplicate>
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
                linearRowFilter_caller<16, T, D, BrdRowConstant>
            },
            {
                0, 
                linearRowFilter_caller<1 , T, D, BrdRowReflect>, 
                linearRowFilter_caller<2 , T, D, BrdRowReflect>,
                linearRowFilter_caller<3 , T, D, BrdRowReflect>, 
                linearRowFilter_caller<4 , T, D, BrdRowReflect>, 
                linearRowFilter_caller<5 , T, D, BrdRowReflect>, 
                linearRowFilter_caller<6 , T, D, BrdRowReflect>, 
                linearRowFilter_caller<7 , T, D, BrdRowReflect>, 
                linearRowFilter_caller<8 , T, D, BrdRowReflect>,
                linearRowFilter_caller<9 , T, D, BrdRowReflect>,
                linearRowFilter_caller<10, T, D, BrdRowReflect>, 
                linearRowFilter_caller<11, T, D, BrdRowReflect>, 
                linearRowFilter_caller<12, T, D, BrdRowReflect>, 
                linearRowFilter_caller<13, T, D, BrdRowReflect>,
                linearRowFilter_caller<14, T, D, BrdRowReflect>,
                linearRowFilter_caller<15, T, D, BrdRowReflect>, 
                linearRowFilter_caller<16, T, D, BrdRowReflect>
            },
            {
                0, 
                linearRowFilter_caller<1 , T, D, BrdRowWrap>, 
                linearRowFilter_caller<2 , T, D, BrdRowWrap>,
                linearRowFilter_caller<3 , T, D, BrdRowWrap>, 
                linearRowFilter_caller<4 , T, D, BrdRowWrap>, 
                linearRowFilter_caller<5 , T, D, BrdRowWrap>, 
                linearRowFilter_caller<6 , T, D, BrdRowWrap>, 
                linearRowFilter_caller<7 , T, D, BrdRowWrap>, 
                linearRowFilter_caller<8 , T, D, BrdRowWrap>,
                linearRowFilter_caller<9 , T, D, BrdRowWrap>,
                linearRowFilter_caller<10, T, D, BrdRowWrap>, 
                linearRowFilter_caller<11, T, D, BrdRowWrap>, 
                linearRowFilter_caller<12, T, D, BrdRowWrap>, 
                linearRowFilter_caller<13, T, D, BrdRowWrap>,
                linearRowFilter_caller<14, T, D, BrdRowWrap>,
                linearRowFilter_caller<15, T, D, BrdRowWrap>, 
                linearRowFilter_caller<16, T, D, BrdRowWrap>
            }
        };
        
        filter_row::loadKernel(kernel, ksize);

        callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
    }

    template void linearRowFilter_gpu<uchar , float >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<uchar4, float4>(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearRowFilter_gpu<short , float >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearRowFilter_gpu<short2, float2>(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<short3, float3>(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<int   , float >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<float , float >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
}}}
