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

namespace filter_column
{
    __constant__ float c_kernel[MAX_KERNEL_SIZE];

    void loadKernel(const float kernel[], int ksize)
    {
        cudaSafeCall( cudaMemcpyToSymbol(c_kernel, kernel, ksize * sizeof(float)) );
    }

    template <int KERNEL_SIZE, typename T, typename D, typename B>
    __global__ void linearColumnFilter(const DevMem2D_<T> src, PtrStep<D> dst, int anchor, const B b)
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

        __shared__ T smem[BLOCK_DIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCK_DIM_Y + 1];

        //Offset to the upper halo edge
        const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        const int y = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCK_DIM_Y + threadIdx.y;

        if (x < src.cols)
        {
            const T* src_col = src.ptr() + x;

            //Main data
            #pragma unroll
            for(int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; ++i)
                smem[threadIdx.x][threadIdx.y + i * BLOCK_DIM_Y] = b.at_high(y + i * BLOCK_DIM_Y, src_col, src.step);

            //Upper halo
            #pragma unroll
            for(int i = 0; i < HALO_STEPS; ++i)
                smem[threadIdx.x][threadIdx.y + i * BLOCK_DIM_Y] = b.at_low(y + i * BLOCK_DIM_Y, src_col, src.step);

            //Lower halo
            #pragma unroll
            for(int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; ++i)
                smem[threadIdx.x][threadIdx.y + i * BLOCK_DIM_Y]=  b.at_high(y + i * BLOCK_DIM_Y, src_col, src.step);

            __syncthreads();

            #pragma unroll
            for(int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; ++i)
            {
                sum_t sum = VecTraits<sum_t>::all(0);

                #pragma unroll
                for(int j = 0; j < KERNEL_SIZE; ++j)
                    sum = sum + smem[threadIdx.x][threadIdx.y + i * BLOCK_DIM_Y + j - anchor] * c_kernel[j];

                int dstY = y + i * BLOCK_DIM_Y;

                if (dstY < src.rows)
                    dst.ptr(dstY)[x] = saturate_cast<D>(sum);
            }
        }
    }
}

namespace cv { namespace gpu { namespace filters
{
    template <int ksize, typename T, typename D, template<typename> class B>
    void linearColumnFilter_caller(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream)
    {        
        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, RESULT_STEPS * BLOCK_DIM_Y));

        B<T> b(src.rows);

        filter_column::linearColumnFilter<ksize, T, D><<<grid, block, 0, stream>>>(src, dst, anchor, b);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename D>
    void linearColumnFilter_gpu(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
    {
        typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2D_<D>& dst, int anchor, cudaStream_t stream);
        static const caller_t callers[5][17] = 
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
                linearColumnFilter_caller<16, T, D, BrdColReflect101> 
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
                linearColumnFilter_caller<16, T, D, BrdColReplicate>
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
                linearColumnFilter_caller<16, T, D, BrdColConstant> 
            },
            {
                0, 
                linearColumnFilter_caller<1 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<2 , T, D, BrdColReflect>,
                linearColumnFilter_caller<3 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<4 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<5 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<6 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<7 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<8 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<9 , T, D, BrdColReflect>, 
                linearColumnFilter_caller<10, T, D, BrdColReflect>, 
                linearColumnFilter_caller<11, T, D, BrdColReflect>, 
                linearColumnFilter_caller<12, T, D, BrdColReflect>, 
                linearColumnFilter_caller<13, T, D, BrdColReflect>, 
                linearColumnFilter_caller<14, T, D, BrdColReflect>, 
                linearColumnFilter_caller<15, T, D, BrdColReflect>, 
                linearColumnFilter_caller<16, T, D, BrdColReflect>
            },
            {
                0, 
                linearColumnFilter_caller<1 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<2 , T, D, BrdColWrap>,
                linearColumnFilter_caller<3 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<4 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<5 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<6 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<7 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<8 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<9 , T, D, BrdColWrap>, 
                linearColumnFilter_caller<10, T, D, BrdColWrap>, 
                linearColumnFilter_caller<11, T, D, BrdColWrap>, 
                linearColumnFilter_caller<12, T, D, BrdColWrap>, 
                linearColumnFilter_caller<13, T, D, BrdColWrap>, 
                linearColumnFilter_caller<14, T, D, BrdColWrap>, 
                linearColumnFilter_caller<15, T, D, BrdColWrap>, 
                linearColumnFilter_caller<16, T, D, BrdColWrap>,
            }
        };
        
        filter_column::loadKernel(kernel, ksize);

        callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
    }

    template void linearColumnFilter_gpu<float , uchar >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float4, uchar4>(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearColumnFilter_gpu<float , short >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearColumnFilter_gpu<float2, short2>(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float3, short3>(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , int   >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , float >(const DevMem2Db& src, const DevMem2Db& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
}}}
