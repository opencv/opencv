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
#include "opencv2/gpu/device/static_check.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace row_filter 
    {
        #define MAX_KERNEL_SIZE 32

        __constant__ float c_kernel[MAX_KERNEL_SIZE];

        void loadKernel(const float kernel[], int ksize)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_kernel, kernel, ksize * sizeof(float)) );
        }

        template <int KSIZE, typename T, typename D, typename B>
        __global__ void linearRowFilter(const DevMem2D_<T> src, PtrStep<D> dst, const int anchor, const B brd)
        {
            #if __CUDA_ARCH__ >= 200
                const int BLOCK_DIM_X = 32;
                const int BLOCK_DIM_Y = 8;
                const int PATCH_PER_BLOCK = 4;
                const int HALO_SIZE = 1;
            #else
                const int BLOCK_DIM_X = 32;
                const int BLOCK_DIM_Y = 4;
                const int PATCH_PER_BLOCK = 4;
                const int HALO_SIZE = 1;
            #endif

            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

            __shared__ sum_t smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];
            
            const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

            if (y >= src.rows)
                return;

            const T* src_row = src.ptr(y);

            const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_low(xStart - (HALO_SIZE - j) * BLOCK_DIM_X, src_row));

            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_high(xStart + j * BLOCK_DIM_X, src_row));

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_high(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src_row));

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
            {
                const int x = xStart + j * BLOCK_DIM_X;

                if (x < src.cols)
                {
                    sum_t sum = VecTraits<sum_t>::all(0);

                    #pragma unroll
                    for (int k = 0; k < KSIZE; ++k)
                        sum = sum + smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X - anchor + k] * c_kernel[k];

                    dst(y, x) = saturate_cast<D>(sum);
                }
            }
        }

        template <int KSIZE, typename T, typename D, template<typename> class B>
        void linearRowFilter_caller(DevMem2D_<T> src, DevMem2D_<D> dst, int anchor, int cc, cudaStream_t stream)
        {
            int BLOCK_DIM_X;
            int BLOCK_DIM_Y;
            int PATCH_PER_BLOCK;

            if (cc >= 20)
            {
                BLOCK_DIM_X = 32;
                BLOCK_DIM_Y = 8;
                PATCH_PER_BLOCK = 4;
            }
            else
            {
                BLOCK_DIM_X = 32;
                BLOCK_DIM_Y = 4;
                PATCH_PER_BLOCK = 4;
            }

            const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
            const dim3 grid(divUp(src.cols, BLOCK_DIM_X * PATCH_PER_BLOCK), divUp(src.rows, BLOCK_DIM_Y));

            B<T> brd(src.cols);

            linearRowFilter<KSIZE, T, D><<<grid, block, 0, stream>>>(src, dst, anchor, brd);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <typename T, typename D>
        void linearRowFilter_gpu(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, int cc, cudaStream_t stream)
        {
            typedef void (*caller_t)(DevMem2D_<T> src, DevMem2D_<D> dst, int anchor, int cc, cudaStream_t stream);

            static const caller_t callers[5][33] = 
            {
                {
                    0,
                    linearRowFilter_caller< 1, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 2, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 3, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 4, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 5, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 6, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 7, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 8, T, D, BrdRowReflect101>,
                    linearRowFilter_caller< 9, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<10, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<11, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<12, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<13, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<14, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<15, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<16, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<17, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<18, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<19, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<20, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<21, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<22, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<23, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<24, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<25, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<26, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<27, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<28, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<29, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<30, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<31, T, D, BrdRowReflect101>,
                    linearRowFilter_caller<32, T, D, BrdRowReflect101>
                },
                {
                    0,
                    linearRowFilter_caller< 1, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 2, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 3, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 4, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 5, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 6, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 7, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 8, T, D, BrdRowReplicate>,
                    linearRowFilter_caller< 9, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<10, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<11, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<12, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<13, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<14, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<15, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<16, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<17, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<18, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<19, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<20, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<21, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<22, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<23, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<24, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<25, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<26, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<27, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<28, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<29, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<30, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<31, T, D, BrdRowReplicate>,
                    linearRowFilter_caller<32, T, D, BrdRowReplicate>
                },
                {
                    0,
                    linearRowFilter_caller< 1, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 2, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 3, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 4, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 5, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 6, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 7, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 8, T, D, BrdRowConstant>,
                    linearRowFilter_caller< 9, T, D, BrdRowConstant>,
                    linearRowFilter_caller<10, T, D, BrdRowConstant>,
                    linearRowFilter_caller<11, T, D, BrdRowConstant>,
                    linearRowFilter_caller<12, T, D, BrdRowConstant>,
                    linearRowFilter_caller<13, T, D, BrdRowConstant>,
                    linearRowFilter_caller<14, T, D, BrdRowConstant>,
                    linearRowFilter_caller<15, T, D, BrdRowConstant>,
                    linearRowFilter_caller<16, T, D, BrdRowConstant>,
                    linearRowFilter_caller<17, T, D, BrdRowConstant>,
                    linearRowFilter_caller<18, T, D, BrdRowConstant>,
                    linearRowFilter_caller<19, T, D, BrdRowConstant>,
                    linearRowFilter_caller<20, T, D, BrdRowConstant>,
                    linearRowFilter_caller<21, T, D, BrdRowConstant>,
                    linearRowFilter_caller<22, T, D, BrdRowConstant>,
                    linearRowFilter_caller<23, T, D, BrdRowConstant>,
                    linearRowFilter_caller<24, T, D, BrdRowConstant>,
                    linearRowFilter_caller<25, T, D, BrdRowConstant>,
                    linearRowFilter_caller<26, T, D, BrdRowConstant>,
                    linearRowFilter_caller<27, T, D, BrdRowConstant>,
                    linearRowFilter_caller<28, T, D, BrdRowConstant>,
                    linearRowFilter_caller<29, T, D, BrdRowConstant>,
                    linearRowFilter_caller<30, T, D, BrdRowConstant>,
                    linearRowFilter_caller<31, T, D, BrdRowConstant>,
                    linearRowFilter_caller<32, T, D, BrdRowConstant>
                },
                {
                    0,
                    linearRowFilter_caller< 1, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 2, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 3, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 4, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 5, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 6, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 7, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 8, T, D, BrdRowReflect>,
                    linearRowFilter_caller< 9, T, D, BrdRowReflect>,
                    linearRowFilter_caller<10, T, D, BrdRowReflect>,
                    linearRowFilter_caller<11, T, D, BrdRowReflect>,
                    linearRowFilter_caller<12, T, D, BrdRowReflect>,
                    linearRowFilter_caller<13, T, D, BrdRowReflect>,
                    linearRowFilter_caller<14, T, D, BrdRowReflect>,
                    linearRowFilter_caller<15, T, D, BrdRowReflect>,
                    linearRowFilter_caller<16, T, D, BrdRowReflect>,
                    linearRowFilter_caller<17, T, D, BrdRowReflect>,
                    linearRowFilter_caller<18, T, D, BrdRowReflect>,
                    linearRowFilter_caller<19, T, D, BrdRowReflect>,
                    linearRowFilter_caller<20, T, D, BrdRowReflect>,
                    linearRowFilter_caller<21, T, D, BrdRowReflect>,
                    linearRowFilter_caller<22, T, D, BrdRowReflect>,
                    linearRowFilter_caller<23, T, D, BrdRowReflect>,
                    linearRowFilter_caller<24, T, D, BrdRowReflect>,
                    linearRowFilter_caller<25, T, D, BrdRowReflect>,
                    linearRowFilter_caller<26, T, D, BrdRowReflect>,
                    linearRowFilter_caller<27, T, D, BrdRowReflect>,
                    linearRowFilter_caller<28, T, D, BrdRowReflect>,
                    linearRowFilter_caller<29, T, D, BrdRowReflect>,
                    linearRowFilter_caller<30, T, D, BrdRowReflect>,
                    linearRowFilter_caller<31, T, D, BrdRowReflect>,
                    linearRowFilter_caller<32, T, D, BrdRowReflect>
                },
                {
                    0,
                    linearRowFilter_caller< 1, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 2, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 3, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 4, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 5, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 6, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 7, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 8, T, D, BrdRowWrap>,
                    linearRowFilter_caller< 9, T, D, BrdRowWrap>,
                    linearRowFilter_caller<10, T, D, BrdRowWrap>,
                    linearRowFilter_caller<11, T, D, BrdRowWrap>,
                    linearRowFilter_caller<12, T, D, BrdRowWrap>,
                    linearRowFilter_caller<13, T, D, BrdRowWrap>,
                    linearRowFilter_caller<14, T, D, BrdRowWrap>,
                    linearRowFilter_caller<15, T, D, BrdRowWrap>,
                    linearRowFilter_caller<16, T, D, BrdRowWrap>,
                    linearRowFilter_caller<17, T, D, BrdRowWrap>,
                    linearRowFilter_caller<18, T, D, BrdRowWrap>,
                    linearRowFilter_caller<19, T, D, BrdRowWrap>,
                    linearRowFilter_caller<20, T, D, BrdRowWrap>,
                    linearRowFilter_caller<21, T, D, BrdRowWrap>,
                    linearRowFilter_caller<22, T, D, BrdRowWrap>,
                    linearRowFilter_caller<23, T, D, BrdRowWrap>,
                    linearRowFilter_caller<24, T, D, BrdRowWrap>,
                    linearRowFilter_caller<25, T, D, BrdRowWrap>,
                    linearRowFilter_caller<26, T, D, BrdRowWrap>,
                    linearRowFilter_caller<27, T, D, BrdRowWrap>,
                    linearRowFilter_caller<28, T, D, BrdRowWrap>,
                    linearRowFilter_caller<29, T, D, BrdRowWrap>,
                    linearRowFilter_caller<30, T, D, BrdRowWrap>,
                    linearRowFilter_caller<31, T, D, BrdRowWrap>,
                    linearRowFilter_caller<32, T, D, BrdRowWrap>
                }               
            };
            
            loadKernel(kernel, ksize);

            callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, cc, stream);
        }

        template void linearRowFilter_gpu<uchar , float >(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);
        template void linearRowFilter_gpu<uchar4, float4>(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);
        template void linearRowFilter_gpu<short3, float3>(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);
        template void linearRowFilter_gpu<int   , float >(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);
        template void linearRowFilter_gpu<float , float >(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);
    } // namespace row_filter
}}} // namespace cv { namespace gpu { namespace device
