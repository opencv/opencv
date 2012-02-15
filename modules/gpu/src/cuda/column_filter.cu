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
    namespace column_filter 
    {
        #define MAX_KERNEL_SIZE 32

        __constant__ float c_kernel[MAX_KERNEL_SIZE];

        void loadKernel(const float kernel[], int ksize)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_kernel, kernel, ksize * sizeof(float)) );
        }

        template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int PATCH_PER_BLOCK, int HALO_SIZE, int KSIZE, typename T, typename D, typename B>
        __global__ void linearColumnFilter(const DevMem2D_<T> src, PtrStep<D> dst, const int anchor, const B brd)
        {
            Static<KSIZE <= MAX_KERNEL_SIZE>::check();
            Static<HALO_SIZE * BLOCK_DIM_Y >= KSIZE>::check();
            Static<VecTraits<T>::cn == VecTraits<D>::cn>::check();

            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

            __shared__ sum_t smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];

            const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

            if (x >= src.cols)
                return;

            const T* src_col = src.ptr() + x;

            const int yStart = blockIdx.y * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(brd.at_low(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, src_col, src.step));

            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(brd.at_high(yStart + j * BLOCK_DIM_Y, src_col, src.step));

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = saturate_cast<sum_t>(brd.at_high(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, src_col, src.step));

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
            {
                const int y = yStart + j * BLOCK_DIM_Y;

                if (y >= src.rows)
                    return;

                sum_t sum = VecTraits<sum_t>::all(0);

                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum = sum + smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y - anchor + k][threadIdx.x] * c_kernel[k];

                dst(y, x) = saturate_cast<D>(sum);
            }
        }

        template <int KSIZE, typename T, typename D, template<typename> class B>
        void linearColumnFilter_caller(DevMem2D_<T> src, DevMem2D_<D> dst, int anchor, cudaStream_t stream)
        {
            const int BLOCK_DIM_X = 16;
            const int BLOCK_DIM_Y = 16;
            const int PATCH_PER_BLOCK = 4;

            const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
            const dim3 grid(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y * PATCH_PER_BLOCK));
            
            B<T> brd(src.rows);

            linearColumnFilter<BLOCK_DIM_X, BLOCK_DIM_Y, PATCH_PER_BLOCK, KSIZE <= 16 ? 1 : 2, KSIZE, T, D><<<grid, block, 0, stream>>>(src, dst, anchor, brd);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <typename T, typename D>
        void linearColumnFilter_gpu(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
        {
            typedef void (*caller_t)(DevMem2D_<T> src, DevMem2D_<D> dst, int anchor, cudaStream_t stream);

            static const caller_t callers[5][33] = 
            {
                {
                    0,
                    linearColumnFilter_caller< 1, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 2, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 3, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 4, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 5, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 6, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 7, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 8, T, D, BrdColReflect101>,
                    linearColumnFilter_caller< 9, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<10, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<11, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<12, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<13, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<14, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<15, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<16, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<17, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<18, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<19, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<20, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<21, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<22, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<23, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<24, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<25, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<26, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<27, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<28, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<29, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<30, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<31, T, D, BrdColReflect101>,
                    linearColumnFilter_caller<32, T, D, BrdColReflect101>
                },
                {
                    0,
                    linearColumnFilter_caller< 1, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 2, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 3, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 4, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 5, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 6, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 7, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 8, T, D, BrdColReplicate>,
                    linearColumnFilter_caller< 9, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<10, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<11, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<12, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<13, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<14, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<15, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<16, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<17, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<18, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<19, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<20, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<21, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<22, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<23, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<24, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<25, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<26, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<27, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<28, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<29, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<30, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<31, T, D, BrdColReplicate>,
                    linearColumnFilter_caller<32, T, D, BrdColReplicate>
                },
                {
                    0,
                    linearColumnFilter_caller< 1, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 2, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 3, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 4, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 5, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 6, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 7, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 8, T, D, BrdColConstant>,
                    linearColumnFilter_caller< 9, T, D, BrdColConstant>,
                    linearColumnFilter_caller<10, T, D, BrdColConstant>,
                    linearColumnFilter_caller<11, T, D, BrdColConstant>,
                    linearColumnFilter_caller<12, T, D, BrdColConstant>,
                    linearColumnFilter_caller<13, T, D, BrdColConstant>,
                    linearColumnFilter_caller<14, T, D, BrdColConstant>,
                    linearColumnFilter_caller<15, T, D, BrdColConstant>,
                    linearColumnFilter_caller<16, T, D, BrdColConstant>,
                    linearColumnFilter_caller<17, T, D, BrdColConstant>,
                    linearColumnFilter_caller<18, T, D, BrdColConstant>,
                    linearColumnFilter_caller<19, T, D, BrdColConstant>,
                    linearColumnFilter_caller<20, T, D, BrdColConstant>,
                    linearColumnFilter_caller<21, T, D, BrdColConstant>,
                    linearColumnFilter_caller<22, T, D, BrdColConstant>,
                    linearColumnFilter_caller<23, T, D, BrdColConstant>,
                    linearColumnFilter_caller<24, T, D, BrdColConstant>,
                    linearColumnFilter_caller<25, T, D, BrdColConstant>,
                    linearColumnFilter_caller<26, T, D, BrdColConstant>,
                    linearColumnFilter_caller<27, T, D, BrdColConstant>,
                    linearColumnFilter_caller<28, T, D, BrdColConstant>,
                    linearColumnFilter_caller<29, T, D, BrdColConstant>,
                    linearColumnFilter_caller<30, T, D, BrdColConstant>,
                    linearColumnFilter_caller<31, T, D, BrdColConstant>,
                    linearColumnFilter_caller<32, T, D, BrdColConstant>
                },
                {
                    0,
                    linearColumnFilter_caller< 1, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 2, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 3, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 4, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 5, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 6, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 7, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 8, T, D, BrdColReflect>,
                    linearColumnFilter_caller< 9, T, D, BrdColReflect>,
                    linearColumnFilter_caller<10, T, D, BrdColReflect>,
                    linearColumnFilter_caller<11, T, D, BrdColReflect>,
                    linearColumnFilter_caller<12, T, D, BrdColReflect>,
                    linearColumnFilter_caller<13, T, D, BrdColReflect>,
                    linearColumnFilter_caller<14, T, D, BrdColReflect>,
                    linearColumnFilter_caller<15, T, D, BrdColReflect>,
                    linearColumnFilter_caller<16, T, D, BrdColReflect>,
                    linearColumnFilter_caller<17, T, D, BrdColReflect>,
                    linearColumnFilter_caller<18, T, D, BrdColReflect>,
                    linearColumnFilter_caller<19, T, D, BrdColReflect>,
                    linearColumnFilter_caller<20, T, D, BrdColReflect>,
                    linearColumnFilter_caller<21, T, D, BrdColReflect>,
                    linearColumnFilter_caller<22, T, D, BrdColReflect>,
                    linearColumnFilter_caller<23, T, D, BrdColReflect>,
                    linearColumnFilter_caller<24, T, D, BrdColReflect>,
                    linearColumnFilter_caller<25, T, D, BrdColReflect>,
                    linearColumnFilter_caller<26, T, D, BrdColReflect>,
                    linearColumnFilter_caller<27, T, D, BrdColReflect>,
                    linearColumnFilter_caller<28, T, D, BrdColReflect>,
                    linearColumnFilter_caller<29, T, D, BrdColReflect>,
                    linearColumnFilter_caller<30, T, D, BrdColReflect>,
                    linearColumnFilter_caller<31, T, D, BrdColReflect>,
                    linearColumnFilter_caller<32, T, D, BrdColReflect>
                },
                {
                    0,
                    linearColumnFilter_caller< 1, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 2, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 3, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 4, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 5, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 6, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 7, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 8, T, D, BrdColWrap>,
                    linearColumnFilter_caller< 9, T, D, BrdColWrap>,
                    linearColumnFilter_caller<10, T, D, BrdColWrap>,
                    linearColumnFilter_caller<11, T, D, BrdColWrap>,
                    linearColumnFilter_caller<12, T, D, BrdColWrap>,
                    linearColumnFilter_caller<13, T, D, BrdColWrap>,
                    linearColumnFilter_caller<14, T, D, BrdColWrap>,
                    linearColumnFilter_caller<15, T, D, BrdColWrap>,
                    linearColumnFilter_caller<16, T, D, BrdColWrap>,
                    linearColumnFilter_caller<17, T, D, BrdColWrap>,
                    linearColumnFilter_caller<18, T, D, BrdColWrap>,
                    linearColumnFilter_caller<19, T, D, BrdColWrap>,
                    linearColumnFilter_caller<20, T, D, BrdColWrap>,
                    linearColumnFilter_caller<21, T, D, BrdColWrap>,
                    linearColumnFilter_caller<22, T, D, BrdColWrap>,
                    linearColumnFilter_caller<23, T, D, BrdColWrap>,
                    linearColumnFilter_caller<24, T, D, BrdColWrap>,
                    linearColumnFilter_caller<25, T, D, BrdColWrap>,
                    linearColumnFilter_caller<26, T, D, BrdColWrap>,
                    linearColumnFilter_caller<27, T, D, BrdColWrap>,
                    linearColumnFilter_caller<28, T, D, BrdColWrap>,
                    linearColumnFilter_caller<29, T, D, BrdColWrap>,
                    linearColumnFilter_caller<30, T, D, BrdColWrap>,
                    linearColumnFilter_caller<31, T, D, BrdColWrap>,
                    linearColumnFilter_caller<32, T, D, BrdColWrap>
                }               
            };
            
            loadKernel(kernel, ksize);

            callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
        }

        template void linearColumnFilter_gpu<float , uchar >(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
        template void linearColumnFilter_gpu<float4, uchar4>(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
        template void linearColumnFilter_gpu<float3, short3>(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
        template void linearColumnFilter_gpu<float , int   >(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
        template void linearColumnFilter_gpu<float , float >(DevMem2Db src, DevMem2Db dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    } // namespace column_filter
}}} // namespace cv { namespace gpu { namespace device
