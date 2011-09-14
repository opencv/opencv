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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

#define MAX_KERNEL_SIZE 16
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

namespace filter_krnls_column
{
    __constant__ float cLinearKernel[MAX_KERNEL_SIZE];

    void loadLinearKernel(const float kernel[], int ksize)
    {
        cudaSafeCall( cudaMemcpyToSymbol(cLinearKernel, kernel, ksize * sizeof(float)) );
    }

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

            sDataColumn[ threadIdx.y                    * BLOCK_DIM_X] = b.at_low(y - BLOCK_DIM_Y, srcCol, src.step);
            sDataColumn[(threadIdx.y + BLOCK_DIM_Y)     * BLOCK_DIM_X] = b.at_high(y, srcCol, src.step);
            sDataColumn[(threadIdx.y + BLOCK_DIM_Y * 2) * BLOCK_DIM_X] = b.at_high(y + BLOCK_DIM_Y, srcCol, src.step);

            __syncthreads();

            if (y < src.rows)
            {
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;
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

        B<T> b(src.rows);

        if (!b.is_range_safe(-BLOCK_DIM_Y, (grid.y + 1) * BLOCK_DIM_Y - 1))
        {
            cv::gpu::error("linearColumnFilter: can't use specified border extrapolation, image is too small, "
                           "try bigger image or another border extrapolation mode", __FILE__, __LINE__);
        }

        filter_krnls_column::linearColumnFilter<ksize, T, D><<<grid, threads, 0, stream>>>(src, dst, anchor, b);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename D>
    void linearColumnFilter_gpu(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
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
        
        filter_krnls_column::loadLinearKernel(kernel, ksize);

        callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
    }

    template void linearColumnFilter_gpu<float , uchar >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float4, uchar4>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearColumnFilter_gpu<float , short >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearColumnFilter_gpu<float2, short2>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float3, short3>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , int   >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearColumnFilter_gpu<float , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
}}}
