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

namespace filter_krnls_row
{
    __constant__ float cLinearKernel[MAX_KERNEL_SIZE];

    void loadLinearKernel(const float kernel[], int ksize)
    {
        cudaSafeCall( cudaMemcpyToSymbol(cLinearKernel, kernel, ksize * sizeof(float)) );
    }

    template <typename T, size_t size> struct SmemType_
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type smem_t;
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
                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;
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

        typedef typename filter_krnls_row::SmemType<T>::smem_t smem_t;
        B<smem_t> b(src.cols);

        if (!b.is_range_safe(-BLOCK_DIM_X, (grid.x + 1) * BLOCK_DIM_X - 1))
        {
            cv::gpu::error("linearRowFilter: can't use specified border extrapolation, image is too small, "
                           "try bigger image or another border extrapolation mode", __FILE__, __LINE__);
        }

        filter_krnls_row::linearRowFilter<ksize, T, D><<<grid, threads, 0, stream>>>(src, dst, anchor, b);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T, typename D>
    void linearRowFilter_gpu(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream)
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
        
        filter_krnls_row::loadLinearKernel(kernel, ksize);

        callers[brd_type][ksize]((DevMem2D_<T>)src, (DevMem2D_<D>)dst, anchor, stream);
    }

    template void linearRowFilter_gpu<uchar , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<uchar4, float4>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearRowFilter_gpu<short , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    //template void linearRowFilter_gpu<short2, float2>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<short3, float3>(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<int   , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
    template void linearRowFilter_gpu<float , float >(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor, int brd_type, cudaStream_t stream);
}}}
