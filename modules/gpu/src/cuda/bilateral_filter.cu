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

#include "internal_shared.hpp"

#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

using namespace cv::gpu;

typedef unsigned char uchar;
typedef unsigned short ushort;

//////////////////////////////////////////////////////////////////////////////////
/// Bilateral filtering

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        __device__ __forceinline__ float norm_l1(const float& a)  { return ::fabs(a); }
        __device__ __forceinline__ float norm_l1(const float2& a) { return ::fabs(a.x) + ::fabs(a.y); }
        __device__ __forceinline__ float norm_l1(const float3& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z); }
        __device__ __forceinline__ float norm_l1(const float4& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z) + ::fabs(a.w); }

        __device__ __forceinline__ float sqr(const float& a)  { return a * a; }

        template<typename T, typename B>
        __global__ void bilateral_kernel(const PtrStepSz<T> src, PtrStep<T> dst, const B b, const int ksz, const float sigma_spatial2_inv_half, const float sigma_color2_inv_half)
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type value_type;

            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= src.cols || y >= src.rows)
                return;

            value_type center = saturate_cast<value_type>(src(y, x));

            value_type sum1 = VecTraits<value_type>::all(0);
            float sum2 = 0;

            int r = ksz / 2;
            float r2 = (float)(r * r);

            int tx = x - r + ksz;
            int ty = y - r + ksz;

            if (x - ksz/2 >=0 && y - ksz/2 >=0 && tx < src.cols && ty < src.rows)
            {
                for (int cy = y - r; cy < ty; ++cy)
                    for (int cx = x - r; cx < tx; ++cx)
                    {
                        float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                        if (space2 > r2)
                            continue;

                        value_type value = saturate_cast<value_type>(src(cy, cx));

                        float weight = ::exp(space2 * sigma_spatial2_inv_half + sqr(norm_l1(value - center)) * sigma_color2_inv_half);
                        sum1 = sum1 + weight * value;
                        sum2 = sum2 + weight;
                    }
            }
            else
            {
                for (int cy = y - r; cy < ty; ++cy)
                    for (int cx = x - r; cx < tx; ++cx)
                    {
                        float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                        if (space2 > r2)
                            continue;

                        value_type value = saturate_cast<value_type>(b.at(cy, cx, src.data, src.step));

                        float weight = ::exp(space2 * sigma_spatial2_inv_half + sqr(norm_l1(value - center)) * sigma_color2_inv_half);

                        sum1 = sum1 + weight * value;
                        sum2 = sum2 + weight;
                    }
            }
            dst(y, x) = saturate_cast<T>(sum1 / sum2);
        }

        template<typename T, template <typename> class B>
        void bilateral_caller(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float sigma_spatial, float sigma_color, cudaStream_t stream)
        {
            dim3 block (32, 8);
            dim3 grid (divUp (src.cols, block.x), divUp (src.rows, block.y));

            B<T> b(src.rows, src.cols);

            float sigma_spatial2_inv_half = -0.5f/(sigma_spatial * sigma_spatial);
             float sigma_color2_inv_half = -0.5f/(sigma_color * sigma_color);

            cudaSafeCall( cudaFuncSetCacheConfig (bilateral_kernel<T, B<T> >, cudaFuncCachePreferL1) );
            bilateral_kernel<<<grid, block>>>((PtrStepSz<T>)src, (PtrStepSz<T>)dst, b, kernel_size, sigma_spatial2_inv_half, sigma_color2_inv_half);
            cudaSafeCall ( cudaGetLastError () );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template<typename T>
        void bilateral_filter_gpu(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float gauss_spatial_coeff, float gauss_color_coeff, int borderMode, cudaStream_t stream)
        {
            typedef void (*caller_t)(const PtrStepSzb& src, PtrStepSzb dst, int kernel_size, float sigma_spatial, float sigma_color, cudaStream_t stream);

            static caller_t funcs[] =
            {
                bilateral_caller<T, BrdReflect101>,
                bilateral_caller<T, BrdReplicate>,
                bilateral_caller<T, BrdConstant>,
                bilateral_caller<T, BrdReflect>,
                bilateral_caller<T, BrdWrap>,
            };
            funcs[borderMode](src, dst, kernel_size, gauss_spatial_coeff, gauss_color_coeff, stream);
        }
    }
}}}


#define OCV_INSTANTIATE_BILATERAL_FILTER(T) \
    template void cv::gpu::device::imgproc::bilateral_filter_gpu<T>(const PtrStepSzb&, PtrStepSzb, int, float, float, int, cudaStream_t);

OCV_INSTANTIATE_BILATERAL_FILTER(uchar)
//OCV_INSTANTIATE_BILATERAL_FILTER(uchar2)
OCV_INSTANTIATE_BILATERAL_FILTER(uchar3)
OCV_INSTANTIATE_BILATERAL_FILTER(uchar4)

//OCV_INSTANTIATE_BILATERAL_FILTER(schar)
//OCV_INSTANTIATE_BILATERAL_FILTER(schar2)
//OCV_INSTANTIATE_BILATERAL_FILTER(schar3)
//OCV_INSTANTIATE_BILATERAL_FILTER(schar4)

OCV_INSTANTIATE_BILATERAL_FILTER(short)
//OCV_INSTANTIATE_BILATERAL_FILTER(short2)
OCV_INSTANTIATE_BILATERAL_FILTER(short3)
OCV_INSTANTIATE_BILATERAL_FILTER(short4)

OCV_INSTANTIATE_BILATERAL_FILTER(ushort)
//OCV_INSTANTIATE_BILATERAL_FILTER(ushort2)
OCV_INSTANTIATE_BILATERAL_FILTER(ushort3)
OCV_INSTANTIATE_BILATERAL_FILTER(ushort4)

//OCV_INSTANTIATE_BILATERAL_FILTER(int)
//OCV_INSTANTIATE_BILATERAL_FILTER(int2)
//OCV_INSTANTIATE_BILATERAL_FILTER(int3)
//OCV_INSTANTIATE_BILATERAL_FILTER(int4)

OCV_INSTANTIATE_BILATERAL_FILTER(float)
//OCV_INSTANTIATE_BILATERAL_FILTER(float2)
OCV_INSTANTIATE_BILATERAL_FILTER(float3)
OCV_INSTANTIATE_BILATERAL_FILTER(float4)


#endif /* CUDA_DISABLER */
