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

#include "internal_shared.hpp"

#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

using namespace cv::gpu;

typedef unsigned char uchar;
typedef unsigned short ushort;

//////////////////////////////////////////////////////////////////////////////////
/// Non local means denosings

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

            for(float cy = -search_radius; cy <= search_radius; ++cy)
                for(float cx = -search_radius; cx <= search_radius; ++cx)
                {
                    float color2 = 0;
                    for(float by = -block_radius; by <= block_radius; ++by)
                        for(float bx = -block_radius; bx <= block_radius; ++bx)
                        {
                            value_type v1 = saturate_cast<value_type>(src(y + by, x + bx));
                            value_type v2 = saturate_cast<value_type>(src(y + cy + by, x + cx + bx));
                            color2 += norm2(v1 - v2);
                        }

                    float dist2 = cx * cx + cy * cy;
                    float w = __expf(color2 * h2_inv_half + dist2 * block_radius2_inv);
                    
                    sum1 = sum1 + saturate_cast<value_type>(src(y + cy, x + cy)) * w;
                    sum2 += w;
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
        template void nlm_bruteforce_gpu<uchar3>(const PtrStepSzb&, PtrStepSzb, int, int, float, int, cudaStream_t);
    }
}}}
