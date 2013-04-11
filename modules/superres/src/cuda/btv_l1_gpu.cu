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

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_GPU

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace btv_l1_device
{
    void buildMotionMaps(PtrStepSzf forwardMotionX, PtrStepSzf forwardMotionY,
                         PtrStepSzf backwardMotionX, PtrStepSzf bacwardMotionY,
                         PtrStepSzf forwardMapX, PtrStepSzf forwardMapY,
                         PtrStepSzf backwardMapX, PtrStepSzf backwardMapY);

    template <int cn>
    void upscale(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream);

    void diffSign(PtrStepSzf src1, PtrStepSzf src2, PtrStepSzf dst, cudaStream_t stream);

    void loadBtvWeights(const float* weights, size_t count);
    template <int cn> void calcBtvRegularization(PtrStepSzb src, PtrStepSzb dst, int ksize);
}

namespace btv_l1_device
{
    __global__ void buildMotionMapsKernel(const PtrStepSzf forwardMotionX, const PtrStepf forwardMotionY,
                                          PtrStepf backwardMotionX, PtrStepf backwardMotionY,
                                          PtrStepf forwardMapX, PtrStepf forwardMapY,
                                          PtrStepf backwardMapX, PtrStepf backwardMapY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= forwardMotionX.cols || y >= forwardMotionX.rows)
            return;

        const float fx = forwardMotionX(y, x);
        const float fy = forwardMotionY(y, x);

        const float bx = backwardMotionX(y, x);
        const float by = backwardMotionY(y, x);

        forwardMapX(y, x) = x + bx;
        forwardMapY(y, x) = y + by;

        backwardMapX(y, x) = x + fx;
        backwardMapY(y, x) = y + fy;
    }

    void buildMotionMaps(PtrStepSzf forwardMotionX, PtrStepSzf forwardMotionY,
                         PtrStepSzf backwardMotionX, PtrStepSzf bacwardMotionY,
                         PtrStepSzf forwardMapX, PtrStepSzf forwardMapY,
                         PtrStepSzf backwardMapX, PtrStepSzf backwardMapY)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(forwardMapX.cols, block.x), divUp(forwardMapX.rows, block.y));

        buildMotionMapsKernel<<<grid, block>>>(forwardMotionX, forwardMotionY,
                                               backwardMotionX, bacwardMotionY,
                                               forwardMapX, forwardMapY,
                                               backwardMapX, backwardMapY);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    __global__ void upscaleKernel(const PtrStepSz<T> src, PtrStep<T> dst, const int scale)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src.cols || y >= src.rows)
            return;

        dst(y * scale, x * scale) = src(y, x);
    }

    template <int cn>
    void upscale(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream)
    {
        typedef typename TypeVec<float, cn>::vec_type src_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        upscaleKernel<src_t><<<grid, block, 0, stream>>>((PtrStepSz<src_t>) src, (PtrStepSz<src_t>) dst, scale);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void upscale<1>(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream);
    template void upscale<3>(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream);
    template void upscale<4>(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream);

    __device__ __forceinline__ float diffSign(float a, float b)
    {
        return a > b ? 1.0f : a < b ? -1.0f : 0.0f;
    }
    __device__ __forceinline__ float3 diffSign(const float3& a, const float3& b)
    {
        return make_float3(
            a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f,
            a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f,
            a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f
        );
    }
    __device__ __forceinline__ float4 diffSign(const float4& a, const float4& b)
    {
        return make_float4(
            a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f,
            a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f,
            a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f,
            0.0f
        );
    }

    struct DiffSign : binary_function<float, float, float>
    {
        __device__ __forceinline__ float operator ()(float a, float b) const
        {
            return diffSign(a, b);
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits<btv_l1_device::DiffSign> : DefaultTransformFunctorTraits<btv_l1_device::DiffSign>
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
}}}

namespace btv_l1_device
{
    void diffSign(PtrStepSzf src1, PtrStepSzf src2, PtrStepSzf dst, cudaStream_t stream)
    {
        transform(src1, src2, dst, DiffSign(), WithOutMask(), stream);
    }

    __constant__ float c_btvRegWeights[16*16];

    template <typename T>
    __global__ void calcBtvRegularizationKernel(const PtrStepSz<T> src, PtrStep<T> dst, const int ksize)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x + ksize;
        const int y = blockIdx.y * blockDim.y + threadIdx.y + ksize;

        if (y >= src.rows - ksize || x >= src.cols - ksize)
            return;

        const T srcVal = src(y, x);

        T dstVal = VecTraits<T>::all(0);

        for (int m = 0, count = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++count)
                dstVal = dstVal + c_btvRegWeights[count] * (diffSign(srcVal, src(y + m, x + l)) - diffSign(src(y - m, x - l), srcVal));
        }

        dst(y, x) = dstVal;
    }

    void loadBtvWeights(const float* weights, size_t count)
    {
        cudaSafeCall( cudaMemcpyToSymbol(c_btvRegWeights, weights, count * sizeof(float)) );
    }

    template <int cn>
    void calcBtvRegularization(PtrStepSzb src, PtrStepSzb dst, int ksize)
    {
        typedef typename TypeVec<float, cn>::vec_type src_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        calcBtvRegularizationKernel<src_t><<<grid, block>>>((PtrStepSz<src_t>) src, (PtrStepSz<src_t>) dst, ksize);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void calcBtvRegularization<1>(PtrStepSzb src, PtrStepSzb dst, int ksize);
    template void calcBtvRegularization<3>(PtrStepSzb src, PtrStepSzb dst, int ksize);
    template void calcBtvRegularization<4>(PtrStepSzb src, PtrStepSzb dst, int ksize);
}

#endif /* HAVE_OPENCV_GPU */
