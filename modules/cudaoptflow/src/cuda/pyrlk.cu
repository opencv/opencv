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

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/filters.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

#include <iostream>

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace pyrlk
{
    __constant__ int c_winSize_x;
    __constant__ int c_winSize_y;
    __constant__ int c_halfWin_x;
    __constant__ int c_halfWin_y;
    __constant__ int c_iters;

    texture<uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> tex_I8U(false, cudaFilterModeLinear, cudaAddressModeClamp);
    texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex_I8UC4(false, cudaFilterModeLinear, cudaAddressModeClamp);

    texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex_I16UC4(false, cudaFilterModeLinear, cudaAddressModeClamp);


    texture<float, cudaTextureType2D, cudaReadModeElementType> tex_If(false, cudaFilterModeLinear, cudaAddressModeClamp);
    texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_If4(false, cudaFilterModeLinear, cudaAddressModeClamp);

    texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_Ib(false, cudaFilterModePoint, cudaAddressModeClamp);

    texture<uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> tex_J8U(false, cudaFilterModeLinear, cudaAddressModeClamp);
    texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex_J8UC4(false, cudaFilterModeLinear, cudaAddressModeClamp);

    texture<ushort4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex_J16UC4(false, cudaFilterModeLinear, cudaAddressModeClamp);


    texture<float, cudaTextureType2D, cudaReadModeElementType> tex_Jf(false, cudaFilterModeLinear, cudaAddressModeClamp);
    texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_Jf4(false, cudaFilterModeLinear, cudaAddressModeClamp);


    template <int cn, typename T> struct Tex_I
    {
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<typename TypeVec<T, cn>::vec_type> I)
        {
            (void)I;
        }
    };

    template <> struct Tex_I<1, uchar>
    {
        static __device__ __forceinline__ float read(float x, float y)
        {
            return tex2D(tex_I8U, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<uchar>& I)
        {
            bindTexture(&tex_I8U, I);
        }
    };
    template <> struct Tex_I<1, ushort>
    {
        static __device__ __forceinline__ float read(float x, float y)
        {
            return 0.0;
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<ushort>& I)
        {
            (void)I;
        }
    };
    template <> struct Tex_I<1, int>
    {
        static __device__ __forceinline__ float read(float x, float y)
        {
            return 0.0;
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<int>& I)
        {
            (void)I;
        }
    };
    template <> struct Tex_I<1, float>
    {
        static __device__ __forceinline__ float read(float x, float y)
        {
            return tex2D(tex_If, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<float>& I)
        {
            bindTexture(&tex_If, I);
        }
    };
    // ****************** 3 channel specializations ************************
    template <> struct Tex_I<3, uchar>
    {
        static __device__ __forceinline__ float3 read(float x, float y)
        {
            return make_float3(0,0,0);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<uchar3> I)
        {
            (void)I;
        }
    };
    template <> struct Tex_I<3, ushort>
    {
        static __device__ __forceinline__ float3 read(float x, float y)
        {
            return make_float3(0, 0, 0);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<ushort3> I)
        {
            (void)I;
        }
    };
    template <> struct Tex_I<3, int>
    {
        static __device__ __forceinline__ float3 read(float x, float y)
        {
            return make_float3(0, 0, 0);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<int3> I)
        {
            (void)I;
        }
    };
    template <> struct Tex_I<3, float>
    {
        static __device__ __forceinline__ float3 read(float x, float y)
        {
            return make_float3(0, 0, 0);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<float3> I)
        {
            (void)I;
        }
    };
    // ****************** 4 channel specializations ************************

    template <> struct Tex_I<4, uchar>
    {
        static __device__ __forceinline__ float4 read(float x, float y)
        {
            return tex2D(tex_I8UC4, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<uchar4>& I)
        {
            bindTexture(&tex_I8UC4, I);
        }
    };
    template <> struct Tex_I<4, ushort>
    {
        static __device__ __forceinline__ float4 read(float x, float y)
        {
            return tex2D(tex_I16UC4, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<ushort4>& I)
        {
            bindTexture(&tex_I16UC4, I);
        }
    };
    template <> struct Tex_I<4, float>
    {
        static __device__ __forceinline__ float4 read(float x, float y)
        {
            return tex2D(tex_If4, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<float4>& I)
        {
            bindTexture(&tex_If4, I);
        }
    };
    // ************* J  ***************
    template <int cn, typename T> struct Tex_J
    {
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<typename TypeVec<T,cn>::vec_type>& J)
        {
            (void)J;
        }
    };
    template <> struct Tex_J<1, uchar>
    {
        static __device__ __forceinline__ float read(float x, float y)
        {
            return tex2D(tex_J8U, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<uchar>& J)
        {
            bindTexture(&tex_J8U, J);
        }
    };
    template <> struct Tex_J<1, float>
    {
        static __device__ __forceinline__ float read(float x, float y)
        {
            return tex2D(tex_Jf, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<float>& J)
        {
            bindTexture(&tex_Jf, J);
        }
    };
    // ************* 4 channel specializations ***************
    template <> struct Tex_J<4, uchar>
    {
        static __device__ __forceinline__ float4 read(float x, float y)
        {
            return tex2D(tex_J8UC4, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<uchar4>& J)
        {
            bindTexture(&tex_J8UC4, J);
        }
    };
    template <> struct Tex_J<4, ushort>
    {
        static __device__ __forceinline__ float4 read(float x, float y)
        {
            return tex2D(tex_J16UC4, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<ushort4>& J)
        {
            bindTexture(&tex_J16UC4, J);
        }
    };
    template <> struct Tex_J<4, float>
    {
        static __device__ __forceinline__ float4 read(float x, float y)
        {
            return tex2D(tex_Jf4, x, y);
        }
        static __host__ __forceinline__ void bindTexture_(PtrStepSz<float4>& J)
        {
            bindTexture(&tex_Jf4, J);
        }
    };

    __device__ __forceinline__ void accum(float& dst, const float& val)
    {
        dst += val;
    }
    __device__ __forceinline__ void accum(float& dst, const float2& val)
    {
        dst += val.x + val.y;
    }
    __device__ __forceinline__ void accum(float& dst, const float3& val)
    {
        dst += val.x + val.y + val.z;
    }
    __device__ __forceinline__ void accum(float& dst, const float4& val)
    {
        dst += val.x + val.y + val.z + val.w;
    }

    __device__ __forceinline__ float abs_(float a)
    {
        return ::fabsf(a);
    }
    __device__ __forceinline__ float4 abs_(const float4& a)
    {
        return abs(a);
    }
    __device__ __forceinline__ float2 abs_(const float2& a)
    {
        return abs(a);
    }
    __device__ __forceinline__ float3 abs_(const float3& a)
    {
        return abs(a);
    }


    template<typename T> __device__ __forceinline__ typename TypeVec<float, 1>::vec_type ToFloat(const typename TypeVec<T, 1>::vec_type& other)
    {
        return other;
    }
    template<typename T> __device__ __forceinline__  typename TypeVec<float, 2>::vec_type ToFloat(const typename TypeVec<T, 2>::vec_type& other)
    {
        typename TypeVec<float, 2>::vec_type ret;
        ret.x = other.x;
        ret.y = other.y;
        return ret;
    }
    template<typename T> __device__ __forceinline__  typename TypeVec<float, 3>::vec_type ToFloat(const typename TypeVec<T, 3>::vec_type& other)
    {
        typename TypeVec<float, 3>::vec_type ret;
        ret.x = other.x;
        ret.y = other.y;
        ret.z = other.z;
        return ret;
    }
    template<typename T> __device__ __forceinline__  typename TypeVec<float, 4>::vec_type ToFloat(const typename TypeVec<T, 4>::vec_type& other)
    {
        typename TypeVec<float, 4>::vec_type ret;
        ret.x = other.x;
        ret.y = other.y;
        ret.z = other.z;
        ret.w = other.w;
        return ret;
    }

    template <typename T>
    struct DenormalizationFactor
    {
        static __device__ __forceinline__ float factor()
        {
            return 1.0f;
        }
    };

    template <>
    struct DenormalizationFactor<uchar>
    {
        static __device__ __forceinline__ float factor()
        {
            return 255.0f;
        }
    };

    template <int cn, int PATCH_X, int PATCH_Y, bool calcErr, typename T>
    __global__ void sparseKernel(const float2* prevPts, float2* nextPts, uchar* status, float* err, const int level, const int rows, const int cols)
    {
    #if __CUDA_ARCH__ <= 110
        const int BLOCK_SIZE = 128;
    #else
        const int BLOCK_SIZE = 256;
    #endif

        __shared__ float smem1[BLOCK_SIZE];
        __shared__ float smem2[BLOCK_SIZE];
        __shared__ float smem3[BLOCK_SIZE];

        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        float2 prevPt = prevPts[blockIdx.x];
        prevPt.x *= (1.0f / (1 << level));
        prevPt.y *= (1.0f / (1 << level));

        if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[blockIdx.x] = 0;

            return;
        }

        prevPt.x -= c_halfWin_x;
        prevPt.y -= c_halfWin_y;

        // extract the patch from the first image, compute covariation matrix of derivatives

        float A11 = 0;
        float A12 = 0;
        float A22 = 0;

        typedef typename TypeVec<float, cn>::vec_type work_type;

        work_type I_patch   [PATCH_Y][PATCH_X];
        work_type dIdx_patch[PATCH_Y][PATCH_X];
        work_type dIdy_patch[PATCH_Y][PATCH_X];

        for (int yBase = threadIdx.y, i = 0; yBase < c_winSize_y; yBase += blockDim.y, ++i)
        {
            for (int xBase = threadIdx.x, j = 0; xBase < c_winSize_x; xBase += blockDim.x, ++j)
            {
                float x = prevPt.x + xBase + 0.5f;
                float y = prevPt.y + yBase + 0.5f;

                I_patch[i][j] = Tex_I<cn, T>::read(x, y);

                // Sharr Deriv

                work_type dIdx = 3.0f * Tex_I<cn,T>::read(x+1, y-1) + 10.0f * Tex_I<cn, T>::read(x+1, y) + 3.0f * Tex_I<cn,T>::read(x+1, y+1) -
                                 (3.0f * Tex_I<cn,T>::read(x-1, y-1) + 10.0f * Tex_I<cn, T>::read(x-1, y) + 3.0f * Tex_I<cn,T>::read(x-1, y+1));

                work_type dIdy = 3.0f * Tex_I<cn,T>::read(x-1, y+1) + 10.0f * Tex_I<cn, T>::read(x, y+1) + 3.0f * Tex_I<cn,T>::read(x+1, y+1) -
                                (3.0f * Tex_I<cn,T>::read(x-1, y-1) + 10.0f * Tex_I<cn, T>::read(x, y-1) + 3.0f * Tex_I<cn,T>::read(x+1, y-1));

                dIdx_patch[i][j] = dIdx;
                dIdy_patch[i][j] = dIdy;

                accum(A11, dIdx * dIdx);
                accum(A12, dIdx * dIdy);
                accum(A22, dIdy * dIdy);
            }
        }

        reduce<BLOCK_SIZE>(smem_tuple(smem1, smem2, smem3), thrust::tie(A11, A12, A22), tid, thrust::make_tuple(plus<float>(), plus<float>(), plus<float>()));

    #if __CUDA_ARCH__ >= 300
        if (tid == 0)
        {
            smem1[0] = A11;
            smem2[0] = A12;
            smem3[0] = A22;
        }
    #endif

        __syncthreads();

        A11 = smem1[0];
        A12 = smem2[0];
        A22 = smem3[0];

        float D = A11 * A22 - A12 * A12;

        if (D < numeric_limits<float>::epsilon())
        {
            if (tid == 0 && level == 0)
                status[blockIdx.x] = 0;

            return;
        }

        D = 1.f / D;

        A11 *= D;
        A12 *= D;
        A22 *= D;

        float2 nextPt = nextPts[blockIdx.x];
        nextPt.x *= 2.f;
        nextPt.y *= 2.f;

        nextPt.x -= c_halfWin_x;
        nextPt.y -= c_halfWin_y;

        for (int k = 0; k < c_iters; ++k)
        {
            if (nextPt.x < -c_halfWin_x || nextPt.x >= cols || nextPt.y < -c_halfWin_y || nextPt.y >= rows)
            {
                if (tid == 0 && level == 0)
                    status[blockIdx.x] = 0;

                return;
            }

            float b1 = 0;
            float b2 = 0;

            for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
            {
                for (int x = threadIdx.x, j = 0; x < c_winSize_x; x += blockDim.x, ++j)
                {
                    work_type I_val = I_patch[i][j];
                    work_type J_val = Tex_J<cn, T>::read(nextPt.x + x + 0.5f, nextPt.y + y + 0.5f);

                    work_type diff = (J_val - I_val) * 32.0f;

                    accum(b1, diff * dIdx_patch[i][j]);
                    accum(b2, diff * dIdy_patch[i][j]);
                }
            }

            reduce<BLOCK_SIZE>(smem_tuple(smem1, smem2), thrust::tie(b1, b2), tid, thrust::make_tuple(plus<float>(), plus<float>()));

        #if __CUDA_ARCH__ >= 300
            if (tid == 0)
            {
                smem1[0] = b1;
                smem2[0] = b2;
            }
        #endif

            __syncthreads();

            b1 = smem1[0];
            b2 = smem2[0];

            float2 delta;
            delta.x = A12 * b2 - A22 * b1;
            delta.y = A12 * b1 - A11 * b2;

            nextPt.x += delta.x;
            nextPt.y += delta.y;

            if (::fabs(delta.x) < 0.01f && ::fabs(delta.y) < 0.01f)
                break;
        }

        float errval = 0;
        if (calcErr)
        {
            for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
            {
                for (int x = threadIdx.x, j = 0; x < c_winSize_x; x += blockDim.x, ++j)
                {
                    work_type I_val = I_patch[i][j];
                    work_type J_val = Tex_J<cn, T>::read(nextPt.x + x + 0.5f, nextPt.y + y + 0.5f);

                    work_type diff = J_val - I_val;

                    accum(errval, abs_(diff));
                }
            }

            reduce<BLOCK_SIZE>(smem1, errval, tid, plus<float>());
        }

        if (tid == 0)
        {
            nextPt.x += c_halfWin_x;
            nextPt.y += c_halfWin_y;

            nextPts[blockIdx.x] = nextPt;

            if (calcErr)
                err[blockIdx.x] = static_cast<float>(errval) / (::min(cn, 3) * c_winSize_x * c_winSize_y) * DenormalizationFactor<T>::factor();
        }
    }

    // Kernel, uses non texture fetches
    template <int PATCH_X, int PATCH_Y, bool calcErr, int cn, typename T, typename Ptr2D>
    __global__ void sparseKernel_(Ptr2D I, Ptr2D J, const float2* prevPts, float2* nextPts, uchar* status, float* err, const int level, const int rows, const int cols)
    {
#if __CUDA_ARCH__ <= 110
        const int BLOCK_SIZE = 128;
#else
        const int BLOCK_SIZE = 256;
#endif

        __shared__ float smem1[BLOCK_SIZE];
        __shared__ float smem2[BLOCK_SIZE];
        __shared__ float smem3[BLOCK_SIZE];

        const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

        float2 prevPt = prevPts[blockIdx.x];
        prevPt.x *= (1.0f / (1 << level));
        prevPt.y *= (1.0f / (1 << level));

        if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[blockIdx.x] = 0;

            return;
        }

        prevPt.x -= c_halfWin_x;
        prevPt.y -= c_halfWin_y;

        // extract the patch from the first image, compute covariation matrix of derivatives

        float A11 = 0;
        float A12 = 0;
        float A22 = 0;

        typedef typename TypeVec<float, cn>::vec_type work_type;

        work_type I_patch[PATCH_Y][PATCH_X];
        work_type dIdx_patch[PATCH_Y][PATCH_X];
        work_type dIdy_patch[PATCH_Y][PATCH_X];

        for (int yBase = threadIdx.y, i = 0; yBase < c_winSize_y; yBase += blockDim.y, ++i)
        {
            for (int xBase = threadIdx.x, j = 0; xBase < c_winSize_x; xBase += blockDim.x, ++j)
            {
                float x = prevPt.x + xBase + 0.5f;
                float y = prevPt.y + yBase + 0.5f;

                I_patch[i][j] = ToFloat<T>(I(y, x));

                // Sharr Deriv

                work_type dIdx = 3.0f * I(y - 1, x + 1) + 10.0f * I(y, x + 1) + 3.0f * I(y + 1, x + 1) -
                    (3.0f * I(y - 1, x - 1) + 10.0f * I(y, x - 1) + 3.0f * I(y + 1 , x - 1));

                work_type dIdy = 3.0f * I(y + 1, x - 1) + 10.0f * I(y + 1, x) + 3.0f * I(y+1, x + 1) -
                    (3.0f * I(y - 1, x - 1) + 10.0f * I(y-1, x) + 3.0f * I(y - 1, x + 1));

                dIdx_patch[i][j] = dIdx;
                dIdy_patch[i][j] = dIdy;

                accum(A11, dIdx * dIdx);
                accum(A12, dIdx * dIdy);
                accum(A22, dIdy * dIdy);
            }
        }

        reduce<BLOCK_SIZE>(smem_tuple(smem1, smem2, smem3), thrust::tie(A11, A12, A22), tid, thrust::make_tuple(plus<float>(), plus<float>(), plus<float>()));

#if __CUDA_ARCH__ >= 300
        if (tid == 0)
        {
            smem1[0] = A11;
            smem2[0] = A12;
            smem3[0] = A22;
        }
#endif

        __syncthreads();

        A11 = smem1[0];
        A12 = smem2[0];
        A22 = smem3[0];

        float D = A11 * A22 - A12 * A12;

        if (D < numeric_limits<float>::epsilon())
        {
            if (tid == 0 && level == 0)
                status[blockIdx.x] = 0;

            return;
        }

        D = 1.f / D;

        A11 *= D;
        A12 *= D;
        A22 *= D;

        float2 nextPt = nextPts[blockIdx.x];
        nextPt.x *= 2.f;
        nextPt.y *= 2.f;

        nextPt.x -= c_halfWin_x;
        nextPt.y -= c_halfWin_y;

        for (int k = 0; k < c_iters; ++k)
        {
            if (nextPt.x < -c_halfWin_x || nextPt.x >= cols || nextPt.y < -c_halfWin_y || nextPt.y >= rows)
            {
                if (tid == 0 && level == 0)
                    status[blockIdx.x] = 0;

                return;
            }

            float b1 = 0;
            float b2 = 0;

            for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
            {
                for (int x = threadIdx.x, j = 0; x < c_winSize_x; x += blockDim.x, ++j)
                {
                    work_type I_val = I_patch[i][j];
                    work_type J_val = ToFloat<T>(J(nextPt.y + y + 0.5f, nextPt.x + x + 0.5f));

                    work_type diff = (J_val - I_val) * 32.0f;

                    accum(b1, diff * dIdx_patch[i][j]);
                    accum(b2, diff * dIdy_patch[i][j]);
                }
            }

            reduce<BLOCK_SIZE>(smem_tuple(smem1, smem2), thrust::tie(b1, b2), tid, thrust::make_tuple(plus<float>(), plus<float>()));

#if __CUDA_ARCH__ >= 300
            if (tid == 0)
            {
                smem1[0] = b1;
                smem2[0] = b2;
            }
#endif

            __syncthreads();

            b1 = smem1[0];
            b2 = smem2[0];

            float2 delta;
            delta.x = A12 * b2 - A22 * b1;
            delta.y = A12 * b1 - A11 * b2;

            nextPt.x += delta.x;
            nextPt.y += delta.y;

            if (::fabs(delta.x) < 0.01f && ::fabs(delta.y) < 0.01f)
                break;
        }

        float errval = 0;
        if (calcErr)
        {
            for (int y = threadIdx.y, i = 0; y < c_winSize_y; y += blockDim.y, ++i)
            {
                for (int x = threadIdx.x, j = 0; x < c_winSize_x; x += blockDim.x, ++j)
                {
                    work_type I_val = I_patch[i][j];
                    work_type J_val = ToFloat<T>(J(nextPt.y + y + 0.5f, nextPt.x + x + 0.5f));

                    work_type diff = J_val - I_val;

                    accum(errval, abs_(diff));
                }
            }

            reduce<BLOCK_SIZE>(smem1, errval, tid, plus<float>());
        }

        if (tid == 0)
        {
            nextPt.x += c_halfWin_x;
            nextPt.y += c_halfWin_y;

            nextPts[blockIdx.x] = nextPt;

            if (calcErr)
                err[blockIdx.x] = static_cast<float>(errval) / (::min(cn, 3)*c_winSize_x * c_winSize_y);
        }
    } // __global__ void sparseKernel_


    template <int cn, int PATCH_X, int PATCH_Y, typename T> class sparse_caller
    {
    public:
        static void call(PtrStepSz<typename TypeVec<T, cn>::vec_type> I, PtrStepSz<typename TypeVec<T, cn>::vec_type> J, int rows, int cols, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, cudaStream_t stream)
        {
            dim3 grid(ptcount);
            (void)I;
            (void)J;
            if (level == 0 && err)
                sparseKernel<cn, PATCH_X, PATCH_Y, true, T> <<<grid, block, 0, stream >>>(prevPts, nextPts, status, err, level, rows, cols);
            else
                sparseKernel<cn, PATCH_X, PATCH_Y, false, T> <<<grid, block, 0, stream >>>(prevPts, nextPts, status, err, level, rows, cols);

            cudaSafeCall(cudaGetLastError());

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    };
    // Specialization to use non texture path because for some reason the texture path keeps failing accuracy tests
    template<int PATCH_X, int PATCH_Y> class sparse_caller<1, PATCH_X, PATCH_Y, unsigned short>
    {
    public:
        typedef typename TypeVec<unsigned short, 1>::vec_type work_type;
        typedef PtrStepSz<work_type> Ptr2D;
        typedef BrdConstant<work_type> BrdType;
        typedef BorderReader<Ptr2D, BrdType> Reader;
        typedef LinearFilter<Reader> Filter;
        static void call(Ptr2D I, Ptr2D J, int rows, int cols, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, cudaStream_t stream)
        {
            dim3 grid(ptcount);
            if (level == 0 && err)
            {
                sparseKernel_<PATCH_X, PATCH_Y, true, 1, unsigned short> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            else
            {
                sparseKernel_<PATCH_X, PATCH_Y, false, 1, unsigned short> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            cudaSafeCall(cudaGetLastError());

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    };
    // Specialization for int because the texture path keeps failing
    template<int PATCH_X, int PATCH_Y> class sparse_caller<1, PATCH_X, PATCH_Y, int>
    {
    public:
        typedef typename TypeVec<int, 1>::vec_type work_type;
        typedef PtrStepSz<work_type> Ptr2D;
        typedef BrdConstant<work_type> BrdType;
        typedef BorderReader<Ptr2D, BrdType> Reader;
        typedef LinearFilter<Reader> Filter;
        static void call(Ptr2D I, Ptr2D J, int rows, int cols, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, cudaStream_t stream)
        {
            dim3 grid(ptcount);
            if (level == 0 && err)
            {
                sparseKernel_<PATCH_X, PATCH_Y, true, 1, int> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            else
            {
                sparseKernel_<PATCH_X, PATCH_Y, false, 1, int> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            cudaSafeCall(cudaGetLastError());

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    };
    template<int PATCH_X, int PATCH_Y> class sparse_caller<4, PATCH_X, PATCH_Y, int>
    {
    public:
        typedef typename TypeVec<int, 4>::vec_type work_type;
        typedef PtrStepSz<work_type> Ptr2D;
        typedef BrdConstant<work_type> BrdType;
        typedef BorderReader<Ptr2D, BrdType> Reader;
        typedef LinearFilter<Reader> Filter;
        static void call(Ptr2D I, Ptr2D J, int rows, int cols, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, cudaStream_t stream)
        {
            dim3 grid(ptcount);
            if (level == 0 && err)
            {
                sparseKernel_<PATCH_X, PATCH_Y, true, 4, int> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            else
            {
                sparseKernel_<PATCH_X, PATCH_Y, false, 4, int> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            cudaSafeCall(cudaGetLastError());

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    };
    using namespace cv::cuda::device;
    template <int PATCH_X, int PATCH_Y, typename T> class sparse_caller<3, PATCH_X, PATCH_Y, T>
    {
    public:
        typedef typename TypeVec<T, 3>::vec_type work_type;
        typedef PtrStepSz<work_type> Ptr2D;
        typedef BrdConstant<work_type> BrdType;
        typedef BorderReader<Ptr2D, BrdType> Reader;
        typedef LinearFilter<Reader> Filter;
        static void call(Ptr2D I, Ptr2D J, int rows, int cols, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, cudaStream_t stream)
        {
            dim3 grid(ptcount);
            if (level == 0 && err)
            {
                sparseKernel_<PATCH_X, PATCH_Y, true, 3, T> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            else
            {
                sparseKernel_<PATCH_X, PATCH_Y, false, 3, T> <<<grid, block, 0, stream >>>(
                    Filter(Reader(I, BrdType(rows, cols))),
                    Filter(Reader(J, BrdType(rows, cols))),
                    prevPts, nextPts, status, err, level, rows, cols);
            }
            cudaSafeCall(cudaGetLastError());

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    };


    template <bool calcErr>
    __global__ void denseKernel(PtrStepf u, PtrStepf v, const PtrStepf prevU, const PtrStepf prevV, PtrStepf err, const int rows, const int cols)
    {
        extern __shared__ int smem[];

        const int patchWidth  = blockDim.x + 2 * c_halfWin_x;
        const int patchHeight = blockDim.y + 2 * c_halfWin_y;

        int* I_patch = smem;
        int* dIdx_patch = I_patch + patchWidth * patchHeight;
        int* dIdy_patch = dIdx_patch + patchWidth * patchHeight;

        const int xBase = blockIdx.x * blockDim.x;
        const int yBase = blockIdx.y * blockDim.y;

        for (int i = threadIdx.y; i < patchHeight; i += blockDim.y)
        {
            for (int j = threadIdx.x; j < patchWidth; j += blockDim.x)
            {
                float x = xBase - c_halfWin_x + j + 0.5f;
                float y = yBase - c_halfWin_y + i + 0.5f;

                I_patch[i * patchWidth + j] = tex2D(tex_If, x, y);

                // Sharr Deriv

                dIdx_patch[i * patchWidth + j] = 3 * tex2D(tex_If, x+1, y-1) + 10 * tex2D(tex_If, x+1, y) + 3 * tex2D(tex_If, x+1, y+1) -
                                                (3 * tex2D(tex_If, x-1, y-1) + 10 * tex2D(tex_If, x-1, y) + 3 * tex2D(tex_If, x-1, y+1));

                dIdy_patch[i * patchWidth + j] = 3 * tex2D(tex_If, x-1, y+1) + 10 * tex2D(tex_If, x, y+1) + 3 * tex2D(tex_If, x+1, y+1) -
                                                (3 * tex2D(tex_If, x-1, y-1) + 10 * tex2D(tex_If, x, y-1) + 3 * tex2D(tex_If, x+1, y-1));
            }
        }

        __syncthreads();

        const int x = xBase + threadIdx.x;
        const int y = yBase + threadIdx.y;

        if (x >= cols || y >= rows)
            return;


        int A11i = 0;
        int A12i = 0;
        int A22i = 0;

        for (int i = 0; i < c_winSize_y; ++i)
        {
            for (int j = 0; j < c_winSize_x; ++j)
            {
                int dIdx = dIdx_patch[(threadIdx.y + i) * patchWidth + (threadIdx.x + j)];
                int dIdy = dIdy_patch[(threadIdx.y + i) * patchWidth + (threadIdx.x + j)];

                A11i += dIdx * dIdx;
                A12i += dIdx * dIdy;
                A22i += dIdy * dIdy;
            }
        }

        float A11 = A11i;
        float A12 = A12i;
        float A22 = A22i;

        float D = A11 * A22 - A12 * A12;

        if (D < numeric_limits<float>::epsilon())
        {
            if (calcErr)
                err(y, x) = numeric_limits<float>::max();
            return;
        }

        D = 1.f / D;

        A11 *= D;
        A12 *= D;
        A22 *= D;

        float2 nextPt;
        nextPt.x = x + prevU(y/2, x/2) * 2.0f;
        nextPt.y = y + prevV(y/2, x/2) * 2.0f;

        for (int k = 0; k < c_iters; ++k)
        {
            if (nextPt.x < 0 || nextPt.x >= cols || nextPt.y < 0 || nextPt.y >= rows)
            {
                if (calcErr)
                    err(y, x) = numeric_limits<float>::max();

                return;
            }

            int b1 = 0;
            int b2 = 0;

            for (int i = 0; i < c_winSize_y; ++i)
            {
                for (int j = 0; j < c_winSize_x; ++j)
                {
                    int I = I_patch[(threadIdx.y + i) * patchWidth + threadIdx.x + j];
                    int J = tex2D(tex_Jf, nextPt.x - c_halfWin_x + j + 0.5f, nextPt.y - c_halfWin_y + i + 0.5f);

                    int diff = (J - I) * 32;

                    int dIdx = dIdx_patch[(threadIdx.y + i) * patchWidth + (threadIdx.x + j)];
                    int dIdy = dIdy_patch[(threadIdx.y + i) * patchWidth + (threadIdx.x + j)];

                    b1 += diff * dIdx;
                    b2 += diff * dIdy;
                }
            }


            float2 delta;
            delta.x = A12 * b2 - A22 * b1;
            delta.y = A12 * b1 - A11 * b2;

            nextPt.x += delta.x;
            nextPt.y += delta.y;

            if (::fabs(delta.x) < 0.01f && ::fabs(delta.y) < 0.01f)
                break;
        }

        u(y, x) = nextPt.x - x;
        v(y, x) = nextPt.y - y;

        if (calcErr)
        {
            int errval = 0;

            for (int i = 0; i < c_winSize_y; ++i)
            {
                for (int j = 0; j < c_winSize_x; ++j)
                {
                    int I = I_patch[(threadIdx.y + i) * patchWidth + threadIdx.x + j];
                    int J = tex2D(tex_Jf, nextPt.x - c_halfWin_x + j + 0.5f, nextPt.y - c_halfWin_y + i + 0.5f);

                    errval += ::abs(J - I);
                }
            }

            err(y, x) = static_cast<float>(errval) / (c_winSize_x * c_winSize_y);
        }
    }

    void loadWinSize(int* winSize, int* halfWinSize, cudaStream_t stream)
    {
        cudaSafeCall( cudaMemcpyToSymbolAsync(c_winSize_x, winSize, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
        cudaSafeCall( cudaMemcpyToSymbolAsync(c_winSize_y, winSize + 1, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );

        cudaSafeCall( cudaMemcpyToSymbolAsync(c_halfWin_x, halfWinSize, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
        cudaSafeCall( cudaMemcpyToSymbolAsync(c_halfWin_y, halfWinSize + 1, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
    }

    void loadIters(int* iters, cudaStream_t stream)
    {
        cudaSafeCall( cudaMemcpyToSymbolAsync(c_iters, iters, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
    }

    void loadConstants(int2 winSize_, int iters_, cudaStream_t stream)
    {
        static int2 winSize = make_int2(0,0);
        if(winSize.x != winSize_.x || winSize.y != winSize_.y)
        {
            winSize = winSize_;
            cudaSafeCall( cudaMemcpyToSymbolAsync(c_winSize_x, &winSize.x, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
            cudaSafeCall( cudaMemcpyToSymbolAsync(c_winSize_y, &winSize.y, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
        }

        static int2 halfWin = make_int2(0,0);
        int2 half = make_int2((winSize.x - 1) / 2, (winSize.y - 1) / 2);
        if(halfWin.x != half.x || halfWin.y != half.y)
        {
            halfWin = half;
            cudaSafeCall( cudaMemcpyToSymbolAsync(c_halfWin_x, &halfWin.x, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
            cudaSafeCall( cudaMemcpyToSymbolAsync(c_halfWin_y, &halfWin.y, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
        }

        static int iters = 0;
        if(iters != iters_)
        {
            iters = iters_;
            cudaSafeCall( cudaMemcpyToSymbolAsync(c_iters, &iters, sizeof(int), 0, cudaMemcpyHostToDevice, stream) );
        }
    }

    template<typename T, int cn> struct pyrLK_caller
    {
        static void sparse(PtrStepSz<typename TypeVec<T, cn>::vec_type> I, PtrStepSz<typename TypeVec<T, cn>::vec_type> J, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
            int level, dim3 block, dim3 patch, cudaStream_t stream)
        {
            typedef void(*func_t)(PtrStepSz<typename TypeVec<T, cn>::vec_type> I, PtrStepSz<typename TypeVec<T, cn>::vec_type> J,
                int rows, int cols, const float2* prevPts, float2* nextPts, uchar* status, float* err, int ptcount,
                int level, dim3 block, cudaStream_t stream);

            static const func_t funcs[5][5] =
            {
                { sparse_caller<cn, 1, 1,T>::call, sparse_caller<cn, 2, 1,T>::call, sparse_caller<cn, 3, 1,T>::call, sparse_caller<cn, 4, 1,T>::call, sparse_caller<cn, 5, 1,T>::call },
                { sparse_caller<cn, 1, 2,T>::call, sparse_caller<cn, 2, 2,T>::call, sparse_caller<cn, 3, 2,T>::call, sparse_caller<cn, 4, 2,T>::call, sparse_caller<cn, 5, 2,T>::call },
                { sparse_caller<cn, 1, 3,T>::call, sparse_caller<cn, 2, 3,T>::call, sparse_caller<cn, 3, 3,T>::call, sparse_caller<cn, 4, 3,T>::call, sparse_caller<cn, 5, 3,T>::call },
                { sparse_caller<cn, 1, 4,T>::call, sparse_caller<cn, 2, 4,T>::call, sparse_caller<cn, 3, 4,T>::call, sparse_caller<cn, 4, 4,T>::call, sparse_caller<cn, 5, 4,T>::call },
                { sparse_caller<cn, 1, 5,T>::call, sparse_caller<cn, 2, 5,T>::call, sparse_caller<cn, 3, 5,T>::call, sparse_caller<cn, 4, 5,T>::call, sparse_caller<cn, 5, 5,T>::call }
            };

            Tex_I<cn, T>::bindTexture_(I);
            Tex_J<cn, T>::bindTexture_(J);

            funcs[patch.y - 1][patch.x - 1](I, J, I.rows, I.cols, prevPts, nextPts, status, err, ptcount,
                level, block, stream);
        }
        static void dense(PtrStepSz<T> I, PtrStepSz<T> J, PtrStepSzf u, PtrStepSzf v, PtrStepSzf prevU, PtrStepSzf prevV, PtrStepSzf err, int2 winSize, cudaStream_t stream)
        {
            dim3 block(16, 16);
            dim3 grid(divUp(I.cols, block.x), divUp(I.rows, block.y));
            Tex_I<1, T>::bindTexture_(I);
            Tex_J<1, T>::bindTexture_(J);

            int2 halfWin = make_int2((winSize.x - 1) / 2, (winSize.y - 1) / 2);
            const int patchWidth = block.x + 2 * halfWin.x;
            const int patchHeight = block.y + 2 * halfWin.y;
            size_t smem_size = 3 * patchWidth * patchHeight * sizeof(int);

            if (err.data)
            {
                denseKernel<true> << <grid, block, smem_size, stream >> >(u, v, prevU, prevV, err, I.rows, I.cols);
                cudaSafeCall(cudaGetLastError());
            }
            else
            {
                denseKernel<false> << <grid, block, smem_size, stream >> >(u, v, prevU, prevV, PtrStepf(), I.rows, I.cols);
                cudaSafeCall(cudaGetLastError());
            }

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    };

    template class pyrLK_caller<unsigned char,1>;
    template class pyrLK_caller<unsigned short,1>;
    template class pyrLK_caller<int,1>;
    template class pyrLK_caller<float,1>;

    template class pyrLK_caller<unsigned char, 3>;
    template class pyrLK_caller<unsigned short, 3>;
    template class pyrLK_caller<int, 3>;
    template class pyrLK_caller<float, 3>;

    template class pyrLK_caller<unsigned char, 4>;
    template class pyrLK_caller<unsigned short, 4>;
    template class pyrLK_caller<int, 4>;
    template class pyrLK_caller<float, 4>;
}

#endif /* CUDA_DISABLER */
