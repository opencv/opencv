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
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

////////////////////////////////////////////////////////////
// centeredGradient

namespace tvl1flow
{
    __global__ void centeredGradientKernel(const PtrStepSzf src, PtrStepf dx, PtrStepf dy)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src.cols || y >= src.rows)
            return;

        dx(y, x) = 0.5f * (src(y, ::min(x + 1, src.cols - 1)) - src(y, ::max(x - 1, 0)));
        dy(y, x) = 0.5f * (src(::min(y + 1, src.rows - 1), x) - src(::max(y - 1, 0), x));
    }

    void centeredGradient(PtrStepSzf src, PtrStepSzf dx, PtrStepSzf dy, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        centeredGradientKernel<<<grid, block, 0, stream>>>(src, dx, dy);
        cudaSafeCall( cudaGetLastError() );

        if (!stream)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

////////////////////////////////////////////////////////////
// warpBackward

namespace tvl1flow
{
    static __device__ __forceinline__ float bicubicCoeff(float x_)
    {
        float x = fabsf(x_);
        if (x <= 1.0f)
        {
            return x * x * (1.5f * x - 2.5f) + 1.0f;
        }
        else if (x < 2.0f)
        {
            return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
        }
        else
        {
            return 0.0f;
        }
    }

    struct SrcTex
    {
        virtual ~SrcTex() {}

        __device__ __forceinline__ virtual float I1(float x, float y) const = 0;
        __device__ __forceinline__ virtual float I1x(float x, float y) const = 0;
        __device__ __forceinline__ virtual float I1y(float x, float y) const = 0;
    };

    texture<float, cudaTextureType2D, cudaReadModeElementType> tex_I1 (false, cudaFilterModePoint, cudaAddressModeClamp);
    texture<float, cudaTextureType2D, cudaReadModeElementType> tex_I1x(false, cudaFilterModePoint, cudaAddressModeClamp);
    texture<float, cudaTextureType2D, cudaReadModeElementType> tex_I1y(false, cudaFilterModePoint, cudaAddressModeClamp);
    struct SrcTexRef : SrcTex
    {
        __device__ __forceinline__ float I1(float x, float y) const CV_OVERRIDE
        {
            return tex2D(tex_I1, x, y);
        }
        __device__ __forceinline__ float I1x(float x, float y) const CV_OVERRIDE
        {
            return tex2D(tex_I1x, x, y);
        }
        __device__ __forceinline__ float I1y(float x, float y) const CV_OVERRIDE
        {
            return tex2D(tex_I1y, x, y);
        }
    };

    struct SrcTexObj : SrcTex
    {
        __host__ SrcTexObj(cudaTextureObject_t tex_obj_I1_, cudaTextureObject_t tex_obj_I1x_, cudaTextureObject_t tex_obj_I1y_)
            : tex_obj_I1(tex_obj_I1_), tex_obj_I1x(tex_obj_I1x_), tex_obj_I1y(tex_obj_I1y_) {}

        __device__ __forceinline__ float I1(float x, float y) const CV_OVERRIDE
        {
            return tex2D<float>(tex_obj_I1, x, y);
        }
        __device__ __forceinline__ float I1x(float x, float y) const CV_OVERRIDE
        {
            return tex2D<float>(tex_obj_I1x, x, y);
        }
        __device__ __forceinline__ float I1y(float x, float y) const CV_OVERRIDE
        {
            return tex2D<float>(tex_obj_I1y, x, y);
        }

        cudaTextureObject_t tex_obj_I1;
        cudaTextureObject_t tex_obj_I1x;
        cudaTextureObject_t tex_obj_I1y;
    };

    template <
        typename T
    >
    __global__ void warpBackwardKernel(
        const PtrStepSzf I0, const T src, const PtrStepf u1, const PtrStepf u2,
        PtrStepf I1w, PtrStepf I1wx, PtrStepf I1wy, PtrStepf grad, PtrStepf rho)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= I0.cols || y >= I0.rows)
            return;

        const float u1Val = u1(y, x);
        const float u2Val = u2(y, x);

        const float wx = x + u1Val;
        const float wy = y + u2Val;

        const int xmin = ::ceilf(wx - 2.0f);
        const int xmax = ::floorf(wx + 2.0f);

        const int ymin = ::ceilf(wy - 2.0f);
        const int ymax = ::floorf(wy + 2.0f);

        float sum  = 0.0f;
        float sumx = 0.0f;
        float sumy = 0.0f;
        float wsum = 0.0f;

        for (int cy = ymin; cy <= ymax; ++cy)
        {
            for (int cx = xmin; cx <= xmax; ++cx)
            {
                const float w = bicubicCoeff(wx - cx) * bicubicCoeff(wy - cy);

                sum  += w * src.I1(cx, cy);
                sumx += w * src.I1x(cx, cy);
                sumy += w * src.I1y(cx, cy);

                wsum += w;
            }
        }

        const float coeff = 1.0f / wsum;

        const float I1wVal  = sum  * coeff;
        const float I1wxVal = sumx * coeff;
        const float I1wyVal = sumy * coeff;

        I1w(y, x)  = I1wVal;
        I1wx(y, x) = I1wxVal;
        I1wy(y, x) = I1wyVal;

        const float Ix2 = I1wxVal * I1wxVal;
        const float Iy2 = I1wyVal * I1wyVal;

        // store the |Grad(I1)|^2
        grad(y, x) = Ix2 + Iy2;

        // compute the constant part of the rho function
        const float I0Val = I0(y, x);
        rho(y, x) = I1wVal - I1wxVal * u1Val - I1wyVal * u2Val - I0Val;
    }

    void warpBackward(PtrStepSzf I0, PtrStepSzf I1, PtrStepSzf I1x, PtrStepSzf I1y,
                      PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf I1w, PtrStepSzf I1wx,
                      PtrStepSzf I1wy, PtrStepSzf grad, PtrStepSzf rho,
                      cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(I0.cols, block.x), divUp(I0.rows, block.y));

        bool cc30 = deviceSupports(FEATURE_SET_COMPUTE_30);

        if (cc30)
        {
            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.addressMode[2] = cudaAddressModeClamp;

            cudaTextureObject_t texObj_I1 = 0, texObj_I1x = 0, texObj_I1y = 0;

            createTextureObjectPitch2D(&texObj_I1, I1, texDesc);
            createTextureObjectPitch2D(&texObj_I1x, I1x, texDesc);
            createTextureObjectPitch2D(&texObj_I1y, I1y, texDesc);

            warpBackwardKernel << <grid, block, 0, stream >> > (I0, SrcTexObj(texObj_I1, texObj_I1x, texObj_I1y), u1, u2, I1w, I1wx, I1wy, grad, rho);
            cudaSafeCall(cudaGetLastError());

            if (!stream)
                cudaSafeCall(cudaDeviceSynchronize());
            else
                cudaSafeCall(cudaStreamSynchronize(stream));

            cudaSafeCall(cudaDestroyTextureObject(texObj_I1));
            cudaSafeCall(cudaDestroyTextureObject(texObj_I1x));
            cudaSafeCall(cudaDestroyTextureObject(texObj_I1y));
        }
        else
        {
            bindTexture(&tex_I1, I1);
            bindTexture(&tex_I1x, I1x);
            bindTexture(&tex_I1y, I1y);

            warpBackwardKernel << <grid, block, 0, stream >> > (I0, SrcTexRef(), u1, u2, I1w, I1wx, I1wy, grad, rho);
            cudaSafeCall(cudaGetLastError());

            if (!stream)
                cudaSafeCall(cudaDeviceSynchronize());
        }
    }
}

////////////////////////////////////////////////////////////
// estimateU

namespace tvl1flow
{
    __device__ float divergence(const PtrStepf& v1, const PtrStepf& v2, int y, int x)
    {
        if (x > 0 && y > 0)
        {
            const float v1x = v1(y, x) - v1(y, x - 1);
            const float v2y = v2(y, x) - v2(y - 1, x);
            return v1x + v2y;
        }
        else
        {
            if (y > 0)
                return v1(y, 0) + v2(y, 0) - v2(y - 1, 0);
            else
            {
                if (x > 0)
                    return v1(0, x) - v1(0, x - 1) + v2(0, x);
                else
                    return v1(0, 0) + v2(0, 0);
            }
        }
    }

    __global__ void estimateUKernel(const PtrStepSzf I1wx, const PtrStepf I1wy,
                              const PtrStepf grad, const PtrStepf rho_c,
                              const PtrStepf p11, const PtrStepf p12,
                              const PtrStepf p21, const PtrStepf p22,
                              const PtrStepf p31, const PtrStepf p32,
                              PtrStepf u1, PtrStepf u2, PtrStepf u3, PtrStepf error,
                              const float l_t, const float theta, const float gamma, const bool calcError)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= I1wx.cols || y >= I1wx.rows)
            return;

        const float I1wxVal = I1wx(y, x);
        const float I1wyVal = I1wy(y, x);
        const float gradVal = grad(y, x);
        const float u1OldVal = u1(y, x);
        const float u2OldVal = u2(y, x);
        const float u3OldVal = gamma ? u3(y, x) : 0;

        const float rho = rho_c(y, x) + (I1wxVal * u1OldVal + I1wyVal * u2OldVal + gamma * u3OldVal);

        // estimate the values of the variable (v1, v2) (thresholding operator TH)

        float d1 = 0.0f;
        float d2 = 0.0f;
        float d3 = 0.0f;

        if (rho < -l_t * gradVal)
        {
            d1 = l_t * I1wxVal;
            d2 = l_t * I1wyVal;
            if (gamma)
                d3 = l_t * gamma;
        }
        else if (rho > l_t * gradVal)
        {
            d1 = -l_t * I1wxVal;
            d2 = -l_t * I1wyVal;
            if (gamma)
                d3 = -l_t * gamma;
        }
        else if (gradVal > numeric_limits<float>::epsilon())
        {
            const float fi = -rho / gradVal;
            d1 = fi * I1wxVal;
            d2 = fi * I1wyVal;
            if (gamma)
                d3 = fi * gamma;
        }

        const float v1 = u1OldVal + d1;
        const float v2 = u2OldVal + d2;
        const float v3 = u3OldVal + d3;

        // compute the divergence of the dual variable (p1, p2)

        const float div_p1 = divergence(p11, p12, y, x);
        const float div_p2 = divergence(p21, p22, y, x);
        const float div_p3 = gamma ? divergence(p31, p32, y, x) : 0;

        // estimate the values of the optical flow (u1, u2)

        const float u1NewVal = v1 + theta * div_p1;
        const float u2NewVal = v2 + theta * div_p2;
        const float u3NewVal = gamma ? v3 + theta * div_p3 : 0;

        u1(y, x) = u1NewVal;
        u2(y, x) = u2NewVal;
        if (gamma)
            u3(y, x) = u3NewVal;

        if (calcError)
        {
            const float n1 = (u1OldVal - u1NewVal) * (u1OldVal - u1NewVal);
            const float n2 = (u2OldVal - u2NewVal) * (u2OldVal - u2NewVal);
            error(y, x) = n1 + n2;
        }
    }

    void estimateU(PtrStepSzf I1wx, PtrStepSzf I1wy,
                   PtrStepSzf grad, PtrStepSzf rho_c,
                   PtrStepSzf p11, PtrStepSzf p12, PtrStepSzf p21, PtrStepSzf p22, PtrStepSzf p31, PtrStepSzf p32,
                   PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf u3, PtrStepSzf error,
                   float l_t, float theta, float gamma, bool calcError,
                   cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(I1wx.cols, block.x), divUp(I1wx.rows, block.y));

        estimateUKernel<<<grid, block, 0, stream>>>(I1wx, I1wy, grad, rho_c, p11, p12, p21, p22, p31, p32, u1, u2, u3, error, l_t, theta, gamma, calcError);
        cudaSafeCall( cudaGetLastError() );

        if (!stream)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

////////////////////////////////////////////////////////////
// estimateDualVariables

namespace tvl1flow
{
    __global__ void estimateDualVariablesKernel(const PtrStepSzf u1, const PtrStepf u2, const PtrStepSzf u3,
                                                PtrStepf p11, PtrStepf p12, PtrStepf p21, PtrStepf p22, PtrStepf p31, PtrStepf p32, const float taut, const float gamma)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= u1.cols || y >= u1.rows)
            return;

        const float u1x = u1(y, ::min(x + 1, u1.cols - 1)) - u1(y, x);
        const float u1y = u1(::min(y + 1, u1.rows - 1), x) - u1(y, x);

        const float u2x = u2(y, ::min(x + 1, u1.cols - 1)) - u2(y, x);
        const float u2y = u2(::min(y + 1, u1.rows - 1), x) - u2(y, x);

        const float u3x = gamma ? u3(y, ::min(x + 1, u1.cols - 1)) - u3(y, x) : 0;
        const float u3y = gamma ? u3(::min(y + 1, u1.rows - 1), x) - u3(y, x) : 0;

        const float g1 = ::hypotf(u1x, u1y);
        const float g2 = ::hypotf(u2x, u2y);
        const float g3 = gamma ? ::hypotf(u3x, u3y) : 0;

        const float ng1 = 1.0f + taut * g1;
        const float ng2 = 1.0f + taut * g2;
        const float ng3 = gamma ? 1.0f + taut * g3 : 0;

        p11(y, x) = (p11(y, x) + taut * u1x) / ng1;
        p12(y, x) = (p12(y, x) + taut * u1y) / ng1;
        p21(y, x) = (p21(y, x) + taut * u2x) / ng2;
        p22(y, x) = (p22(y, x) + taut * u2y) / ng2;
        if (gamma)
        {
            p31(y, x) = (p31(y, x) + taut * u3x) / ng3;
            p32(y, x) = (p32(y, x) + taut * u3y) / ng3;
        }
    }

    void estimateDualVariables(PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf u3,
                               PtrStepSzf p11, PtrStepSzf p12, PtrStepSzf p21, PtrStepSzf p22, PtrStepSzf p31, PtrStepSzf p32,
                               float taut, float gamma,
                               cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(u1.cols, block.x), divUp(u1.rows, block.y));

        estimateDualVariablesKernel<<<grid, block, 0, stream>>>(u1, u2, u3, p11, p12, p21, p22, p31, p32, taut, gamma);
        cudaSafeCall( cudaGetLastError() );

        if (!stream)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

#endif // !defined CUDA_DISABLER
