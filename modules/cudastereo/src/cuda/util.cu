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
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/reduce.hpp"

namespace cv { namespace cuda { namespace device
{
    /////////////////////////////////// reprojectImageTo3D ///////////////////////////////////////////////

    __constant__ float cq[16];

    template <typename T, typename D>
    __global__ void reprojectImageTo3D(const PtrStepSz<T> disp, PtrStep<D> xyz)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y >= disp.rows || x >= disp.cols)
            return;

        const float qx = x * cq[ 0] + y * cq[ 1] + cq[ 3];
        const float qy = x * cq[ 4] + y * cq[ 5] + cq[ 7];
        const float qz = x * cq[ 8] + y * cq[ 9] + cq[11];
        const float qw = x * cq[12] + y * cq[13] + cq[15];

        const T d = disp(y, x);

        const float iW = 1.f / (qw + cq[14] * d);

        D v = VecTraits<D>::all(1.0f);
        v.x = (qx + cq[2] * d) * iW;
        v.y = (qy + cq[6] * d) * iW;
        v.z = (qz + cq[10] * d) * iW;

        xyz(y, x) = v;
    }

    template <typename T, typename D>
    void reprojectImageTo3D_gpu(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(disp.cols, block.x), divUp(disp.rows, block.y));

        cudaSafeCall( cudaMemcpyToSymbol(cq, q, 16 * sizeof(float)) );

        reprojectImageTo3D<T, D><<<grid, block, 0, stream>>>((PtrStepSz<T>)disp, (PtrStepSz<D>)xyz);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void reprojectImageTo3D_gpu<uchar, float3>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<uchar, float4>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<short, float3>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<short, float4>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<int, float3>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<int, float4>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<float, float3>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);
    template void reprojectImageTo3D_gpu<float, float4>(const PtrStepSzb disp, PtrStepSzb xyz, const float* q, cudaStream_t stream);

    /////////////////////////////////// drawColorDisp ///////////////////////////////////////////////

    template <typename T>
    __device__ unsigned int cvtPixel(T d, int ndisp, float S = 1, float V = 1)
    {
        unsigned int H = ((ndisp-d) * 240)/ndisp;

        unsigned int hi = (H/60) % 6;
        float f = H/60.f - H/60;
        float p = V * (1 - S);
        float q = V * (1 - f * S);
        float t = V * (1 - (1 - f) * S);

        float3 res;

        if (hi == 0) //R = V,	G = t,	B = p
        {
            res.x = p;
            res.y = t;
            res.z = V;
        }

        if (hi == 1) // R = q,	G = V,	B = p
        {
            res.x = p;
            res.y = V;
            res.z = q;
        }

        if (hi == 2) // R = p,	G = V,	B = t
        {
            res.x = t;
            res.y = V;
            res.z = p;
        }

        if (hi == 3) // R = p,	G = q,	B = V
        {
            res.x = V;
            res.y = q;
            res.z = p;
        }

        if (hi == 4) // R = t,	G = p,	B = V
        {
            res.x = V;
            res.y = p;
            res.z = t;
        }

        if (hi == 5) // R = V,	G = p,	B = q
        {
            res.x = q;
            res.y = p;
            res.z = V;
        }
        const unsigned int b = (unsigned int)(::max(0.f, ::min(res.x, 1.f)) * 255.f);
        const unsigned int g = (unsigned int)(::max(0.f, ::min(res.y, 1.f)) * 255.f);
        const unsigned int r = (unsigned int)(::max(0.f, ::min(res.z, 1.f)) * 255.f);
        const unsigned int a = 255U;

        return (a << 24) + (r << 16) + (g << 8) + b;
    }

    __global__ void drawColorDisp(uchar* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height)
        {
            uchar4 d4 = *(uchar4*)(disp + y * disp_step + x);

            uint4 res;
            res.x = cvtPixel(d4.x, ndisp);
            res.y = cvtPixel(d4.y, ndisp);
            res.z = cvtPixel(d4.z, ndisp);
            res.w = cvtPixel(d4.w, ndisp);

            uint4* line = (uint4*)(out_image + y * out_step);
            line[x >> 2] = res;
        }
    }

    __global__ void drawColorDisp(short* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height)
        {
            short2 d2 = *(short2*)(disp + y * disp_step + x);

            uint2 res;
            res.x = cvtPixel(d2.x, ndisp);
            res.y = cvtPixel(d2.y, ndisp);

            uint2* line = (uint2*)(out_image + y * out_step);
            line[x >> 1] = res;
        }
    }

    __global__ void drawColorDisp(int* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height)
        {
            uint *line = (uint*)(out_image + y * out_step);
            line[x] = cvtPixel(disp[y*disp_step + x], ndisp);
        }
    }

    __global__ void drawColorDisp(float* disp, size_t disp_step, uchar* out_image, size_t out_step, int width, int height, int ndisp)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height)
        {
            uint *line = (uint*)(out_image + y * out_step);
            line[x] = cvtPixel(disp[y*disp_step + x], ndisp);
        }
    }

    void drawColorDisp_gpu(const PtrStepSzb& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream)
    {
        dim3 threads(16, 16, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(src.cols, threads.x << 2);
        grid.y = divUp(src.rows, threads.y);

        drawColorDisp<<<grid, threads, 0, stream>>>(src.data, src.step, dst.data, dst.step, src.cols, src.rows, ndisp);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    void drawColorDisp_gpu(const PtrStepSz<short>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(src.cols, threads.x << 1);
        grid.y = divUp(src.rows, threads.y);

        drawColorDisp<<<grid, threads, 0, stream>>>(src.data, src.step / sizeof(short), dst.data, dst.step, src.cols, src.rows, ndisp);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    void drawColorDisp_gpu(const PtrStepSz<int>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(src.cols, threads.x);
        grid.y = divUp(src.rows, threads.y);

        drawColorDisp<<<grid, threads, 0, stream>>>(src.data, src.step / sizeof(int), dst.data, dst.step, src.cols, src.rows, ndisp);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    void drawColorDisp_gpu(const PtrStepSz<float>& src, const PtrStepSzb& dst, int ndisp, const cudaStream_t& stream)
    {
        dim3 threads(32, 8, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp(src.cols, threads.x);
        grid.y = divUp(src.rows, threads.y);

        drawColorDisp<<<grid, threads, 0, stream>>>(src.data, src.step / sizeof(float), dst.data, dst.step, src.cols, src.rows, ndisp);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
