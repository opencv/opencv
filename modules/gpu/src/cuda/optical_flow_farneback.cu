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
#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y

#define BORDER_SIZE 5
#define MAX_KSIZE_HALF 100

namespace cv { namespace gpu { namespace device { namespace optflow_farneback
{
    __constant__ float c_g[8];
    __constant__ float c_xg[8];
    __constant__ float c_xxg[8];
    __constant__ float c_ig11, c_ig03, c_ig33, c_ig55;


    template <int polyN>
    __global__ void polynomialExpansion(
            const int height, const int width, const PtrStepf src, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * (bdx - 2*polyN) + tx - polyN;

        if (y < height)
        {
            extern __shared__ float smem[];
            volatile float *row = smem + tx;
            int xWarped = ::min(::max(x, 0), width - 1);

            row[0] = src(y, xWarped) * c_g[0];
            row[bdx] = 0.f;
            row[2*bdx] = 0.f;

            for (int k = 1; k <= polyN; ++k)
            {
                float t0 = src(::max(y - k, 0), xWarped);
                float t1 = src(::min(y + k, height - 1), xWarped);

                row[0] += c_g[k] * (t0 + t1);
                row[bdx] += c_xg[k] * (t1 - t0);
                row[2*bdx] += c_xxg[k] * (t0 + t1);
            }

            __syncthreads();

            if (tx >= polyN && tx + polyN < bdx && x < width)
            {
                float b1 = c_g[0] * row[0];
                float b3 = c_g[0] * row[bdx];
                float b5 = c_g[0] * row[2*bdx];
                float b2 = 0, b4 = 0, b6 = 0;

                for (int k = 1; k <= polyN; ++k)
                {
                    b1 += (row[k] + row[-k]) * c_g[k];
                    b4 += (row[k] + row[-k]) * c_xxg[k];
                    b2 += (row[k] - row[-k]) * c_xg[k];
                    b3 += (row[k + bdx] + row[-k + bdx]) * c_g[k];
                    b6 += (row[k + bdx] - row[-k + bdx]) * c_xg[k];
                    b5 += (row[k + 2*bdx] + row[-k + 2*bdx]) * c_g[k];
                }

                dst(y, xWarped) = b3*c_ig11;
                dst(height + y, xWarped) = b2*c_ig11;
                dst(2*height + y, xWarped) = b1*c_ig03 + b5*c_ig33;
                dst(3*height + y, xWarped) = b1*c_ig03 + b4*c_ig33;
                dst(4*height + y, xWarped) = b6*c_ig55;
            }
        }
    }


    void setPolynomialExpansionConsts(
            int polyN, const float *g, const float *xg, const float *xxg,
            float ig11, float ig03, float ig33, float ig55)
    {
        cudaSafeCall(cudaMemcpyToSymbol(c_g, g, (polyN + 1) * sizeof(*g)));
        cudaSafeCall(cudaMemcpyToSymbol(c_xg, xg, (polyN + 1) * sizeof(*xg)));
        cudaSafeCall(cudaMemcpyToSymbol(c_xxg, xxg, (polyN + 1) * sizeof(*xxg)));
        cudaSafeCall(cudaMemcpyToSymbol(c_ig11, &ig11, sizeof(ig11)));
        cudaSafeCall(cudaMemcpyToSymbol(c_ig03, &ig03, sizeof(ig03)));
        cudaSafeCall(cudaMemcpyToSymbol(c_ig33, &ig33, sizeof(ig33)));
        cudaSafeCall(cudaMemcpyToSymbol(c_ig55, &ig55, sizeof(ig55)));
    }


    void polynomialExpansionGpu(const PtrStepSzf &src, int polyN, PtrStepSzf dst, cudaStream_t stream)
    {
        dim3 block(256);
        dim3 grid(divUp(src.cols, block.x - 2*polyN), src.rows);
        int smem = 3 * block.x * sizeof(float);

        if (polyN == 5)
            polynomialExpansion<5><<<grid, block, smem, stream>>>(src.rows, src.cols, src, dst);
        else if (polyN == 7)
            polynomialExpansion<7><<<grid, block, smem, stream>>>(src.rows, src.cols, src, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    __constant__ float c_border[BORDER_SIZE + 1];

    __global__ void updateMatrices(
            const int height, const int width, const PtrStepf flowx, const PtrStepf flowy,
            const PtrStepf R0, const PtrStepf R1, PtrStepf M)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        if (y < height && x < width)
        {
            float dx = flowx(y, x);
            float dy = flowy(y, x);
            float fx = x + dx;
            float fy = y + dy;

            int x1 = floorf(fx);
            int y1 = floorf(fy);
            fx -= x1; fy -= y1;

            float r2, r3, r4, r5, r6;

            if (x1 >= 0 && y1 >= 0 && x1 < width - 1 && y1 < height - 1)
            {
                float a00 = (1.f - fx) * (1.f - fy);
                float a01 = fx * (1.f - fy);
                float a10 = (1.f - fx) * fy;
                float a11 = fx * fy;

                r2 = a00 * R1(y1, x1) +
                     a01 * R1(y1, x1 + 1) +
                     a10 * R1(y1 + 1, x1) +
                     a11 * R1(y1 + 1, x1 + 1);

                r3 = a00 * R1(height + y1, x1) +
                     a01 * R1(height + y1, x1 + 1) +
                     a10 * R1(height + y1 + 1, x1) +
                     a11 * R1(height + y1 + 1, x1 + 1);

                r4 = a00 * R1(2*height + y1, x1) +
                     a01 * R1(2*height + y1, x1 + 1) +
                     a10 * R1(2*height + y1 + 1, x1) +
                     a11 * R1(2*height + y1 + 1, x1 + 1);

                r5 = a00 * R1(3*height + y1, x1) +
                     a01 * R1(3*height + y1, x1 + 1) +
                     a10 * R1(3*height + y1 + 1, x1) +
                     a11 * R1(3*height + y1 + 1, x1 + 1);

                r6 = a00 * R1(4*height + y1, x1) +
                     a01 * R1(4*height + y1, x1 + 1) +
                     a10 * R1(4*height + y1 + 1, x1) +
                     a11 * R1(4*height + y1 + 1, x1 + 1);

                r4 = (R0(2*height + y, x) + r4) * 0.5f;
                r5 = (R0(3*height + y, x) + r5) * 0.5f;
                r6 = (R0(4*height + y, x) + r6) * 0.25f;
            }
            else
            {
                r2 = r3 = 0.f;
                r4 = R0(2*height + y, x);
                r5 = R0(3*height + y, x);
                r6 = R0(4*height + y, x) * 0.5f;
            }

            r2 = (R0(y, x) - r2) * 0.5f;
            r3 = (R0(height + y, x) - r3) * 0.5f;

            r2 += r4*dy + r6*dx;
            r3 += r6*dy + r5*dx;

            float scale =
                    c_border[::min(x, BORDER_SIZE)] *
                    c_border[::min(y, BORDER_SIZE)] *
                    c_border[::min(width - x - 1, BORDER_SIZE)] *
                    c_border[::min(height - y - 1, BORDER_SIZE)];

            r2 *= scale; r3 *= scale; r4 *= scale;
            r5 *= scale; r6 *= scale;

            M(y, x) = r4*r4 + r6*r6;
            M(height + y, x) = (r4 + r5)*r6;
            M(2*height + y, x) = r5*r5 + r6*r6;
            M(3*height + y, x) = r4*r2 + r6*r3;
            M(4*height + y, x) = r6*r2 + r5*r3;
        }
    }


    void setUpdateMatricesConsts()
    {
        static const float border[BORDER_SIZE + 1] = {0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f, 1.f};
        cudaSafeCall(cudaMemcpyToSymbol(c_border, border, (BORDER_SIZE + 1) * sizeof(*border)));
    }


    void updateMatricesGpu(
            const PtrStepSzf flowx, const PtrStepSzf flowy, const PtrStepSzf R0, const PtrStepSzf R1,
            PtrStepSzf M, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(flowx.cols, block.x), divUp(flowx.rows, block.y));

        updateMatrices<<<grid, block, 0, stream>>>(flowx.rows, flowx.cols, flowx, flowy, R0, R1, M);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    __global__ void updateFlow(
            const int height, const int width, const PtrStepf M, PtrStepf flowx, PtrStepf flowy)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        if (y < height && x < width)
        {
            float g11 = M(y, x);
            float g12 = M(height + y, x);
            float g22 = M(2*height + y, x);
            float h1 = M(3*height + y, x);
            float h2 = M(4*height + y, x);

            float detInv = 1.f / (g11*g22 - g12*g12 + 1e-3f);

            flowx(y, x) = (g11*h2 - g12*h1) * detInv;
            flowy(y, x) = (g22*h1 - g12*h2) * detInv;
        }
    }


    void updateFlowGpu(const PtrStepSzf M, PtrStepSzf flowx, PtrStepSzf flowy, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(flowx.cols, block.x), divUp(flowx.rows, block.y));

        updateFlow<<<grid, block, 0, stream>>>(flowx.rows, flowx.cols, M, flowx, flowy);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    /*__global__ void boxFilter(
            const int height, const int width, const PtrStepf src,
            const int ksizeHalf, const float boxAreaInv, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];
        volatile float *row = smem + ty * (bdx + 2*ksizeHalf);

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = ::min(::max(xExt, 0), width - 1);

                row[i] = src(y, xExt);
                for (int j = 1; j <= ksizeHalf; ++j)
                    row[i] += src(::max(y - j, 0), xExt) + src(::min(y + j, height - 1), xExt);
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal passs
                row += tx + ksizeHalf;
                float res = row[0];
                for (int i = 1; i <= ksizeHalf; ++i)
                    res += row[-i] + row[i];
                dst(y, x) = res * boxAreaInv;
            }
        }
    }


    void boxFilterGpu(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        dim3 block(256);
        dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
        int smem = (block.x + 2*ksizeHalf) * block.y * sizeof(float);

        float boxAreaInv = 1.f / ((1 + 2*ksizeHalf) * (1 + 2*ksizeHalf));
        boxFilter<<<grid, block, smem, stream>>>(src.rows, src.cols, src, ksizeHalf, boxAreaInv, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }*/


    __global__ void boxFilter5(
            const int height, const int width, const PtrStepf src,
            const int ksizeHalf, const float boxAreaInv, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];

        const int smw = bdx + 2*ksizeHalf; // shared memory "width"
        volatile float *row = smem + 5 * ty * smw;

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = ::min(::max(xExt, 0), width - 1);

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    row[k*smw + i] = src(k*height + y, xExt);

                for (int j = 1; j <= ksizeHalf; ++j)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        row[k*smw + i] +=
                                src(k*height + ::max(y - j, 0), xExt) +
                                src(k*height + ::min(y + j, height - 1), xExt);
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal passs

                row += tx + ksizeHalf;
                float res[5];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    res[k] = row[k*smw];

                for (int i = 1; i <= ksizeHalf; ++i)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        res[k] += row[k*smw - i] + row[k*smw + i];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    dst(k*height + y, x) = res[k] * boxAreaInv;
            }
        }
    }


    void boxFilter5Gpu(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows / 5;
        int width = src.cols;

        dim3 block(256);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * 5 * block.y * sizeof(float);

        float boxAreaInv = 1.f / ((1 + 2*ksizeHalf) * (1 + 2*ksizeHalf));
        boxFilter5<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, boxAreaInv, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void boxFilter5Gpu_CC11(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows / 5;
        int width = src.cols;

        dim3 block(128);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * 5 * block.y * sizeof(float);

        float boxAreaInv = 1.f / ((1 + 2*ksizeHalf) * (1 + 2*ksizeHalf));
        boxFilter5<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, boxAreaInv, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    __constant__ float c_gKer[MAX_KSIZE_HALF + 1];

    template <typename Border>
    __global__ void gaussianBlur(
            const int height, const int width, const PtrStepf src, const int ksizeHalf,
            const Border b, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];
        volatile float *row = smem + ty * (bdx + 2*ksizeHalf);

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = b.idx_col(xExt);
                row[i] = src(y, xExt) * c_gKer[0];
                for (int j = 1; j <= ksizeHalf; ++j)
                    row[i] +=
                            (src(b.idx_row_low(y - j), xExt) +
                             src(b.idx_row_high(y + j), xExt)) * c_gKer[j];
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal pass
                row += tx + ksizeHalf;
                float res = row[0] * c_gKer[0];
                for (int i = 1; i <= ksizeHalf; ++i)
                    res += (row[-i] + row[i]) * c_gKer[i];
                dst(y, x) = res;
            }
        }
    }


    void setGaussianBlurKernel(const float *gKer, int ksizeHalf)
    {
        cudaSafeCall(cudaMemcpyToSymbol(c_gKer, gKer, (ksizeHalf + 1) * sizeof(*gKer)));
    }


    template <typename Border>
    void gaussianBlurCaller(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows;
        int width = src.cols;

        dim3 block(256);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * block.y * sizeof(float);
        Border b(height, width);

        gaussianBlur<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, b, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void gaussianBlurGpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            gaussianBlurCaller<BrdReflect101<float> >,
            gaussianBlurCaller<BrdReplicate<float> >,
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }


    template <typename Border>
    __global__ void gaussianBlur5(
            const int height, const int width, const PtrStepf src, const int ksizeHalf,
            const Border b, PtrStepf dst)
    {
        const int y = by * bdy + ty;
        const int x = bx * bdx + tx;

        extern __shared__ float smem[];

        const int smw = bdx + 2*ksizeHalf; // shared memory "width"
        volatile float *row = smem + 5 * ty * smw;

        if (y < height)
        {
            // Vertical pass
            for (int i = tx; i < bdx + 2*ksizeHalf; i += bdx)
            {
                int xExt = int(bx * bdx) + i - ksizeHalf;
                xExt = b.idx_col(xExt);

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    row[k*smw + i] = src(k*height + y, xExt) * c_gKer[0];

                for (int j = 1; j <= ksizeHalf; ++j)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        row[k*smw + i] +=
                                (src(k*height + b.idx_row_low(y - j), xExt) +
                                 src(k*height + b.idx_row_high(y + j), xExt)) * c_gKer[j];
            }

            if (x < width)
            {
                __syncthreads();

                // Horizontal pass

                row += tx + ksizeHalf;
                float res[5];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    res[k] = row[k*smw] * c_gKer[0];

                for (int i = 1; i <= ksizeHalf; ++i)
                    #pragma unroll
                    for (int k = 0; k < 5; ++k)
                        res[k] += (row[k*smw - i] + row[k*smw + i]) * c_gKer[i];

                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    dst(k*height + y, x) = res[k];
            }
        }
    }


    template <typename Border, int blockDimX>
    void gaussianBlur5Caller(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream)
    {
        int height = src.rows / 5;
        int width = src.cols;

        dim3 block(blockDimX);
        dim3 grid(divUp(width, block.x), divUp(height, block.y));
        int smem = (block.x + 2*ksizeHalf) * 5 * block.y * sizeof(float);
        Border b(height, width);

        gaussianBlur5<<<grid, block, smem, stream>>>(height, width, src, ksizeHalf, b, dst);

        cudaSafeCall(cudaGetLastError());

        if (stream == 0)
            cudaSafeCall(cudaDeviceSynchronize());
    }


    void gaussianBlur5Gpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            gaussianBlur5Caller<BrdReflect101<float>,256>,
            gaussianBlur5Caller<BrdReplicate<float>,256>,
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }

    void gaussianBlur5Gpu_CC11(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderMode, cudaStream_t stream)
    {
        typedef void (*caller_t)(const PtrStepSzf, int, PtrStepSzf, cudaStream_t);

        static const caller_t callers[] =
        {
            gaussianBlur5Caller<BrdReflect101<float>,128>,
            gaussianBlur5Caller<BrdReplicate<float>,128>,
        };

        callers[borderMode](src, ksizeHalf, dst, stream);
    }

}}}} // namespace cv { namespace gpu { namespace device { namespace optflow_farneback


#endif /* CUDA_DISABLER */
