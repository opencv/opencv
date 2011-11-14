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
#include "opencv2/gpu/device/vec_math.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace match_template 
    {
        __device__ __forceinline__ float sum(float v) { return v; }
        __device__ __forceinline__ float sum(float2 v) { return v.x + v.y; }
        __device__ __forceinline__ float sum(float3 v) { return v.x + v.y + v.z; }
        __device__ __forceinline__ float sum(float4 v) { return v.x + v.y + v.z + v.w; }

        __device__ __forceinline__ float first(float v) { return v; }
        __device__ __forceinline__ float first(float2 v) { return v.x; }
        __device__ __forceinline__ float first(float3 v) { return v.x; }
        __device__ __forceinline__ float first(float4 v) { return v.x; }

        __device__ __forceinline__ float mul(float a, float b) { return a * b; }
        __device__ __forceinline__ float2 mul(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
        __device__ __forceinline__ float3 mul(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
        __device__ __forceinline__ float4 mul(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

        __device__ __forceinline__ float mul(uchar a, uchar b) { return a * b; }
        __device__ __forceinline__ float2 mul(uchar2 a, uchar2 b) { return make_float2(a.x * b.x, a.y * b.y); }
        __device__ __forceinline__ float3 mul(uchar3 a, uchar3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
        __device__ __forceinline__ float4 mul(uchar4 a, uchar4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

        __device__ __forceinline__ float sub(float a, float b) { return a - b; }
        __device__ __forceinline__ float2 sub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
        __device__ __forceinline__ float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
        __device__ __forceinline__ float4 sub(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }

        __device__ __forceinline__ float sub(uchar a, uchar b) { return a - b; }
        __device__ __forceinline__ float2 sub(uchar2 a, uchar2 b) { return make_float2(a.x - b.x, a.y - b.y); }
        __device__ __forceinline__ float3 sub(uchar3 a, uchar3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
        __device__ __forceinline__ float4 sub(uchar4 a, uchar4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }

        //////////////////////////////////////////////////////////////////////
        // Naive_CCORR

        template <typename T, int cn> 
        __global__ void matchTemplateNaiveKernel_CCORR(int w, int h, const PtrStepb image, const PtrStepb templ, DevMem2Df result)
        {
            typedef typename TypeVec<T, cn>::vec_type Type;
            typedef typename TypeVec<float, cn>::vec_type Typef;

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                Typef res = VecTraits<Typef>::all(0);

                for (int i = 0; i < h; ++i)
                {
                    const Type* image_ptr = (const Type*)image.ptr(y + i);
                    const Type* templ_ptr = (const Type*)templ.ptr(i);
                    for (int j = 0; j < w; ++j)
                        res = res + mul(image_ptr[x + j], templ_ptr[j]);
                }

                result.ptr(y)[x] = sum(res);
            }
        }

        template <typename T, int cn>
        void matchTemplateNaive_CCORR(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, cudaStream_t stream)
        {
            const dim3 threads(32, 8);
            const dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplateNaiveKernel_CCORR<T, cn><<<grid, threads, 0, stream>>>(templ.cols, templ.rows, image, templ, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void matchTemplateNaive_CCORR_32F(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream)
        {
            typedef void (*caller_t)(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                0, matchTemplateNaive_CCORR<float, 1>, matchTemplateNaive_CCORR<float, 2>, matchTemplateNaive_CCORR<float, 3>, matchTemplateNaive_CCORR<float, 4>
            };

            callers[cn](image, templ, result, stream);
        }


        void matchTemplateNaive_CCORR_8U(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream)
        {
            typedef void (*caller_t)(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                0, matchTemplateNaive_CCORR<uchar, 1>, matchTemplateNaive_CCORR<uchar, 2>, matchTemplateNaive_CCORR<uchar, 3>, matchTemplateNaive_CCORR<uchar, 4>
            };

            callers[cn](image, templ, result, stream);
        }

        //////////////////////////////////////////////////////////////////////
        // Naive_SQDIFF

        template <typename T, int cn>
        __global__ void matchTemplateNaiveKernel_SQDIFF(int w, int h, const PtrStepb image, const PtrStepb templ, DevMem2Df result)
        {
            typedef typename TypeVec<T, cn>::vec_type Type;
            typedef typename TypeVec<float, cn>::vec_type Typef;

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                Typef res = VecTraits<Typef>::all(0);
                Typef delta;

                for (int i = 0; i < h; ++i)
                {
                    const Type* image_ptr = (const Type*)image.ptr(y + i);
                    const Type* templ_ptr = (const Type*)templ.ptr(i);
                    for (int j = 0; j < w; ++j)
                    {
                        delta = sub(image_ptr[x + j], templ_ptr[j]);
                        res = res + delta * delta;
                    }
                }

                result.ptr(y)[x] = sum(res);
            }
        }

        template <typename T, int cn>
        void matchTemplateNaive_SQDIFF(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, cudaStream_t stream)
        {
            const dim3 threads(32, 8);
            const dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplateNaiveKernel_SQDIFF<T, cn><<<grid, threads, 0, stream>>>(templ.cols, templ.rows, image, templ, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void matchTemplateNaive_SQDIFF_32F(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream)
        {
            typedef void (*caller_t)(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                0, matchTemplateNaive_SQDIFF<float, 1>, matchTemplateNaive_SQDIFF<float, 2>, matchTemplateNaive_SQDIFF<float, 3>, matchTemplateNaive_SQDIFF<float, 4>
            };

            callers[cn](image, templ, result, stream);
        }

        void matchTemplateNaive_SQDIFF_8U(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, int cn, cudaStream_t stream)
        {
            typedef void (*caller_t)(const DevMem2Db image, const DevMem2Db templ, DevMem2Df result, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                0, matchTemplateNaive_SQDIFF<uchar, 1>, matchTemplateNaive_SQDIFF<uchar, 2>, matchTemplateNaive_SQDIFF<uchar, 3>, matchTemplateNaive_SQDIFF<uchar, 4>
            };

            callers[cn](image, templ, result, stream);
        }

        //////////////////////////////////////////////////////////////////////
        // Prepared_SQDIFF

        template <int cn>
        __global__ void matchTemplatePreparedKernel_SQDIFF_8U(int w, int h, const PtrStep<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sqsum_ = (float)(
                        (image_sqsum.ptr(y + h)[(x + w) * cn] - image_sqsum.ptr(y)[(x + w) * cn]) -
                        (image_sqsum.ptr(y + h)[x * cn] - image_sqsum.ptr(y)[x * cn]));
                float ccorr = result.ptr(y)[x];
                result.ptr(y)[x] = image_sqsum_ - 2.f * ccorr + templ_sqsum;
            }
        }

        template <int cn>
        void matchTemplatePrepared_SQDIFF_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result, cudaStream_t stream)
        {
            const dim3 threads(32, 8);
            const dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplatePreparedKernel_SQDIFF_8U<cn><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void matchTemplatePrepared_SQDIFF_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result, int cn, 
                                             cudaStream_t stream)
        {
            typedef void (*caller_t)(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result, cudaStream_t stream);

            static const caller_t callers[] = 
            {
                0, matchTemplatePrepared_SQDIFF_8U<1>, matchTemplatePrepared_SQDIFF_8U<2>, matchTemplatePrepared_SQDIFF_8U<3>, matchTemplatePrepared_SQDIFF_8U<4>
            };

            callers[cn](w, h, image_sqsum, templ_sqsum, result, stream);
        }

        //////////////////////////////////////////////////////////////////////
        // Prepared_SQDIFF_NORMED

        // normAcc* are accurate normalization routines which make GPU matchTemplate
        // consistent with CPU one

        __device__ float normAcc(float num, float denum)
        {
            if (::fabs(num) < denum)
                return num / denum;
            if (::fabs(num) < denum * 1.125f)
                return num > 0 ? 1 : -1;
            return 0;
        }


        __device__ float normAcc_SQDIFF(float num, float denum)
        {
            if (::fabs(num) < denum)
                return num / denum;
            if (::fabs(num) < denum * 1.125f)
                return num > 0 ? 1 : -1;
            return 1;
        }


        template <int cn>
        __global__ void matchTemplatePreparedKernel_SQDIFF_NORMED_8U(int w, int h, const PtrStep<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sqsum_ = (float)(
                        (image_sqsum.ptr(y + h)[(x + w) * cn] - image_sqsum.ptr(y)[(x + w) * cn]) -
                        (image_sqsum.ptr(y + h)[x * cn] - image_sqsum.ptr(y)[x * cn]));
                float ccorr = result.ptr(y)[x];
                result.ptr(y)[x] = normAcc_SQDIFF(image_sqsum_ - 2.f * ccorr + templ_sqsum,
                                                  sqrtf(image_sqsum_ * templ_sqsum));
            }
        }

        template <int cn>
        void matchTemplatePrepared_SQDIFF_NORMED_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, 
                                                    DevMem2Df result, cudaStream_t stream)
        {
            const dim3 threads(32, 8);
            const dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplatePreparedKernel_SQDIFF_NORMED_8U<cn><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }


        void matchTemplatePrepared_SQDIFF_NORMED_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, 
                                                    DevMem2Df result, int cn, cudaStream_t stream)
        {
            typedef void (*caller_t)(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, unsigned int templ_sqsum, DevMem2Df result, cudaStream_t stream);
            static const caller_t callers[] = 
            {
                0, matchTemplatePrepared_SQDIFF_NORMED_8U<1>, matchTemplatePrepared_SQDIFF_NORMED_8U<2>, matchTemplatePrepared_SQDIFF_NORMED_8U<3>, matchTemplatePrepared_SQDIFF_NORMED_8U<4>
            };

            callers[cn](w, h, image_sqsum, templ_sqsum, result, stream);
        }

        //////////////////////////////////////////////////////////////////////
        // Prepared_CCOFF

        __global__ void matchTemplatePreparedKernel_CCOFF_8U(int w, int h, float templ_sum_scale, const PtrStep<unsigned int> image_sum, DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_ = (float)(
                        (image_sum.ptr(y + h)[x + w] - image_sum.ptr(y)[x + w]) -
                        (image_sum.ptr(y + h)[x] - image_sum.ptr(y)[x]));
                float ccorr = result.ptr(y)[x];
                result.ptr(y)[x] = ccorr - image_sum_ * templ_sum_scale;
            }
        }

        void matchTemplatePrepared_CCOFF_8U(int w, int h, const DevMem2D_<unsigned int> image_sum, unsigned int templ_sum, DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplatePreparedKernel_CCOFF_8U<<<grid, threads, 0, stream>>>(w, h, (float)templ_sum / (w * h), image_sum, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }



        __global__ void matchTemplatePreparedKernel_CCOFF_8UC2(
                int w, int h, float templ_sum_scale_r, float templ_sum_scale_g,
                const PtrStep<unsigned int> image_sum_r,
                const PtrStep<unsigned int> image_sum_g,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_r_ = (float)(
                        (image_sum_r.ptr(y + h)[x + w] - image_sum_r.ptr(y)[x + w]) -
                        (image_sum_r.ptr(y + h)[x] - image_sum_r.ptr(y)[x]));
                float image_sum_g_ = (float)(
                        (image_sum_g.ptr(y + h)[x + w] - image_sum_g.ptr(y)[x + w]) -
                        (image_sum_g.ptr(y + h)[x] - image_sum_g.ptr(y)[x]));
                float ccorr = result.ptr(y)[x];
                result.ptr(y)[x] = ccorr - image_sum_r_ * templ_sum_scale_r 
                                         - image_sum_g_ * templ_sum_scale_g;
            }
        }

        void matchTemplatePrepared_CCOFF_8UC2(
                int w, int h, 
                const DevMem2D_<unsigned int> image_sum_r, 
                const DevMem2D_<unsigned int> image_sum_g,
                unsigned int templ_sum_r, unsigned int templ_sum_g, 
                DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplatePreparedKernel_CCOFF_8UC2<<<grid, threads, 0, stream>>>(
                    w, h, (float)templ_sum_r / (w * h), (float)templ_sum_g / (w * h),
                    image_sum_r, image_sum_g, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }



        __global__ void matchTemplatePreparedKernel_CCOFF_8UC3(
                int w, int h, 
                float templ_sum_scale_r,
                float templ_sum_scale_g,
                float templ_sum_scale_b,
                const PtrStep<unsigned int> image_sum_r,
                const PtrStep<unsigned int> image_sum_g,
                const PtrStep<unsigned int> image_sum_b,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_r_ = (float)(
                        (image_sum_r.ptr(y + h)[x + w] - image_sum_r.ptr(y)[x + w]) -
                        (image_sum_r.ptr(y + h)[x] - image_sum_r.ptr(y)[x]));
                float image_sum_g_ = (float)(
                        (image_sum_g.ptr(y + h)[x + w] - image_sum_g.ptr(y)[x + w]) -
                        (image_sum_g.ptr(y + h)[x] - image_sum_g.ptr(y)[x]));
                float image_sum_b_ = (float)(
                        (image_sum_b.ptr(y + h)[x + w] - image_sum_b.ptr(y)[x + w]) -
                        (image_sum_b.ptr(y + h)[x] - image_sum_b.ptr(y)[x]));
                float ccorr = result.ptr(y)[x];
                result.ptr(y)[x] = ccorr - image_sum_r_ * templ_sum_scale_r
                                         - image_sum_g_ * templ_sum_scale_g
                                         - image_sum_b_ * templ_sum_scale_b;
            }
        }

        void matchTemplatePrepared_CCOFF_8UC3(
                int w, int h, 
                const DevMem2D_<unsigned int> image_sum_r, 
                const DevMem2D_<unsigned int> image_sum_g,
                const DevMem2D_<unsigned int> image_sum_b,
                unsigned int templ_sum_r, 
                unsigned int templ_sum_g, 
                unsigned int templ_sum_b, 
                DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplatePreparedKernel_CCOFF_8UC3<<<grid, threads, 0, stream>>>(
                    w, h, 
                    (float)templ_sum_r / (w * h),
                    (float)templ_sum_g / (w * h),
                    (float)templ_sum_b / (w * h),
                    image_sum_r, image_sum_g, image_sum_b, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }



        __global__ void matchTemplatePreparedKernel_CCOFF_8UC4(
                int w, int h, 
                float templ_sum_scale_r, 
                float templ_sum_scale_g,
                float templ_sum_scale_b,
                float templ_sum_scale_a,
                const PtrStep<unsigned int> image_sum_r,
                const PtrStep<unsigned int> image_sum_g,
                const PtrStep<unsigned int> image_sum_b,
                const PtrStep<unsigned int> image_sum_a,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_r_ = (float)(
                        (image_sum_r.ptr(y + h)[x + w] - image_sum_r.ptr(y)[x + w]) -
                        (image_sum_r.ptr(y + h)[x] - image_sum_r.ptr(y)[x]));
                float image_sum_g_ = (float)(
                        (image_sum_g.ptr(y + h)[x + w] - image_sum_g.ptr(y)[x + w]) -
                        (image_sum_g.ptr(y + h)[x] - image_sum_g.ptr(y)[x]));
                float image_sum_b_ = (float)(
                        (image_sum_b.ptr(y + h)[x + w] - image_sum_b.ptr(y)[x + w]) -
                        (image_sum_b.ptr(y + h)[x] - image_sum_b.ptr(y)[x]));
                float image_sum_a_ = (float)(
                        (image_sum_a.ptr(y + h)[x + w] - image_sum_a.ptr(y)[x + w]) -
                        (image_sum_a.ptr(y + h)[x] - image_sum_a.ptr(y)[x]));
                float ccorr = result.ptr(y)[x];
                result.ptr(y)[x] = ccorr - image_sum_r_ * templ_sum_scale_r 
                                         - image_sum_g_ * templ_sum_scale_g
                                         - image_sum_b_ * templ_sum_scale_b
                                         - image_sum_a_ * templ_sum_scale_a;
            }
        }

        void matchTemplatePrepared_CCOFF_8UC4(
                int w, int h, 
                const DevMem2D_<unsigned int> image_sum_r, 
                const DevMem2D_<unsigned int> image_sum_g,
                const DevMem2D_<unsigned int> image_sum_b,
                const DevMem2D_<unsigned int> image_sum_a,
                unsigned int templ_sum_r, 
                unsigned int templ_sum_g, 
                unsigned int templ_sum_b, 
                unsigned int templ_sum_a, 
                DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            matchTemplatePreparedKernel_CCOFF_8UC4<<<grid, threads, 0, stream>>>(
                    w, h, 
                    (float)templ_sum_r / (w * h), 
                    (float)templ_sum_g / (w * h), 
                    (float)templ_sum_b / (w * h),
                    (float)templ_sum_a / (w * h),
                    image_sum_r, image_sum_g, image_sum_b, image_sum_a,
                    result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        //////////////////////////////////////////////////////////////////////
        // Prepared_CCOFF_NORMED

        __global__ void matchTemplatePreparedKernel_CCOFF_NORMED_8U(
                int w, int h, float weight, 
                float templ_sum_scale, float templ_sqsum_scale,
                const PtrStep<unsigned int> image_sum, 
                const PtrStep<unsigned long long> image_sqsum,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float ccorr = result.ptr(y)[x];
                float image_sum_ = (float)(
                        (image_sum.ptr(y + h)[x + w] - image_sum.ptr(y)[x + w]) -
                        (image_sum.ptr(y + h)[x] - image_sum.ptr(y)[x]));
                float image_sqsum_ = (float)(
                        (image_sqsum.ptr(y + h)[x + w] - image_sqsum.ptr(y)[x + w]) -
                        (image_sqsum.ptr(y + h)[x] - image_sqsum.ptr(y)[x]));
                result.ptr(y)[x] = normAcc(ccorr - image_sum_ * templ_sum_scale,
                                           sqrtf(templ_sqsum_scale * (image_sqsum_ - weight * image_sum_ * image_sum_)));
            }
        }

        void matchTemplatePrepared_CCOFF_NORMED_8U(
                    int w, int h, const DevMem2D_<unsigned int> image_sum, 
                    const DevMem2D_<unsigned long long> image_sqsum,
                    unsigned int templ_sum, unsigned int templ_sqsum,
                    DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            float weight = 1.f / (w * h);
            float templ_sum_scale = templ_sum * weight;
            float templ_sqsum_scale = templ_sqsum - weight * templ_sum * templ_sum;

            matchTemplatePreparedKernel_CCOFF_NORMED_8U<<<grid, threads, 0, stream>>>(
                    w, h, weight, templ_sum_scale, templ_sqsum_scale, 
                    image_sum, image_sqsum, result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }



        __global__ void matchTemplatePreparedKernel_CCOFF_NORMED_8UC2(
                int w, int h, float weight, 
                float templ_sum_scale_r, float templ_sum_scale_g, 
                float templ_sqsum_scale,
                const PtrStep<unsigned int> image_sum_r, const PtrStep<unsigned long long> image_sqsum_r,
                const PtrStep<unsigned int> image_sum_g, const PtrStep<unsigned long long> image_sqsum_g,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_r_ = (float)(
                        (image_sum_r.ptr(y + h)[x + w] - image_sum_r.ptr(y)[x + w]) -
                        (image_sum_r.ptr(y + h)[x] - image_sum_r.ptr(y)[x]));
                float image_sqsum_r_ = (float)(
                        (image_sqsum_r.ptr(y + h)[x + w] - image_sqsum_r.ptr(y)[x + w]) -
                        (image_sqsum_r.ptr(y + h)[x] - image_sqsum_r.ptr(y)[x]));
                float image_sum_g_ = (float)(
                        (image_sum_g.ptr(y + h)[x + w] - image_sum_g.ptr(y)[x + w]) -
                        (image_sum_g.ptr(y + h)[x] - image_sum_g.ptr(y)[x]));
                float image_sqsum_g_ = (float)(
                        (image_sqsum_g.ptr(y + h)[x + w] - image_sqsum_g.ptr(y)[x + w]) -
                        (image_sqsum_g.ptr(y + h)[x] - image_sqsum_g.ptr(y)[x]));

                float num = result.ptr(y)[x] - image_sum_r_ * templ_sum_scale_r
                                             - image_sum_g_ * templ_sum_scale_g;
                float denum = sqrtf(templ_sqsum_scale * (image_sqsum_r_ - weight * image_sum_r_ * image_sum_r_
                                                         + image_sqsum_g_ - weight * image_sum_g_ * image_sum_g_));
                result.ptr(y)[x] = normAcc(num, denum);
            }
        }

        void matchTemplatePrepared_CCOFF_NORMED_8UC2(
                    int w, int h, 
                    const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
                    const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
                    unsigned int templ_sum_r, unsigned int templ_sqsum_r,
                    unsigned int templ_sum_g, unsigned int templ_sqsum_g,
                    DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            float weight = 1.f / (w * h);
            float templ_sum_scale_r = templ_sum_r * weight;
            float templ_sum_scale_g = templ_sum_g * weight;
            float templ_sqsum_scale = templ_sqsum_r - weight * templ_sum_r * templ_sum_r 
                                       + templ_sqsum_g - weight * templ_sum_g * templ_sum_g;

            matchTemplatePreparedKernel_CCOFF_NORMED_8UC2<<<grid, threads, 0, stream>>>(
                    w, h, weight, 
                    templ_sum_scale_r, templ_sum_scale_g,
                    templ_sqsum_scale,
                    image_sum_r, image_sqsum_r, 
                    image_sum_g, image_sqsum_g, 
                    result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }



        __global__ void matchTemplatePreparedKernel_CCOFF_NORMED_8UC3(
                int w, int h, float weight, 
                float templ_sum_scale_r, float templ_sum_scale_g, float templ_sum_scale_b, 
                float templ_sqsum_scale,
                const PtrStep<unsigned int> image_sum_r, const PtrStep<unsigned long long> image_sqsum_r,
                const PtrStep<unsigned int> image_sum_g, const PtrStep<unsigned long long> image_sqsum_g,
                const PtrStep<unsigned int> image_sum_b, const PtrStep<unsigned long long> image_sqsum_b,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_r_ = (float)(
                        (image_sum_r.ptr(y + h)[x + w] - image_sum_r.ptr(y)[x + w]) -
                        (image_sum_r.ptr(y + h)[x] - image_sum_r.ptr(y)[x]));
                float image_sqsum_r_ = (float)(
                        (image_sqsum_r.ptr(y + h)[x + w] - image_sqsum_r.ptr(y)[x + w]) -
                        (image_sqsum_r.ptr(y + h)[x] - image_sqsum_r.ptr(y)[x]));
                float image_sum_g_ = (float)(
                        (image_sum_g.ptr(y + h)[x + w] - image_sum_g.ptr(y)[x + w]) -
                        (image_sum_g.ptr(y + h)[x] - image_sum_g.ptr(y)[x]));
                float image_sqsum_g_ = (float)(
                        (image_sqsum_g.ptr(y + h)[x + w] - image_sqsum_g.ptr(y)[x + w]) -
                        (image_sqsum_g.ptr(y + h)[x] - image_sqsum_g.ptr(y)[x]));
                float image_sum_b_ = (float)(
                        (image_sum_b.ptr(y + h)[x + w] - image_sum_b.ptr(y)[x + w]) -
                        (image_sum_b.ptr(y + h)[x] - image_sum_b.ptr(y)[x]));
                float image_sqsum_b_ = (float)(
                        (image_sqsum_b.ptr(y + h)[x + w] - image_sqsum_b.ptr(y)[x + w]) -
                        (image_sqsum_b.ptr(y + h)[x] - image_sqsum_b.ptr(y)[x]));

                float num = result.ptr(y)[x] - image_sum_r_ * templ_sum_scale_r
                                             - image_sum_g_ * templ_sum_scale_g
                                             - image_sum_b_ * templ_sum_scale_b;
                float denum = sqrtf(templ_sqsum_scale * (image_sqsum_r_ - weight * image_sum_r_ * image_sum_r_
                                                         + image_sqsum_g_ - weight * image_sum_g_ * image_sum_g_
                                                         + image_sqsum_b_ - weight * image_sum_b_ * image_sum_b_));
                result.ptr(y)[x] = normAcc(num, denum);
            }
        }

        void matchTemplatePrepared_CCOFF_NORMED_8UC3(
                    int w, int h, 
                    const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
                    const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
                    const DevMem2D_<unsigned int> image_sum_b, const DevMem2D_<unsigned long long> image_sqsum_b,
                    unsigned int templ_sum_r, unsigned int templ_sqsum_r,
                    unsigned int templ_sum_g, unsigned int templ_sqsum_g,
                    unsigned int templ_sum_b, unsigned int templ_sqsum_b,
                    DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            float weight = 1.f / (w * h);
            float templ_sum_scale_r = templ_sum_r * weight;
            float templ_sum_scale_g = templ_sum_g * weight;
            float templ_sum_scale_b = templ_sum_b * weight;
            float templ_sqsum_scale = templ_sqsum_r - weight * templ_sum_r * templ_sum_r 
                                      + templ_sqsum_g - weight * templ_sum_g * templ_sum_g
                                      + templ_sqsum_b - weight * templ_sum_b * templ_sum_b;

            matchTemplatePreparedKernel_CCOFF_NORMED_8UC3<<<grid, threads, 0, stream>>>(
                    w, h, weight, 
                    templ_sum_scale_r, templ_sum_scale_g, templ_sum_scale_b, 
                    templ_sqsum_scale, 
                    image_sum_r, image_sqsum_r, 
                    image_sum_g, image_sqsum_g, 
                    image_sum_b, image_sqsum_b, 
                    result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }



        __global__ void matchTemplatePreparedKernel_CCOFF_NORMED_8UC4(
                int w, int h, float weight, 
                float templ_sum_scale_r, float templ_sum_scale_g, float templ_sum_scale_b, 
                float templ_sum_scale_a, float templ_sqsum_scale,
                const PtrStep<unsigned int> image_sum_r, const PtrStep<unsigned long long> image_sqsum_r,
                const PtrStep<unsigned int> image_sum_g, const PtrStep<unsigned long long> image_sqsum_g,
                const PtrStep<unsigned int> image_sum_b, const PtrStep<unsigned long long> image_sqsum_b,
                const PtrStep<unsigned int> image_sum_a, const PtrStep<unsigned long long> image_sqsum_a,
                DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sum_r_ = (float)(
                        (image_sum_r.ptr(y + h)[x + w] - image_sum_r.ptr(y)[x + w]) -
                        (image_sum_r.ptr(y + h)[x] - image_sum_r.ptr(y)[x]));
                float image_sqsum_r_ = (float)(
                        (image_sqsum_r.ptr(y + h)[x + w] - image_sqsum_r.ptr(y)[x + w]) -
                        (image_sqsum_r.ptr(y + h)[x] - image_sqsum_r.ptr(y)[x]));
                float image_sum_g_ = (float)(
                        (image_sum_g.ptr(y + h)[x + w] - image_sum_g.ptr(y)[x + w]) -
                        (image_sum_g.ptr(y + h)[x] - image_sum_g.ptr(y)[x]));
                float image_sqsum_g_ = (float)(
                        (image_sqsum_g.ptr(y + h)[x + w] - image_sqsum_g.ptr(y)[x + w]) -
                        (image_sqsum_g.ptr(y + h)[x] - image_sqsum_g.ptr(y)[x]));
                float image_sum_b_ = (float)(
                        (image_sum_b.ptr(y + h)[x + w] - image_sum_b.ptr(y)[x + w]) -
                        (image_sum_b.ptr(y + h)[x] - image_sum_b.ptr(y)[x]));
                float image_sqsum_b_ = (float)(
                        (image_sqsum_b.ptr(y + h)[x + w] - image_sqsum_b.ptr(y)[x + w]) -
                        (image_sqsum_b.ptr(y + h)[x] - image_sqsum_b.ptr(y)[x]));
                float image_sum_a_ = (float)(
                        (image_sum_a.ptr(y + h)[x + w] - image_sum_a.ptr(y)[x + w]) -
                        (image_sum_a.ptr(y + h)[x] - image_sum_a.ptr(y)[x]));
                float image_sqsum_a_ = (float)(
                        (image_sqsum_a.ptr(y + h)[x + w] - image_sqsum_a.ptr(y)[x + w]) -
                        (image_sqsum_a.ptr(y + h)[x] - image_sqsum_a.ptr(y)[x]));

                float num = result.ptr(y)[x] - image_sum_r_ * templ_sum_scale_r - image_sum_g_ * templ_sum_scale_g
                                             - image_sum_b_ * templ_sum_scale_b - image_sum_a_ * templ_sum_scale_a;
                float denum = sqrtf(templ_sqsum_scale * (image_sqsum_r_ - weight * image_sum_r_ * image_sum_r_
                                                         + image_sqsum_g_ - weight * image_sum_g_ * image_sum_g_
                                                         + image_sqsum_b_ - weight * image_sum_b_ * image_sum_b_
                                                         + image_sqsum_a_ - weight * image_sum_a_ * image_sum_a_));
                result.ptr(y)[x] = normAcc(num, denum);
            }
        }

        void matchTemplatePrepared_CCOFF_NORMED_8UC4(
                    int w, int h, 
                    const DevMem2D_<unsigned int> image_sum_r, const DevMem2D_<unsigned long long> image_sqsum_r,
                    const DevMem2D_<unsigned int> image_sum_g, const DevMem2D_<unsigned long long> image_sqsum_g,
                    const DevMem2D_<unsigned int> image_sum_b, const DevMem2D_<unsigned long long> image_sqsum_b,
                    const DevMem2D_<unsigned int> image_sum_a, const DevMem2D_<unsigned long long> image_sqsum_a,
                    unsigned int templ_sum_r, unsigned int templ_sqsum_r,
                    unsigned int templ_sum_g, unsigned int templ_sqsum_g,
                    unsigned int templ_sum_b, unsigned int templ_sqsum_b,
                    unsigned int templ_sum_a, unsigned int templ_sqsum_a,
                    DevMem2Df result, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            float weight = 1.f / (w * h);
            float templ_sum_scale_r = templ_sum_r * weight;
            float templ_sum_scale_g = templ_sum_g * weight;
            float templ_sum_scale_b = templ_sum_b * weight;
            float templ_sum_scale_a = templ_sum_a * weight;
            float templ_sqsum_scale = templ_sqsum_r - weight * templ_sum_r * templ_sum_r
                                      + templ_sqsum_g - weight * templ_sum_g * templ_sum_g
                                      + templ_sqsum_b - weight * templ_sum_b * templ_sum_b
                                      + templ_sqsum_a - weight * templ_sum_a * templ_sum_a;

            matchTemplatePreparedKernel_CCOFF_NORMED_8UC4<<<grid, threads, 0, stream>>>(
                    w, h, weight, 
                    templ_sum_scale_r, templ_sum_scale_g, templ_sum_scale_b, templ_sum_scale_a, 
                    templ_sqsum_scale, 
                    image_sum_r, image_sqsum_r, 
                    image_sum_g, image_sqsum_g, 
                    image_sum_b, image_sqsum_b, 
                    image_sum_a, image_sqsum_a, 
                    result);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        //////////////////////////////////////////////////////////////////////
        // normalize

        template <int cn>
        __global__ void normalizeKernel_8U(
                int w, int h, const PtrStep<unsigned long long> image_sqsum, 
                unsigned int templ_sqsum, DevMem2Df result)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                float image_sqsum_ = (float)(
                        (image_sqsum.ptr(y + h)[(x + w) * cn] - image_sqsum.ptr(y)[(x + w) * cn]) -
                        (image_sqsum.ptr(y + h)[x * cn] - image_sqsum.ptr(y)[x * cn]));
                result.ptr(y)[x] = normAcc(result.ptr(y)[x], sqrtf(image_sqsum_ * templ_sqsum));
            }
        }

        void normalize_8U(int w, int h, const DevMem2D_<unsigned long long> image_sqsum, 
                          unsigned int templ_sqsum, DevMem2Df result, int cn, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            switch (cn)
            {
            case 1:
                normalizeKernel_8U<1><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
                break;
            case 2:
                normalizeKernel_8U<2><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
                break;
            case 3:
                normalizeKernel_8U<3><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
                break;
            case 4:
                normalizeKernel_8U<4><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
                break;
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        //////////////////////////////////////////////////////////////////////
        // extractFirstChannel

        template <int cn>
        __global__ void extractFirstChannel_32F(const PtrStepb image, DevMem2Df result)
        {
            typedef typename TypeVec<float, cn>::vec_type Typef;

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < result.cols && y < result.rows)
            {
                Typef val = ((const Typef*)image.ptr(y))[x];
                result.ptr(y)[x] = first(val);
            }
        }

        void extractFirstChannel_32F(const DevMem2Db image, DevMem2Df result, int cn, cudaStream_t stream)
        {
            dim3 threads(32, 8);
            dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

            switch (cn)
            {
            case 1:
                extractFirstChannel_32F<1><<<grid, threads, 0, stream>>>(image, result);
                break;
            case 2:
                extractFirstChannel_32F<2><<<grid, threads, 0, stream>>>(image, result);
                break;
            case 3:
                extractFirstChannel_32F<3><<<grid, threads, 0, stream>>>(image, result);
                break;
            case 4:
                extractFirstChannel_32F<4><<<grid, threads, 0, stream>>>(image, result);
                break;
            }
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    } //namespace match_template
}}} // namespace cv { namespace gpu { namespace device
