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

namespace cv { namespace gpu { namespace cudev
{
    #define SOLVE_PNP_RANSAC_MAX_NUM_ITERS 200

    namespace transform_points
    {
        __constant__ float3 crot0;
        __constant__ float3 crot1;
        __constant__ float3 crot2;
        __constant__ float3 ctransl;

        struct TransformOp : unary_function<float3, float3>
        {
            __device__ __forceinline__ float3 operator()(const float3& p) const
            {
                return make_float3(
                        crot0.x * p.x + crot0.y * p.y + crot0.z * p.z + ctransl.x,
                        crot1.x * p.x + crot1.y * p.y + crot1.z * p.z + ctransl.y,
                        crot2.x * p.x + crot2.y * p.y + crot2.z * p.z + ctransl.z);
            }
            __device__ __forceinline__ TransformOp() {}
            __device__ __forceinline__ TransformOp(const TransformOp&) {}
        };

        void call(const PtrStepSz<float3> src, const float* rot,
                  const float* transl, PtrStepSz<float3> dst,
                  cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(crot0, rot, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot1, rot + 3, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot2, rot + 6, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(ctransl, transl, sizeof(float) * 3));
            cv::gpu::cudev::transform(src, dst, TransformOp(), WithOutMask(), stream);
        }
    } // namespace transform_points

    namespace project_points
    {
        __constant__ float3 crot0;
        __constant__ float3 crot1;
        __constant__ float3 crot2;
        __constant__ float3 ctransl;
        __constant__ float3 cproj0;
        __constant__ float3 cproj1;

        struct ProjectOp : unary_function<float3, float3>
        {
            __device__ __forceinline__ float2 operator()(const float3& p) const
            {
                // Rotate and translate in 3D
                float3 t = make_float3(
                        crot0.x * p.x + crot0.y * p.y + crot0.z * p.z + ctransl.x,
                        crot1.x * p.x + crot1.y * p.y + crot1.z * p.z + ctransl.y,
                        crot2.x * p.x + crot2.y * p.y + crot2.z * p.z + ctransl.z);
                // Project on 2D plane
                return make_float2(
                        (cproj0.x * t.x + cproj0.y * t.y) / t.z + cproj0.z,
                        (cproj1.x * t.x + cproj1.y * t.y) / t.z + cproj1.z);
            }
            __device__ __forceinline__ ProjectOp() {}
            __device__ __forceinline__ ProjectOp(const ProjectOp&) {}
        };

        void call(const PtrStepSz<float3> src, const float* rot,
                  const float* transl, const float* proj, PtrStepSz<float2> dst,
                  cudaStream_t stream)
        {
            cudaSafeCall(cudaMemcpyToSymbol(crot0, rot, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot1, rot + 3, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(crot2, rot + 6, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(ctransl, transl, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(cproj0, proj, sizeof(float) * 3));
            cudaSafeCall(cudaMemcpyToSymbol(cproj1, proj + 3, sizeof(float) * 3));
            cv::gpu::cudev::transform(src, dst, ProjectOp(), WithOutMask(), stream);
        }
    } // namespace project_points

    namespace solve_pnp_ransac
    {
        __constant__ float3 crot_matrices[SOLVE_PNP_RANSAC_MAX_NUM_ITERS * 3];
        __constant__ float3 ctransl_vectors[SOLVE_PNP_RANSAC_MAX_NUM_ITERS];

        int maxNumIters()
        {
            return SOLVE_PNP_RANSAC_MAX_NUM_ITERS;
        }

        __device__ __forceinline__ float sqr(float x)
        {
            return x * x;
        }

        template <int BLOCK_SIZE>
        __global__ void computeHypothesisScoresKernel(
                const int num_points, const float3* object, const float2* image,
                const float dist_threshold, int* g_num_inliers)
        {
            const float3* const &rot_mat = crot_matrices + blockIdx.x * 3;
            const float3 &transl_vec = ctransl_vectors[blockIdx.x];
            int num_inliers = 0;

            for (int i = threadIdx.x; i < num_points; i += blockDim.x)
            {
                float3 p = object[i];
                p = make_float3(
                        rot_mat[0].x * p.x + rot_mat[0].y * p.y + rot_mat[0].z * p.z + transl_vec.x,
                        rot_mat[1].x * p.x + rot_mat[1].y * p.y + rot_mat[1].z * p.z + transl_vec.y,
                        rot_mat[2].x * p.x + rot_mat[2].y * p.y + rot_mat[2].z * p.z + transl_vec.z);
                p.x /= p.z;
                p.y /= p.z;
                float2 image_p = image[i];
                if (sqr(p.x - image_p.x) + sqr(p.y - image_p.y) < dist_threshold)
                    ++num_inliers;
            }

            __shared__ int s_num_inliers[BLOCK_SIZE];
            reduce<BLOCK_SIZE>(s_num_inliers, num_inliers, threadIdx.x, plus<int>());

            if (threadIdx.x == 0)
                g_num_inliers[blockIdx.x] = num_inliers;
        }

        void computeHypothesisScores(
                const int num_hypotheses, const int num_points, const float* rot_matrices,
                const float3* transl_vectors, const float3* object, const float2* image,
                const float dist_threshold, int* hypothesis_scores)
        {
            cudaSafeCall(cudaMemcpyToSymbol(crot_matrices, rot_matrices, num_hypotheses * 3 * sizeof(float3)));
            cudaSafeCall(cudaMemcpyToSymbol(ctransl_vectors, transl_vectors, num_hypotheses * sizeof(float3)));

            dim3 threads(256);
            dim3 grid(num_hypotheses);

            computeHypothesisScoresKernel<256><<<grid, threads>>>(
                    num_points, object, image, dist_threshold, hypothesis_scores);
            cudaSafeCall( cudaGetLastError() );

            cudaSafeCall( cudaDeviceSynchronize() );
        }
    } // namespace solvepnp_ransac



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
}}} // namespace cv { namespace gpu { namespace cudev


#endif /* CUDA_DISABLER */
