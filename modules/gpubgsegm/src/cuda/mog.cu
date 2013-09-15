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
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/limits.hpp"

namespace cv { namespace gpu { namespace cudev
{
    namespace mog
    {
        ///////////////////////////////////////////////////////////////
        // Utility

        __device__ __forceinline__ float cvt(uchar val)
        {
            return val;
        }
        __device__ __forceinline__ float3 cvt(const uchar3& val)
        {
            return make_float3(val.x, val.y, val.z);
        }
        __device__ __forceinline__ float4 cvt(const uchar4& val)
        {
            return make_float4(val.x, val.y, val.z, val.w);
        }

        __device__ __forceinline__ float sqr(float val)
        {
            return val * val;
        }
        __device__ __forceinline__ float sqr(const float3& val)
        {
            return val.x * val.x + val.y * val.y + val.z * val.z;
        }
        __device__ __forceinline__ float sqr(const float4& val)
        {
            return val.x * val.x + val.y * val.y + val.z * val.z;
        }

        __device__ __forceinline__ float sum(float val)
        {
            return val;
        }
        __device__ __forceinline__ float sum(const float3& val)
        {
            return val.x + val.y + val.z;
        }
        __device__ __forceinline__ float sum(const float4& val)
        {
            return val.x + val.y + val.z;
        }

        __device__ __forceinline__ float clamp(float var, float learningRate, float diff, float minVar)
        {
             return ::fmaxf(var + learningRate * (diff * diff - var), minVar);
        }
        __device__ __forceinline__ float3 clamp(const float3& var, float learningRate, const float3& diff, float minVar)
        {
             return make_float3(::fmaxf(var.x + learningRate * (diff.x * diff.x - var.x), minVar),
                                ::fmaxf(var.y + learningRate * (diff.y * diff.y - var.y), minVar),
                                ::fmaxf(var.z + learningRate * (diff.z * diff.z - var.z), minVar));
        }
        __device__ __forceinline__ float4 clamp(const float4& var, float learningRate, const float4& diff, float minVar)
        {
             return make_float4(::fmaxf(var.x + learningRate * (diff.x * diff.x - var.x), minVar),
                                ::fmaxf(var.y + learningRate * (diff.y * diff.y - var.y), minVar),
                                ::fmaxf(var.z + learningRate * (diff.z * diff.z - var.z), minVar),
                                0.0f);
        }

        ///////////////////////////////////////////////////////////////
        // MOG without learning

        template <typename SrcT, typename WorkT>
        __global__ void mog_withoutLearning(const PtrStepSz<SrcT> frame, PtrStepb fgmask,
                                            const PtrStepf gmm_weight, const PtrStep<WorkT> gmm_mean, const PtrStep<WorkT> gmm_var,
                                            const int nmixtures, const float varThreshold, const float backgroundRatio)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= frame.cols || y >= frame.rows)
                return;

            WorkT pix = cvt(frame(y, x));

            int kHit = -1;
            int kForeground = -1;

            for (int k = 0; k < nmixtures; ++k)
            {
                if (gmm_weight(k * frame.rows + y, x) < numeric_limits<float>::epsilon())
                    break;

                WorkT mu = gmm_mean(k * frame.rows + y, x);
                WorkT var = gmm_var(k * frame.rows + y, x);

                WorkT diff = pix - mu;

                if (sqr(diff) < varThreshold * sum(var))
                {
                    kHit = k;
                    break;
                }
            }

            if (kHit >= 0)
            {
                float wsum = 0.0f;
                for (int k = 0; k < nmixtures; ++k)
                {
                    wsum += gmm_weight(k * frame.rows + y, x);

                    if (wsum > backgroundRatio)
                    {
                        kForeground = k + 1;
                        break;
                    }
                }
            }

            fgmask(y, x) = (uchar) (-(kHit < 0 || kHit >= kForeground));
        }

        template <typename SrcT, typename WorkT>
        void mog_withoutLearning_caller(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb var,
                                        int nmixtures, float varThreshold, float backgroundRatio, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(mog_withoutLearning<SrcT, WorkT>, cudaFuncCachePreferL1) );

            mog_withoutLearning<SrcT, WorkT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, fgmask,
                                                                         weight, (PtrStepSz<WorkT>) mean, (PtrStepSz<WorkT>) var,
                                                                         nmixtures, varThreshold, backgroundRatio);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ///////////////////////////////////////////////////////////////
        // MOG with learning

        template <typename SrcT, typename WorkT>
        __global__ void mog_withLearning(const PtrStepSz<SrcT> frame, PtrStepb fgmask,
                                         PtrStepf gmm_weight, PtrStepf gmm_sortKey, PtrStep<WorkT> gmm_mean, PtrStep<WorkT> gmm_var,
                                         const int nmixtures, const float varThreshold, const float backgroundRatio, const float learningRate, const float minVar)
        {
            const float w0 = 0.05f;
            const float sk0 = w0 / (30.0f * 0.5f * 2.0f);
            const float var0 = 30.0f * 0.5f * 30.0f * 0.5f * 4.0f;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= frame.cols || y >= frame.rows)
                return;

            WorkT pix = cvt(frame(y, x));

            float wsum = 0.0f;
            int kHit = -1;
            int kForeground = -1;

            int k = 0;
            for (; k < nmixtures; ++k)
            {
                float w = gmm_weight(k * frame.rows + y, x);
                wsum += w;

                if (w < numeric_limits<float>::epsilon())
                    break;

                WorkT mu = gmm_mean(k * frame.rows + y, x);
                WorkT var = gmm_var(k * frame.rows + y, x);

                WorkT diff = pix - mu;

                if (sqr(diff) < varThreshold * sum(var))
                {
                    wsum -= w;
                    float dw = learningRate * (1.0f - w);

                    var = clamp(var, learningRate, diff, minVar);

                    float sortKey_prev = w / ::sqrtf(sum(var));
                    gmm_sortKey(k * frame.rows + y, x) = sortKey_prev;

                    float weight_prev = w + dw;
                    gmm_weight(k * frame.rows + y, x) = weight_prev;

                    WorkT mean_prev = mu + learningRate * diff;
                    gmm_mean(k * frame.rows + y, x) = mean_prev;

                    WorkT var_prev = var;
                    gmm_var(k * frame.rows + y, x) = var_prev;

                    int k1 = k - 1;

                    if (k1 >= 0)
                    {
                        float sortKey_next = gmm_sortKey(k1 * frame.rows + y, x);
                        float weight_next = gmm_weight(k1 * frame.rows + y, x);
                        WorkT mean_next = gmm_mean(k1 * frame.rows + y, x);
                        WorkT var_next = gmm_var(k1 * frame.rows + y, x);

                        for (; sortKey_next < sortKey_prev && k1 >= 0; --k1)
                        {
                            gmm_sortKey(k1 * frame.rows + y, x) = sortKey_prev;
                            gmm_sortKey((k1 + 1) * frame.rows + y, x) = sortKey_next;

                            gmm_weight(k1 * frame.rows + y, x) = weight_prev;
                            gmm_weight((k1 + 1) * frame.rows + y, x) = weight_next;

                            gmm_mean(k1 * frame.rows + y, x) = mean_prev;
                            gmm_mean((k1 + 1) * frame.rows + y, x) = mean_next;

                            gmm_var(k1 * frame.rows + y, x) = var_prev;
                            gmm_var((k1 + 1) * frame.rows + y, x) = var_next;

                            sortKey_prev = sortKey_next;
                            sortKey_next = k1 > 0 ? gmm_sortKey((k1 - 1) * frame.rows + y, x) : 0.0f;

                            weight_prev = weight_next;
                            weight_next = k1 > 0 ? gmm_weight((k1 - 1) * frame.rows + y, x) : 0.0f;

                            mean_prev = mean_next;
                            mean_next = k1 > 0 ? gmm_mean((k1 - 1) * frame.rows + y, x) : VecTraits<WorkT>::all(0.0f);

                            var_prev = var_next;
                            var_next = k1 > 0 ? gmm_var((k1 - 1) * frame.rows + y, x) : VecTraits<WorkT>::all(0.0f);
                        }
                    }

                    kHit = k1 + 1;
                    break;
                }
            }

            if (kHit < 0)
            {
                // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                kHit = k = ::min(k, nmixtures - 1);
                wsum += w0 - gmm_weight(k * frame.rows + y, x);

                gmm_weight(k * frame.rows + y, x) = w0;
                gmm_mean(k * frame.rows + y, x) = pix;
                gmm_var(k * frame.rows + y, x) = VecTraits<WorkT>::all(var0);
                gmm_sortKey(k * frame.rows + y, x) = sk0;
            }
            else
            {
                for( ; k < nmixtures; k++)
                    wsum += gmm_weight(k * frame.rows + y, x);
            }

            float wscale = 1.0f / wsum;
            wsum = 0;
            for (k = 0; k < nmixtures; ++k)
            {
                float w = gmm_weight(k * frame.rows + y, x);
                wsum += w *= wscale;

                gmm_weight(k * frame.rows + y, x) = w;
                gmm_sortKey(k * frame.rows + y, x) *= wscale;

                if (wsum > backgroundRatio && kForeground < 0)
                    kForeground = k + 1;
            }

            fgmask(y, x) = (uchar)(-(kHit >= kForeground));
        }

        template <typename SrcT, typename WorkT>
        void mog_withLearning_caller(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzf weight, PtrStepSzf sortKey, PtrStepSzb mean, PtrStepSzb var,
                                     int nmixtures, float varThreshold, float backgroundRatio, float learningRate, float minVar,
                                     cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(mog_withLearning<SrcT, WorkT>, cudaFuncCachePreferL1) );

            mog_withLearning<SrcT, WorkT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, fgmask,
                                                                      weight, sortKey, (PtrStepSz<WorkT>) mean, (PtrStepSz<WorkT>) var,
                                                                      nmixtures, varThreshold, backgroundRatio, learningRate, minVar);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ///////////////////////////////////////////////////////////////
        // MOG

        void mog_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzf weight, PtrStepSzf sortKey, PtrStepSzb mean, PtrStepSzb var, int nmixtures, float varThreshold, float learningRate, float backgroundRatio, float noiseSigma, cudaStream_t stream)
        {
            typedef void (*withoutLearning_t)(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb var, int nmixtures, float varThreshold, float backgroundRatio, cudaStream_t stream);
            typedef void (*withLearning_t)(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzf weight, PtrStepSzf sortKey, PtrStepSzb mean, PtrStepSzb var, int nmixtures, float varThreshold, float backgroundRatio, float learningRate, float minVar, cudaStream_t stream);

            static const withoutLearning_t withoutLearning[] =
            {
                0, mog_withoutLearning_caller<uchar, float>, 0, mog_withoutLearning_caller<uchar3, float3>, mog_withoutLearning_caller<uchar4, float4>
            };
            static const withLearning_t withLearning[] =
            {
                0, mog_withLearning_caller<uchar, float>, 0, mog_withLearning_caller<uchar3, float3>, mog_withLearning_caller<uchar4, float4>
            };

            const float minVar = noiseSigma * noiseSigma;

            if (learningRate > 0.0f)
                withLearning[cn](frame, fgmask, weight, sortKey, mean, var, nmixtures, varThreshold, backgroundRatio, learningRate, minVar, stream);
            else
                withoutLearning[cn](frame, fgmask, weight, mean, var, nmixtures, varThreshold, backgroundRatio, stream);
        }

        template <typename WorkT, typename OutT>
        __global__ void getBackgroundImage(const PtrStepf gmm_weight, const PtrStep<WorkT> gmm_mean, PtrStepSz<OutT> dst, const int nmixtures, const float backgroundRatio)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= dst.cols || y >= dst.rows)
                return;

            WorkT meanVal = VecTraits<WorkT>::all(0.0f);
            float totalWeight = 0.0f;

            for (int mode = 0; mode < nmixtures; ++mode)
            {
                float weight = gmm_weight(mode * dst.rows + y, x);

                WorkT mean = gmm_mean(mode * dst.rows + y, x);
                meanVal = meanVal + weight * mean;

                totalWeight += weight;

                if(totalWeight > backgroundRatio)
                    break;
            }

            meanVal = meanVal * (1.f / totalWeight);

            dst(y, x) = saturate_cast<OutT>(meanVal);
        }

        template <typename WorkT, typename OutT>
        void getBackgroundImage_caller(PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, int nmixtures, float backgroundRatio, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(getBackgroundImage<WorkT, OutT>, cudaFuncCachePreferL1) );

            getBackgroundImage<WorkT, OutT><<<grid, block, 0, stream>>>(weight, (PtrStepSz<WorkT>) mean, (PtrStepSz<OutT>) dst, nmixtures, backgroundRatio);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void getBackgroundImage_gpu(int cn, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, int nmixtures, float backgroundRatio, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, int nmixtures, float backgroundRatio, cudaStream_t stream);

            static const func_t funcs[] =
            {
                0, getBackgroundImage_caller<float, uchar>, 0, getBackgroundImage_caller<float3, uchar3>, getBackgroundImage_caller<float4, uchar4>
            };

            funcs[cn](weight, mean, dst, nmixtures, backgroundRatio, stream);
        }
    }
}}}


#endif /* CUDA_DISABLER */
