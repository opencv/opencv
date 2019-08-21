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

namespace cv { namespace cuda { namespace device
{
    namespace mog2
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

        template <class Ptr2D>
        __device__ __forceinline__ void swap(Ptr2D& ptr, int x, int y, int k, int rows)
        {
            typename Ptr2D::elem_type val = ptr(k * rows + y, x);
            ptr(k * rows + y, x) = ptr((k + 1) * rows + y, x);
            ptr((k + 1) * rows + y, x) = val;
        }

        ///////////////////////////////////////////////////////////////
        // MOG2

        __constant__ int           c_nmixtures;
        __constant__ float         c_Tb;
        __constant__ float         c_TB;
        __constant__ float         c_Tg;
        __constant__ float         c_varInit;
        __constant__ float         c_varMin;
        __constant__ float         c_varMax;
        __constant__ float         c_tau;
        __constant__ unsigned char c_shadowVal;

        void loadConstants(int nmixtures, float Tb, float TB, float Tg, float varInit, float varMin, float varMax, float tau, unsigned char shadowVal)
        {
            varMin = ::fminf(varMin, varMax);
            varMax = ::fmaxf(varMin, varMax);

            cudaSafeCall( cudaMemcpyToSymbol(c_nmixtures, &nmixtures, sizeof(int)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_Tb, &Tb, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_TB, &TB, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_Tg, &Tg, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_varInit, &varInit, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_varMin, &varMin, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_varMax, &varMax, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_tau, &tau, sizeof(float)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_shadowVal, &shadowVal, sizeof(unsigned char)) );
        }

        template <bool detectShadows, typename SrcT, typename WorkT>
        __global__ void mog2(const PtrStepSz<SrcT> frame, PtrStepb fgmask, PtrStepb modesUsed,
                             PtrStepf gmm_weight, PtrStepf gmm_variance, PtrStep<WorkT> gmm_mean,
                             const float alphaT, const float alpha1, const float prune)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= frame.cols || y >= frame.rows)
                return;

            WorkT pix = cvt(frame(y, x));

            //calculate distances to the modes (+ sort)
            //here we need to go in descending order!!!

            bool background = false; // true - the pixel classified as background

            //internal:

            bool fitsPDF = false; //if it remains zero a new GMM mode will be added

            int nmodes = modesUsed(y, x);
            int nNewModes = nmodes; //current number of modes in GMM

            float totalWeight = 0.0f;

            //go through all modes

            for (int mode = 0; mode < nmodes; ++mode)
            {
                //need only weight if fit is found
                float weight = alpha1 * gmm_weight(mode * frame.rows + y, x) + prune;
                int swap_count = 0;
                //fit not found yet
                if (!fitsPDF)
                {
                    //check if it belongs to some of the remaining modes
                    float var = gmm_variance(mode * frame.rows + y, x);

                    WorkT mean = gmm_mean(mode * frame.rows + y, x);

                    //calculate difference and distance
                    WorkT diff = mean - pix;
                    float dist2 = sqr(diff);

                    //background? - Tb - usually larger than Tg
                    if (totalWeight < c_TB && dist2 < c_Tb * var)
                        background = true;

                    //check fit
                    if (dist2 < c_Tg * var)
                    {
                        //belongs to the mode
                        fitsPDF = true;

                        //update distribution

                        //update weight
                        weight += alphaT;
                        float k = alphaT / weight;

                        //update mean
                        gmm_mean(mode * frame.rows + y, x) = mean - k * diff;

                        //update variance
                        float varnew = var + k * (dist2 - var);

                        //limit the variance
                        varnew = ::fmaxf(varnew, c_varMin);
                        varnew = ::fminf(varnew, c_varMax);

                        gmm_variance(mode * frame.rows + y, x) = varnew;

                        //sort
                        //all other weights are at the same place and
                        //only the matched (iModes) is higher -> just find the new place for it

                        for (int i = mode; i > 0; --i)
                        {
                            //check one up
                            if (weight < gmm_weight((i - 1) * frame.rows + y, x))
                                break;

                            swap_count++;
                            //swap one up
                            swap(gmm_weight, x, y, i - 1, frame.rows);
                            swap(gmm_variance, x, y, i - 1, frame.rows);
                            swap(gmm_mean, x, y, i - 1, frame.rows);
                        }

                        //belongs to the mode - bFitsPDF becomes 1
                    }
                } // !fitsPDF

                //check prune
                if (weight < -prune)
                {
                    weight = 0.0f;
                    nmodes--;
                }

                gmm_weight((mode - swap_count) * frame.rows + y, x) = weight; //update weight by the calculated value
                totalWeight += weight;
            }

            //renormalize weights

            totalWeight = 1.f / totalWeight;
            for (int mode = 0; mode < nmodes; ++mode)
                gmm_weight(mode * frame.rows + y, x) *= totalWeight;

            nmodes = nNewModes;

            //make new mode if needed and exit

            if (!fitsPDF)
            {
                // replace the weakest or add a new one
                int mode = nmodes == c_nmixtures ? c_nmixtures - 1 : nmodes++;

                if (nmodes == 1)
                    gmm_weight(mode * frame.rows + y, x) = 1.f;
                else
                {
                    gmm_weight(mode * frame.rows + y, x) = alphaT;

                    // renormalize all other weights

                    for (int i = 0; i < nmodes - 1; ++i)
                        gmm_weight(i * frame.rows + y, x) *= alpha1;
                }

                // init

                gmm_mean(mode * frame.rows + y, x) = pix;
                gmm_variance(mode * frame.rows + y, x) = c_varInit;

                //sort
                //find the new place for it

                for (int i = nmodes - 1; i > 0; --i)
                {
                    // check one up
                    if (alphaT < gmm_weight((i - 1) * frame.rows + y, x))
                        break;

                    //swap one up
                    swap(gmm_weight, x, y, i - 1, frame.rows);
                    swap(gmm_variance, x, y, i - 1, frame.rows);
                    swap(gmm_mean, x, y, i - 1, frame.rows);
                }
            }

            //set the number of modes
            modesUsed(y, x) = nmodes;

            bool isShadow = false;
            if (detectShadows && !background)
            {
                float tWeight = 0.0f;

                // check all the components  marked as background:
                for (int mode = 0; mode < nmodes; ++mode)
                {
                    WorkT mean = gmm_mean(mode * frame.rows + y, x);

                    WorkT pix_mean = pix * mean;

                    float numerator = sum(pix_mean);
                    float denominator = sqr(mean);

                    // no division by zero allowed
                    if (denominator == 0)
                        break;

                    // if tau < a < 1 then also check the color distortion
                    if (numerator <= denominator && numerator >= c_tau * denominator)
                    {
                        float a = numerator / denominator;

                        WorkT dD = a * mean - pix;

                        if (sqr(dD) < c_Tb * gmm_variance(mode * frame.rows + y, x) * a * a)
                        {
                            isShadow = true;
                            break;
                        }
                    };

                    tWeight += gmm_weight(mode * frame.rows + y, x);
                    if (tWeight > c_TB)
                        break;
                }
            }

            fgmask(y, x) = background ? 0 : isShadow ? c_shadowVal : 255;
        }

        template <typename SrcT, typename WorkT>
        void mog2_caller(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzf variance, PtrStepSzb mean,
                         float alphaT, float prune, bool detectShadows, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

            const float alpha1 = 1.0f - alphaT;

            if (detectShadows)
            {
                cudaSafeCall( cudaFuncSetCacheConfig(mog2<true, SrcT, WorkT>, cudaFuncCachePreferL1) );

                mog2<true, SrcT, WorkT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, fgmask, modesUsed,
                                                                    weight, variance, (PtrStepSz<WorkT>) mean,
                                                                    alphaT, alpha1, prune);
            }
            else
            {
                cudaSafeCall( cudaFuncSetCacheConfig(mog2<false, SrcT, WorkT>, cudaFuncCachePreferL1) );

                mog2<false, SrcT, WorkT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, fgmask, modesUsed,
                                                                    weight, variance, (PtrStepSz<WorkT>) mean,
                                                                    alphaT, alpha1, prune);
            }

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void mog2_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzf variance, PtrStepSzb mean,
                      float alphaT, float prune, bool detectShadows, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzf variance, PtrStepSzb mean, float alphaT, float prune, bool detectShadows, cudaStream_t stream);

            static const func_t funcs[] =
            {
                0, mog2_caller<uchar, float>, 0, mog2_caller<uchar3, float3>, mog2_caller<uchar4, float4>
            };

            funcs[cn](frame, fgmask, modesUsed, weight, variance, mean, alphaT, prune, detectShadows, stream);
        }

        template <typename WorkT, typename OutT>
        __global__ void getBackgroundImage2(const PtrStepSzb modesUsed, const PtrStepf gmm_weight, const PtrStep<WorkT> gmm_mean, PtrStep<OutT> dst)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= modesUsed.cols || y >= modesUsed.rows)
                return;

            int nmodes = modesUsed(y, x);

            WorkT meanVal = VecTraits<WorkT>::all(0.0f);
            float totalWeight = 0.0f;

            for (int mode = 0; mode < nmodes; ++mode)
            {
                float weight = gmm_weight(mode * modesUsed.rows + y, x);

                WorkT mean = gmm_mean(mode * modesUsed.rows + y, x);
                meanVal = meanVal + weight * mean;

                totalWeight += weight;

                if(totalWeight > c_TB)
                    break;
            }

            meanVal = meanVal * (1.f / totalWeight);

            dst(y, x) = saturate_cast<OutT>(meanVal);
        }

        template <typename WorkT, typename OutT>
        void getBackgroundImage2_caller(PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, cudaStream_t stream)
        {
            dim3 block(32, 8);
            dim3 grid(divUp(modesUsed.cols, block.x), divUp(modesUsed.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(getBackgroundImage2<WorkT, OutT>, cudaFuncCachePreferL1) );

            getBackgroundImage2<WorkT, OutT><<<grid, block, 0, stream>>>(modesUsed, weight, (PtrStepSz<WorkT>) mean, (PtrStepSz<OutT>) dst);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        void getBackgroundImage2_gpu(int cn, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, cudaStream_t stream)
        {
            typedef void (*func_t)(PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, cudaStream_t stream);

            static const func_t funcs[] =
            {
                0, getBackgroundImage2_caller<float, uchar>, 0, getBackgroundImage2_caller<float3, uchar3>, getBackgroundImage2_caller<float4, uchar4>
            };

            funcs[cn](modesUsed, weight, mean, dst, stream);
        }
    }
}}}


#endif /* CUDA_DISABLER */
