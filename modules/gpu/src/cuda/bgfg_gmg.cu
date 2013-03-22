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

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/limits.hpp"

namespace cv { namespace gpu { namespace device {
    namespace bgfg_gmg
    {
        __constant__ int   c_width;
        __constant__ int   c_height;
        __constant__ float c_minVal;
        __constant__ float c_maxVal;
        __constant__ int   c_quantizationLevels;
        __constant__ float c_backgroundPrior;
        __constant__ float c_decisionThreshold;
        __constant__ int   c_maxFeatures;
        __constant__ int   c_numInitializationFrames;

        void loadConstants(int width, int height, float minVal, float maxVal, int quantizationLevels, float backgroundPrior,
                           float decisionThreshold, int maxFeatures, int numInitializationFrames)
        {
            cudaSafeCall( cudaMemcpyToSymbol(c_width, &width, sizeof(width)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_height, &height, sizeof(height)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_minVal, &minVal, sizeof(minVal)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_maxVal, &maxVal, sizeof(maxVal)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_quantizationLevels, &quantizationLevels, sizeof(quantizationLevels)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_backgroundPrior, &backgroundPrior, sizeof(backgroundPrior)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_decisionThreshold, &decisionThreshold, sizeof(decisionThreshold)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_maxFeatures, &maxFeatures, sizeof(maxFeatures)) );
            cudaSafeCall( cudaMemcpyToSymbol(c_numInitializationFrames, &numInitializationFrames, sizeof(numInitializationFrames)) );
        }

        __device__ float findFeature(const int color, const PtrStepi& colors, const PtrStepf& weights, const int x, const int y, const int nfeatures)
        {
            for (int i = 0, fy = y; i < nfeatures; ++i, fy += c_height)
            {
                if (color == colors(fy, x))
                    return weights(fy, x);
            }

            // not in histogram, so return 0.
            return 0.0f;
        }

        __device__ void normalizeHistogram(PtrStepf weights, const int x, const int y, const int nfeatures)
        {
            float total = 0.0f;
            for (int i = 0, fy = y; i < nfeatures; ++i, fy += c_height)
                total += weights(fy, x);

            if (total != 0.0f)
            {
                for (int i = 0, fy = y; i < nfeatures; ++i, fy += c_height)
                    weights(fy, x) /= total;
            }
        }

        __device__ bool insertFeature(const int color, const float weight, PtrStepi colors, PtrStepf weights, const int x, const int y, int& nfeatures)
        {
            for (int i = 0, fy = y; i < nfeatures; ++i, fy += c_height)
            {
                if (color == colors(fy, x))
                {
                    // feature in histogram

                    weights(fy, x) += weight;

                    return false;
                }
            }

            if (nfeatures == c_maxFeatures)
            {
                // discard oldest feature

                int idx = -1;
                float minVal = numeric_limits<float>::max();
                for (int i = 0, fy = y; i < nfeatures; ++i, fy += c_height)
                {
                    const float w = weights(fy, x);
                    if (w < minVal)
                    {
                        minVal = w;
                        idx = fy;
                    }
                }

                colors(idx, x) = color;
                weights(idx, x) = weight;

                return false;
            }

            colors(nfeatures * c_height + y, x) = color;
            weights(nfeatures * c_height + y, x) = weight;

            ++nfeatures;

            return true;
        }

        namespace detail
        {
            template <int cn> struct Quantization
            {
                template <typename T>
                __device__ static int apply(const T& val)
                {
                    int res = 0;
                    res |= static_cast<int>((val.x - c_minVal) * c_quantizationLevels / (c_maxVal - c_minVal));
                    res |= static_cast<int>((val.y - c_minVal) * c_quantizationLevels / (c_maxVal - c_minVal)) << 8;
                    res |= static_cast<int>((val.z - c_minVal) * c_quantizationLevels / (c_maxVal - c_minVal)) << 16;
                    return res;
                }
            };

            template <> struct Quantization<1>
            {
                template <typename T>
                __device__ static int apply(T val)
                {
                    return static_cast<int>((val - c_minVal) * c_quantizationLevels / (c_maxVal - c_minVal));
                }
            };
        }

        template <typename T> struct Quantization : detail::Quantization<VecTraits<T>::cn> {};

        template <typename SrcT>
        __global__ void update(const PtrStep<SrcT> frame, PtrStepb fgmask, PtrStepi colors_, PtrStepf weights_, PtrStepi nfeatures_,
                               const int frameNum, const float learningRate, const bool updateBackgroundModel)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= c_width || y >= c_height)
                return;

            const SrcT pix = frame(y, x);
            const int newFeatureColor = Quantization<SrcT>::apply(pix);

            int nfeatures = nfeatures_(y, x);

            if (frameNum >= c_numInitializationFrames)
            {
                // typical operation

                const float weight = findFeature(newFeatureColor, colors_, weights_, x, y, nfeatures);

                // see Godbehere, Matsukawa, Goldberg (2012) for reasoning behind this implementation of Bayes rule
                const float posterior = (weight * c_backgroundPrior) / (weight * c_backgroundPrior + (1.0f - weight) * (1.0f - c_backgroundPrior));

                const bool isForeground = ((1.0f - posterior) > c_decisionThreshold);
                fgmask(y, x) = (uchar)(-isForeground);

                // update histogram.

                if (updateBackgroundModel)
                {
                    for (int i = 0, fy = y; i < nfeatures; ++i, fy += c_height)
                        weights_(fy, x) *= 1.0f - learningRate;

                    bool inserted = insertFeature(newFeatureColor, learningRate, colors_, weights_, x, y, nfeatures);

                    if (inserted)
                    {
                        normalizeHistogram(weights_, x, y, nfeatures);
                        nfeatures_(y, x) = nfeatures;
                    }
                }
            }
            else if (updateBackgroundModel)
            {
                // training-mode update

                insertFeature(newFeatureColor, 1.0f, colors_, weights_, x, y, nfeatures);

                if (frameNum == c_numInitializationFrames - 1)
                    normalizeHistogram(weights_, x, y, nfeatures);
            }
        }

        template <typename SrcT>
        void update_gpu(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures,
                        int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream)
        {
            const dim3 block(32, 8);
            const dim3 grid(divUp(frame.cols, block.x), divUp(frame.rows, block.y));

            cudaSafeCall( cudaFuncSetCacheConfig(update<SrcT>, cudaFuncCachePreferL1) );

            update<SrcT><<<grid, block, 0, stream>>>((PtrStepSz<SrcT>) frame, fgmask, colors, weights, nfeatures, frameNum, learningRate, updateBackgroundModel);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void update_gpu<uchar  >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        template void update_gpu<uchar3 >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        template void update_gpu<uchar4 >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);

        template void update_gpu<ushort >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        template void update_gpu<ushort3>(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        template void update_gpu<ushort4>(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);

        template void update_gpu<float  >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        template void update_gpu<float3 >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        template void update_gpu<float4 >(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures, int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
    }
}}}


#endif /* CUDA_DISABLER */
