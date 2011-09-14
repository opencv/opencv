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
// any express or bpied warranties, including, but not limited to, the bpied
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
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/vec_distance.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace cv { namespace gpu { namespace bfmatcher
{
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    __global__ void radiusMatch(const PtrStep_<T> query, const DevMem2D_<T> train, float maxDistance, const Mask mask, 
        DevMem2Di trainIdx_, unsigned int* nMatches, PtrStepf distance)
    {
        #if __CUDA_ARCH__ >= 110

        __shared__ typename Dist::result_type smem[BLOCK_DIM_X * BLOCK_DIM_Y];

        typename Dist::result_type* sdiff_row = smem + BLOCK_DIM_X * threadIdx.y;
        
        const int queryIdx = blockIdx.x;
        const T* queryDescs = query.ptr(queryIdx);

        const int trainIdx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (trainIdx < train.rows)
        {
            const T* trainDescs = train.ptr(trainIdx);

            if (mask(queryIdx, trainIdx))
            {
                Dist dist;

                calcVecDiffGlobal<BLOCK_DIM_X>(queryDescs, trainDescs, train.cols, dist, sdiff_row, threadIdx.x);

                if (threadIdx.x == 0)
                {
                    if (dist < maxDistance)
                    {
                        unsigned int i = atomicInc(nMatches + queryIdx, (unsigned int) -1);
                        if (i < trainIdx_.cols)
                        {
                            distance.ptr(queryIdx)[i] = dist;
                            trainIdx_.ptr(queryIdx)[i] = trainIdx;
                        }
                    }
                }
            }
        }

        #endif
    }
        
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    void radiusMatch_caller(const DevMem2D_<T>& query, const DevMem2D_<T>& train, float maxDistance, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2D_<unsigned int>& nMatches, const DevMem2Df& distance, 
        cudaStream_t stream)
    {
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        const dim3 grid(query.rows, divUp(train.rows, BLOCK_DIM_Y), 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, T><<<grid, threads, 0, stream>>>(query, train, maxDistance, mask, trainIdx, nMatches.data, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match Dispatcher

    template <typename Dist, typename T, typename Mask>
    void radiusMatchDispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, float maxDistance, const Mask& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream)
    {
        radiusMatch_caller<16, 16, Dist>(query, train, maxDistance, mask, 
            static_cast<DevMem2Di>(trainIdx), static_cast< const DevMem2D_<unsigned int> >(nMatches), static_cast<DevMem2Df>(distance), 
            stream);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match caller

    template <typename T> void radiusMatchL1_gpu(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream)
    {
        if (mask.data)
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, SingleMask(mask), 
                trainIdx, nMatches, distance, 
                stream);
        }
        else
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, WithOutMask(), 
                trainIdx, nMatches, distance, 
                stream);
        }
    }

    template void radiusMatchL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);

    template <typename T> void radiusMatchL2_gpu(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream)
    {
        if (mask.data)
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, SingleMask(mask), 
                trainIdx, nMatches, distance, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, WithOutMask(), 
                trainIdx, nMatches, distance, 
                stream);
        }
    }

    //template void radiusMatchL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);

    template <typename T> void radiusMatchHamming_gpu(const DevMem2D& query, const DevMem2D& train, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, 
        cudaStream_t stream)
    {
        if (mask.data)
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, SingleMask(mask), 
                trainIdx, nMatches, distance, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, WithOutMask(), 
                trainIdx, nMatches, distance, 
                stream);
        }
    }

    template void radiusMatchHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
}}}
