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

namespace cv { namespace gpu { namespace bf_radius_match
{
    __device__ __forceinline__ void store(const int* sidx, const float* sdist, const unsigned int scount, int* trainIdx, float* distance, int& sglob_ind, const int tid)
    {
        if (tid < scount)
        {
            trainIdx[sglob_ind + tid] = sidx[tid];
            distance[sglob_ind + tid] = sdist[tid];
        }

        if (tid == 0)
            sglob_ind += scount;
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_STACK, typename VecDiff, typename Dist, typename T, typename Mask>
    __global__ void radiusMatch(const PtrStep_<T> query, const DevMem2D_<T> train, const float maxDistance, const Mask mask, 
        DevMem2Di trainIdx_, PtrStepf distance, unsigned int* nMatches)
    {
        #if __CUDA_ARCH__ >= 120

        typedef typename Dist::result_type result_type;
        typedef typename Dist::value_type value_type;

        __shared__ result_type smem[BLOCK_DIM_X * BLOCK_DIM_Y];
        __shared__ int sidx[BLOCK_STACK];
        __shared__ float sdist[BLOCK_STACK];
        __shared__ unsigned int scount;
        __shared__ int sglob_ind;

        const int queryIdx = blockIdx.x;
        const int tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;

        if (tid == 0)
        {
            scount = 0;
            sglob_ind = 0;
        }
        __syncthreads();

        int* trainIdx_row = trainIdx_.ptr(queryIdx);
        float* distance_row = distance.ptr(queryIdx);

        const VecDiff vecDiff(query.ptr(queryIdx), train.cols, (typename Dist::value_type*)smem, tid, threadIdx.x);
        
        typename Dist::result_type* sdiffRow = smem + BLOCK_DIM_X * threadIdx.y;

        for (int trainIdx = threadIdx.y; trainIdx < train.rows; trainIdx += BLOCK_DIM_Y)
        {
            if (mask(queryIdx, trainIdx))
            {
                Dist dist;

                const T* trainRow = train.ptr(trainIdx);
                
                vecDiff.calc(trainRow, train.cols, dist, sdiffRow, threadIdx.x);

                const typename Dist::result_type val = dist;

                if (threadIdx.x == 0 && val < maxDistance)
                {
                    unsigned int i = atomicInc(&scount, (unsigned int) -1);
                    sidx[i] = trainIdx;
                    sdist[i] = val;
                }
            }
            __syncthreads();

            if (scount > BLOCK_STACK - BLOCK_DIM_Y)
            {
                store(sidx, sdist, scount, trainIdx_row, distance_row, sglob_ind, tid);
                if (tid == 0)
                    scount = 0;
            }
            __syncthreads();
        }

        store(sidx, sdist, scount, trainIdx_row, distance_row, sglob_ind, tid);

        if (tid == 0)
            nMatches[queryIdx] = sglob_ind;

        #endif
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_STACK, typename Dist, typename T, typename Mask>
    void radiusMatchSimple_caller(const DevMem2D_<T>& query, const DevMem2D_<T>& train, float maxDistance, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2Df& distance, unsigned int* nMatches,
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_STACK >= BLOCK_DIM_Y>::check();
        StaticAssert<BLOCK_STACK <= BLOCK_DIM_X * BLOCK_DIM_Y>::check();

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_STACK, VecDiffGlobal<BLOCK_DIM_X, T>, Dist, T>
            <<<grid, threads, 0, stream>>>(query, train, maxDistance, mask, trainIdx, distance, nMatches);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_STACK, int MAX_LEN, bool LEN_EQ_MAX_LEN, typename Dist, typename T, typename Mask>
    void radiusMatchCached_caller(const DevMem2D_<T>& query, const DevMem2D_<T>& train, float maxDistance, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2Df& distance, unsigned int* nMatches, 
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_STACK >= BLOCK_DIM_Y>::check();
        StaticAssert<BLOCK_STACK <= BLOCK_DIM_X * BLOCK_DIM_Y>::check();
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_LEN>::check();
        StaticAssert<MAX_LEN % BLOCK_DIM_X == 0>::check();

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_STACK, VecDiffCachedRegister<BLOCK_DIM_X, MAX_LEN, LEN_EQ_MAX_LEN, typename Dist::value_type>, Dist, T>
              <<<grid, threads, 0, stream>>>(query, train, maxDistance, mask, trainIdx, distance, nMatches);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match Dispatcher
    
    template <typename Dist, typename T, typename Mask>
    void radiusMatchDispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, float maxDistance, const Mask& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        if (query.cols < 64)
        {
            radiusMatchCached_caller<16, 16, 64, 64, false, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data,
                stream);
        }
        else if (query.cols == 64)
        {
            radiusMatchCached_caller<16, 16, 64, 64, true, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data,
                stream);
        }
        else if (query.cols < 128)
        {
            radiusMatchCached_caller<16, 16, 64, 128, false, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data,
                stream);
        }
        else if (query.cols == 128)
        {
            radiusMatchCached_caller<16, 16, 64, 128, true, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data,
                stream);
        }
        else if (query.cols < 256)
        {
            radiusMatchCached_caller<16, 16, 64, 256, false, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data,
                stream);
        }
        else if (query.cols == 256)
        {
            radiusMatchCached_caller<16, 16, 64, 256, true, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data, 
                stream);
        }
        else
        {
            radiusMatchSimple_caller<16, 16, 64, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), (unsigned int*)nMatches.data,
                stream);
        }
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
                trainIdx, distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, WithOutMask(), 
                trainIdx, distance, nMatches, 
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
                trainIdx, distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, WithOutMask(), 
                trainIdx, distance, nMatches, 
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
                trainIdx, distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), maxDistance, WithOutMask(), 
                trainIdx, distance, nMatches, 
                stream);
        }
    }

    template void radiusMatchHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    //template void radiusMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
}}}
