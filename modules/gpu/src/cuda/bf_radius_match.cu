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
    template <typename T> struct SingleTrain
    {
        enum {USE_IMG_IDX = 0};

        explicit SingleTrain(const DevMem2D_<T>& train_) : train(train_)
        {
        }

        static __device__ __forceinline__ void store(const int* s_trainIdx, const int* s_imgIdx, const float* s_dist, unsigned int& s_count, int& s_globInd, 
            int* trainIdx, int* imgIdx, float* distance, int maxCount)
        {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            if (tid < s_count && s_globInd + tid < maxCount)
            {
                trainIdx[s_globInd + tid] = s_trainIdx[tid];
                distance[s_globInd + tid] = s_dist[tid];
            }

            if (tid == 0)
            {
                s_globInd += s_count;
                s_count = 0;
            }
        }

        template <int BLOCK_STACK, typename Dist, typename VecDiff, typename Mask>
        __device__ __forceinline__ void loop(float maxDistance, Mask& mask, const VecDiff& vecDiff, 
            int* s_trainIdx, int* s_imgIdx, float* s_dist, unsigned int& s_count, int& s_globInd, 
            int* trainIdxRow, int* imgIdxRow, float* distanceRow, int maxCount, 
            typename Dist::result_type* s_diffRow) const
        {
            #if __CUDA_ARCH__ >= 120

            for (int i = 0; i < train.rows; i += blockDim.y)
            {
                int trainIdx = i + threadIdx.y;

                if (trainIdx < train.rows && mask(blockIdx.x, trainIdx))
                {
                    Dist dist;
                    
                    vecDiff.calc(train.ptr(trainIdx), train.cols, dist, s_diffRow, threadIdx.x);

                    const typename Dist::result_type val = dist;

                    if (threadIdx.x == 0 && val < maxDistance)
                    {
                        unsigned int ind = atomicInc(&s_count, (unsigned int) -1);
                        s_trainIdx[ind] = trainIdx;
                        s_dist[ind] = val;
                    }
                }

                __syncthreads();

                if (s_count >= BLOCK_STACK - blockDim.y)
                    store(s_trainIdx, s_imgIdx, s_dist, s_count, s_globInd, trainIdxRow, imgIdxRow, distanceRow, maxCount);

                __syncthreads();
            }

            store(s_trainIdx, s_imgIdx, s_dist, s_count, s_globInd, trainIdxRow, imgIdxRow, distanceRow, maxCount);

            #endif
        }

        __device__ __forceinline__ int descLen() const
        {
            return train.cols;
        }

        const DevMem2D_<T> train;
    };

    template <typename T> struct TrainCollection
    {
        enum {USE_IMG_IDX = 1};

        TrainCollection(const DevMem2D_<T>* trainCollection_, int nImg_, int desclen_) : 
            trainCollection(trainCollection_), nImg(nImg_), desclen(desclen_)
        {
        }

        static __device__ __forceinline__ void store(const int* s_trainIdx, const int* s_imgIdx, const float* s_dist, unsigned int& s_count, int& s_globInd, 
            int* trainIdx, int* imgIdx, float* distance, int maxCount)
        {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            if (tid < s_count && s_globInd + tid < maxCount)
            {
                trainIdx[s_globInd + tid] = s_trainIdx[tid];
                imgIdx[s_globInd + tid] = s_imgIdx[tid];
                distance[s_globInd + tid] = s_dist[tid];
            }

            if (tid == 0)
            {
                s_globInd += s_count;
                s_count = 0;
            }
        }

        template <int BLOCK_STACK, typename Dist, typename VecDiff, typename Mask>
        __device__ void loop(float maxDistance, Mask& mask, const VecDiff& vecDiff, 
            int* s_trainIdx, int* s_imgIdx, float* s_dist, unsigned int& s_count, int& s_globInd, 
            int* trainIdxRow, int* imgIdxRow, float* distanceRow, int maxCount, 
            typename Dist::result_type* s_diffRow) const
        {
            #if __CUDA_ARCH__ >= 120

            for (int imgIdx = 0; imgIdx < nImg; ++imgIdx)
            {
                const DevMem2D_<T> train = trainCollection[imgIdx];

                mask.next();

                for (int i = 0; i < train.rows; i += blockDim.y)
                {
                    int trainIdx = i + threadIdx.y;

                    if (trainIdx < train.rows && mask(blockIdx.x, trainIdx))
                    {
                        Dist dist;
                        
                        vecDiff.calc(train.ptr(trainIdx), desclen, dist, s_diffRow, threadIdx.x);

                        const typename Dist::result_type val = dist;

                        if (threadIdx.x == 0 && val < maxDistance)
                        {
                            unsigned int ind = atomicInc(&s_count, (unsigned int) -1);
                            s_trainIdx[ind] = trainIdx;
                            s_imgIdx[ind] = imgIdx;
                            s_dist[ind] = val;
                        }
                    }

                    __syncthreads();

                    if (s_count >= BLOCK_STACK - blockDim.y)
                        store(s_trainIdx, s_imgIdx, s_dist, s_count, s_globInd, trainIdxRow, imgIdxRow, distanceRow, maxCount);

                    __syncthreads();
                }
            }

            store(s_trainIdx, s_imgIdx, s_dist, s_count, s_globInd, trainIdxRow, imgIdxRow, distanceRow, maxCount);

            #endif
        }

        __device__ __forceinline__ int descLen() const
        {
            return desclen;
        }

        const DevMem2D_<T>* trainCollection;
        const int nImg;
        const int desclen;
    };

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_STACK, typename VecDiff, typename Dist, typename T, typename Train, typename Mask>
    __global__ void radiusMatch(const PtrStep_<T> query, const Train train, float maxDistance, const Mask mask, 
        PtrStepi trainIdx, PtrStepi imgIdx, PtrStepf distance, int* nMatches, int maxCount)
    {
        typedef typename Dist::result_type result_type;
        typedef typename Dist::value_type value_type;

        __shared__ result_type s_mem[BLOCK_DIM_X * BLOCK_DIM_Y];

        __shared__ int s_trainIdx[BLOCK_STACK];
        __shared__ int s_imgIdx[Train::USE_IMG_IDX ? BLOCK_STACK : 1];
        __shared__ float s_dist[BLOCK_STACK];
        __shared__ unsigned int s_count;

        __shared__ int s_globInd;

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            s_count = 0;
            s_globInd = 0;
        }
        __syncthreads();

        const VecDiff vecDiff(query.ptr(blockIdx.x), train.descLen(), (typename Dist::value_type*)s_mem, threadIdx.y * BLOCK_DIM_X + threadIdx.x, threadIdx.x);

        Mask m = mask;

        train.template loop<BLOCK_STACK, Dist>(maxDistance, m, vecDiff, 
            s_trainIdx, s_imgIdx, s_dist, s_count, s_globInd, 
            trainIdx.ptr(blockIdx.x), imgIdx.ptr(blockIdx.x), distance.ptr(blockIdx.x), maxCount, 
            s_mem + BLOCK_DIM_X * threadIdx.y);

        if (threadIdx.x == 0 && threadIdx.y == 0)
            nMatches[blockIdx.x] = s_globInd;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_STACK, typename Dist, typename T, typename Train, typename Mask>
    void radiusMatchSimple_caller(const DevMem2D_<T>& query, const Train& train, float maxDistance, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, int* nMatches,
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_STACK >= BLOCK_DIM_Y>::check();
        StaticAssert<BLOCK_STACK <= BLOCK_DIM_X * BLOCK_DIM_Y>::check();

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_STACK, VecDiffGlobal<BLOCK_DIM_X, T>, Dist, T>
            <<<grid, threads, 0, stream>>>(query, train, maxDistance, mask, trainIdx, imgIdx, distance, nMatches, trainIdx.cols);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_STACK, int MAX_LEN, bool LEN_EQ_MAX_LEN, typename Dist, typename T, typename Train, typename Mask>
    void radiusMatchCached_caller(const DevMem2D_<T>& query, const Train& train, float maxDistance, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, int* nMatches, 
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_STACK >= BLOCK_DIM_Y>::check();
        StaticAssert<BLOCK_STACK <= BLOCK_DIM_X * BLOCK_DIM_Y>::check();
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_LEN>::check();
        StaticAssert<MAX_LEN % BLOCK_DIM_X == 0>::check();

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_STACK, VecDiffCachedRegister<BLOCK_DIM_X, MAX_LEN, LEN_EQ_MAX_LEN, typename Dist::value_type>, Dist, T>
              <<<grid, threads, 0, stream>>>(query, train, maxDistance, mask, trainIdx, imgIdx, distance, nMatches, trainIdx.cols);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match Dispatcher
    
    template <typename Dist, typename T, typename Train, typename Mask>
    void radiusMatchDispatcher(const DevMem2D_<T>& query, const Train& train, float maxDistance, const Mask& mask, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        if (query.cols < 64)
        {
            radiusMatchCached_caller<16, 16, 64, 64, false, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data,
                stream);
        }
        else if (query.cols == 64)
        {
            radiusMatchCached_caller<16, 16, 64, 64, true, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data,
                stream);
        }
        else if (query.cols < 128)
        {
            radiusMatchCached_caller<16, 16, 64, 128, false, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data,
                stream);
        }
        else if (query.cols == 128)
        {
            radiusMatchCached_caller<16, 16, 64, 128, true, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data,
                stream);
        }
        else if (query.cols < 256)
        {
            radiusMatchCached_caller<16, 16, 64, 256, false, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data,
                stream);
        }
        else if (query.cols == 256)
        {
            radiusMatchCached_caller<16, 16, 64, 256, true, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data, 
                stream);
        }
        else
        {
            radiusMatchSimple_caller<16, 16, 64, Dist>(
                query, train, maxDistance, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), (int*)nMatches.data,
                stream);
        }
    }    
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match caller

    template <typename T> void radiusMatchSingleL1_gpu(const DevMem2D& query, const DevMem2D& train_, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        SingleTrain<T> train(static_cast< DevMem2D_<T> >(train_));

        if (mask.data)
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, maxDistance, SingleMask(mask), 
                trainIdx, DevMem2D(), distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, maxDistance, WithOutMask(), 
                trainIdx, DevMem2D(), distance, nMatches, 
                stream);
        }
    }

    template void radiusMatchSingleL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);

    template <typename T> void radiusMatchSingleL2_gpu(const DevMem2D& query, const DevMem2D& train_, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        SingleTrain<T> train(static_cast< DevMem2D_<T> >(train_));

        if (mask.data)
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, SingleMask(mask), 
                trainIdx, DevMem2D(), distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, WithOutMask(), 
                trainIdx, DevMem2D(), distance, nMatches, 
                stream);
        }
    }

    //template void radiusMatchSingleL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);

    template <typename T> void radiusMatchSingleHamming_gpu(const DevMem2D& query, const DevMem2D& train_, float maxDistance, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        SingleTrain<T> train(static_cast< DevMem2D_<T> >(train_));

        if (mask.data)
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, SingleMask(mask), 
                trainIdx, DevMem2D(), distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, WithOutMask(), 
                trainIdx, DevMem2D(), distance, nMatches, 
                stream);
        }
    }

    template void radiusMatchSingleHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchSingleHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchSingleHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);

    template <typename T> void radiusMatchCollectionL1_gpu(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, query.cols);

        if (maskCollection.data)
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, maxDistance, MaskCollection(maskCollection.data), 
                trainIdx, imgIdx, distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, maxDistance, WithOutMask(), 
                trainIdx, imgIdx, distance, nMatches, 
                stream);
        }
    }

    template void radiusMatchCollectionL1_gpu<uchar >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionL1_gpu<schar >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionL1_gpu<ushort>(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionL1_gpu<short >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionL1_gpu<int   >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionL1_gpu<float >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);

    template <typename T> void radiusMatchCollectionL2_gpu(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, query.cols);

        if (maskCollection.data)
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, MaskCollection(maskCollection.data), 
                trainIdx, imgIdx, distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, WithOutMask(), 
                trainIdx, imgIdx, distance, nMatches, 
                stream);
        }
    }

    //template void radiusMatchCollectionL2_gpu<uchar >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionL2_gpu<schar >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionL2_gpu<ushort>(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionL2_gpu<short >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionL2_gpu<int   >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionL2_gpu<float >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);

    template <typename T> void radiusMatchCollectionHamming_gpu(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, 
        cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, query.cols);

        if (maskCollection.data)
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, MaskCollection(maskCollection.data), 
                trainIdx, imgIdx, distance, nMatches, 
                stream);
        }
        else
        {
            radiusMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, maxDistance, WithOutMask(), 
                trainIdx, imgIdx, distance, nMatches, 
                stream);
        }
    }

    template void radiusMatchCollectionHamming_gpu<uchar >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionHamming_gpu<schar >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionHamming_gpu<ushort>(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    //template void radiusMatchCollectionHamming_gpu<short >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
    template void radiusMatchCollectionHamming_gpu<int   >(const DevMem2D& query, const DevMem2D& trainCollection, float maxDistance, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, const DevMem2D& nMatches, cudaStream_t stream);
}}}
