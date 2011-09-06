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

///////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Match //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

    template <int BLOCK_DIM_Y, typename T>
    __device__ void findBestMatch(T& myDist, int2& myIdx, T* smin, int2* sIdx)
    {
        if (threadIdx.x == 0)
        {
            smin[threadIdx.y] = myDist;
            sIdx[threadIdx.y] = myIdx;
        }
        __syncthreads();

        reducePredVal<BLOCK_DIM_Y>(smin, myDist, sIdx, myIdx, threadIdx.y * blockDim.x + threadIdx.x, less<volatile T>());
    }

    template <typename Dist, typename VecDiff, typename T, typename Mask>
    __device__ void matchDescs(int queryIdx, int imgIdx, const DevMem2D_<T>& train, const Mask& m, const VecDiff& vecDiff,
        typename Dist::result_type& myDist, int2& myIdx, typename Dist::result_type* sdiff_row)
    {
        for (int trainIdx = threadIdx.y; trainIdx < train.rows; trainIdx += blockDim.y)
        {
            if (m(queryIdx, trainIdx))
            {
                const T* trainDescs = train.ptr(trainIdx);

                Dist dist;

                vecDiff.calc(trainDescs, train.cols, dist, sdiff_row, threadIdx.x);

                const typename Dist::result_type res = dist;

                if (res < myDist)
                {
                    myDist = res;
                    myIdx.x = trainIdx;
                    myIdx.y = imgIdx;
                }
            }
        }
    }

    template <typename T> struct SingleTrain
    {
        explicit SingleTrain(const DevMem2D_<T>& train_) : train(train_)
        {
        }

        template <typename Dist, typename VecDiff, typename Mask>
        __device__ __forceinline__ void loop(int queryIdx, Mask& m, const VecDiff& vecDiff, 
            typename Dist::result_type& myDist, int2& myIdx, typename Dist::result_type* sdiff_row) const
        {
            matchDescs<Dist>(queryIdx, 0, train, m, vecDiff, myDist, myIdx, sdiff_row);
        }

        __device__ __forceinline__ int desc_len() const
        {
            return train.cols;
        }

        static __device__ __forceinline__ void storeResult(float* distance, int* trainIdx, int* imgIdx, 
            float myDist, const int2& myIdx, int queryIdx)
        {
            trainIdx[queryIdx] = myIdx.x;
            distance[queryIdx] = myDist;
        }

        const DevMem2D_<T> train;
    };

    template <typename T> struct TrainCollection
    {
        TrainCollection(const DevMem2D_<T>* trainCollection_, int nImg_, int desclen_) : 
            trainCollection(trainCollection_), nImg(nImg_), desclen(desclen_)
        {
        }

        template <typename Dist, typename VecDiff, typename Mask>
        __device__ void loop(int queryIdx, Mask& m, const VecDiff& vecDiff, 
            typename Dist::result_type& myDist, int2& myIdx, typename Dist::result_type* sdiff_row) const
        {
            for (int imgIdx = 0; imgIdx < nImg; ++imgIdx)
            {
                const DevMem2D_<T> train = trainCollection[imgIdx];
                m.next();
                matchDescs<Dist>(queryIdx, imgIdx, train, m, vecDiff, myDist, myIdx, sdiff_row);
            }
        }

        __device__ __forceinline__ int desc_len() const
        {
            return desclen;
        }

        static __device__ __forceinline__ void storeResult(float* distance, int* trainIdx, int* imgIdx, 
            float myDist, const int2& myIdx, int queryIdx)
        {
            trainIdx[queryIdx] = myIdx.x;
            imgIdx[queryIdx] = myIdx.y;
            distance[queryIdx] = myDist;
        }

        const DevMem2D_<T>* trainCollection;
        const int nImg;
        const int desclen;
    };

    template <typename VecDiff, typename Dist, typename T, typename Train, typename Mask>
    __device__ void distanceCalcLoop(const PtrStep_<T>& query, const Train& train, const Mask& mask, int queryIdx, 
        typename Dist::result_type& myDist, int2& myIdx, typename Dist::result_type* smem)
    {
        const VecDiff vecDiff(query.ptr(queryIdx), train.desc_len(), (typename Dist::value_type*)smem, threadIdx.y * blockDim.x + threadIdx.x, threadIdx.x);
    
        typename Dist::result_type* sdiff_row = smem + blockDim.x * threadIdx.y;

        Mask m = mask;

        myIdx.x = -1;
        myIdx.y = -1;
        myDist = numeric_limits<typename Dist::result_type>::max();

        train.template loop<Dist>(queryIdx, m, vecDiff, myDist, myIdx, sdiff_row);
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename VecDiff, typename Dist, typename T, typename Train, typename Mask>
    __global__ void match(const PtrStep_<T> query, const Train train, const Mask mask, int* trainIdx, int* imgIdx, float* distance)
    {
        __shared__ typename Dist::result_type smem[BLOCK_DIM_X * BLOCK_DIM_Y];        
        
        const int queryIdx = blockIdx.x;
        
        int2 myIdx;
        typename Dist::result_type myDist;

        distanceCalcLoop<VecDiff, Dist>(query, train, mask, queryIdx, myDist, myIdx, smem);
        __syncthreads();

        typename Dist::result_type* smin = smem;
        int2* sIdx = (int2*)(smin + BLOCK_DIM_Y);

        findBestMatch<BLOCK_DIM_Y>(myDist, myIdx, smin, sIdx);

        if (threadIdx.x == 0 && threadIdx.y == 0)
            Train::storeResult(distance, trainIdx, imgIdx, myDist, myIdx, queryIdx);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Match kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Train, typename Mask>
    void matchSimple_caller(const DevMem2D_<T>& query, const Train& train, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_DIM_Y <= 64>::check(); // blockDimY vals must reduce by warp

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        match<BLOCK_DIM_X, BLOCK_DIM_Y, VecDiffGlobal<BLOCK_DIM_X, T>, Dist, T>
            <<<grid, threads, 0, stream>>>(query, train, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_LEN, bool LEN_EQ_MAX_LEN, typename Dist, typename T, typename Train, typename Mask>
    void matchCached_caller(const DevMem2D_<T>& query, const Train& train, const Mask& mask, 
        const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_DIM_Y <= 64>::check();                    // blockDimY vals must reduce by warp
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_LEN>::check(); // block size must be greter than descriptors length
        StaticAssert<MAX_LEN % BLOCK_DIM_X == 0>::check();           // max descriptors length must divide to blockDimX

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        match<BLOCK_DIM_X, BLOCK_DIM_Y, VecDiffCachedRegister<BLOCK_DIM_X, MAX_LEN, LEN_EQ_MAX_LEN, typename Dist::value_type>, Dist, T>
              <<<grid, threads, 0, stream>>>(query, train, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Match Dispatcher

    template <typename Dist, typename T, typename Train, typename Mask>
    void matchDispatcher(const DevMem2D_<T>& query, const Train& train, const Mask& mask, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance,
        int cc, cudaStream_t stream)
    {
        if (query.cols < 64)
        {
            matchCached_caller<16, 16, 64, false, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
        else if (query.cols == 64)
        {
            matchCached_caller<16, 16, 64, true, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
        else if (query.cols < 128)
        {
            matchCached_caller<16, 16, 128, false, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
        else if (query.cols == 128 && cc >= 12)
        {
            matchCached_caller<16, 16, 128, true, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
        else if (query.cols < 256 && cc >= 12)
        {
            matchCached_caller<16, 16, 256, false, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
        else if (query.cols == 256 && cc >= 12)
        {
            matchCached_caller<16, 16, 256, true, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
        else
        {
            matchSimple_caller<16, 16, Dist>(
                query, train, mask, 
                static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Di>(imgIdx), static_cast<DevMem2Df>(distance), 
                stream);
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Match caller

    template <typename T> void matchSingleL1_gpu(const DevMem2D& query, const DevMem2D& train_, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance,
        int cc, cudaStream_t stream)
    {
        SingleTrain<T> train(static_cast< DevMem2D_<T> >(train_));
        if (mask.data)
            matchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, SingleMask(mask), trainIdx, DevMem2D(), distance, cc, stream);
        else
            matchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, WithOutMask(), trainIdx, DevMem2D(), distance, cc, stream);
    }

    template void matchSingleL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);

    template <typename T> void matchSingleL2_gpu(const DevMem2D& query, const DevMem2D& train_, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream)
    {
        SingleTrain<T> train(static_cast< DevMem2D_<T> >(train_));
        if (mask.data)
            matchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, SingleMask(mask), trainIdx, DevMem2D(), distance, cc, stream);
        else
            matchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, WithOutMask(), trainIdx, DevMem2D(), distance, cc, stream);
    }

    template void matchSingleL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);

    template <typename T> void matchSingleHamming_gpu(const DevMem2D& query, const DevMem2D& train_, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream)
    {
        SingleTrain<T> train(static_cast< DevMem2D_<T> >(train_));
        if (mask.data)
            matchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, SingleMask(mask), trainIdx, DevMem2D(), distance, cc, stream);
        else
            matchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, WithOutMask(), trainIdx, DevMem2D(), distance, cc, stream);
    }

    template void matchSingleHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);

    template <typename T> void matchCollectionL1_gpu(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, query.cols);
        if (maskCollection.data)
            matchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, MaskCollection(maskCollection.data), trainIdx, imgIdx, distance, cc, stream);
        else
            matchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), train, WithOutMask(), trainIdx, imgIdx, distance, cc, stream);
    }

    template void matchCollectionL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);

    template <typename T> void matchCollectionL2_gpu(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, query.cols);
        if (maskCollection.data)
            matchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, MaskCollection(maskCollection.data), trainIdx, imgIdx, distance, cc, stream);
        else
            matchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), train, WithOutMask(), trainIdx, imgIdx, distance, cc, stream);
    }

    template void matchCollectionL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);

    template <typename T> void matchCollectionHamming_gpu(const DevMem2D& query, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, 
        const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, query.cols);
        if (maskCollection.data)
            matchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, MaskCollection(maskCollection.data), trainIdx, imgIdx, distance, cc, stream);
        else
            matchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), train, WithOutMask(), trainIdx, imgIdx, distance, cc, stream);
    }

    template void matchCollectionHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Knn Match ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

    template <typename VecDiff, typename Dist, typename T, typename Mask>
    __device__ void distanceCalcLoop(const PtrStep_<T>& query, const DevMem2D_<T>& train, const Mask& m, int queryIdx,
        typename Dist::result_type& distMin1, typename Dist::result_type& distMin2, int& bestTrainIdx1, int& bestTrainIdx2, 
        typename Dist::result_type* smem)
    {
        const VecDiff vecDiff(query.ptr(queryIdx), train.cols, (typename Dist::value_type*)smem, threadIdx.y * blockDim.x + threadIdx.x, threadIdx.x);
        
        typename Dist::result_type* sdiffRow = smem + blockDim.x * threadIdx.y;
        
        distMin1 = numeric_limits<typename Dist::result_type>::max();
        distMin2 = numeric_limits<typename Dist::result_type>::max();

        bestTrainIdx1 = -1;
        bestTrainIdx2 = -1;

        for (int trainIdx = threadIdx.y; trainIdx < train.rows; trainIdx += blockDim.y)
        {
            if (m(queryIdx, trainIdx))
            {
                Dist dist;

                const T* trainRow = train.ptr(trainIdx);
                
                vecDiff.calc(trainRow, train.cols, dist, sdiffRow, threadIdx.x);

                const typename Dist::result_type val = dist;

                if (val < distMin1)
                {
                    distMin1 = val;
                    bestTrainIdx1 = trainIdx;
                }
                else if (val < distMin2)
                {
                    distMin2 = val;
                    bestTrainIdx2 = trainIdx;
                }
            }
        }
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename VecDiff, typename Dist, typename T, typename Mask>
    __global__ void knnMatch2(const PtrStep_<T> query, const DevMem2D_<T> train, const Mask m, int2* trainIdx, float2* distance)
    {
        typedef typename Dist::result_type result_type;
        typedef typename Dist::value_type value_type;

        __shared__ result_type smem[BLOCK_DIM_X * BLOCK_DIM_Y];

        const int queryIdx = blockIdx.x;

        result_type distMin1;
        result_type distMin2;

        int bestTrainIdx1;
        int bestTrainIdx2;

        distanceCalcLoop<VecDiff, Dist>(query, train, m, queryIdx, distMin1, distMin2, bestTrainIdx1, bestTrainIdx2, smem);
        __syncthreads();

        volatile result_type* sdistMinRow = smem;
        volatile int* sbestTrainIdxRow = (int*)(sdistMinRow + 2 * BLOCK_DIM_Y);

        if (threadIdx.x == 0)
        {
            sdistMinRow[threadIdx.y] = distMin1;
            sdistMinRow[threadIdx.y + BLOCK_DIM_Y] = distMin2;

            sbestTrainIdxRow[threadIdx.y] = bestTrainIdx1;            
            sbestTrainIdxRow[threadIdx.y + BLOCK_DIM_Y] = bestTrainIdx2;
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            distMin1 = numeric_limits<result_type>::max();
            distMin2 = numeric_limits<result_type>::max();

            bestTrainIdx1 = -1;
            bestTrainIdx2 = -1;

            #pragma unroll
            for (int i = 0; i < BLOCK_DIM_Y; ++i)
            {
                result_type val = sdistMinRow[i];

                if (val < distMin1)
                {
                    distMin1 = val;
                    bestTrainIdx1 = sbestTrainIdxRow[i];
                }
                else if (val < distMin2)
                {
                    distMin2 = val;
                    bestTrainIdx2 = sbestTrainIdxRow[i];
                }
            }

            #pragma unroll
            for (int i = BLOCK_DIM_Y; i < 2 * BLOCK_DIM_Y; ++i)
            {
                result_type val = sdistMinRow[i];

                if (val < distMin2)
                {
                    distMin2 = val;
                    bestTrainIdx2 = sbestTrainIdxRow[i];
                }
            }

            trainIdx[queryIdx] = make_int2(bestTrainIdx1, bestTrainIdx2);
            distance[queryIdx] = make_float2(distMin1, distMin2);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Knn 2 Match kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    void knnMatch2Simple_caller(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
        const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, 
        cudaStream_t stream)
    {
        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        knnMatch2<BLOCK_DIM_X, BLOCK_DIM_Y, VecDiffGlobal<BLOCK_DIM_X, T>, Dist, T>
            <<<grid, threads, 0, stream>>>(query, train, mask, trainIdx, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_LEN, bool LEN_EQ_MAX_LEN, typename Dist, typename T, typename Mask>
    void knnMatch2Cached_caller(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
        const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, 
        cudaStream_t stream)
    {
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_LEN>::check(); // block size must be greter than descriptors length
        StaticAssert<MAX_LEN % BLOCK_DIM_X == 0>::check();           // max descriptors length must divide to blockDimX

        const dim3 grid(query.rows, 1, 1);
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        knnMatch2<BLOCK_DIM_X, BLOCK_DIM_Y, VecDiffCachedRegister<BLOCK_DIM_X, MAX_LEN, LEN_EQ_MAX_LEN, typename Dist::value_type>, Dist, T>
              <<<grid, threads, 0, stream>>>(query, train, mask, trainIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Knn 2 Match Dispatcher
    
    template <typename Dist, typename T, typename Mask>
    void knnMatch2Dispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, 
        int cc, cudaStream_t stream)
    {
        if (query.cols < 64)
        {
            knnMatch2Cached_caller<16, 16, 64, false, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance),
                stream);
        }
        else if (query.cols == 64)
        {
            knnMatch2Cached_caller<16, 16, 64, true, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance), 
                stream);
        }
        else if (query.cols < 128)
        {
            knnMatch2Cached_caller<16, 16, 128, false, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance), 
                stream);
        }
        else if (query.cols == 128 && cc >= 12)
        {
            knnMatch2Cached_caller<16, 16, 128, true, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance), 
                stream);
        }
        else if (query.cols < 256 && cc >= 12)
        {
            knnMatch2Cached_caller<16, 16, 256, false, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance), 
                stream);
        }
        else if (query.cols == 256 && cc >= 12)
        {
            knnMatch2Cached_caller<16, 16, 256, true, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance), 
                stream);
        }
        else
        {
            knnMatch2Simple_caller<16, 16, Dist>(
                query, train, mask, 
                static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> >(distance),
                stream);
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    __global__ void calcDistance(const PtrStep_<T> query, const DevMem2D_<T> train, const Mask mask, PtrStepf distance)
    {
        __shared__ typename Dist::result_type sdiff[BLOCK_DIM_X * BLOCK_DIM_Y];

        typename Dist::result_type* sdiff_row = sdiff + BLOCK_DIM_X * threadIdx.y;
        
        const int queryIdx = blockIdx.x;
        const T* queryDescs = query.ptr(queryIdx);

        const int trainIdx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (trainIdx < train.rows)
        {
            const T* trainDescs = train.ptr(trainIdx);

            typename Dist::result_type myDist = numeric_limits<typename Dist::result_type>::max();

            if (mask(queryIdx, trainIdx))
            {
                Dist dist;

                calcVecDiffGlobal<BLOCK_DIM_X>(queryDescs, trainDescs, train.cols, dist, sdiff_row, threadIdx.x);

                myDist = dist;
            }
            
            if (threadIdx.x == 0)
                distance.ptr(queryIdx)[trainIdx] = myDist;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    void calcDistance_caller(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, const DevMem2Df& distance, cudaStream_t stream)
    {
        const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        const dim3 grid(query.rows, divUp(train.rows, BLOCK_DIM_Y), 1);

        calcDistance<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, T><<<grid, threads, 0, stream>>>(query, train, mask, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename Dist, typename T, typename Mask>
    void calcDistanceDispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, const DevMem2D& allDist, cudaStream_t stream)
    {
        calcDistance_caller<16, 16, Dist>(query, train, mask, static_cast<DevMem2Df>(allDist), stream);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // find knn match kernel

    template <int BLOCK_SIZE> __global__ void findBestMatch(DevMem2Df allDist_, int i, PtrStepi trainIdx_, PtrStepf distance_)
    {
        const int SMEM_SIZE = BLOCK_SIZE > 64 ? BLOCK_SIZE : 64;
        __shared__ float sdist[SMEM_SIZE];
        __shared__ int strainIdx[SMEM_SIZE];

        const int queryIdx = blockIdx.x;

        float* allDist = allDist_.ptr(queryIdx);
        int* trainIdx = trainIdx_.ptr(queryIdx);
        float* distance = distance_.ptr(queryIdx);

        float dist = numeric_limits<float>::max();
        int bestIdx = -1;
        
        for (int i = threadIdx.x; i < allDist_.cols; i += BLOCK_SIZE)
        {
            float reg = allDist[i];
            if (reg < dist)
            {
                dist = reg;
                bestIdx = i;
            }
        }

        sdist[threadIdx.x] = dist;
        strainIdx[threadIdx.x] = bestIdx;
        __syncthreads();

        reducePredVal<BLOCK_SIZE>(sdist, dist, strainIdx, bestIdx, threadIdx.x, less<volatile float>());

        if (threadIdx.x == 0)
        {
            if (dist < numeric_limits<float>::max())
            {
                allDist[bestIdx] = numeric_limits<float>::max();
                trainIdx[i] = bestIdx;
                distance[i] = dist;
            }
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // find knn match kernel caller

    template <int BLOCK_SIZE> void findKnnMatch_caller(int k, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, cudaStream_t stream)
    {
        const dim3 threads(BLOCK_SIZE, 1, 1);
        const dim3 grid(trainIdx.rows, 1, 1);

        for (int i = 0; i < k; ++i)
        {
            findBestMatch<BLOCK_SIZE><<<grid, threads, 0, stream>>>(allDist, i, trainIdx, distance);
            cudaSafeCall( cudaGetLastError() );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    void findKnnMatchDispatcher(int k, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, cudaStream_t stream)
    {
        findKnnMatch_caller<256>(k, static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), static_cast<DevMem2Df>(allDist), stream);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // knn match Dispatcher

    template <typename Dist, typename T>
    void knnMatchDispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream)
    {
        if (mask.data)
        {
            if (k == 2)
            {
                knnMatch2Dispatcher<Dist>(query, train, SingleMask(mask), trainIdx, distance, cc, stream);
                return;
            }

            calcDistanceDispatcher<Dist>(query, train, SingleMask(mask), allDist, stream);
        }
        else
        {
            if (k == 2)
            {
                knnMatch2Dispatcher<Dist>(query, train, WithOutMask(), trainIdx, distance, cc, stream);
                return;
            }

            calcDistanceDispatcher<Dist>(query, train, WithOutMask(), allDist, stream);
        }

        findKnnMatchDispatcher(k, trainIdx, distance, allDist, stream);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // knn match caller

    template <typename T> void knnMatchL1_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream)
    {
        knnMatchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, mask, trainIdx, distance, allDist, cc, stream);
    }

    template void knnMatchL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);

    template <typename T> void knnMatchL2_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask, 
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist,
        int cc, cudaStream_t stream)
    {
        knnMatchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, mask, trainIdx, distance, allDist, cc, stream);
    }

    template void knnMatchL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);

    template <typename T> void knnMatchHamming_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask,
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream)
    {
        knnMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, mask, trainIdx, distance, allDist, cc, stream);
    }

    template void knnMatchHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Radius Match //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

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
    template void radiusMatchL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
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

    template void radiusMatchL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
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
    template void radiusMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& nMatches, const DevMem2D& distance, cudaStream_t stream);
}}}
