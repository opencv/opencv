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

namespace cv { namespace gpu { namespace bf_match
{
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
    //template void matchSingleL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
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

    //template void matchSingleL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchSingleL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchSingleL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchSingleL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchSingleL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
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
    //template void matchSingleHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchSingleHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchSingleHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
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
    //template void matchCollectionL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
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

    //template void matchCollectionL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchCollectionL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchCollectionL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchCollectionL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchCollectionL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
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
    //template void matchCollectionHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    //template void matchCollectionHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
    template void matchCollectionHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2D& trainIdx, const DevMem2D& imgIdx, const DevMem2D& distance, int cc, cudaStream_t stream);
}}}
