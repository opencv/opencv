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

namespace cv { namespace gpu { namespace bf_knnmatch
{
    ///////////////////////////////////////////////////////////////////////////////
    // Reduction

    template <int BLOCK_SIZE> 
    __device__ void findBestMatch(float& bestDistance1, float& bestDistance2, 
                                  int& bestTrainIdx1, int& bestTrainIdx2, 
                                  float* s_distance, int* s_trainIdx)
    {
        float myBestDistance1 = numeric_limits<float>::max(); 
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;

        s_distance += threadIdx.y * BLOCK_SIZE;
        s_trainIdx += threadIdx.y * BLOCK_SIZE;

        s_distance[threadIdx.x] = bestDistance1;
        s_trainIdx[threadIdx.x] = bestTrainIdx1;

        __syncthreads();

        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                float val = s_distance[i];

                if (val < myBestDistance1)
                {
                    myBestDistance2 = myBestDistance1;
                    myBestTrainIdx2 = myBestTrainIdx1;

                    myBestDistance1 = val;
                    myBestTrainIdx1 = s_trainIdx[i];
                }
                else if (val < myBestDistance2)
                {
                    myBestDistance2 = val;
                    myBestTrainIdx2 = s_trainIdx[i];
                }
            }
        }

        __syncthreads();

        s_distance[threadIdx.x] = bestDistance2;
        s_trainIdx[threadIdx.x] = bestTrainIdx2;

        __syncthreads();

        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                float val = s_distance[i];

                if (val < myBestDistance2)
                {
                    myBestDistance2 = val;
                    myBestTrainIdx2 = s_trainIdx[i];
                }
            }
        }

        bestDistance1 = myBestDistance1;
        bestDistance2 = myBestDistance2;

        bestTrainIdx1 = myBestTrainIdx1;
        bestTrainIdx2 = myBestTrainIdx2;
    }

    template <int BLOCK_SIZE> 
    __device__ void findBestMatch(float& bestDistance1, float& bestDistance2, 
                                   int& bestTrainIdx1, int& bestTrainIdx2, 
                                   int& bestImgIdx1, int& bestImgIdx2, 
                                   float* s_distance, int* s_trainIdx, int* s_imgIdx)
    {
        float myBestDistance1 = numeric_limits<float>::max(); 
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;
        int myBestImgIdx1 = -1;
        int myBestImgIdx2 = -1;

        s_distance += threadIdx.y * BLOCK_SIZE;
        s_trainIdx += threadIdx.y * BLOCK_SIZE;
        s_imgIdx   += threadIdx.y * BLOCK_SIZE;

        s_distance[threadIdx.x] = bestDistance1;
        s_trainIdx[threadIdx.x] = bestTrainIdx1;
        s_imgIdx[threadIdx.x]   = bestImgIdx1;

        __syncthreads();

        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                float val = s_distance[i];

                if (val < myBestDistance1)
                {
                    myBestDistance2 = myBestDistance1;
                    myBestTrainIdx2 = myBestTrainIdx1;
                    myBestImgIdx2   = myBestImgIdx1;

                    myBestDistance1 = val;
                    myBestTrainIdx1 = s_trainIdx[i];
                    myBestImgIdx1   = s_imgIdx[i];
                }
                else if (val < myBestDistance2)
                {
                    myBestDistance2 = val;
                    myBestTrainIdx2 = s_trainIdx[i];
                    myBestImgIdx2   = s_imgIdx[i];
                }
            }
        }

        __syncthreads();

        s_distance[threadIdx.x] = bestDistance2;
        s_trainIdx[threadIdx.x] = bestTrainIdx2;
        s_imgIdx[threadIdx.x]   = bestImgIdx2;

        __syncthreads();

        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                float val = s_distance[i];

                if (val < myBestDistance2)
                {
                    myBestDistance2 = val;
                    myBestTrainIdx2 = s_trainIdx[i];
                    myBestImgIdx2   = s_imgIdx[i];
                }
            }
        }

        bestDistance1 = myBestDistance1;
        bestDistance2 = myBestDistance2;

        bestTrainIdx1 = myBestTrainIdx1;
        bestTrainIdx2 = myBestTrainIdx2;

        bestImgIdx1 = myBestImgIdx1;
        bestImgIdx2 = myBestImgIdx2;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Match Unrolled Cached

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename T, typename U> 
    __device__ void loadQueryToSmem(int queryIdx, const DevMem2D_<T>& query, U* s_query)
    {
        #pragma unroll
        for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
        {
            const int loadX = threadIdx.x + i * BLOCK_SIZE;
            s_query[threadIdx.y * MAX_DESC_LEN + loadX] = loadX < query.cols ? query.ptr(min(queryIdx, query.rows - 1))[loadX] : 0;
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    __device__ void loopUnrolledCached(int queryIdx, const DevMem2D_<T>& query, int imgIdx, const DevMem2D_<T>& train, const Mask& mask, 
                                       typename Dist::value_type* s_query, typename Dist::value_type* s_train, 
                                       float& bestDistance1, float& bestDistance2, 
                                       int& bestTrainIdx1, int& bestTrainIdx2, 
                                       int& bestImgIdx1, int& bestImgIdx2)
    {
        for (int t = 0, endt = (train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE; t < endt; ++t)
        {
            Dist dist;

            #pragma unroll
            for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
            {
                const int loadX = threadIdx.x + i * BLOCK_SIZE;

                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = loadX < train.cols ? train.ptr(min(t * BLOCK_SIZE + threadIdx.y, train.rows - 1))[loadX] : 0;

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; ++j)
                    dist.reduceIter(s_query[threadIdx.y * MAX_DESC_LEN + i * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                __syncthreads();
            }

            typename Dist::result_type distVal = dist;

            const int trainIdx = t * BLOCK_SIZE + threadIdx.x;

            if (queryIdx < query.rows && trainIdx < train.rows && mask(queryIdx, trainIdx))
            {
                if (distVal < bestDistance1)
                {
                    bestImgIdx2   = bestImgIdx1;
                    bestDistance2 = bestDistance1;
                    bestTrainIdx2 = bestTrainIdx1;

                    bestImgIdx1   = imgIdx;
                    bestDistance1 = distVal;
                    bestTrainIdx1 = trainIdx;
                }
                else if (distVal < bestDistance2)
                {
                    bestImgIdx2   = imgIdx;
                    bestDistance2 = distVal;
                    bestTrainIdx2 = trainIdx;
                }
            }
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    __global__ void matchUnrolledCached(const DevMem2D_<T> query, const DevMem2D_<T> train, const Mask mask, int2* bestTrainIdx, float2* bestDistance)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * MAX_DESC_LEN);

        loadQueryToSmem<BLOCK_SIZE, MAX_DESC_LEN>(queryIdx, query, s_query);

        float myBestDistance1 = numeric_limits<float>::max();
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;

        loopUnrolledCached<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, 0, train, mask, s_query, s_train, myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestTrainIdx1, myBestTrainIdx2);

        __syncthreads();

        float* s_distance = (float*)(smem);
        int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        findBestMatch<BLOCK_SIZE>(myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, s_distance, s_trainIdx);

        if (queryIdx < query.rows && threadIdx.x == 0)
        {
            bestTrainIdx[queryIdx] = make_int2(myBestTrainIdx1, myBestTrainIdx2);
            bestDistance[queryIdx] = make_float2(myBestDistance1, myBestDistance2);
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    void matchUnrolledCached(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
                             const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, 
                             cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (BLOCK_SIZE * (MAX_DESC_LEN >= BLOCK_SIZE ? MAX_DESC_LEN : BLOCK_SIZE) + BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        matchUnrolledCached<BLOCK_SIZE, MAX_DESC_LEN, Dist><<<grid, block, smemSize, stream>>>(query, train, mask, trainIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    __global__ void matchUnrolledCached(const DevMem2D_<T> query, const DevMem2D_<T>* trains, int n, const Mask mask, int2* bestTrainIdx, int2* bestImgIdx, float2* bestDistance)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * MAX_DESC_LEN);

        loadQueryToSmem<BLOCK_SIZE, MAX_DESC_LEN>(queryIdx, query, s_query);

        float myBestDistance1 = numeric_limits<float>::max();
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;
        int myBestImgIdx1 = -1;
        int myBestImgIdx2 = -1;

        Mask m = mask;

        for (int imgIdx = 0; imgIdx < n; ++imgIdx)
        {
            const DevMem2D_<T> train = trains[imgIdx];
            m.next();
            loopUnrolledCached<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, imgIdx, train, m, s_query, s_train, myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestImgIdx1, myBestImgIdx2);
        }

        __syncthreads();

        float* s_distance = (float*)(smem);
        int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);
        int* s_imgIdx = (int*)(smem + 2 * BLOCK_SIZE * BLOCK_SIZE);

        findBestMatch<BLOCK_SIZE>(myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestImgIdx1, myBestImgIdx2, s_distance, s_trainIdx, s_imgIdx);

        if (queryIdx < query.rows && threadIdx.x == 0)
        {
            bestTrainIdx[queryIdx] = make_int2(myBestTrainIdx1, myBestTrainIdx2);
            bestImgIdx[queryIdx] = make_int2(myBestImgIdx1, myBestImgIdx2);
            bestDistance[queryIdx] = make_float2(myBestDistance1, myBestDistance2);
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    void matchUnrolledCached(const DevMem2D_<T>& query, const DevMem2D_<T>* trains, int n, const Mask& mask, 
                             const DevMem2D_<int2>& trainIdx, const DevMem2D_<int2>& imgIdx, const DevMem2D_<float2>& distance, 
                             cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (BLOCK_SIZE * (MAX_DESC_LEN >= 2 * BLOCK_SIZE ? MAX_DESC_LEN : 2 * BLOCK_SIZE) + BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        matchUnrolledCached<BLOCK_SIZE, MAX_DESC_LEN, Dist><<<grid, block, smemSize, stream>>>(query, trains, n, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Match Unrolled

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    __device__ void loopUnrolled(int queryIdx, const DevMem2D_<T>& query, int imgIdx, const DevMem2D_<T>& train, const Mask& mask, 
                                 typename Dist::value_type* s_query, typename Dist::value_type* s_train, 
                                 float& bestDistance1, float& bestDistance2, 
                                 int& bestTrainIdx1, int& bestTrainIdx2, 
                                 int& bestImgIdx1, int& bestImgIdx2)
    {
        for (int t = 0, endt = (train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE; t < endt; ++t)
        {
            Dist dist;

            #pragma unroll
            for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
            {
                const int loadX = threadIdx.x + i * BLOCK_SIZE;

                if (loadX < query.cols)
                {
                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = query.ptr(min(queryIdx, query.rows - 1))[loadX];
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = train.ptr(min(t * BLOCK_SIZE + threadIdx.y, train.rows - 1))[loadX];
                }
                else
                {                
                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;
                }

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; ++j)
                    dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                __syncthreads();
            }

            typename Dist::result_type distVal = dist;

            const int trainIdx = t * BLOCK_SIZE + threadIdx.x;

            if (queryIdx < query.rows && trainIdx < train.rows && mask(queryIdx, trainIdx))
            {
                if (distVal < bestDistance1)
                {
                    bestImgIdx2   = bestImgIdx1;
                    bestDistance2 = bestDistance1;
                    bestTrainIdx2 = bestTrainIdx1;

                    bestImgIdx1   = imgIdx;
                    bestDistance1 = distVal;
                    bestTrainIdx1 = trainIdx;
                }
                else if (distVal < bestDistance2)
                {
                    bestImgIdx2   = imgIdx;
                    bestDistance2 = distVal;
                    bestTrainIdx2 = trainIdx;
                }
            }
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    __global__ void matchUnrolled(const DevMem2D_<T> query, const DevMem2D_<T> train, const Mask mask, int2* bestTrainIdx, float2* bestDistance)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        float myBestDistance1 = numeric_limits<float>::max();
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;

        loopUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, 0, train, mask, s_query, s_train, myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestTrainIdx1, myBestTrainIdx2);

        __syncthreads();

        float* s_distance = (float*)(smem);
        int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        findBestMatch<BLOCK_SIZE>(myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, s_distance, s_trainIdx);

        if (queryIdx < query.rows && threadIdx.x == 0)
        {
            bestTrainIdx[queryIdx] = make_int2(myBestTrainIdx1, myBestTrainIdx2);
            bestDistance[queryIdx] = make_float2(myBestDistance1, myBestDistance2);
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    void matchUnrolled(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
                       const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, 
                       cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        matchUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist><<<grid, block, smemSize, stream>>>(query, train, mask, trainIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    __global__ void matchUnrolled(const DevMem2D_<T> query, const DevMem2D_<T>* trains, int n, const Mask mask, int2* bestTrainIdx, int2* bestImgIdx, float2* bestDistance)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        float myBestDistance1 = numeric_limits<float>::max();
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;
        int myBestImgIdx1 = -1;
        int myBestImgIdx2 = -1;

        Mask m = mask;

        for (int imgIdx = 0; imgIdx < n; ++imgIdx)
        {
            const DevMem2D_<T> train = trains[imgIdx];
            m.next();
            loopUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, imgIdx, train, m, s_query, s_train, myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestImgIdx1, myBestImgIdx2);
        }

        __syncthreads();

        float* s_distance = (float*)(smem);
        int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);
        int* s_imgIdx = (int*)(smem + 2 * BLOCK_SIZE * BLOCK_SIZE);

        findBestMatch<BLOCK_SIZE>(myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestImgIdx1, myBestImgIdx2, s_distance, s_trainIdx, s_imgIdx);

        if (queryIdx < query.rows && threadIdx.x == 0)
        {
            bestTrainIdx[queryIdx] = make_int2(myBestTrainIdx1, myBestTrainIdx2);
            bestImgIdx[queryIdx] = make_int2(myBestImgIdx1, myBestImgIdx2);
            bestDistance[queryIdx] = make_float2(myBestDistance1, myBestDistance2);
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    void matchUnrolled(const DevMem2D_<T>& query, const DevMem2D_<T>* trains, int n, const Mask& mask, 
                       const DevMem2D_<int2>& trainIdx, const DevMem2D_<int2>& imgIdx, const DevMem2D_<float2>& distance, 
                       cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (3 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        matchUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist><<<grid, block, smemSize, stream>>>(query, trains, n, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Match

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask> 
    __device__ void loop(int queryIdx, const DevMem2D_<T>& query, int imgIdx, const DevMem2D_<T>& train, const Mask& mask, 
                         typename Dist::value_type* s_query, typename Dist::value_type* s_train, 
                         float& bestDistance1, float& bestDistance2, 
                         int& bestTrainIdx1, int& bestTrainIdx2, 
                         int& bestImgIdx1, int& bestImgIdx2)
    {
        for (int t = 0, endt = (train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE; t < endt; ++t)
        {
            Dist dist;

            for (int i = 0, endi = (query.cols + BLOCK_SIZE - 1) / BLOCK_SIZE; i < endi; ++i)
            {
                const int loadX = threadIdx.x + i * BLOCK_SIZE;

                if (loadX < query.cols)
                {
                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = query.ptr(min(queryIdx, query.rows - 1))[loadX];
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = train.ptr(min(t * BLOCK_SIZE + threadIdx.y, train.rows - 1))[loadX];
                }
                else
                {                
                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;
                }

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; ++j)
                    dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                __syncthreads();
            }

            typename Dist::result_type distVal = dist;

            const int trainIdx = t * BLOCK_SIZE + threadIdx.x;

            if (queryIdx < query.rows && trainIdx < train.rows && mask(queryIdx, trainIdx))
            {
                if (distVal < bestDistance1)
                {
                    bestImgIdx2   = bestImgIdx1;
                    bestDistance2 = bestDistance1;
                    bestTrainIdx2 = bestTrainIdx1;

                    bestImgIdx1   = imgIdx;
                    bestDistance1 = distVal;
                    bestTrainIdx1 = trainIdx;
                }
                else if (distVal < bestDistance2)
                {
                    bestImgIdx2   = imgIdx;
                    bestDistance2 = distVal;
                    bestTrainIdx2 = trainIdx;
                }
            }
        }
    }

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask> 
    __global__ void match(const DevMem2D_<T> query, const DevMem2D_<T> train, const Mask mask, int2* bestTrainIdx, float2* bestDistance)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        float myBestDistance1 = numeric_limits<float>::max();
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;

        loop<BLOCK_SIZE, Dist>(queryIdx, query, 0, train, mask, s_query, s_train, myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestTrainIdx1, myBestTrainIdx2);

        __syncthreads();

        float* s_distance = (float*)(smem);
        int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        findBestMatch<BLOCK_SIZE>(myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, s_distance, s_trainIdx);

        if (queryIdx < query.rows && threadIdx.x == 0)
        {
            bestTrainIdx[queryIdx] = make_int2(myBestTrainIdx1, myBestTrainIdx2);
            bestDistance[queryIdx] = make_float2(myBestDistance1, myBestDistance2);
        }
    }

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask> 
    void match(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
               const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, 
               cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        match<BLOCK_SIZE, Dist><<<grid, block, smemSize, stream>>>(query, train, mask, trainIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask> 
    __global__ void match(const DevMem2D_<T> query, const DevMem2D_<T>* trains, int n, const Mask mask, int2* bestTrainIdx, int2* bestImgIdx, float2* bestDistance)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        float myBestDistance1 = numeric_limits<float>::max();
        float myBestDistance2 = numeric_limits<float>::max();
        int myBestTrainIdx1 = -1;
        int myBestTrainIdx2 = -1;
        int myBestImgIdx1 = -1;
        int myBestImgIdx2 = -1;

        Mask m = mask;

        for (int imgIdx = 0; imgIdx < n; ++imgIdx)
        {
            const DevMem2D_<T> train = trains[imgIdx];
            m.next();
            loop<BLOCK_SIZE, Dist>(queryIdx, query, imgIdx, train, m, s_query, s_train, myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestImgIdx1, myBestImgIdx2);
        }

        __syncthreads();

        float* s_distance = (float*)(smem);
        int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);
        int* s_imgIdx = (int*)(smem + 2 * BLOCK_SIZE * BLOCK_SIZE);

        findBestMatch<BLOCK_SIZE>(myBestDistance1, myBestDistance2, myBestTrainIdx1, myBestTrainIdx2, myBestImgIdx1, myBestImgIdx2, s_distance, s_trainIdx, s_imgIdx);

        if (queryIdx < query.rows && threadIdx.x == 0)
        {
            bestTrainIdx[queryIdx] = make_int2(myBestTrainIdx1, myBestTrainIdx2);
            bestImgIdx[queryIdx] = make_int2(myBestImgIdx1, myBestImgIdx2);
            bestDistance[queryIdx] = make_float2(myBestDistance1, myBestDistance2);
        }
    }

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask> 
    void match(const DevMem2D_<T>& query, const DevMem2D_<T>* trains, int n, const Mask& mask, 
               const DevMem2D_<int2>& trainIdx, const DevMem2D_<int2>& imgIdx, const DevMem2D_<float2>& distance, 
               cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (3 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        match<BLOCK_SIZE, Dist><<<grid, block, smemSize, stream>>>(query, trains, n, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // knnMatch 2 dispatcher

    template <typename Dist, typename T, typename Mask> 
    void match2Dispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
                          const DevMem2Db& trainIdx, const DevMem2Db& distance, 
                          int cc, cudaStream_t stream)
    {
        if (query.cols <= 64)
        {
            matchUnrolledCached<16, 64, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        else if (query.cols <= 128)
        {
            matchUnrolledCached<16, 128, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        /*else if (query.cols <= 256)
        {
            matchUnrolled<16, 256, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        else if (query.cols <= 512)
        {            
            matchUnrolled<16, 512, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        else if (query.cols <= 1024)
        {            
            matchUnrolled<16, 1024, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }*/
        else
        {
            match<16, Dist>(query, train, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
    }

    template <typename Dist, typename T, typename Mask> 
    void match2Dispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>* trains, int n, const Mask& mask, 
                          const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
                          int cc, cudaStream_t stream)
    {
        if (query.cols <= 64)
        {
            matchUnrolledCached<16, 64, Dist>(query, trains, n, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<int2> >(imgIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        else if (query.cols <= 128)
        {
            matchUnrolledCached<16, 128, Dist>(query, trains, n, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<int2> >(imgIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        /*else if (query.cols <= 256)
        {
            matchUnrolled<16, 256, Dist>(query, trains, n, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<int2> >(imgIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        else if (query.cols <= 512)
        {            
            matchUnrolled<16, 512, Dist>(query, trains, n, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<int2> >(imgIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
        else if (query.cols <= 1024)
        {            
            matchUnrolled<16, 1024, Dist>(query, trains, n, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<int2> >(imgIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }*/
        else
        {
            match<16, Dist>(query, trains, n, mask, static_cast< DevMem2D_<int2> >(trainIdx), static_cast< DevMem2D_<int2> >(imgIdx), static_cast< DevMem2D_<float2> > (distance), stream);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
    __global__ void calcDistanceUnrolled(const DevMem2D_<T> query, const DevMem2D_<T> train, const Mask mask, PtrStepf allDist)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        const int trainIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        Dist dist;

        #pragma unroll
        for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
        {
            const int loadX = threadIdx.x + i * BLOCK_SIZE;

            if (loadX < query.cols)
            {
                s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = query.ptr(min(queryIdx, query.rows - 1))[loadX];
                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = train.ptr(min(blockIdx.x * BLOCK_SIZE + threadIdx.y, train.rows - 1))[loadX];
            }
            else
            {                
                s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;
            }

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j)
                dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

            __syncthreads();
        }

        if (queryIdx < query.rows && trainIdx < train.rows)
        {
            float distVal = numeric_limits<float>::max();

            if (mask(queryIdx, trainIdx))
                distVal = (typename Dist::result_type)dist;

            allDist.ptr(queryIdx)[trainIdx] = distVal;
        }
    }

    template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask> 
    void calcDistanceUnrolled(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, const DevMem2Df& allDist, cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(train.rows, BLOCK_SIZE), divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        calcDistanceUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist><<<grid, block, smemSize, stream>>>(query, train, mask, allDist);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask>
    __global__ void calcDistance(const DevMem2D_<T> query, const DevMem2D_<T> train, const Mask mask, PtrStepf allDist)
    {
        extern __shared__ int smem[];

        const int queryIdx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        const int trainIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
        typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

        Dist dist;

        for (int i = 0, endi = (query.cols + BLOCK_SIZE - 1) / BLOCK_SIZE; i < endi; ++i)
        {
            const int loadX = threadIdx.x + i * BLOCK_SIZE;

            if (loadX < query.cols)
            {
                s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = query.ptr(min(queryIdx, query.rows - 1))[loadX];
                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = train.ptr(min(blockIdx.x * BLOCK_SIZE + threadIdx.y, train.rows - 1))[loadX];
            }
            else
            {                
                s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;
            }

            __syncthreads();

            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j)
                dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

            __syncthreads();
        }

        if (queryIdx < query.rows && trainIdx < train.rows)
        {
            float distVal = numeric_limits<float>::max();

            if (mask(queryIdx, trainIdx))
                distVal = (typename Dist::result_type)dist;

            allDist.ptr(queryIdx)[trainIdx] = distVal;
        }
    }

    template <int BLOCK_SIZE, typename Dist, typename T, typename Mask> 
    void calcDistance(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, const DevMem2Df& allDist, cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(divUp(train.rows, BLOCK_SIZE), divUp(query.rows, BLOCK_SIZE));

        const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

        calcDistance<BLOCK_SIZE, Dist><<<grid, block, smemSize, stream>>>(query, train, mask, allDist);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Calc Distance dispatcher

    template <typename Dist, typename T, typename Mask> 
    void calcDistanceDispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
                                const DevMem2Df& allDist, 
                                int cc, cudaStream_t stream)
    {
        if (query.cols <= 64)
        {
            calcDistanceUnrolled<16, 64, Dist>(query, train, mask, allDist, stream);
        }
        else if (query.cols <= 128)
        {
            calcDistanceUnrolled<16, 128, Dist>(query, train, mask, allDist, stream);
        }
        /*else if (query.cols <= 256)
        {
            calcDistanceUnrolled<16, 256, Dist>(query, train, mask, allDist, stream);
        }
        else if (query.cols <= 512)
        {            
            calcDistanceUnrolled<16, 512, Dist>(query, train, mask, allDist, stream);
        }
        else if (query.cols <= 1024)
        {            
            calcDistanceUnrolled<16, 1024, Dist>(query, train, mask, allDist, stream);
        }*/
        else
        {
            calcDistance<16, Dist>(query, train, mask, allDist, stream);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // find knn match kernel

    template <int BLOCK_SIZE> 
    __global__ void findBestMatch(DevMem2Df allDist, int i, PtrStepi trainIdx, PtrStepf distance)
    {
        const int SMEM_SIZE = BLOCK_SIZE > 64 ? BLOCK_SIZE : 64;
        __shared__ float s_dist[SMEM_SIZE];
        __shared__ int s_trainIdx[SMEM_SIZE];

        const int queryIdx = blockIdx.x;

        float* allDistRow = allDist.ptr(queryIdx);

        float dist = numeric_limits<float>::max();
        int bestIdx = -1;
        
        for (int i = threadIdx.x; i < allDist.cols; i += BLOCK_SIZE)
        {
            float reg = allDistRow[i];
            if (reg < dist)
            {
                dist = reg;
                bestIdx = i;
            }
        }

        s_dist[threadIdx.x] = dist;
        s_trainIdx[threadIdx.x] = bestIdx;
        __syncthreads();

        reducePredVal<BLOCK_SIZE>(s_dist, dist, s_trainIdx, bestIdx, threadIdx.x, less<volatile float>());

        if (threadIdx.x == 0)
        {
            if (dist < numeric_limits<float>::max())
            {
                allDistRow[bestIdx] = numeric_limits<float>::max();
                trainIdx.ptr(queryIdx)[i] = bestIdx;
                distance.ptr(queryIdx)[i] = dist;
            }
        }
    }

    template <int BLOCK_SIZE> 
    void findKnnMatch(int k, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, cudaStream_t stream)
    {
        const dim3 block(BLOCK_SIZE, 1, 1);
        const dim3 grid(trainIdx.rows, 1, 1);

        for (int i = 0; i < k; ++i)
        {
            findBestMatch<BLOCK_SIZE><<<grid, block, 0, stream>>>(allDist, i, trainIdx, distance);
            cudaSafeCall( cudaGetLastError() );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    void findKnnMatchDispatcher(int k, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream)
    {
        findKnnMatch<256>(k, static_cast<DevMem2Di>(trainIdx), static_cast<DevMem2Df>(distance), allDist, stream);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // knn match Dispatcher

    template <typename Dist, typename T, typename Mask>
    void matchDispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, int k, const Mask& mask, 
        const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
        int cc, cudaStream_t stream)
    {
        if (k == 2)
        {
            match2Dispatcher<Dist>(query, train, mask, trainIdx, distance, cc, stream);
        }
        else
        {
            calcDistanceDispatcher<Dist>(query, train, mask, allDist, cc, stream);
            findKnnMatchDispatcher(k, trainIdx, distance, allDist, cc, stream);
        }
    }     
    
    ///////////////////////////////////////////////////////////////////////////////
    // knn match caller

    template <typename T> void matchL1_gpu(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask, 
        const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
        int cc, cudaStream_t stream)
    {
        if (mask.data)
            matchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, SingleMask(mask), trainIdx, distance, allDist, cc, stream);
        else
            matchDispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, WithOutMask(), trainIdx, distance, allDist, cc, stream);
    }

    template void matchL1_gpu<uchar >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchL1_gpu<schar >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchL1_gpu<ushort>(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchL1_gpu<short >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchL1_gpu<int   >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchL1_gpu<float >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);

    template <typename T> void matchL2_gpu(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask, 
        const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist,
        int cc, cudaStream_t stream)
    {
        if (mask.data)
            matchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, SingleMask(mask), trainIdx, distance, allDist, cc, stream);
        else
            matchDispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, WithOutMask(), trainIdx, distance, allDist, cc, stream);
    }

    //template void matchL2_gpu<uchar >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchL2_gpu<schar >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchL2_gpu<ushort>(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchL2_gpu<short >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchL2_gpu<int   >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchL2_gpu<float >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);

    template <typename T> void matchHamming_gpu(const DevMem2Db& query, const DevMem2Db& train, int k, const DevMem2Db& mask,
        const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, 
        int cc, cudaStream_t stream)
    {
        if (mask.data)
            matchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, SingleMask(mask), trainIdx, distance, allDist, cc, stream);
        else
            matchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, WithOutMask(), trainIdx, distance, allDist, cc, stream);
    }

    template void matchHamming_gpu<uchar >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchHamming_gpu<schar >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchHamming_gpu<ushort>(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    //template void matchHamming_gpu<short >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);
    template void matchHamming_gpu<int   >(const DevMem2Db& queryDescs, const DevMem2Db& trainDescs, int k, const DevMem2Db& mask, const DevMem2Db& trainIdx, const DevMem2Db& distance, const DevMem2Df& allDist, int cc, cudaStream_t stream);

    template <typename T> void match2L1_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
        const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
        int cc, cudaStream_t stream)
    {
        if (masks.data)
            match2Dispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), (const DevMem2D_<T>*)trains.ptr(), trains.cols, MaskCollection(masks.data), trainIdx, imgIdx, distance, cc, stream);
        else
            match2Dispatcher< L1Dist<T> >(static_cast< DevMem2D_<T> >(query), (const DevMem2D_<T>*)trains.ptr(), trains.cols, WithOutMask(), trainIdx, imgIdx, distance, cc, stream);
    }
    
    template void match2L1_gpu<uchar >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2L1_gpu<schar >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2L1_gpu<ushort>(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2L1_gpu<short >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2L1_gpu<int   >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2L1_gpu<float >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);

    template <typename T> void match2L2_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
        const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
        int cc, cudaStream_t stream)
    {
        if (masks.data)
            match2Dispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), (const DevMem2D_<T>*)trains.ptr(), trains.cols, MaskCollection(masks.data), trainIdx, imgIdx, distance, cc, stream);
        else
            match2Dispatcher<L2Dist>(static_cast< DevMem2D_<T> >(query), (const DevMem2D_<T>*)trains.ptr(), trains.cols, WithOutMask(), trainIdx, imgIdx, distance, cc, stream);
    }
    
    //template void match2L2_gpu<uchar >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2L2_gpu<schar >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2L2_gpu<ushort>(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2L2_gpu<short >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2L2_gpu<int   >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Di& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2L2_gpu<float >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    
    template <typename T> void match2Hamming_gpu(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, 
        const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, 
        int cc, cudaStream_t stream)
    {
        if (masks.data)
            match2Dispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), (const DevMem2D_<T>*)trains.ptr(), trains.cols, MaskCollection(masks.data), trainIdx, imgIdx, distance, cc, stream);
        else
            match2Dispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), (const DevMem2D_<T>*)trains.ptr(), trains.cols, WithOutMask(), trainIdx, imgIdx, distance, cc, stream);
    }
    
    template void match2Hamming_gpu<uchar >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2Hamming_gpu<schar >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2Hamming_gpu<ushort>(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    //template void match2Hamming_gpu<short >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
    template void match2Hamming_gpu<int   >(const DevMem2Db& query, const DevMem2Db& trains, const DevMem2D_<PtrStepb>& masks, const DevMem2Db& trainIdx, const DevMem2Db& imgIdx, const DevMem2Db& distance, int cc, cudaStream_t stream);
}}}
