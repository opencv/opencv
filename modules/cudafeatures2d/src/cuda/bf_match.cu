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
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/vec_distance.hpp"
#include "opencv2/core/cuda/datamov_utils.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace bf_match
    {
        ///////////////////////////////////////////////////////////////////////////////
        // Reduction

        template <int BLOCK_SIZE>
        __device__ void findBestMatch(float& bestDistance, int& bestTrainIdx, float* s_distance, int* s_trainIdx)
        {
            s_distance += threadIdx.y * BLOCK_SIZE;
            s_trainIdx += threadIdx.y * BLOCK_SIZE;

            reduceKeyVal<BLOCK_SIZE>(s_distance, bestDistance, s_trainIdx, bestTrainIdx, threadIdx.x, less<float>());
        }

        template <int BLOCK_SIZE>
        __device__ void findBestMatch(float& bestDistance, int& bestTrainIdx, int& bestImgIdx, float* s_distance, int* s_trainIdx, int* s_imgIdx)
        {
            s_distance += threadIdx.y * BLOCK_SIZE;
            s_trainIdx += threadIdx.y * BLOCK_SIZE;
            s_imgIdx   += threadIdx.y * BLOCK_SIZE;

            reduceKeyVal<BLOCK_SIZE>(s_distance, bestDistance, smem_tuple(s_trainIdx, s_imgIdx), thrust::tie(bestTrainIdx, bestImgIdx), threadIdx.x, less<float>());
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Match Unrolled Cached

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename T, typename U>
        __device__ void loadQueryToSmem(int queryIdx, const PtrStepSz<T>& query, U* s_query)
        {
            #pragma unroll
            for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
            {
                const int loadX = threadIdx.x + i * BLOCK_SIZE;
                s_query[threadIdx.y * MAX_DESC_LEN + loadX] = loadX < query.cols ? query.ptr(::min(queryIdx, query.rows - 1))[loadX] : 0;
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        __device__ void loopUnrolledCached(int queryIdx, const PtrStepSz<T>& query,volatile int imgIdx, const PtrStepSz<T>& train, const Mask& mask,
                                           typename Dist::value_type* s_query, typename Dist::value_type* s_train,
                                           float& bestDistance, int& bestTrainIdx, int& bestImgIdx)
        {
            for (int t = 0, endt = (train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE; t < endt; ++t)
            {
                Dist dist;

                #pragma unroll
                for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
                {
                    const int loadX = threadIdx.x + i * BLOCK_SIZE;

                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;

                    if (loadX < train.cols)
                    {
                        T val;

                        ForceGlob<T>::Load(train.ptr(::min(t * BLOCK_SIZE + threadIdx.y, train.rows - 1)), loadX, val);
                        s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = val;
                    }

                    __syncthreads();

                    #pragma unroll
                    for (int j = 0; j < BLOCK_SIZE; ++j)
                        dist.reduceIter(s_query[threadIdx.y * MAX_DESC_LEN + i * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                    __syncthreads();
                }

                typename Dist::result_type distVal = dist;

                const int trainIdx = t * BLOCK_SIZE + threadIdx.x;

                if (queryIdx < query.rows && trainIdx < train.rows && distVal < bestDistance && mask(queryIdx, trainIdx))
                {
                    bestImgIdx = imgIdx;
                    bestDistance = distVal;
                    bestTrainIdx = trainIdx;
                }
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        __global__ void matchUnrolledCached(const PtrStepSz<T> query, const PtrStepSz<T> train, const Mask mask, int* bestTrainIdx, float* bestDistance)
        {
            extern __shared__ int smem[];

            const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

            typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
            typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * MAX_DESC_LEN);

            loadQueryToSmem<BLOCK_SIZE, MAX_DESC_LEN>(queryIdx, query, s_query);

            float myBestDistance = numeric_limits<float>::max();
            int myBestTrainIdx = -1;

            loopUnrolledCached<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, 0, train, mask, s_query, s_train, myBestDistance, myBestTrainIdx, myBestTrainIdx);

            __syncthreads();

            float* s_distance = (float*)(smem);
            int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            findBestMatch<BLOCK_SIZE>(myBestDistance, myBestTrainIdx, s_distance, s_trainIdx);

            if (queryIdx < query.rows && threadIdx.x == 0)
            {
                bestTrainIdx[queryIdx] = myBestTrainIdx;
                bestDistance[queryIdx] = myBestDistance;
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        void matchUnrolledCached(const PtrStepSz<T>& query, const PtrStepSz<T>& train, const Mask& mask,
                                 const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
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
        __global__ void matchUnrolledCached(const PtrStepSz<T> query, const PtrStepSz<T>* trains, int n, const Mask mask,
                                            int* bestTrainIdx, int* bestImgIdx, float* bestDistance)
        {
            extern __shared__ int smem[];

            const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

            typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
            typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * MAX_DESC_LEN);

            loadQueryToSmem<BLOCK_SIZE, MAX_DESC_LEN>(queryIdx, query, s_query);

            float myBestDistance = numeric_limits<float>::max();
            int myBestTrainIdx = -1;
            int myBestImgIdx = -1;

            Mask m = mask;

            for (int imgIdx = 0; imgIdx < n; ++imgIdx)
            {
                const PtrStepSz<T> train = trains[imgIdx];
                m.next();
                loopUnrolledCached<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, imgIdx, train, m, s_query, s_train, myBestDistance, myBestTrainIdx, myBestImgIdx);
            }

            __syncthreads();

            float* s_distance = (float*)(smem);
            int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);
            int* s_imgIdx = (int*)(smem + 2 * BLOCK_SIZE * BLOCK_SIZE);

            findBestMatch<BLOCK_SIZE>(myBestDistance, myBestTrainIdx, myBestImgIdx, s_distance, s_trainIdx, s_imgIdx);

            if (queryIdx < query.rows && threadIdx.x == 0)
            {
                bestTrainIdx[queryIdx] = myBestTrainIdx;
                bestImgIdx[queryIdx] = myBestImgIdx;
                bestDistance[queryIdx] = myBestDistance;
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        void matchUnrolledCached(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, const Mask& mask,
                                 const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
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
        __device__ void loopUnrolled(int queryIdx, const PtrStepSz<T>& query,volatile int imgIdx, const PtrStepSz<T>& train, const Mask& mask,
                                     typename Dist::value_type* s_query, typename Dist::value_type* s_train,
                                     float& bestDistance, int& bestTrainIdx, int& bestImgIdx)
        {
            for (int t = 0, endt = (train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE; t < endt; ++t)
            {
                Dist dist;

                #pragma unroll
                for (int i = 0; i < MAX_DESC_LEN / BLOCK_SIZE; ++i)
                {
                    const int loadX = threadIdx.x + i * BLOCK_SIZE;

                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;

                    if (loadX < query.cols)
                    {
                        T val;

                        ForceGlob<T>::Load(query.ptr(::min(queryIdx, query.rows - 1)), loadX, val);
                        s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = val;

                        ForceGlob<T>::Load(train.ptr(::min(t * BLOCK_SIZE + threadIdx.y, train.rows - 1)), loadX, val);
                        s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = val;
                    }

                    __syncthreads();

                    #pragma unroll
                    for (int j = 0; j < BLOCK_SIZE; ++j)
                        dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                    __syncthreads();
                }

                typename Dist::result_type distVal = dist;

                const int trainIdx = t * BLOCK_SIZE + threadIdx.x;

                if (queryIdx < query.rows && trainIdx < train.rows && distVal < bestDistance && mask(queryIdx, trainIdx))
                {
                    bestImgIdx = imgIdx;
                    bestDistance = distVal;
                    bestTrainIdx = trainIdx;
                }
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        __global__ void matchUnrolled(const PtrStepSz<T> query, const PtrStepSz<T> train, const Mask mask, int* bestTrainIdx, float* bestDistance)
        {
            extern __shared__ int smem[];

            const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

            float myBestDistance = numeric_limits<float>::max();
            int myBestTrainIdx = -1;

            typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
            typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            loopUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, 0, train, mask, s_query, s_train, myBestDistance, myBestTrainIdx, myBestTrainIdx);

            __syncthreads();

            float* s_distance = (float*)(smem);
            int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            findBestMatch<BLOCK_SIZE>(myBestDistance, myBestTrainIdx, s_distance, s_trainIdx);

            if (queryIdx < query.rows && threadIdx.x == 0)
            {
                bestTrainIdx[queryIdx] = myBestTrainIdx;
                bestDistance[queryIdx] = myBestDistance;
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        void matchUnrolled(const PtrStepSz<T>& query, const PtrStepSz<T>& train, const Mask& mask,
                           const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
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
        __global__ void matchUnrolled(const PtrStepSz<T> query, const PtrStepSz<T>* trains, int n, const Mask mask,
                                      int* bestTrainIdx, int* bestImgIdx, float* bestDistance)
        {
            extern __shared__ int smem[];

            const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

            float myBestDistance = numeric_limits<float>::max();
            int myBestTrainIdx = -1;
            int myBestImgIdx = -1;

            typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
            typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            Mask m = mask;

            for (int imgIdx = 0; imgIdx < n; ++imgIdx)
            {
                const PtrStepSz<T> train = trains[imgIdx];
                m.next();
                loopUnrolled<BLOCK_SIZE, MAX_DESC_LEN, Dist>(queryIdx, query, imgIdx, train, m, s_query, s_train, myBestDistance, myBestTrainIdx, myBestImgIdx);
            }

            __syncthreads();

            float* s_distance = (float*)(smem);
            int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);
            int* s_imgIdxIdx = (int*)(smem + 2 * BLOCK_SIZE * BLOCK_SIZE);

            findBestMatch<BLOCK_SIZE>(myBestDistance, myBestTrainIdx, myBestImgIdx, s_distance, s_trainIdx, s_imgIdxIdx);

            if (queryIdx < query.rows && threadIdx.x == 0)
            {
                bestTrainIdx[queryIdx] = myBestTrainIdx;
                bestImgIdx[queryIdx] = myBestImgIdx;
                bestDistance[queryIdx] = myBestDistance;
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        void matchUnrolled(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, const Mask& mask,
                           const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
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
        __device__ void loop(int queryIdx, const PtrStepSz<T>& query, volatile int imgIdx, const PtrStepSz<T>& train, const Mask& mask,
                             typename Dist::value_type* s_query, typename Dist::value_type* s_train,
                             float& bestDistance, int& bestTrainIdx, int& bestImgIdx)
        {
            for (int t = 0, endt = (train.rows + BLOCK_SIZE - 1) / BLOCK_SIZE; t < endt; ++t)
            {
                Dist dist;

                for (int i = 0, endi = (query.cols + BLOCK_SIZE - 1) / BLOCK_SIZE; i < endi; ++i)
                {
                    const int loadX = threadIdx.x + i * BLOCK_SIZE;

                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;

                    if (loadX < query.cols)
                    {
                        T val;

                        ForceGlob<T>::Load(query.ptr(::min(queryIdx, query.rows - 1)), loadX, val);
                        s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = val;

                        ForceGlob<T>::Load(train.ptr(::min(t * BLOCK_SIZE + threadIdx.y, train.rows - 1)), loadX, val);
                        s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = val;
                    }

                    __syncthreads();

                    #pragma unroll
                    for (int j = 0; j < BLOCK_SIZE; ++j)
                        dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                    __syncthreads();
                }

                typename Dist::result_type distVal = dist;

                const int trainIdx = t * BLOCK_SIZE + threadIdx.x;

                if (queryIdx < query.rows && trainIdx < train.rows && distVal < bestDistance && mask(queryIdx, trainIdx))
                {
                    bestImgIdx = imgIdx;
                    bestDistance = distVal;
                    bestTrainIdx = trainIdx;
                }
            }
        }

        template <int BLOCK_SIZE, typename Dist, typename T, typename Mask>
        __global__ void match(const PtrStepSz<T> query, const PtrStepSz<T> train, const Mask mask, int* bestTrainIdx, float* bestDistance)
        {
            extern __shared__ int smem[];

            const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

            float myBestDistance = numeric_limits<float>::max();
            int myBestTrainIdx = -1;

            typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
            typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            loop<BLOCK_SIZE, Dist>(queryIdx, query, 0, train, mask, s_query, s_train, myBestDistance, myBestTrainIdx, myBestTrainIdx);

            __syncthreads();

            float* s_distance = (float*)(smem);
            int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            findBestMatch<BLOCK_SIZE>(myBestDistance, myBestTrainIdx, s_distance, s_trainIdx);

            if (queryIdx < query.rows && threadIdx.x == 0)
            {
                bestTrainIdx[queryIdx] = myBestTrainIdx;
                bestDistance[queryIdx] = myBestDistance;
            }
        }

        template <int BLOCK_SIZE, typename Dist, typename T, typename Mask>
        void match(const PtrStepSz<T>& query, const PtrStepSz<T>& train, const Mask& mask,
                   const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
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
        __global__ void match(const PtrStepSz<T> query, const PtrStepSz<T>* trains, int n, const Mask mask,
                              int* bestTrainIdx, int* bestImgIdx, float* bestDistance)
        {
            extern __shared__ int smem[];

            const int queryIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;

            float myBestDistance = numeric_limits<float>::max();
            int myBestTrainIdx = -1;
            int myBestImgIdx = -1;

            typename Dist::value_type* s_query = (typename Dist::value_type*)(smem);
            typename Dist::value_type* s_train = (typename Dist::value_type*)(smem + BLOCK_SIZE * BLOCK_SIZE);

            Mask m = mask;
            for (int imgIdx = 0; imgIdx < n; ++imgIdx)
            {
                const PtrStepSz<T> train = trains[imgIdx];
                m.next();
                loop<BLOCK_SIZE, Dist>(queryIdx, query, imgIdx, train, m, s_query, s_train, myBestDistance, myBestTrainIdx, myBestImgIdx);
            }

            __syncthreads();

            float* s_distance = (float*)(smem);
            int* s_trainIdx = (int*)(smem + BLOCK_SIZE * BLOCK_SIZE);
            int* s_imgIdxIdx = (int*)(smem + 2 * BLOCK_SIZE * BLOCK_SIZE);

            findBestMatch<BLOCK_SIZE>(myBestDistance, myBestTrainIdx, myBestImgIdx, s_distance, s_trainIdx, s_imgIdxIdx);

            if (queryIdx < query.rows && threadIdx.x == 0)
            {
                bestTrainIdx[queryIdx] = myBestTrainIdx;
                bestImgIdx[queryIdx] = myBestImgIdx;
                bestDistance[queryIdx] = myBestDistance;
            }
        }

        template <int BLOCK_SIZE, typename Dist, typename T, typename Mask>
        void match(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, const Mask& mask,
                   const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
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
        // Match dispatcher

        template <typename Dist, typename T, typename Mask>
        void matchDispatcher(const PtrStepSz<T>& query, const PtrStepSz<T>& train, const Mask& mask,
                             const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
                             cudaStream_t stream)
        {
            if (query.cols <= 64)
            {
                matchUnrolledCached<16, 64, Dist>(query, train, mask, trainIdx, distance, stream);
            }
            else if (query.cols <= 128)
            {
                matchUnrolledCached<16, 128, Dist>(query, train, mask, trainIdx, distance, stream);
            }
            /*else if (query.cols <= 256)
            {
                matchUnrolled<16, 256, Dist>(query, train, mask, trainIdx, distance, stream);
            }
            else if (query.cols <= 512)
            {
                matchUnrolled<16, 512, Dist>(query, train, mask, trainIdx, distance, stream);
            }
            else if (query.cols <= 1024)
            {
                matchUnrolled<16, 1024, Dist>(query, train, mask, trainIdx, distance, stream);
            }*/
            else
            {
                match<16, Dist>(query, train, mask, trainIdx, distance, stream);
            }
        }

        template <typename Dist, typename T, typename Mask>
        void matchDispatcher(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, const Mask& mask,
                             const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
                             cudaStream_t stream)
        {
            if (query.cols <= 64)
            {
                matchUnrolledCached<16, 64, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
            }
            else if (query.cols <= 128)
            {
                matchUnrolledCached<16, 128, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
            }
            /*else if (query.cols <= 256)
            {
                matchUnrolled<16, 256, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
            }
            else if (query.cols <= 512)
            {
                matchUnrolled<16, 512, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
            }
            else if (query.cols <= 1024)
            {
                matchUnrolled<16, 1024, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
            }*/
            else
            {
                match<16, Dist>(query, trains, n, mask, trainIdx, imgIdx, distance, stream);
            }
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Match caller

        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
                                               const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
                                               cudaStream_t stream)
        {
            if (mask.data)
            {
                matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), SingleMask(mask),
                    trainIdx, distance,
                    stream);
            }
            else
            {
                matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), WithOutMask(),
                    trainIdx, distance,
                    stream);
            }
        }

        template void matchL1_gpu<uchar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL1_gpu<schar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<ushort>(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<short >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<int   >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<float >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);

        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
                                               const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
                                               cudaStream_t stream)
        {
            if (mask.data)
            {
                matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), SingleMask(mask),
                    trainIdx, distance,
                    stream);
            }
            else
            {
                matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), WithOutMask(),
                    trainIdx, distance,
                    stream);
            }
        }

        //template void matchL2_gpu<uchar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<schar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<ushort>(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<short >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<int   >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL2_gpu<float >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);

        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& train, const PtrStepSzb& mask,
                                                    const PtrStepSzi& trainIdx, const PtrStepSzf& distance,
                                                    cudaStream_t stream)
        {
            if (mask.data)
            {
                matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), SingleMask(mask),
                    trainIdx, distance,
                    stream);
            }
            else
            {
                matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), WithOutMask(),
                    trainIdx, distance,
                    stream);
            }
        }

        template void matchHamming_gpu<uchar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchHamming_gpu<schar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchHamming_gpu<ushort>(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchHamming_gpu<short >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchHamming_gpu<int   >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, cudaStream_t stream);

        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
                                               const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
                                                cudaStream_t stream)
        {
            if (masks.data)
            {
                matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains.ptr(), trains.cols, MaskCollection(masks.data),
                    trainIdx, imgIdx, distance,
                    stream);
            }
            else
            {
                matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains.ptr(), trains.cols, WithOutMask(),
                    trainIdx, imgIdx, distance,
                    stream);
            }
        }

        template void matchL1_gpu<uchar >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL1_gpu<schar >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<ushort>(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<short >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<int   >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL1_gpu<float >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);

        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
                                               const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
                                               cudaStream_t stream)
        {
            if (masks.data)
            {
                matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains.ptr(), trains.cols, MaskCollection(masks.data),
                    trainIdx, imgIdx, distance,
                    stream);
            }
            else
            {
                matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains.ptr(), trains.cols, WithOutMask(),
                    trainIdx, imgIdx, distance,
                    stream);
            }
        }

        //template void matchL2_gpu<uchar >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<schar >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<ushort>(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<short >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchL2_gpu<int   >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchL2_gpu<float >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& maskCollection, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);

        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks,
                                                    const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance,
                                                    cudaStream_t stream)
        {
            if (masks.data)
            {
                matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains.ptr(), trains.cols, MaskCollection(masks.data),
                    trainIdx, imgIdx, distance,
                    stream);
            }
            else
            {
                matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains.ptr(), trains.cols, WithOutMask(),
                    trainIdx, imgIdx, distance,
                    stream);
            }
        }

        template void matchHamming_gpu<uchar >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchHamming_gpu<schar >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchHamming_gpu<ushort>(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        //template void matchHamming_gpu<short >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
        template void matchHamming_gpu<int   >(const PtrStepSzb& query, const PtrStepSzb& trains, const PtrStepSz<PtrStepb>& masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, cudaStream_t stream);
    } // namespace bf_match
}}} // namespace cv { namespace cuda { namespace cudev {


#endif /* CUDA_DISABLER */
