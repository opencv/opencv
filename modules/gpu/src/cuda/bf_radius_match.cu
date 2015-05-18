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
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/vec_distance.hpp"
#include "opencv2/gpu/device/datamov_utils.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace bf_radius_match
    {
        ///////////////////////////////////////////////////////////////////////////////
        // Match Unrolled

        template <int BLOCK_SIZE, int MAX_DESC_LEN, bool SAVE_IMG_IDX, typename Dist, typename T, typename Mask>
        __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE)
        __global__ void matchUnrolled(const PtrStepSz<T> query, int imgIdx, const PtrStepSz<T> train, float maxDistance, const Mask mask,
            PtrStepi bestTrainIdx, PtrStepi bestImgIdx, PtrStepf bestDistance, unsigned int* nMatches, int maxCount)
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

                s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;

                if (loadX < query.cols)
                {
                    T val;

                    ForceGlob<T>::Load(query.ptr(::min(queryIdx, query.rows - 1)), loadX, val);
                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = val;

                    ForceGlob<T>::Load(train.ptr(::min(blockIdx.x * BLOCK_SIZE + threadIdx.y, train.rows - 1)), loadX, val);
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = val;
                }

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; ++j)
                    dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                __syncthreads();
            }

            float distVal = (typename Dist::result_type)dist;

            if (queryIdx < query.rows && trainIdx < train.rows && mask(queryIdx, trainIdx) && distVal < maxDistance)
            {
                unsigned int ind = atomicInc(nMatches + queryIdx, (unsigned int) -1);
                if (ind < maxCount)
                {
                    bestTrainIdx.ptr(queryIdx)[ind] = trainIdx;
                    if (SAVE_IMG_IDX) bestImgIdx.ptr(queryIdx)[ind] = imgIdx;
                    bestDistance.ptr(queryIdx)[ind] = distVal;
                }
            }
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T, typename Mask>
        void matchUnrolled(const PtrStepSz<T>& query, const PtrStepSz<T>& train, float maxDistance, const Mask& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream)
        {
            const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            const dim3 grid(divUp(train.rows, BLOCK_SIZE), divUp(query.rows, BLOCK_SIZE));

            const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

            matchUnrolled<BLOCK_SIZE, MAX_DESC_LEN, false, Dist><<<grid, block, smemSize, stream>>>(query, 0, train, maxDistance, mask,
                trainIdx, PtrStepi(), distance, nMatches.data, trainIdx.cols);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <int BLOCK_SIZE, int MAX_DESC_LEN, typename Dist, typename T>
        void matchUnrolled(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

            const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

            for (int i = 0; i < n; ++i)
            {
                const PtrStepSz<T> train = trains[i];

                const dim3 grid(divUp(train.rows, BLOCK_SIZE), divUp(query.rows, BLOCK_SIZE));

                if (masks != 0 && masks[i].data)
                {
                    matchUnrolled<BLOCK_SIZE, MAX_DESC_LEN, true, Dist><<<grid, block, smemSize, stream>>>(query, i, train, maxDistance, SingleMask(masks[i]),
                        trainIdx, imgIdx, distance, nMatches.data, trainIdx.cols);
                }
                else
                {
                    matchUnrolled<BLOCK_SIZE, MAX_DESC_LEN, true, Dist><<<grid, block, smemSize, stream>>>(query, i, train, maxDistance, WithOutMask(),
                        trainIdx, imgIdx, distance, nMatches.data, trainIdx.cols);
                }
                cudaSafeCall( cudaGetLastError() );
            }

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Match

        template <int BLOCK_SIZE, bool SAVE_IMG_IDX, typename Dist, typename T, typename Mask>
        __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE)
        __global__ void match(const PtrStepSz<T> query, int imgIdx, const PtrStepSz<T> train, float maxDistance, const Mask mask,
            PtrStepi bestTrainIdx, PtrStepi bestImgIdx, PtrStepf bestDistance, unsigned int* nMatches, int maxCount)
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

                s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
                s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;

                if (loadX < query.cols)
                {
                    T val;

                    ForceGlob<T>::Load(query.ptr(::min(queryIdx, query.rows - 1)), loadX, val);
                    s_query[threadIdx.y * BLOCK_SIZE + threadIdx.x] = val;

                    ForceGlob<T>::Load(train.ptr(::min(blockIdx.x * BLOCK_SIZE + threadIdx.y, train.rows - 1)), loadX, val);
                    s_train[threadIdx.x * BLOCK_SIZE + threadIdx.y] = val;
                }

                __syncthreads();

                #pragma unroll
                for (int j = 0; j < BLOCK_SIZE; ++j)
                    dist.reduceIter(s_query[threadIdx.y * BLOCK_SIZE + j], s_train[j * BLOCK_SIZE + threadIdx.x]);

                __syncthreads();
            }

            float distVal = (typename Dist::result_type)dist;

            if (queryIdx < query.rows && trainIdx < train.rows && mask(queryIdx, trainIdx) && distVal < maxDistance)
            {
                unsigned int ind = atomicInc(nMatches + queryIdx, (unsigned int) -1);
                if (ind < maxCount)
                {
                    bestTrainIdx.ptr(queryIdx)[ind] = trainIdx;
                    if (SAVE_IMG_IDX) bestImgIdx.ptr(queryIdx)[ind] = imgIdx;
                    bestDistance.ptr(queryIdx)[ind] = distVal;
                }
            }
        }

        template <int BLOCK_SIZE, typename Dist, typename T, typename Mask>
        void match(const PtrStepSz<T>& query, const PtrStepSz<T>& train, float maxDistance, const Mask& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            const dim3 grid(divUp(train.rows, BLOCK_SIZE), divUp(query.rows, BLOCK_SIZE));

            const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

            match<BLOCK_SIZE, false, Dist><<<grid, block, smemSize, stream>>>(query, 0, train, maxDistance, mask,
                trainIdx, PtrStepi(), distance, nMatches.data, trainIdx.cols);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template <int BLOCK_SIZE, typename Dist, typename T>
        void match(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

            const size_t smemSize = (2 * BLOCK_SIZE * BLOCK_SIZE) * sizeof(int);

            for (int i = 0; i < n; ++i)
            {
                const PtrStepSz<T> train = trains[i];

                const dim3 grid(divUp(train.rows, BLOCK_SIZE), divUp(query.rows, BLOCK_SIZE));

                if (masks != 0 && masks[i].data)
                {
                    match<BLOCK_SIZE, true, Dist><<<grid, block, smemSize, stream>>>(query, i, train, maxDistance, SingleMask(masks[i]),
                        trainIdx, imgIdx, distance, nMatches.data, trainIdx.cols);
                }
                else
                {
                    match<BLOCK_SIZE, true, Dist><<<grid, block, smemSize, stream>>>(query, i, train, maxDistance, WithOutMask(),
                        trainIdx, imgIdx, distance, nMatches.data, trainIdx.cols);
                }
                cudaSafeCall( cudaGetLastError() );
            }

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Match dispatcher

        template <typename Dist, typename T, typename Mask>
        void matchDispatcher(const PtrStepSz<T>& query, const PtrStepSz<T>& train, float maxDistance, const Mask& mask,
                             const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
                             cudaStream_t stream)
        {
            if (query.cols <= 64)
            {
                matchUnrolled<16, 64, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
            }
            else if (query.cols <= 128)
            {
                matchUnrolled<16, 128, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
            }
            /*else if (query.cols <= 256)
            {
                matchUnrolled<16, 256, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
            }
            else if (query.cols <= 512)
            {
                matchUnrolled<16, 512, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
            }
            else if (query.cols <= 1024)
            {
                matchUnrolled<16, 1024, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
            }*/
            else
            {
                match<16, Dist>(query, train, maxDistance, mask, trainIdx, distance, nMatches, stream);
            }
        }

        template <typename Dist, typename T>
        void matchDispatcher(const PtrStepSz<T>& query, const PtrStepSz<T>* trains, int n, float maxDistance, const PtrStepSzb* masks,
                             const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
                             cudaStream_t stream)
        {
            if (query.cols <= 64)
            {
                matchUnrolled<16, 64, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
            }
            else if (query.cols <= 128)
            {
                matchUnrolled<16, 128, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
            }
            /*else if (query.cols <= 256)
            {
                matchUnrolled<16, 256, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
            }
            else if (query.cols <= 512)
            {
                matchUnrolled<16, 512, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
            }
            else if (query.cols <= 1024)
            {
                matchUnrolled<16, 1024, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
            }*/
            else
            {
                match<16, Dist>(query, trains, n, maxDistance, masks, trainIdx, imgIdx, distance, nMatches, stream);
            }
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Radius Match caller

        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            if (mask.data)
            {
                matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), maxDistance, SingleMask(mask),
                    trainIdx, distance, nMatches,
                    stream);
            }
            else
            {
                matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), maxDistance, WithOutMask(),
                    trainIdx, distance, nMatches,
                    stream);
            }
        }

#ifndef OPENCV_TINY_GPU_MODULE
        template void matchL1_gpu<uchar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL1_gpu<schar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<ushort>(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<short >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<int   >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<float >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#endif

        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            if (mask.data)
            {
                matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), maxDistance, SingleMask(mask),
                    trainIdx, distance, nMatches,
                    stream);
            }
            else
            {
                matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), maxDistance, WithOutMask(),
                    trainIdx, distance, nMatches,
                    stream);
            }
        }

#ifndef OPENCV_TINY_GPU_MODULE
        //template void matchL2_gpu<uchar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<schar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<ushort>(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<short >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<int   >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#endif
        template void matchL2_gpu<float >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);

        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb& train, float maxDistance, const PtrStepSzb& mask,
            const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            if (mask.data)
            {
                matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), maxDistance, SingleMask(mask),
                    trainIdx, distance, nMatches,
                    stream);
            }
            else
            {
                matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), static_cast< PtrStepSz<T> >(train), maxDistance, WithOutMask(),
                    trainIdx, distance, nMatches,
                    stream);
            }
        }

        template void matchHamming_gpu<uchar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#ifndef OPENCV_TINY_GPU_MODULE
        //template void matchHamming_gpu<schar >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchHamming_gpu<ushort>(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchHamming_gpu<short >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchHamming_gpu<int   >(const PtrStepSzb& queryDescs, const PtrStepSzb& trainDescs, float maxDistance, const PtrStepSzb& mask, const PtrStepSzi& trainIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#endif

        template <typename T> void matchL1_gpu(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            matchDispatcher< L1Dist<T> >(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains, n, maxDistance, masks,
                trainIdx, imgIdx, distance, nMatches,
                stream);
        }

#ifndef OPENCV_TINY_GPU_MODULE
        template void matchL1_gpu<uchar >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL1_gpu<schar >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<ushort>(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<short >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<int   >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL1_gpu<float >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#endif

        template <typename T> void matchL2_gpu(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            matchDispatcher<L2Dist>(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains, n, maxDistance, masks,
                trainIdx, imgIdx, distance, nMatches,
                stream);
        }

#ifndef OPENCV_TINY_GPU_MODULE
        //template void matchL2_gpu<uchar >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<schar >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<ushort>(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<short >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchL2_gpu<int   >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchL2_gpu<float >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#endif

        template <typename T> void matchHamming_gpu(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks,
            const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches,
            cudaStream_t stream)
        {
            matchDispatcher<HammingDist>(static_cast< PtrStepSz<T> >(query), (const PtrStepSz<T>*)trains, n, maxDistance, masks,
                trainIdx, imgIdx, distance, nMatches,
                stream);
        }

#ifndef OPENCV_TINY_GPU_MODULE
        template void matchHamming_gpu<uchar >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchHamming_gpu<schar >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchHamming_gpu<ushort>(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        //template void matchHamming_gpu<short >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
        template void matchHamming_gpu<int   >(const PtrStepSzb& query, const PtrStepSzb* trains, int n, float maxDistance, const PtrStepSzb* masks, const PtrStepSzi& trainIdx, const PtrStepSzi& imgIdx, const PtrStepSzf& distance, const PtrStepSz<unsigned int>& nMatches, cudaStream_t stream);
#endif
    } // namespace bf_radius_match
}}} // namespace cv { namespace gpu { namespace device


#endif /* CUDA_DISABLER */
