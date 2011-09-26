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
    //template void knnMatchL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
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

    //template void knnMatchL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    //template void knnMatchL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    //template void knnMatchL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    //template void knnMatchL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    //template void knnMatchL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);

    template <typename T> void knnMatchHamming_gpu(const DevMem2D& query, const DevMem2D& train, int k, const DevMem2D& mask,
        const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, 
        int cc, cudaStream_t stream)
    {
        knnMatchDispatcher<HammingDist>(static_cast< DevMem2D_<T> >(query), static_cast< DevMem2D_<T> >(train), k, mask, trainIdx, distance, allDist, cc, stream);
    }

    template void knnMatchHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    //template void knnMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    //template void knnMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
    template void knnMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int k, const DevMem2D& mask, const DevMem2D& trainIdx, const DevMem2D& distance, const DevMem2D& allDist, int cc, cudaStream_t stream);
}}}
