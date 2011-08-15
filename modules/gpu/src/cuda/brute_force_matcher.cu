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
#include "opencv2/gpu/device/datamov_utils.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace cv { namespace gpu { namespace bfmatcher
{
///////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// General funcs //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////
    // Mask strategy

    struct SingleMask
    {
        explicit SingleMask(const PtrStep& mask_) : mask(mask_) {}
        
        __device__ __forceinline__ bool operator()(int queryIdx, int trainIdx) const
        {            
            return mask.ptr(queryIdx)[trainIdx] != 0;
        }

        const PtrStep mask;
    };

    struct MaskCollection
    {
        explicit MaskCollection(PtrStep* maskCollection_) : maskCollection(maskCollection_) {}

        __device__ __forceinline__ void nextMask()
        {
            curMask = *maskCollection++;
        }
        
        __device__ __forceinline__ bool operator()(int queryIdx, int trainIdx) const
        {
            uchar val;
            return curMask.data == 0 || (ForceGlob<uchar>::Load(curMask.ptr(queryIdx), trainIdx, val), (val != 0));
        }

        const PtrStep* maskCollection;
        PtrStep curMask;
    };

    struct WithOutMask
    {
        __device__ __forceinline__ void nextMask() const
        {
        }
        __device__ __forceinline__ bool operator()(int queryIdx, int trainIdx) const
        {
            return true;
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Reduce Sum

    template <int BLOCK_DIM_X> struct SumReductor;
    template <> struct SumReductor<16>
    {
        template <typename T> static __device__ void reduce(volatile T* sdiff_row, T& mySum)
        {
            sdiff_row[threadIdx.x] = mySum;
            
            if (threadIdx.x < 8) 
            {
                sdiff_row[threadIdx.x] = mySum += sdiff_row[threadIdx.x + 8]; 
                sdiff_row[threadIdx.x] = mySum += sdiff_row[threadIdx.x + 4]; 
                sdiff_row[threadIdx.x] = mySum += sdiff_row[threadIdx.x + 2];
                sdiff_row[threadIdx.x] = mySum += sdiff_row[threadIdx.x + 1];  
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Distance

    template <typename T> struct L1Dist
    {
        typedef int ResultType;
        typedef int ValueType;

        __device__ __forceinline__ L1Dist() : mySum(0) {}

        __device__ __forceinline__ void reduceIter(int val1, int val2)
        {
            mySum = __sad(val1, val2, mySum);
        }

        template <int BLOCK_DIM_X> __device__ __forceinline__ void reduceAll(int* sdiff_row)
        {
            SumReductor<BLOCK_DIM_X>::reduce(sdiff_row, mySum);
        }

        __device__ __forceinline__ operator int() const
        {
            return mySum;
        }

        int mySum;
    };
    template <> struct L1Dist<float>
    {
        typedef float ResultType;
        typedef float ValueType;

        __device__ __forceinline__ L1Dist() : mySum(0.0f) {}

        __device__ __forceinline__ void reduceIter(float val1, float val2)
        {
            mySum += fabs(val1 - val2);
        }

        template <int BLOCK_DIM_X> __device__ __forceinline__ void reduceAll(float* sdiff_row)
        {
            SumReductor<BLOCK_DIM_X>::reduce(sdiff_row, mySum);
        }

        __device__ __forceinline__ operator float() const
        {
            return mySum;
        }

        float mySum;
    };

    struct L2Dist
    {
        typedef float ResultType;
        typedef float ValueType;

        __device__ __forceinline__ L2Dist() : mySum(0.0f) {}

        __device__ __forceinline__ void reduceIter(float val1, float val2)
        {
            float reg = val1 - val2;
            mySum += reg * reg;
        }

        template <int BLOCK_DIM_X> __device__ __forceinline__ void reduceAll(float* sdiff_row)
        {
            SumReductor<BLOCK_DIM_X>::reduce(sdiff_row, mySum);
        }

        __device__ __forceinline__ operator float() const
        {
            return sqrtf(mySum);
        }

        float mySum;
    };

    struct HammingDist
    {
        typedef int ResultType;
        typedef int ValueType;

        __device__ __forceinline__ HammingDist() : mySum(0) {}

        __device__ __forceinline__ void reduceIter(int val1, int val2)
        {
            mySum += __popc(val1 ^ val2);
        }

        template <int BLOCK_DIM_X> __device__ __forceinline__ void reduceAll(int* sdiff_row)
        {
            SumReductor<BLOCK_DIM_X>::reduce(sdiff_row, mySum);
        }

        __device__ __forceinline__ operator int() const
        {
            return mySum;
        }

        int mySum;
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    // reduceDescDiff

    template <int BLOCK_DIM_X, typename Dist, typename T> 
    __device__ void reduceDescDiff(const T* queryDescs, const T* trainDescs, int desc_len, Dist& dist, typename Dist::ResultType* sdiff_row)
    {
        for (int i = threadIdx.x; i < desc_len; i += BLOCK_DIM_X)
        {
            T trainVal;
            ForceGlob<T>::Load(trainDescs, i, trainVal);
            dist.reduceIter(queryDescs[i], trainVal);
        }

        dist.reduceAll<BLOCK_DIM_X>(sdiff_row);
    }

///////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Match //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////////
    // loadDescsVals

    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN, typename T, typename U> 
    __device__ void loadDescsVals(const T* descs, int desc_len, U* queryVals, U* smem)
    {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (tid < desc_len)
        {
            smem[tid] = descs[tid];
        }
        __syncthreads();

        #pragma unroll
        for (int i = threadIdx.x; i < MAX_DESCRIPTORS_LEN; i += BLOCK_DIM_X)
        {
            *queryVals = smem[i];
            ++queryVals;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // reduceDescDiffCached

    template <int N> struct UnrollDescDiff
    {
        template <typename Dist, typename T>
        static __device__ void calcCheck(const typename Dist::ValueType* queryVals, const T* trainDescs, int desc_len, Dist& dist, int ind)
        {
            if (ind < desc_len)
            {
                T trainVal;
                ForceGlob<T>::Load(trainDescs, ind, trainVal);
                dist.reduceIter(*queryVals, trainVal);

                ++queryVals;

                UnrollDescDiff<N - 1>::calcCheck(queryVals, trainDescs, desc_len, dist, ind + blockDim.x);
            }
        }

        template <typename Dist, typename T>
        static __device__ void calcWithoutCheck(const typename Dist::ValueType* queryVals, const T* trainDescs, Dist& dist)
        {
            T trainVal;
            ForceGlob<T>::Load(trainDescs, 0, trainVal);
            dist.reduceIter(*queryVals, trainVal);

            ++queryVals;
            trainDescs += blockDim.x;

            UnrollDescDiff<N - 1>::calcWithoutCheck(queryVals, trainDescs, dist);
        }
    };
    template <> struct UnrollDescDiff<0>
    {
        template <typename Dist, typename T>
        static __device__ __forceinline__ void calcCheck(const typename Dist::ValueType* queryVals, const T* trainDescs, int desc_len, 
            Dist& dist, int ind)
        {
        }

        template <typename Dist, typename T>
        static __device__ __forceinline__ void calcWithoutCheck(const typename Dist::ValueType* queryVals, const T* trainDescs, Dist& dist)
        {
        }
    };

    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN, bool WITH_OUT_CHECK> struct DescDiffCalculator;
    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN> 
    struct DescDiffCalculator<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, false>
    {
        template <typename Dist, typename T>
        static __device__ __forceinline__ void calc(const typename Dist::ValueType* queryVals, const T* trainDescs, int desc_len, Dist& dist)
        {
            UnrollDescDiff<MAX_DESCRIPTORS_LEN / BLOCK_DIM_X>::calcCheck(queryVals, trainDescs, desc_len, dist, threadIdx.x);
        }
    };
    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN> 
    struct DescDiffCalculator<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, true>
    {
        template <typename Dist, typename T>
        static __device__ __forceinline__ void calc(const typename Dist::ValueType* queryVals, const T* trainDescs, int desc_len, Dist& dist)
        {
            UnrollDescDiff<MAX_DESCRIPTORS_LEN / BLOCK_DIM_X>::calcWithoutCheck(queryVals, trainDescs + threadIdx.x, dist);
        }
    };

    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, typename Dist, typename T>
    __device__ __forceinline__ void reduceDescDiffCached(const typename Dist::ValueType* queryVals, const T* trainDescs, int desc_len, Dist& dist, typename Dist::ResultType* sdiff_row)
    {        
        DescDiffCalculator<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN>::calc(queryVals, trainDescs, desc_len, dist);
        
        dist.reduceAll<BLOCK_DIM_X>(sdiff_row);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // warpReduceMinIdxIdx

    template <int BLOCK_DIM_Y> struct MinIdxIdxWarpReductor;    
    template <> struct MinIdxIdxWarpReductor<16>
    {
        template <typename T> 
        static __device__ void reduce(T& myMin, int& myBestTrainIdx, int& myBestImgIdx, volatile T* smin, volatile int* strainIdx, volatile int* simgIdx)
        {
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;

            if (tid < 8)
            {
                myMin = smin[tid];
                myBestTrainIdx = strainIdx[tid];
                myBestImgIdx = simgIdx[tid];

                float reg = smin[tid + 8];
                if (reg < myMin)
                {
                    smin[tid] = myMin = reg;
                    strainIdx[tid] = myBestTrainIdx = strainIdx[tid + 8];
                    simgIdx[tid] = myBestImgIdx = simgIdx[tid + 8];
                }

                reg = smin[tid + 4];
                if (reg < myMin)
                {
                    smin[tid] = myMin = reg;
                    strainIdx[tid] = myBestTrainIdx = strainIdx[tid + 4];
                    simgIdx[tid] = myBestImgIdx = simgIdx[tid + 4];
                }
            
                reg = smin[tid + 2];
                if (reg < myMin)
                {
                    smin[tid] = myMin = reg;
                    strainIdx[tid] = myBestTrainIdx = strainIdx[tid + 2];
                    simgIdx[tid] = myBestImgIdx = simgIdx[tid + 2];
                }
            
                reg = smin[tid + 1];
                if (reg < myMin)
                {
                    smin[tid] = myMin = reg;
                    strainIdx[tid] = myBestTrainIdx = strainIdx[tid + 1];
                    simgIdx[tid] = myBestImgIdx = simgIdx[tid + 1];
                }
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // findBestMatch

    template <int BLOCK_DIM_Y, typename T>
    __device__ void findBestMatch(T& myMin, int& myBestTrainIdx, int& myBestImgIdx, T* smin, int* strainIdx, int* simgIdx)
    {
        if (threadIdx.x == 0)
        {
            smin[threadIdx.y] = myMin;
            strainIdx[threadIdx.y] = myBestTrainIdx;
            simgIdx[threadIdx.y] = myBestImgIdx;
        }
        __syncthreads();

        MinIdxIdxWarpReductor<BLOCK_DIM_Y>::reduce(myMin, myBestTrainIdx, myBestImgIdx, smin, strainIdx, simgIdx);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // ReduceDescCalculator

    template <int BLOCK_DIM_X, typename T> struct ReduceDescCalculatorSimple
    {
        __device__ __forceinline__ void prepare(const T* queryDescs_, int, void*)
        {
            queryDescs = queryDescs_;
        }

        template <typename Dist>
        __device__ __forceinline__ void calc(const T* trainDescs, int desc_len, Dist& dist, typename Dist::ResultType* sdiff_row) const
        {
            reduceDescDiff<BLOCK_DIM_X>(queryDescs, trainDescs, desc_len, dist, sdiff_row);
        }

        const T* queryDescs;
    };

    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, typename T, typename U>
    struct ReduceDescCalculatorCached
    {
        __device__ __forceinline__ void prepare(const T* queryDescs, int desc_len, U* smem)
        {
            loadDescsVals<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN>(queryDescs, desc_len, queryVals, smem);
            __syncthreads();
        }

        template <typename Dist>
        __device__ __forceinline__ void calc(const T* trainDescs, int desc_len, Dist& dist, typename Dist::ResultType* sdiff_row) const
        {
            reduceDescDiffCached<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN>(queryVals, trainDescs, desc_len, dist, sdiff_row);
        }

        U queryVals[MAX_DESCRIPTORS_LEN / BLOCK_DIM_X];
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    // matchDescs loop

    template <typename Dist, typename ReduceDescCalculator, typename T, typename Mask>
    __device__ void matchDescs(int queryIdx, int imgIdx, const DevMem2D_<T>& trainDescs_,  
        const Mask& m, const ReduceDescCalculator& reduceDescCalc,
        typename Dist::ResultType& myMin, int& myBestTrainIdx, int& myBestImgIdx, typename Dist::ResultType* sdiff_row)
    {
        for (int trainIdx = threadIdx.y; trainIdx < trainDescs_.rows; trainIdx += blockDim.y)
        {
            if (m(queryIdx, trainIdx))
            {
                const T* trainDescs = trainDescs_.ptr(trainIdx);

                Dist dist;

                reduceDescCalc.calc(trainDescs, trainDescs_.cols, dist, sdiff_row);

                if (threadIdx.x == 0)
                {
                    if (dist < myMin)
                    {
                        myMin = dist;
                        myBestTrainIdx = trainIdx;
                        myBestImgIdx = imgIdx;
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Train collection loop strategy

    template <typename T> struct SingleTrain
    {
        explicit SingleTrain(const DevMem2D_<T>& trainDescs_) : trainDescs(trainDescs_)
        {
        }

        template <typename Dist, typename ReduceDescCalculator, typename Mask>
        __device__ __forceinline__ void loop(int queryIdx, Mask& m, const ReduceDescCalculator& reduceDescCalc, 
            typename Dist::ResultType& myMin, int& myBestTrainIdx, int& myBestImgIdx, typename Dist::ResultType* sdiff_row) const
        {
            matchDescs<Dist>(queryIdx, 0, trainDescs, m, reduceDescCalc, myMin, myBestTrainIdx, myBestImgIdx, sdiff_row);
        }

        __device__ __forceinline__ int desc_len() const
        {
            return trainDescs.cols;
        }

        const DevMem2D_<T> trainDescs;
    };

    template <typename T> struct TrainCollection
    {
        TrainCollection(const DevMem2D_<T>* trainCollection_, int nImg_, int desclen_) : 
            trainCollection(trainCollection_), nImg(nImg_), desclen(desclen_)
        {
        }

        template <typename Dist, typename ReduceDescCalculator, typename Mask>
        __device__ void loop(int queryIdx, Mask& m, const ReduceDescCalculator& reduceDescCalc, 
            typename Dist::ResultType& myMin, int& myBestTrainIdx, int& myBestImgIdx, typename Dist::ResultType* sdiff_row) const
        {
            for (int imgIdx = 0; imgIdx < nImg; ++imgIdx)
            {
                const DevMem2D_<T> trainDescs = trainCollection[imgIdx];
                m.nextMask();
                matchDescs<Dist>(queryIdx, imgIdx, trainDescs, m, reduceDescCalc, myMin, myBestTrainIdx, myBestImgIdx, sdiff_row);
            }
        }

        __device__ __forceinline__ int desc_len() const
        {
            return desclen;
        }

        const DevMem2D_<T>* trainCollection;
        int nImg;
        int desclen;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Match kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename ReduceDescCalculator, typename Dist, typename T, typename Train, typename Mask>
    __global__ void match(const PtrStep_<T> queryDescs_, const Train train, const Mask mask, int* trainIdx, int* imgIdx, float* distance)
    {
        __shared__ typename Dist::ResultType smem[BLOCK_DIM_X * BLOCK_DIM_Y];        
        
        const int queryIdx = blockIdx.x;
        
        int myBestTrainIdx = -1;
        int myBestImgIdx = -1;
        typename Dist::ResultType myMin = numeric_limits<typename Dist::ResultType>::max();

        {
            typename Dist::ResultType* sdiff_row = smem + BLOCK_DIM_X * threadIdx.y;

            Mask m = mask;

            ReduceDescCalculator reduceDescCalc;

            reduceDescCalc.prepare(queryDescs_.ptr(queryIdx), train.desc_len(), (typename Dist::ValueType*)smem);
        
            train.template loop<Dist>(queryIdx, m, reduceDescCalc, myMin, myBestTrainIdx, myBestImgIdx, sdiff_row);
        }
        __syncthreads();

        typename Dist::ResultType* smin = smem;
        int* strainIdx = (int*)(smin + BLOCK_DIM_Y);
        int* simgIdx = strainIdx + BLOCK_DIM_Y;

        findBestMatch<BLOCK_DIM_Y>(myMin, myBestTrainIdx, myBestImgIdx, smin, strainIdx, simgIdx);

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            imgIdx[queryIdx] = myBestImgIdx;
            trainIdx[queryIdx] = myBestTrainIdx;
            distance[queryIdx] = myMin;
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Match kernel callers

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Train, typename Mask>
    void matchSimple_caller(const DevMem2D_<T>& queryDescs, const Train& train, 
        const Mask& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, cudaStream_t stream)
    {
        StaticAssert<BLOCK_DIM_Y <= 64>::check(); // blockDimY vals must reduce by warp

        dim3 grid(queryDescs.rows, 1, 1);
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        match<BLOCK_DIM_X, BLOCK_DIM_Y, ReduceDescCalculatorSimple<BLOCK_DIM_X, T>, Dist, T>
            <<<grid, threads, 0, stream>>>(queryDescs, train, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, typename Dist, typename T, typename Train, typename Mask>
    void matchCached_caller(const DevMem2D_<T>& queryDescs, const Train& train, 
        const Mask& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, cudaStream_t stream)
    {
        StaticAssert<BLOCK_DIM_Y <= 64>::check();                                // blockDimY vals must reduce by warp
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_DESCRIPTORS_LEN>::check(); // block size must be greter than descriptors length
        StaticAssert<MAX_DESCRIPTORS_LEN % BLOCK_DIM_X == 0>::check();           // max descriptors length must divide to blockDimX

        dim3 grid(queryDescs.rows, 1, 1);
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        match<BLOCK_DIM_X, BLOCK_DIM_Y, ReduceDescCalculatorCached<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN, T, typename Dist::ValueType>, Dist, T>
              <<<grid, threads, 0, stream>>>(queryDescs, train, mask, trainIdx.data, imgIdx.data, distance.data);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Match caller

    template <typename Dist, typename T, typename Train, typename Mask>
    void matchDispatcher(const DevMem2D_<T>& queryDescs, const Train& train, 
        const Mask& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance,
        bool cc_12, cudaStream_t stream)
    {
        if (queryDescs.cols < 64)
            matchCached_caller<16, 16, 64, false, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
        else if (queryDescs.cols == 64)
            matchCached_caller<16, 16, 64, true, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
        else if (queryDescs.cols < 128)
            matchCached_caller<16, 16, 128, false, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
        else if (queryDescs.cols == 128 && cc_12)
            matchCached_caller<16, 16, 128, true, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
        else if (queryDescs.cols < 256 && cc_12)
            matchCached_caller<16, 16, 256, false, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
        else if (queryDescs.cols == 256 && cc_12)
            matchCached_caller<16, 16, 256, true, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
        else
            matchSimple_caller<16, 16, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance, stream);
    }

    template <typename T>
    void matchSingleL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, 
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance,
        bool cc_12, cudaStream_t stream)
    {
        SingleTrain<T> train((DevMem2D_<T>)trainDescs);
        if (mask.data)
        {
            SingleMask m(mask);
            matchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, train, m, trainIdx, imgIdx, distance, cc_12, stream);
        }
        else
        {
            matchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance, cc_12, stream);
        }
    }

    template void matchSingleL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);

    template <typename T>
    void matchSingleL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, 
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
        bool cc_12, cudaStream_t stream)
    {
        SingleTrain<T> train((DevMem2D_<T>)trainDescs);
        if (mask.data)
        {
            SingleMask m(mask);
            matchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, train, m, trainIdx, imgIdx, distance, cc_12, stream);
        }
        else
        {
            matchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance, cc_12, stream);
        }
    }

    template void matchSingleL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);

    template <typename T>
    void matchSingleHamming_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, 
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, 
        bool cc_12, cudaStream_t stream)
    {
        SingleTrain<T> train((DevMem2D_<T>)trainDescs);
        if (mask.data)
        {
            SingleMask m(mask);
            matchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, train, m, trainIdx, imgIdx, distance, cc_12, stream);
        }
        else
        {
            matchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance, cc_12, stream);
        }
    }

    template void matchSingleHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchSingleHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);

    template <typename T>
    void matchCollectionL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainCollection, 
        const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, 
        const DevMem2Df& distance, bool cc_12, cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, queryDescs.cols);
        if (maskCollection.data)
        {
            MaskCollection mask(maskCollection.data);
            matchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, train, mask, trainIdx, imgIdx, distance, cc_12, stream);
        }
        else
        {
            matchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance, cc_12, stream);
        }
    }

    template void matchCollectionL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);

    template <typename T>
    void matchCollectionL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainCollection, 
        const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, 
        const DevMem2Df& distance, bool cc_12, cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, queryDescs.cols);
        if (maskCollection.data)
        {
            MaskCollection mask(maskCollection.data);
            matchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, train, mask, trainIdx, imgIdx, distance, cc_12, stream);
        }
        else
        {
            matchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance, cc_12, stream);
        }
    }

    template void matchCollectionL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);

    template <typename T>
    void matchCollectionHamming_gpu(const DevMem2D& queryDescs, const DevMem2D& trainCollection, 
        const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, 
        const DevMem2Df& distance, bool cc_12, cudaStream_t stream)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, queryDescs.cols);
        if (maskCollection.data)
        {
            MaskCollection mask(maskCollection.data);
            matchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, train, mask, trainIdx, imgIdx, distance, cc_12, stream);
        }
        else
        {
            matchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance, cc_12, stream);
        }
    }

    template void matchCollectionHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    template void matchCollectionHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance, bool cc_12, cudaStream_t stream);
    
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Knn Match ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename ReduceDescCalculator, typename T, typename Mask>
    __device__ void distanceCalcLoop(const PtrStep_<T>& query, const DevMem2D_<T>& train, const Mask& m, int queryIdx,
        typename Dist::ResultType& distMin1, typename Dist::ResultType& distMin2, int& bestTrainIdx1, int& bestTrainIdx2, 
        typename Dist::ResultType* smem)
    {
        ReduceDescCalculator reduceDescCalc;

        reduceDescCalc.prepare(query.ptr(queryIdx), train.cols, (typename Dist::ValueType*)smem);
        
        typename Dist::ResultType* sdiffRow = smem + BLOCK_DIM_X * threadIdx.y;

        for (int trainIdx = threadIdx.y; trainIdx < train.rows; trainIdx += BLOCK_DIM_Y)
        {
            if (m(queryIdx, trainIdx))
            {
                Dist dist;

                const T* trainRow = train.ptr(trainIdx);
                
                reduceDescCalc.calc(trainRow, train.cols, dist, sdiffRow);

                if (threadIdx.x == 0)
                {
                    typename Dist::ResultType val = dist;

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
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename ReduceDescCalculator, typename T, typename Mask>
    __global__ void knnMatch2(const PtrStep_<T> query, const DevMem2D_<T> train, const Mask m, PtrStep_<int2> trainIdx, PtrStep_<float2> distance)
    {
        typedef typename Dist::ResultType ResultType;
        typedef typename Dist::ValueType ValueType;

        __shared__ ResultType smem[BLOCK_DIM_X * BLOCK_DIM_Y];

        const int queryIdx = blockIdx.x;

        ResultType distMin1 = numeric_limits<ResultType>::max();
        ResultType distMin2 = numeric_limits<ResultType>::max();

        int bestTrainIdx1 = -1;
        int bestTrainIdx2 = -1;

        distanceCalcLoop<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, ReduceDescCalculator>(query, train, m, queryIdx, 
            distMin1, distMin2, bestTrainIdx1, bestTrainIdx2, smem);
        __syncthreads();

        volatile ResultType* sdistMinRow = smem;
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
            distMin1 = numeric_limits<ResultType>::max();
            distMin2 = numeric_limits<ResultType>::max();

            bestTrainIdx1 = -1;
            bestTrainIdx2 = -1;

            #pragma unroll
            for (int i = 0; i < BLOCK_DIM_Y; ++i)
            {
                ResultType val = sdistMinRow[i];

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
                ResultType val = sdistMinRow[i];

                if (val < distMin2)
                {
                    distMin2 = val;
                    bestTrainIdx2 = sbestTrainIdxRow[i];
                }
            }

            trainIdx.ptr(queryIdx)[0] = make_int2(bestTrainIdx1, bestTrainIdx2);
            distance.ptr(queryIdx)[0] = make_float2(distMin1, distMin2);
        }
    }

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    void knnMatch2Simple_caller(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, const Mask& mask, 
        const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, cudaStream_t stream)
    {
        dim3 grid(queryDescs.rows, 1, 1);
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        knnMatch2<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, ReduceDescCalculatorSimple<BLOCK_DIM_X, T>, T>
            <<<grid, threads, 0, stream>>>(queryDescs, trainDescs, mask, trainIdx, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, typename Dist, typename T, typename Mask>
    void knnMatch2Cached_caller(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, const Mask& mask, 
        const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, cudaStream_t stream)
    {
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_DESCRIPTORS_LEN>::check(); // block size must be greter than descriptors length
        StaticAssert<MAX_DESCRIPTORS_LEN % BLOCK_DIM_X == 0>::check();           // max descriptors length must divide to blockDimX

        dim3 grid(queryDescs.rows, 1, 1);
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        knnMatch2<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, ReduceDescCalculatorCached<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN, T, typename Dist::ValueType>, T>
              <<<grid, threads, 0, stream>>>(queryDescs, trainDescs, mask, trainIdx, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    
    template <typename Dist, typename T, typename Mask>
    void knnMatch2Dispatcher(const DevMem2D_<T>& query, const DevMem2D_<T>& train, const Mask& mask, 
        const DevMem2D_<int2>& trainIdx, const DevMem2D_<float2>& distance, bool cc_12, cudaStream_t stream)
    {
        if (query.cols < 64)
            knnMatch2Cached_caller<16, 16, 64, false, Dist>(query, train, mask, trainIdx, distance, stream);
        else if (query.cols == 64)
            knnMatch2Cached_caller<16, 16, 64, true, Dist>(query, train, mask, trainIdx, distance, stream);
        else if (query.cols < 128)
            knnMatch2Cached_caller<16, 16, 128, false, Dist>(query, train, mask, trainIdx, distance, stream);
        else if (query.cols == 128 && cc_12)
            knnMatch2Cached_caller<16, 16, 128, true, Dist>(query, train, mask, trainIdx, distance, stream);
        else if (query.cols < 256 && cc_12)
            knnMatch2Cached_caller<16, 16, 256, false, Dist>(query, train, mask, trainIdx, distance, stream);
        else if (query.cols == 256 && cc_12)
            knnMatch2Cached_caller<16, 16, 256, true, Dist>(query, train, mask, trainIdx, distance, stream);
        else
            knnMatch2Simple_caller<16, 16, Dist>(query, train, mask, trainIdx, distance, stream);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    __global__ void calcDistance(const PtrStep_<T> queryDescs_, const DevMem2D_<T> trainDescs_, const Mask mask, PtrStepf distance)
    {
        __shared__ typename Dist::ResultType sdiff[BLOCK_DIM_X * BLOCK_DIM_Y];

        typename Dist::ResultType* sdiff_row = sdiff + BLOCK_DIM_X * threadIdx.y;
        
        const int queryIdx = blockIdx.x;
        const T* queryDescs = queryDescs_.ptr(queryIdx);

        const int trainIdx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (trainIdx < trainDescs_.rows)
        {
            const T* trainDescs = trainDescs_.ptr(trainIdx);

            typename Dist::ResultType myDist = numeric_limits<typename Dist::ResultType>::max();

            if (mask(queryIdx, trainIdx))
            {
                Dist dist;

                reduceDescDiff<BLOCK_DIM_X>(queryDescs, trainDescs, trainDescs_.cols, dist, sdiff_row);

                if (threadIdx.x == 0)
                    myDist = dist;
            }
            
            if (threadIdx.x == 0)
                distance.ptr(queryIdx)[trainIdx] = myDist;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    void calcDistance_caller(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, 
        const Mask& mask, const DevMem2Df& distance, cudaStream_t stream)
    {
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        dim3 grid(queryDescs.rows, divUp(trainDescs.rows, BLOCK_DIM_Y), 1);

        calcDistance<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, T><<<grid, threads, 0, stream>>>(
            queryDescs, trainDescs, mask, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
        
    ///////////////////////////////////////////////////////////////////////////////
    // warpReduceMinIdx

    template <int BLOCK_SIZE, typename T> 
    __device__ void warpReduceMinIdx(volatile T* sdist, volatile int* strainIdx, T& myMin, int tid)
    {
        if (tid < 32)
        {
            if (BLOCK_SIZE >= 64) 
            { 
                T reg = sdist[tid + 32];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 32];
                }
            }
            if (BLOCK_SIZE >= 32) 
            { 
                T reg = sdist[tid + 16];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 16];
                }
            }
            if (BLOCK_SIZE >= 16) 
            { 
                T reg = sdist[tid + 8];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 8];
                }
            }
            if (BLOCK_SIZE >= 8) 
            { 
                T reg = sdist[tid + 4];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 4];
                }
            }
            if (BLOCK_SIZE >= 4) 
            { 
                T reg = sdist[tid + 2];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 2];
                } 
            }
            if (BLOCK_SIZE >= 2) 
            { 
                T reg = sdist[tid + 1];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 1];
                }
            }
        }
    }
    
    template <int BLOCK_SIZE, typename T> 
    __device__ void reduceMinIdx(const T* dist, int n, T* sdist, int* strainIdx)
    {
        const int tid = threadIdx.x;
        
        T myMin = numeric_limits<T>::max();
        int myMinIdx = -1;

        for (int i = tid; i < n; i += BLOCK_SIZE)
        {
            T reg = dist[i];
            if (reg < myMin)
            {
                myMin = reg;
                myMinIdx = i;
            }
        }

        sdist[tid] = myMin;
        strainIdx[tid] = myMinIdx;
        __syncthreads();

        if (BLOCK_SIZE >= 512 && tid < 256) 
        {
            T reg = sdist[tid + 256];

            if (reg < myMin)
            {
                sdist[tid] = myMin = reg;
                strainIdx[tid] = strainIdx[tid + 256];
            }
            __syncthreads(); 
        }
        if (BLOCK_SIZE >= 256 && tid < 128) 
        {
            T reg = sdist[tid + 128];

            if (reg < myMin)
            {
                sdist[tid] = myMin = reg;
                strainIdx[tid] = strainIdx[tid + 128];
            }
            __syncthreads(); 
        }
        if (BLOCK_SIZE >= 128 && tid < 64) 
        {
            T reg = sdist[tid + 64];

            if (reg < myMin)
            {
                sdist[tid] = myMin = reg;
                strainIdx[tid] = strainIdx[tid + 64];
            }
            __syncthreads(); 
        }
        
        warpReduceMinIdx<BLOCK_SIZE>(sdist, strainIdx, myMin, tid);
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

        reduceMinIdx<BLOCK_SIZE>(allDist, allDist_.cols, sdist, strainIdx);

        if (threadIdx.x == 0)
        {
            float dist = sdist[0];
            if (dist < numeric_limits<float>::max())
            {
                int bestIdx = strainIdx[0];
                allDist[bestIdx] = numeric_limits<float>::max();
                trainIdx[i] = bestIdx;
                distance[i] = dist;
            }
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // find knn match kernel caller

    template <int BLOCK_SIZE>
    void findKnnMatch_caller(int knn, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, cudaStream_t stream)
    {
        dim3 threads(BLOCK_SIZE, 1, 1);
        dim3 grid(trainIdx.rows, 1, 1);

        for (int i = 0; i < knn; ++i)
        {
            findBestMatch<BLOCK_SIZE><<<grid, threads, 0, stream>>>(allDist, i, trainIdx, distance);
            cudaSafeCall( cudaGetLastError() );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // knn match caller

    template <typename Dist, typename T, typename Mask>
    void calcDistanceDispatcher(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, const Mask& mask, const DevMem2Df& allDist, cudaStream_t stream)
    {
        calcDistance_caller<16, 16, Dist>(queryDescs, trainDescs, mask, allDist, stream);
    }

    void findKnnMatchDispatcher(int knn, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, cudaStream_t stream)
    {
        findKnnMatch_caller<256>(knn, trainIdx, distance, allDist, stream);
    }

    template < typename Dist, typename T >
    void knnMatchDispatcher(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, int knn,
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream)
    {
        if (mask.data)
        {
            if (knn == 2)
            {
                knnMatch2Dispatcher<Dist>(queryDescs, trainDescs, SingleMask(mask), (DevMem2D_<int2>)trainIdx, (DevMem2D_<float2>)distance, cc_12, stream);
                return;
            }

            calcDistanceDispatcher<Dist>(queryDescs, trainDescs, SingleMask(mask), allDist, stream);
        }
        else
        {
            if (knn == 2)
            {
                knnMatch2Dispatcher<Dist>(queryDescs, trainDescs, WithOutMask(), (DevMem2D_<int2>)trainIdx, (DevMem2D_<float2>)distance, cc_12, stream);
                return;
            }

            calcDistanceDispatcher<Dist>(queryDescs, trainDescs, WithOutMask(), allDist, stream);
        }

        findKnnMatchDispatcher(knn, trainIdx, distance, allDist, stream);
    }

    template <typename T>
    void knnMatchL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn,
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream)
    {
        knnMatchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, knn, mask, trainIdx, distance, allDist, cc_12, stream);
    }

    template void knnMatchL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);

    template <typename T>
    void knnMatchL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn,
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream)
    {
        knnMatchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, knn, mask, trainIdx, distance, allDist, cc_12, stream);
    }

    template void knnMatchL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);

    template <typename T>
    void knnMatchHamming_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn,
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream)
    {
        knnMatchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, knn, mask, trainIdx, distance, allDist, cc_12, stream);
    }

    template void knnMatchHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);
    template void knnMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist, bool cc_12, cudaStream_t stream);

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Radius Match //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    __global__ void radiusMatch(const PtrStep_<T> queryDescs_, const DevMem2D_<T> trainDescs_, 
        float maxDistance, const Mask mask, DevMem2Di trainIdx_, unsigned int* nMatches, PtrStepf distance)
    {
        #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 110

        __shared__ typename Dist::ResultType smem[BLOCK_DIM_X * BLOCK_DIM_Y];

        typename Dist::ResultType* sdiff_row = smem + BLOCK_DIM_X * threadIdx.y;
        
        const int queryIdx = blockIdx.x;
        const T* queryDescs = queryDescs_.ptr(queryIdx);

        const int trainIdx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (trainIdx < trainDescs_.rows)
        {
            const T* trainDescs = trainDescs_.ptr(trainIdx);

            if (mask(queryIdx, trainIdx))
            {
                Dist dist;

                reduceDescDiff<BLOCK_DIM_X>(queryDescs, trainDescs, trainDescs_.cols, dist, sdiff_row);

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
    void radiusMatch_caller(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, 
        float maxDistance, const Mask& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, 
        const DevMem2Df& distance, cudaStream_t stream)
    {
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        dim3 grid(queryDescs.rows, divUp(trainDescs.rows, BLOCK_DIM_Y), 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, Dist, T><<<grid, threads, 0, stream>>>(
            queryDescs, trainDescs, maxDistance, mask, trainIdx, nMatches, distance);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match caller

    template <typename Dist, typename T, typename Mask>
    void radiusMatchDispatcher(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, 
        float maxDistance, const Mask& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, 
        const DevMem2Df& distance, cudaStream_t stream)
    {
        radiusMatch_caller<16, 16, Dist>(queryDescs, trainDescs, maxDistance, mask, 
            trainIdx, nMatches, distance, stream);
    }

    template <typename T>
    void radiusMatchL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance,
        const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream)
    {
        if (mask.data)
        {
            radiusMatchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, SingleMask(mask), trainIdx, nMatches, distance, stream);
        }
        else
        {
            radiusMatchDispatcher< L1Dist<T> >((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, WithOutMask(), trainIdx, nMatches, distance, stream);
        }
    }

    template void radiusMatchL1_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL1_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);

    template <typename T>
    void radiusMatchL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance,
        const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream)
    {
        if (mask.data)
        {
            radiusMatchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, SingleMask(mask), trainIdx, nMatches, distance, stream);
        }
        else
        {
            radiusMatchDispatcher<L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, WithOutMask(), trainIdx, nMatches, distance, stream);
        }
    }

    template void radiusMatchL2_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchL2_gpu<float >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);

    template <typename T>
    void radiusMatchHamming_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance,
        const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream)
    {
        if (mask.data)
        {
            radiusMatchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, SingleMask(mask), trainIdx, nMatches, distance, stream);
        }
        else
        {
            radiusMatchDispatcher<HammingDist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, WithOutMask(), trainIdx, nMatches, distance, stream);
        }
    }

    template void radiusMatchHamming_gpu<uchar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<schar >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<ushort>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<short >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
    template void radiusMatchHamming_gpu<int   >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance, cudaStream_t stream);
}}}
