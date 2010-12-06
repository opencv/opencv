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

#include "cuda_shared.hpp"
#include "limits_gpu.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace cv { namespace gpu { namespace bfmatcher
{
///////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// General funcs //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
    
    template <bool expr> struct StaticAssert;
    template <> struct StaticAssert<true> {static __host__ __device__ void check(){}};

    ///////////////////////////////////////////////////////////////////////////////
    // Mask strategy

    class SingleMask
    {
    public:
        explicit SingleMask(const PtrStep& mask_) : mask(mask_) {}
        
        __device__ bool operator()(int queryIdx, int trainIdx) const
        {            
            return mask.ptr(queryIdx)[trainIdx] != 0;
        }
    private:
        PtrStep mask;
    };

    class MaskCollection
    {
    public:
        explicit MaskCollection(PtrStep* maskCollection_) : maskCollection(maskCollection_) {}

        __device__ void nextMask()
        {
            curMask = *maskCollection++;
        }
        
        __device__ bool operator()(int queryIdx, int trainIdx) const
        {            
            return curMask.data == 0 || curMask.ptr(queryIdx)[trainIdx] != 0;
        }
    private:
        PtrStep* maskCollection;
        PtrStep curMask;
    };

    class WithOutMask
    {
    public:
        __device__ void nextMask()
        {
        }
        __device__ bool operator()(int queryIdx, int trainIdx) const
        {
            return true;
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Reduce Sum
    
    template <int BLOCK_DIM_X>
    __device__ void reduceSum(float* sdiff, float mySum, int tid)
    {
        sdiff[tid] = mySum;
        __syncthreads();

        if (BLOCK_DIM_X == 512) 
        {
            if (tid < 256) 
            { 
                sdiff[tid] = mySum += sdiff[tid + 256]; __syncthreads(); 
                sdiff[tid] = mySum += sdiff[tid + 128]; __syncthreads();
                sdiff[tid] = mySum += sdiff[tid +  64]; __syncthreads();
            }
            volatile float* smem = sdiff;
            smem[tid] = mySum += smem[tid + 32]; 
            smem[tid] = mySum += smem[tid + 16]; 
            smem[tid] = mySum += smem[tid +  8]; 
            smem[tid] = mySum += smem[tid +  4]; 
            smem[tid] = mySum += smem[tid +  2];
            smem[tid] = mySum += smem[tid +  1]; 
        }
        if (BLOCK_DIM_X == 256)
        {
            if (tid < 128) 
            { 
                sdiff[tid] = mySum += sdiff[tid + 128]; __syncthreads(); 
                sdiff[tid] = mySum += sdiff[tid +  64]; __syncthreads();
            }
            volatile float* smem = sdiff;
            smem[tid] = mySum += smem[tid + 32]; 
            smem[tid] = mySum += smem[tid + 16]; 
            smem[tid] = mySum += smem[tid +  8]; 
            smem[tid] = mySum += smem[tid +  4]; 
            smem[tid] = mySum += smem[tid +  2];
            smem[tid] = mySum += smem[tid +  1];
        }
        if (BLOCK_DIM_X == 128)
        {
            if (tid <  64) 
            { 
                sdiff[tid] = mySum += sdiff[tid +  64]; __syncthreads(); 
            }
            volatile float* smem = sdiff;
            smem[tid] = mySum += smem[tid + 32]; 
            smem[tid] = mySum += smem[tid + 16]; 
            smem[tid] = mySum += smem[tid +  8]; 
            smem[tid] = mySum += smem[tid +  4]; 
            smem[tid] = mySum += smem[tid +  2];
            smem[tid] = mySum += smem[tid +  1];
        }
        
        volatile float* smem = sdiff;
        if (BLOCK_DIM_X == 64) 
        {
            if (tid < 32) 
            {
                smem[tid] = mySum += smem[tid + 32]; 
                smem[tid] = mySum += smem[tid + 16]; 
                smem[tid] = mySum += smem[tid +  8]; 
                smem[tid] = mySum += smem[tid +  4]; 
                smem[tid] = mySum += smem[tid +  2];
                smem[tid] = mySum += smem[tid +  1];  
            }
        }
        if (BLOCK_DIM_X == 32) 
        {
            if (tid < 16) 
            {
                smem[tid] = mySum += smem[tid + 16]; 
                smem[tid] = mySum += smem[tid +  8]; 
                smem[tid] = mySum += smem[tid +  4]; 
                smem[tid] = mySum += smem[tid +  2];
                smem[tid] = mySum += smem[tid +  1];  
            }
        }
        if (BLOCK_DIM_X == 16) 
        {
            if (tid < 8) 
            {
                smem[tid] = mySum += smem[tid +  8]; 
                smem[tid] = mySum += smem[tid +  4]; 
                smem[tid] = mySum += smem[tid +  2];
                smem[tid] = mySum += smem[tid +  1];  
            }
        }
        if (BLOCK_DIM_X == 8) 
        {
            if (tid < 4) 
            {
                smem[tid] = mySum += smem[tid +  4]; 
                smem[tid] = mySum += smem[tid +  2];
                smem[tid] = mySum += smem[tid +  1];  
            }
        }
        if (BLOCK_DIM_X == 4) 
        {
            if (tid < 2) 
            {
                smem[tid] = mySum += smem[tid +  2];
                smem[tid] = mySum += smem[tid +  1];  
            }
        }
        if (BLOCK_DIM_X == 2) 
        {
            if (tid < 1) 
            {
                smem[tid] = mySum += smem[tid +  1];  
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // loadDescsVals

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_DESCRIPTORS_LEN, typename T> 
    __device__ void loadDescsVals(const T* descs, int desc_len, float* smem, float* queryVals)
    {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (tid < desc_len)
        {
            smem[tid] = (float)descs[tid];
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
    // Distance

    template <int BLOCK_DIM_X>
    class L1Dist
    {
    public:
        __device__ L1Dist() : mySum(0) {}

        __device__ void reduceIter(float val1, float val2)
        {
            mySum += fabs(val1 - val2);
        }

        __device__ void reduceAll(float* sdiff, int tid)
        {
            reduceSum<BLOCK_DIM_X>(sdiff, mySum, tid);
        }

        static __device__ float finalResult(float res)
        {
            return res;
        }
    private:
        float mySum;
    };

    template <int BLOCK_DIM_X>
    class L2Dist
    {
    public:
        __device__ L2Dist() : mySum(0) {}

        __device__ void reduceIter(float val1, float val2)
        {
            float reg = val1 - val2;
            mySum += reg * reg;
        }

        __device__ void reduceAll(float* sdiff, int tid)
        {
            reduceSum<BLOCK_DIM_X>(sdiff, mySum, tid);
        }

        static __device__ float finalResult(float res)
        {
            return sqrtf(res);
        }
    private:
        float mySum;
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    // reduceDescDiff

    template <int BLOCK_DIM_X, typename Dist, typename T> 
    __device__ void reduceDescDiff(const T* queryDescs, const T* trainDescs, int desc_len, float* sdiff)
    {
        const int tid = threadIdx.x;

        Dist dist;

        for (int i = tid; i < desc_len; i += BLOCK_DIM_X)
            dist.reduceIter(queryDescs[i], trainDescs[i]);

        dist.reduceAll(sdiff, tid);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // reduceDescDiff_smem

    template <int N> struct UnrollDescDiff
    {
        template <typename Dist, typename T>
        static __device__ void calcCheck(Dist& dist, const float* queryVals, const T* trainDescs, 
            int ind, int desc_len)
        {
            if (ind < desc_len)
                dist.reduceIter(*queryVals, trainDescs[ind]);

            ++queryVals;

            UnrollDescDiff<N - 1>::calcCheck(dist, queryVals, trainDescs, ind + blockDim.x, desc_len);
        }

        template <typename Dist, typename T>
        static __device__ void calcWithoutCheck(Dist& dist, const float* queryVals, const T* trainDescs)
        {
            dist.reduceIter(*queryVals, *trainDescs);

            ++queryVals;
            trainDescs += blockDim.x;

            UnrollDescDiff<N - 1>::calcWithoutCheck(dist, queryVals, trainDescs);
        }
    };
    template <> struct UnrollDescDiff<0>
    {
        template <typename Dist, typename T>
        static __device__ void calcCheck(Dist& dist, const float* queryVals, const T* trainDescs, 
            int ind, int desc_len)
        {
        }

        template <typename Dist, typename T>
        static __device__ void calcWithoutCheck(Dist& dist, const float* queryVals, const T* trainDescs)
        {
        }
    };

    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN, bool WITH_OUT_CHECK> struct DescDiffCalculator;
    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN> 
    struct DescDiffCalculator<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, false>
    {
        template <typename Dist, typename T>
        static __device__ void calc(Dist& dist, const float* queryVals, const T* trainDescs, int desc_len)
        {
            UnrollDescDiff<MAX_DESCRIPTORS_LEN / BLOCK_DIM_X>::calcCheck(dist, queryVals, trainDescs, 
                threadIdx.x, desc_len);
        }
    };
    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN> 
    struct DescDiffCalculator<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, true>
    {
        template <typename Dist, typename T>
        static __device__ void calc(Dist& dist, const float* queryVals, const T* trainDescs, int desc_len)
        {
            UnrollDescDiff<MAX_DESCRIPTORS_LEN / BLOCK_DIM_X>::calcWithoutCheck(dist, queryVals, 
                trainDescs + threadIdx.x);
        }
    };

    template <int BLOCK_DIM_X, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, typename Dist, typename T>
    __device__ void reduceDescDiff_smem(const float* queryVals, const T* trainDescs, int desc_len, float* sdiff)
    {
        const int tid = threadIdx.x;
        
        Dist dist;

        DescDiffCalculator<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN>::calc(dist, queryVals, 
            trainDescs, desc_len);
        
        dist.reduceAll(sdiff, tid);
    }

///////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// Match //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////
    // warpReduceMin

    template <int BLOCK_DIM_Y> 
    __device__ void warpReduceMin(int tid, volatile float* sdata, volatile int* strainIdx, volatile int* simgIdx)
    {
        float minSum = sdata[tid];

        if (BLOCK_DIM_Y >= 64) 
        {
            float reg = sdata[tid + 32];
            if (reg < minSum)
            {
                sdata[tid] = minSum = reg;
                strainIdx[tid] = strainIdx[tid + 32];
                simgIdx[tid] = simgIdx[tid + 32];
            }
        }
        if (BLOCK_DIM_Y >= 32) 
        {
            float reg = sdata[tid + 16];
            if (reg < minSum)
            {
                sdata[tid] = minSum = reg;
                strainIdx[tid] = strainIdx[tid + 16];
                simgIdx[tid] = simgIdx[tid + 16];
            }
        }
        if (BLOCK_DIM_Y >= 16) 
        {
            float reg = sdata[tid + 8];
            if (reg < minSum)
            {
                sdata[tid] = minSum = reg;
                strainIdx[tid] = strainIdx[tid + 8];
                simgIdx[tid] = simgIdx[tid + 8];
            }
        }
        if (BLOCK_DIM_Y >= 8) 
        { 
            float reg = sdata[tid + 4];
            if (reg < minSum)
            {
                sdata[tid] = minSum = reg;
                strainIdx[tid] = strainIdx[tid + 4];
                simgIdx[tid] = simgIdx[tid + 4];
            }
        }
        if (BLOCK_DIM_Y >= 4) 
        { 
            float reg = sdata[tid + 2];
            if (reg < minSum)
            {
                sdata[tid] = minSum = reg;
                strainIdx[tid] = strainIdx[tid + 2];
                simgIdx[tid] = simgIdx[tid + 2];
            }
        }
        if (BLOCK_DIM_Y >= 2) 
        {
            float reg = sdata[tid + 1];
            if (reg < minSum)
            {
                sdata[tid] = minSum = reg;
                strainIdx[tid] = strainIdx[tid + 1];
                simgIdx[tid] = simgIdx[tid + 1];
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // findBestMatch

    template <int BLOCK_DIM_Y, typename Dist>
    __device__ void findBestMatch(int queryIdx, float myMin, int myBestTrainIdx, int myBestImgIdx, 
        float* smin, int* strainIdx, int* simgIdx, int* trainIdx, int* imgIdx, float* distance)
    {
        if (threadIdx.x == 0)
        {
            smin[threadIdx.y] = myMin;
            strainIdx[threadIdx.y] = myBestTrainIdx;
            simgIdx[threadIdx.y] = myBestImgIdx;
        }
        __syncthreads();

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        if (tid < 32)
            warpReduceMin<BLOCK_DIM_Y>(tid, smin, strainIdx, simgIdx);

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            float minSum = smin[0];
            int bestTrainIdx = strainIdx[0];
            int bestImgIdx = simgIdx[0];

            imgIdx[queryIdx] = bestImgIdx;
            trainIdx[queryIdx] = bestTrainIdx;
            distance[queryIdx] = Dist::finalResult(minSum);
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // ReduceDescCalculator

    template <int BLOCK_DIM_X, typename Dist, typename T>
    class ReduceDescCalculatorSimple
    {
    public:
        __device__ void prepare(const T* queryDescs_, int, float*)
        {
            queryDescs = queryDescs_;
        }

        __device__ void calc(const T* trainDescs, int desc_len, float* sdiff_row) const
        {
            reduceDescDiff<BLOCK_DIM_X, Dist>(queryDescs, trainDescs, desc_len, sdiff_row);
        }

    private:
        const T* queryDescs;
    };

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, 
        typename Dist, typename T>
    class ReduceDescCalculatorSmem
    {
    public:
        __device__ void prepare(const T* queryDescs, int desc_len, float* smem)
        {
            loadDescsVals<BLOCK_DIM_X, BLOCK_DIM_Y, MAX_DESCRIPTORS_LEN>(queryDescs, desc_len, smem, queryVals);
        }

        __device__ void calc(const T* trainDescs, int desc_len, float* sdiff_row) const
        {
            reduceDescDiff_smem<BLOCK_DIM_X, MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN, Dist>(queryVals, trainDescs, 
                desc_len, sdiff_row);
        }

    private:
        float queryVals[MAX_DESCRIPTORS_LEN / BLOCK_DIM_X];
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    // matchDescs loop

    template <typename ReduceDescCalculator, typename T, typename Mask>
    __device__ void matchDescs(int queryIdx, const int imgIdx, const DevMem2D_<T>& trainDescs_,  
        const Mask& m, const ReduceDescCalculator& reduceDescCalc,
        float* sdiff_row, float& myMin, int& myBestTrainIdx, int& myBestImgIdx)
    {
        const T* trainDescs = trainDescs_.ptr(threadIdx.y);
        const int trainDescsStep = blockDim.y * trainDescs_.step / sizeof(T);
        for (int trainIdx = threadIdx.y; trainIdx < trainDescs_.rows; 
             trainIdx += blockDim.y, trainDescs += trainDescsStep)
        {
            if (m(queryIdx, trainIdx))
            {
                reduceDescCalc.calc(trainDescs, trainDescs_.cols, sdiff_row);

                if (threadIdx.x == 0)
                {
                    float reg = sdiff_row[0];
                    if (reg < myMin)
                    {
                        myMin = reg;
                        myBestTrainIdx = trainIdx;
                        myBestImgIdx = imgIdx;
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Train collection loop strategy

    template <typename T>
    class SingleTrain
    {
    public:
        explicit SingleTrain(const DevMem2D_<T>& trainDescs_) : trainDescs(trainDescs_)
        {
        }

        template <typename ReduceDescCalculator, typename Mask>
        __device__ void loop(int queryIdx, Mask& m, const ReduceDescCalculator& reduceDescCalc, 
            float* sdiff_row, float& myMin, int& myBestTrainIdx, int& myBestImgIdx) const
        {
            matchDescs(queryIdx, 0, trainDescs, m, reduceDescCalc, 
                sdiff_row, myMin, myBestTrainIdx, myBestImgIdx);
        }

        __device__ int desc_len() const
        {
            return trainDescs.cols;
        }
    private:
        DevMem2D_<T> trainDescs;
    };

    template <typename T>
    class TrainCollection
    {
    public:
        TrainCollection(const DevMem2D_<T>* trainCollection_, int nImg_, int desclen_) : 
            trainCollection(trainCollection_), nImg(nImg_), desclen(desclen_)
        {
        }

        template <typename ReduceDescCalculator, typename Mask>
        __device__ void loop(int queryIdx, Mask& m, const ReduceDescCalculator& reduceDescCalc, 
            float* sdiff_row, float& myMin, int& myBestTrainIdx, int& myBestImgIdx) const
        {
            for (int imgIdx = 0; imgIdx < nImg; ++imgIdx)
            {
                DevMem2D_<T> trainDescs = trainCollection[imgIdx];
                m.nextMask();
                matchDescs(queryIdx, imgIdx, trainDescs, m, reduceDescCalc, 
                    sdiff_row, myMin, myBestTrainIdx, myBestImgIdx);
            }
        }

        __device__ int desc_len() const
        {
            return desclen;
        }
    private:
        const DevMem2D_<T>* trainCollection;
        int nImg;
        int desclen;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Match kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename ReduceDescCalculator, typename Dist, typename T, 
        typename Train, typename Mask>
    __global__ void match(PtrStep_<T> queryDescs_, Train train, Mask mask, int* trainIdx, int* imgIdx, float* distance)
    {
        __shared__ float sdiff[BLOCK_DIM_X * BLOCK_DIM_Y];
        __shared__ float smin[64];
        __shared__ int strainIdx[64];
        __shared__ int simgIdx[64];
        
        const int queryIdx = blockIdx.x;
        
        int myBestTrainIdx = -1;
        int myBestImgIdx = -1;
        float myMin = numeric_limits_gpu<float>::max();

        {
            float* sdiff_row = sdiff + BLOCK_DIM_X * threadIdx.y;
            Mask m = mask;
            ReduceDescCalculator reduceDescCalc;
            reduceDescCalc.prepare(queryDescs_.ptr(queryIdx), train.desc_len(), sdiff);
        
            train.loop(queryIdx, m, reduceDescCalc, sdiff_row, myMin, myBestTrainIdx, myBestImgIdx);
        }

        findBestMatch<BLOCK_DIM_Y, Dist>(queryIdx, myMin, myBestTrainIdx, myBestImgIdx, 
            smin, strainIdx, simgIdx, trainIdx, imgIdx, distance);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Match kernel callers

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, template <int> class Dist, typename T, 
        typename Train, typename Mask>
    void match_caller(const DevMem2D_<T>& queryDescs, const Train& train, 
        const Mask& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        StaticAssert<BLOCK_DIM_Y <= 64>::check(); // blockDimY vals must reduce by warp

        dim3 grid(queryDescs.rows, 1, 1);
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        match<BLOCK_DIM_X, BLOCK_DIM_Y, ReduceDescCalculatorSimple<BLOCK_DIM_X, Dist<BLOCK_DIM_X>, T>, 
            Dist<BLOCK_DIM_X>, T><<<grid, threads>>>(queryDescs, train, mask, trainIdx.data, 
            imgIdx.data, distance.data);

        cudaSafeCall( cudaThreadSynchronize() );
    }
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int MAX_DESCRIPTORS_LEN, bool DESC_LEN_EQ_MAX_LEN, 
        template <int> class Dist, typename T, typename Train, typename Mask>
    void match_smem_caller(const DevMem2D_<T>& queryDescs, const Train& train, 
        const Mask& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        StaticAssert<BLOCK_DIM_Y <= 64>::check();                                // blockDimY vals must reduce by warp
        StaticAssert<BLOCK_DIM_X * BLOCK_DIM_Y >= MAX_DESCRIPTORS_LEN>::check(); // block size must be greter than descriptors length
        StaticAssert<MAX_DESCRIPTORS_LEN % BLOCK_DIM_X == 0>::check();           // max descriptors length must divide to blockDimX

        dim3 grid(queryDescs.rows, 1, 1);
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);

        match<BLOCK_DIM_X, BLOCK_DIM_Y, ReduceDescCalculatorSmem<BLOCK_DIM_X, BLOCK_DIM_Y, 
              MAX_DESCRIPTORS_LEN, DESC_LEN_EQ_MAX_LEN, Dist<BLOCK_DIM_X>, T>, 
              Dist<BLOCK_DIM_X>, T><<<grid, threads>>>(queryDescs, train, mask, trainIdx.data, 
              imgIdx.data, distance.data);

        cudaSafeCall( cudaThreadSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Match kernel chooser

    template <template <int> class Dist, typename T, typename Train, typename Mask>
    void match_chooser(const DevMem2D_<T>& queryDescs, const Train& train, 
        const Mask& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        if (queryDescs.cols < 64)
            match_smem_caller<16, 16, 64, false, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);
        else if (queryDescs.cols == 64)
            match_smem_caller<16, 16, 64, true, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);
        else if (queryDescs.cols < 128)
            match_smem_caller<16, 16, 128, false, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);
        else if (queryDescs.cols == 128)
            match_smem_caller<16, 16, 128, true, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);
        else if (queryDescs.cols < 256)
            match_smem_caller<16, 16, 256, false, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);
        else if (queryDescs.cols == 256)
            match_smem_caller<16, 16, 256, true, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);
        else
            match_caller<16, 16, Dist>(queryDescs, train, mask, trainIdx, imgIdx, distance);

        cudaSafeCall( cudaThreadSynchronize() );
    }

    template <typename T>
    void matchSingleL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, 
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        SingleTrain<T> train((DevMem2D_<T>)trainDescs);
        if (mask.data)
        {
            SingleMask m(mask);
            match_chooser<L1Dist>((DevMem2D_<T>)queryDescs, train, m, trainIdx, imgIdx, distance);
        }
        else
        {
            match_chooser<L1Dist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance);
        }
    }

    template void matchSingleL1_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL1_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL1_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL1_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL1_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL1_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);

    template <typename T>
    void matchSingleL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, 
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        SingleTrain<T> train((DevMem2D_<T>)trainDescs);
        if (mask.data)
        {
            SingleMask m(mask);
            match_chooser<L2Dist>((DevMem2D_<T>)queryDescs, train, m, trainIdx, imgIdx, distance);
        }
        else
        {
            match_chooser<L2Dist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance);
        }
    }

    template void matchSingleL2_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL2_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL2_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL2_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL2_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchSingleL2_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);

    template <typename T>
    void matchCollectionL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainCollection, 
        const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, queryDescs.cols);
        if (maskCollection.data)
        {
            MaskCollection mask(maskCollection.data);
            match_chooser<L1Dist>((DevMem2D_<T>)queryDescs, train, mask, trainIdx, imgIdx, distance);
        }
        else
        {
            match_chooser<L1Dist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance);
        }
    }

    template void matchCollectionL1_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL1_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL1_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL1_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL1_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL1_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);

    template <typename T>
    void matchCollectionL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainCollection, 
        const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance)
    {
        TrainCollection<T> train((DevMem2D_<T>*)trainCollection.ptr(), trainCollection.cols, queryDescs.cols);
        if (maskCollection.data)
        {
            MaskCollection mask(maskCollection.data);
            match_chooser<L2Dist>((DevMem2D_<T>)queryDescs, train, mask, trainIdx, imgIdx, distance);
        }
        else
        {
            match_chooser<L2Dist>((DevMem2D_<T>)queryDescs, train, WithOutMask(), trainIdx, imgIdx, distance);
        }
    }

    template void matchCollectionL2_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL2_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL2_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL2_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL2_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    template void matchCollectionL2_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainCollection, const DevMem2D_<PtrStep>& maskCollection, const DevMem2Di& trainIdx, const DevMem2Di& imgIdx, const DevMem2Df& distance);
    
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Knn Match ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    __global__ void calcDistance(PtrStep_<T> queryDescs_, DevMem2D_<T> trainDescs_, Mask mask, PtrStepf distance)
    {
        __shared__ float sdiff[BLOCK_DIM_X * BLOCK_DIM_Y];

        float* sdiff_row = sdiff + BLOCK_DIM_X * threadIdx.y;
        
        const int queryIdx = blockIdx.x;
        const T* queryDescs = queryDescs_.ptr(queryIdx);

        const int trainIdx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (trainIdx < trainDescs_.rows)
        {
            const T* trainDescs = trainDescs_.ptr(trainIdx);

            float dist = numeric_limits_gpu<float>::max();

            if (mask(queryIdx, trainIdx))
            {
                reduceDescDiff<BLOCK_DIM_X, Dist>(queryDescs, trainDescs, trainDescs_.cols, sdiff_row);

                if (threadIdx.x == 0)
                {
                    dist = Dist::finalResult(sdiff_row[0]);
                }
            }
            
            if (threadIdx.x == 0)
                distance.ptr(queryIdx)[trainIdx] = dist;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Calc distance kernel caller

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, template <int> class Dist, typename T, typename Mask>
    void calcDistance_caller(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, 
        const Mask& mask, const DevMem2Df& distance)
    {
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        dim3 grid(queryDescs.rows, divUp(trainDescs.rows, BLOCK_DIM_Y), 1);

        calcDistance<BLOCK_DIM_X, BLOCK_DIM_Y, Dist<BLOCK_DIM_X>, T><<<grid, threads>>>(
            queryDescs, trainDescs, mask, distance);

        cudaSafeCall( cudaThreadSynchronize() );
    }
        
    ///////////////////////////////////////////////////////////////////////////////
    // reduceMin

    template <int BLOCK_SIZE> 
    __device__ void warpReduceMinIdx(volatile float* sdist, volatile int* strainIdx, float& myMin, int tid)
    {
        if (tid < 32)
        {
            if (BLOCK_SIZE >= 64) 
            { 
                float reg = sdist[tid + 32];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 32];
                }
            }
            if (BLOCK_SIZE >= 32) 
            { 
                float reg = sdist[tid + 16];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 16];
                }
            }
            if (BLOCK_SIZE >= 16) 
            { 
                float reg = sdist[tid + 8];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 8];
                }
            }
            if (BLOCK_SIZE >= 8) 
            { 
                float reg = sdist[tid + 4];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 4];
                }
            }
            if (BLOCK_SIZE >= 4) 
            { 
                float reg = sdist[tid + 2];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 2];
                } 
            }
            if (BLOCK_SIZE >= 2) 
            { 
                float reg = sdist[tid + 1];

                if (reg < myMin)
                {
                    sdist[tid] = myMin = reg;
                    strainIdx[tid] = strainIdx[tid + 1];
                }
            }
        }
    }
    
    template <int BLOCK_SIZE> 
    __device__ void reduceMinIdx(const float* dist, int n, float* sdist, int* strainIdx)
    {
        const int tid = threadIdx.x;
        
        float myMin = numeric_limits_gpu<float>::max();
        int myMinIdx = -1;

        for (int i = tid; i < n; i += BLOCK_SIZE)
        {
            float reg = dist[i];
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
            float reg = sdist[tid + 256];

            if (reg < myMin)
            {
                sdist[tid] = myMin = reg;
                strainIdx[tid] = strainIdx[tid + 256];
            }
            __syncthreads(); 
        }
        if (BLOCK_SIZE >= 256 && tid < 128) 
        {
            float reg = sdist[tid + 128];

            if (reg < myMin)
            {
                sdist[tid] = myMin = reg;
                strainIdx[tid] = strainIdx[tid + 128];
            }
            __syncthreads(); 
        }
        if (BLOCK_SIZE >= 128 && tid < 64) 
        {
            float reg = sdist[tid + 64];

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

    template <int BLOCK_SIZE>
    __global__ void findBestMatch(DevMem2Df allDist_, int i, PtrStepi trainIdx_, PtrStepf distance_)
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
            if (dist < numeric_limits_gpu<float>::max())
            {
                int bestIdx = strainIdx[0];
                allDist[bestIdx] = numeric_limits_gpu<float>::max();
                trainIdx[i] = bestIdx;
                distance[i] = dist;
            }
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // find knn match kernel caller

    template <int BLOCK_SIZE>
    void findKnnMatch_caller(int knn, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist)
    {
        dim3 threads(BLOCK_SIZE, 1, 1);
        dim3 grid(trainIdx.rows, 1, 1);

        for (int i = 0; i < knn; ++i)
            findBestMatch<BLOCK_SIZE><<<grid, threads>>>(allDist, i, trainIdx, distance);
        
        cudaSafeCall( cudaThreadSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // knn match caller

    template <typename T>
    void knnMatchL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn,
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist)
    {
        if (mask.data)
        {
            calcDistance_caller<16, 16, L1Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                SingleMask(mask), allDist);
        }
        else
        {
            calcDistance_caller<16, 16, L1Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                WithOutMask(), allDist);
        }

        findKnnMatch_caller<256>(knn, trainIdx, distance, allDist);
    }

    template void knnMatchL1_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL1_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL1_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL1_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL1_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL1_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);

    template <typename T>
    void knnMatchL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn,
        const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist)
    {
        if (mask.data)
        {
            calcDistance_caller<16, 16, L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                SingleMask(mask), allDist);
        }
        else
        {
            calcDistance_caller<16, 16, L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                WithOutMask(), allDist);
        }

        findKnnMatch_caller<256>(knn, trainIdx, distance, allDist);
    }

    template void knnMatchL2_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL2_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL2_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL2_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL2_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);
    template void knnMatchL2_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, int knn, const DevMem2D& mask, const DevMem2Di& trainIdx, const DevMem2Df& distance, const DevMem2Df& allDist);

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Radius Match //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match kernel

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, typename Dist, typename T, typename Mask>
    __global__ void radiusMatch(PtrStep_<T> queryDescs_, DevMem2D_<T> trainDescs_, 
        float maxDistance, Mask mask, DevMem2Di trainIdx_, unsigned int* nMatches, PtrStepf distance)
    {
        #if defined (__CUDA_ARCH__) && __CUDA_ARCH__ >= 110

        __shared__ float sdiff[BLOCK_DIM_X * BLOCK_DIM_Y];

        float* sdiff_row = sdiff + BLOCK_DIM_X * threadIdx.y;
        
        const int queryIdx = blockIdx.x;
        const T* queryDescs = queryDescs_.ptr(queryIdx);

        const int trainIdx = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
        if (trainIdx < trainDescs_.rows)
        {
            const T* trainDescs = trainDescs_.ptr(trainIdx);

            if (mask(queryIdx, trainIdx))
            {
                reduceDescDiff<BLOCK_DIM_X, Dist>(queryDescs, trainDescs, trainDescs_.cols, sdiff_row);

                if (threadIdx.x == 0)
                {
                    float dist = Dist::finalResult(sdiff_row[0]);
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

    template <int BLOCK_DIM_X, int BLOCK_DIM_Y, template <int> class Dist, typename T, typename Mask>
    void radiusMatch_caller(const DevMem2D_<T>& queryDescs, const DevMem2D_<T>& trainDescs, 
        float maxDistance, const Mask& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, 
        const DevMem2Df& distance)
    {
        dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        dim3 grid(queryDescs.rows, divUp(trainDescs.rows, BLOCK_DIM_Y), 1);

        radiusMatch<BLOCK_DIM_X, BLOCK_DIM_Y, Dist<BLOCK_DIM_X>, T><<<grid, threads>>>(
            queryDescs, trainDescs, maxDistance, mask, trainIdx, nMatches, distance);

        cudaSafeCall( cudaThreadSynchronize() );
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Radius Match kernel chooser

    template <typename T>
    void radiusMatchL1_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance,
        const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance)
    {
        if (mask.data)
        {
            radiusMatch_caller<16, 16, L1Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, SingleMask(mask), trainIdx, nMatches, distance);
        }
        else
        {
            radiusMatch_caller<16, 16, L1Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, WithOutMask(), trainIdx, nMatches, distance);
        }
    }

    template void radiusMatchL1_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL1_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL1_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL1_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL1_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL1_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);

    template <typename T>
    void radiusMatchL2_gpu(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance,
        const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance)
    {
        if (mask.data)
        {
            radiusMatch_caller<16, 16, L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, SingleMask(mask), trainIdx, nMatches, distance);
        }
        else
        {
            radiusMatch_caller<16, 16, L2Dist>((DevMem2D_<T>)queryDescs, (DevMem2D_<T>)trainDescs, 
                maxDistance, WithOutMask(), trainIdx, nMatches, distance);
        }
    }

    template void radiusMatchL2_gpu<unsigned char >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL2_gpu<char          >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL2_gpu<unsigned short>(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL2_gpu<short         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL2_gpu<int           >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
    template void radiusMatchL2_gpu<float         >(const DevMem2D& queryDescs, const DevMem2D& trainDescs, float maxDistance, const DevMem2D& mask, const DevMem2Di& trainIdx, unsigned int* nMatches, const DevMem2Df& distance);
}}}
