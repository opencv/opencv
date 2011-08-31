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

#ifndef __OPENCV_GPU_UTILITY_HPP__
#define __OPENCV_GPU_UTILITY_HPP__

#include "internal_shared.hpp"
#include "saturate_cast.hpp"
#include "datamov_utils.hpp"
#include "functional.hpp"
#include "detail/utility_detail.hpp"

#define OPENCV_GPU_LOG_WARP_SIZE	    (5)
#define OPENCV_GPU_WARP_SIZE	        (1 << OPENCV_GPU_LOG_WARP_SIZE)
#define OPENCV_GPU_LOG_MEM_BANKS        ((__CUDA_ARCH__ >= 200) ? 5 : 4) // 32 banks on fermi, 16 on tesla
#define OPENCV_GPU_MEM_BANKS            (1 << OPENCV_GPU_LOG_MEM_BANKS)

namespace cv {  namespace gpu { namespace device
{
    ///////////////////////////////////////////////////////////////////////////////
    // swap

    template <typename T> void __device__ __forceinline__ swap(T& a, T& b) 
    {
        const T temp = a;
        a = b;
        b = temp;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Mask Reader

    struct SingleMask
    {
        explicit __host__ __device__ __forceinline__ SingleMask(const PtrStep& mask_) : mask(mask_) {}
        
        __device__ __forceinline__ bool operator()(int y, int x) const
        {            
            return mask.ptr(y)[x] != 0;
        }

        const PtrStep mask;
    };

    struct MaskCollection
    {
        explicit __host__ __device__ __forceinline__ MaskCollection(PtrStep* maskCollection_) : maskCollection(maskCollection_) {}

        __device__ __forceinline__ void next()
        {
            curMask = *maskCollection++;
        }
        __device__ __forceinline__ void setMask(int z)
        {
            curMask = maskCollection[z];
        }
        
        __device__ __forceinline__ bool operator()(int y, int x) const
        {
            uchar val;
            return curMask.data == 0 || (ForceGlob<uchar>::Load(curMask.ptr(y), x, val), (val != 0));
        }

        const PtrStep* maskCollection;
        PtrStep curMask;
    };

    struct WithOutMask
    {
        __device__ __forceinline__ void next() const
        {
        }
        __device__ __forceinline__ void setMask(int) const
        {
        }

        __device__ __forceinline__ bool operator()(int, int) const
        {
            return true;
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Reduction

    // reduction
    template <int n, typename T, typename Op> __device__ __forceinline__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
    {
        StaticAssert<n >= 8 && n <= 512>::check();
        detail::ReductionDispatcher<n <= 64>::reduce<n>(data, partial_reduction, tid, op);
    }

    template <int n, typename T, typename V, typename Pred> 
    __device__ __forceinline__ void reducePredVal(volatile T* sdata, T& myData, V* sval, V& myVal, int tid, const Pred& pred)
    {
        StaticAssert<n >= 8 && n <= 512>::check();
        detail::PredValReductionDispatcher<n <= 64>::reduce<n>(myData, myVal, sdata, sval, tid, pred);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Vector Distance

    template <typename T> struct L1Dist
    {
        typedef int value_type;
        typedef int result_type;

        __device__ __forceinline__ L1Dist() : mySum(0) {}

        __device__ __forceinline__ void reduceIter(int val1, int val2)
        {
            mySum = __sad(val1, val2, mySum);
        }

        template <int THREAD_DIM> __device__ __forceinline__ void reduceAll(int* smem, int tid)
        {
            reduce<THREAD_DIM>(smem, mySum, tid, plus<volatile int>());
        }

        __device__ __forceinline__ operator int() const
        {
            return mySum;
        }

        int mySum;
    };
    template <> struct L1Dist<float>
    {
        typedef float value_type;
        typedef float result_type;

        __device__ __forceinline__ L1Dist() : mySum(0.0f) {}

        __device__ __forceinline__ void reduceIter(float val1, float val2)
        {
            mySum += ::fabs(val1 - val2);
        }

        template <int THREAD_DIM> __device__ __forceinline__ void reduceAll(float* smem, int tid)
        {
            reduce<THREAD_DIM>(smem, mySum, tid, plus<volatile float>());
        }

        __device__ __forceinline__ operator float() const
        {
            return mySum;
        }

        float mySum;
    };

    struct L2Dist
    {
        typedef float value_type;
        typedef float result_type;

        __device__ __forceinline__ L2Dist() : mySum(0.0f) {}

        __device__ __forceinline__ void reduceIter(float val1, float val2)
        {
            float reg = val1 - val2;
            mySum += reg * reg;
        }

        template <int THREAD_DIM> __device__ __forceinline__ void reduceAll(float* smem, int tid)
        {
            reduce<THREAD_DIM>(smem, mySum, tid, plus<volatile float>());
        }

        __device__ __forceinline__ operator float() const
        {
            return sqrtf(mySum);
        }

        float mySum;
    };

    struct HammingDist
    {
        typedef int value_type;
        typedef int result_type;

        __device__ __forceinline__ HammingDist() : mySum(0) {}

        __device__ __forceinline__ void reduceIter(int val1, int val2)
        {
            mySum += __popc(val1 ^ val2);
        }

        template <int THREAD_DIM> __device__ __forceinline__ void reduceAll(int* smem, int tid)
        {
            reduce<THREAD_DIM>(smem, mySum, tid, plus<volatile int>());
        }

        __device__ __forceinline__ operator int() const
        {
            return mySum;
        }

        int mySum;
    };

    // calc distance between two vectors in global memory
    template <int THREAD_DIM, typename Dist, typename T1, typename T2> 
    __device__ void calcVecDiffGlobal(const T1* vec1, const T2* vec2, int len, Dist& dist, typename Dist::result_type* smem, int tid)
    {
        for (int i = tid; i < len; i += THREAD_DIM)
        {
            T1 val1;
            ForceGlob<T1>::Load(vec1, i, val1);

            T2 val2;
            ForceGlob<T2>::Load(vec2, i, val2);

            dist.reduceIter(val1, val2);
        }

        dist.reduceAll<THREAD_DIM>(smem, tid);
    }

    // calc distance between two vectors, first vector is cached in register or shared memory, second vector is in global memory
    template <int THREAD_DIM, int MAX_LEN, bool LEN_EQ_MAX_LEN, typename Dist, typename T1, typename T2>
    __device__ __forceinline__ void calcVecDiffCached(const T1* vecCached, const T2* vecGlob, int len, Dist& dist, typename Dist::result_type* smem, int tid)
    {        
        detail::VecDiffCachedCalculator<THREAD_DIM, MAX_LEN, LEN_EQ_MAX_LEN>::calc(vecCached, vecGlob, len, dist, tid);
        
        dist.reduceAll<THREAD_DIM>(smem, tid);
    }

    // calc distance between two vectors in global memory
    template <int THREAD_DIM, typename T1> struct VecDiffGlobal
    {
        explicit __device__ __forceinline__ VecDiffGlobal(const T1* vec1_, int = 0, void* = 0, int = 0, int = 0)
        {
            vec1 = vec1_;
        }

        template <typename T2, typename Dist>
        __device__ __forceinline__ void calc(const T2* vec2, int len, Dist& dist, typename Dist::result_type* smem, int tid) const
        {
            calcVecDiffGlobal<THREAD_DIM>(vec1, vec2, len, dist, smem, tid);
        }

        const T1* vec1;
    };

    // calc distance between two vectors, first vector is cached in register memory, second vector is in global memory
    template <int THREAD_DIM, int MAX_LEN, bool LEN_EQ_MAX_LEN, typename U> struct VecDiffCachedRegister
    {
        template <typename T1> __device__ __forceinline__ VecDiffCachedRegister(const T1* vec1, int len, U* smem, int glob_tid, int tid)
        {
            if (glob_tid < len)
                smem[glob_tid] = vec1[glob_tid];
            __syncthreads();

            U* vec1ValsPtr = vec1Vals;

            #pragma unroll
            for (int i = tid; i < MAX_LEN; i += THREAD_DIM)
                *vec1ValsPtr++ = smem[i];

            __syncthreads();
        }

        template <typename T2, typename Dist>
        __device__ __forceinline__ void calc(const T2* vec2, int len, Dist& dist, typename Dist::result_type* smem, int tid) const
        {
            calcVecDiffCached<THREAD_DIM, MAX_LEN, LEN_EQ_MAX_LEN>(vec1Vals, vec2, len, dist, smem, tid);
        }

        U vec1Vals[MAX_LEN / THREAD_DIM];
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    // Solve linear system

    // solve 2x2 linear system Ax=b
    template <typename T> __device__ __forceinline__ bool solve2x2(const T A[2][2], const T b[2], T x[2])
    {
        T det = A[0][0] * A[1][1] - A[1][0] * A[0][1];

        if (det != 0)
        {
            double invdet = 1.0 / det;

            x[0] = saturate_cast<T>(invdet * (b[0] * A[1][1] - b[1] * A[0][1]));

            x[1] = saturate_cast<T>(invdet * (A[0][0] * b[1] - A[1][0] * b[0]));

            return true;
        }

        return false;
    }

    // solve 3x3 linear system Ax=b
    template <typename T> __device__ __forceinline__ bool solve3x3(const T A[3][3], const T b[3], T x[3])
    {
        T det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
              - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
              + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

        if (det != 0)
        {
            double invdet = 1.0 / det;

            x[0] = saturate_cast<T>(invdet * 
                (b[0]    * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                 A[0][1] * (b[1]    * A[2][2] - A[1][2] * b[2]   ) +
                 A[0][2] * (b[1]    * A[2][1] - A[1][1] * b[2]   )));

            x[1] = saturate_cast<T>(invdet * 
                (A[0][0] * (b[1]    * A[2][2] - A[1][2] * b[2]   ) -
                 b[0]    * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * b[2]    - b[1]    * A[2][0])));

            x[2] = saturate_cast<T>(invdet * 
                (A[0][0] * (A[1][1] * b[2]    - b[1]    * A[2][1]) -
                 A[0][1] * (A[1][0] * b[2]    - b[1]    * A[2][0]) +
                 b[0]    * (A[1][0] * A[2][1] - A[1][1] * A[2][0])));

            return true;
        }

        return false;
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Filters    

    template <typename Ptr2D> struct PointFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ PointFilter(const Ptr2D& src_) : src(src_) {}
         
        __device__ __forceinline__ elem_type operator ()(float y, float x) const
        {
            return src(__float2int_rn(y), __float2int_rn(x));
        }

        const Ptr2D src;
    };

    template <typename Ptr2D> struct LinearFilter
    {
        typedef typename Ptr2D::elem_type elem_type;
        typedef float index_type;

        explicit __host__ __device__ __forceinline__ LinearFilter(const Ptr2D& src_) : src(src_) {}

        __device__ __forceinline__ elem_type operator ()(float y, float x) const
        {
            typedef typename TypeVec<float, VecTraits<elem_type>::cn>::vec_type work_type;

            work_type out = VecTraits<work_type>::all(0);

            const int x1 = __float2int_rd(x);
            const int y1 = __float2int_rd(y);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            elem_type src_reg = src(y1, x1);
            out = out + src_reg * ((x2 - x) * (y2 - y));

            src_reg = src(y1, x2);
            out = out + src_reg * ((x - x1) * (y2 - y));

            src_reg = src(y2, x1);
            out = out + src_reg * ((x2 - x) * (y - y1));

            src_reg = src(y2, x2);
            out = out + src_reg * ((x - x1) * (y - y1));

            return saturate_cast<elem_type>(out);
        }

        const Ptr2D src;
    };
}}}

#endif // __OPENCV_GPU_UTILITY_HPP__
