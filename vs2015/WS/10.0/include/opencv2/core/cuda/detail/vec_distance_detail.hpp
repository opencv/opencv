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

#ifndef __OPENCV_CUDA_VEC_DISTANCE_DETAIL_HPP__
#define __OPENCV_CUDA_VEC_DISTANCE_DETAIL_HPP__

#include "../datamov_utils.hpp"

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    namespace vec_distance_detail
    {
        template <int THREAD_DIM, int N> struct UnrollVecDiffCached
        {
            template <typename Dist, typename T1, typename T2>
            static __device__ void calcCheck(const T1* vecCached, const T2* vecGlob, int len, Dist& dist, int ind)
            {
                if (ind < len)
                {
                    T1 val1 = *vecCached++;

                    T2 val2;
                    ForceGlob<T2>::Load(vecGlob, ind, val2);

                    dist.reduceIter(val1, val2);

                    UnrollVecDiffCached<THREAD_DIM, N - 1>::calcCheck(vecCached, vecGlob, len, dist, ind + THREAD_DIM);
                }
            }

            template <typename Dist, typename T1, typename T2>
            static __device__ void calcWithoutCheck(const T1* vecCached, const T2* vecGlob, Dist& dist)
            {
                T1 val1 = *vecCached++;

                T2 val2;
                ForceGlob<T2>::Load(vecGlob, 0, val2);
                vecGlob += THREAD_DIM;

                dist.reduceIter(val1, val2);

                UnrollVecDiffCached<THREAD_DIM, N - 1>::calcWithoutCheck(vecCached, vecGlob, dist);
            }
        };
        template <int THREAD_DIM> struct UnrollVecDiffCached<THREAD_DIM, 0>
        {
            template <typename Dist, typename T1, typename T2>
            static __device__ __forceinline__ void calcCheck(const T1*, const T2*, int, Dist&, int)
            {
            }

            template <typename Dist, typename T1, typename T2>
            static __device__ __forceinline__ void calcWithoutCheck(const T1*, const T2*, Dist&)
            {
            }
        };

        template <int THREAD_DIM, int MAX_LEN, bool LEN_EQ_MAX_LEN> struct VecDiffCachedCalculator;
        template <int THREAD_DIM, int MAX_LEN> struct VecDiffCachedCalculator<THREAD_DIM, MAX_LEN, false>
        {
            template <typename Dist, typename T1, typename T2>
            static __device__ __forceinline__ void calc(const T1* vecCached, const T2* vecGlob, int len, Dist& dist, int tid)
            {
                UnrollVecDiffCached<THREAD_DIM, MAX_LEN / THREAD_DIM>::calcCheck(vecCached, vecGlob, len, dist, tid);
            }
        };
        template <int THREAD_DIM, int MAX_LEN> struct VecDiffCachedCalculator<THREAD_DIM, MAX_LEN, true>
        {
            template <typename Dist, typename T1, typename T2>
            static __device__ __forceinline__ void calc(const T1* vecCached, const T2* vecGlob, int len, Dist& dist, int tid)
            {
                UnrollVecDiffCached<THREAD_DIM, MAX_LEN / THREAD_DIM>::calcWithoutCheck(vecCached, vecGlob + tid, dist);
            }
        };
    } // namespace vec_distance_detail
}}} // namespace cv { namespace cuda { namespace cudev

//! @endcond

#endif // __OPENCV_CUDA_VEC_DISTANCE_DETAIL_HPP__
