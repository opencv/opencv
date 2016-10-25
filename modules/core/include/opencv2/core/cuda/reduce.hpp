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

#ifndef OPENCV_CUDA_REDUCE_HPP
#define OPENCV_CUDA_REDUCE_HPP

#include <thrust/tuple.h>
#include "detail/reduce.hpp"
#include "detail/reduce_key_val.hpp"

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    template <int N, typename T, class Op>
    __device__ __forceinline__ void reduce(volatile T* smem, T& val, unsigned int tid, const Op& op)
    {
        reduce_detail::Dispatcher<N>::reductor::template reduce<volatile T*, T&, const Op&>(smem, val, tid, op);
    }
    template <int N,
              typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
              typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
              class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
    __device__ __forceinline__ void reduce(const thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                           const thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                           unsigned int tid,
                                           const thrust::tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
    {
        reduce_detail::Dispatcher<N>::reductor::template reduce<
                const thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>&,
                const thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>&,
                const thrust::tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>&>(smem, val, tid, op);
    }

    template <unsigned int N, typename K, typename V, class Cmp>
    __device__ __forceinline__ void reduceKeyVal(volatile K* skeys, K& key, volatile V* svals, V& val, unsigned int tid, const Cmp& cmp)
    {
        reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&, volatile V*, V&, const Cmp&>(skeys, key, svals, val, tid, cmp);
    }
    template <unsigned int N,
              typename K,
              typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
              class Cmp>
    __device__ __forceinline__ void reduceKeyVal(volatile K* skeys, K& key,
                                                 const thrust::tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                                 const thrust::tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                                 unsigned int tid, const Cmp& cmp)
    {
        reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<volatile K*, K&,
                const thrust::tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>&,
                const thrust::tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>&,
                const Cmp&>(skeys, key, svals, val, tid, cmp);
    }
    template <unsigned int N,
              typename KP0, typename KP1, typename KP2, typename KP3, typename KP4, typename KP5, typename KP6, typename KP7, typename KP8, typename KP9,
              typename KR0, typename KR1, typename KR2, typename KR3, typename KR4, typename KR5, typename KR6, typename KR7, typename KR8, typename KR9,
              typename VP0, typename VP1, typename VP2, typename VP3, typename VP4, typename VP5, typename VP6, typename VP7, typename VP8, typename VP9,
              typename VR0, typename VR1, typename VR2, typename VR3, typename VR4, typename VR5, typename VR6, typename VR7, typename VR8, typename VR9,
              class Cmp0, class Cmp1, class Cmp2, class Cmp3, class Cmp4, class Cmp5, class Cmp6, class Cmp7, class Cmp8, class Cmp9>
    __device__ __forceinline__ void reduceKeyVal(const thrust::tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>& skeys,
                                                 const thrust::tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>& key,
                                                 const thrust::tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>& svals,
                                                 const thrust::tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>& val,
                                                 unsigned int tid,
                                                 const thrust::tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>& cmp)
    {
        reduce_key_val_detail::Dispatcher<N>::reductor::template reduce<
                const thrust::tuple<KP0, KP1, KP2, KP3, KP4, KP5, KP6, KP7, KP8, KP9>&,
                const thrust::tuple<KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7, KR8, KR9>&,
                const thrust::tuple<VP0, VP1, VP2, VP3, VP4, VP5, VP6, VP7, VP8, VP9>&,
                const thrust::tuple<VR0, VR1, VR2, VR3, VR4, VR5, VR6, VR7, VR8, VR9>&,
                const thrust::tuple<Cmp0, Cmp1, Cmp2, Cmp3, Cmp4, Cmp5, Cmp6, Cmp7, Cmp8, Cmp9>&
                >(skeys, key, svals, val, tid, cmp);
    }

    // smem_tuple

    template <typename T0>
    __device__ __forceinline__
    thrust::tuple<volatile T0*>
    smem_tuple(T0* t0)
    {
        return thrust::make_tuple((volatile T0*) t0);
    }

    template <typename T0, typename T1>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*>
    smem_tuple(T0* t0, T1* t1)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1);
    }

    template <typename T0, typename T1, typename T2>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*>
    smem_tuple(T0* t0, T1* t1, T2* t2)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2);
    }

    template <typename T0, typename T1, typename T2, typename T3>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3);
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*, volatile T4*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3, T4* t4)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3, (volatile T4*) t4);
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*, volatile T4*, volatile T5*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3, (volatile T4*) t4, (volatile T5*) t5);
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*, volatile T4*, volatile T5*, volatile T6*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3, (volatile T4*) t4, (volatile T5*) t5, (volatile T6*) t6);
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*, volatile T4*, volatile T5*, volatile T6*, volatile T7*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3, (volatile T4*) t4, (volatile T5*) t5, (volatile T6*) t6, (volatile T7*) t7);
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*, volatile T4*, volatile T5*, volatile T6*, volatile T7*, volatile T8*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7, T8* t8)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3, (volatile T4*) t4, (volatile T5*) t5, (volatile T6*) t6, (volatile T7*) t7, (volatile T8*) t8);
    }

    template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
    __device__ __forceinline__
    thrust::tuple<volatile T0*, volatile T1*, volatile T2*, volatile T3*, volatile T4*, volatile T5*, volatile T6*, volatile T7*, volatile T8*, volatile T9*>
    smem_tuple(T0* t0, T1* t1, T2* t2, T3* t3, T4* t4, T5* t5, T6* t6, T7* t7, T8* t8, T9* t9)
    {
        return thrust::make_tuple((volatile T0*) t0, (volatile T1*) t1, (volatile T2*) t2, (volatile T3*) t3, (volatile T4*) t4, (volatile T5*) t5, (volatile T6*) t6, (volatile T7*) t7, (volatile T8*) t8, (volatile T9*) t9);
    }
}}}

//! @endcond

#endif // OPENCV_CUDA_REDUCE_HPP
