/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#pragma once

#ifndef __OPENCV_CUDEV_WARP_REDUCE_DETAIL_HPP__
#define __OPENCV_CUDEV_WARP_REDUCE_DETAIL_HPP__

#include "../../common.hpp"
#include "../../util/tuple.hpp"
#include "../../warp/shuffle.hpp"

namespace cv { namespace cudev {

namespace warp_reduce_detail
{
    // GetType

    template <typename T> struct GetType;

    template <typename T> struct GetType<T*>
    {
        typedef T type;
    };

    template <typename T> struct GetType<volatile T*>
    {
        typedef T type;
    };

    template <typename T> struct GetType<T&>
    {
        typedef T type;
    };

    // For

    template <int I, int N> struct For
    {
        template <class PointerTuple, class ValTuple>
        __device__ static void loadToSmem(const PointerTuple& smem, const ValTuple& val, uint tid)
        {
            get<I>(smem)[tid] = get<I>(val);

            For<I + 1, N>::loadToSmem(smem, val, tid);
        }

        template <class PointerTuple, class ValTuple, class OpTuple>
        __device__ static void merge(const PointerTuple& smem, const ValTuple& val, uint tid, uint delta, const OpTuple& op)
        {
            typename GetType<typename tuple_element<I, PointerTuple>::type>::type reg = get<I>(smem)[tid + delta];
            get<I>(smem)[tid] = get<I>(val) = get<I>(op)(get<I>(val), reg);

            For<I + 1, N>::merge(smem, val, tid, delta, op);
        }

#if CV_CUDEV_ARCH >= 300
        template <class ValTuple, class OpTuple>
        __device__ static void mergeShfl(const ValTuple& val, uint delta, uint width, const OpTuple& op)
        {
            typename GetType<typename tuple_element<I, ValTuple>::type>::type reg = shfl_down(get<I>(val), delta, width);
            get<I>(val) = get<I>(op)(get<I>(val), reg);

            For<I + 1, N>::mergeShfl(val, delta, width, op);
        }
#endif
    };

    template <int N> struct For<N, N>
    {
        template <class PointerTuple, class ValTuple>
        __device__ __forceinline__ static void loadToSmem(const PointerTuple&, const ValTuple&, uint)
        {
        }

        template <class PointerTuple, class ValTuple, class OpTuple>
        __device__ __forceinline__ static void merge(const PointerTuple&, const ValTuple&, uint, uint, const OpTuple&)
        {
        }

#if CV_CUDEV_ARCH >= 300
        template <class ValTuple, class OpTuple>
        __device__ __forceinline__ static void mergeShfl(const ValTuple&, uint, uint, const OpTuple&)
        {
        }
#endif
    };

    // loadToSmem

    template <typename T>
    __device__ __forceinline__ void loadToSmem(volatile T* smem, T& val, uint tid)
    {
        smem[tid] = val;
    }

    template <typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
              typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9>
    __device__ __forceinline__ void loadToSmem(const tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                               const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                               uint tid)
    {
        For<0, tuple_size<tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value>::loadToSmem(smem, val, tid);
    }

    // merge

    template <typename T, class Op>
    __device__ __forceinline__ void merge(volatile T* smem, T& val, uint tid, uint delta, const Op& op)
    {
        T reg = smem[tid + delta];
        smem[tid] = val = op(val, reg);
    }

    template <typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
              typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
              class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
    __device__ __forceinline__ void merge(const tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                          const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                          uint tid,
                                          uint delta,
                                          const tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
    {
        For<0, tuple_size<tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value>::merge(smem, val, tid, delta, op);
    }

    // mergeShfl

#if CV_CUDEV_ARCH >= 300
    template <typename T, class Op>
    __device__ __forceinline__ void mergeShfl(T& val, uint delta, uint width, const Op& op)
    {
        T reg = shfl_down(val, delta, width);
        val = op(val, reg);
    }

    template <typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
              class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
    __device__ __forceinline__ void mergeShfl(const tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                              uint delta,
                                              uint width,
                                              const tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
    {
        For<0, tuple_size<tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9> >::value>::mergeShfl(val, delta, width, op);
    }
#endif

    // WarpReductor

    struct WarpReductor
    {
        template <typename Pointer, typename Reference, class Op>
        __device__ static void reduce(Pointer smem, Reference val, uint tid, Op op)
        {
        #if CV_CUDEV_ARCH >= 300
            (void) smem;
            (void) tid;

            mergeShfl(val, 16, 32, op);
            mergeShfl(val, 8, 32, op);
            mergeShfl(val, 4, 32, op);
            mergeShfl(val, 2, 32, op);
            mergeShfl(val, 1, 32, op);
        #else
            loadToSmem(smem, val, tid);

            if (tid < 16)
            {
                merge(smem, val, tid, 16, op);
                merge(smem, val, tid, 8, op);
                merge(smem, val, tid, 4, op);
                merge(smem, val, tid, 2, op);
                merge(smem, val, tid, 1, op);
            }
        #endif
        }
    };
}

}}

#endif
