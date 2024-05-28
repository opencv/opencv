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

#ifndef OPENCV_CUDA_REDUCE_DETAIL_HPP
#define OPENCV_CUDA_REDUCE_DETAIL_HPP

#include <thrust/tuple.h>
#include "../warp.hpp"
#include "../warp_shuffle.hpp"

//! @cond IGNORED

namespace cv { namespace cuda { namespace device
{
    namespace reduce_detail
    {
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

        template <unsigned int I, unsigned int N>
        struct For
        {
            template <class PointerTuple, class ValTuple>
            static __device__ void loadToSmem(const PointerTuple& smem, const ValTuple& val, unsigned int tid)
            {
                thrust::get<I>(smem)[tid] = thrust::get<I>(val);

                For<I + 1, N>::loadToSmem(smem, val, tid);
            }
            template <class PointerTuple, class ValTuple>
            static __device__ void loadFromSmem(const PointerTuple& smem, const ValTuple& val, unsigned int tid)
            {
                thrust::get<I>(val) = thrust::get<I>(smem)[tid];

                For<I + 1, N>::loadFromSmem(smem, val, tid);
            }

            template <class PointerTuple, class ValTuple, class OpTuple>
            static __device__ void merge(const PointerTuple& smem, const ValTuple& val, unsigned int tid, unsigned int delta, const OpTuple& op)
            {
                typename GetType<typename thrust::tuple_element<I, PointerTuple>::type>::type reg = thrust::get<I>(smem)[tid + delta];
                thrust::get<I>(smem)[tid] = thrust::get<I>(val) = thrust::get<I>(op)(thrust::get<I>(val), reg);

                For<I + 1, N>::merge(smem, val, tid, delta, op);
            }
            template <class ValTuple, class OpTuple>
            static __device__ void mergeShfl(const ValTuple& val, unsigned int delta, unsigned int width, const OpTuple& op)
            {
                typename GetType<typename thrust::tuple_element<I, ValTuple>::type>::type reg = shfl_down(thrust::get<I>(val), delta, width);
                thrust::get<I>(val) = thrust::get<I>(op)(thrust::get<I>(val), reg);

                For<I + 1, N>::mergeShfl(val, delta, width, op);
            }
        };
        template <unsigned int N>
        struct For<N, N>
        {
            template <class PointerTuple, class ValTuple>
            static __device__ void loadToSmem(const PointerTuple&, const ValTuple&, unsigned int)
            {
            }
            template <class PointerTuple, class ValTuple>
            static __device__ void loadFromSmem(const PointerTuple&, const ValTuple&, unsigned int)
            {
            }

            template <class PointerTuple, class ValTuple, class OpTuple>
            static __device__ void merge(const PointerTuple&, const ValTuple&, unsigned int, unsigned int, const OpTuple&)
            {
            }
            template <class ValTuple, class OpTuple>
            static __device__ void mergeShfl(const ValTuple&, unsigned int, unsigned int, const OpTuple&)
            {
            }
        };

        template <typename T>
        __device__ __forceinline__ void loadToSmem(volatile T* smem, T& val, unsigned int tid)
        {
            smem[tid] = val;
        }
        template <typename T>
        __device__ __forceinline__ void loadFromSmem(volatile T* smem, T& val, unsigned int tid)
        {
            val = smem[tid];
        }

        template <typename T, class Op>
        __device__ __forceinline__ void merge(volatile T* smem, T& val, unsigned int tid, unsigned int delta, const Op& op)
        {
            T reg = smem[tid + delta];
            smem[tid] = val = op(val, reg);
        }

        template <typename T, class Op>
        __device__ __forceinline__ void mergeShfl(T& val, unsigned int delta, unsigned int width, const Op& op)
        {
            T reg = shfl_down(val, delta, width);
            val = op(val, reg);
        }

#if (CUDART_VERSION < 12040) // details: https://github.com/opencv/opencv_contrib/issues/3690
        template <typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
                  typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9>
        __device__ __forceinline__ void loadToSmem(const thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                                       const thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                                       unsigned int tid)
        {
            For<0, thrust::tuple_size<thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value>::loadToSmem(smem, val, tid);
        }

        template <typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
                  typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9>
        __device__ __forceinline__ void loadFromSmem(const thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                                         const thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                                         unsigned int tid)
        {
            For<0, thrust::tuple_size<thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value>::loadFromSmem(smem, val, tid);
        }

        template <typename P0, typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9,
                  typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
                  class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
        __device__ __forceinline__ void merge(const thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9>& smem,
                                              const thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                              unsigned int tid,
                                              unsigned int delta,
                                              const thrust::tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
        {
            For<0, thrust::tuple_size<thrust::tuple<P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> >::value>::merge(smem, val, tid, delta, op);
        }
        template <typename R0, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7, typename R8, typename R9,
                  class Op0, class Op1, class Op2, class Op3, class Op4, class Op5, class Op6, class Op7, class Op8, class Op9>
        __device__ __forceinline__ void mergeShfl(const thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9>& val,
                                                  unsigned int delta,
                                                  unsigned int width,
                                                  const thrust::tuple<Op0, Op1, Op2, Op3, Op4, Op5, Op6, Op7, Op8, Op9>& op)
        {
            For<0, thrust::tuple_size<thrust::tuple<R0, R1, R2, R3, R4, R5, R6, R7, R8, R9> >::value>::mergeShfl(val, delta, width, op);
        }
#else
        template <typename... P, typename... R>
        __device__ __forceinline__ void loadToSmem(const thrust::tuple<P...>& smem, const thrust::tuple<R...>& val, unsigned int tid)
        {
            For<0, thrust::tuple_size<thrust::tuple<P...> >::value>::loadToSmem(smem, val, tid);
        }

        template <typename... P, typename... R>
        __device__ __forceinline__ void loadFromSmem(const thrust::tuple<P...>& smem, const thrust::tuple<R...>& val, unsigned int tid)
        {
            For<0, thrust::tuple_size<thrust::tuple<P...> >::value>::loadFromSmem(smem, val, tid);
        }

        template <typename... P, typename... R, class... Op>
        __device__ __forceinline__ void merge(const thrust::tuple<P...>& smem, const thrust::tuple<R...>& val, unsigned int tid, unsigned int delta, const thrust::tuple<Op...>& op)
        {
            For<0, thrust::tuple_size<thrust::tuple<P...> >::value>::merge(smem, val, tid, delta, op);
        }

        template <typename... R, class... Op>
        __device__ __forceinline__ void mergeShfl(const thrust::tuple<R...>& val, unsigned int delta, unsigned int width, const thrust::tuple<Op...>& op)
        {
            For<0, thrust::tuple_size<thrust::tuple<R...> >::value>::mergeShfl(val, delta, width, op);
        }
#endif
        template <unsigned int N> struct Generic
        {
            template <typename Pointer, typename Reference, class Op>
            static __device__ void reduce(Pointer smem, Reference val, unsigned int tid, Op op)
            {
                loadToSmem(smem, val, tid);
                if (N >= 32)
                    __syncthreads();

                if (N >= 2048)
                {
                    if (tid < 1024)
                        merge(smem, val, tid, 1024, op);

                    __syncthreads();
                }
                if (N >= 1024)
                {
                    if (tid < 512)
                        merge(smem, val, tid, 512, op);

                    __syncthreads();
                }
                if (N >= 512)
                {
                    if (tid < 256)
                        merge(smem, val, tid, 256, op);

                    __syncthreads();
                }
                if (N >= 256)
                {
                    if (tid < 128)
                        merge(smem, val, tid, 128, op);

                    __syncthreads();
                }
                if (N >= 128)
                {
                    if (tid < 64)
                        merge(smem, val, tid, 64, op);

                    __syncthreads();
                }
                if (N >= 64)
                {
                    if (tid < 32)
                        merge(smem, val, tid, 32, op);
                }

                if (tid < 16)
                {
                    merge(smem, val, tid, 16, op);
                    merge(smem, val, tid, 8, op);
                    merge(smem, val, tid, 4, op);
                    merge(smem, val, tid, 2, op);
                    merge(smem, val, tid, 1, op);
                }
            }
        };

        template <unsigned int I, typename Pointer, typename Reference, class Op>
        struct Unroll
        {
            static __device__ void loopShfl(Reference val, Op op, unsigned int N)
            {
                mergeShfl(val, I, N, op);
                Unroll<I / 2, Pointer, Reference, Op>::loopShfl(val, op, N);
            }
            static __device__ void loop(Pointer smem, Reference val, unsigned int tid, Op op)
            {
                merge(smem, val, tid, I, op);
                Unroll<I / 2, Pointer, Reference, Op>::loop(smem, val, tid, op);
            }
        };
        template <typename Pointer, typename Reference, class Op>
        struct Unroll<0, Pointer, Reference, Op>
        {
            static __device__ void loopShfl(Reference, Op, unsigned int)
            {
            }
            static __device__ void loop(Pointer, Reference, unsigned int, Op)
            {
            }
        };

        template <unsigned int N> struct WarpOptimized
        {
            template <typename Pointer, typename Reference, class Op>
            static __device__ void reduce(Pointer smem, Reference val, unsigned int tid, Op op)
            {
            #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
                CV_UNUSED(smem);
                CV_UNUSED(tid);

                Unroll<N / 2, Pointer, Reference, Op>::loopShfl(val, op, N);
            #else
                loadToSmem(smem, val, tid);

                if (tid < N / 2)
                    Unroll<N / 2, Pointer, Reference, Op>::loop(smem, val, tid, op);
            #endif
            }
        };

        template <unsigned int N> struct GenericOptimized32
        {
            enum { M = N / 32 };

            template <typename Pointer, typename Reference, class Op>
            static __device__ void reduce(Pointer smem, Reference val, unsigned int tid, Op op)
            {
                const unsigned int laneId = Warp::laneId();

            #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
                Unroll<16, Pointer, Reference, Op>::loopShfl(val, op, warpSize);

                if (laneId == 0)
                    loadToSmem(smem, val, tid / 32);
            #else
                loadToSmem(smem, val, tid);

                if (laneId < 16)
                    Unroll<16, Pointer, Reference, Op>::loop(smem, val, tid, op);

                __syncthreads();

                if (laneId == 0)
                    loadToSmem(smem, val, tid / 32);
            #endif

                __syncthreads();

                loadFromSmem(smem, val, tid);

                if (tid < 32)
                {
                #if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
                    Unroll<M / 2, Pointer, Reference, Op>::loopShfl(val, op, M);
                #else
                    Unroll<M / 2, Pointer, Reference, Op>::loop(smem, val, tid, op);
                #endif
                }
            }
        };

        template <bool val, class T1, class T2> struct StaticIf;
        template <class T1, class T2> struct StaticIf<true, T1, T2>
        {
            typedef T1 type;
        };
        template <class T1, class T2> struct StaticIf<false, T1, T2>
        {
            typedef T2 type;
        };

        template <unsigned int N> struct IsPowerOf2
        {
            enum { value = ((N != 0) && !(N & (N - 1))) };
        };

        template <unsigned int N> struct Dispatcher
        {
            typedef typename StaticIf<
                (N <= 32) && IsPowerOf2<N>::value,
                WarpOptimized<N>,
                typename StaticIf<
                    (N <= 1024) && IsPowerOf2<N>::value,
                    GenericOptimized32<N>,
                    Generic<N>
                >::type
            >::type reductor;
        };
    }
}}}

//! @endcond

#endif // OPENCV_CUDA_REDUCE_DETAIL_HPP
