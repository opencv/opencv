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

#ifndef __OPENCV_GPU_UTILITY_DETAIL_HPP__
#define __OPENCV_GPU_UTILITY_DETAIL_HPP__

namespace cv { namespace gpu { namespace device
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////////
        // Reduction

        template <int n> struct WarpReductor
        {
            template <typename T, typename Op> static __device__ __forceinline__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                if (tid < n)
                    data[tid] = partial_reduction;                
                if (n > 32) __syncthreads();

                if (n > 32)
                {
                    if (tid < n - 32) 
                        data[tid] = partial_reduction = op(partial_reduction, data[tid + 32]);
                    if (tid < 16)
                    {
                        data[tid] = partial_reduction = op(partial_reduction, data[tid + 16]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  8]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  4]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  2]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  1]);
                    }
                }
                else if (n > 16)
                {
                    if (tid < n - 16) 
                        data[tid] = partial_reduction = op(partial_reduction, data[tid + 16]);
                    if (tid < 8)
                    {
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  8]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  4]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  2]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  1]);
                    }
                }
                else if (n > 8)
                {
                    if (tid < n - 8) 
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  8]);
                    if (tid < 4)
                    {
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  4]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  2]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  1]);
                    }
                }
                else if (n > 4)
                {
                    if (tid < n - 4) 
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  4]);
                    if (tid < 2)
                    {
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  2]);
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  1]);
                    }
                }   
                else if (n > 2)
                {
                    if (tid < n - 2) 
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  2]);
                    if (tid < 2)
                    {
                        data[tid] = partial_reduction = op(partial_reduction, data[tid +  1]);
                    }
                }      
            }
        };
        template <> struct WarpReductor<64>
        {
            template <typename T, typename Op> static __device__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                data[tid] = partial_reduction;
                __syncthreads();
                
                if (tid < 32) 
                {
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 32]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 16]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 8 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 4 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 2 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 1 ]); 
                }
            }
        };
        template <> struct WarpReductor<32>
        {
            template <typename T, typename Op> static __device__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                data[tid] = partial_reduction;
                
                if (tid < 16) 
                {
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 16]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 8 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 4 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 2 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 1 ]); 
                }
            }
        };
        template <> struct WarpReductor<16>
        {
            template <typename T, typename Op> static __device__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                data[tid] = partial_reduction;
                
                if (tid < 8) 
                {
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 8 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 4 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 2 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 1 ]); 
                }
            }
        };
        template <> struct WarpReductor<8>
        {
            template <typename T, typename Op> static __device__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                data[tid] = partial_reduction;
                
                if (tid < 4) 
                {
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 4 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 2 ]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 1 ]); 
                }
            }
        };

        template <bool warp> struct ReductionDispatcher;
        template <> struct ReductionDispatcher<true>
        {
            template <int n, typename T, typename Op> static __device__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                WarpReductor<n>::reduce(data, partial_reduction, tid, op);
            }
        };
        template <> struct ReductionDispatcher<false>
        {
            template <int n, typename T, typename Op> static __device__ void reduce(volatile T* data, T& partial_reduction, int tid, const Op& op)
            {
                if (tid < n)
                    data[tid] = partial_reduction;
                __syncthreads();


                if (n == 512) { if (tid < 256) { data[tid] = partial_reduction = op(partial_reduction, data[tid + 256]); } __syncthreads(); }
                if (n >= 256) { if (tid < 128) { data[tid] = partial_reduction = op(partial_reduction, data[tid + 128]); } __syncthreads(); }
                if (n >= 128) { if (tid <  64) { data[tid] = partial_reduction = op(partial_reduction, data[tid +  64]); } __syncthreads(); }

                if (tid < 32)
                {
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 32]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid + 16]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid +  8]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid +  4]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid +  2]);
                    data[tid] = partial_reduction = op(partial_reduction, data[tid +  1]);
                }
            }
        };

        
        template <int n> struct PredValWarpReductor;
        template <> struct PredValWarpReductor<64>
        {
            template <typename T, typename V, typename Pred> 
            static __device__ void reduce(T& myData, V& myVal, volatile T* sdata, V* sval, int tid, const Pred& pred)
            {
                if (tid < 32)
                {
                    myData = sdata[tid];
                    myVal = sval[tid];

                    T reg = sdata[tid + 32];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 32];
                    }

                    reg = sdata[tid + 16];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 16];
                    }

                    reg = sdata[tid + 8];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 8];
                    }

                    reg = sdata[tid + 4];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 4];
                    }
                
                    reg = sdata[tid + 2];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 2];
                    }
                
                    reg = sdata[tid + 1];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 1];
                    }
                }
            }
        };
        template <> struct PredValWarpReductor<32>
        {
            template <typename T, typename V, typename Pred> 
            static __device__ void reduce(T& myData, V& myVal, volatile T* sdata, V* sval, int tid, const Pred& pred)
            {
                if (tid < 16)
                {
                    myData = sdata[tid];
                    myVal = sval[tid];

                    T reg = sdata[tid + 16];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 16];
                    }

                    reg = sdata[tid + 8];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 8];
                    }

                    reg = sdata[tid + 4];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 4];
                    }
                
                    reg = sdata[tid + 2];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 2];
                    }
                
                    reg = sdata[tid + 1];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 1];
                    }
                }
            }
        };

        template <> struct PredValWarpReductor<16>
        {
            template <typename T, typename V, typename Pred> 
            static __device__ void reduce(T& myData, V& myVal, volatile T* sdata, V* sval, int tid, const Pred& pred)
            {
                if (tid < 8)
                {
                    myData = sdata[tid];
                    myVal = sval[tid];

                    T reg = reg = sdata[tid + 8];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 8];
                    }

                    reg = sdata[tid + 4];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 4];
                    }
                
                    reg = sdata[tid + 2];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 2];
                    }
                
                    reg = sdata[tid + 1];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 1];
                    }
                }
            }
        };
        template <> struct PredValWarpReductor<8>
        {
            template <typename T, typename V, typename Pred> 
            static __device__ void reduce(T& myData, V& myVal, volatile T* sdata, V* sval, int tid, const Pred& pred)
            {
                if (tid < 4)
                {
                    myData = sdata[tid];
                    myVal = sval[tid];

                    T reg = reg = sdata[tid + 4];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 4];
                    }
                
                    reg = sdata[tid + 2];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 2];
                    }
                
                    reg = sdata[tid + 1];
                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 1];
                    }
                }
            }
        };

        template <bool warp> struct PredValReductionDispatcher;
        template <> struct PredValReductionDispatcher<true>
        {
            template <int n, typename T, typename V, typename Pred> static __device__ void reduce(T& myData, V& myVal, volatile T* sdata, V* sval, int tid, const Pred& pred)
            {
                PredValWarpReductor<n>::reduce(myData, myVal, sdata, sval, tid, pred);
            }
        };
        template <> struct PredValReductionDispatcher<false>
        {
            template <int n, typename T, typename V, typename Pred> static __device__ void reduce(T& myData, V& myVal, volatile T* sdata, V* sval, int tid, const Pred& pred)
            {
                myData = sdata[tid];
                myVal = sval[tid];

                if (n >= 512 && tid < 256) 
                {
                    T reg = sdata[tid + 256];

                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 256];
                    }
                    __syncthreads(); 
                }
                if (n >= 256 && tid < 128) 
                {
                    T reg = sdata[tid + 128];

                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 128];
                    }
                    __syncthreads(); 
                }
                if (n >= 128 && tid < 64) 
                {
                    T reg = sdata[tid + 64];

                    if (pred(reg, myData))
                    {
                        sdata[tid] = myData = reg;
                        sval[tid] = myVal = sval[tid + 64];
                    }
                    __syncthreads(); 
                }        

                if (tid < 32)
                {
                    if (n >= 64) 
                    { 
                        T reg = sdata[tid + 32];

                        if (pred(reg, myData))
                        {
                            sdata[tid] = myData = reg;
                            sval[tid] = myVal = sval[tid + 32];
                        }
                    }
                    if (n >= 32) 
                    { 
                        T reg = sdata[tid + 16];

                        if (pred(reg, myData))
                        {
                            sdata[tid] = myData = reg;
                            sval[tid] = myVal = sval[tid + 16];
                        }
                    }
                    if (n >= 16) 
                    { 
                        T reg = sdata[tid + 8];

                        if (pred(reg, myData))
                        {
                            sdata[tid] = myData = reg;
                            sval[tid] = myVal = sval[tid + 8];
                        }
                    }
                    if (n >= 8) 
                    { 
                        T reg = sdata[tid + 4];

                        if (pred(reg, myData))
                        {
                            sdata[tid] = myData = reg;
                            sval[tid] = myVal = sval[tid + 4];
                        }
                    }
                    if (n >= 4) 
                    { 
                        T reg = sdata[tid + 2];

                        if (pred(reg, myData))
                        {
                            sdata[tid] = myData = reg;
                            sval[tid] = myVal = sval[tid + 2];
                        } 
                    }
                    if (n >= 2) 
                    { 
                        T reg = sdata[tid + 1];

                        if (pred(reg, myData))
                        {
                            sdata[tid] = myData = reg;
                            sval[tid] = myVal = sval[tid + 1];
                        }
                    }
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////////////
        // Vector Distance

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
    }
}}}

#endif // __OPENCV_GPU_UTILITY_DETAIL_HPP__
