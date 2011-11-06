/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2009-2010, NVIDIA Corporation, all rights reserved.
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

#ifndef _ncv_alg_hpp_
#define _ncv_alg_hpp_

#include "NCV.hpp"


template<typename T>
static T divUp(T a, T b)
{
    return (a + b - 1) / b;
}


template<typename T>
struct functorAddValues
{
    static __device__ __inline__ void reduce(T &in1out, T &in2)
    {
        in1out += in2;
    }
};


template<typename T>
struct functorMinValues
{
    static __device__ __inline__ void reduce(T &in1out, T &in2)
    {
        in1out = in1out > in2 ? in2 : in1out;
    }
};


template<typename T>
struct functorMaxValues
{
    static __device__ __inline__ void reduce(T &in1out, T &in2)
    {
        in1out = in1out > in2 ? in1out : in2;
    }
};


template<typename Tdata, class Tfunc, Ncv32u nThreads>
static __device__ Tdata subReduce(Tdata threadElem)
{
    Tfunc functor;

    __shared__ Tdata reduceArr[nThreads];
    reduceArr[threadIdx.x] = threadElem;
    __syncthreads();

    if (nThreads >= 256 && threadIdx.x < 128)
    {
        functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 128]);
    }
    __syncthreads();

    if (nThreads >= 128 && threadIdx.x < 64)
    {
        functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 64]);
    }
    __syncthreads();

    if (threadIdx.x < 32)
    {
        if (nThreads >= 64)
        {
            functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 32]);
        }
        if (nThreads >= 32)
        {
            functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 16]);
        }
        functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 8]);
        functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 4]);
        functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 2]);
        functor.reduce(reduceArr[threadIdx.x], reduceArr[threadIdx.x + 1]);
    }

    __syncthreads();
    return reduceArr[0];
}


#endif //_ncv_alg_hpp_
