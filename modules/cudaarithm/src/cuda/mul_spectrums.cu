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

#if !defined CUDA_DISABLER

#include "cvconfig.h"

#ifdef HAVE_CUFFT

#include <cufft.h>

#include "opencv2/core/cuda/common.hpp"

namespace cv { namespace cuda { namespace device
{
    //////////////////////////////////////////////////////////////////////////
    // mulSpectrums

    __global__ void mulSpectrumsKernel(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, PtrStepSz<cufftComplex> c)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < c.cols && y < c.rows)
        {
            c.ptr(y)[x] = cuCmulf(a.ptr(y)[x], b.ptr(y)[x]);
        }
    }


    void mulSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, PtrStepSz<cufftComplex> c, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

        mulSpectrumsKernel<<<grid, threads, 0, stream>>>(a, b, c);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    //////////////////////////////////////////////////////////////////////////
    // mulSpectrums_CONJ

    __global__ void mulSpectrumsKernel_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, PtrStepSz<cufftComplex> c)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < c.cols && y < c.rows)
        {
            c.ptr(y)[x] = cuCmulf(a.ptr(y)[x], cuConjf(b.ptr(y)[x]));
        }
    }


    void mulSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, PtrStepSz<cufftComplex> c, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

        mulSpectrumsKernel_CONJ<<<grid, threads, 0, stream>>>(a, b, c);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    //////////////////////////////////////////////////////////////////////////
    // mulAndScaleSpectrums

    __global__ void mulAndScaleSpectrumsKernel(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, PtrStepSz<cufftComplex> c)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < c.cols && y < c.rows)
        {
            cufftComplex v = cuCmulf(a.ptr(y)[x], b.ptr(y)[x]);
            c.ptr(y)[x] = make_cuFloatComplex(cuCrealf(v) * scale, cuCimagf(v) * scale);
        }
    }


    void mulAndScaleSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, PtrStepSz<cufftComplex> c, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

        mulAndScaleSpectrumsKernel<<<grid, threads, 0, stream>>>(a, b, scale, c);
        cudaSafeCall( cudaGetLastError() );

        if (stream)
            cudaSafeCall( cudaDeviceSynchronize() );
    }


    //////////////////////////////////////////////////////////////////////////
    // mulAndScaleSpectrums_CONJ

    __global__ void mulAndScaleSpectrumsKernel_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, PtrStepSz<cufftComplex> c)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < c.cols && y < c.rows)
        {
            cufftComplex v = cuCmulf(a.ptr(y)[x], cuConjf(b.ptr(y)[x]));
            c.ptr(y)[x] = make_cuFloatComplex(cuCrealf(v) * scale, cuCimagf(v) * scale);
        }
    }


    void mulAndScaleSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, PtrStepSz<cufftComplex> c, cudaStream_t stream)
    {
        dim3 threads(256);
        dim3 grid(divUp(c.cols, threads.x), divUp(c.rows, threads.y));

        mulAndScaleSpectrumsKernel_CONJ<<<grid, threads, 0, stream>>>(a, b, scale, c);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}}} // namespace cv { namespace cuda { namespace cudev

#endif // HAVE_CUFFT

#endif /* CUDA_DISABLER */
