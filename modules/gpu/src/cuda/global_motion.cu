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
// Copyright (C) 1993-2011, NVIDIA Corporation, all rights reserved.
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

#include "thrust/device_ptr.h"
#include "thrust/remove.h"
#include "thrust/functional.h"
#include "internal_shared.hpp"

using namespace thrust;

namespace cv { namespace gpu { namespace device { namespace globmotion {

__constant__ float cml[9];
__constant__ float cmr[9];

int compactPoints(int N, float *points0, float *points1, const uchar *mask)
{
    thrust::device_ptr<float2> dpoints0((float2*)points0);
    thrust::device_ptr<float2> dpoints1((float2*)points1);
    thrust::device_ptr<const uchar> dmask(mask);

    return thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(dpoints0, dpoints1)),
                             thrust::make_zip_iterator(thrust::make_tuple(dpoints0 + N, dpoints1 + N)),
                             dmask, thrust::not1(thrust::identity<uchar>()))
           - make_zip_iterator(make_tuple(dpoints0, dpoints1));
}


__global__ void calcWobbleSuppressionMapsKernel(
        const int left, const int idx, const int right, const int width, const int height,
        PtrElemStepf mapx, PtrElemStepf mapy)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height)
    {
        float xl = cml[0]*x + cml[1]*y + cml[2];
        float yl = cml[3]*x + cml[4]*y + cml[5];
        float izl = 1.f / (cml[6]*x + cml[7]*y + cml[8]);
        xl *= izl;
        yl *= izl;

        float xr = cmr[0]*x + cmr[1]*y + cmr[2];
        float yr = cmr[3]*x + cmr[4]*y + cmr[5];
        float izr = 1.f / (cmr[6]*x + cmr[7]*y + cmr[8]);
        xr *= izr;
        yr *= izr;

        float wl = idx - left;
        float wr = right - idx;
        mapx(y,x) = (wr * xl + wl * xr) / (wl + wr);
        mapy(y,x) = (wr * yl + wl * yr) / (wl + wr);
    }
}


void calcWobbleSuppressionMaps(
        int left, int idx, int right, int width, int height,
        const float *ml, const float *mr, DevMem2Df mapx, DevMem2Df mapy)
{
    cudaSafeCall(cudaMemcpyToSymbol(cml, ml, 9*sizeof(float)));
    cudaSafeCall(cudaMemcpyToSymbol(cmr, mr, 9*sizeof(float)));

    dim3 threads(32, 8);
    dim3 grid(divUp(width, threads.x), divUp(height, threads.y));

    calcWobbleSuppressionMapsKernel<<<grid, threads>>>(
            left, idx, right, width, height, mapx, mapy);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

}}}}
