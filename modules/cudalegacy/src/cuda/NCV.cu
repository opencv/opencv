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

#include <iostream>
#include <vector>

#include "opencv2/cudalegacy/NCV.hpp"

//===================================================================
//
// Operations with rectangles
//
//===================================================================


const Ncv32u NUMTHREADS_DRAWRECTS = 32;
const Ncv32u NUMTHREADS_DRAWRECTS_LOG2 = 5;


template <class T>
__global__ void drawRects(T *d_dst,
                          Ncv32u dstStride,
                          Ncv32u dstWidth,
                          Ncv32u dstHeight,
                          NcvRect32u *d_rects,
                          Ncv32u numRects,
                          T color)
{
    Ncv32u blockId = blockIdx.y * 65535 + blockIdx.x;
    if (blockId > numRects * 4)
    {
        return;
    }

    NcvRect32u curRect = d_rects[blockId >> 2];
    NcvBool bVertical = blockId & 0x1;
    NcvBool bTopLeft = blockId & 0x2;

    Ncv32u pt0x, pt0y;
    if (bVertical)
    {
        Ncv32u numChunks = (curRect.height + NUMTHREADS_DRAWRECTS - 1) >> NUMTHREADS_DRAWRECTS_LOG2;

        pt0x = bTopLeft ? curRect.x : curRect.x + curRect.width - 1;
        pt0y = curRect.y;

        if (pt0x < dstWidth)
        {
            for (Ncv32u chunkId = 0; chunkId < numChunks; chunkId++)
            {
                Ncv32u ptY = pt0y + chunkId * NUMTHREADS_DRAWRECTS + threadIdx.x;
                if (ptY < pt0y + curRect.height && ptY < dstHeight)
                {
                    d_dst[ptY * dstStride + pt0x] = color;
                }
            }
        }
    }
    else
    {
        Ncv32u numChunks = (curRect.width + NUMTHREADS_DRAWRECTS - 1) >> NUMTHREADS_DRAWRECTS_LOG2;

        pt0x = curRect.x;
        pt0y = bTopLeft ? curRect.y : curRect.y + curRect.height - 1;

        if (pt0y < dstHeight)
        {
            for (Ncv32u chunkId = 0; chunkId < numChunks; chunkId++)
            {
                Ncv32u ptX = pt0x + chunkId * NUMTHREADS_DRAWRECTS + threadIdx.x;
                if (ptX < pt0x + curRect.width && ptX < dstWidth)
                {
                    d_dst[pt0y * dstStride + ptX] = color;
                }
            }
        }
    }
}


template <class T>
static NCVStatus drawRectsWrapperDevice(T *d_dst,
                                        Ncv32u dstStride,
                                        Ncv32u dstWidth,
                                        Ncv32u dstHeight,
                                        NcvRect32u *d_rects,
                                        Ncv32u numRects,
                                        T color,
                                        cudaStream_t cuStream)
{
    CV_UNUSED(cuStream);
    ncvAssertReturn(d_dst != NULL && d_rects != NULL, NCV_NULL_PTR);
    ncvAssertReturn(dstWidth > 0 && dstHeight > 0, NCV_DIMENSIONS_INVALID);
    ncvAssertReturn(dstStride >= dstWidth, NCV_INVALID_STEP);
    ncvAssertReturn(numRects <= dstWidth * dstHeight, NCV_DIMENSIONS_INVALID);

    if (numRects == 0)
    {
        return NCV_SUCCESS;
    }

    dim3 grid(numRects * 4);
    dim3 block(NUMTHREADS_DRAWRECTS);
    if (grid.x > 65535)
    {
        grid.y = (grid.x + 65534) / 65535;
        grid.x = 65535;
    }

    drawRects<T><<<grid, block>>>(d_dst, dstStride, dstWidth, dstHeight, d_rects, numRects, color);

    ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);

    return NCV_SUCCESS;
}


NCVStatus ncvDrawRects_8u_device(Ncv8u *d_dst,
                                 Ncv32u dstStride,
                                 Ncv32u dstWidth,
                                 Ncv32u dstHeight,
                                 NcvRect32u *d_rects,
                                 Ncv32u numRects,
                                 Ncv8u color,
                                 cudaStream_t cuStream)
{
    return drawRectsWrapperDevice(d_dst, dstStride, dstWidth, dstHeight, d_rects, numRects, color, cuStream);
}


NCVStatus ncvDrawRects_32u_device(Ncv32u *d_dst,
                                  Ncv32u dstStride,
                                  Ncv32u dstWidth,
                                  Ncv32u dstHeight,
                                  NcvRect32u *d_rects,
                                  Ncv32u numRects,
                                  Ncv32u color,
                                  cudaStream_t cuStream)
{
    return drawRectsWrapperDevice(d_dst, dstStride, dstWidth, dstHeight, d_rects, numRects, color, cuStream);
}
