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

#include "cuda_shared.hpp"

using namespace cv::gpu;

namespace cv { namespace gpu { namespace imgproc {

texture<unsigned char, 2> imageTex_8U;
texture<unsigned char, 2> templTex_8U;


__global__ void matchTemplateKernel_8U_SqDiff(int w, int h, DevMem2Df result)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < result.cols && y < result.rows)
    {
        float sum = 0.f;
        float delta;

        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                delta = (float)tex2D(imageTex_8U, x + j, y + i) - 
                        (float)tex2D(templTex_8U, j, i);
                sum += delta * delta;
            }
        }

        result.ptr(y)[x] = sum;
    }
}


void matchTemplateCaller_8U_SqDiff(const DevMem2D image, const DevMem2D templ, DevMem2Df result)
{
    dim3 threads(32, 8);
    dim3 grid(divUp(image.cols - templ.cols + 1, threads.x), 
              divUp(image.rows - templ.rows + 1, threads.y));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaBindTexture2D(0, imageTex_8U, image.data, desc, image.cols, image.rows, image.step);
    cudaBindTexture2D(0, templTex_8U, templ.data, desc, templ.cols, templ.rows, templ.step);
    imageTex_8U.filterMode = cudaFilterModePoint;
    templTex_8U.filterMode = cudaFilterModePoint;

    matchTemplateKernel_8U_SqDiff<<<grid, threads>>>(templ.cols, templ.rows, result);
    cudaSafeCall(cudaThreadSynchronize());
    cudaSafeCall(cudaUnbindTexture(imageTex_8U));
    cudaSafeCall(cudaUnbindTexture(templTex_8U));
}


}}}
