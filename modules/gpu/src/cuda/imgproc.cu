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

namespace imgproc
{
    texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex;

    __global__ void kernel_remap(const float *mapx, const float *mapy, size_t map_step, unsigned char* out, size_t out_step, int width, int height)
    {    
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x < width && y < height)
        {
            int idx = y * (map_step >> 2) + x; /* map_step >> 2  <=> map_step / sizeof(float)*/

            float xcoo = mapx[idx];
            float ycoo = mapy[idx];
            
            out[y * out_step + x] = (unsigned char)(255.f * tex2D(tex, xcoo, ycoo));            
        }
    }
}

namespace cv { namespace gpu { namespace impl {
    extern "C" void remap_gpu(const DevMem2D& src, const DevMem2D_<float>& xmap, const DevMem2D_<float>& ymap, DevMem2D dst, size_t width, size_t height)
    {
        dim3 block(16, 16, 1);
        dim3 grid(1, 1, 1);
        grid.x = divUp( width, block.x);
        grid.y = divUp(height, block.y);

        ::imgproc::tex.filterMode = cudaFilterModeLinear;	    
        ::imgproc::tex.addressMode[0] = ::imgproc::tex.addressMode[1] = cudaAddressModeWrap;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
        cudaSafeCall( cudaBindTexture2D(0, ::imgproc::tex, src.ptr, desc, width, height, src.step) );

        ::imgproc::kernel_remap<<<grid, block>>>(xmap.ptr, ymap.ptr, xmap.step, dst.ptr, dst.step, width, height);

        cudaSafeCall( cudaThreadSynchronize() );  
        cudaSafeCall( cudaUnbindTexture(::imgproc::tex) );
    }
}}}