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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#include <icf.hpp>

namespace cv { namespace gpu {


 namespace device {

__global__ void rgb2grayluv(const uchar3* __restrict__ rgb, uchar* __restrict__ hog,
                            const int rgbPitch, const int hogPitch)
{
}

__global__ void gray2hog(const uchar* __restrict__ gray, uchar* __restrict__ hog,
                         const int pitch)
{
}

__global__ void decimate(const uchar* __restrict__ hogluv, uchar* __restrict__ shrank,
                        const int inPitch, const int outPitch )
{
}

__global__ void intRow(const uchar* __restrict__ hogluv, ushort* __restrict__ sum,
                       const int inPitch, const int outPitch)
{

}

__global__ void intCol(ushort* __restrict__ sum, const int pitch)
{

}


__global__ void detect(const cv::gpu::icf::Cascade cascade, const uchar* __restrict__ hogluv, const int pitch)
{
    cascade.detectAt();
}

}

void __device icf::Cascade::detectAt() const
{

}

void icf::Cascade::detect(const cv::gpu::PtrStepSzb& hogluv, cudaStream_t stream) const
{
    // detection kernel

}

void icf::ChannelStorage::frame(const cv::gpu::PtrStepSz<uchar3>& rgb, cudaStream_t stream)
{
    // color convertin kernel
    dim3 block(32, 8);
    dim3 grid(FRAME_WIDTH / 32, FRAME_HEIGHT / 8);

    uchar * channels = (uchar*)dmem.ptr(FRAME_HEIGHT * HOG_BINS);
    device::rgb2grayluv<<<grid, block, 0, stream>>>((uchar3*)rgb.ptr(), channels, rgb.step, dmem.step);
    cudaSafeCall( cudaGetLastError());

    // hog calculation kernel
    channels = (uchar*)dmem.ptr(FRAME_HEIGHT * HOG_LUV_BINS);
    device::gray2hog<<<grid, block, 0, stream>>>(channels, (uchar*)dmem.ptr(), dmem.step);
    cudaSafeCall( cudaGetLastError() );

    const int shrWidth  = FRAME_WIDTH / shrinkage;
    const int shrHeight = FRAME_HEIGHT / shrinkage;

    // decimate kernel
    grid = dim3(shrWidth / 32, shrHeight / 8);
    device::decimate<<<grid, block, 0, stream>>>((uchar*)dmem.ptr(), (uchar*)shrunk.ptr(), dmem.step, shrunk.step);
    cudaSafeCall( cudaGetLastError() );

    // integrate rows
    block = dim3(shrWidth, 1);
    grid = dim3(shrHeight * HOG_LUV_BINS, 1);
    device::intRow<<<grid, block, 0, stream>>>((uchar*)shrunk.ptr(), (ushort*)hogluv.ptr(), shrunk.step, hogluv.step);
    cudaSafeCall( cudaGetLastError() );

    // integrate cols
    block = dim3(128, 1);
    grid = dim3(shrWidth * HOG_LUV_BINS, 1);
    device::intCol<<<grid, block, 0, stream>>>((ushort*)hogluv.ptr(), hogluv.step);
    cudaSafeCall( cudaGetLastError() );
}

}}