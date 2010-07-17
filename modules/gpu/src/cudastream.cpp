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

#include "precomp.hpp"
//#include "opencv2/gpu/stream_access.hpp"

using namespace cv;
using namespace cv::gpu;


cv::gpu::CudaStream::CudaStream() //: impl( (Impl*)fastMalloc(sizeof(Impl)) )
{
    //cudaSafeCall( cudaStreamCreate( &impl->stream) );
}
cv::gpu::CudaStream::~CudaStream()
{
    if (impl)
    {
        cudaSafeCall( cudaStreamDestroy( *(cudaStream_t*)impl ) );
        cv::fastFree( impl );
    }
}

bool cv::gpu::CudaStream::queryIfComplete()
{
    //cudaError_t err = cudaStreamQuery( *(cudaStream_t*)impl );

    //if (err == cudaSuccess)
    //    return true;

    //if (err == cudaErrorNotReady)
    //    return false;

    ////cudaErrorInvalidResourceHandle
    //cudaSafeCall( err );
    return true;
}
void cv::gpu::CudaStream::waitForCompletion()
{
    cudaSafeCall( cudaStreamSynchronize(  *(cudaStream_t*)impl ) );
}

void cv::gpu::CudaStream::enqueueDownload(const GpuMat& src, Mat& dst)
{
//    cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, src.cols * src.elemSize(), src.rows, cudaMemcpyDeviceToHost,
}
void cv::gpu::CudaStream::enqueueUpload(const Mat& src, GpuMat& dst)
{
    CV_Assert(!"Not implemented");
}
void cv::gpu::CudaStream::enqueueCopy(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(!"Not implemented");
}

void cv::gpu::CudaStream::enqueueMemSet(const GpuMat& src, Scalar val)
{
    CV_Assert(!"Not implemented");
}

void cv::gpu::CudaStream::enqueueMemSet(const GpuMat& src, Scalar val, const GpuMat& mask)
{
    CV_Assert(!"Not implemented");
}

void cv::gpu::CudaStream::enqueueConvert(const GpuMat& src, GpuMat& dst, int type)
{
    CV_Assert(!"Not implemented");
}

//struct cudaStream_t& cv::gpu::CudaStream::getStream() { return stream; }


