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

using namespace cv;
using namespace cv::gpu;


#if !defined (HAVE_CUDA)

void cv::gpu::CudaStream::create() { throw_nogpu(); }
void cv::gpu::CudaStream::release() { throw_nogpu(); }
cv::gpu::CudaStream::CudaStream() : impl(0) { throw_nogpu(); }
cv::gpu::CudaStream::~CudaStream() { throw_nogpu(); }
cv::gpu::CudaStream::CudaStream(const CudaStream& stream) { throw_nogpu(); }
CudaStream& cv::gpu::CudaStream::operator=(const CudaStream& stream) { throw_nogpu(); return *this; }
bool cv::gpu::CudaStream::queryIfComplete() { throw_nogpu(); return true; }
void cv::gpu::CudaStream::waitForCompletion() { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueDownload(const GpuMat& src, Mat& dst) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueDownload(const GpuMat& src, MatPL& dst) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueUpload(const MatPL& src, GpuMat& dst) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueUpload(const Mat& src, GpuMat& dst) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueCopy(const GpuMat& src, GpuMat& dst) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueMemSet(const GpuMat& src, Scalar val) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueMemSet(const GpuMat& src, Scalar val, const GpuMat& mask) { throw_nogpu(); }
void cv::gpu::CudaStream::enqueueConvert(const GpuMat& src, GpuMat& dst, int type, double a, double b) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

#include "opencv2/gpu/stream_accessor.hpp"

struct CudaStream::Impl
{
    cudaStream_t stream;
    int ref_counter;
};
namespace 
{
    template<class S, class D> void devcopy(const S& src, D& dst, cudaStream_t s, cudaMemcpyKind k)
    {
        dst.create(src.size(), src.type());
        size_t bwidth = src.cols * src.elemSize();
        cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, k, s) ); 
    };
}

CV_EXPORTS cudaStream_t cv::gpu::StreamAccessor::getStream(const CudaStream& stream) { return stream.impl->stream; };

void cv::gpu::CudaStream::create()
{
    if (impl)
        release();

    cudaStream_t stream;
    cudaSafeCall( cudaStreamCreate( &stream ) );

    impl = (CudaStream::Impl*)fastMalloc(sizeof(CudaStream::Impl));

    impl->stream = stream;
    impl->ref_counter = 1;    
}

void cv::gpu::CudaStream::release()
{
    if( impl && CV_XADD(&impl->ref_counter, -1) == 1 )
    {
        cudaSafeCall( cudaStreamDestroy( impl->stream ) );
        cv::fastFree( impl );
    }
}

cv::gpu::CudaStream::CudaStream() : impl(0) { create(); }
cv::gpu::CudaStream::~CudaStream() { release(); }

cv::gpu::CudaStream::CudaStream(const CudaStream& stream) : impl(stream.impl)
{
    if( impl )
        CV_XADD(&impl->ref_counter, 1);
}
CudaStream& cv::gpu::CudaStream::operator=(const CudaStream& stream)
{
    if( this != &stream )
    {
        if( stream.impl )
            CV_XADD(&stream.impl->ref_counter, 1);

        release();
        impl = stream.impl;        
    }
    return *this;
}

bool cv::gpu::CudaStream::queryIfComplete()
{
    cudaError_t err = cudaStreamQuery( impl->stream );

    if (err == cudaErrorNotReady || err == cudaSuccess)
        return err == cudaSuccess;

    cudaSafeCall(err);
}

void cv::gpu::CudaStream::waitForCompletion() { cudaSafeCall( cudaStreamSynchronize( impl->stream ) ); }

void cv::gpu::CudaStream::enqueueDownload(const GpuMat& src, Mat& dst) 
{ 
    // if not -> allocation will be done, but after that dst will not point to page locked memory
    CV_Assert(src.cols == dst.cols && src.rows == dst.rows && src.type() == dst.type() )
     devcopy(src, dst, impl->stream, cudaMemcpyDeviceToHost); 
}
void cv::gpu::CudaStream::enqueueDownload(const GpuMat& src, MatPL& dst) { devcopy(src, dst, impl->stream, cudaMemcpyDeviceToHost); }

void cv::gpu::CudaStream::enqueueUpload(const MatPL& src, GpuMat& dst){ devcopy(src, dst, impl->stream,   cudaMemcpyHostToDevice); }
void cv::gpu::CudaStream::enqueueUpload(const Mat& src, GpuMat& dst)  { devcopy(src, dst, impl->stream,   cudaMemcpyHostToDevice); }   
void cv::gpu::CudaStream::enqueueCopy(const GpuMat& src, GpuMat& dst) { devcopy(src, dst, impl->stream, cudaMemcpyDeviceToDevice); }

void cv::gpu::CudaStream::enqueueMemSet(const GpuMat& src, Scalar val)
{
    CV_Assert(!"Not implemented");
}

void cv::gpu::CudaStream::enqueueMemSet(const GpuMat& src, Scalar val, const GpuMat& mask)
{
    CV_Assert(!"Not implemented");
}

void cv::gpu::CudaStream::enqueueConvert(const GpuMat& src, GpuMat& dst, int type, double a, double b)
{
    CV_Assert(!"Not implemented");
}


#endif /* !defined (HAVE_CUDA) */