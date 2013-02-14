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

using namespace std;
using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

cv::gpu::Stream::Stream() { throw_nogpu(); }
cv::gpu::Stream::~Stream() {}
cv::gpu::Stream::Stream(const Stream&) { throw_nogpu(); }
Stream& cv::gpu::Stream::operator=(const Stream&) { throw_nogpu(); return *this; }
bool cv::gpu::Stream::queryIfComplete() { throw_nogpu(); return false; }
void cv::gpu::Stream::waitForCompletion() { throw_nogpu(); }
void cv::gpu::Stream::enqueueDownload(const GpuMat&, Mat&) { throw_nogpu(); }
void cv::gpu::Stream::enqueueDownload(const GpuMat&, CudaMem&) { throw_nogpu(); }
void cv::gpu::Stream::enqueueUpload(const CudaMem&, GpuMat&) { throw_nogpu(); }
void cv::gpu::Stream::enqueueUpload(const Mat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::Stream::enqueueCopy(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::Stream::enqueueMemSet(GpuMat&, Scalar) { throw_nogpu(); }
void cv::gpu::Stream::enqueueMemSet(GpuMat&, Scalar, const GpuMat&) { throw_nogpu(); }
void cv::gpu::Stream::enqueueConvert(const GpuMat&, GpuMat&, int, double, double) { throw_nogpu(); }
void cv::gpu::Stream::enqueueHostCallback(StreamCallback, void*) { throw_nogpu(); }
Stream& cv::gpu::Stream::Null() { throw_nogpu(); static Stream s; return s; }
cv::gpu::Stream::operator bool() const { throw_nogpu(); return false; }
cv::gpu::Stream::Stream(Impl*) { throw_nogpu(); }
void cv::gpu::Stream::create() { throw_nogpu(); }
void cv::gpu::Stream::release() { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

#include "opencv2/gpu/stream_accessor.hpp"

namespace cv { namespace gpu
{
    void copyWithMask(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream);
    void convertTo(const GpuMat& src, GpuMat& dst, double alpha, double beta, cudaStream_t stream);
    void setTo(GpuMat& src, Scalar s, cudaStream_t stream);
    void setTo(GpuMat& src, Scalar s, const GpuMat& mask, cudaStream_t stream);
}}

struct Stream::Impl
{
    static cudaStream_t getStream(const Impl* impl)
    {
        return impl ? impl->stream : 0;
    }

    cudaStream_t stream;
    int ref_counter;
};

cudaStream_t cv::gpu::StreamAccessor::getStream(const Stream& stream)
{
    return Stream::Impl::getStream(stream.impl);
}

cv::gpu::Stream::Stream() : impl(0)
{
    create();
}

cv::gpu::Stream::~Stream()
{
    release();
}

cv::gpu::Stream::Stream(const Stream& stream) : impl(stream.impl)
{
    if (impl)
        CV_XADD(&impl->ref_counter, 1);
}

Stream& cv::gpu::Stream::operator =(const Stream& stream)
{
    if (this != &stream)
    {
        release();
        impl = stream.impl;
        if (impl)
            CV_XADD(&impl->ref_counter, 1);
    }

    return *this;
}

bool cv::gpu::Stream::queryIfComplete()
{
    cudaStream_t stream = Impl::getStream(impl);
    cudaError_t err = cudaStreamQuery(stream);

    if (err == cudaErrorNotReady || err == cudaSuccess)
        return err == cudaSuccess;

    cudaSafeCall(err);
    return false;
}

void cv::gpu::Stream::waitForCompletion()
{
    cudaStream_t stream = Impl::getStream(impl);
    cudaSafeCall( cudaStreamSynchronize(stream) );
}

void cv::gpu::Stream::enqueueDownload(const GpuMat& src, Mat& dst)
{
    // if not -> allocation will be done, but after that dst will not point to page locked memory
    CV_Assert( src.size() == dst.size() && src.type() == dst.type() );

    cudaStream_t stream = Impl::getStream(impl);
    size_t bwidth = src.cols * src.elemSize();
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, cudaMemcpyDeviceToHost, stream) );
}

void cv::gpu::Stream::enqueueDownload(const GpuMat& src, CudaMem& dst)
{
    dst.create(src.size(), src.type(), CudaMem::ALLOC_PAGE_LOCKED);

    cudaStream_t stream = Impl::getStream(impl);
    size_t bwidth = src.cols * src.elemSize();
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, cudaMemcpyDeviceToHost, stream) );
}

void cv::gpu::Stream::enqueueUpload(const CudaMem& src, GpuMat& dst)
{
    dst.create(src.size(), src.type());

    cudaStream_t stream = Impl::getStream(impl);
    size_t bwidth = src.cols * src.elemSize();
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, cudaMemcpyHostToDevice, stream) );
}

void cv::gpu::Stream::enqueueUpload(const Mat& src, GpuMat& dst)
{
    dst.create(src.size(), src.type());

    cudaStream_t stream = Impl::getStream(impl);
    size_t bwidth = src.cols * src.elemSize();
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, cudaMemcpyHostToDevice, stream) );
}

void cv::gpu::Stream::enqueueCopy(const GpuMat& src, GpuMat& dst)
{
    dst.create(src.size(), src.type());

    cudaStream_t stream = Impl::getStream(impl);
    size_t bwidth = src.cols * src.elemSize();
    cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, cudaMemcpyDeviceToDevice, stream) );
}

void cv::gpu::Stream::enqueueMemSet(GpuMat& src, Scalar val)
{
    const int sdepth = src.depth();

    if (sdepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    cudaStream_t stream = Impl::getStream(impl);

    if (val[0] == 0.0 && val[1] == 0.0 && val[2] == 0.0 && val[3] == 0.0)
    {
        cudaSafeCall( cudaMemset2DAsync(src.data, src.step, 0, src.cols * src.elemSize(), src.rows, stream) );
        return;
    }

    if (sdepth == CV_8U)
    {
        int cn = src.channels();

        if (cn == 1 || (cn == 2 && val[0] == val[1]) || (cn == 3 && val[0] == val[1] && val[0] == val[2]) || (cn == 4 && val[0] == val[1] && val[0] == val[2] && val[0] == val[3]))
        {
            int ival = saturate_cast<uchar>(val[0]);
            cudaSafeCall( cudaMemset2DAsync(src.data, src.step, ival, src.cols * src.elemSize(), src.rows, stream) );
            return;
        }
    }

    setTo(src, val, stream);
}

void cv::gpu::Stream::enqueueMemSet(GpuMat& src, Scalar val, const GpuMat& mask)
{
    const int sdepth = src.depth();

    if (sdepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    CV_Assert(mask.type() == CV_8UC1);

    cudaStream_t stream = Impl::getStream(impl);

    setTo(src, val, mask, stream);
}

void cv::gpu::Stream::enqueueConvert(const GpuMat& src, GpuMat& dst, int dtype, double alpha, double beta)
{
    if (dtype < 0)
        dtype = src.type();
    else
        dtype = CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels());

    const int sdepth = src.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    bool noScale = fabs(alpha - 1) < numeric_limits<double>::epsilon() && fabs(beta) < numeric_limits<double>::epsilon();

    if (sdepth == ddepth && noScale)
    {
        enqueueCopy(src, dst);
        return;
    }

    dst.create(src.size(), dtype);

    cudaStream_t stream = Impl::getStream(impl);
    convertTo(src, dst, alpha, beta, stream);
}

#if CUDA_VERSION >= 5000

namespace
{
    struct CallbackData
    {
        cv::gpu::Stream::StreamCallback callback;
        void* userData;
        Stream stream;
    };

    void CUDART_CB cudaStreamCallback(cudaStream_t, cudaError_t status, void* userData)
    {
        CallbackData* data = reinterpret_cast<CallbackData*>(userData);
        data->callback(data->stream, static_cast<int>(status), data->userData);
        delete data;
    }
}

#endif

void cv::gpu::Stream::enqueueHostCallback(StreamCallback callback, void* userData)
{
#if CUDA_VERSION >= 5000
    CallbackData* data = new CallbackData;
    data->callback = callback;
    data->userData = userData;
    data->stream = *this;

    cudaStream_t stream = Impl::getStream(impl);

    cudaSafeCall( cudaStreamAddCallback(stream, cudaStreamCallback, data, 0) );
#else
    (void) callback;
    (void) userData;
    CV_Error(CV_StsNotImplemented, "This function requires CUDA 5.0");
#endif
}

cv::gpu::Stream& cv::gpu::Stream::Null()
{
    static Stream s((Impl*) 0);
    return s;
}

cv::gpu::Stream::operator bool() const
{
    return impl && impl->stream;
}

cv::gpu::Stream::Stream(Impl* impl_) : impl(impl_)
{
}

void cv::gpu::Stream::create()
{
    if (impl)
        release();

    cudaStream_t stream;
    cudaSafeCall( cudaStreamCreate( &stream ) );

    impl = (Stream::Impl*) fastMalloc(sizeof(Stream::Impl));

    impl->stream = stream;
    impl->ref_counter = 1;
}

void cv::gpu::Stream::release()
{
    if (impl && CV_XADD(&impl->ref_counter, -1) == 1)
    {
        cudaSafeCall( cudaStreamDestroy(impl->stream) );
        cv::fastFree(impl);
    }
}

#endif /* !defined (HAVE_CUDA) */
