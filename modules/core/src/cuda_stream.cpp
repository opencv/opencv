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
using namespace cv::cuda;

////////////////////////////////////////////////////////////////
// Stream

#ifndef HAVE_CUDA

class cv::cuda::Stream::Impl
{
public:
    Impl(void* ptr = 0)
    {
        (void) ptr;
        throw_no_cuda();
    }
};

#else

class cv::cuda::Stream::Impl
{
public:
    cudaStream_t stream;

    Impl();
    Impl(cudaStream_t stream);

    ~Impl();
};

cv::cuda::Stream::Impl::Impl() : stream(0)
{
    cudaSafeCall( cudaStreamCreate(&stream) );
}

cv::cuda::Stream::Impl::Impl(cudaStream_t stream_) : stream(stream_)
{
}

cv::cuda::Stream::Impl::~Impl()
{
    if (stream)
        cudaStreamDestroy(stream);
}

cudaStream_t cv::cuda::StreamAccessor::getStream(const Stream& stream)
{
    return stream.impl_->stream;
}

#endif

cv::cuda::Stream::Stream()
{
#ifndef HAVE_CUDA
    throw_no_cuda();
#else
    impl_ = makePtr<Impl>();
#endif
}

bool cv::cuda::Stream::queryIfComplete() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    cudaError_t err = cudaStreamQuery(impl_->stream);

    if (err == cudaErrorNotReady || err == cudaSuccess)
        return err == cudaSuccess;

    cudaSafeCall(err);
    return false;
#endif
}

void cv::cuda::Stream::waitForCompletion()
{
#ifndef HAVE_CUDA
    throw_no_cuda();
#else
    cudaSafeCall( cudaStreamSynchronize(impl_->stream) );
#endif
}

void cv::cuda::Stream::waitEvent(const Event& event)
{
#ifndef HAVE_CUDA
    (void) event;
    throw_no_cuda();
#else
    cudaSafeCall( cudaStreamWaitEvent(impl_->stream, EventAccessor::getEvent(event), 0) );
#endif
}

#if defined(HAVE_CUDA) && (CUDART_VERSION >= 5000)

namespace
{
    struct CallbackData
    {
        Stream::StreamCallback callback;
        void* userData;

        CallbackData(Stream::StreamCallback callback_, void* userData_) : callback(callback_), userData(userData_) {}
    };

    void CUDART_CB cudaStreamCallback(cudaStream_t, cudaError_t status, void* userData)
    {
        CallbackData* data = reinterpret_cast<CallbackData*>(userData);
        data->callback(static_cast<int>(status), data->userData);
        delete data;
    }
}

#endif

void cv::cuda::Stream::enqueueHostCallback(StreamCallback callback, void* userData)
{
#ifndef HAVE_CUDA
    (void) callback;
    (void) userData;
    throw_no_cuda();
#else
    #if CUDART_VERSION < 5000
        (void) callback;
        (void) userData;
        CV_Error(cv::Error::StsNotImplemented, "This function requires CUDA 5.0");
    #else
        CallbackData* data = new CallbackData(callback, userData);

        cudaSafeCall( cudaStreamAddCallback(impl_->stream, cudaStreamCallback, data, 0) );
    #endif
#endif
}

Stream& cv::cuda::Stream::Null()
{
    static Stream s(Ptr<Impl>(new Impl(0)));
    return s;
}

cv::cuda::Stream::operator bool_type() const
{
#ifndef HAVE_CUDA
    return 0;
#else
    return (impl_->stream != 0) ? &Stream::this_type_does_not_support_comparisons : 0;
#endif
}


////////////////////////////////////////////////////////////////
// Stream

#ifndef HAVE_CUDA

class cv::cuda::Event::Impl
{
public:
    Impl(unsigned int)
    {
        throw_no_cuda();
    }
};

#else

class cv::cuda::Event::Impl
{
public:
    cudaEvent_t event;

    Impl(unsigned int flags);
    ~Impl();
};

cv::cuda::Event::Impl::Impl(unsigned int flags) : event(0)
{
    cudaSafeCall( cudaEventCreateWithFlags(&event, flags) );
}

cv::cuda::Event::Impl::~Impl()
{
    if (event)
        cudaEventDestroy(event);
}

cudaEvent_t cv::cuda::EventAccessor::getEvent(const Event& event)
{
    return event.impl_->event;
}

#endif

cv::cuda::Event::Event(CreateFlags flags)
{
#ifndef HAVE_CUDA
    (void) flags;
    throw_no_cuda();
#else
    impl_ = makePtr<Impl>(flags);
#endif
}

void cv::cuda::Event::record(Stream& stream)
{
#ifndef HAVE_CUDA
    (void) stream;
    throw_no_cuda();
#else
    cudaSafeCall( cudaEventRecord(impl_->event, StreamAccessor::getStream(stream)) );
#endif
}

bool cv::cuda::Event::queryIfComplete() const
{
#ifndef HAVE_CUDA
    throw_no_cuda();
    return false;
#else
    cudaError_t err = cudaEventQuery(impl_->event);

    if (err == cudaErrorNotReady || err == cudaSuccess)
        return err == cudaSuccess;

    cudaSafeCall(err);
    return false;
#endif
}

void cv::cuda::Event::waitForCompletion()
{
#ifndef HAVE_CUDA
    throw_no_cuda();
#else
    cudaSafeCall( cudaEventSynchronize(impl_->event) );
#endif
}

float cv::cuda::Event::elapsedTime(const Event& start, const Event& end)
{
#ifndef HAVE_CUDA
    (void) start;
    (void) end;
    throw_no_cuda();
    return 0.0f;
#else
    float ms;
    cudaSafeCall( cudaEventElapsedTime(&ms, start.impl_->event, end.impl_->event) );
    return ms;
#endif
}
