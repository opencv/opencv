// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_GPU_STREAM_IMPL_HPP
#define OPENCV_HAL_GPU_STREAM_IMPL_HPP

// Non-CPU HAL, CUDA and HIP: errors, the current device, events and streams, which the backend and
// its operations share.

#include "opencv2/core/private/accel_backend.hpp"
#include "runtime.hpp"

#if !defined(CV_ACCEL_GPU_HIP) && CUDART_VERSION >= 12000
#include <cuda.h>   // the types of a driver function reached through the runtime; libcuda is not linked
#endif

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

void check(CV_GPU(Error_t) e, const char* what)
{
    if (e == CV_GPU(Success))
        return;
    (void)CV_GPU(GetLastError)();   // a non-sticky error must not reach the thread's next call
    CV_Error_(Error::GpuApiCallError, ("accel(" CV_GPU_RUNTIME "): %s failed: %s", what, CV_GPU(GetErrorString)(e)));
}

#if !defined(CV_ACCEL_GPU_HIP) && CUDART_VERSION >= 12000
typedef CUresult (CUDAAPI *FN_cuDevicePrimaryCtxGetState)(CUdevice, unsigned int*, int*);

FN_cuDevicePrimaryCtxGetState loadPrimaryCtxGetState()
{
    void* fn = 0;
    cudaDriverEntryPointQueryResult found = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const cudaError_t r = cudaGetDriverEntryPointByVersion("cuDevicePrimaryCtxGetState", &fn, 12000,
                                                           cudaEnableDefault, &found);
#else
    const cudaError_t r = cudaGetDriverEntryPoint("cuDevicePrimaryCtxGetState", &fn, cudaEnableDefault, &found);
#endif
    if (r != cudaSuccess || found != cudaDriverEntryPointSuccess)
    {
        (void)cudaGetLastError();
        return 0;
    }
    return (FN_cuDevicePrimaryCtxGetState)fn;
}
#endif

// whether making @p device current costs nothing: since CUDA 12 SetDevice creates a primary
// context, which takes memory on a GPU the process may never use
bool hasContext(int device)
{
#if !defined(CV_ACCEL_GPU_HIP) && CUDART_VERSION >= 12000
    static const FN_cuDevicePrimaryCtxGetState getState = loadPrimaryCtxGetState();
    if (!getState)
        return true;    // cannot tell: go back, as before CUDA 12
    unsigned int flags = 0;
    int active = 0;
    return getState((CUdevice)device, &flags, &active) == CUDA_SUCCESS && active != 0;
#else
    CV_UNUSED(device);
    return true;
#endif
}

/* Makes @p device current for the calling thread while it lives. The previous device is made
   current again only when that is free (hasContext); otherwise the thread stays on @p device. */
class DeviceGuard
{
public:
    explicit DeviceGuard(int device) : prev_(-1)
    {
        int current = -1;
        check(CV_GPU(GetDevice)(&current), "GetDevice");
        if (current != device)
        {
            check(CV_GPU(SetDevice)(device), "SetDevice");
            prev_ = hasContext(current) ? current : -1;
        }
    }
    ~DeviceGuard()
    {
        if (prev_ >= 0)
            (void)CV_GPU(SetDevice)(prev_);
    }
private:
    int prev_;
};

class GpuEvent CV_FINAL : public Event
{
public:
    explicit GpuEvent(CV_GPU(Event_t) e) : e_(e) {}
    ~GpuEvent() { (void)CV_GPU(EventDestroy)(e_); }

    bool query() CV_OVERRIDE
    {
        const CV_GPU(Error_t) r = CV_GPU(EventQuery)(e_);
        if (r == CV_GPU(ErrorNotReady))
        {
            (void)CV_GPU(GetLastError)();
            return false;
        }
        check(r, "EventQuery");
        return true;
    }

    void wait() CV_OVERRIDE
    {
        check(CV_GPU(EventSynchronize)(e_), "EventSynchronize");
    }

    CV_GPU(Event_t) e_;
};

class GpuStream CV_FINAL : public Stream
{
public:
    GpuStream(int device, unsigned int eventFlags) : device_(device), s_(0), eventFlags_(eventFlags)
    {
        DeviceGuard g(device);
        check(CV_GPU(StreamCreateWithFlags)(&s_, CV_GPU(StreamNonBlocking)), "StreamCreateWithFlags");
    }
    ~GpuStream() { (void)CV_GPU(StreamDestroy)(s_); }   // queued work still completes

    Ptr<Event> record() CV_OVERRIDE
    {
        DeviceGuard g(device_);
        CV_GPU(Event_t) e = 0;
        check(CV_GPU(EventCreateWithFlags)(&e, eventFlags_), "EventCreateWithFlags");
        const CV_GPU(Error_t) r = CV_GPU(EventRecord)(e, s_);
        if (r != CV_GPU(Success))
        {
            (void)CV_GPU(EventDestroy)(e);
            check(r, "EventRecord");
        }
        return makePtr<GpuEvent>(e);
    }

    void synchronize() CV_OVERRIDE
    {
        check(CV_GPU(StreamSynchronize)(s_), "StreamSynchronize");
    }

    const int device_;
    CV_GPU(Stream_t) s_;
    const unsigned int eventFlags_;
};

GpuStream& gpuStreamOf(Stream& s)
{
    GpuStream* gs = dynamic_cast<GpuStream*>(&s);
    CV_Assert(gs && "accel(" CV_GPU_RUNTIME "): the stream belongs to another backend");
    return *gs;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // OPENCV_HAL_GPU_STREAM_IMPL_HPP
