// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_GPU_BACKEND_IMPL_HPP
#define OPENCV_CORE_SRC_ACCEL_GPU_BACKEND_IMPL_HPP

// Non-CPU HAL: what the CUDA and HIP backends share. Their handles are addresses, so a handle plus
// an offset is an address too.

#include "opencv2/core/accel_backend.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "gpu_kernels.hpp"

#include <memory>
#include <mutex>
#include <string.h>
#include <vector>

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

    bool waitEvent(const Ptr<Event>& event) CV_OVERRIDE
    {
        GpuEvent* ge = dynamic_cast<GpuEvent*>(event.get());
        if (!ge)
            return false;
        DeviceGuard g(device_);
        check(CV_GPU(StreamWaitEvent)(s_, ge->e_, 0), "StreamWaitEvent");  // also across devices
        return true;
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

class GpuCoreOps CV_FINAL : public CoreOps
{
public:
    bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !kernels::supportedElementSize(esz))
            return false;
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::flip((const unsigned char*)devicePtr(src), src.step[0],
                            (unsigned char*)devicePtr(dst), dst.step[0],
                            src.rows, src.cols, esz, flipCode, s.s_), "flip");
        return true;
    }

    bool transpose(const UMat& src, UMat& dst, Stream& stream) CV_OVERRIDE
    {
        const int esz = (int)src.elemSize();
        if (src.dims != 2 || !kernels::supportedElementSize(esz))
            return false;
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(s.device_);
        check(kernels::transpose((const unsigned char*)devicePtr(src), src.step[0],
                                 (unsigned char*)devicePtr(dst), dst.step[0],
                                 src.rows, src.cols, esz, s.s_), "transpose");
        return true;
    }
};

//! one device; a backend's constructor says what it offers (shared, kinds)
struct GpuDevice
{
    GpuDevice() : shared(false), kinds(MEMORY_DEVICE), poolsSupported(false), pool(0), allocStream(0) {}

    cvGpuDeviceProp prop;
    bool shared;                    // host memory mapped into the GPU is the default kind
    int kinds;                      // bitmask of MemoryKind
    bool poolsSupported;
    std::once_flag poolOnce;
    CV_GPU(MemPool_t) pool;         // the backend's own pool for device memory, 0 without one
    CV_GPU(Stream_t) allocStream;   // where pool memory is allocated and freed
};

/* Devices, streams, copies, device memory and coherent shared memory, and the core operations.
   A backend that offers more memory kinds allocates those itself and passes the rest here. */
class GpuBackend : public Backend
{
public:
    GpuBackend()
    {
        int n = 0;
        if (CV_GPU(GetDeviceCount)(&n) != CV_GPU(Success))
        {
            (void)CV_GPU(GetLastError)();   // no driver or no device
            n = 0;
        }
        for (int i = 0; i < n; i++)
        {
            std::unique_ptr<GpuDevice> d(new GpuDevice());
            check(CV_GPU(GetDeviceProperties)(&d->prop, i), "GetDeviceProperties");
            d->poolsSupported = attr(CV_GPU_ATTR_MEMORY_POOLS, i) != 0;
            devs_.push_back(std::move(d));
        }
    }

    int deviceCount() const CV_OVERRIDE { return (int)devs_.size(); }
    String deviceDescription(int device) const CV_OVERRIDE { return devs_[device]->prop.name; }
    int memoryKinds(int device) const CV_OVERRIDE { return devs_[device]->kinds; }

    void* allocate(int device, size_t size, MemoryKind kind) CV_OVERRIDE
    {
        GpuDevice& d = *devs_[device];
        CV_Assert((d.kinds & (int)kind) != 0 && "accel(" CV_GPU_RUNTIME "): a kind this device does not provide");
        CV_Assert((kind == MEMORY_DEVICE || kind == MEMORY_SHARED_COHERENT) && "accel: a kind the backend did not allocate");
        DeviceGuard g(device);
        void* p = 0;
        CV_GPU(Error_t) r;
        if (kind == MEMORY_SHARED_COHERENT)
            r = CV_GPU_HOST_ALLOC(&p, size, CV_GPU_HOST_ALLOC_MAPPED | CV_GPU_HOST_ALLOC_PORTABLE);  // one address for CPU and GPU
        else if (usesPool(device))
            r = allocateFromPool(d, size, &p);
        else
            r = CV_GPU(Malloc)(&p, size);
        return allocated(r, p);
    }

    void deallocate(int device, void* ptr, MemoryKind kind) CV_OVERRIDE
    {
        // called while UMats are released: report failures, never throw
        try
        {
            GpuDevice& d = *devs_[device];
            DeviceGuard g(device);
            // freed the way it was allocated; Free takes managed memory too
            const CV_GPU(Error_t) r =
                kind == MEMORY_SHARED_COHERENT || kind == MEMORY_SHARED_NONCOHERENT ? CV_GPU_HOST_FREE(ptr)
                : kind == MEMORY_DEVICE && usesPool(device) ? CV_GPU(FreeAsync)(ptr, d.allocStream)
                : CV_GPU(Free)(ptr);
            if (r != CV_GPU(Success))
            {
                (void)CV_GPU(GetLastError)();
                CV_LOG_WARNING(NULL, "accel(" CV_GPU_RUNTIME "): freeing device memory failed: " << CV_GPU(GetErrorString)(r));
            }
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_WARNING(NULL, "accel(" CV_GPU_RUNTIME "): freeing device memory failed: " << e.what());
        }
    }

    void* allocateHost(int device, size_t size) CV_OVERRIDE
    {
        DeviceGuard g(device);
        void* p = 0;
        if (CV_GPU_HOST_ALLOC(&p, size, CV_GPU_HOST_ALLOC_PORTABLE) != CV_GPU(Success))
        {
            (void)CV_GPU(GetLastError)();
            return 0;
        }
        return p;
    }

    void deallocateHost(int, void* ptr) CV_OVERRIDE
    {
        if (CV_GPU_HOST_FREE(ptr) != CV_GPU(Success))
            (void)CV_GPU(GetLastError)();
    }

    void copy(int device, CopyKind kind, DeviceRef dst, size_t dstStep, DeviceRef src, size_t srcStep,
              size_t widthBytes, size_t rows, Stream& stream, bool hostStable) CV_OVERRIDE
    {
        void* d = (uchar*)dst.handle + dst.offset;
        const void* sp = (const uchar*)src.handle + src.offset;
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(device);
        check(CV_GPU(Memcpy2DAsync)(d, dstStep, sp, srcStep, widthBytes, rows, CV_GPU(MemcpyDefault), s.s_),
              "Memcpy2DAsync");
        if (kind == COPY_HOST_TO_DEVICE && !hostStable)
        {
            // the caller's memory may change once this returns, and pinned memory is read asynchronously
            Ptr<Event> e = s.record();
            e->wait();
        }
    }

    Ptr<Stream> createStream(int device) CV_OVERRIDE
    {
        return makePtr<GpuStream>(device, eventFlags(device));
    }

    Ops* ops(const String& module) CV_OVERRIDE
    {
        return module == CoreOps::module() ? &core_ : 0;
    }

protected:
    //! the flags of the events a stream of @p device records
    virtual unsigned int eventFlags(int /*device*/) const { return CV_GPU(EventDisableTiming); }

    //! a device attribute, 0 when the runtime will not tell
    static int attr(cvGpuDeviceAttr a, int device)
    {
        int v = 0;
        if (CV_GPU(DeviceGetAttribute)(&v, a, device) != CV_GPU(Success))
        {
            (void)CV_GPU(GetLastError)();
            return 0;
        }
        return v;
    }

    //! @p p, or NULL when the device is out of memory; throws on any other failure
    static void* allocated(CV_GPU(Error_t) r, void* p)
    {
        if (r == CV_GPU_ERROR_OUT_OF_MEMORY)
        {
            (void)CV_GPU(GetLastError)();
            return 0;
        }
        check(r, "allocate");
        return p;
    }

    std::vector<std::unique_ptr<GpuDevice> > devs_;

private:
    //! device memory of @p device comes from the backend's pool; creates the pool on first use
    bool usesPool(int device)
    {
        GpuDevice& d = *devs_[device];
        if (d.shared || !d.poolsSupported)
            return false;
        std::call_once(d.poolOnce, &GpuBackend::createPool, this, device);
        return d.pool != 0;
    }

    // a pool of our own: its release threshold and the memory it keeps stay away from other users
    void createPool(int device)
    {
        GpuDevice& d = *devs_[device];
        DeviceGuard g(device);
        // first: when this throws nothing has been created, and call_once tries again next time
        check(CV_GPU(StreamCreateWithFlags)(&d.allocStream, CV_GPU(StreamNonBlocking)), "StreamCreateWithFlags");
        CV_GPU(MemPoolProps) props;
        memset(&props, 0, sizeof(props));
        props.allocType = CV_GPU(MemAllocationTypePinned);
        props.location.type = CV_GPU(MemLocationTypeDevice);
        props.location.id = device;
        if (CV_GPU(MemPoolCreate)(&d.pool, &props) != CV_GPU(Success))
        {
            (void)CV_GPU(GetLastError)();
            d.pool = 0;
            return;
        }
        uint64_t threshold = UINT64_MAX;
        if (CV_GPU(MemPoolSetAttribute)(d.pool, CV_GPU(MemPoolAttrReleaseThreshold), &threshold) != CV_GPU(Success))
            (void)CV_GPU(GetLastError)();
    }

    CV_GPU(Error_t) allocateFromPool(GpuDevice& d, size_t size, void** p)
    {
        CV_GPU(Error_t) r = CV_GPU(MallocFromPoolAsync)(p, size, d.pool, d.allocStream);
        if (r == CV_GPU_ERROR_OUT_OF_MEMORY)
        {
            // what the pool keeps in blocks of other sizes goes back to the driver, then once more
            (void)CV_GPU(GetLastError)();
            (void)CV_GPU(MemPoolTrimTo)(d.pool, 0);
            r = CV_GPU(MallocFromPoolAsync)(p, size, d.pool, d.allocStream);
        }
        if (r == CV_GPU(Success))
            r = CV_GPU(StreamSynchronize)(d.allocStream);   // from now on usable on every stream
        return r;
    }

    GpuCoreOps core_;
};

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // OPENCV_CORE_SRC_ACCEL_GPU_BACKEND_IMPL_HPP
