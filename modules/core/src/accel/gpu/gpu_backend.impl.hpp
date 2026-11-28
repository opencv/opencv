// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_GPU_BACKEND_IMPL_HPP
#define OPENCV_CORE_SRC_ACCEL_GPU_BACKEND_IMPL_HPP

// Non-CPU HAL: what the CUDA and HIP backends share. Their handles are addresses, so a handle plus
// an offset is an address too.

#include "opencv2/core/private/accel_backend.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "gpu_runtime.hpp"
#include "gpu/core/include/gpu_hal_core.hpp"         // non_cpu_hal/: the operations of each module
#include "gpu/imgproc/include/gpu_hal_imgproc.hpp"

#include <algorithm>
#include <list>
#include <map>
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

    void synchronize() CV_OVERRIDE
    {
        check(CV_GPU(StreamSynchronize)(s_), "StreamSynchronize");
    }

    const int device_;
    CV_GPU(Stream_t) s_;
    const unsigned int eventFlags_;
};

//! the launch failed because the build has no code for this GPU; ROCm reports that in several ways
bool noKernelImage(CV_GPU(Error_t) r)
{
#ifdef CV_ACCEL_GPU_HIP
    return r == hipErrorNoBinaryForGpu || r == hipErrorInvalidImage ||
           r == hipErrorInvalidKernelFile || r == hipErrorInvalidDeviceFunction;
#else
    return r == CV_GPU_ERROR_NO_KERNEL_IMAGE;
#endif
}

GpuStream& gpuStreamOf(Stream& s)
{
    GpuStream* gs = dynamic_cast<GpuStream*>(&s);
    CV_Assert(gs && "accel(" CV_GPU_RUNTIME "): the stream belongs to another backend");
    return *gs;
}

//! sets the launches of the tunable kernels (non_cpu_hal/gpu/precomp.hpp)
void setKernelTuning(const String& spec);

/** @brief Bytes of freed shared memory a device keeps for reuse: OPENCV_ACCEL_SHARED_BUFFERPOOL_LIMIT,
1 GiB by default, 0 to free every block at once. */
size_t sharedPoolLimit()
{
    static const size_t limit = utils::getConfigurationParameterSizeT("OPENCV_ACCEL_SHARED_BUFFERPOOL_LIMIT",
                                                                      (size_t)1 << 30);
    return limit;
}

/** @brief @p size rounded up as the OpenCL buffer pools round it (4 KiB, 64 KiB or 1 MiB steps),
so that close sizes share blocks. */
size_t sharedCapacity(size_t size)
{
    const size_t granularity = size < ((size_t)1 << 20) ? 4096 : size < ((size_t)16 << 20) ? (64 << 10) : (1 << 20);
    return (size + granularity - 1) & ~(granularity - 1);
}

//! one device; a backend's constructor says whether the CPU addresses its memory (shared)
struct GpuDevice
{
    cvGpuDeviceProp prop;
    bool shared = false;                // memory is host memory mapped into the GPU
    std::mutex sharedMutex;             // guards the three below
    std::map<void*, size_t> sharedInUse;                // the capacity of each shared block handed out
    std::list<std::pair<void*, size_t> > sharedKept;    // freed shared blocks, most recent first
    size_t sharedKeptBytes = 0;
#ifdef CV_ACCEL_GPU_HAVE_POOLS
    bool poolsSupported = false;
    std::once_flag poolOnce;
    CV_GPU(MemPool_t) pool = 0;         // the backend's own pool for device memory, 0 without one
    CV_GPU(Stream_t) allocStream = 0;   // where pool memory is allocated and freed
#endif
};

/* Devices, streams, copies, memory and the core operations. Device memory comes from a
   stream-ordered pool where the device has pools; shared devices get mapped host memory, whose
   freed blocks are kept for reuse. */
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
#ifdef CV_ACCEL_GPU_HAVE_POOLS
            d->poolsSupported = attr(CV_GPU_ATTR_MEMORY_POOLS, i) != 0;
#endif
            devs_.push_back(std::move(d));
        }
    }

    int deviceCount() const CV_OVERRIDE { return (int)devs_.size(); }
    String deviceDescription(int device) const CV_OVERRIDE { return devs_[device]->prop.name; }

    void* allocate(int device, size_t size) CV_OVERRIDE
    {
        GpuDevice& d = *devs_[device];
        DeviceGuard g(device);
        void* p = 0;
        CV_GPU(Error_t) r;
        if (d.shared)
            r = allocateShared(d, size, &p);
#ifdef CV_ACCEL_GPU_HAVE_POOLS
        else if (usesPool(device))
            r = allocateFromPool(d, size, &p);
#endif
        else
            r = CV_GPU(Malloc)(&p, size);
        if (r == CV_GPU_ERROR_OUT_OF_MEMORY)
        {
            (void)CV_GPU(GetLastError)();
            return 0;
        }
        check(r, "allocate");
        return p;
    }

    void deallocate(int device, void* handle) CV_OVERRIDE
    {
        // called while UMats are released: report failures, never throw
        try
        {
            GpuDevice& d = *devs_[device];
            DeviceGuard g(device);
            CV_GPU(Error_t) r;
            if (d.shared)
                r = releaseShared(d, handle);
#ifdef CV_ACCEL_GPU_HAVE_POOLS
            else if (usesPool(device))
            {
                // allocStream only allocates and frees, so this returns at once and gives what the
                // pool keeps above its release threshold back to the driver
                r = CV_GPU(FreeAsync)(handle, d.allocStream);
                if (r == CV_GPU(Success))
                    r = CV_GPU(StreamSynchronize)(d.allocStream);
            }
#endif
            else
                r = CV_GPU(Free)(handle);
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

    void* hostAddress(int device, void* handle) CV_OVERRIDE
    {
        return devs_[device]->shared ? handle : 0;
    }

    void copy(int device, CopyKind kind, DeviceRef dst, size_t dstStep, DeviceRef src, size_t srcStep,
              size_t widthBytes, size_t rows, Stream& stream) CV_OVERRIDE
    {
        void* d = (uchar*)dst.handle + dst.offset;
        const void* sp = (const uchar*)src.handle + src.offset;
        GpuStream& s = gpuStreamOf(stream);
        DeviceGuard g(device);
        // a contiguous region goes as one span: Memcpy2D refuses a pitch above cudaDevAttrMaxPitch
        // (2 GiB), and map() passes a whole buffer as one row of its full size
        if (rows == 1 || (dstStep == widthBytes && srcStep == widthBytes))
            check(CV_GPU(MemcpyAsync)(d, sp, widthBytes * rows, CV_GPU(MemcpyDefault), s.s_),
                  "MemcpyAsync");
        else
            check(CV_GPU(Memcpy2DAsync)(d, dstStep, sp, srcStep, widthBytes, rows,
                                        CV_GPU(MemcpyDefault), s.s_), "Memcpy2DAsync");
        if (kind != COPY_DEVICE_TO_DEVICE)
            s.synchronize();
    }

    Ptr<Stream> createStream(int device) CV_OVERRIDE
    {
        return makePtr<GpuStream>(device, eventFlags(device));
    }

    Ops* ops(const char* module) CV_OVERRIDE
    {
        if (strcmp(module, "core") == 0)
            return core_.get();
        return strcmp(module, "imgproc") == 0 ? imgproc_.get() : 0;
    }

    void setTuning(const String& spec) CV_OVERRIDE
    {
        setKernelTuning(spec);
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

    std::vector<std::unique_ptr<GpuDevice> > devs_;

private:
    /** @brief Shared memory for @p size bytes: a kept block that fits, else newly pinned host memory.

    Pinning and mapping a block costs milliseconds, so freed blocks are kept (releaseShared) and
    reused by the OpenCL buffer pools' rule: the smallest one wasting less than an eighth. */
    CV_GPU(Error_t) allocateShared(GpuDevice& d, size_t size, void** p)
    {
        {
            std::lock_guard<std::mutex> lock(d.sharedMutex);
            auto best = d.sharedKept.end();
            for (auto i = d.sharedKept.begin(); i != d.sharedKept.end(); ++i)
                if (i->second >= size && i->second - size < std::max((size_t)4096, size / 8) &&
                    (best == d.sharedKept.end() || i->second < best->second))
                    best = i;
            if (best != d.sharedKept.end())
            {
                *p = best->first;
                d.sharedKeptBytes -= best->second;
                d.sharedInUse[*p] = best->second;
                d.sharedKept.erase(best);
                return CV_GPU(Success);
            }
        }
        const size_t capacity = sharedCapacity(size);
        CV_GPU(Error_t) r = CV_GPU_HOST_ALLOC(p, capacity, CV_GPU_HOST_ALLOC_SHARED);  // one address for CPU and GPU
        if (r == CV_GPU_ERROR_OUT_OF_MEMORY)
        {
            (void)CV_GPU(GetLastError)();
            (void)trimShared(d, 0);     // the kept blocks go back to the driver, then once more
            r = CV_GPU_HOST_ALLOC(p, capacity, CV_GPU_HOST_ALLOC_SHARED);
        }
        if (r == CV_GPU(Success))
        {
            std::lock_guard<std::mutex> lock(d.sharedMutex);
            d.sharedInUse[*p] = capacity;
        }
        return r;
    }

    /** @brief Keeps the freed shared block @p handle for reuse, then trims the kept blocks to
    sharedPoolLimit(). A block larger than the limit is freed at once. */
    CV_GPU(Error_t) releaseShared(GpuDevice& d, void* handle)
    {
        {
            std::lock_guard<std::mutex> lock(d.sharedMutex);
            auto i = d.sharedInUse.find(handle);
            if (i != d.sharedInUse.end())
            {
                if (i->second <= sharedPoolLimit())
                {
                    d.sharedKept.push_front(*i);
                    d.sharedKeptBytes += i->second;
                    handle = 0;
                }
                d.sharedInUse.erase(i);
            }
        }
        const CV_GPU(Error_t) r = handle ? CV_GPU_HOST_FREE(handle) : CV_GPU(Success);
        const CV_GPU(Error_t) t = trimShared(d, sharedPoolLimit());
        return r != CV_GPU(Success) ? r : t;
    }

    /** @brief Frees the least recently kept shared blocks until at most @p limit bytes are kept;
    returns the last error of the frees. */
    CV_GPU(Error_t) trimShared(GpuDevice& d, size_t limit)
    {
        std::vector<void*> blocks;
        {
            std::lock_guard<std::mutex> lock(d.sharedMutex);
            for (; d.sharedKeptBytes > limit; d.sharedKept.pop_back())
            {
                blocks.push_back(d.sharedKept.back().first);
                d.sharedKeptBytes -= d.sharedKept.back().second;
            }
        }
        CV_GPU(Error_t) r = CV_GPU(Success);
        for (size_t i = 0; i < blocks.size(); i++)
        {
            const CV_GPU(Error_t) e = CV_GPU_HOST_FREE(blocks[i]);
            if (e != CV_GPU(Success))
                r = e;
        }
        return r;
    }

#ifdef CV_ACCEL_GPU_HAVE_POOLS
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
        // what the pool may hold, in use and kept for reuse together; freed memory above it goes back
        uint64_t threshold = (uint64_t)d.prop.totalGlobalMem / 4;
        if (CV_GPU(MemPoolSetAttribute)(d.pool, CV_GPU(MemPoolAttrReleaseThreshold), &threshold) != CV_GPU(Success))
            (void)CV_GPU(GetLastError)();
    }

    CV_GPU(Error_t) allocateFromPool(GpuDevice& d, size_t size, void** p)
    {
        CV_GPU(Error_t) r = CV_GPU(MallocFromPoolAsync)(p, size, d.pool, d.allocStream);
        if (r == CV_GPU_ERROR_OUT_OF_MEMORY)
        {
            // what the pool keeps in blocks of other sizes goes back to the driver, then once more;
            // blocks freed on allocStream can be trimmed only once the stream has reached the frees
            (void)CV_GPU(GetLastError)();
            (void)CV_GPU(StreamSynchronize)(d.allocStream);
            (void)CV_GPU(MemPoolTrimTo)(d.pool, 0);
            r = CV_GPU(MallocFromPoolAsync)(p, size, d.pool, d.allocStream);
            if (r != CV_GPU(Success))
                (void)CV_GPU(MemPoolTrimTo)(d.pool, 0);    // a failed attempt may leave memory reserved
        }
        if (r == CV_GPU(Success))
            r = CV_GPU(StreamSynchronize)(d.allocStream);   // from now on usable on every stream
        return r;
    }
#endif

    Ptr<CoreOps> core_ = createCoreOps();
    Ptr<ImgprocOps> imgproc_ = createImgprocOps();
};

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#include "gpu/precomp.hpp"   // non_cpu_hal/: the operation half of each family

#endif // OPENCV_CORE_SRC_ACCEL_GPU_BACKEND_IMPL_HPP
