// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_GPU_BACKEND_IMPL_HPP
#define OPENCV_HAL_GPU_BACKEND_IMPL_HPP

// Non-CPU HAL: what the CUDA and HIP backends share. Their handles are addresses, so a handle plus
// an offset is an address too.

#include "opencv2/core/utils/logger.hpp"
#include "ops_core.impl.hpp"
#include "ops_imgproc.impl.hpp"

#include <memory>
#include <mutex>
#include <string.h>
#include <vector>

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! one device; a backend's constructor says whether the CPU addresses its memory (shared)
struct GpuDevice
{
    cvGpuDeviceProp prop;
    bool shared = false;                // memory is host memory mapped into the GPU
#ifdef CV_ACCEL_GPU_HAVE_POOLS
    bool poolsSupported = false;
    std::once_flag poolOnce;
    CV_GPU(MemPool_t) pool = 0;         // the backend's own pool for device memory, 0 without one
    CV_GPU(Stream_t) allocStream = 0;   // where pool memory is allocated and freed
#endif
};

/* Devices, streams, copies, memory and the core operations. Device memory comes from a
   stream-ordered pool where the device has pools; shared devices get mapped host memory. */
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
            r = CV_GPU_HOST_ALLOC(&p, size, CV_GPU_HOST_ALLOC_MAPPED | CV_GPU_HOST_ALLOC_PORTABLE);  // one address for CPU and GPU
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
                r = CV_GPU_HOST_FREE(handle);
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
            return &core_;
        return strcmp(module, "imgproc") == 0 ? &imgproc_ : 0;
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

    GpuCoreOps core_;
    GpuImgprocOps imgproc_;
};

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // OPENCV_HAL_GPU_BACKEND_IMPL_HPP
