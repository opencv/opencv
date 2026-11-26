// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: device memory behind UMat. Copies and CPU access go through the calling thread's
// stream, so they come after the work that thread queued before.

#include "../precomp.hpp"
#include "../umatrix.hpp"
#include "accel_impl.hpp"

#include <atomic>

namespace cv { namespace accel { namespace detail {

namespace {

inline Device::Impl* deviceOf(const UMatData* u)
{
    return static_cast<Device::Impl*>(u->allocatorContext.get());
}

/* On an integrated GPU a kernel reading shared memory that the CPU wrote, and still caches, runs up to three
   times as slow, read after read, and flushing the lines costs about two such reads. So, as in renting before
   buying, the device reads twice at that cost and the caches are flushed before its third read.
   UMatData::allocatorFlags_ of shared memory counts the device's reads since the CPU's write
   (beforeDeviceAccess()). Lines the CPU only read slow the device less and often not past its first read. */
enum { CPU_CACHED = 1, DEVICE_READ = 2, READS_BEFORE_FLUSH = 2 };

/* Copies @p n bytes into shared memory with stores that bypass the CPU caches, which is also faster than
   memcpy; false when the bytes went through the caches. */
bool copyPastCaches(uchar* dst, const uchar* src, size_t n)
{
#if CV_SIMD
    const size_t V = (size_t)VTraits<v_uint8>::vlanes();
    if (n >= 256)
    {
        const size_t head = std::min(n, (V - (size_t)dst % V) % V);
        memcpy(dst, src, head);
        size_t i = head;
        for (; i + V <= n; i += V)
            v_store_aligned_nocache(dst + i, vx_load(src + i));
        memcpy(dst + i, src + i, n - i);
        vx_cleanup();
        return true;
    }
#endif
    memcpy(dst, src, n);
    return false;
}

#if CV_SSE2
// writes back and drops the lines of @p n bytes at @p p from the CPU caches, in stripes: lines that several cores
// cache go about twice as fast
class FlushStripes CV_FINAL : public ParallelLoopBody
{
public:
    FlushStripes(const uchar* p, size_t n, size_t stripe) : p_(p), n_(n), stripe_(stripe) {}
    void operator()(const Range& r) const CV_OVERRIDE
    {
        const uchar* end = p_ + std::min(n_, (size_t)r.end * stripe_);
        for (const uchar* q = (const uchar*)((size_t)(p_ + r.start * stripe_) & ~(size_t)63); q < end; q += 64)
            _mm_clflush(q);
        _mm_mfence();
    }
private:
    const uchar* p_;
    size_t n_, stripe_;
};
#endif

//! drops @p n bytes at @p p from the CPU caches; nothing where there is no instruction for it
void flushCpuCaches(const uchar* p, size_t n)
{
#if CV_SSE2
    const int stripes = n >= ((size_t)1 << 20) ? 8 : 1;
    const size_t stripe = ((n + stripes - 1) / stripes + 63) & ~(size_t)63;
    parallel_for_(Range(0, (int)((n + stripe - 1) / stripe)), FlushStripes(p, n, stripe), stripes);
#else
    CV_UNUSED(p); CV_UNUSED(n);
#endif
}

//! a header that keeps @p u alive, for Stream::retain()
UMat holderOf(UMatData* u)
{
    UMat m;
    m.u = u;
    m.addref();
    return m;
}

// calls fn(srcOffset, dstOffset, srcPitch, dstPitch, widthBytes, rows) per 2-D plane of an N-D
// region; as everywhere in MatAllocator, the last dimension's sizes and offsets are in bytes
template<typename Fn>
void forEachPlane(int dims, const size_t sz[], const size_t srcofs[], const size_t srcstep[],
                  const size_t dstofs[], const size_t dststep[], Fn fn)
{
    CV_Assert(dims >= 1 && dims <= CV_MAX_DIM);
    for (int i = 0; i < dims; i++)
        if (sz[i] == 0)
            return;
    size_t srcBase = 0, dstBase = 0;
    for (int i = 0; i < dims; i++)
    {
        if (srcofs)
            srcBase += srcofs[i] * (i < dims - 1 ? srcstep[i] : 1);
        if (dstofs)
            dstBase += dstofs[i] * (i < dims - 1 ? dststep[i] : 1);
    }
    const size_t width = sz[dims - 1];
    const size_t rows = dims >= 2 ? sz[dims - 2] : 1;
    const size_t srcPitch = dims >= 2 ? srcstep[dims - 2] : width;
    const size_t dstPitch = dims >= 2 ? dststep[dims - 2] : width;
    const int outer = dims - 2;     // dimensions iterated around the planes
    size_t idx[CV_MAX_DIM] = {0};
    for (;;)
    {
        size_t so = srcBase, dof = dstBase;
        for (int i = 0; i < outer; i++)
        {
            so += idx[i] * srcstep[i];
            dof += idx[i] * dststep[i];
        }
        fn(so, dof, srcPitch, dstPitch, width, rows);
        int i = outer - 1;
        for (; i >= 0; i--)
        {
            if (++idx[i] < sz[i])
                break;
            idx[i] = 0;
        }
        if (i < 0)
            break;
    }
}

} // namespace

UMatData* BackendAllocator::allocate(Device::Impl* dev, int dims, const int* sizes, int type, size_t* step) const
{
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--)
    {
        step[i] = total;
        total *= (size_t)sizes[i];
    }
    CV_Assert(total > 0);

    Backend& b = *dev->backend;
    void* handle = b.allocate(dev->index, total);
    if (!handle)
    {
        // memory kept for this thread's finished work is given back first
        Stream& s = threadStream(dev);
        s.synchronize();
        s.releaseFinished();
        handle = b.allocate(dev->index, total);
    }
    if (!handle)
        CV_Error_(Error::StsNoMem, ("accel: %s is out of memory (%llu bytes requested)",
                                    dev->name.c_str(), (unsigned long long)total));

    UMatData* u = new UMatData(this);
    u->flags |= UMatData::ACCEL_MEMORY;     // what getDevice() and UMat::getMat() test
    u->size = total;
    u->handle = handle;
    u->allocatorContext = std::shared_ptr<void>(dev, [](void*) {});    // devices live as long as the process
    u->data = u->origdata = (uchar*)b.hostAddress(dev->index, handle);
    if (!u->data)
        u->flags |= UMatData::COPY_ON_MAP;  // the host copy is made on the first CPU access
    return u;
}

UMatData* BackendAllocator::allocate(int, const int*, int, void*, size_t*, AccessFlag, UMatUsageFlags) const
{
    CV_Error(Error::StsNotImplemented, "accel: device data is created with UMat::createOn()");
}

bool BackendAllocator::allocate(UMatData*, AccessFlag, UMatUsageFlags) const
{
    return false;   // never takes over memory of another allocator (Mat::getUMat)
}

void BackendAllocator::deallocate(UMatData* u) const
{
    CV_Assert(u);
    CV_Assert(u->urefcount == 0);
    CV_Assert(u->refcount == 0 && "UMat deallocation error: some derived Mat is still alive");
    // device work using the memory holds references (Stream::retain), so it has finished
    Device::Impl* dev = deviceOf(u);
    if (u->copyOnMap())
        fastFree(u->data);
    dev->backend->deallocate(dev->index, u->handle);
    u->handle = 0;
    u->data = u->origdata = 0;
    delete u;
}

void beforeDeviceAccess(UMatData* u, AccessFlag access, bool wholeBuffer)
{
    UMatDataAutoLock lock(u);
    if (!(u->allocatorFlags_ & CPU_CACHED))
        return;
    if (!(access & ACCESS_READ))
    {
        if (wholeBuffer)
            u->allocatorFlags_ = 0;     // the device's writes drop the lines from the CPU caches
    }
    else if (u->allocatorFlags_ / DEVICE_READ < READS_BEFORE_FLUSH)
        u->allocatorFlags_ += DEVICE_READ;
    else
    {
        flushCpuCaches(u->data, u->size);
        u->allocatorFlags_ = 0;
    }
}

void BackendAllocator::map(UMatData* u, AccessFlag accessFlags) const
{
    CV_Assert(u && u->handle);
    UMatDataAutoLock lock(u);
    Device::Impl* dev = deviceOf(u);
    Stream& s = threadStream(dev);
    if (!u->copyOnMap())
    {
        s.synchronize();    // the CPU uses the device's own bytes
        s.releaseFinished();
        if (!!(accessFlags & ACCESS_WRITE))
            u->allocatorFlags_ = CPU_CACHED;
        return;
    }
    if (!u->data)
        u->data = u->origdata = (uchar*)fastMalloc(u->size);
    if (u->hostCopyObsolete())
    {
        dev->backend->copy(dev->index, COPY_DEVICE_TO_HOST, DeviceRef(u->data), u->size,
                           DeviceRef(u->handle), u->size, u->size, 1, s);
        u->markHostCopyObsolete(false);
        s.releaseFinished();    // the copy waited for the work before it
    }
    // getMat() maps every view, so a later write view marks it here
    if (!!(accessFlags & ACCESS_WRITE))
        u->markDeviceCopyObsolete(true);
}

void BackendAllocator::unmap(UMatData* u) const
{
    CV_Assert(u);
    if (u->refcount == 0 && u->copyOnMap() && u->deviceCopyObsolete())
    {
        UMatDataAutoLock lock(u);
        Device::Impl* dev = deviceOf(u);
        dev->backend->copy(dev->index, COPY_HOST_TO_DEVICE, DeviceRef(u->handle), u->size,
                           DeviceRef(u->data), u->size, u->size, 1, threadStream(dev));
        u->markDeviceCopyObsolete(false);
    }
    if (u->urefcount == 0 && u->refcount == 0)
        deallocate(u);
}

void BackendAllocator::download(UMatData* u, void* dst, int dims, const size_t sz[],
                                const size_t srcofs[], const size_t srcstep[],
                                const size_t dststep[]) const
{
    CV_Assert(u && dst);
    UMatDataAutoLock lock(u);
    Device::Impl* dev = deviceOf(u);
    Stream& s = threadStream(dev);
    if (!u->copyOnMap() || (u->data && !u->hostCopyObsolete()))
    {
        // the CPU reaches these bytes: shared memory or a current host copy
        if (!u->copyOnMap())
            s.synchronize();
        forEachPlane(dims, sz, srcofs, srcstep, 0, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                memcpy((uchar*)dst + dof + r * dp, u->data + so + r * sp, w);
        });
        return;
    }
    forEachPlane(dims, sz, srcofs, srcstep, 0, dststep,
                 [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
        dev->backend->copy(dev->index, COPY_DEVICE_TO_HOST, DeviceRef(dst, dof), dp,
                           DeviceRef(u->handle, so), sp, w, rows, s);
    });
    s.releaseFinished();    // the copies waited for the work before them
}

void BackendAllocator::upload(UMatData* u, const void* src, int dims, const size_t sz[],
                              const size_t dstofs[], const size_t dststep[],
                              const size_t srcstep[]) const
{
    CV_Assert(u && src);
    UMatDataAutoLock lock(u);
    Device::Impl* dev = deviceOf(u);
    Stream& s = threadStream(dev);
    if (!u->copyOnMap() || u->refcount > 0)
    {
        // shared memory, or the host copy Mat views use; unmap() uploads it
        if (!u->copyOnMap())
            s.synchronize();
        const bool shared = !u->copyOnMap();
        bool cached = false;
        forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                if (shared)
                    cached |= !copyPastCaches(u->data + dof + r * dp, (const uchar*)src + so + r * sp, w);
                else
                    memcpy(u->data + dof + r * dp, (const uchar*)src + so + r * sp, w);
        });
        if (shared)
        {
            std::atomic_thread_fence(std::memory_order_seq_cst);   // the streaming stores land before later GPU work
            if (cached)
                u->allocatorFlags_ = CPU_CACHED;
        }
        else
            u->markDeviceCopyObsolete(true);
        return;
    }
    forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                 [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
        dev->backend->copy(dev->index, COPY_HOST_TO_DEVICE, DeviceRef(u->handle, dof), dp,
                           DeviceRef(src, so), sp, w, rows, s);
    });
    u->markHostCopyObsolete(true);
    u->markDeviceCopyObsolete(false);
}

void BackendAllocator::copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
                            const size_t srcofs[], const size_t srcstep[],
                            const size_t dstofs[], const size_t dststep[], bool sync) const
{
    CV_Assert(srcdata && dstdata);
    Device::Impl* dev = deviceOf(srcdata);
    if (dev == deviceOf(dstdata) && srcdata->refcount == 0 && dstdata->refcount == 0)
    {
        UMatDataAutoLock lock(srcdata, dstdata);
        Stream& s = threadStream(dev);
        forEachPlane(dims, sz, srcofs, srcstep, dstofs, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            dev->backend->copy(dev->index, COPY_DEVICE_TO_DEVICE, DeviceRef(dstdata->handle, dof), dp,
                               DeviceRef(srcdata->handle, so), sp, w, rows, s);
        });
        s.retain({ holderOf(srcdata), holderOf(dstdata) });
        dstdata->markHostCopyObsolete(true);
        if (sync)
            s.synchronize();
        return;
    }
    // across devices, or with Mat views alive: through host memory
    size_t tmpstep[CV_MAX_DIM] = {0};
    size_t total = sz[dims - 1];
    tmpstep[dims - 1] = 1;
    for (int k = dims - 2; k >= 0; k--)
    {
        tmpstep[k] = total;
        total *= sz[k];
    }
    AutoBuffer<uchar> tmp(total);
    download(srcdata, tmp.data(), dims, sz, srcofs, srcstep, tmpstep);
    upload(dstdata, tmp.data(), dims, sz, dstofs, dststep, tmpstep);
}

}}} // namespace cv::accel::detail
