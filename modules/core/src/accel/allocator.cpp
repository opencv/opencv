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

namespace cv { namespace accel { namespace detail {

namespace {

inline Device::Impl* deviceOf(const UMatData* u)
{
    return static_cast<Device::Impl*>(u->allocatorContext.get());
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

bool isBackendAllocator(const MatAllocator* a)
{
    return a != 0 && dynamic_cast<const BackendAllocator*>(a) != 0;
}

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

void BackendAllocator::map(UMatData* u, AccessFlag accessFlags) const
{
    CV_Assert(u && u->handle);
    UMatDataAutoLock lock(u);
    Device::Impl* dev = deviceOf(u);
    Stream& s = threadStream(dev);
    if (!u->copyOnMap())
    {
        s.synchronize();    // the CPU uses the device's own bytes
        return;
    }
    if (!u->data)
        u->data = u->origdata = (uchar*)fastMalloc(u->size);
    if (u->hostCopyObsolete())
    {
        dev->backend->copy(dev->index, COPY_DEVICE_TO_HOST, DeviceRef(u->data), u->size,
                           DeviceRef(u->handle), u->size, u->size, 1, s);
        u->markHostCopyObsolete(false);
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
        forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                memcpy(u->data + dof + r * dp, (const uchar*)src + so + r * sp, w);
        });
        if (u->copyOnMap())
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
