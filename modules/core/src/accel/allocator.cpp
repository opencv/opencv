// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: device memory behind UMat (DeviceAllocator), the per-buffer events that order
// device work against the CPU, dma-buf import and export, and the list of released memory
// still used by device work.

#include "../precomp.hpp"
#include "../umatrix.hpp"
#include "accel_impl.hpp"
#include "opencv2/core/utils/logger.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif

namespace cv { namespace accel { namespace detail {

std::atomic<int> g_liveBuffers(0);

namespace {

inline BufferState* state(const UMatData* u)
{
    CV_DbgAssert(u && u->allocatorContext);
    return static_cast<BufferState*>(u->allocatorContext.get());
}

void closeFd(int fd)
{
#ifdef _WIN32
    CV_UNUSED(fd);
#else
    ::close(fd);
#endif
}

/* Calls fn(srcOffset, dstOffset, srcPitch, dstPitch, widthBytes, rows) for each 2-D plane of
   an N-D region. As everywhere in MatAllocator, sz[dims-1] and ofs[dims-1] are in bytes and
   step[dims-1] is the element size. */
template<typename Fn>
size_t forEachPlane(int dims, const size_t sz[], const size_t srcofs[], const size_t srcstep[],
                    const size_t dstofs[], const size_t dststep[], Fn fn)
{
    CV_Assert(dims >= 1 && dims <= CV_MAX_DIM);
    for (int i = 0; i < dims; i++)
        if (sz[i] == 0)
            return 0;
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
    size_t bytes = 0;
    for (;;)
    {
        size_t so = srcBase, dof = dstBase;
        for (int i = 0; i < outer; i++)
        {
            so += idx[i] * srcstep[i];
            dof += idx[i] * dststep[i];
        }
        fn(so, dof, srcPitch, dstPitch, width, rows);
        bytes += width * rows;
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
    return bytes;
}

//! CPU waits for @p e, then drops it. A failure propagates and keeps the event, so later accesses fail too.
void waitAndDrop(Ptr<Event>& e)
{
    if (!e)
        return;
    if (!e->query())
    {
        countHostWait();
        e->wait();
    }
    e.release();
}

//! true when @p e has completed; failed work counts as completed (memory can be freed after it)
bool finished(const Ptr<Event>& e)
{
    try
    {
        return e->query();
    }
    catch (const cv::Exception& ex)
    {
        CV_LOG_WARNING(NULL, "accel: device work on released memory failed: " << ex.what());
        return true;
    }
}

struct Retired
{
    Device::Impl* dev;
    void* devPtr;
    void* hostCopy;
    std::vector<UMatData*> aliases;
    std::vector<Ptr<Event> > events;
};

std::mutex& retiredMutex()
{
    static std::mutex* m = new std::mutex();
    return *m;
}

std::vector<Retired>& retiredList()
{
    static std::vector<Retired>* l = new std::vector<Retired>();
    return *l;
}

std::atomic<int> g_retiredCount(0);

void freeNow(const Retired& r)
{
    // imports of these bytes on other devices go first; their work is ordered by the same events
    for (size_t i = 0; i < r.aliases.size(); i++)
    {
        UMatData* a = r.aliases[i];
        a->urefcount = 0;
        a->currAllocator->deallocate(a);
    }
    if (r.devPtr)
        r.dev->backend->deallocate(r.dev->index, r.devPtr);
    if (r.hostCopy)
        r.dev->backend->deallocateHost(r.dev->index, r.hostCopy);
}

void* allocateDevice(Device::Impl* dev, size_t size)
{
    collectRetired(false);
    void* p = dev->backend->allocate(dev->index, size);
    if (!p && g_retiredCount.load() > 0)
    {
        // memory released while device work still used it: wait for that work, then retry
        collectRetired(true);
        p = dev->backend->allocate(dev->index, size);
    }
    return p;
}

//! a UMat header of @p size and @p type over all of @p u, rows @p step bytes apart
class WrapAllocator CV_FINAL : public MatAllocator
{
public:
    WrapAllocator(UMatData* u, size_t step) : u_(u), step_(step) {}
    UMatData* allocate(int dims, const int*, int type, void*, size_t* step, AccessFlag, UMatUsageFlags) const CV_OVERRIDE
    {
        CV_Assert(dims == 2);
        step[0] = step_;
        step[1] = CV_ELEM_SIZE(type);
        return u_;
    }
    bool allocate(UMatData*, AccessFlag, UMatUsageFlags) const CV_OVERRIDE { return false; }
    void deallocate(UMatData*) const CV_OVERRIDE { CV_Error(Error::StsInternal, "accel: unexpected deallocation"); }
private:
    UMatData* u_;
    size_t step_;
};

} // namespace

void collectRetired(bool wait)
{
    if (g_retiredCount.load(std::memory_order_relaxed) == 0)
        return;
    std::vector<Retired> done;
    {
        std::lock_guard<std::mutex> lock(retiredMutex());
        std::vector<Retired>& list = retiredList();
        for (size_t i = 0; i < list.size(); )
        {
            bool ready = true;
            for (size_t j = 0; j < list[i].events.size() && ready; j++)
            {
                if (wait)
                {
                    try { list[i].events[j]->wait(); }
                    catch (const cv::Exception& ex) { CV_LOG_WARNING(NULL, "accel: device work on released memory failed: " << ex.what()); }
                }
                else
                    ready = finished(list[i].events[j]);
            }
            if (ready)
            {
                done.push_back(list[i]);
                list[i] = list.back();
                list.pop_back();
            }
            else
                i++;
        }
        g_retiredCount.store((int)list.size());
    }
    for (size_t i = 0; i < done.size(); i++)
        freeNow(done[i]);
}

bool isDeviceAllocator(const MatAllocator* a)
{
    return a != 0 && dynamic_cast<const DeviceAllocator*>(a) != 0;
}

BufferState* stateOf(const UMatData* u)
{
    return (u && isDeviceAllocator(u->currAllocator)) ? state(u) : 0;
}

UMatData* owningData(UMatData* u)
{
    BufferState* st = stateOf(u);
    return (st && st->aliasOf) ? st->aliasOf : u;
}

void waitOnCpu(BufferState* st, bool write)
{
    waitAndDrop(st->write);
    st->writeStream.release();
    if (write)
    {
        for (size_t i = 0; i < st->reads.size(); i++)
            waitAndDrop(st->reads[i].second);
        st->reads.clear();
    }
}

void orderOnStream(BufferState* st, const Ptr<Stream>& s, bool write)
{
    if (st->write && st->writeStream != s && !s->waitEvent(st->write))
    {
        waitAndDrop(st->write);
        st->writeStream.release();
    }
    if (write)
    {
        for (size_t i = 0; i < st->reads.size(); i++)
        {
            Ptr<Event>& e = st->reads[i].second;
            if (e && st->reads[i].first != s && !s->waitEvent(e))
                waitAndDrop(e);
        }
    }
}

void recordUse(BufferState* st, const Ptr<Stream>& s, const Ptr<Event>& e, bool write)
{
    if (write)
    {
        // the write is ordered after every earlier use (orderOnStream), so it alone remains
        st->write = e;
        st->writeStream = s;
        st->reads.clear();
        return;
    }
    for (size_t i = 0; i < st->reads.size(); i++)
    {
        if (st->reads[i].first == s)
        {
            st->reads[i].second = e;
            return;
        }
    }
    st->reads.push_back(std::make_pair(s, e));
}

void prepareHandle(UMatData* u, AccessFlag accessFlags)
{
    BufferState* st = stateOf(u);
    if (!st)
        return;
    UMatDataAutoLock lock(u);
    waitOnCpu(st, !!(accessFlags & ACCESS_WRITE));
}

UMatData* aliasOn(UMatData* u, Device::Impl* dev)
{
    BufferState* st = stateOf(u);
    CV_Assert(st && !st->aliasOf);
    if (st->dev == dev)
        return u;
    UMatDataAutoLock lock(u);
    for (size_t i = 0; i < st->aliases.size(); i++)
        if (state(st->aliases[i])->dev == dev)
            return st->aliases[i];
    for (size_t i = 0; i < st->refused.size(); i++)
        if (st->refused[i] == dev)
            return 0;

    UMatData* a = 0;
    const int fd = st->dev->backend->exportDmaBuf(st->dev->index, u->handle, u->size);
    if (fd >= 0)
    {
        try
        {
            a = dev->allocator->importDmaBuf(fd, u->size);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_DEBUG(NULL, "accel: " << dev->name << " cannot import memory of " << st->dev->name << ": " << e.what());
            a = 0;
        }
        closeFd(fd);
    }
    if (!a)
    {
        st->refused.push_back(dev);
        return 0;
    }
    state(a)->aliasOf = u;
    a->urefcount = 1;   // held by u until u is freed
    st->aliases.push_back(a);
    return a;
}

UMatData* DeviceAllocator::allocate(int dims, const int* sizes, int type, void* data0, size_t* step,
                                    AccessFlag /*flags*/, UMatUsageFlags /*usageFlags*/) const
{
    if (data0)
        CV_Error(Error::StsNotImplemented, "accel: device memory cannot wrap user host memory");

    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--)
    {
        if (step)
            step[i] = total;
        total *= (size_t)sizes[i];
    }

    UMatData* u = new UMatData(this);
    u->size = total;
    BufferState* st = new BufferState(dev_);
    u->allocatorContext = std::shared_ptr<void>(st, [](void* p) { delete static_cast<BufferState*>(p); });
    if (total > 0)
    {
        u->handle = allocateDevice(dev_, total);
        if (!u->handle)
        {
            delete u;
            CV_Error_(Error::StsNoMem, ("accel: %s is out of memory (%llu bytes requested)",
                                        dev_->name.c_str(), (unsigned long long)total));
        }
    }
    if (dev_->shared)
    {
        u->data = u->origdata = total > 0 ? (uchar*)dev_->backend->hostAddress(dev_->index, u->handle) : 0;
    }
    else
    {
        u->flags |= UMatData::COPY_ON_MAP;
        u->markHostCopyObsolete(true);  // the host copy is created on the first CPU access
    }
    g_liveBuffers++;
    return u;
}

UMatData* DeviceAllocator::importDmaBuf(int fd, size_t size) const
{
    void* p = dev_->backend->importDmaBuf(dev_->index, fd, size);
    if (!p)
        return 0;
    UMatData* u = new UMatData(this);
    u->size = size;
    u->handle = p;
    u->flags |= UMatData::USER_ALLOCATED;   // never released and created again elsewhere
    BufferState* st = new BufferState(dev_);
    st->imported = true;
    st->initialized = true;
    u->allocatorContext = std::shared_ptr<void>(st, [](void* q) { delete static_cast<BufferState*>(q); });
    if (dev_->shared)
        u->data = u->origdata = (uchar*)dev_->backend->hostAddress(dev_->index, p);
    else
    {
        u->flags |= UMatData::COPY_ON_MAP;
        u->markHostCopyObsolete(true);
    }
    g_liveBuffers++;
    return u;
}

bool DeviceAllocator::allocate(UMatData* /*u*/, AccessFlag /*accessFlags*/, UMatUsageFlags /*usageFlags*/) const
{
    return false;   // never takes over memory of another allocator (Mat::getUMat)
}

void DeviceAllocator::deallocate(UMatData* u) const
{
    CV_Assert(u);
    CV_Assert(u->urefcount == 0);
    CV_Assert(u->refcount == 0 && "UMat deallocation error: some derived Mat is still alive");
    BufferState* st = state(u);
    Retired r;
    r.dev = dev_;
    r.devPtr = u->handle;
    r.hostCopy = (!dev_->shared && u->data) ? (void*)u->data : 0;
    r.aliases.swap(st->aliases);
    if (st->write && !finished(st->write))
        r.events.push_back(st->write);
    for (size_t i = 0; i < st->reads.size(); i++)
        if (st->reads[i].second && !finished(st->reads[i].second))
            r.events.push_back(st->reads[i].second);
    u->handle = 0;
    u->data = u->origdata = 0;
    delete u;
    g_liveBuffers--;

    if (r.events.empty())
    {
        freeNow(r);
        return;
    }
    std::lock_guard<std::mutex> lock(retiredMutex());
    retiredList().push_back(r);
    g_retiredCount.store((int)retiredList().size());
}

void DeviceAllocator::map(UMatData* u, AccessFlag accessFlags) const
{
    CV_Assert(u);
    UMatDataAutoLock lock(u);
    BufferState* st = state(u);
    const bool write = !!(accessFlags & ACCESS_WRITE);
    if (st->mapped && (st->mappedWrite || !write))
        return;     // already mapped as needed (UMat::getMat calls map for every view)

    waitOnCpu(st, write);
    Backend& b = *dev_->backend;
    const int i = dev_->index;
    if (dev_->shared)
    {
        if (u->size > 0)
        {
            if (st->mapped)
                b.endCpuAccess(i, u->handle, u->size, false);  // a read mapping becomes a write mapping
            b.beginCpuAccess(i, u->handle, u->size, write);
        }
    }
    else
    {
        if (!u->data && u->size > 0)
        {
            u->data = u->origdata = (uchar*)b.allocateHost(i, u->size);
            if (!u->data)
                CV_Error_(Error::StsNoMem, ("accel: no host memory for a copy of %llu bytes", (unsigned long long)u->size));
        }
        if (u->hostCopyObsolete() && !st->initialized)
            u->markHostCopyObsolete(false);     // fresh memory: its contents are undefined on both sides
        if (u->hostCopyObsolete() && u->size > 0)
        {
            const Ptr<Stream>& s = currentStream(dev_);
            b.copy(i, COPY_DEVICE_TO_HOST, u->data, u->size, u->handle, u->size, u->size, 1, *s, true);
            Ptr<Event> e = s->record();
            waitAndDrop(e);
            countCopy(COPY_DEVICE_TO_HOST, u->size);
            u->markHostCopyObsolete(false);
        }
        if (write)
            u->markDeviceCopyObsolete(true);
    }
    st->mapped = true;
    st->mappedWrite = st->mappedWrite || write;
    st->initialized = st->initialized || write;
}

void DeviceAllocator::unmap(UMatData* u) const
{
    CV_Assert(u);
    if (u->refcount == 0)
    {
        UMatDataAutoLock lock(u);
        BufferState* st = state(u);
        if (st->mapped)
        {
            Backend& b = *dev_->backend;
            const int i = dev_->index;
            if (dev_->shared)
            {
                if (u->size > 0)
                    b.endCpuAccess(i, u->handle, u->size, st->mappedWrite);
            }
            else if (u->deviceCopyObsolete())
            {
                // the CPU wrote: the upload is queued now, from the host copy, which stays
                // unchanged until it completes (CPU writes wait for it in map())
                const Ptr<Stream>& s = currentStream(dev_);
                orderOnStream(st, s, true);
                b.copy(i, COPY_HOST_TO_DEVICE, u->handle, u->size, u->data, u->size, u->size, 1, *s, true);
                recordUse(st, s, s->record(), true);
                countCopy(COPY_HOST_TO_DEVICE, u->size);
                u->markDeviceCopyObsolete(false);
            }
            st->mapped = false;
            st->mappedWrite = false;
        }
    }
    if (u->urefcount == 0 && u->refcount == 0)
        deallocate(u);
}

void DeviceAllocator::download(UMatData* u, void* dst, int dims, const size_t sz[],
                               const size_t srcofs[], const size_t srcstep[],
                               const size_t dststep[]) const
{
    CV_Assert(u && dst);
    UMatDataAutoLock lock(u);
    BufferState* st = state(u);
    Backend& b = *dev_->backend;
    const int i = dev_->index;
    if (dev_->shared || (!u->hostCopyObsolete() && u->data))
    {
        // the CPU reaches the bytes: the shared memory itself, or a host copy that is current
        waitOnCpu(st, false);
        const bool bracket = dev_->shared && !st->mapped && u->size > 0;
        if (bracket)
            b.beginCpuAccess(i, u->handle, u->size, false);
        forEachPlane(dims, sz, srcofs, srcstep, 0, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                memcpy((uchar*)dst + dof + r * dp, u->data + so + r * sp, w);
        });
        if (bracket)
            b.endCpuAccess(i, u->handle, u->size, false);
        return;
    }
    const Ptr<Stream>& s = currentStream(dev_);
    orderOnStream(st, s, false);
    const size_t bytes = forEachPlane(dims, sz, srcofs, srcstep, 0, dststep,
                                      [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
        b.copy(i, COPY_DEVICE_TO_HOST, (uchar*)dst + dof, dp, (uchar*)u->handle + so, sp, w, rows, *s, false);
    });
    Ptr<Event> e = s->record();
    waitAndDrop(e);     // downloads are synchronous: the caller reads the data next
    countCopy(COPY_DEVICE_TO_HOST, bytes);
}

void DeviceAllocator::upload(UMatData* u, const void* src, int dims, const size_t sz[],
                             const size_t dstofs[], const size_t dststep[],
                             const size_t srcstep[]) const
{
    CV_Assert(u && src);
    UMatDataAutoLock lock(u);
    BufferState* st = state(u);
    Backend& b = *dev_->backend;
    const int i = dev_->index;
    if (dev_->shared || st->mapped)
    {
        // shared memory, or a copy-only buffer whose Mat views the CPU uses: the CPU writes the
        // bytes it reaches, and unmap() uploads a written host copy
        waitOnCpu(st, true);
        const bool bracket = dev_->shared && !st->mapped && u->size > 0;
        if (bracket)
            b.beginCpuAccess(i, u->handle, u->size, true);
        if (st->mapped && !st->mappedWrite)
        {
            // the CPU now writes into a read mapping: make it a write mapping, so unmap() flushes it
            if (dev_->shared && u->size > 0)
            {
                b.endCpuAccess(i, u->handle, u->size, false);
                b.beginCpuAccess(i, u->handle, u->size, true);
            }
            st->mappedWrite = true;
        }
        forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                memcpy(u->data + dof + r * dp, (const uchar*)src + so + r * sp, w);
        });
        if (bracket)
            b.endCpuAccess(i, u->handle, u->size, true);
        if (!dev_->shared)
            u->markDeviceCopyObsolete(true);
        st->initialized = true;
        return;
    }
    const Ptr<Stream>& s = currentStream(dev_);
    orderOnStream(st, s, true);
    const size_t bytes = forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                                      [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
        // the caller's memory: the backend reads it before returning
        b.copy(i, COPY_HOST_TO_DEVICE, (uchar*)u->handle + dof, dp, (const uchar*)src + so, sp, w, rows, *s, false);
    });
    recordUse(st, s, s->record(), true);
    countCopy(COPY_HOST_TO_DEVICE, bytes);
    u->markHostCopyObsolete(true);
    u->markDeviceCopyObsolete(false);
    st->initialized = true;
}

void DeviceAllocator::copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
                           const size_t srcofs[], const size_t srcstep[],
                           const size_t dstofs[], const size_t dststep[], bool sync) const
{
    CV_Assert(srcdata && dstdata);
    {
        UMatDataAutoLock lock(srcdata, dstdata);
        BufferState* ss = state(srcdata);
        BufferState* ds = state(dstdata);
        if (!ss->mapped && !ds->mapped)
        {
            Backend& b = *dev_->backend;
            const int i = dev_->index;
            const Ptr<Stream>& s = currentStream(dev_);
            orderOnStream(ss, s, false);
            orderOnStream(ds, s, true);
            const size_t bytes = forEachPlane(dims, sz, srcofs, srcstep, dstofs, dststep,
                                              [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
                b.copy(i, COPY_DEVICE_TO_DEVICE, (uchar*)dstdata->handle + dof, dp,
                       (const uchar*)srcdata->handle + so, sp, w, rows, *s, true);
            });
            Ptr<Event> e = s->record();
            recordUse(ss, s, e, false);
            recordUse(ds, s, e, true);
            ds->initialized = true;
            if (!dev_->shared)
            {
                dstdata->markHostCopyObsolete(true);
                dstdata->markDeviceCopyObsolete(false);
            }
            countCopy(COPY_DEVICE_TO_DEVICE, bytes);
            if (sync)
                waitAndDrop(e);
            return;
        }
    }
    // a Mat view of one of them is alive, so its newest bytes may be on the CPU: copy through it
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

} // namespace detail

UMat wrapDmaBuf(const Device& device, int fd, Size size, int type, size_t step, size_t offset)
{
    CV_Assert(!device.empty() && "accel: wrapDmaBuf needs a device");
    CV_Assert(fd >= 0 && size.width > 0 && size.height > 0);
    type = CV_MAT_TYPE(type);
    const size_t row = (size_t)size.width * CV_ELEM_SIZE(type);
    if (step == Mat::AUTO_STEP)
        step = row;
    CV_Assert(step >= row && step % CV_ELEM_SIZE1(type) == 0);
    const size_t needed = offset + step * (size_t)(size.height - 1) + row;
    size_t bufferSize = needed;
#ifndef _WIN32
    const off_t end = ::lseek(fd, 0, SEEK_END);   // the size of a dma-buf
    if (end > 0)
        bufferSize = (size_t)end;
#endif
    if (needed > bufferSize)
        CV_Error_(Error::StsOutOfRange, ("accel: the image needs %llu bytes, the dma-buf has %llu",
                                         (unsigned long long)needed, (unsigned long long)bufferSize));

    Device::Impl* dev = device.impl();
    UMatData* u = dev->allocator->importDmaBuf(fd, bufferSize);
    if (!u)
        CV_Error_(Error::StsNotImplemented, ("accel: %s cannot import dma-bufs", dev->name.c_str()));

    // a UMat header over the import, then the view the caller asked for
    UMat m;
    detail::WrapAllocator wrap(u, step);
    m.allocator = &wrap;
    try
    {
        m.create(size, type);
    }
    catch (...)
    {
        m.allocator = 0;
        if (m.u != u)
            dev->allocator->deallocate(u);
        throw;
    }
    m.allocator = 0;
    CV_Assert(m.u == u);
    m.offset = offset;
    return m;
}

int exportDmaBuf(const UMat& m)
{
    detail::BufferState* st = detail::stateOf(m.u);
    if (!st)
        CV_Error(Error::StsBadArg, "accel: exportDmaBuf needs a UMat in device memory");
    UMatData* u = m.u;
    UMatDataAutoLock lock(u);
    if (st->mapped)
        CV_Error(Error::StsError, "accel: release the Mat views of a UMat before exporting it");
    detail::waitOnCpu(st, true);
    const int fd = st->dev->backend->exportDmaBuf(st->dev->index, u->handle, u->size);
    if (fd < 0)
        CV_Error_(Error::StsNotImplemented, ("accel: %s cannot export its memory as a dma-buf", st->dev->name.c_str()));
    if (!st->dev->shared)
        u->markHostCopyObsolete(true);  // someone else may write it from now on
    st->initialized = true;
    return fd;
}

}} // namespace cv::accel
