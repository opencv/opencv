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
#include "opencv2/core/utils/configuration.private.hpp"
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

//! for messages; also the spelling OPENCV_ACCEL_FORCE_MEMORY_KIND accepts
const char* kindName(MemoryKind kind)
{
    switch (kind)
    {
    case MEMORY_DEVICE:             return "device";
    case MEMORY_SHARED_COHERENT:    return "shared";
    case MEMORY_SHARED_NONCOHERENT: return "shared_nc";
    case MEMORY_MANAGED:            return "managed";
    }
    return "?";
}

void closeFd(int fd)
{
#ifdef _WIN32
    CV_UNUSED(fd);
#else
    ::close(fd);
#endif
}

// calls fn(srcOffset, dstOffset, srcPitch, dstPitch, widthBytes, rows) per 2-D plane of an N-D
// region; as everywhere in MatAllocator, the last dimension's sizes and offsets are in bytes
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
    MemoryKind kind;        // the kind devPtr was allocated with, needed to free it
    size_t size;            // of devPtr, for the cache; 0 for memory that must not be reused
    size_t pinnedSize;      // bytes this counted against dev->pinnedBytes; 0 when it counted none
    void* hostCopy;
    size_t hostCopySize;    // and what the host copy counted, allocated lazily in map()
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

// the allocation cache (below); kind is a MemoryKind, or CACHE_HOST for a device buffer's host copy
const int CACHE_HOST = 0;   // no MemoryKind uses it
void* cacheTake(Device::Impl* dev, size_t size, int kind);
bool cachePut(Device::Impl* dev, void* ptr, size_t size, int kind);
bool cacheHas(Device::Impl* dev, size_t size, int kind);
void cacheFlush(Device::Impl* dev);

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
    {
        // kept for the next buffer of this size and kind where there is room; it stays counted
        // against the pinned budget while it sits there, because it is still allocated
        if (!r.size || !cachePut(r.dev, r.devPtr, r.size, r.kind))
        {
            r.dev->backend->deallocate(r.dev->index, r.devPtr, r.kind);
            if (r.pinnedSize)
                r.dev->pinnedBytes -= r.pinnedSize;
        }
    }
    if (r.hostCopy && !cachePut(r.dev, r.hostCopy, r.hostCopySize, CACHE_HOST))
    {
        r.dev->backend->deallocateHost(r.dev->index, r.hostCopy);
        r.dev->pinnedBytes -= r.hostCopySize;
    }
}

//! the OPENCV_ACCEL_* settings that steer allocations, read once (reloadSettings() for tests)
struct Settings
{
    std::atomic<int> forcedKind;        // OPENCV_ACCEL_FORCE_MEMORY_KIND, 0 when not set
    std::atomic<size_t> pinnedLimit;    // OPENCV_ACCEL_PINNED_LIMIT_MB in bytes, 0 = unlimited
    std::atomic<size_t> zeroCopyMax;    // OPENCV_ACCEL_ZERO_COPY_MAX_BYTES, 0 = no promotion
    std::atomic<size_t> cacheLimit;     // OPENCV_ACCEL_CACHE_MAX_MB in bytes, 0 = no cache
    std::atomic<bool> warnedForced;     // the warning about an override a device cannot follow

    Settings() { read(); }

    void read()
    {
        forcedKind = parseKind(utils::getConfigurationParameterString("OPENCV_ACCEL_FORCE_MEMORY_KIND", ""));
        pinnedLimit = (size_t)utils::getConfigurationParameterSizeT("OPENCV_ACCEL_PINNED_LIMIT_MB", 0) * 1024 * 1024;
        zeroCopyMax = (size_t)utils::getConfigurationParameterSizeT("OPENCV_ACCEL_ZERO_COPY_MAX_BYTES", 0);
        cacheLimit = (size_t)utils::getConfigurationParameterSizeT("OPENCV_ACCEL_CACHE_MAX_MB", 0) * 1024 * 1024;
        warnedForced = false;
    }

    static int parseKind(const std::string& s)
    {
        if (s.empty())          return 0;
        if (s == "device")      return MEMORY_DEVICE;
        if (s == "shared")      return MEMORY_SHARED_COHERENT;
        if (s == "shared_nc")   return MEMORY_SHARED_NONCOHERENT;
        if (s == "managed")     return MEMORY_MANAGED;
        CV_LOG_WARNING(NULL, "accel: OPENCV_ACCEL_FORCE_MEMORY_KIND=" << s
                       << " is not one of device, shared, shared_nc, managed; ignored");
        return 0;
    }
};

Settings& settings()
{
    static Settings* s = new Settings();
    return *s;
}

//! OPENCV_ACCEL_FORCE_MEMORY_KIND: a MemoryKind, or 0 for "not set"
int forcedKind()
{
    return settings().forcedKind.load(std::memory_order_relaxed);
}

//! the budget for memory the CPU addresses, 0 = unlimited
size_t pinnedLimit()
{
    return settings().pinnedLimit.load(std::memory_order_relaxed);
}

// the kind of a new buffer: the debug override, else the UMatUsageFlags, else the device default;
// shared memory past the pinned budget becomes device memory
MemoryKind chooseKind(Device::Impl* dev, size_t size, UMatUsageFlags usage, bool& requested)
{
    const char* why = "device default";
    requested = false;      // set once something other than the device default decides
    MemoryKind kind = newBufferKind(dev);
    const int forced = forcedKind();
    const bool overridden = forced && (dev->kinds & forced);

    if (overridden)
    {
        why = "OPENCV_ACCEL_FORCE_MEMORY_KIND";
        requested = true;
    }
    else if (forced)
    {
        if (!settings().warnedForced.exchange(true))    // once: this is the same for every buffer
            CV_LOG_WARNING(NULL, "accel: " << dev->name << " does not provide "
                           << kindName((MemoryKind)forced) << " memory; OPENCV_ACCEL_FORCE_MEMORY_KIND ignored");
    }
    else if (usage & USAGE_ALLOCATE_DEVICE_MEMORY)
    {
        if (dev->kinds & MEMORY_DEVICE)
        {
            kind = MEMORY_DEVICE;
            why = "USAGE_ALLOCATE_DEVICE_MEMORY";
            requested = true;
        }
    }
    else if (usage & (USAGE_ALLOCATE_SHARED_MEMORY | USAGE_ALLOCATE_HOST_MEMORY))
    {
        if (dev->kinds & MEMORY_SHARED_COHERENT)
        {
            kind = MEMORY_SHARED_COHERENT;
            why = "USAGE_ALLOCATE_SHARED_MEMORY";
            requested = true;
        }
        else if (dev->kinds & MEMORY_SHARED_NONCOHERENT)
        {
            kind = MEMORY_SHARED_NONCOHERENT;
            why = "USAGE_ALLOCATE_SHARED_MEMORY";
            requested = true;
        }
    }

    // soft per-device limit, checked without a lock: concurrent allocations may overshoot it
    const size_t limit = pinnedLimit();
    if (limit && isSharedKind(kind) && (dev->kinds & MEMORY_DEVICE)
        && dev->pinnedBytes.load() + size > limit
        // a block of this size and kind is already allocated and already counted: reusing it
        // adds nothing to the total, so the budget has no reason to refuse it
        && !cacheHas(dev, size, kind))
    {
        CV_LOG_DEBUG(NULL, "accel: " << dev->name << " is at its pinned budget ("
                     << dev->pinnedBytes.load() << " + " << size << " > " << limit
                     << " bytes); " << kindName(kind) << " becomes device memory");
        kind = MEMORY_DEVICE;
        why = "OPENCV_ACCEL_PINNED_LIMIT_MB";
        requested = true;   // promotion must not undo the budget on the first CPU access
    }

    CV_LOG_DEBUG(NULL, "accel: " << dev->name << " " << size << " bytes as "
                 << kindName(kind) << " (" << why << ")");
    CV_UNUSED(why);     // the log is compiled out in release builds
    return kind;
}

//! the largest buffer promoted to shared memory; 0 turns promotion off, which is the default
size_t zeroCopyMaxBytes()
{
    return settings().zeroCopyMax.load(std::memory_order_relaxed);
}

//! how much released memory to keep; 0 turns the cache off
size_t cacheLimit()
{
    return settings().cacheLimit.load(std::memory_order_relaxed);
}

//! memory of exactly this size and kind that was released earlier, or NULL
void* cacheTake(Device::Impl* dev, size_t size, int kind)
{
    if (!cacheLimit())
        return 0;
    std::lock_guard<std::mutex> lock(dev->cacheMutex);
    auto it = dev->cache.find(std::make_pair(size, kind));
    if (it == dev->cache.end() || it->second.empty())
        return 0;
    void* p = it->second.back();
    it->second.pop_back();
    dev->cachedBytes -= size;
    return p;
}

//! true when cacheTake() would return something, without taking it
bool cacheHas(Device::Impl* dev, size_t size, int kind)
{
    if (!cacheLimit())
        return false;
    std::lock_guard<std::mutex> lock(dev->cacheMutex);
    auto it = dev->cache.find(std::make_pair(size, kind));
    return it != dev->cache.end() && !it->second.empty();
}

/** @brief Keeps @p ptr for the next buffer of this size and kind. False when the cache is off
or full, and then the caller frees it. The device work that used it has finished. */
bool cachePut(Device::Impl* dev, void* ptr, size_t size, int kind)
{
    const size_t limit = cacheLimit();
    if (!limit || size == 0 || size > limit)
        return false;
    std::lock_guard<std::mutex> lock(dev->cacheMutex);
    if (dev->cachedBytes + size > limit)
        return false;
    dev->cache[std::make_pair(size, kind)].push_back(ptr);
    dev->cachedBytes += size;
    return true;
}

//! gives everything the cache holds back to the device, for when an allocation has failed
void cacheFlush(Device::Impl* dev)
{
    std::map<std::pair<size_t, int>, std::vector<void*> > taken;
    {
        std::lock_guard<std::mutex> lock(dev->cacheMutex);
        taken.swap(dev->cache);
        dev->cachedBytes = 0;
    }
    for (auto it = taken.begin(); it != taken.end(); ++it)
    {
        const size_t size = it->first.first;
        const int kind = it->first.second;
        for (size_t i = 0; i < it->second.size(); i++)
        {
            if (kind == CACHE_HOST)
                dev->backend->deallocateHost(dev->index, it->second[i]);
            else
                dev->backend->deallocate(dev->index, it->second[i], (MemoryKind)kind);
            if (kind == CACHE_HOST || isSharedKind(kind))
                dev->pinnedBytes -= size;
        }
    }
}

void* allocateDevice(Device::Impl* dev, size_t size, MemoryKind kind);

// a device buffer moves to shared memory on its first CPU access when it is small enough
// (OPENCV_ACCEL_ZERO_COPY_MAX_BYTES) and nothing else holds or is using its address
bool canPromote(Device::Impl* dev, const UMatData* u, const BufferState* st)
{
    const size_t limit = zeroCopyMaxBytes();
    if (!limit || u->size == 0 || u->size > limit)
        return false;
    if (isSharedKind(st->kind) || !dev->canShare())
        return false;
    // the caller asked for this kind, or the pinned budget could not afford another one
    if (st->kindRequested)
        return false;
    if (st->imported || st->externallyReferenced || st->mapped || !st->aliases.empty() ||
        st->inflight > 0)
        return false;
    // the newest bytes are in the host copy, which this would not carry over
    if (u->deviceCopyObsolete())
        return false;
    // shared memory is what the budget counts, so promotion has to respect it as well
    const size_t pinLimit = pinnedLimit();
    if (pinLimit && dev->pinnedBytes.load() + u->size > pinLimit)
    {
        CV_LOG_DEBUG(NULL, "accel: " << dev->name << " is at its pinned budget; "
                     << u->size << " bytes stay in device memory");
        return false;
    }
    return true;
}

bool promoteToShared(Device::Impl* dev, UMatData* u, BufferState* st)
{
    if (!canPromote(dev, u, st))
        return false;

    const MemoryKind to = mostSharedKind(dev->kinds);
    void* p = allocateDevice(dev, u->size, to);
    if (!p)
        return false;       // no shared memory to be had: the copy path still works

    Backend& b = *dev->backend;
    if (st->initialized)
    {
        // on the device, so it costs no host bandwidth; the CPU reads it right after
        const Ptr<Stream>& s = currentStream(dev);
        try
        {
            b.copy(dev->index, COPY_DEVICE_TO_DEVICE, DeviceRef(p), u->size,
                   DeviceRef(u->handle), u->size, u->size, 1, *s, true);
            s->record()->wait();
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_WARNING(NULL, "accel: promoting a buffer to shared memory failed: " << e.what());
            b.deallocate(dev->index, p, to);
            dev->pinnedBytes -= u->size;
            return false;
        }
        countCopy(COPY_DEVICE_TO_DEVICE, u->size);
    }

    // all work on it has finished, so it can go straight to the next buffer of its size
    if (!cachePut(dev, u->handle, u->size, st->kind))
        b.deallocate(dev->index, u->handle, st->kind);
    if (u->data && !cachePut(dev, u->data, u->size, CACHE_HOST))
    {
        b.deallocateHost(dev->index, u->data);          // the host copy is not needed any more
        dev->pinnedBytes -= u->size;
    }
    u->handle = p;
    st->kind = to;
    u->data = u->origdata = (uchar*)b.hostAddress(dev->index, p, to);
    u->flags &= ~UMatData::COPY_ON_MAP;
    u->markHostCopyObsolete(false);
    u->markDeviceCopyObsolete(false);
    CV_LOG_DEBUG(NULL, "accel: " << dev->name << " promoted " << u->size << " bytes to "
                 << kindName(to) << " on first CPU access");
    return true;
}

void* allocateDevice(Device::Impl* dev, size_t size, MemoryKind kind)
{
    // memory of this size and kind released earlier is already of the right kind and already
    // counted, so it is handed straight back
    if (void* cached = cacheTake(dev, size, kind))
        return cached;

    collectRetired(false);
    void* p = dev->backend->allocate(dev->index, size, kind);
    if (!p)
    {
        // out of memory: give back what the cache is sitting on, and what device work had
        // finished with, then ask once more
        cacheFlush(dev);
        if (g_retiredCount.load() > 0)
            collectRetired(true);
        p = dev->backend->allocate(dev->index, size, kind);
    }
    // managed memory is not pinned up front, but counting it errs on the safe side
    if (p && isSharedKind(kind))
        dev->pinnedBytes += size;
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

void reloadSettings()
{
    settings().read();
}

MemoryKind newBufferKind(Device::Impl* dev)
{
    // what chooseKind() picks before it knows size and usage; dispatch.cpp predicts outputs with it
    const int forced = forcedKind();
    return (forced && (dev->kinds & forced)) ? (MemoryKind)forced : dev->defaultKind;
}

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
    st->externallyReferenced = true;    // the caller keeps the device pointer; the memory stays put
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
    const int fd = st->dev->backend->exportDmaBuf(st->dev->index, u->handle, u->size, st->kind);
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
                                    AccessFlag /*flags*/, UMatUsageFlags usageFlags) const
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
    st->kind = chooseKind(dev_, total, usageFlags, st->kindRequested);
    u->allocatorContext = std::shared_ptr<void>(st, [](void* p) { delete static_cast<BufferState*>(p); });
    if (total > 0)
    {
        try
        {
            u->handle = allocateDevice(dev_, total, st->kind);
            // an integrated GPU's device memory is a small carve-out: when it is full, a buffer
            // that did not ask for device memory takes shared memory instead of failing
            if (!u->handle && st->kind == MEMORY_DEVICE && !st->kindRequested && dev_->canShare())
            {
                const MemoryKind fallback = isSharedKind(dev_->defaultKind) ? dev_->defaultKind
                                                                            : mostSharedKind(dev_->kinds);
                u->handle = allocateDevice(dev_, total, fallback);
                if (u->handle)
                {
                    CV_LOG_DEBUG(NULL, "accel: " << dev_->name << " has no device memory for "
                                 << total << " bytes; using " << kindName(fallback) << " instead");
                    st->kind = fallback;
                }
            }
        }
        catch (...)
        {
            delete u;
            throw;
        }
        if (!u->handle)
        {
            delete u;
            CV_Error_(Error::StsNoMem, ("accel: %s is out of memory (%llu bytes of %s requested)",
                                        dev_->name.c_str(), (unsigned long long)total,
                                        kindName(st->kind)));
        }
    }
    if (isSharedKind(st->kind))
    {
        u->data = u->origdata = total > 0
            ? (uchar*)dev_->backend->hostAddress(dev_->index, u->handle, st->kind) : 0;
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
    // the kind of an import is the exporter's, not this device's default
    MemoryKind kind = dev_->defaultKind;
    void* p = dev_->backend->importDmaBuf(dev_->index, fd, size, kind);
    if (!p)
        return 0;
    CV_Assert((dev_->kinds & (int)kind) != 0 && "accel: an import of a kind the device does not provide");
    UMatData* u = new UMatData(this);
    u->size = size;
    u->handle = p;
    u->flags |= UMatData::USER_ALLOCATED;   // never released and created again elsewhere
    BufferState* st = new BufferState(dev_);
    st->kind = kind;
    st->imported = true;
    st->initialized = true;
    st->externallyReferenced = true;    // the exporter still holds these bytes
    u->allocatorContext = std::shared_ptr<void>(st, [](void* q) { delete static_cast<BufferState*>(q); });
    if (isSharedKind(kind))
        u->data = u->origdata = (uchar*)dev_->backend->hostAddress(dev_->index, p, kind);
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
    r.kind = st->kind;
    // imported memory, or memory whose address went outside OpenCV, must never be reused
    r.size = (st->imported || st->externallyReferenced) ? 0 : u->size;
    r.pinnedSize = (!st->imported && isSharedKind(st->kind)) ? u->size : 0;
    r.hostCopy = (!isSharedKind(st->kind) && u->data) ? (void*)u->data : 0;
    r.hostCopySize = r.hostCopy ? u->size : 0;
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

    // promotion frees the old memory, so it waits for the readers as well as the writers
    const bool promoting = canPromote(dev_, u, st);
    waitOnCpu(st, write || promoting);
    if (promoting)
        promoteToShared(dev_, u, st);

    Backend& b = *dev_->backend;
    const int i = dev_->index;
    if (isSharedKind(st->kind))
    {
        if (u->size > 0)
        {
            if (st->mapped)
                b.endCpuAccess(i, u->handle, u->size, st->kind, false);  // a read mapping becomes a write mapping
            b.beginCpuAccess(i, u->handle, u->size, st->kind, write);
        }
    }
    else
    {
        if (!u->data && u->size > 0)
        {
            u->data = u->origdata = (uchar*)cacheTake(dev_, u->size, CACHE_HOST);
            if (!u->data)
                u->data = u->origdata = (uchar*)b.allocateHost(i, u->size);
            else
                dev_->pinnedBytes -= u->size;   // added again below, as for a fresh one
            if (!u->data)
                CV_Error_(Error::StsNoMem, ("accel: no host memory for a copy of %llu bytes", (unsigned long long)u->size));
            // counted as pinned: backends pin host copies so copies to and from them are async
            dev_->pinnedBytes += u->size;
        }
        if (u->hostCopyObsolete() && !st->initialized)
            u->markHostCopyObsolete(false);     // fresh memory: its contents are undefined on both sides
        if (u->hostCopyObsolete() && u->size > 0)
        {
            const Ptr<Stream>& s = currentStream(dev_);
            b.copy(i, COPY_DEVICE_TO_HOST, DeviceRef(u->data), u->size,
                   DeviceRef(u->handle), u->size, u->size, 1, *s, true);
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
            if (isSharedKind(st->kind))
            {
                if (u->size > 0)
                    b.endCpuAccess(i, u->handle, u->size, st->kind, st->mappedWrite);
            }
            else if (u->deviceCopyObsolete())
            {
                // the CPU wrote: the upload is queued now, from the host copy, which stays
                // unchanged until it completes (CPU writes wait for it in map())
                const Ptr<Stream>& s = currentStream(dev_);
                orderOnStream(st, s, true);
                b.copy(i, COPY_HOST_TO_DEVICE, DeviceRef(u->handle), u->size,
                       DeviceRef(u->data), u->size, u->size, 1, *s, true);
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
    if (isSharedKind(st->kind) || (!u->hostCopyObsolete() && u->data))
    {
        // the CPU reaches the bytes: the shared memory itself, or a host copy that is current
        waitOnCpu(st, false);
        const bool bracket = isSharedKind(st->kind) && !st->mapped && u->size > 0;
        if (bracket)
            b.beginCpuAccess(i, u->handle, u->size, st->kind, false);
        forEachPlane(dims, sz, srcofs, srcstep, 0, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                memcpy((uchar*)dst + dof + r * dp, u->data + so + r * sp, w);
        });
        if (bracket)
            b.endCpuAccess(i, u->handle, u->size, st->kind, false);
        return;
    }
    const Ptr<Stream>& s = currentStream(dev_);
    orderOnStream(st, s, false);
    const size_t bytes = forEachPlane(dims, sz, srcofs, srcstep, 0, dststep,
                                      [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
        b.copy(i, COPY_DEVICE_TO_HOST, DeviceRef(dst, dof), dp,
               DeviceRef(u->handle, so), sp, w, rows, *s, false);
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
    if (isSharedKind(st->kind) || st->mapped)
    {
        // shared memory, or a copy-only buffer whose Mat views the CPU uses: the CPU writes the
        // bytes it reaches, and unmap() uploads a written host copy
        waitOnCpu(st, true);
        const bool bracket = isSharedKind(st->kind) && !st->mapped && u->size > 0;
        if (bracket)
            b.beginCpuAccess(i, u->handle, u->size, st->kind, true);
        if (st->mapped && !st->mappedWrite)
        {
            // the CPU now writes into a read mapping: make it a write mapping, so unmap() flushes it
            if (isSharedKind(st->kind) && u->size > 0)
            {
                b.endCpuAccess(i, u->handle, u->size, st->kind, false);
                b.beginCpuAccess(i, u->handle, u->size, st->kind, true);
            }
            st->mappedWrite = true;
        }
        forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                     [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
            for (size_t r = 0; r < rows; r++)
                memcpy(u->data + dof + r * dp, (const uchar*)src + so + r * sp, w);
        });
        if (bracket)
            b.endCpuAccess(i, u->handle, u->size, st->kind, true);
        if (!isSharedKind(st->kind))
            u->markDeviceCopyObsolete(true);
        st->initialized = true;
        return;
    }
    const Ptr<Stream>& s = currentStream(dev_);
    orderOnStream(st, s, true);
    const size_t bytes = forEachPlane(dims, sz, 0, srcstep, dstofs, dststep,
                                      [&](size_t so, size_t dof, size_t sp, size_t dp, size_t w, size_t rows) {
        // the caller's memory: the backend reads it before returning
        b.copy(i, COPY_HOST_TO_DEVICE, DeviceRef(u->handle, dof), dp,
               DeviceRef(src, so), sp, w, rows, *s, false);
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
                b.copy(i, COPY_DEVICE_TO_DEVICE, DeviceRef(dstdata->handle, dof), dp,
                       DeviceRef(srcdata->handle, so), sp, w, rows, *s, true);
            });
            Ptr<Event> e = s->record();
            recordUse(ss, s, e, false);
            recordUse(ds, s, e, true);
            ds->initialized = true;
            if (!isSharedKind(ds->kind))    // the destination buffer's own kind, not the device's
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
    CV_Assert(step >= row && step % CV_ELEM_SIZE1(type) == 0 && offset % CV_ELEM_SIZE1(type) == 0);
    // offset + step * (height - 1) + row, refusing a step or offset that would wrap it
    const size_t rowsBefore = (size_t)(size.height - 1);
    CV_Assert((rowsBefore == 0 || step <= (SIZE_MAX - row) / rowsBefore) && "accel: the dma-buf step is too large");
    const size_t span = step * rowsBefore + row;
    CV_Assert(offset <= SIZE_MAX - span && "accel: the dma-buf offset is too large");
    const size_t needed = offset + span;
    size_t bufferSize = 0;
#ifndef _WIN32
    const off_t end = ::lseek(fd, 0, SEEK_END);   // the size of a dma-buf
    if (end > 0)
        bufferSize = (size_t)end;
#endif
    if (bufferSize == 0)
        CV_Error(Error::StsBadArg, "accel: cannot tell the size of the dma-buf");
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
    const int fd = st->dev->backend->exportDmaBuf(st->dev->index, u->handle, u->size, st->kind);
    if (fd < 0)
        CV_Error_(Error::StsNotImplemented, ("accel: %s cannot export its memory as a dma-buf", st->dev->name.c_str()));
    if (!isSharedKind(st->kind))
        u->markHostCopyObsolete(true);  // someone else may write it from now on
    st->initialized = true;
    st->externallyReferenced = true;    // the caller owns the fd; we never learn when it is done
    return fd;
}

}} // namespace cv::accel
