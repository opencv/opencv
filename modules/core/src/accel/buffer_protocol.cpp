// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the coherence protocol for UMat data that lives on an accelerator, and the
// generic allocator that puts it there.
//
// Rules, in the order they are applied:
// 1. Ordering. The CPU never touches a buffer before device work that conflicts with the
//    access has completed: reading waits for the last device write, writing also waits
//    for device reads (write-after-read). Device work waits the same way, on the device
//    when the fence belongs to the same backend.
// 2. Exclusivity. While Mat views of a buffer may exist, the device may read it only if
//    no view was requested for writing, and may never write it. A conflicting operation
//    falls back to the CPU instead of racing with the views.
// 3. Coherence. Memory with a single physical copy that the CPU caches do not snoop
//    (NONCOHERENT) is bracketed by begin_cpu_access / end_cpu_access: caches are
//    invalidated after the device wrote and cleaned before the device may touch the
//    memory again. Discrete memory (COPY) is reached through a host copy that is only
//    refreshed when it is stale.
// 4. Minimal transfers. Read-only CPU access never writes back. A stale host copy is
//    refreshed only when it is read; a stale device copy only when the device reads it or
//    writes part of it. Uninitialized buffers are never transferred.
// 5. Lifetime. A released buffer is freed only after its fences complete. Memory that
//    belongs to the user is not handed back before the device is done with it.
// 6. Device-idle memory (managed memory without concurrent access) is touched by the CPU
//    only after the device drained, and no device work starts while the CPU holds it
//    (IdleGate). Every CPU touch of memory with a CPU view goes through beginCpuAccess /
//    endCpuAccess, so no path can skip a cache operation or the gate.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <functional>

namespace cv { namespace accel { namespace impl {

enum ShadowSource
{
    SHADOW_NONE = 0,
    SHADOW_FASTMALLOC = 1,
    SHADOW_BACKEND = 2,     //!< pinned host memory from the backend: one DMA per transfer
    SHADOW_USER = 3         //!< the Mat that the UMat was created from (Mat::getUMat)
};

BufferState::BufferState(Backend* b, int device_)
    : backend(b), device(device_), shadow(0), shadowFromBackend(false),
      busy(false), pendingReaders(0), pendingWriters(0), hostMapped(false), hostAccess(0),
      imported(false), fromMat(false), hostImport(0), idleGateHeld(false), poisoned(false), generation(0),
      matGeneration(0)
{
    memset(&alloc, 0, sizeof(alloc));
    memset(&shadowAlloc, 0, sizeof(shadowAlloc));
}

static inline int shadowSource(const BufferState* st)
{
    if (!st->shadow)
        return SHADOW_NONE;
    if (st->fromMat)
        return SHADOW_USER;
    return st->shadowFromBackend ? SHADOW_BACKEND : SHADOW_FASTMALLOC;
}

static inline int hostAccessClass(const BufferState* st)
{
    return (int)st->alloc.desc.host_access;
}

static inline bool usesShadow(const BufferState* st)
{
    int c = hostAccessClass(st);
    return c == CV_ACCEL_HOST_ACCESS_COPY || c == CV_ACCEL_HOST_ACCESS_UNCACHED;
}

static inline bool isUninitialized(const UMatData* u)
{
    return u->hostCopyObsolete() && u->deviceCopyObsolete();
}

cvAccelMemoryDesc hostMemoryDesc()
{
    cvAccelMemoryDesc d;
    memset(&d, 0, sizeof(d));
    d.kind = CV_ACCEL_MEM_HOST;
    d.host_access = CV_ACCEL_HOST_ACCESS_COHERENT;
    d.handle_kind = CV_ACCEL_HANDLE_HOST_POINTER;
    return d;
}

bool queryMemoryDesc(const UMatData* u, cvAccelMemoryDesc& desc)
{
    return u && u->currAllocator && u->currAllocator->getMemoryDesc(u, &desc);
}

bool viewFitsAllocation(const cvAccelView& v)
{
    if (!v.alloc || v.dims <= 0 || v.dims > CV_ACCEL_MAX_DIMS)
        return false;
    const uint64_t esz = (uint64_t)CV_ELEM_SIZE(v.type);
    uint64_t last = v.offset;
    for (int i = 0; i < v.dims; i++)
    {
        if (v.size[i] < 0)
            return false;
        if (v.size[i] == 0)
            return true;    // empty: nothing is accessed
        const uint64_t n = (uint64_t)(v.size[i] - 1);
        if (n != 0 && v.step[i] > (UINT64_MAX - last) / n)
            return false;   // overflow
        last += n * v.step[i];
    }
    if (last > UINT64_MAX - esz)
        return false;
    return last + esz <= v.alloc->size;
}

//////////////////////////////////// helpers ////////////////////////////////////

namespace {

//! true while device work that conflicts with an access of this kind is between acquire and record
inline bool deviceUseInFlight(const BufferState* st, bool write)
{
    return st->pendingWriters != 0 || (write && st->pendingReaders != 0);
}

/** Keeps BufferState::busy set for the lifetime of the object, and wakes waiters at the end.
    Busy is set only once no other transition runs and no conflicting device operation sits
    between acquire and record: a transition never holds busy while it waits for one,
    because that operation may itself be waiting for busy (a call holding another buffer). */
struct BusyScope
{
    BusyScope(BufferState* st_, std::unique_lock<std::mutex>& lk_, bool write) : st(st_), lk(lk_)
    {
        st->cond.wait(lk, [this, write]{ return !st->busy && !deviceUseInFlight(st, write); });
        st->busy = true;
    }
    ~BusyScope()
    {
        if (!lk.owns_lock())
            lk.lock();
        st->busy = false;
        st->cond.notify_all();
    }
    BufferState* st;
    std::unique_lock<std::mutex>& lk;
};

//! release path, never throws: true when no fence is pending (a failed device counts as done)
bool fencesSettled(BufferState* st)
{
    if (!st->writeFence.empty())
    {
        if (st->writeFence.poll() == FenceRef::PENDING)
            return false;
        st->writeFence.reset();
    }
    for (size_t i = 0; i < st->readFences.size(); )
    {
        if (st->readFences[i].poll() == FenceRef::PENDING)
            return false;
        st->readFences.erase(st->readFences.begin() + i);
    }
    return true;
}

//! written whole: earlier failures no longer matter
void clearPoison(BufferState* st)
{
    st->poisoned = false;
    st->poisonReason.clear();
}

void checkNotPoisoned(const BufferState* st)
{
    if (st->poisoned)
        CV_Error_(Error::StsError, ("accel(%s): the device operation that produced this UMat failed: %s",
                                    st->backend->name().c_str(), st->poisonReason.c_str()));
}

/** Waits on the CPU for device work that conflicts with a CPU access. Called with st->busy
    set, so no new device work on this buffer can start meanwhile; the mutex is released
    during the waits so that operations in flight can record their fences. A poisoned buffer
    throws, unless the caller is about to overwrite all of it (@p overwriteAll). */
void waitDeviceWork(BufferState* st, std::unique_lock<std::mutex>& lk, bool write, bool overwriteAll = false)
{
    CV_DbgAssert(st->busy && !deviceUseInFlight(st, write));  // see BusyScope
    FenceRef lastWrite;
    if (!st->writeFence.empty() && !st->writeFence.isComplete())
        lastWrite = st->writeFence;
    std::vector<FenceRef> reads;
    if (write)
    {
        for (size_t i = 0; i < st->readFences.size(); i++)
            if (!st->readFences[i].isComplete())
                reads.push_back(st->readFences[i]);
    }
    if (lastWrite.empty() && reads.empty())
    {
        if (!overwriteAll)
            checkNotPoisoned(st);
        return;
    }
    lk.unlock();
    try
    {
        lastWrite.wait();
    }
    catch (const cv::Exception& e)
    {
        // async errors surface here: the content is not what the operation should have written
        lk.lock();
        st->poisoned = true;
        st->poisonReason = e.err;
        throw;
    }
    for (size_t i = 0; i < reads.size(); i++)
        reads[i].wait();
    lk.lock();
    // completed fences carry no information any more
    if (st->writeFence.isComplete())
        st->writeFence.reset();
    if (write)
        st->readFences.clear();
    if (!overwriteAll)
        checkNotPoisoned(st);
}

void ensureShadow(BufferState* st)
{
    if (st->shadow)
        return;
    Backend* b = st->backend;
    if ((b->deviceInfo(st->device).caps & CV_ACCEL_CAP_HOST_PINNED) && b->pinnedShadowAllowed(st->device))
    {
        cvAccelAllocation a;
        memset(&a, 0, sizeof(a));
        if (b->takePinned(st->device, st->alloc.size, a) ||
            b->fn().allocate(st->device, st->alloc.size, CV_ACCEL_MEM_HOST_PINNED, 0, &a) == CV_ACCEL_OK)
        {
            if (a.desc.host_access == CV_ACCEL_HOST_ACCESS_COHERENT && a.desc.handle_kind == CV_ACCEL_HANDLE_HOST_POINTER)
            {
                a.desc.backend_id = (uint32_t)b->id();
                st->shadowAlloc = a;
                st->shadow = (uchar*)(uintptr_t)a.handle;
                st->shadowFromBackend = true;
                return;
            }
            // The CPU works in the host copy: memory it reaches uncached (pinned memory on
            // Jetson before Xavier), write-combined or without coherence would slow every
            // access down or need cache maintenance. Pageable memory from now on.
            CV_LOG_INFO(NULL, "accel(" << b->name() << "): pinned memory of device " << st->device
                        << " is not CPU-cached; host copies use pageable memory");
            b->disallowPinnedShadow(st->device);
            if (b->fn().deallocate(&a) != CV_ACCEL_OK)
                CV_LOG_ERROR(NULL, "accel(" << b->name() << "): failed to free pinned memory: " << b->fn().get_last_error());
        }
        else
        {
            // pinned memory is a limited resource: fall back to pageable memory
            CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): no pinned memory for a host copy, using pageable memory");
        }
    }
    st->shadow = (uchar*)fastMalloc((size_t)st->alloc.size);
    st->shadowFromBackend = false;
}

void fillRegion1D(cvAccelCopyRegion& r, uint64_t bytes)
{
    memset(&r, 0, sizeof(r));
    r.struct_size = (uint32_t)sizeof(r);
    r.dims = 1;
    r.size[0] = bytes;
}

//! whole-buffer device -> host copy into the shadow; waits for completion
void downloadWhole(BufferState* st)
{
    Backend* b = st->backend;
    StreamPtr s = b->defaultStream(st->device);
    cvAccelCopyRegion r;
    fillRegion1D(r, st->alloc.size);
    r.src.alloc = &st->alloc;
    r.dst.host = st->shadow;
    if (shadowSource(st) == SHADOW_BACKEND)
        r.flags |= CV_ACCEL_COPY_HOST_PINNED;
    FenceRef f;
    try
    {
        b->check(b->fn().copy(st->device, s->handle, &r), "copy (device to host)", st->device);
        f = recordFence(s);
        f.wait();
    }
    catch (...)
    {
        if (f.empty())
            fenceAfterFailure(s).waitNoThrow();     // part of the copy may be queued, writing the host copy
        throw;
    }
    Counters& c = counters();
    c.bytesD2H += st->alloc.size;
    c.transfers++;
}

//! whole-buffer host -> device copy from the shadow, enqueued on @p s
void uploadWhole(BufferState* st, const StreamPtr& s)
{
    Backend* b = st->backend;
    cvAccelCopyRegion r;
    fillRegion1D(r, st->alloc.size);
    r.src.host = st->shadow;
    r.dst.alloc = &st->alloc;
    if (shadowSource(st) == SHADOW_BACKEND)
        r.flags |= CV_ACCEL_COPY_HOST_PINNED;
    b->check(b->fn().copy(st->device, s->handle, &r), "copy (host to device)", st->device);
    Counters& c = counters();
    c.bytesH2D += st->alloc.size;
    c.transfers++;
}

/** Makes the device copy current from the shadow. The upload overwrites the device copy,
    so it waits for every device user of the old content first. */
void uploadShadowLocked(BufferState* st, UMatData* u, const StreamPtr& s, bool waitNow)
{
    if (!st->writeFence.empty())
        streamWait(s, st->writeFence);
    for (size_t i = 0; i < st->readFences.size(); i++)
        streamWait(s, st->readFences[i]);
    FenceRef f;
    try
    {
        uploadWhole(st, s);
        f = recordFence(s);
        if (waitNow)
            f.wait();
    }
    catch (...)
    {
        // part of the copy may be queued: later users wait for it. The device copy stays
        // obsolete, so its next use uploads again.
        if (f.empty())
        {
            FenceRef after = fenceAfterFailure(s);
            if (!after.empty())
                st->writeFence = after;
        }
        throw;
    }
    st->writeFence = f;
    st->readFences.clear();
    u->markDeviceCopyObsolete(false);
}

/** Announces CPU access to DEVICE_IDLE memory on the device's gate and waits until no
    submission is in progress. Must be created before the buffer is locked (see IdleGate).
    One-shot accesses release the count when the guard goes away; a mapping keeps it
    (commit) until endHostAccessLocked. Does nothing for other memory. */
class IdleCpuGuard
{
public:
    explicit IdleCpuGuard(BufferState* st) : gate_(0), committed_(false)
    {
        if (st->alloc.desc.host_access != CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
            return;
        gate_ = &st->backend->idleGate(st->device);
        gate_->cpuMappings++;
        std::unique_lock<std::mutex> lk(gate_->mutex);
        if (std::find(gate_->holders.begin(), gate_->holders.end(), std::this_thread::get_id()) != gate_->holders.end())
        {
            // waiting would mean waiting for ourselves
            gate_->cpuMappings--;
            gate_ = 0;
            CV_Error(Error::StsError, "accel: a DeviceAccess open on this thread may run device work at any time, "
                                      "so the CPU cannot touch device-idle memory of that device until it is destroyed");
        }
        gate_->cond.wait(lk, [this]{ return gate_->submitting.load() == 0; });
    }
    ~IdleCpuGuard()
    {
        if (gate_ && !committed_)
            gate_->cpuMappings--;
    }
    bool active() const { return gate_ != 0; }
    void commit() { committed_ = true; }
private:
    IdleCpuGuard(const IdleCpuGuard&);
    IdleCpuGuard& operator=(const IdleCpuGuard&);
    IdleGate* gate_;
    bool committed_;
};

/** Opens CPU access to memory that has a CPU view (all classes but COPY, UNCACHED and NONE).
    Every CPU touch of such memory goes through this pair, so none of them can skip the cache
    maintenance. For DEVICE_IDLE memory the caller holds an IdleCpuGuard, and this drains the
    device first. */
void* beginCpuAccess(BufferState* st, int access)
{
    Backend* b = st->backend;
    if (hostAccessClass(st) == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
        b->check(b->fn().device_synchronize(st->device), "device_synchronize", st->device);
    void* p = 0;
    cvAccelStatus status = b->fn().begin_cpu_access(&st->alloc, 0, st->alloc.size, (uint32_t)access, &p);
    if (status == CV_ACCEL_OK && !p)
        status = CV_ACCEL_INTERNAL_ERROR;
    b->check(status, "begin_cpu_access", st->device);
    return p;
}

//! closes CPU access; never throws (also runs from Mat destructors)
bool endCpuAccess(BufferState* st, int access)
{
    Backend* b = st->backend;
    cvAccelStatus status = b->fn().end_cpu_access(&st->alloc, 0, st->alloc.size, (uint32_t)access);
    if (status == CV_ACCEL_OK)
        return true;
    CV_LOG_ERROR(NULL, "accel(" << b->name() << "): end_cpu_access failed: " << b->fn().get_last_error()
                 << ". CPU writes may not be visible to the device.");
    if (status == CV_ACCEL_DEVICE_LOST)
        b->markLost(st->device);
    return false;
}

void beginHostAccessLocked(BufferState* st, UMatData* u, int access)
{
    if (usesShadow(st))
    {
        const bool fresh = !st->shadow;     // a new host copy holds nothing yet, whatever the flags say
        ensureShadow(st);
        u->data = st->shadow;
        if ((u->hostCopyObsolete() || fresh) && !u->deviceCopyObsolete())
        {
            downloadWhole(st);
            u->markHostCopyObsolete(false);
        }
        else if (isUninitialized(u) && (access & ACCESS_WRITE))
        {
            // nothing to fetch: from now on the host copy is the content
            u->markHostCopyObsolete(false);
        }
        return;
    }
    u->data = (uchar*)beginCpuAccess(st, access);
    u->markHostCopyObsolete(false);
}

void upgradeHostAccessLocked(BufferState* st, UMatData* u)
{
    if (usesShadow(st))
    {
        if (isUninitialized(u))
            u->markHostCopyObsolete(false);
        return;
    }
    // the mapping is already open (and holds the device gate): only announce the write
    void* p = 0;
    Backend* b = st->backend;
    b->check(b->fn().begin_cpu_access(&st->alloc, 0, st->alloc.size, ACCESS_WRITE, &p), "begin_cpu_access", st->device);
    CV_Assert(p == (void*)u->data);
}

//! ends CPU access; never throws (runs from Mat destructors)
void endHostAccessLocked(BufferState* st, UMatData* u)
{
    const int access = st->hostAccess;
    if (usesShadow(st))
    {
        if (access & ACCESS_WRITE)
        {
            // the device copy is refreshed lazily, when the device needs it
            u->markHostCopyObsolete(false);
            u->markDeviceCopyObsolete(true);
            st->generation++;
        }
    }
    else
    {
        endCpuAccess(st, access);
        if (access & ACCESS_WRITE)
        {
            u->markDeviceCopyObsolete(false);
            st->generation++;
        }
    }
    if (st->idleGateHeld)
    {
        st->backend->idleGate(st->device).cpuMappings--;
        st->idleGateHeld = false;
    }
    st->hostMapped = false;
    st->hostAccess = 0;
    u->mapcount = 0;
}

void checkHostAccessAllowed(const BufferState* st, int access)
{
    const cvAccelMemoryDesc& d = st->alloc.desc;
    if (d.host_access == CV_ACCEL_HOST_ACCESS_NONE || (d.flags & CV_ACCEL_MEMF_PROTECTED) || d.layout_modifier != 0)
        CV_Error_(Error::StsNotImplemented,
                  ("accel(%s): this UMat is in memory the CPU cannot access (protected content or a "
                   "non-linear layout); use an operation that the device implements",
                   st->backend->name().c_str()));
    if ((access & ACCESS_WRITE) && (d.flags & CV_ACCEL_MEMF_READ_ONLY))
        CV_Error_(Error::StsError, ("accel(%s): this UMat wraps read-only memory", st->backend->name().c_str()));
}

void copyND(uchar* dst, const size_t* dststep, const uchar* src, const size_t* srcstep,
            int dims, const size_t* sz)
{
    // sz[dims-1] is in bytes (MatAllocator convention)
    if (dims <= 1)
    {
        memcpy(dst, src, sz[0]);
        return;
    }
    for (size_t i = 0; i < sz[0]; i++)
        copyND(dst + i * dststep[0], dststep + 1, src + i * srcstep[0], srcstep + 1, dims - 1, sz + 1);
}

size_t offsetND(const size_t* ofs, const size_t* step, int dims)
{
    size_t r = 0;
    if (!ofs)
        return 0;
    for (int i = 0; i < dims - 1; i++)
        r += ofs[i] * step[i];
    return r + ofs[dims - 1];   // last offset is in bytes
}

size_t totalND(const size_t* sz, int dims)
{
    size_t t = 1;
    for (int i = 0; i < dims; i++)
        t *= sz[i];
    return t;
}

//! a DeviceAccess open on this thread would make the caller wait for itself: report it instead
void checkOwnHolds(const BufferState* st, bool write)
{
    const std::thread::id self = std::this_thread::get_id();
    for (size_t i = 0; i < st->longHolds.size(); i++)
    {
        const BufferState::Hold& h = st->longHolds[i];
        if (h.thread == self && (write || (h.mode & DEV_WRITE)))
            CV_Error(Error::StsError, "accel: a DeviceAccess opened on this thread still holds this UMat; "
                                      "destroy it before using the UMat elsewhere");
    }
}

} // namespace

//////////////////////////////////// host access ////////////////////////////////////

Mat getHostView(const UMat& m, AccessFlag accessFlags)
{
    UMatData* u = m.u;
    BufferState* st = u->accelState;
    int access = accessFlags & ACCESS_RW;
    if (access == 0)
        access = ACCESS_READ;
    checkHostAccessAllowed(st, access);

    {
        IdleCpuGuard idle(st);  // before the lock: see IdleGate
        std::unique_lock<std::mutex> lk(st->mutex);
        const bool write = (access & ACCESS_WRITE) != 0;
        checkOwnHolds(st, write);
        st->cond.wait(lk, [st, write]{ return !st->busy && !deviceUseInFlight(st, write); });
        const bool first = !st->hostMapped;
        const bool upgrade = !first && write && !(st->hostAccess & ACCESS_WRITE);
        if (first || upgrade)
        {
            BusyScope busy(st, lk, write);
            waitDeviceWork(st, lk, write);
            if (first)
                beginHostAccessLocked(st, u, access);
            else
                upgradeHostAccessLocked(st, u);
            st->hostMapped = true;
            u->mapcount = 1;    // UMatData::~UMatData unmaps chained views (getMat().getUMat()) only then
            if (first && idle.active())
            {
                st->idleGateHeld = true;    // the mapping holds the gate until it ends
                idle.commit();
            }
        }
        st->hostAccess |= access;
        CV_XADD(&u->refcount, 1);
    }

    Mat hdr(m.dims, m.size.p, m.type(), u->data + m.offset, m.step.p);
    hdr.flags = m.flags;
    hdr.u = u;
    hdr.datastart = u->data;
    hdr.data = u->data + m.offset;
    hdr.datalimit = hdr.dataend = u->data + u->size;
    return hdr;
}

void endHostView(UMatData* u)
{
    BufferState* st = u->accelState;
    std::unique_lock<std::mutex> lk(st->mutex);
    st->cond.wait(lk, [st]{ return !st->busy; });
    // a new view may have been created after the last one went away
    if (!st->hostMapped || u->refcount != 0)
        return;
    endHostAccessLocked(st, u);
    st->cond.notify_all();
}

void* deviceHandleBlocking(UMatData* u, AccessFlag access)
{
    BufferState* st = u->accelState;
    std::unique_lock<std::mutex> lk(st->mutex);
    checkOwnHolds(st, true);
    BusyScope busy(st, lk, true);
    if (st->hostMapped)
    {
        if (u->refcount == 0)
            endHostAccessLocked(st, u);
        else
            CV_Error(Error::StsError, "UMat::handle(): a Mat view of this UMat is alive");
    }
    // the caller's work may go to any stream: wait for everything
    waitDeviceWork(st, lk, true);
    if (usesShadow(st) && u->deviceCopyObsolete() && !u->hostCopyObsolete())
        uploadShadowLocked(st, u, st->backend->defaultStream(st->device), true);
    if (access & ACCESS_WRITE)
    {
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
        st->generation++;
    }
    return (void*)(uintptr_t)st->alloc.handle;
}

//////////////////////////////////// device access ////////////////////////////////////

namespace {

//! acquires one buffer (see DeviceAcquisition); false when a live Mat view conflicts
bool acquireOne(UMatData* u, const StreamPtr& stream, int mode, bool longHold, std::vector<FenceRef>& waits)
{
    BufferState* st = u->accelState;
    std::unique_lock<std::mutex> lk(st->mutex);
    const bool write = (mode & DEV_WRITE) != 0;
    checkOwnHolds(st, write);
    st->cond.wait(lk, [st, write]{ return !st->busy && !deviceUseInFlight(st, write); });
    const bool needContent = (mode & DEV_READ) || !(mode & DEV_DISCARD);
    if (needContent)
        checkNotPoisoned(st);   // a write of the whole buffer may reuse a poisoned one

    if (st->hostMapped)
    {
        if (u->refcount == 0)
            endHostAccessLocked(st, u);     // the last view is gone, its unmap has not run yet
        else if (write || (st->hostAccess & ACCESS_WRITE))
            return false;                   // live Mat views: the device must not race with them
    }
    if (write && (st->alloc.desc.flags & CV_ACCEL_MEMF_READ_ONLY))
        return false;

    if (usesShadow(st) && u->deviceCopyObsolete() && !u->hostCopyObsolete() && needContent)
    {
        // the host copy is newer: the backend that owns the memory uploads it
        const bool sameBackend = stream->backend == st->backend;
        StreamPtr us = sameBackend ? stream : st->backend->defaultStream(st->device);
        uploadShadowLocked(st, u, us, !sameBackend);
    }

    if (!st->writeFence.empty() && st->writeFence.streamUid() != stream->uid)
        waits.push_back(st->writeFence);
    if (write)
    {
        for (size_t i = 0; i < st->readFences.size(); i++)
            if (st->readFences[i].streamUid() != stream->uid)
                waits.push_back(st->readFences[i]);
        st->pendingWriters++;
    }
    else
        st->pendingReaders++;
    if (longHold)
    {
        BufferState::Hold h;
        h.thread = std::this_thread::get_id();
        h.mode = mode;
        st->longHolds.push_back(h);
    }
    return true;
}

//! ends one acquisition; with @p fence the work was enqueued and completes at that fence
void finishOne(UMatData* u, int mode, bool longHold, std::thread::id holder, const FenceRef* fence,
               const char* failure = NULL)
{
    BufferState* st = u->accelState;
    std::lock_guard<std::mutex> lk(st->mutex);
    const bool write = (mode & DEV_WRITE) != 0;
    if (fence)
    {
        if (write)
        {
            // writers exclude everyone, so every read fence here was waited for by this work
            st->writeFence = *fence;
            st->readFences.clear();
            u->markHostCopyObsolete(true);
            u->markDeviceCopyObsolete(false);
            st->generation++;
            if (failure)
            {
                // part of the content may have been written: invalid until written whole
                st->poisoned = true;
                st->poisonReason = failure;
            }
            else if (mode & DEV_DISCARD)
            {
                clearPoison(st);
            }
        }
        else if (!fence->empty())
        {
            bool replaced = false;
            for (size_t i = 0; i < st->readFences.size(); i++)
            {
                if (st->readFences[i].streamUid() == fence->streamUid())
                {
                    st->readFences[i] = *fence;     // in-order stream: the newer fence covers the older one
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
                st->readFences.push_back(*fence);
        }
    }
    int& pending = write ? st->pendingWriters : st->pendingReaders;
    CV_DbgAssert(pending > 0);
    if (pending > 0)
        pending--;
    if (longHold)
    {
        for (size_t i = 0; i < st->longHolds.size(); i++)
        {
            if (st->longHolds[i].thread == holder && st->longHolds[i].mode == mode)
            {
                st->longHolds.erase(st->longHolds.begin() + i);
                break;
            }
        }
    }
    st->cond.notify_all();
}

} // namespace

void DeviceAcquisition::add(UMatData* u, int mode)
{
    CV_Assert(u && u->accelState && acquired_ == 0);
    for (size_t i = 0; i < uses_.size(); i++)
    {
        if (uses_[i].u == u)
        {
            // the same buffer twice: one acquisition doing both; discard only if both overwrite it all
            const int m = uses_[i].mode;
            const bool discard = (m & DEV_DISCARD) && (mode & DEV_DISCARD) && !((m | mode) & DEV_READ);
            uses_[i].mode = ((m | mode) & (DEV_READ | DEV_WRITE)) | (discard ? DEV_DISCARD : 0);
            return;
        }
    }
    DeviceUse d;
    d.u = u;
    d.mode = mode;
    uses_.push_back(d);
}

bool DeviceAcquisition::acquire(const StreamPtr& stream, std::vector<FenceRef>& waits)
{
    CV_Assert(acquired_ == 0);
    // one global order: an acquirer holding a buffer only ever waits for buffers after it
    std::sort(uses_.begin(), uses_.end(), [](const DeviceUse& a, const DeviceUse& b) {
        return std::less<const BufferState*>()(a.u->accelState, b.u->accelState);
    });
    holder_ = std::this_thread::get_id();
    try
    {
        for (; acquired_ < uses_.size(); acquired_++)
        {
            if (!acquireOne(uses_[acquired_].u, stream, uses_[acquired_].mode, longHold_, waits))
            {
                abandon();
                return false;
            }
        }
    }
    catch (...)
    {
        abandon();
        throw;
    }
    return true;
}

void DeviceAcquisition::record(const FenceRef& fence)
{
    for (size_t i = 0; i < acquired_; i++)
        finishOne(uses_[i].u, uses_[i].mode, longHold_, holder_, &fence);
    acquired_ = 0;
}

void DeviceAcquisition::fail(const FenceRef& fence, const std::string& reason)
{
    for (size_t i = 0; i < acquired_; i++)
        finishOne(uses_[i].u, uses_[i].mode, longHold_, holder_, &fence, reason.c_str());
    acquired_ = 0;
}

void DeviceAcquisition::abandon()
{
    while (acquired_ > 0)
    {
        acquired_--;
        finishOne(uses_[acquired_].u, uses_[acquired_].mode, longHold_, holder_, NULL);
    }
}

//////////////////////////////////// release ////////////////////////////////////

void destroyBufferState(BufferState* st)
{
    Backend* b = st->backend;
    // A lost device rejects every call, or is still running after a timed-out wait (where a
    // free would block, as cudaFree does): its allocations are leaked, host memory is freed.
    const bool lost = b->isLost(st->device);
    if (shadowSource(st) == SHADOW_BACKEND)
    {
        if (!lost)
        {
            st->shadowAlloc.desc.device_index = (uint32_t)st->device;
            b->recyclePinned(st->shadowAlloc);
        }
    }
    else if (shadowSource(st) == SHADOW_FASTMALLOC)
    {
        fastFree(st->shadow);
    }
    st->shadow = 0;
    if (!lost && (st->alloc.handle || st->alloc.backend_data))
    {
        if (b->fn().deallocate(&st->alloc) != CV_ACCEL_OK)
            CV_LOG_ERROR(NULL, "accel(" << b->name() << "): failed to free device memory: " << b->fn().get_last_error());
    }
    if (st->hostImport)
        b->releaseHostImport(st->device, st->hostImport);
    delete st;
    noteBufferDestroyed();
}

bool isDeviceAllocator(const MatAllocator* a)
{
    return dynamic_cast<const DeviceAllocator*>(a) != NULL;
}

bool sameAccelDevice(const UMatData* a, const UMatData* b)
{
    return a && b && a->accelState && b->accelState &&
           a->accelState->backend == b->accelState->backend &&
           a->accelState->device == b->accelState->device;
}

/** Release path: waits for every fence on a buffer that nothing refers to any more. Runs
    from ~UMat, so it never throws; false when the device failed (logged, device marked lost). */
static bool drainForRelease(BufferState* st) CV_NOEXCEPT
{
    bool ok = st->writeFence.waitNoThrow();
    for (size_t i = 0; i < st->readFences.size(); i++)
        ok = st->readFences[i].waitNoThrow() && ok;
    st->writeFence.reset();
    st->readFences.clear();
    return ok && !st->backend->isLost(st->device);
}

//! release path: the device result back into the memory of the Mat the UMat came from; never throws
static void copyBackToMat(UMatData* u, BufferState* st) CV_NOEXCEPT
{
    try
    {
        if (usesShadow(st))
        {
            downloadWhole(st);  // into the Mat's memory
        }
        else
        {
            IdleCpuGuard idle(st);
            void* p = beginCpuAccess(st, ACCESS_READ);
            memcpy(u->origdata, p, u->size);
            endCpuAccess(st, ACCESS_READ);
        }
        return;
    }
    catch (const std::exception& e)
    {
        try { CV_LOG_ERROR(NULL, "accel: " << e.what()); } catch (...) {}
    }
    catch (...) {}
    try
    {
        CV_LOG_ERROR(NULL, "accel(" << st->backend->name() << "): the device result could not be copied back; "
                     "the Mat keeps its previous content");
    }
    catch (...) {}
}

static bool readyToFree(BufferState* st)
{
    std::lock_guard<std::mutex> lk(st->mutex);
    return st->pendingReaders == 0 && st->pendingWriters == 0 && fencesSettled(st);
}

//////////////////////////////////// DeviceAllocator ////////////////////////////////////

DeviceAllocator::DeviceAllocator(Backend* b, int device, int kind, bool protectedMem)
    : b_(b), device_(device), kind_(kind), protected_(protectedMem)
{}

int DeviceAllocator::kindFor(UMatUsageFlags usageFlags) const
{
    if (usageFlags & USAGE_ALLOCATE_SHARED_MEMORY)
        return CV_ACCEL_MEM_SHARED;
    if (usageFlags & USAGE_ALLOCATE_HOST_MEMORY)
        return CV_ACCEL_MEM_HOST_PINNED;
    if (usageFlags & USAGE_ALLOCATE_DEVICE_MEMORY)
        return CV_ACCEL_MEM_DEVICE;
    return kind_;
}

UMatData* DeviceAllocator::createState(UMatData* u, const cvAccelAllocation& a, bool imported) const
{
    BufferState* st = new BufferState(b_, device_);
    st->alloc = a;
    st->alloc.desc.backend_id = (uint32_t)b_->id();
    st->alloc.desc.device_index = (uint32_t)device_;
    st->imported = imported;
    u->accelState = st;
    u->handle = (void*)(uintptr_t)a.handle;
    if (a.desc.host_access == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
        b_->idleGate(device_).used = true;
    noteBufferCreated();
    return u;
}

UMatData* DeviceAllocator::allocate(int dims, const int* sizes, int type, void* data0, size_t* step,
                                    AccessFlag /*flags*/, UMatUsageFlags usageFlags) const
{
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--)
    {
        if (step)
            step[i] = total;
        total *= (size_t)sizes[i];
    }
    if (data0)
    {
        // a UMat over existing host memory: import it when the device can, never copy silently
        cvAccelExternalMemory ext;
        memset(&ext, 0, sizeof(ext));
        ext.struct_size = (uint32_t)sizeof(ext);
        ext.type = CV_ACCEL_EXTMEM_HOST_POINTER;
        ext.handle = (int64_t)(intptr_t)data0;
        ext.size = total;
        return wrap(ext, total);
    }

    cvAccelAllocation a;
    memset(&a, 0, sizeof(a));
    const int kind = kindFor(usageFlags);
    const uint32_t allocFlags = protected_ ? CV_ACCEL_MEMF_PROTECTED : 0;
    cvAccelStatus status = CV_ACCEL_OK;
    if (total > 0)
    {
        status = b_->fn().allocate(device_, total, (uint32_t)kind, allocFlags, &a);
        if (status == CV_ACCEL_OUT_OF_MEMORY)
        {
            // memory held by buffers that are released but still in flight
            b_->collectDeferred(true);
            status = b_->fn().allocate(device_, total, (uint32_t)kind, allocFlags, &a);
        }
        if (protected_ && status == CV_ACCEL_NOT_IMPLEMENTED)
            CV_Error_(Error::StsNotImplemented, ("accel(%s): device %d has no protected memory for results "
                                                 "computed from protected content", b_->name().c_str(), device_));
        b_->check(status, "allocate", device_);
        if (protected_ && !(a.desc.flags & CV_ACCEL_MEMF_PROTECTED))
        {
            if (b_->fn().deallocate(&a) != CV_ACCEL_OK)
                CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): failed to free device memory: " << b_->fn().get_last_error());
            CV_Error_(Error::StsError, ("accel(%s): the backend returned ordinary memory for protected content",
                                        b_->name().c_str()));
        }
    }
    UMatData* u = new UMatData(this);
    u->size = total;
    createState(u, a, false);
    // nothing has been written yet: no copy is ever needed for this content
    u->markHostCopyObsolete(true);
    u->markDeviceCopyObsolete(true);
    return u;
}

UMatData* DeviceAllocator::wrap(const cvAccelExternalMemory& ext, size_t total) const
{
    if (!b_->fn().import_memory)
        CV_Error_(Error::StsNotImplemented, ("accel(%s): the backend cannot import external memory", b_->name().c_str()));
    const void* hostPtr = ext.type == CV_ACCEL_EXTMEM_HOST_POINTER ? (const void*)((uchar*)(intptr_t)ext.handle + ext.offset) : 0;
    const uint32_t align = hostPtr ? b_->deviceInfo(device_).host_import_alignment : 0;
    if (align > 1 && (uintptr_t)hostPtr % align != 0)
        CV_Error_(Error::StsBadArg, ("accel(%s): the device imports host memory on %u-byte boundaries only; "
                                     "copy it into a UMat instead", b_->name().c_str(), (unsigned)align));
    if (hostPtr && !b_->reserveHostImport(device_, hostPtr, (size_t)ext.size))
        CV_Error_(Error::StsError, ("accel(%s): this host memory overlaps memory that is already imported "
                                    "(pinning works on whole pages); copy it into a UMat instead", b_->name().c_str()));
    cvAccelAllocation a;
    memset(&a, 0, sizeof(a));
    cvAccelStatus status = b_->fn().import_memory(device_, &ext, &a);
    if (status != CV_ACCEL_OK && hostPtr)
        b_->releaseHostImport(device_, hostPtr);
    if (status == CV_ACCEL_NOT_IMPLEMENTED)
        CV_Error_(Error::StsNotImplemented, ("accel(%s): this kind of external memory cannot be imported "
                                             "(type %d); copy it into a UMat instead", b_->name().c_str(), (int)ext.type));
    b_->check(status, "import_memory", device_);
    if (hostPtr && a.desc.host_access == CV_ACCEL_HOST_ACCESS_NONCOHERENT)
    {
        // cache maintenance works on whole granules: a partial one at either end would write
        // back or drop bytes of the data next to it (backend rule 3)
        const uint64_t g = a.desc.coherence_granule;
        if (g == 0 || (uintptr_t)hostPtr % g != 0 || ext.size % g != 0)
        {
            if (b_->fn().deallocate(&a) != CV_ACCEL_OK)
                CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): failed to release an import: " << b_->fn().get_last_error());
            b_->releaseHostImport(device_, hostPtr);
            CV_Error_(Error::StsBadArg, ("accel(%s): host memory imported without cache coherence must start and end "
                                         "on %u-byte boundaries; copy it into a UMat instead",
                                         b_->name().c_str(), (unsigned)g));
        }
    }
    CV_Assert(a.size >= total);
    UMatData* u = new UMatData(this);
    u->size = total;
    u->flags |= UMatData::USER_ALLOCATED;
    createState(u, a, true);
    u->accelState->hostImport = hostPtr;
    u->accelState->alloc.desc.flags |= CV_ACCEL_MEMF_IMPORTED;
    // The content belongs to the owner and lives in the memory, which the device addresses.
    // Memory with a CPU view is also the host copy; COPY and UNCACHED memory get their host
    // copy on the first CPU access, memory without CPU access never.
    BufferState* st = u->accelState;
    u->markDeviceCopyObsolete(false);
    u->markHostCopyObsolete(usesShadow(st) || hostAccessClass(st) == CV_ACCEL_HOST_ACCESS_NONE);
    return u;
}

bool DeviceAllocator::allocate(UMatData* u, AccessFlag /*accessFlags*/, UMatUsageFlags usageFlags) const
{
    // Mat::getUMat(): u describes the Mat's host memory (origdata). This runs on every call
    // that turns a Mat into a UMat, so it never pins the Mat (registering memory is slow and
    // stalls other streams): the data goes to device memory, and only when the device needs
    // it. Zero-copy use of host memory is explicit (wrapExternalMemory), or automatic for a
    // Mat passed straight to an operation on a device that can address host memory.
    if (!u)
        return false;
    if (u->accelState)
        return true;
    CV_Assert(u->origdata != 0);

    cvAccelAllocation a;
    memset(&a, 0, sizeof(a));
    if (b_->fn().allocate(device_, u->size, (uint32_t)kindFor(usageFlags), 0, &a) != CV_ACCEL_OK)
        return false;   // Mat::getUMat falls back to the host allocator
    createState(u, a, false);
    BufferState* st = u->accelState;
    st->fromMat = true;
    if (usesShadow(st))
    {
        // the Mat's memory is the host copy; the device copy is filled when the device needs it
        st->shadow = u->origdata;
        u->data = u->origdata;
        u->markHostCopyObsolete(false);
        u->markDeviceCopyObsolete(true);
    }
    else
    {
        // memory with a CPU view: copy the content in now, the Mat keeps its own bytes and
        // gets them back only if the UMat is written (see matGeneration)
        IdleCpuGuard idle(st);
        void* p = beginCpuAccess(st, ACCESS_WRITE);
        memcpy(p, u->origdata, u->size);
        endCpuAccess(st, ACCESS_WRITE);
        u->markHostCopyObsolete(false);
        u->markDeviceCopyObsolete(false);
    }
    st->matGeneration = st->generation;
    u->prevAllocator = u->currAllocator;
    u->currAllocator = this;
    u->flags |= UMatData::TEMP_UMAT;
    return true;
}

void DeviceAllocator::deallocate(UMatData* u) const
{
    if (!u)
        return;
    CV_Assert(u->urefcount == 0);
    CV_Assert(u->refcount == 0 && "UMat deallocation error: some derived Mat is still alive");
    BufferState* st = u->accelState;
    CV_Assert(st);

    // From here on errors are logged, not thrown: this runs from ~UMat.
    if (st->hostMapped)
    {
        // the last view went without an unmap: flush its CPU writes, release the device gate
        std::lock_guard<std::mutex> lk(st->mutex);
        endHostAccessLocked(st, u);
    }
    if (st->fromMat)
    {
        // the memory or the content belongs to a Mat: finish everything, then hand it back
        const bool deviceOk = drainForRelease(st);
        // For COPY and UNCACHED memory the Mat's memory is the host copy: only a device write
        // leaves it stale. Other memory holds its own copy of the content: any write counts.
        const bool matStale = usesShadow(st) ? u->hostCopyObsolete() && !u->deviceCopyObsolete()
                                             : st->generation != st->matGeneration;
        if (!st->imported && matStale)
        {
            if (deviceOk && !st->poisoned)
                copyBackToMat(u, st);
            else
                CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): the device failed before its result was copied "
                             "back; the Mat keeps its previous content");
        }
        u->accelState = NULL;
        destroyBufferState(st);
        u->handle = 0;
        u->data = u->origdata;
        u->flags &= ~(UMatData::TEMP_UMAT | UMatData::HOST_COPY_OBSOLETE | UMatData::DEVICE_COPY_OBSOLETE);
        u->currAllocator = u->prevAllocator;
        u->prevAllocator = NULL;
        u->currAllocator->deallocate(u);
        b_->collectDeferred(false);
        return;
    }

    u->accelState = NULL;
    u->handle = 0;
    u->data = u->origdata = 0;
    if (st->imported)
        drainForRelease(st);  // the owner may reuse the memory as soon as we return
    if (st->imported || readyToFree(st))
        destroyBufferState(st);
    else
        b_->deferRelease(st);   // freed when its fences complete
    delete u;
    b_->collectDeferred(false);
}

void DeviceAllocator::map(UMatData* u, AccessFlag accessFlags) const
{
    // UMat::getMat() goes through getHostView(); a direct call cannot create a view safely
    CV_UNUSED(u); CV_UNUSED(accessFlags);
    CV_Error(Error::StsNotImplemented, "accel: use UMat::getMat() to access device data from the CPU");
}

void DeviceAllocator::unmap(UMatData* u) const
{
    if (!u || !u->accelState)
        return;
    endHostView(u);
    if (u->urefcount == 0 && u->refcount == 0)
        deallocate(u);  // same contract as MatAllocator::unmap
}

void DeviceAllocator::download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                               const size_t srcofs[], const size_t srcstep[], const size_t dststep[]) const
{
    BufferState* st = u->accelState;
    checkHostAccessAllowed(st, ACCESS_READ);
    IdleCpuGuard idle(st);
    std::unique_lock<std::mutex> lk(st->mutex);
    checkOwnHolds(st, false);
    BusyScope busy(st, lk, false);
    waitDeviceWork(st, lk, false);
    const size_t srcOffset = offsetND(srcofs, srcstep, dims);
    if (usesShadow(st))
    {
        if ((!u->hostCopyObsolete() && st->shadow) || u->deviceCopyObsolete())
        {
            // the host copy is current (or nothing was ever written): no DMA at all
            if (st->shadow)
                copyND((uchar*)dstptr, dststep, st->shadow + srcOffset, srcstep, dims, sz);
            return;
        }
        // one DMA straight into the destination
        StreamPtr s = b_->defaultStream(device_);
        cvAccelCopyRegion r;
        memset(&r, 0, sizeof(r));
        r.struct_size = (uint32_t)sizeof(r);
        r.dims = dims;
        for (int i = 0; i < dims; i++)
        {
            r.size[i] = sz[i];
            r.src.step[i] = srcstep[i];
            r.dst.step[i] = dststep[i];
        }
        r.src.alloc = &st->alloc;
        r.src.offset = srcOffset;
        r.dst.host = dstptr;
        FenceRef f;
        try
        {
            b_->check(b_->fn().copy(device_, s->handle, &r), "copy (device to host)", device_);
            f = recordFence(s);
            f.wait();   // the destination is user memory
        }
        catch (...)
        {
            if (f.empty())
                fenceAfterFailure(s).waitNoThrow();     // part of the copy may be queued, writing user memory
            throw;
        }
        Counters& c = counters();
        c.bytesD2H += totalND(sz, dims);
        c.transfers++;
        return;
    }
    if (st->hostMapped)
    {
        // a Mat view is open: read through its mapping. A second begin_cpu_access would
        // invalidate CPU caches under writes the view has not flushed yet.
        copyND((uchar*)dstptr, dststep, u->data + srcOffset, srcstep, dims, sz);
        return;
    }
    const void* p = beginCpuAccess(st, ACCESS_READ);
    copyND((uchar*)dstptr, dststep, (const uchar*)p + srcOffset, srcstep, dims, sz);
    endCpuAccess(st, ACCESS_READ);
}

void DeviceAllocator::upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                             const size_t dstofs[], const size_t dststep[], const size_t srcstep[]) const
{
    BufferState* st = u->accelState;
    if (st->alloc.desc.flags & CV_ACCEL_MEMF_READ_ONLY)
        CV_Error_(Error::StsError, ("accel(%s): this UMat wraps read-only memory", b_->name().c_str()));
    IdleCpuGuard idle(st);
    std::unique_lock<std::mutex> lk(st->mutex);
    checkOwnHolds(st, true);
    BusyScope busy(st, lk, true);
    if (st->hostMapped && u->refcount > 0)
        CV_Error(Error::StsError, "accel: cannot copy into a UMat while a Mat view of it is alive");
    if (st->hostMapped)
        endHostAccessLocked(st, u);
    const size_t dstOffset = offsetND(dstofs, dststep, dims);
    // a copy into all of the buffer may reuse a poisoned one, a copy into part of it may not
    const bool whole = dstOffset == 0 && totalND(sz, dims) == u->size;
    if (!whole)
        checkNotPoisoned(st);
    // memory without a CPU view (discrete, protected) is written by the device's copy engine;
    // data may enter protected memory, it just never comes out
    if (usesShadow(st) || hostAccessClass(st) == CV_ACCEL_HOST_ACCESS_NONE)
    {
        if (usesShadow(st) && u->deviceCopyObsolete() && !u->hostCopyObsolete())
        {
            // the host copy is current anyway: write there, the device copy follows lazily
            waitDeviceWork(st, lk, true, whole);
            copyND(st->shadow + dstOffset, dststep, (const uchar*)srcptr, srcstep, dims, sz);
            st->generation++;
            if (whole)
                clearPoison(st);
            return;
        }
        // one DMA straight into the device copy, after everyone using the old content
        StreamPtr s = b_->defaultStream(device_);
        if (!st->writeFence.empty())
            streamWait(s, st->writeFence);
        for (size_t i = 0; i < st->readFences.size(); i++)
            streamWait(s, st->readFences[i]);
        cvAccelCopyRegion r;
        memset(&r, 0, sizeof(r));
        r.struct_size = (uint32_t)sizeof(r);
        r.dims = dims;
        for (int i = 0; i < dims; i++)
        {
            r.size[i] = sz[i];
            r.src.step[i] = srcstep[i];
            r.dst.step[i] = dststep[i];
        }
        r.src.host = const_cast<void*>(srcptr);
        r.dst.alloc = &st->alloc;
        r.dst.offset = dstOffset;
        FenceRef f;
        try
        {
            b_->check(b_->fn().copy(device_, s->handle, &r), "copy (host to device)", device_);
            f = recordFence(s);
            lk.unlock();
            f.wait();   // the source is user memory that may be freed when we return
            lk.lock();
        }
        catch (...)
        {
            const std::string reason = currentExceptionMessage();
            if (f.empty())
                fenceAfterFailure(s).waitNoThrow();     // part of the copy may be queued, reading user memory
            if (!lk.owns_lock())
                lk.lock();
            // the device copy may be partly overwritten: invalid until written whole
            st->poisoned = true;
            st->poisonReason = reason;
            u->markHostCopyObsolete(true);
            u->markDeviceCopyObsolete(false);
            st->generation++;
            throw;
        }
        st->writeFence = f;
        st->readFences.clear();
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
        st->generation++;
        if (whole)
            clearPoison(st);
        Counters& c = counters();
        c.bytesH2D += totalND(sz, dims);
        c.transfers++;
        return;
    }
    waitDeviceWork(st, lk, true, whole);
    void* p = beginCpuAccess(st, ACCESS_WRITE);
    copyND((uchar*)p + dstOffset, dststep, (const uchar*)srcptr, srcstep, dims, sz);
    if (!endCpuAccess(st, ACCESS_WRITE))
        CV_Error_(Error::StsError, ("accel(%s): end_cpu_access failed after a copy into the UMat", b_->name().c_str()));
    u->markHostCopyObsolete(false);
    u->markDeviceCopyObsolete(false);
    st->generation++;
    if (whole)
        clearPoison(st);
}

void DeviceAllocator::copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
                           const size_t srcofs[], const size_t srcstep[],
                           const size_t dstofs[], const size_t dststep[], bool /*sync*/) const
{
    // two buffers of one device (the same buffer for a copy between its ROIs): a device copy
    CV_Assert(src->accelState && dst->accelState);
    if ((src->accelState->alloc.desc.flags & CV_ACCEL_MEMF_PROTECTED) &&
        !(dst->accelState->alloc.desc.flags & CV_ACCEL_MEMF_PROTECTED))
        CV_Error_(Error::StsError, ("accel(%s): protected content cannot be copied into ordinary memory",
                                    b_->name().c_str()));
    StreamPtr s = b_->defaultStream(device_);
    const size_t srcOffset = offsetND(srcofs, srcstep, dims);
    const size_t dstOffset = offsetND(dstofs, dststep, dims);
    const bool whole = dstOffset == 0 && totalND(sz, dims) == dst->size;
    DeviceAcquisition acq;
    acq.add(src, DEV_READ);
    acq.add(dst, DEV_WRITE | (whole ? DEV_DISCARD : 0));
    std::vector<FenceRef> waits;
    if (!acq.acquire(s, waits))
        CV_Error(Error::StsError, "accel: cannot copy between UMats while a conflicting Mat view of one of them is alive");
    for (size_t i = 0; i < waits.size(); i++)
        streamWait(s, waits[i]);
    cvAccelCopyRegion r;
    memset(&r, 0, sizeof(r));
    r.struct_size = (uint32_t)sizeof(r);
    r.dims = dims;
    for (int i = 0; i < dims; i++)
    {
        r.size[i] = sz[i];
        r.src.step[i] = srcstep[i];
        r.dst.step[i] = dststep[i];
    }
    r.src.alloc = &src->accelState->alloc;
    r.src.offset = srcOffset;
    r.dst.alloc = &dst->accelState->alloc;
    r.dst.offset = dstOffset;
    const cvAccelStatus status = b_->fn().copy(device_, s->handle, &r);
    if (status == CV_ACCEL_NOT_IMPLEMENTED || status == CV_ACCEL_OUT_OF_MEMORY)
        b_->check(status, "copy (device to device)", device_);     // nothing was queued: the acquisition is abandoned
    FenceRef done;
    try
    {
        b_->check(status, "copy (device to device)", device_);
        done = recordFence(s);
    }
    catch (...)
    {
        // part of the copy may be queued: the buffers wait for it, the destination is invalid
        acq.fail(fenceAfterFailure(s), currentExceptionMessage());
        throw;
    }
    acq.record(done);
    Counters& c = counters();
    c.bytesD2D += totalND(sz, dims);
    c.transfers++;
    if (src->accelState->imported || dst->accelState->imported)
        done.wait();    // user memory: the owner may touch it once we return (as in OpCall)
}

bool DeviceAllocator::getMemoryDesc(const UMatData* u, cvAccelMemoryDesc* desc) const
{
    if (!u || !u->accelState || !desc)
        return false;
    *desc = u->accelState->alloc.desc;
    return true;
}

}}} // namespace cv::accel::impl
