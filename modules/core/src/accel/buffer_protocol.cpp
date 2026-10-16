// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the coherence protocol for UMat data that lives on an accelerator.
//
// The state is kept per piece of physical memory (PhysicalBacking, see backing.cpp), whatever
// UMatData views it, so every rule below covers aliases, planes of one buffer and imports of
// one memory by several backends alike. Rules, in the order they are applied:
// 1. Ordering. The CPU never touches memory before device work that conflicts with the
//    access has completed: reading waits for the last device write, writing also waits for
//    device reads (write-after-read). Device work waits the same way, on the device when the
//    fence belongs to the same backend.
// 2. Exclusivity. While Mat views of memory may exist, the device may read it only if no view
//    was requested for writing, and may never write it. A conflicting operation falls back to
//    the CPU instead of racing with the views.
// 3. Coherence. Memory with a single physical copy that the CPU caches do not snoop
//    (NONCOHERENT) is opened with begin_cpu_access and closed with end_cpu_access: caches are
//    invalidated after the device wrote and cleaned before the device may touch the memory
//    again. The CPU opens all of a backing at once, for all of its views, and closes it when
//    the last view goes, so that no cache maintenance runs under another view's unflushed
//    writes. Discrete memory (COPY) is reached through a host copy that is only refreshed when
//    it is stale.
// 4. Minimal transfers. Read-only CPU access never writes back. A stale host copy is refreshed
//    only when it is read; a stale device copy only when the device reads it or writes part
//    of it. Uninitialized buffers are never transferred.
// 5. Lifetime. Released memory is freed only after its fences complete. Memory that belongs
//    to someone else is not handed back before the device is done with it.
// 6. Device-idle memory (managed memory without concurrent access) is touched by the CPU only
//    after the device drained, and no device work starts while the CPU holds it (IdleGate).

#include "../precomp.hpp"
#include "accel_impl.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/configuration.private.hpp>

#include <algorithm>
#include <functional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace cv { namespace accel { namespace impl {

static inline int hostAccessClass(const BufferState* v)
{
    return (int)v->imp->alloc.desc.host_access;
}

static inline bool usesShadow(const BufferState* v)
{
    int c = hostAccessClass(v);
    return c == CV_ACCEL_HOST_ACCESS_COPY || c == CV_ACCEL_HOST_ACCESS_UNCACHED;
}

//! memory the CPU addresses directly (opened for all views of the backing at once)
static inline bool directClass(int c)
{
    return c == CV_ACCEL_HOST_ACCESS_COHERENT || c == CV_ACCEL_HOST_ACCESS_NONCOHERENT ||
           c == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE;
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
inline bool deviceUseInFlight(const PhysicalBacking* b, bool write)
{
    return b->pendingWriters != 0 || (write && b->pendingReaders != 0);
}

/** Keeps PhysicalBacking::busy set for the lifetime of the object, and wakes waiters at the end.
    Busy is set only once no other transition runs and no conflicting device operation sits
    between acquire and record: a transition never holds busy while it waits for one, because
    that operation may itself be waiting for busy (a call holding another backing). */
struct BusyScope
{
    BusyScope(PhysicalBacking* b_, std::unique_lock<std::mutex>& lk_, bool write) : b(b_), lk(lk_)
    {
        b->cond.wait(lk, [this, write]{ return !b->busy && !deviceUseInFlight(b, write); });
        b->busy = true;
    }
    ~BusyScope()
    {
        if (!lk.owns_lock())
            lk.lock();
        b->busy = false;
        b->cond.notify_all();
    }
    PhysicalBacking* b;
    std::unique_lock<std::mutex>& lk;
};

int accessOfMode(int mode)
{
    return ((mode & DEV_READ) ? ACCESS_READ : 0) | ((mode & DEV_WRITE) ? ACCESS_WRITE : 0);
}

void checkNotPoisoned(const BufferState* v)
{
    if (const PoisonRange* p = findPoison(v->backing, v->start, v->size))
        CV_Error_(Error::StsError, ("accel(%s): the device operation that produced this UMat failed: %s",
                                    v->backend()->name().c_str(), p->reason.c_str()));
}

//! the last device write is the one that covers @p v's bytes and just completed with @p failure
void poisonLastWriteLocked(PhysicalBacking* b, const std::string& failure)
{
    for (size_t i = 0; i < b->writeRanges.size(); i++)
        addPoison(b, b->writeRanges[i].first, b->writeRanges[i].second - b->writeRanges[i].first, failure);
    b->writeRanges.clear();
}

/** Memory with implicit fences of other drivers (CV_ACCEL_MEMF_EXTERNAL_SYNC, e.g. a camera's
    dma-buf): device work on it waits for their fences, and they wait for the work's fence. Core
    imports such memory only through backends that bracket it (backend rule 5). */
void beginExternal(const BufferState* v, const StreamPtr& s, int access)
{
    Backend* b = v->backend();
    if (!(v->backing->flags & CV_ACCEL_MEMF_EXTERNAL_SYNC))
        return;
    if (!b->fn().begin_device_access || !b->fn().end_device_access)
        CV_Error_(Error::StsNotImplemented, ("accel(%s): this memory carries implicit fences of other drivers that the "
                                             "backend cannot handle", b->name().c_str()));
    b->check(b->fn().begin_device_access(&v->imp->alloc, s->handle, (uint32_t)access), "begin_device_access", v->device());
}

/** Never throws: runs when the work is already queued. When the work's fence cannot be handed
    to the other drivers, they would not wait for it: the work is waited for on the CPU instead,
    without limit (the memory is theirs as much as ours), before the call that queued it returns. */
void endExternal(const BufferState* v, const FenceRef& f, int access) CV_NOEXCEPT
{
    Backend* b = v->backend();
    if (!(v->backing->flags & CV_ACCEL_MEMF_EXTERNAL_SYNC) || f.empty())
        return;
    cvAccelStatus st = CV_ACCEL_INTERNAL_ERROR;
    if (!f.isUnknown() && b->fn().end_device_access)
        st = b->fn().end_device_access(&v->imp->alloc, f.get(), (uint32_t)access);
    if (st == CV_ACCEL_OK)
        return;
    try
    {
        CV_LOG_WARNING(NULL, "accel(" << b->name() << "): the fence of this work could not be attached to memory "
                       "shared with other drivers (" << b->fn().get_last_error() << "); waiting for the work instead");
    }
    catch (...) {}
    f.waitNoThrow(false);
}

/** Waits on the CPU for device work that conflicts with a CPU access through @p v. Called with
    b->busy set, so no new device work on the backing can start meanwhile; the mutex is released
    during the waits so that operations in flight can record their fences. When the last write
    failed, the bytes it covered are poisoned; @p v's bytes throw then, unless the caller is
    about to overwrite all of them (@p overwriteAll). */
void waitDeviceWork(PhysicalBacking* b, std::unique_lock<std::mutex>& lk, bool write, const BufferState* v,
                    bool overwriteAll = false)
{
    CV_DbgAssert(b->busy && !deviceUseInFlight(b, write));  // see BusyScope
    FenceRef lastWrite = b->writeFence;
    std::vector<FenceRef> reads;
    if (write)
        reads = b->readFences;
    if (!lastWrite.empty() || !reads.empty())
    {
        lk.unlock();
        std::string failure;
        try
        {
            lastWrite.wait();
        }
        catch (const cv::Exception& e)
        {
            // async errors surface here: the bytes that work wrote are not what it should have written
            failure = e.err;
        }
        std::exception_ptr readError;
        for (size_t i = 0; i < reads.size(); i++)
        {
            try
            {
                reads[i].wait();
            }
            catch (const cv::Exception& e)
            {
                // a failed read changed nothing, but a lost device or an expired wait is an error
                if (reads[i].backend()->isLost(reads[i].device()))
                    readError = std::current_exception();
            }
        }
        lk.lock();
        // busy keeps writers away, so b->writeFence is still lastWrite. Work that may still run
        // (a wait that timed out, an unknown fence) keeps its fence: what it touches stays guarded.
        bool running = false;
        if (!lastWrite.empty())
        {
            const FenceRef::Poll p = lastWrite.poll(false);
            if (p == FenceRef::PENDING)
                running = true;
            else
            {
                if (p == FenceRef::FAILED || !failure.empty())
                    poisonLastWriteLocked(b, failure.empty() ? std::string("the device work failed") : failure);
                b->writeFence.reset();  // completed: carries no information any more
            }
        }
        if (write)
        {
            bool readsDone = true;
            for (size_t i = 0; i < reads.size() && readsDone; i++)
                readsDone = reads[i].poll(false) != FenceRef::PENDING;
            if (readsDone)
                b->readFences.clear();
        }
        if (!failure.empty() && (running || lastWrite.backend()->isLost(lastWrite.device())))
            CV_Error(Error::StsError, failure);     // the device is gone or late: nothing more to do with it
        if (readError)
            std::rethrow_exception(readError);
    }
    if (!overwriteAll)
        checkNotPoisoned(v);
}

//////////////////////////////////// pages a write can fault on ////////////////////////////////////

size_t pageRounded(size_t size)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize ? si.dwPageSize : 4096;
#else
    const long ps = sysconf(_SC_PAGESIZE);
    const size_t page = ps > 0 ? (size_t)ps : 4096;
#endif
    return (size + page - 1) / page * page;
}

uchar* allocPages(size_t size)
{
#ifdef _WIN32
    return (uchar*)VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : (uchar*)p;
#endif
}

void freePages(uchar* p, size_t size)
{
#ifdef _WIN32
    CV_UNUSED(size);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, size);
#endif
}

//! tests: the page protection call after this many successful ones fails (detail::injectPageProtectionFailure)
std::atomic<int>& pageFailureCountdown()
{
    static std::atomic<int>* c = new std::atomic<int>(-1);
    return *c;
}

bool injectedPageFailure()
{
    int n = pageFailureCountdown().load();
    while (n >= 0)
    {
        if (pageFailureCountdown().compare_exchange_weak(n, n - 1))
            return n == 0;
    }
    return false;
}

bool protectPages(uchar* p, size_t size, bool writable)
{
    if (injectedPageFailure())
        return false;
#ifdef _WIN32
    DWORD old = 0;
    return VirtualProtect(p, size, writable ? PAGE_READWRITE : PAGE_READONLY, &old) != 0;
#else
    return mprotect(p, size, writable ? (PROT_READ | PROT_WRITE) : PROT_READ) == 0;
#endif
}

/** A copy of @p size bytes at @p src in pages mapped read-only, so that a write through a Mat
    view of it faults. Fails closed: a copy that cannot be protected is never handed out. */
uchar* protectedCopy(const uchar* src, size_t size, size_t& pages)
{
    pages = pageRounded(size ? size : 1);
    uchar* p = allocPages(pages);
    if (!p)
        CV_Error(Error::StsNoMem, "accel: no pages for a read-only CPU view");
    memcpy(p, src, size);
    if (!protectPages(p, pages, false))
    {
        freePages(p, pages);
        CV_Error(Error::StsError, "accel: a read-only CPU view cannot be protected against writes; refused");
    }
    return p;
}

//////////////////////////////////// validation mode ////////////////////////////////////

/** OPENCV_ACCEL_VALIDATE: every Mat view opened with ACCESS_READ is mapped read-only (a copy in
    protected pages where the memory is not), so that a write through one faults at once. */
std::atomic<bool>& validating()
{
    static std::atomic<bool> v(utils::getConfigurationParameterBool("OPENCV_ACCEL_VALIDATE", false));
    return v;
}

void ensureShadow(BufferState* v)
{
    if (v->shadow)
        return;
    const size_t size = (size_t)v->size;
    if (v->readOnly)
    {
        // Read-only memory: the host copy is mapped read-only, so a write through a Mat view
        // faults instead of corrupting it. Only downloads lift the protection, for their
        // duration. No protection, no host copy: the CPU access is refused.
        const size_t pages = pageRounded(size ? size : 1);
        uchar* p = allocPages(pages);
        if (!p)
            CV_Error(Error::StsNoMem, "accel: no pages for the host copy of read-only memory");
        if (!protectPages(p, pages, false))
        {
            freePages(p, pages);
            CV_Error(Error::StsError, "accel: the host copy of read-only memory cannot be protected against writes; "
                                      "CPU access refused");
        }
        v->shadow = p;
        v->shadowPages = pages;
        v->shadowKind = SHADOW_PAGES;
        return;
    }
    Backend* b = v->backend();
    const int device = v->device();
    if ((b->deviceInfo(device).caps & CV_ACCEL_CAP_HOST_PINNED) && b->pinnedShadowAllowed(device))
    {
        cvAccelAllocation a;
        memset(&a, 0, sizeof(a));
        if (b->takePinned(device, size, a) ||
            b->fn().allocate(device, size, CV_ACCEL_MEM_HOST_PINNED, 0, &a) == CV_ACCEL_OK)
        {
            if (a.desc.host_access == CV_ACCEL_HOST_ACCESS_COHERENT && a.desc.handle_kind == CV_ACCEL_HANDLE_HOST_POINTER)
            {
                a.desc.backend_id = (uint32_t)b->id();
                a.desc.device_index = (uint32_t)device;
                v->shadowAlloc = a;
                v->shadow = (uchar*)(uintptr_t)a.handle;
                v->shadowKind = SHADOW_BACKEND;
                return;
            }
            // The CPU works in the host copy: memory it reaches uncached (pinned memory on
            // Jetson before Xavier), write-combined or without coherence would slow every
            // access down or need cache maintenance. Pageable memory from now on.
            CV_LOG_INFO(NULL, "accel(" << b->name() << "): pinned memory of device " << device
                        << " is not CPU-cached; host copies use pageable memory");
            b->disallowPinnedShadow(device);
            if (b->fn().deallocate(&a) != CV_ACCEL_OK)
                CV_LOG_ERROR(NULL, "accel(" << b->name() << "): failed to free pinned memory: " << b->fn().get_last_error());
        }
        else
        {
            // pinned memory is a limited resource: fall back to pageable memory
            CV_LOG_DEBUG(NULL, "accel(" << b->name() << "): no pinned memory for a host copy, using pageable memory");
        }
    }
    v->shadow = (uchar*)fastMalloc(size);
    v->shadowKind = SHADOW_FASTMALLOC;
}

void fillRegion1D(cvAccelCopyRegion& r, uint64_t bytes)
{
    memset(&r, 0, sizeof(r));
    r.struct_size = (uint32_t)sizeof(r);
    r.dims = 1;
    r.size[0] = bytes;
}

//! forgets the page-protected host copy of a read-only view whose protection could not be restored
void dropProtectedShadow(BufferState* v)
{
    freePages(v->shadow, v->shadowPages);   // unmapped: a stale pointer into it faults
    v->shadow = 0;
    v->shadowPages = 0;
    v->shadowKind = SHADOW_NONE;
    if (v->u)
        v->u->markHostCopyObsolete(true);
}

//! whole-view device -> host copy into the host copy; waits for completion
void downloadWhole(BufferState* v)
{
    Backend* b = v->backend();
    StreamPtr s = b->defaultStream(v->device());
    cvAccelCopyRegion r;
    fillRegion1D(r, v->size);
    r.src.alloc = &v->imp->alloc;
    r.src.offset = v->start;
    r.dst.host = v->shadow;
    if (v->shadowKind == SHADOW_BACKEND)
        r.flags |= CV_ACCEL_COPY_HOST_PINNED;
    FenceRef f;
    // read-only memory: its protected host copy takes the download, writable for its duration
    const bool pages = v->shadowKind == SHADOW_PAGES;
    if (pages && !protectPages(v->shadow, v->shadowPages, true))
        CV_Error(Error::StsError, "accel: cannot unprotect the host copy of read-only memory for a download");
    try
    {
        beginExternal(v, s, ACCESS_READ);
        b->check(b->fn().copy(v->device(), s->handle, &r), "copy (device to host)", v->device());
        f = recordFence(s);
        endExternal(v, f, ACCESS_READ);
        f.wait(false);  // the host copy may be a Mat's memory, and this fence is not kept
    }
    catch (...)
    {
        if (f.empty())
            fenceAfterFailure(s).waitNoThrow(false);    // part of the copy may be queued, writing the host copy
        if (pages && !protectPages(v->shadow, v->shadowPages, false))
            dropProtectedShadow(v);
        throw;
    }
    if (pages && !protectPages(v->shadow, v->shadowPages, false))
    {
        // fails closed: pages that stay writable are never shown; no Mat views them yet
        dropProtectedShadow(v);
        CV_Error(Error::StsError, "accel: the host copy of read-only memory cannot be protected again after a "
                                  "download; CPU access refused");
    }
    Counters& c = counters();
    c.bytesD2H += v->size;
    c.transfers++;
}

//! whole-view host -> device copy from the host copy, enqueued on @p s
void uploadWhole(BufferState* v, const StreamPtr& s)
{
    Backend* b = v->backend();
    cvAccelCopyRegion r;
    fillRegion1D(r, v->size);
    r.src.host = v->shadow;
    r.dst.alloc = &v->imp->alloc;
    r.dst.offset = v->start;
    if (v->shadowKind == SHADOW_BACKEND)
        r.flags |= CV_ACCEL_COPY_HOST_PINNED;
    b->check(b->fn().copy(v->device(), s->handle, &r), "copy (host to device)", v->device());
    Counters& c = counters();
    c.bytesH2D += v->size;
    c.transfers++;
}

//! the last device write of the backing is @p f, which covers @p v's bytes
void recordWriteLocked(PhysicalBacking* b, const BufferState* v, const FenceRef& f)
{
    b->writeFence = f;
    b->readFences.clear();
    b->writeRanges.clear();
    try
    {
        b->writeRanges.push_back(std::make_pair(v->start, v->start + v->size));
    }
    catch (...)
    {
        b->writeRanges.clear();     // no memory: a failure of this write would go unnoticed by readers of v
        b->writeRanges.reserve(0);
    }
}

/** Makes the device copy current from the host copy. The upload overwrites the device copy,
    so it waits for every device user of the old content first. */
void uploadShadowLocked(BufferState* v, UMatData* u, const StreamPtr& s, bool waitNow)
{
    PhysicalBacking* b = v->backing;
    streamWait(s, b->writeFence);
    for (size_t i = 0; i < b->readFences.size(); i++)
        streamWait(s, b->readFences[i]);
    FenceRef f;
    try
    {
        beginExternal(v, s, ACCESS_WRITE);
        uploadWhole(v, s);
        f = recordFence(s);
        endExternal(v, f, ACCESS_WRITE);
        if (waitNow)
            f.wait(false);  // reads the host copy, which may be a Mat's memory
    }
    catch (...)
    {
        // part of the copy may be queued: later users wait for it. The device copy stays
        // obsolete, so its next use uploads again.
        if (f.empty())
        {
            FenceRef after = fenceAfterFailure(s);
            if (!after.empty())
                recordWriteLocked(b, v, after);
        }
        throw;
    }
    recordWriteLocked(b, v, f);
    u->markDeviceCopyObsolete(false);
}

/** Announces CPU access to DEVICE_IDLE memory on the device's gate and waits until no
    submission is in progress. Must be created before the backing is locked (see IdleGate).
    One-shot accesses release the count when the guard goes away; a mapping keeps it (commit)
    until the backing's CPU access ends. Does nothing for other memory. */
class IdleCpuGuard
{
public:
    explicit IdleCpuGuard(const BufferState* v) : gate_(0), committed_(false)
    {
        if (hostAccessClass(v) != CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
            return;
        gate_ = &v->backend()->idleGate(v->device());
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
    IdleGate* gate() const { return gate_; }
    //! the backing's CPU access holds the count from now on (released by closeCpu)
    void commit() { committed_ = true; }
private:
    IdleCpuGuard(const IdleCpuGuard&);
    IdleCpuGuard& operator=(const IdleCpuGuard&);
    IdleGate* gate_;
    bool committed_;
};

//////////////////////////////////// CPU access to the whole backing ////////////////////////////////////

//! a CPU or device access may start only once every failed end_cpu_access went through (backend rule 4)
void requireCpuAccessEnded(PhysicalBacking* b)
{
    if (!retryFailedCpuEndsLocked(b))
        CV_Error(Error::StsError, "accel: an earlier CPU access to this memory could not be ended (end_cpu_access "
                                  "keeps failing); the memory cannot be used until it is");
}

//! the import whose mapping the CPU uses: one it may write when there is one (all of them begin and end)
const Import* cpuImport(const PhysicalBacking* b)
{
    const Import* first = 0;
    for (size_t i = 0; i < b->imports.size(); i++)
    {
        const Import* imp = b->imports[i];
        if (!directClass((int)imp->alloc.desc.host_access) || imp->modifier != 0)
            continue;
        if (!(imp->alloc.desc.flags & CV_ACCEL_MEMF_CPU_MAPPING_READ_ONLY))
            return imp;
        if (!first)
            first = imp;
    }
    return first;
}

//! access of every level the backing has open
int openCpuAccess(const PhysicalBacking* b)
{
    int access = 0;
    for (size_t i = 0; i < b->cpuLevels.size(); i++)
        access |= b->cpuLevels[i];
    return access;
}

//! ends levels [from, end) on every import with a CPU view, innermost first; false when one failed
bool closeCpuLevels(PhysicalBacking* b, size_t from) CV_NOEXCEPT
{
    bool ok = true;
    while (b->cpuLevels.size() > from)
    {
        const int access = b->cpuLevels.back();
        for (size_t i = b->imports.size(); i-- > 0; )
        {
            Import* imp = b->imports[i];
            if (!directClass((int)imp->alloc.desc.host_access) || imp->modifier != 0)
                continue;
            const cvAccelStatus st = imp->backend->fn().end_cpu_access(&imp->alloc, 0, imp->alloc.size, (uint32_t)access);
            if (st == CV_ACCEL_OK)
                continue;
            ok = false;
            if (st == CV_ACCEL_DEVICE_LOST)
            {
                imp->backend->markStopped(imp->device);
                continue;   // a stopped device touches nothing: there is no CPU access to keep open
            }
            // the CPU access counts as open until a retry succeeds (backend rule 4)
            try
            {
                b->failedCpuEnds.push_back(std::make_pair(imp, access));
                CV_LOG_ERROR(NULL, "accel(" << imp->backend->name() << "): end_cpu_access failed: "
                             << imp->backend->fn().get_last_error() << "; no device work uses this memory until it succeeds");
            }
            catch (...) {}
        }
        b->cpuLevels.pop_back();
    }
    if (b->cpuLevels.empty())
        b->cpuBase = 0;
    return ok;
}

/** Opens the backing for CPU access with @p access, or widens what is open: begin_cpu_access on
    every import with a CPU view, nested (backend rule 4), so that the cache maintenance of one
    import covers what any device of the backing needs. Returns the CPU address of byte 0. A write
    is always opened together with a read: a partial write of a cache line must not merge with
    stale bytes. For DEVICE_IDLE memory the caller holds an IdleCpuGuard, and the first level
    drains the device first. Called with the backing locked and busy. */
uchar* openCpuLevel(PhysicalBacking* b, int access)
{
    const int want = (access & ACCESS_WRITE) ? ACCESS_RW : ACCESS_READ;
    if (!b->cpuLevels.empty() && (openCpuAccess(b) & want) == want)
        return b->cpuBase;
    if (b->cpuLevels.empty())
    {
        requireCpuAccessEnded(b);
        for (size_t i = 0; i < b->imports.size(); i++)
        {
            Import* imp = b->imports[i];
            if (imp->alloc.desc.host_access == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
                imp->backend->check(imp->backend->synchronize(imp->device), "device_synchronize", imp->device);
        }
    }
    b->cpuLevels.reserve(b->cpuLevels.size() + 1);
    // the CPU reaches the memory through a mapping it may write, when an import has one
    const Import* through = cpuImport(b);
    uchar* base = 0;
    std::vector<Import*> begun;
    begun.reserve(b->imports.size());
    for (size_t i = 0; i < b->imports.size(); i++)
    {
        Import* imp = b->imports[i];
        if (!directClass((int)imp->alloc.desc.host_access) || imp->modifier != 0)
            continue;
        void* p = 0;
        cvAccelStatus st = imp->backend->fn().begin_cpu_access(&imp->alloc, 0, imp->alloc.size, (uint32_t)want, &p);
        if (st == CV_ACCEL_OK && !p)
            st = CV_ACCEL_INTERNAL_ERROR;
        // a nested level must return the address the outer one returned (backend rule 4)
        const bool moved = st == CV_ACCEL_OK && imp == through && !b->cpuLevels.empty() && (uchar*)p != b->cpuBase;
        if (st != CV_ACCEL_OK || moved)
        {
            if (st == CV_ACCEL_OK)
                begun.push_back(imp);
            // undo this level on the imports begun so far, in reverse order
            for (size_t j = begun.size(); j-- > 0; )
            {
                if (begun[j]->backend->fn().end_cpu_access(&begun[j]->alloc, 0, begun[j]->alloc.size, (uint32_t)want) != CV_ACCEL_OK)
                    CV_LOG_ERROR(NULL, "accel(" << begun[j]->backend->name() << "): end_cpu_access failed: "
                                 << begun[j]->backend->fn().get_last_error());
            }
            if (moved)
                CV_Error_(Error::StsInternal, ("accel(%s): a nested begin_cpu_access returned another address",
                                               imp->backend->name().c_str()));
            imp->backend->check(st, "begin_cpu_access", imp->device);
        }
        if (imp == through)
            base = (uchar*)p;
        begun.push_back(imp);
    }
    if (!base)
        CV_Error(Error::StsInternal, "accel: this memory has no import with a CPU view");
    b->cpuLevels.push_back(want);   // capacity is reserved
    b->cpuBase = base;
    return base;
}

/** One-shot CPU access to the backing (a download or upload through the allocator, a copy back
    to a Mat): opens a level when what is open does not cover @p access, and closes that level
    (and only that one) at the end. */
class CpuScope
{
public:
    CpuScope(PhysicalBacking* b, int access) : b_(b), from_(b->cpuLevels.size())
    {
        base_ = openCpuLevel(b, access);
    }
    ~CpuScope() { close(); }
    uchar* base() const { return base_; }
    //! false when an end_cpu_access failed
    bool close() CV_NOEXCEPT
    {
        if (!b_)
            return true;
        const bool ok = closeCpuLevels(b_, from_);
        b_ = 0;
        return ok;
    }
private:
    CpuScope(const CpuScope&);
    CpuScope& operator=(const CpuScope&);
    PhysicalBacking* b_;
    size_t from_;
    uchar* base_;
};

//! a Mat mapping of @p b conflicts with device work through @p v (that writes, with @p write)
bool cpuConflictLocked(const PhysicalBacking* b, const BufferState* v, bool write)
{
    // memory the CPU addresses directly is open as a whole, and unflushed CPU writes may sit in
    // any of its cache lines until it closes
    if (!b->cpuLevels.empty() && (write || (openCpuAccess(b) & ACCESS_WRITE)))
        return true;
    // host copies: only the device's copy of the same bytes changes under them
    for (size_t i = 0; i < b->views.size(); i++)
    {
        const BufferState* w = b->views[i];
        if (w->hostMapped && usesShadow(w) && rangesOverlap(w->start, w->size, v->start, v->size) &&
            (write || (w->hostAccess & ACCESS_WRITE)))
            return true;
    }
    return false;
}

void beginHostAccessLocked(BufferState* v, UMatData* u, int access)
{
    // the bytes the CPU sees: the host copy, or the memory itself
    uchar* content = 0;
    bool protectedInPlace = false;
    if (usesShadow(v))
    {
        const bool fresh = !v->shadow;  // a new host copy holds nothing yet, whatever the flags say
        ensureShadow(v);
        if ((u->hostCopyObsolete() || fresh) && !u->deviceCopyObsolete())
        {
            downloadWhole(v);
            u->markHostCopyObsolete(false);
        }
        else if (isUninitialized(u) && (access & ACCESS_WRITE))
        {
            // nothing to fetch: from now on the host copy is the content
            u->markHostCopyObsolete(false);
        }
        content = v->shadow;
        protectedInPlace = v->shadowKind == SHADOW_PAGES;
    }
    else
    {
        PhysicalBacking* b = v->backing;
        content = openCpuLevel(b, access) + v->start;
        const Import* through = cpuImport(b);
        protectedInPlace = through && (through->alloc.desc.flags & CV_ACCEL_MEMF_CPU_MAPPING_READ_ONLY);
        u->markHostCopyObsolete(false);
    }
    // A write through a view that may not write must fault: read-only memory always, any view
    // opened for reading under OPENCV_ACCEL_VALIDATE. Where the bytes are not mapped read-only
    // already, the view shows a copy in pages that are.
    const bool mustFault = !(access & ACCESS_WRITE) && (v->readOnly || validating().load());
    if (mustFault && !protectedInPlace)
        u->data = v->cpuCopy = protectedCopy(content, (size_t)v->size, v->cpuCopyPages);
    else
        u->data = content;
    v->cpuCopyWriteBack = false;
}

void upgradeHostAccessLocked(BufferState* v, UMatData* u)
{
    if (!usesShadow(v))
    {
        // a nested level for the write, ended together with the outer one when the backing closes
        uchar* p = openCpuLevel(v->backing, ACCESS_RW) + v->start;
        CV_Assert(v->cpuCopy || p == u->data);
    }
    else if (isUninitialized(u))
        u->markHostCopyObsolete(false);
    if (v->cpuCopy)
    {
        // a read view under OPENCV_ACCEL_VALIDATE turns into a write view: its copy takes the
        // writes, and goes back to the memory when the mapping ends
        if (!protectPages(v->cpuCopy, v->cpuCopyPages, true))
            CV_Error(Error::StsError, "accel: cannot make the CPU view writable");
        v->cpuCopyWriteBack = true;
    }
}

//! the backing's last Mat mapping ended: ends CPU access to all of it; never throws
void closeCpuLocked(PhysicalBacking* b) CV_NOEXCEPT
{
    const bool wrote = (openCpuAccess(b) & ACCESS_WRITE) != 0;
    if (!closeCpuLevels(b, 0) && wrote)
    {
        // the device may see part of the CPU's writes: invalid until written whole
        try
        {
            addPoison(b, 0, b->size, "end_cpu_access failed after a CPU write");
        }
        catch (...) {}
    }
    // device-idle memory: the CPU access is over only once every end succeeded
    if (b->idleGateHeld && b->failedCpuEnds.empty())
    {
        b->idleGateHeld->cpuMappings--;
        b->idleGateHeld = 0;
    }
}

} // namespace

bool retryFailedCpuEndsLocked(PhysicalBacking* b) CV_NOEXCEPT
{
    for (size_t i = 0; i < b->failedCpuEnds.size(); )
    {
        Import* imp = b->failedCpuEnds[i].first;
        const cvAccelStatus st = imp->backend->fn().end_cpu_access(&imp->alloc, 0, imp->alloc.size,
                                                                   (uint32_t)b->failedCpuEnds[i].second);
        if (st == CV_ACCEL_OK || st == CV_ACCEL_DEVICE_LOST)
        {
            if (st == CV_ACCEL_DEVICE_LOST)
                imp->backend->markStopped(imp->device);
            b->failedCpuEnds.erase(b->failedCpuEnds.begin() + i);
            continue;
        }
        i++;
    }
    if (!b->failedCpuEnds.empty())
        return false;
    if (b->idleGateHeld && b->cpuLevels.empty())
    {
        b->idleGateHeld->cpuMappings--;     // the CPU access that held the device gate is over
        b->idleGateHeld = 0;
    }
    return true;
}

namespace {

//! ends the Mat mapping of a view; never throws (runs from Mat destructors)
void endHostAccessLocked(BufferState* v, UMatData* u)
{
    PhysicalBacking* b = v->backing;
    const int access = v->hostAccess;
    if (v->cpuCopy)
    {
        if (v->cpuCopyWriteBack)
        {
            // the writes go where the content lives: the host copy, or the memory (still open
            // with a write level)
            uchar* content = usesShadow(v) ? v->shadow : b->cpuBase + v->start;
            if (content)
                memcpy(content, v->cpuCopy, (size_t)v->size);
        }
        freePages(v->cpuCopy, v->cpuCopyPages);
        v->cpuCopy = 0;
        v->cpuCopyPages = 0;
        v->cpuCopyWriteBack = false;
    }
    if (access & ACCESS_WRITE)
    {
        b->generation++;
        if (usesShadow(v))
        {
            // the device copy is refreshed lazily, when the device needs it
            u->markHostCopyObsolete(false);
            u->markDeviceCopyObsolete(true);
        }
        else
            u->markDeviceCopyObsolete(false);
    }
    v->hostMapped = false;
    v->hostAccess = 0;
    u->mapcount = 0;
    CV_DbgAssert(b->cpuViews > 0);
    if (--b->cpuViews == 0)
        closeCpuLocked(b);
}

//! ends mappings whose last Mat is gone but whose unmap has not run yet (backing mutex held)
void endOrphanMappingsLocked(PhysicalBacking* b)
{
    for (size_t i = 0; i < b->views.size(); i++)
    {
        BufferState* w = b->views[i];
        if (w->hostMapped && w->u->refcount == 0)
            endHostAccessLocked(w, w->u);
    }
}

void checkHostAccessAllowed(const BufferState* v, int access)
{
    const cvAccelMemoryDesc& d = v->imp->alloc.desc;
    if (d.host_access == CV_ACCEL_HOST_ACCESS_NONE || (v->backing->flags & CV_ACCEL_MEMF_PROTECTED) || d.layout_modifier != 0)
        CV_Error_(Error::StsNotImplemented,
                  ("accel(%s): this UMat is in memory the CPU cannot access (protected content or a "
                   "non-linear layout); use an operation that the device implements",
                   v->backend()->name().c_str()));
    if ((access & ACCESS_WRITE) && v->readOnly)
        CV_Error_(Error::StsError, ("accel(%s): this UMat wraps read-only memory", v->backend()->name().c_str()));
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

//! bytes from the first to the last byte touched by a strided region (sz[dims-1] in bytes)
uint64_t extentND(const size_t* sz, const size_t* step, int dims)
{
    uint64_t e = dims > 0 ? sz[dims - 1] : 0;
    for (int i = 0; i < dims - 1; i++)
    {
        if (sz[i] == 0)
            return 0;
        e += (uint64_t)(sz[i] - 1) * step[i];
    }
    return e;
}

//! a DeviceAccess open on this thread would make the caller wait for itself: report it instead
void checkOwnHolds(const PhysicalBacking* b, bool write)
{
    const std::thread::id self = std::this_thread::get_id();
    for (size_t i = 0; i < b->longHolds.size(); i++)
    {
        const PhysicalBacking::Hold& h = b->longHolds[i];
        if (h.thread == self && (write || (h.mode & DEV_WRITE)))
            CV_Error(Error::StsError, "accel: a DeviceAccess opened on this thread still holds this memory; "
                                      "destroy it before using the memory elsewhere");
    }
}

} // namespace

//////////////////////////////////// host access ////////////////////////////////////

Mat getHostView(const UMat& m, AccessFlag accessFlags)
{
    UMatData* u = m.u;
    BufferState* v = u->accelState;
    PhysicalBacking* b = v->backing;
    int access = accessFlags & ACCESS_RW;
    if (access == 0)
        access = ACCESS_READ;
    checkHostAccessAllowed(v, access);

    {
        IdleCpuGuard idle(v);   // before the lock: see IdleGate
        std::unique_lock<std::mutex> lk(b->mutex);
        const bool write = (access & ACCESS_WRITE) != 0;
        checkOwnHolds(b, write);
        b->cond.wait(lk, [b, write]{ return !b->busy && !deviceUseInFlight(b, write); });
        const bool first = !v->hostMapped;
        const bool upgrade = !first && write && !(v->hostAccess & ACCESS_WRITE);
        if (first || upgrade)
        {
            BusyScope busy(b, lk, write);
            if (b->cpuViews == 0)
                requireCpuAccessEnded(b);
            waitDeviceWork(b, lk, write, v);
            const bool opens = first && b->cpuViews == 0;   // the backing's CPU access starts now
            if (first)
            {
                try
                {
                    beginHostAccessLocked(v, u, access);
                }
                catch (...)
                {
                    // the levels this view opened go with it: the backing's other views keep theirs
                    if (b->cpuViews == 0)
                        closeCpuLocked(b);
                    throw;
                }
            }
            else
                upgradeHostAccessLocked(v, u);
            if (first)
            {
                v->hostMapped = true;
                b->cpuViews++;
                u->mapcount = 1;    // UMatData::~UMatData unmaps chained views (getMat().getUMat()) only then
                if (opens && idle.gate())
                {
                    b->idleGateHeld = idle.gate();  // the backing's CPU access holds the gate until it ends
                    idle.commit();
                }
            }
        }
        v->hostAccess |= access;
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
    BufferState* v = u->accelState;
    PhysicalBacking* b = v->backing;
    std::unique_lock<std::mutex> lk(b->mutex);
    b->cond.wait(lk, [b]{ return !b->busy; });
    // a new view may have been created after the last one went away
    if (!v->hostMapped || u->refcount != 0)
        return;
    endHostAccessLocked(v, u);
    b->cond.notify_all();
}

void* deviceHandleBlocking(UMatData* u, AccessFlag access)
{
    BufferState* v = u->accelState;
    PhysicalBacking* b = v->backing;
    std::unique_lock<std::mutex> lk(b->mutex);
    checkOwnHolds(b, true);
    BusyScope busy(b, lk, true);
    endOrphanMappingsLocked(b);
    requireCpuAccessEnded(b);
    // the caller's work may go to any stream at any time: no Mat view of the memory may be open
    if (v->hostMapped || cpuConflictLocked(b, v, true))
        CV_Error(Error::StsError, "UMat::handle(): a Mat view of this memory is alive");
    waitDeviceWork(b, lk, true, v);
    if (usesShadow(v) && u->deviceCopyObsolete() && !u->hostCopyObsolete())
        uploadShadowLocked(v, u, v->backend()->defaultStream(v->device()), true);
    if (access & ACCESS_WRITE)
    {
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
        b->generation++;
    }
    const uint32_t kind = v->imp->alloc.desc.handle_kind;
    if (kind == CV_ACCEL_HANDLE_DEVICE_POINTER || kind == CV_ACCEL_HANDLE_HOST_POINTER)
        return (void*)(uintptr_t)(v->imp->alloc.handle + v->start);
    if (v->start != 0)
        CV_Error(Error::StsNotImplemented, "UMat::handle(): this memory has no flat address and the UMat starts inside "
                                           "it; use cv::accel::DeviceAccess, which gives the offset");
    return (void*)(uintptr_t)v->imp->alloc.handle;
}

//////////////////////////////////// device access ////////////////////////////////////

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
    // one global order: an acquirer holding a backing only ever waits for backings after it
    std::sort(uses_.begin(), uses_.end(), [](const DeviceUse& a, const DeviceUse& b) {
        const PhysicalBacking* pa = a.u->accelState->backing;
        const PhysicalBacking* pb = b.u->accelState->backing;
        if (pa != pb)
            return std::less<const PhysicalBacking*>()(pa, pb);
        return std::less<const UMatData*>()(a.u, b.u);
    });
    groups_.clear();
    for (size_t i = 0; i < uses_.size(); i++)
    {
        PhysicalBacking* b = uses_[i].u->accelState->backing;
        if (groups_.empty() || groups_.back().backing != b)
        {
            Group g;
            g.backing = b;
            g.first = i;
            g.count = 0;
            g.mode = 0;
            groups_.push_back(g);
        }
        groups_.back().count++;
        groups_.back().mode |= uses_[i].mode & (DEV_READ | DEV_WRITE);
    }
    holder_ = std::this_thread::get_id();
    try
    {
        for (; acquired_ < groups_.size(); acquired_++)
        {
            const Group& g = groups_[acquired_];
            PhysicalBacking* b = g.backing;
            std::unique_lock<std::mutex> lk(b->mutex);
            const bool write = (g.mode & DEV_WRITE) != 0;
            checkOwnHolds(b, write);
            b->cond.wait(lk, [b, write]{ return !b->busy && !deviceUseInFlight(b, write); });
            endOrphanMappingsLocked(b);     // views whose last Mat is gone, whose unmap has not run yet
            // a CPU access whose end failed is still open: the device stays away until it ends
            bool conflict = !retryFailedCpuEndsLocked(b);
            for (size_t k = g.first; k < g.first + g.count && !conflict; k++)
            {
                const BufferState* v = uses_[k].u->accelState;
                const bool w = (uses_[k].mode & DEV_WRITE) != 0;
                if (w && v->readOnly)
                    conflict = true;
                else if (cpuConflictLocked(b, v, w))
                    conflict = true;    // live Mat views: the device must not race with them
                else if ((uses_[k].mode & DEV_READ) || !(uses_[k].mode & DEV_DISCARD))
                    checkNotPoisoned(v);    // a write of the whole view may reuse a poisoned one
            }
            if (conflict)
            {
                lk.unlock();
                abandon();
                return false;
            }
            for (size_t k = g.first; k < g.first + g.count; k++)
            {
                BufferState* v = uses_[k].u->accelState;
                UMatData* u = uses_[k].u;
                const bool needContent = (uses_[k].mode & DEV_READ) || !(uses_[k].mode & DEV_DISCARD);
                if (usesShadow(v) && u->deviceCopyObsolete() && !u->hostCopyObsolete() && needContent)
                {
                    // the host copy is newer: the backend that owns the memory uploads it
                    const bool sameBackend = stream->backend == v->backend();
                    StreamPtr us = sameBackend ? stream : v->backend()->defaultStream(v->device());
                    uploadShadowLocked(v, u, us, !sameBackend);
                }
            }
            if (!b->writeFence.empty() && b->writeFence.streamUid() != stream->uid)
                waits.push_back(b->writeFence);
            if (write)
            {
                for (size_t i = 0; i < b->readFences.size(); i++)
                    if (b->readFences[i].streamUid() != stream->uid)
                        waits.push_back(b->readFences[i]);
                b->pendingWriters++;
            }
            else
                b->pendingReaders++;
            if (longHold_)
            {
                PhysicalBacking::Hold h;
                h.thread = holder_;
                h.mode = g.mode;
                b->longHolds.push_back(h);
            }
        }
        std::vector<std::pair<const BufferState*, int> > ext = externalUses();
        for (size_t i = 0; i < ext.size(); i++)
            beginExternal(ext[i].first, stream, ext[i].second);
    }
    catch (...)
    {
        abandon();
        throw;
    }
    return true;
}

std::vector<std::pair<const BufferState*, int> > DeviceAcquisition::externalUses() const
{
    // one bracket per import of memory with implicit fences, with the access of all its uses
    std::vector<std::pair<const BufferState*, int> > r;
    for (size_t i = 0; i < uses_.size(); i++)
    {
        const BufferState* v = uses_[i].u->accelState;
        if (!(v->backing->flags & CV_ACCEL_MEMF_EXTERNAL_SYNC))
            continue;
        size_t j = 0;
        while (j < r.size() && r[j].first->imp != v->imp)
            j++;
        if (j == r.size())
            r.push_back(std::make_pair(v, 0));
        r[j].second |= accessOfMode(uses_[i].mode);
    }
    return r;
}

//! ends the acquisition of one backing; with @p fence the work was enqueued and completes at that fence
void DeviceAcquisition::finish(const Group& g, const FenceRef* fence, const char* failure)
{
    PhysicalBacking* b = g.backing;
    std::lock_guard<std::mutex> lk(b->mutex);
    const bool write = (g.mode & DEV_WRITE) != 0;
    if (fence)
    {
        if (write)
        {
            // writers exclude everyone, so every read fence here was waited for by this work
            b->writeFence = *fence;
            b->readFences.clear();
            b->writeRanges.clear();
            b->generation++;
            for (size_t k = g.first; k < g.first + g.count; k++)
            {
                if (!(uses_[k].mode & DEV_WRITE))
                    continue;
                UMatData* u = uses_[k].u;
                const BufferState* v = u->accelState;
                u->markHostCopyObsolete(true);
                u->markDeviceCopyObsolete(false);
                try
                {
                    b->writeRanges.push_back(std::make_pair(v->start, v->start + v->size));
                    if (failure)
                        addPoison(b, v->start, v->size, failure);   // part of it may have been written
                    else if (uses_[k].mode & DEV_DISCARD)
                        clearPoison(b, v->start, v->size);
                }
                catch (...) {}
            }
        }
        else if (!fence->empty())
        {
            bool replaced = false;
            for (size_t i = 0; i < b->readFences.size() && !fence->isUnknown(); i++)
            {
                if (!b->readFences[i].isUnknown() && b->readFences[i].streamUid() == fence->streamUid())
                {
                    b->readFences[i] = *fence;  // in-order stream: the newer fence covers the older one
                    replaced = true;
                    break;
                }
            }
            if (!replaced)
                b->readFences.push_back(*fence);
        }
    }
    int& pending = write ? b->pendingWriters : b->pendingReaders;
    CV_DbgAssert(pending > 0);
    if (pending > 0)
        pending--;
    if (longHold_)
    {
        for (size_t i = 0; i < b->longHolds.size(); i++)
        {
            if (b->longHolds[i].thread == holder_ && b->longHolds[i].mode == g.mode)
            {
                b->longHolds.erase(b->longHolds.begin() + i);
                break;
            }
        }
    }
    b->cond.notify_all();
}

void DeviceAcquisition::endExternalUses(const FenceRef& fence) CV_NOEXCEPT
{
    if (acquired_ == 0)
        return;
    try
    {
        std::vector<std::pair<const BufferState*, int> > ext = externalUses();
        for (size_t i = 0; i < ext.size(); i++)
            endExternal(ext[i].first, fence, ext[i].second);
    }
    catch (...) {}
}

void DeviceAcquisition::record(const FenceRef& fence)
{
    endExternalUses(fence);
    for (size_t i = 0; i < acquired_; i++)
        finish(groups_[i], &fence, NULL);
    acquired_ = 0;
}

void DeviceAcquisition::fail(const FenceRef& fence, const std::string& reason)
{
    endExternalUses(fence);
    for (size_t i = 0; i < acquired_; i++)
        finish(groups_[i], &fence, reason.c_str());
    acquired_ = 0;
}

void DeviceAcquisition::abandon()
{
    while (acquired_ > 0)
    {
        acquired_--;
        finish(groups_[acquired_], NULL, NULL);
    }
}

//////////////////////////////////// release ////////////////////////////////////

void freeHostCopy(RetiredHostCopy& c) CV_NOEXCEPT
{
    if (!c.shadow)
        return;
    if (c.kind == SHADOW_BACKEND)
    {
        // a lost device's pinned memory is leaked with the rest of its memory
        if (!c.backend->isLost(c.device))
        {
            c.alloc.desc.device_index = (uint32_t)c.device;
            c.backend->recyclePinned(c.alloc);
        }
    }
    else if (c.kind == SHADOW_FASTMALLOC)
        fastFree(c.shadow);
    else if (c.kind == SHADOW_PAGES)
        freePages(c.shadow, c.pages);
    c.shadow = 0;
}

bool isDeviceAllocator(const MatAllocator* a)
{
    return dynamic_cast<const DeviceAllocator*>(a) != NULL;
}

SamePlacement::SamePlacement(const _OutputArray& out, const _InputArray& model) : out_(NULL)
{
    if (out.kind() != _InputArray::UMAT || model.kind() != _InputArray::UMAT)
        return;
    UMat& o = out.getUMatRef();
    if (o.allocator)
        return;     // explicit placement wins
    const UMat& m = *(const UMat*)model.getObj();
    if (!m.u || !m.u->accelState || !isDeviceAllocator(m.u->currAllocator))
        return;
    o.allocator = const_cast<MatAllocator*>(m.u->currAllocator);
    out_ = &o;
}

SamePlacement::~SamePlacement()
{
    if (out_)
        out_->allocator = NULL;     // placement belongs to the data, not to the header
}

bool sameAccelDevice(const UMatData* a, const UMatData* b)
{
    return a && b && a->accelState && b->accelState &&
           a->accelState->backend() == b->accelState->backend() &&
           a->accelState->device() == b->accelState->device();
}

/** Release path: waits for every fence on a backing whose memory goes back to a Mat. Runs from
    ~UMat, so it never throws; false when the device failed (logged). */
static bool drainForRelease(PhysicalBacking* b) CV_NOEXCEPT
{
    std::unique_lock<std::mutex> lk(b->mutex);
    bool ok = true;
    try
    {
        FenceRef w = b->writeFence;
        std::vector<FenceRef> reads = b->readFences;
        lk.unlock();
        // the memory goes back to its owner: no time limit (a device that reports a failure has stopped)
        ok = w.waitNoThrow(false);
        for (size_t i = 0; i < reads.size(); i++)
            ok = reads[i].waitNoThrow(false) && ok;
    }
    catch (...)
    {
        if (!lk.owns_lock())
            lk.lock();
        ok = b->writeFence.waitNoThrow(false);
        for (size_t i = 0; i < b->readFences.size(); i++)
            ok = b->readFences[i].waitNoThrow(false) && ok;
    }
    return ok;
}

//! release path: the device result back into the memory of the Mat the UMat came from; never throws
static void copyBackToMat(UMatData* u, BufferState* v) CV_NOEXCEPT
{
    try
    {
        // other views of the memory may still be in use: the copy is a CPU read like any other
        IdleCpuGuard idle(v);
        PhysicalBacking* b = v->backing;
        std::unique_lock<std::mutex> lk(b->mutex);
        BusyScope busy(b, lk, false);
        waitDeviceWork(b, lk, false, v);
        if (usesShadow(v))
            downloadWhole(v);   // into the Mat's memory
        else
        {
            CpuScope cpu(b, ACCESS_READ);
            memcpy(u->origdata, cpu.base() + v->start, u->size);
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
        CV_LOG_ERROR(NULL, "accel(" << v->backend()->name() << "): the device result could not be copied back; "
                     "the Mat keeps its previous content");
    }
    catch (...) {}
}

/** Release path: CPU writes in the host copy of memory that belongs to someone else reach that
    memory before it goes back. Never throws. */
static void flushHostCopy(UMatData* u, BufferState* v) CV_NOEXCEPT
{
    PhysicalBacking* b = v->backing;
    try
    {
        std::unique_lock<std::mutex> lk(b->mutex);
        if (!usesShadow(v) || !v->shadow || !u->deviceCopyObsolete() || u->hostCopyObsolete() ||
            findPoison(b, v->start, v->size))
            return;
        BusyScope busy(b, lk, true);
        uploadShadowLocked(v, u, v->backend()->defaultStream(v->device()), true);
    }
    catch (const std::exception& e)
    {
        try
        {
            CV_LOG_ERROR(NULL, "accel(" << v->backend()->name() << "): CPU writes to wrapped memory could not be "
                         "written back: " << e.what());
        }
        catch (...) {}
    }
    catch (...) {}
}

void DeviceAllocator::deallocate(UMatData* u) const
{
    if (!u)
        return;
    CV_Assert(u->urefcount == 0);
    CV_Assert(u->refcount == 0 && "UMat deallocation error: some derived Mat is still alive");
    BufferState* v = u->accelState;
    CV_Assert(v);
    PhysicalBacking* b = v->backing;

    // From here on errors are logged, not thrown: this runs from ~UMat.
    {
        std::unique_lock<std::mutex> lk(b->mutex);
        b->cond.wait(lk, [b]{ return !b->busy; });
        v->dying = true;    // a wrap of the same bytes waits until the view is gone
        if (v->hostMapped)
            endHostAccessLocked(v, u);  // the last view went without an unmap: flush, release the gate
    }
    if (v->fromMat)
    {
        // the memory or the content belongs to a Mat: finish everything, then hand it back
        const bool deviceOk = drainForRelease(b);
        bool matStale = false, poisoned = false;
        {
            std::lock_guard<std::mutex> lk(b->mutex);
            // For COPY and UNCACHED memory the Mat's memory is the host copy: only a device write
            // leaves it stale. Other memory holds its own copy of the content: any write counts.
            matStale = usesShadow(v) ? u->hostCopyObsolete() && !u->deviceCopyObsolete()
                                     : b->generation != v->matGeneration;
            poisoned = findPoison(b, v->start, v->size) != NULL;
        }
        if (matStale)
        {
            if (deviceOk && !poisoned && !v->backend()->isLost(v->device()))
                copyBackToMat(u, v);
            else
                CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): the device failed before its result was copied "
                             "back; the Mat keeps its previous content");
        }
        u->accelState = NULL;
        releaseView(v);
        u->handle = 0;
        u->data = u->origdata;
        u->flags &= ~(UMatData::TEMP_UMAT | UMatData::HOST_COPY_OBSOLETE | UMatData::DEVICE_COPY_OBSOLETE);
        u->currAllocator = u->prevAllocator;
        u->prevAllocator = NULL;
        u->currAllocator->deallocate(u);
        collectRetired(false);
        return;
    }

    if (b->borrowed)
        flushHostCopy(u, v);    // the owner gets the CPU's writes
    u->accelState = NULL;
    u->handle = 0;
    u->data = u->origdata = 0;
    releaseView(v);     // the last view of borrowed memory waits for the device here
    delete u;
    collectRetired(false);
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

//////////////////////////////////// Mat::getUMat ////////////////////////////////////

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
    BufferState* v = attachOwnedMemory(u, b_, device_, a);
    PhysicalBacking* b = v->backing;
    v->fromMat = true;
    if (usesShadow(v))
    {
        // the Mat's memory is the host copy; the device copy is filled when the device needs it
        v->shadow = u->origdata;
        v->shadowKind = SHADOW_USER;
        u->data = u->origdata;
        u->markHostCopyObsolete(false);
        u->markDeviceCopyObsolete(true);
    }
    else
    {
        // memory with a CPU view: copy the content in now, the Mat keeps its own bytes and
        // gets them back only if the UMat is written (see matGeneration)
        bool copied = false;
        try
        {
            IdleCpuGuard idle(v);
            std::unique_lock<std::mutex> lk(b->mutex);
            BusyScope busy(b, lk, true);
            CpuScope cpu(b, ACCESS_WRITE);
            memcpy(cpu.base() + v->start, u->origdata, u->size);
            copied = cpu.close();   // a failed flush: the device would not see it
        }
        catch (const std::exception& e)
        {
            CV_LOG_WARNING(NULL, "accel: " << e.what());
        }
        if (!copied)
        {
            u->accelState = NULL;
            u->handle = 0;
            releaseView(v);
            return false;   // Mat::getUMat falls back to the host allocator
        }
        u->markHostCopyObsolete(false);
        u->markDeviceCopyObsolete(false);
    }
    {
        std::lock_guard<std::mutex> lk(b->mutex);
        v->matGeneration = b->generation;
    }
    u->prevAllocator = u->currAllocator;
    u->currAllocator = this;
    u->flags |= UMatData::TEMP_UMAT;
    return true;
}

//////////////////////////////////// transfers ////////////////////////////////////

void DeviceAllocator::download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                               const size_t srcofs[], const size_t srcstep[], const size_t dststep[]) const
{
    BufferState* v = u->accelState;
    PhysicalBacking* b = v->backing;
    checkHostAccessAllowed(v, ACCESS_READ);
    IdleCpuGuard idle(v);
    std::unique_lock<std::mutex> lk(b->mutex);
    checkOwnHolds(b, false);
    BusyScope busy(b, lk, false);
    waitDeviceWork(b, lk, false, v);
    const size_t srcOffset = offsetND(srcofs, srcstep, dims);
    if (usesShadow(v))
    {
        if ((!u->hostCopyObsolete() && v->shadow) || u->deviceCopyObsolete())
        {
            // the host copy is current (or nothing was ever written): no DMA at all
            if (v->shadow)
                copyND((uchar*)dstptr, dststep, v->shadow + srcOffset, srcstep, dims, sz);
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
        r.src.alloc = &v->imp->alloc;
        r.src.offset = v->start + srcOffset;
        r.dst.host = dstptr;
        FenceRef f;
        try
        {
            beginExternal(v, s, ACCESS_READ);
            b_->check(b_->fn().copy(device_, s->handle, &r), "copy (device to host)", device_);
            f = recordFence(s);
            endExternal(v, f, ACCESS_READ);
            f.wait(false);  // the destination is user memory: no time limit
        }
        catch (...)
        {
            if (f.empty())
                fenceAfterFailure(s).waitNoThrow(false);    // part of the copy may be queued, writing user memory
            throw;
        }
        Counters& c = counters();
        c.bytesD2H += totalND(sz, dims);
        c.transfers++;
        return;
    }
    // memory the CPU addresses directly: through what is open, or a level opened for the copy.
    // A second begin_cpu_access would invalidate CPU caches under writes a view has not flushed.
    CpuScope cpu(b, ACCESS_READ);
    copyND((uchar*)dstptr, dststep, cpu.base() + v->start + srcOffset, srcstep, dims, sz);
}

void DeviceAllocator::upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                             const size_t dstofs[], const size_t dststep[], const size_t srcstep[]) const
{
    BufferState* v = u->accelState;
    PhysicalBacking* b = v->backing;
    if (v->readOnly)
        CV_Error_(Error::StsError, ("accel(%s): this UMat wraps read-only memory", b_->name().c_str()));
    IdleCpuGuard idle(v);
    std::unique_lock<std::mutex> lk(b->mutex);
    checkOwnHolds(b, true);
    BusyScope busy(b, lk, true);
    if (v->hostMapped && u->refcount > 0)
        CV_Error(Error::StsError, "accel: cannot copy into a UMat while a Mat view of it is alive");
    if (v->hostMapped)
        endHostAccessLocked(v, u);
    const size_t dstOffset = offsetND(dstofs, dststep, dims);
    // a copy into all of the buffer may reuse a poisoned one, a copy into part of it may not
    const bool whole = dstOffset == 0 && totalND(sz, dims) == u->size;
    // memory without a CPU view (discrete, protected) is written by the device's copy engine;
    // data may enter protected memory, it just never comes out
    if (usesShadow(v) || hostAccessClass(v) == CV_ACCEL_HOST_ACCESS_NONE)
    {
        if (usesShadow(v) && u->deviceCopyObsolete() && !u->hostCopyObsolete())
        {
            // the host copy is current anyway: write there, the device copy follows lazily
            waitDeviceWork(b, lk, true, v, whole);
            copyND(v->shadow + dstOffset, dststep, (const uchar*)srcptr, srcstep, dims, sz);
            b->generation++;
            if (whole)
                clearPoison(b, v->start, v->size);
            return;
        }
        if (!whole)
            checkNotPoisoned(v);
        // one DMA straight into the device copy, after everyone using the old content
        StreamPtr s = b_->defaultStream(device_);
        streamWait(s, b->writeFence);
        for (size_t i = 0; i < b->readFences.size(); i++)
            streamWait(s, b->readFences[i]);
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
        r.dst.alloc = &v->imp->alloc;
        r.dst.offset = v->start + dstOffset;
        FenceRef f;
        try
        {
            beginExternal(v, s, ACCESS_WRITE);
            b_->check(b_->fn().copy(device_, s->handle, &r), "copy (host to device)", device_);
            f = recordFence(s);
            endExternal(v, f, ACCESS_WRITE);
            lk.unlock();
            f.wait(false);  // the source is user memory that may be freed when we return: no time limit
            lk.lock();
        }
        catch (...)
        {
            const std::string reason = currentExceptionMessage();
            FenceRef after = f;
            if (f.empty())
            {
                after = fenceAfterFailure(s);
                after.waitNoThrow(false);   // part of the copy may be queued, reading user memory
            }
            if (!lk.owns_lock())
                lk.lock();
            // the device copy may be partly overwritten: invalid until written whole
            recordWriteLocked(b, v, after);
            addPoison(b, v->start, v->size, reason);
            u->markHostCopyObsolete(true);
            u->markDeviceCopyObsolete(false);
            b->generation++;
            throw;
        }
        recordWriteLocked(b, v, f);
        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);
        b->generation++;
        if (whole)
            clearPoison(b, v->start, v->size);
        Counters& c = counters();
        c.bytesH2D += totalND(sz, dims);
        c.transfers++;
        return;
    }
    waitDeviceWork(b, lk, true, v, whole);
    // a level for the write, closed right after it: its end makes the bytes visible to the device
    CpuScope cpu(b, ACCESS_WRITE);
    copyND(cpu.base() + v->start + dstOffset, dststep, (const uchar*)srcptr, srcstep, dims, sz);
    b->generation++;
    u->markHostCopyObsolete(false);
    u->markDeviceCopyObsolete(false);
    if (!cpu.close())
    {
        // the device may see part of the new content: invalid until written whole
        addPoison(b, v->start, v->size, "end_cpu_access failed after a copy into the UMat");
        CV_Error_(Error::StsError, ("accel(%s): end_cpu_access failed after a copy into the UMat", b_->name().c_str()));
    }
    if (whole)
        clearPoison(b, v->start, v->size);
}

void DeviceAllocator::copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
                           const size_t srcofs[], const size_t srcstep[],
                           const size_t dstofs[], const size_t dststep[], bool sync) const
{
    // two buffers of one device: a device copy
    CV_Assert(src->accelState && dst->accelState);
    BufferState* sv = src->accelState;
    BufferState* dv = dst->accelState;
    if ((sv->backing->flags & CV_ACCEL_MEMF_PROTECTED) && !(dv->backing->flags & CV_ACCEL_MEMF_PROTECTED))
        CV_Error_(Error::StsError, ("accel(%s): protected content cannot be copied into ordinary memory",
                                    b_->name().c_str()));
    const size_t srcOffset = offsetND(srcofs, srcstep, dims);
    const size_t dstOffset = offsetND(dstofs, dststep, dims);
    if (sv->backing == dv->backing)
    {
        // two views of one memory (aliases, ROIs of one buffer): a device copy between bytes that
        // overlap is undefined, so it goes through a temporary unless it would change nothing
        const uint64_t s0 = sv->start + srcOffset, d0 = dv->start + dstOffset;
        const uint64_t se = extentND(sz, srcstep, dims), de = extentND(sz, dststep, dims);
        if (rangesOverlap(s0, se, d0, de))
        {
            bool same = s0 == d0;
            for (int i = 0; same && i < dims - 1; i++)
                same = srcstep[i] == dststep[i];
            if (same)
                return;
            const size_t total = totalND(sz, dims);
            CV_Assert(total <= (size_t)std::numeric_limits<int>::max());
            UMat tmp;
            tmp.allocator = const_cast<DeviceAllocator*>(this);
            tmp.create(1, (int)total, CV_8U);
            // the temporary is continuous: the region is copied into it packed
            std::vector<size_t> packed(dims);
            size_t acc = sz[dims - 1];
            packed[dims - 1] = 1;
            for (int i = dims - 2; i >= 0; i--)
            {
                packed[i] = acc;
                acc *= sz[i];
            }
            copy(src, tmp.u, dims, sz, srcofs, srcstep, NULL, packed.data(), sync);
            copy(tmp.u, dst, dims, sz, NULL, packed.data(), dstofs, dststep, sync);
            return;
        }
    }
    StreamPtr s = b_->defaultStream(device_);
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
    r.src.alloc = &sv->imp->alloc;
    r.src.offset = sv->start + srcOffset;
    r.dst.alloc = &dv->imp->alloc;
    r.dst.offset = dv->start + dstOffset;
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
    // user memory: the owner may touch it once we return (as in OpCall), no time limit; memory
    // shared through implicit fences gets the fence of the copy instead
    const PhysicalBacking* sb = sv->backing;
    const PhysicalBacking* db = dv->backing;
    if ((sb->borrowed && !(sb->flags & CV_ACCEL_MEMF_EXTERNAL_SYNC)) || (db->borrowed && !(db->flags & CV_ACCEL_MEMF_EXTERNAL_SYNC)))
        done.wait(false);
}

bool DeviceAllocator::getMemoryDesc(const UMatData* u, cvAccelMemoryDesc* desc) const
{
    if (!u || !u->accelState || !desc)
        return false;
    const BufferState* v = u->accelState;
    *desc = v->imp->alloc.desc;
    desc->flags |= v->backing->flags & (CV_ACCEL_MEMF_PROTECTED | CV_ACCEL_MEMF_EXTERNAL_SYNC);
    if (v->readOnly)
        desc->flags |= CV_ACCEL_MEMF_READ_ONLY;
    if (v->backing->borrowed)
        desc->flags |= CV_ACCEL_MEMF_IMPORTED;
    return true;
}

}}} // namespace cv::accel::impl

namespace cv { namespace accel { namespace detail {

void setValidation(bool on)
{
    impl::validating().store(on);
}

void injectPageProtectionFailure(int after)
{
    impl::pageFailureCountdown().store(after);
}

}}} // namespace cv::accel::detail
