// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: physical memory. One PhysicalBacking per piece of physical memory, however many
// allocations, imports and UMatData objects reach it.
//
// Identity. Core identifies host memory (virtual addresses) and file-backed memory such as
// dma-bufs (their inode) itself, so that imports through different backends meet; other memory
// is identified by the backend that reaches it (cvAccelAllocation::backing_id). A backing covers
// [start, start + size) of its identity's address space, and two backings of one identity never
// overlap: a wrap that overlaps a backing without lying inside it is refused.
//
// Imports. A backend (device, layout) reaches a backing through one import of all of it, so
// that core can open CPU access to the whole memory at once; a UMatData is a view of a range of
// an import.
//
// Lifetime. A backing lives while views refer to it. When the last one goes, the backing stays
// registered until the device work on it completed and its imports are released, and a wrap of
// the same memory in the meantime revives it, fences included: there is never a second state
// for bytes that the first one still orders.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#include <opencv2/core/utils/logger.hpp>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace cv { namespace accel { namespace impl {

Import::Import() : backend(0), device(-1), modifier(0), hostReservation(0)
{
    memset(&alloc, 0, sizeof(alloc));
}

static std::atomic<uint64>& backingIds()
{
    static std::atomic<uint64>* c = new std::atomic<uint64>(0);
    return *c;
}

PhysicalBacking::PhysicalBacking()
    : id(++backingIds()), start(0), size(0), flags(0), borrowed(false), refs(0), registered(false), retiring(false), deferred(false),
      busy(false), pendingReaders(0), pendingWriters(0), generation(0), cpuViews(0), cpuBase(0), idleGateHeld(0)
{}

PhysicalBacking::~PhysicalBacking()
{
    for (size_t i = 0; i < imports.size(); i++)
        delete imports[i];
}

BufferState::BufferState(PhysicalBacking* b, Import* imp_, uint64_t start_, uint64_t size_)
    : backing(b), imp(imp_), u(0), start(start_), size(size_), readOnly(false), fromMat(false),
      shadow(0), shadowKind(SHADOW_NONE), shadowPages(0), hostMapped(false), hostAccess(0), dying(false),
      matGeneration(0), viewHash(0), viewHashed(false)
{
    memset(&shadowAlloc, 0, sizeof(shadowAlloc));
}

//////////////////////////////////// poisoned ranges ////////////////////////////////////

const PoisonRange* findPoison(const PhysicalBacking* b, uint64_t begin, uint64_t size)
{
    for (size_t i = 0; i < b->poison.size(); i++)
    {
        const PoisonRange& r = b->poison[i];
        if (rangesOverlap(r.begin, r.end - r.begin, begin, size))
            return &r;
    }
    return NULL;
}

void addPoison(PhysicalBacking* b, uint64_t begin, uint64_t size, const std::string& reason)
{
    if (size == 0)
        return;
    PoisonRange r;
    r.begin = begin;
    r.end = begin + size;
    r.reason = reason;
    b->poison.push_back(r);
}

void clearPoison(PhysicalBacking* b, uint64_t begin, uint64_t size)
{
    if (b->poison.empty())
        return;
    // what lies inside the range is gone; what sticks out of it stays poisoned
    const uint64_t end = begin + size;
    std::vector<PoisonRange> keep;
    for (size_t i = 0; i < b->poison.size(); i++)
    {
        const PoisonRange& r = b->poison[i];
        if (r.end <= begin || r.begin >= end)
        {
            keep.push_back(r);
            continue;
        }
        if (r.begin < begin)
        {
            PoisonRange left = r;
            left.end = begin;
            keep.push_back(left);
        }
        if (r.end > end)
        {
            PoisonRange right = r;
            right.begin = end;
            keep.push_back(right);
        }
    }
    b->poison.swap(keep);
}

//////////////////////////////////// the registry of backings ////////////////////////////////////

namespace {

struct Backings
{
    static Backings& instance()
    {
        static Backings* r = new Backings();    // never destroyed: buffers may outlive static destruction
        return *r;
    }

    std::mutex mutex;
    std::condition_variable gone;       //!< a retiring backing was unregistered
    std::map<BackingKey, std::vector<PhysicalBacking*> > keyed;
    std::vector<PhysicalBacking*> deferred;     //!< core's own memory without references, fences pending
    std::mutex collecting;              //!< one collector at a time: only a collector frees deferred backings
};

void overlapError()
{
    CV_Error(Error::StsBadArg, "accel: this memory overlaps memory that is already wrapped without lying inside it; "
                               "wrap the whole memory once and take views of it");
}

/** The registered backing of @p key that holds [start, start + size), with one more reference,
    or NULL. Waits while an overlapping one is being freed. */
PhysicalBacking* findBacking(const BackingKey& key, uint64_t start, uint64_t size)
{
    Backings& r = Backings::instance();
    std::unique_lock<std::mutex> lk(r.mutex);
    for (;;)
    {
        std::map<BackingKey, std::vector<PhysicalBacking*> >::iterator it = r.keyed.find(key);
        if (it == r.keyed.end())
            return NULL;
        PhysicalBacking* hit = NULL;
        bool wait = false;
        for (size_t i = 0; i < it->second.size(); i++)
        {
            PhysicalBacking* b = it->second[i];
            if (!rangesOverlap(b->start, b->size, start, size))
                continue;
            if (b->retiring)
            {
                wait = true;    // being freed: the memory gets a new backing once it is gone
                break;
            }
            if (!rangeInside(b->start, b->size, start, size))
                overlapError();
            hit = b;
            break;
        }
        if (wait)
        {
            r.gone.wait(lk);
            continue;
        }
        if (hit)
            hit->refs++;    // a backing without references (waiting for its fences) is revived
        return hit;
    }
}

/** Registers @p nb (one reference, the caller's). Returns the backing to use: @p nb, or the one
    another thread registered for the same memory first (with one more reference). */
PhysicalBacking* registerBacking(PhysicalBacking* nb)
{
    if (nb->key.domain == BACKING_ANONYMOUS)
        return nb;
    Backings& r = Backings::instance();
    std::unique_lock<std::mutex> lk(r.mutex);
    for (;;)
    {
        std::vector<PhysicalBacking*>& list = r.keyed[nb->key];
        PhysicalBacking* hit = NULL;
        bool wait = false;
        for (size_t i = 0; i < list.size(); i++)
        {
            PhysicalBacking* b = list[i];
            if (!rangesOverlap(b->start, b->size, nb->start, nb->size))
                continue;
            if (b->retiring)
            {
                wait = true;
                break;
            }
            if (!rangeInside(b->start, b->size, nb->start, nb->size))
                overlapError();
            hit = b;
            break;
        }
        if (wait)
        {
            r.gone.wait(lk);
            continue;
        }
        if (hit)
        {
            hit->refs++;
            return hit;
        }
        list.push_back(nb);
        nb->registered = true;
        return nb;
    }
}

//! under the registry mutex
void unregisterLocked(Backings& r, PhysicalBacking* b)
{
    if (!b->registered)
        return;
    std::map<BackingKey, std::vector<PhysicalBacking*> >::iterator it = r.keyed.find(b->key);
    if (it != r.keyed.end())
    {
        std::vector<PhysicalBacking*>& list = it->second;
        list.erase(std::remove(list.begin(), list.end(), b), list.end());
        if (list.empty())
            r.keyed.erase(it);
    }
    b->registered = false;
}

void removeDeferredLocked(Backings& r, PhysicalBacking* b)
{
    if (!b->deferred)
        return;
    r.deferred.erase(std::remove(r.deferred.begin(), r.deferred.end(), b), r.deferred.end());
    b->deferred = false;
}

/** Releases what a backing without references holds: the host copies of its released views and
    its imports. Its fences completed (or its device stopped); nobody can reach it any more. */
void destroyBacking(PhysicalBacking* b) CV_NOEXCEPT
{
    for (size_t i = 0; i < b->retired.size(); i++)
        freeHostCopy(b->retired[i]);
    b->retired.clear();
    b->writeFence.reset();
    b->readFences.clear();
    for (size_t i = 0; i < b->imports.size(); i++)
    {
        Import* imp = b->imports[i];
        Backend* be = imp->backend;
        // A lost device rejects every call, or is still running after a timed-out wait (where a
        // free would block, as cudaFree does): its allocations and imports are leaked.
        if (!be->isLost(imp->device) && (imp->alloc.handle || imp->alloc.backend_data))
        {
            if (be->fn().deallocate(&imp->alloc) != CV_ACCEL_OK)
            {
                try
                {
                    CV_LOG_ERROR(NULL, "accel(" << be->name() << "): failed to release device memory: "
                                 << be->fn().get_last_error());
                }
                catch (...) {}
            }
        }
        if (imp->hostReservation)
            be->releaseHostImport(imp->device, imp->hostReservation);
    }
}

//! the backing has no reference left: frees it, unless a wrap revived it meanwhile
void retireIfUnused(PhysicalBacking* b) CV_NOEXCEPT
{
    Backings& r = Backings::instance();
    {
        std::lock_guard<std::mutex> lk(r.mutex);
        if (b->refs > 0)
            return;         // revived by a wrap of the same memory: it lives on
        b->retiring = true; // lookups wait until it is gone, so no second state appears meanwhile
        removeDeferredLocked(r, b);
    }
    destroyBacking(b);
    {
        std::lock_guard<std::mutex> lk(r.mutex);
        unregisterLocked(r, b);
    }
    r.gone.notify_all();
    delete b;
}

//! memory of someone else: waits without limit for every fence on it (never throws)
void drainBorrowed(PhysicalBacking* b) CV_NOEXCEPT
{
    std::unique_lock<std::mutex> lk(b->mutex);
    try
    {
        // waits outside of the lock: a wrap that revives the backing meanwhile adds its own work,
        // which the next release waits for
        FenceRef w = b->writeFence;
        std::vector<FenceRef> reads = b->readFences;
        lk.unlock();
        w.waitNoThrow(false);
        for (size_t i = 0; i < reads.size(); i++)
            reads[i].waitNoThrow(false);
    }
    catch (...)
    {
        // no memory for the copies: wait under the lock instead
        if (!lk.owns_lock())
            lk.lock();
        b->writeFence.waitNoThrow(false);
        for (size_t i = 0; i < b->readFences.size(); i++)
            b->readFences[i].waitNoThrow(false);
    }
}

} // namespace

bool fencesSettledLocked(PhysicalBacking* b) CV_NOEXCEPT
{
    if (!b->writeFence.empty())
    {
        if (b->writeFence.poll() == FenceRef::PENDING)
            return false;
        b->writeFence.reset();
    }
    for (size_t i = 0; i < b->readFences.size(); )
    {
        if (b->readFences[i].poll() == FenceRef::PENDING)
            return false;
        b->readFences.erase(b->readFences.begin() + i);
    }
    pruneRetiredLocked(b);
    return b->retired.empty();
}

void pruneRetiredLocked(PhysicalBacking* b) CV_NOEXCEPT
{
    for (size_t i = 0; i < b->retired.size(); )
    {
        RetiredHostCopy& c = b->retired[i];
        bool done = c.writeFence.poll() != FenceRef::PENDING;
        for (size_t j = 0; done && j < c.readFences.size(); j++)
            done = c.readFences[j].poll() != FenceRef::PENDING;
        if (!done)
        {
            i++;
            continue;
        }
        freeHostCopy(c);
        b->retired.erase(b->retired.begin() + i);
    }
}

void releaseBacking(PhysicalBacking* b) CV_NOEXCEPT
{
    Backings& r = Backings::instance();
    {
        std::lock_guard<std::mutex> lk(r.mutex);
        CV_DbgAssert(b->refs > 0);
        if (--b->refs > 0)
            return;
        if (!b->borrowed)
        {
            // core's own memory: freed once its fences complete, and findable (revivable) until then
            if (!b->deferred)
            {
                try
                {
                    r.deferred.push_back(b);
                    b->deferred = true;
                }
                catch (...) {}  // no memory to remember it: leaked, which is safe
            }
            return;
        }
    }
    // memory of someone else is never handed back while the device may still use it
    drainBorrowed(b);
    retireIfUnused(b);
}

void collectRetired(bool waitAll) CV_NOEXCEPT
{
    Backings& r = Backings::instance();
    std::unique_lock<std::mutex> one(r.collecting, std::defer_lock);
    if (waitAll)
        one.lock();
    else if (!one.try_lock())
        return;     // another thread collects right now
    std::vector<PhysicalBacking*> pending;
    {
        std::lock_guard<std::mutex> lk(r.mutex);
        if (r.deferred.empty())
            return;
        try
        {
            pending = r.deferred;
        }
        catch (...)
        {
            return;
        }
    }
    // Only a collector removes backings from the list, and there is one at a time, so every
    // backing in it stays alive here (a revived one is merely taken off the list).
    for (size_t i = 0; i < pending.size(); i++)
    {
        PhysicalBacking* b = pending[i];
        bool settled = false;
        {
            std::unique_lock<std::mutex> lk(b->mutex);
            if (waitAll)
            {
                try
                {
                    FenceRef w = b->writeFence;
                    std::vector<FenceRef> reads = b->readFences;
                    lk.unlock();
                    w.waitNoThrow();
                    for (size_t j = 0; j < reads.size(); j++)
                        reads[j].waitNoThrow();
                }
                catch (...) {}
                if (!lk.owns_lock())
                    lk.lock();
            }
            settled = fencesSettledLocked(b);
        }
        if (!settled)
            continue;
        bool mine = false;
        {
            std::lock_guard<std::mutex> lk(r.mutex);
            if (b->refs == 0)
            {
                removeDeferredLocked(r, b);
                b->retiring = true;
                mine = true;
            }
            else
                removeDeferredLocked(r, b);     // revived: in use again
        }
        if (!mine)
            continue;
        destroyBacking(b);
        {
            std::lock_guard<std::mutex> lk(r.mutex);
            unregisterLocked(r, b);
        }
        r.gone.notify_all();
        delete b;
    }
}

//////////////////////////////////// views ////////////////////////////////////

bool addRefIfAlive(int* counter)
{
#if defined(_MSC_VER)
    long cur = *(volatile long*)counter;
    while (cur > 0)
    {
        const long prev = _InterlockedCompareExchange((volatile long*)counter, cur + 1, cur);
        if (prev == cur)
            return true;
        cur = prev;
    }
    return false;
#else
    int cur = __atomic_load_n(counter, __ATOMIC_ACQUIRE);
    while (cur > 0)
        if (__atomic_compare_exchange_n(counter, &cur, cur + 1, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return true;
    return false;
#endif
}

namespace {

//! frees the backend memory of a fresh allocation or import unless it was committed
struct MemoryGuard
{
    MemoryGuard(Backend* b_, cvAccelAllocation& a_) : b(b_), a(&a_), reservation(0), device(-1), armed(true) {}
    ~MemoryGuard()
    {
        if (!armed)
            return;
        if ((a->handle || a->backend_data) && b->fn().deallocate(a) != CV_ACCEL_OK)
        {
            try { CV_LOG_ERROR(NULL, "accel(" << b->name() << "): failed to release memory: " << b->fn().get_last_error()); }
            catch (...) {}
        }
        if (reservation)
            b->releaseHostImport(device, reservation);
    }
    Backend* b;
    cvAccelAllocation* a;
    const void* reservation;
    int device;
    bool armed;
};

//! the flat address of byte @p offset of an import, or its handle when the memory has none
void* viewHandle(const Import* imp, uint64_t offset)
{
    const uint32_t kind = imp->alloc.desc.handle_kind;
    if (kind == CV_ACCEL_HANDLE_DEVICE_POINTER || kind == CV_ACCEL_HANDLE_HOST_POINTER)
        return (void*)(uintptr_t)(imp->alloc.handle + offset);
    return (void*)(uintptr_t)imp->alloc.handle;
}

//! a new view of [start, start + size) of @p imp, set on @p u (backing mutex held)
BufferState* addViewLocked(PhysicalBacking* b, Import* imp, UMatData* u, uint64_t start, uint64_t size)
{
    std::unique_ptr<BufferState> v(new BufferState(b, imp, start, size));
    v->u = u;
    b->views.push_back(v.get());
    u->accelState = v.get();
    u->handle = viewHandle(imp, start);
    return v.release();
}

} // namespace

BufferState* attachOwnedMemory(UMatData* u, Backend* be, int device, const cvAccelAllocation& a0)
{
    cvAccelAllocation a = a0;
    MemoryGuard guard(be, a);   // frees the memory if anything below fails
    std::unique_ptr<PhysicalBacking> nb(new PhysicalBacking());
    nb->imports.reserve(1);
    std::unique_ptr<Import> imp(new Import());
    imp->backend = be;
    imp->device = device;
    imp->alloc = a;
    imp->alloc.desc.backend_id = (uint32_t)be->id();
    imp->alloc.desc.device_index = (uint32_t)device;
    imp->modifier = a.desc.layout_modifier;
    // a backend identity makes a later wrap of this memory (UMat::handle(), DeviceAccess) find it
    uint64_t end = 0;
    if (a.backing_id && rangeEnd(a.backing_offset, a.size, end))
    {
        nb->key = BackingKey(BACKING_BACKEND, (uint64_t)be->id(), a.backing_id);
        nb->start = a.backing_offset;
    }
    nb->size = a.size;
    nb->flags = a.desc.flags & CV_ACCEL_MEMF_PROTECTED;
    nb->borrowed = false;
    nb->refs = 1;
    nb->imports.push_back(imp.release());   // capacity is reserved
    BufferState* v = 0;
    {
        std::lock_guard<std::mutex> lk(nb->mutex);
        v = addViewLocked(nb.get(), nb->imports[0], u, 0, a.size);
    }
    PhysicalBacking* got = 0;
    try
    {
        got = registerBacking(nb.get());
    }
    catch (...)
    {
        u->accelState = NULL;
        u->handle = 0;
        delete v;
        throw;
    }
    if (got != nb.get())
    {
        // the backend gave a new allocation the identity of memory in use: never share that
        releaseBacking(got);
        u->accelState = NULL;
        u->handle = 0;
        delete v;
        CV_Error_(Error::StsInternal, ("accel(%s): the backend reported the identity of memory in use for a new allocation",
                                       be->name().c_str()));
    }
    guard.armed = false;
    nb.release();
    if (a.desc.host_access == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
        be->idleGate(device).used = true;
    noteBufferCreated();
    return v;
}

void releaseView(BufferState* v) CV_NOEXCEPT
{
    PhysicalBacking* b = v->backing;
    {
        std::lock_guard<std::mutex> lk(b->mutex);
        b->views.erase(std::remove(b->views.begin(), b->views.end(), v), b->views.end());
        if (v->shadow && v->shadowKind != SHADOW_USER)
        {
            // an upload from this host copy may still be queued: freed once the device is past it
            try
            {
                RetiredHostCopy c;
                c.shadow = v->shadow;
                c.kind = v->shadowKind;
                c.pages = v->shadowPages;
                c.alloc = v->shadowAlloc;
                c.backend = v->backend();
                c.device = v->device();
                c.writeFence = b->writeFence;
                c.readFences = b->readFences;
                b->retired.push_back(c);
            }
            catch (...) {}  // no memory to remember it: leaked, which is safe
            v->shadow = 0;
        }
        pruneRetiredLocked(b);
        b->cond.notify_all();   // wraps of the same bytes wait for a dying view to go
    }
    delete v;
    noteBufferDestroyed();
    releaseBacking(b);
}

//////////////////////////////////// DeviceAllocator: memory ////////////////////////////////////

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

cvAccelAllocation DeviceAllocator::allocateMemory(uint64_t total, int kind) const
{
    cvAccelAllocation a;
    memset(&a, 0, sizeof(a));
    if (total == 0)
        return a;
    const uint32_t allocFlags = protected_ ? CV_ACCEL_MEMF_PROTECTED : 0;
    cvAccelStatus status = b_->fn().allocate(device_, total, (uint32_t)kind, allocFlags, &a);
    if (status == CV_ACCEL_OUT_OF_MEMORY)
    {
        // memory held by buffers that are released but still in flight
        collectRetired(true);
        memset(&a, 0, sizeof(a));
        status = b_->fn().allocate(device_, total, (uint32_t)kind, allocFlags, &a);
    }
    if (protected_ && status == CV_ACCEL_NOT_IMPLEMENTED)
        CV_Error_(Error::StsNotImplemented, ("accel(%s): device %d has no protected memory for results "
                                             "computed from protected content", b_->name().c_str(), device_));
    b_->check(status, "allocate", device_);
    if (protected_ && !(a.desc.flags & CV_ACCEL_MEMF_PROTECTED))
    {
        MemoryGuard guard(b_, a);
        CV_Error_(Error::StsError, ("accel(%s): the backend returned ordinary memory for protected content",
                                    b_->name().c_str()));
    }
    if (a.size < total)
    {
        MemoryGuard guard(b_, a);
        CV_Error_(Error::StsInternal, ("accel(%s): the backend allocated fewer bytes than asked for", b_->name().c_str()));
    }
    return a;
}

UMatData* DeviceAllocator::allocate(int dims, const int* sizes, int type, void* data0, size_t* step,
                                    AccessFlag /*flags*/, UMatUsageFlags usageFlags) const
{
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; i--)
    {
        if (step)
            step[i] = total;
        CV_Assert(sizes[i] >= 0);
        if (sizes[i] != 0 && total > std::numeric_limits<size_t>::max() / (size_t)sizes[i])
            CV_Error(Error::StsNoMem, "accel: the array is larger than the address space");
        total *= (size_t)sizes[i];
    }
    if (data0)
    {
        // a UMat over existing host memory: import it when the device can, never copy silently
        WrapRequest req;
        req.ext.type = CV_ACCEL_EXTMEM_HOST_POINTER;
        req.ext.handle = (int64_t)(intptr_t)data0;
        req.ext.size = total;
        size_t offset = 0;
        bool referenced = false;
        UMatData* u = wrap(req, offset, referenced);
        CV_Assert(!referenced && offset == 0);  // a fresh view: UMat::create takes the reference
        return u;
    }

    cvAccelAllocation a = allocateMemory(total, kindFor(usageFlags));
    std::unique_ptr<UMatData> u;
    try
    {
        u.reset(new UMatData(this));
    }
    catch (...)
    {
        MemoryGuard guard(b_, a);
        throw;
    }
    u->size = total;
    attachOwnedMemory(u.get(), b_, device_, a);
    // nothing has been written yet: no copy is ever needed for this content
    u->markHostCopyObsolete(true);
    u->markDeviceCopyObsolete(true);
    return u.release();
}

//////////////////////////////////// DeviceAllocator: wraps ////////////////////////////////////

namespace {

//! a fresh import from a backend, released unless committed into a backing
struct FreshImport
{
    FreshImport(Backend* b, int device) : guard(b, alloc), committed(false)
    {
        memset(&alloc, 0, sizeof(alloc));
        guard.device = device;
    }
    cvAccelAllocation alloc;
    MemoryGuard guard;
    bool committed;

    Import* commit(Backend* b, int device, uint64_t modifier)
    {
        std::unique_ptr<Import> imp(new Import());
        imp->backend = b;
        imp->device = device;
        imp->modifier = modifier;
        imp->alloc = alloc;
        imp->hostReservation = guard.reservation;
        guard.armed = false;
        committed = true;
        return imp.release();
    }
};

void releaseImportError(Backend* b)
{
    try { CV_LOG_ERROR(NULL, "accel(" << b->name() << "): failed to release an import: " << b->fn().get_last_error()); }
    catch (...) {}
}

/** The identity of memory that core knows without asking a backend: host memory (its
    addresses) and file-backed memory such as dma-bufs (their inode). @p reqStart: where the
    requested bytes start in the identity's address space; [objStart, objStart + objSize): the
    memory a new backing for them covers. */
bool coreIdentity(const cvAccelExternalMemory& ext, BackingKey& key, uint64_t& reqStart,
                  uint64_t& objStart, uint64_t& objSize)
{
    if (ext.type == CV_ACCEL_EXTMEM_HOST_POINTER)
    {
        if (!rangeEnd((uint64_t)(uintptr_t)ext.handle, ext.offset, reqStart) || !rangeEnd(reqStart, ext.size, objStart))
            CV_Error(Error::StsBadArg, "accel: the host memory range does not fit in the address space");
        key = BackingKey(BACKING_HOST, 0, 0);
        objStart = reqStart;    // host memory has no known extent: the first wrap is the backing
        objSize = ext.size;
        return true;
    }
#ifndef _WIN32
    if (ext.type == CV_ACCEL_EXTMEM_DMABUF_FD)
    {
        struct stat st;
        if (fstat((int)ext.handle, &st) != 0)
            CV_Error(Error::StsBadArg, "accel: the dma-buf file descriptor is not open");
        key = BackingKey(BACKING_FILE, (uint64_t)st.st_dev, (uint64_t)st.st_ino);
        reqStart = ext.offset;
        uint64_t reqEnd = 0;
        CV_Assert(rangeEnd(ext.offset, ext.size, reqEnd));
        if (st.st_size > 0)
        {
            // the whole buffer: dma-buf implicit fences and cache syncs cover it all
            objStart = 0;
            objSize = (uint64_t)st.st_size;
            if (reqEnd > objSize)
                CV_Error(Error::StsBadArg, "accel: the range lies outside of the dma-buf");
        }
        else
        {
            objStart = ext.offset;  // a file without a size: what is wrapped
            objSize = ext.size;
        }
        return true;
    }
#endif
    return false;
}

} // namespace

/** Imports [@p ext.offset, + @p ext.size) of the memory: the backend call and the checks that
    do not depend on other imports. */
static void importChecked(Backend* b, int device, const cvAccelExternalMemory& ext, FreshImport& out)
{
    const void* hostPtr = ext.type == CV_ACCEL_EXTMEM_HOST_POINTER
                          ? (const void*)((const uchar*)(intptr_t)ext.handle + ext.offset) : 0;
    const uint32_t align = hostPtr ? b->deviceInfo(device).host_import_alignment : 0;
    if (align > 1 && (uintptr_t)hostPtr % align != 0)
        CV_Error_(Error::StsBadArg, ("accel(%s): the device imports host memory on %u-byte boundaries only; "
                                     "copy it into a UMat instead", b->name().c_str(), (unsigned)align));
    if (hostPtr)
    {
        if (!b->reserveHostImport(device, hostPtr, (size_t)ext.size))
            CV_Error_(Error::StsError, ("accel(%s): this host memory shares pages with memory that is already "
                                        "imported (pinning works on whole pages); copy it into a UMat instead",
                                        b->name().c_str()));
        out.guard.reservation = hostPtr;
    }
    cvAccelStatus status = b->fn().import_memory(device, &ext, &out.alloc);
    if (status != CV_ACCEL_OK)
        memset(&out.alloc, 0, sizeof(out.alloc));   // nothing to release
    if (status == CV_ACCEL_NOT_IMPLEMENTED)
        CV_Error_(Error::StsNotImplemented, ("accel(%s): this kind of external memory cannot be imported "
                                             "(type %d); copy it into a UMat instead", b->name().c_str(), (int)ext.type));
    b->check(status, "import_memory", device);
    cvAccelAllocation& a = out.alloc;
    a.desc.backend_id = (uint32_t)b->id();
    a.desc.device_index = (uint32_t)device;
    if (a.size < ext.size)
        CV_Error_(Error::StsInternal, ("accel(%s): the backend imported fewer bytes than asked for", b->name().c_str()));
    if (hostPtr && a.desc.host_access == CV_ACCEL_HOST_ACCESS_NONCOHERENT)
    {
        // cache maintenance works on whole granules: a partial one at either end would write
        // back or drop bytes of the data next to it (backend rule 3)
        const uint64_t g = a.desc.coherence_granule;
        if (g == 0 || (uintptr_t)hostPtr % g != 0 || ext.size % g != 0)
            CV_Error_(Error::StsBadArg, ("accel(%s): host memory imported without cache coherence must start and end "
                                         "on %u-byte boundaries; copy it into a UMat instead",
                                         b->name().c_str(), (unsigned)g));
    }
    if (a.desc.layout_modifier != ext.layout_modifier)
    {
        // a backend that does not know the layout (or predates the field) reports it as linear:
        // using tiled or compressed memory as linear would read garbage
        CV_Error_(Error::StsNotImplemented, ("accel(%s): the backend cannot import memory with layout modifier 0x%llx",
                                             b->name().c_str(), (unsigned long long)ext.layout_modifier));
    }
}

/** The request for all of [objStart, objStart + objSize) of the memory that @p ext reaches at
    @p reqStart (both in the coordinates of the memory's identity). */
static cvAccelExternalMemory wholeRequest(Backend* b, const cvAccelExternalMemory& ext, uint64_t reqStart,
                                          uint64_t objStart, uint64_t objSize)
{
    const uint64_t front = reqStart - objStart;    // bytes of the memory before the request
    cvAccelExternalMemory whole = ext;
    whole.size = objSize;
    if (ext.type == CV_ACCEL_EXTMEM_HOST_POINTER || ext.type == CV_ACCEL_EXTMEM_DEVICE_POINTER)
    {
        // a pointer: the start of the memory
        const uint64_t ptr = (uint64_t)(uintptr_t)((const uchar*)(intptr_t)ext.handle + ext.offset);
        whole.handle = (int64_t)(intptr_t)(uintptr_t)(ptr - front);
        whole.offset = 0;
    }
    else
    {
        // a handle of the whole object (a file descriptor, an NT handle): an offset in it
        if (ext.offset < front)
            CV_Error_(Error::StsInternal, ("accel(%s): the backend reports an identity that this handle cannot reach",
                                           b->name().c_str()));
        whole.offset = ext.offset - front;
    }
    return whole;
}

/** An import of all of [objStart, objStart + objSize): @p candidate when it is one already (an
    import of the requested bytes made to learn their identity), a new one otherwise. */
static std::unique_ptr<FreshImport> importWhole(Backend* b, int device, const cvAccelExternalMemory& ext, uint64_t reqStart,
                                                uint64_t objStart, uint64_t objSize, std::unique_ptr<FreshImport>& candidate)
{
    std::unique_ptr<FreshImport> whole;
    if (candidate && reqStart == objStart && candidate->alloc.size == objSize)
    {
        whole.swap(candidate);
        return whole;
    }
    candidate.reset();      // the partial import is not needed
    whole.reset(new FreshImport(b, device));
    importChecked(b, device, wholeRequest(b, ext, reqStart, objStart, objSize), *whole);
    return whole;
}

//! releases the imports of a backing that nobody else saw
static void dropNewBacking(PhysicalBacking* nb) CV_NOEXCEPT
{
    for (size_t i = 0; i < nb->imports.size(); i++)
    {
        Import* x = nb->imports[i];
        Backend* be = x->backend;
        if ((x->alloc.handle || x->alloc.backend_data) && be->fn().deallocate(&x->alloc) != CV_ACCEL_OK)
            releaseImportError(be);
        if (x->hostReservation)
            be->releaseHostImport(x->device, x->hostReservation);
    }
    delete nb;
}

//! the import of all of @p bk for this backend, device and layout; made now if there is none
static Import* importFor(PhysicalBacking* bk, Backend* b, int device, const cvAccelExternalMemory& ext,
                         uint64_t reqStart, std::unique_ptr<FreshImport>& candidate)
{
    {
        std::lock_guard<std::mutex> lk(bk->mutex);
        for (size_t i = 0; i < bk->imports.size(); i++)
        {
            Import* imp = bk->imports[i];
            if (imp->backend == b && imp->device == device && imp->modifier == ext.layout_modifier)
                return imp;
        }
    }
    // outside of the lock: importing may be slow
    std::unique_ptr<FreshImport> fresh = importWhole(b, device, ext, reqStart, bk->start, bk->size, candidate);
    std::lock_guard<std::mutex> lk(bk->mutex);
    for (size_t i = 0; i < bk->imports.size(); i++)
    {
        Import* imp = bk->imports[i];
        if (imp->backend == b && imp->device == device && imp->modifier == ext.layout_modifier)
            return imp;     // another thread was faster; ours is released
    }
    const cvAccelMemoryDesc& d = fresh->alloc.desc;
    // protection is attested by the backend: every import of one memory must say the same
    if ((d.flags & CV_ACCEL_MEMF_PROTECTED) != (bk->flags & CV_ACCEL_MEMF_PROTECTED))
        CV_Error_(Error::StsBadArg, ("accel(%s): the backends that import this memory disagree on whether it is protected",
                                     b->name().c_str()));
    if (d.layout_modifier == 0)
    {
        for (size_t i = 0; i < bk->imports.size(); i++)
        {
            const cvAccelMemoryDesc& o = bk->imports[i]->alloc.desc;
            if (o.layout_modifier != 0)
                continue;
            // one memory, one way for the CPU to reach it: core opens CPU access once for all imports
            if (o.host_access != d.host_access)
                CV_Error_(Error::StsNotImplemented, ("accel(%s): this memory is already imported with another way for the "
                                                     "CPU to reach it", b->name().c_str()));
            if (d.host_access == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
                CV_Error_(Error::StsNotImplemented, ("accel(%s): memory the CPU may touch only while its device is idle "
                                                     "cannot be shared with another device", b->name().c_str()));
        }
    }
    if (!bk->cpuLevels.empty())
        CV_Error_(Error::StsError, ("accel(%s): the CPU has this memory open (a Mat view of it is alive); release it "
                                    "before importing the memory into another device", b->name().c_str()));
    bk->imports.reserve(bk->imports.size() + 1);
    Import* imp = fresh->commit(b, device, ext.layout_modifier);
    bk->imports.push_back(imp);     // capacity is reserved
    bk->flags |= d.flags & CV_ACCEL_MEMF_EXTERNAL_SYNC;
    if (d.host_access == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
        b->idleGate(device).used = true;
    return imp;
}

UMatData* DeviceAllocator::wrap(const WrapRequest& req, size_t& viewOffset, bool& referenced) const
{
    viewOffset = 0;
    referenced = false;
    const cvAccelExternalMemory& ext = req.ext;
    if (!b_->fn().import_memory)
        CV_Error_(Error::StsNotImplemented, ("accel(%s): the backend cannot import external memory", b_->name().c_str()));
    uint64_t extEnd = 0;
    if (ext.size == 0 || !rangeEnd(ext.offset, ext.size, extEnd))
        CV_Error(Error::StsBadArg, "accel: the external memory range is empty or does not fit in 64 bits");

    // 1. the identity of the memory: known to core, or reported by the backend for an import
    BackingKey key;
    uint64_t reqStart = 0, objStart = 0, objSize = 0;
    std::unique_ptr<FreshImport> first;     // the import that told the identity
    if (!coreIdentity(ext, key, reqStart, objStart, objSize))
    {
        first.reset(new FreshImport(b_, device_));
        importChecked(b_, device_, ext, *first);
        const cvAccelAllocation& a = first->alloc;
        if (a.backing_id)
        {
            uint64_t reqEnd = 0;
            if (!rangeEnd(a.backing_offset, ext.size, reqEnd) || (a.backing_size && reqEnd > a.backing_size))
                CV_Error_(Error::StsBadArg, ("accel(%s): the range lies outside of the allocation it belongs to",
                                             b_->name().c_str()));
            key = BackingKey(BACKING_BACKEND, (uint64_t)b_->id(), a.backing_id);
            reqStart = a.backing_offset;
            objStart = a.backing_size ? 0 : a.backing_offset;
            objSize = a.backing_size ? a.backing_size : ext.size;
        }
        else
        {
            objSize = ext.size;     // nothing identifies it: a backing of its own
        }
    }

    // 2. its backing: a registered one (revived if it waits for its fences), or a new one
    PhysicalBacking* bk = 0;
    Import* imp = 0;
    for (;;)
    {
        bk = key.domain != BACKING_ANONYMOUS ? findBacking(key, reqStart, ext.size) : 0;
        if (bk)
            break;
        std::unique_ptr<FreshImport> whole = importWhole(b_, device_, ext, reqStart, objStart, objSize, first);
        std::unique_ptr<PhysicalBacking> nb(new PhysicalBacking());
        nb->key = key;
        nb->start = objStart;
        nb->size = objSize;
        nb->borrowed = true;
        nb->refs = 1;
        // the backend attests protection; every dma-buf may carry implicit fences of other drivers
        nb->flags = (whole->alloc.desc.flags & (CV_ACCEL_MEMF_PROTECTED | CV_ACCEL_MEMF_EXTERNAL_SYNC)) |
                    (key.domain == BACKING_FILE ? CV_ACCEL_MEMF_EXTERNAL_SYNC : 0);
        nb->imports.reserve(1);
        nb->imports.push_back(whole->commit(b_, device_, ext.layout_modifier));
        PhysicalBacking* got = 0;
        try
        {
            got = registerBacking(nb.get());
        }
        catch (...)
        {
            dropNewBacking(nb.release());
            throw;
        }
        if (got != nb.get())
        {
            // another thread registered the same memory first: its backing is the one
            dropNewBacking(nb.release());
            bk = got;
            break;
        }
        if (nb->imports[0]->alloc.desc.host_access == CV_ACCEL_HOST_ACCESS_DEVICE_IDLE)
            b_->idleGate(device_).used = true;
        bk = nb.release();
        imp = bk->imports[0];
        break;
    }

    // 3. a view of the requested bytes; the backing holds one reference for it, released on failure
    try
    {
        if (!imp)
            imp = importFor(bk, b_, device_, ext, reqStart, first);
        first.reset();
        const cvAccelMemoryDesc& d = imp->alloc.desc;
        if (req.protectedRequired && !(bk->flags & CV_ACCEL_MEMF_PROTECTED))
        {
            // protection is attested by the backend, never taken on the caller's word
            CV_Error_(Error::StsBadArg, ("accel(%s): the backend does not attest that this memory is protected",
                                         b_->name().c_str()));
        }
        const bool readOnly = req.readOnly || (d.flags & CV_ACCEL_MEMF_READ_ONLY) != 0;
        const bool hostCopy = d.host_access == CV_ACCEL_HOST_ACCESS_COPY || d.host_access == CV_ACCEL_HOST_ACCESS_UNCACHED;
        const uint64_t vStart = reqStart - bk->start;

        std::unique_ptr<UMatData> nu(new UMatData(this));
        std::unique_lock<std::mutex> lk(bk->mutex);
        for (;;)
        {
            // No byte has two host copies: bytes that a view with a host copy covers are reached
            // through that view (see BufferState). A view that is going away may still write its
            // host copy back: wait for it.
            BufferState* holder = 0;
            bool wait = false;
            for (size_t i = 0; hostCopy && i < bk->views.size(); i++)
            {
                BufferState* w = bk->views[i];
                if (w->imp != imp || !rangesOverlap(w->start, w->size, vStart, ext.size))
                    continue;
                if (w->dying)
                {
                    wait = true;
                    break;
                }
                if (!req.allowExistingView || !rangeInside(w->start, w->size, vStart, ext.size) || w->readOnly != readOnly)
                    CV_Error_(Error::StsBadArg, ("accel(%s): these bytes are already wrapped as another buffer with its own "
                                                 "host copy%s; wrap the whole memory once and take views of it",
                                                 b_->name().c_str(), w->readOnly != readOnly ? " and other access rights" : ""));
                holder = w;
                break;
            }
            if (wait)
            {
                bk->cond.wait(lk);
                continue;
            }
            if (holder)
            {
                if (!addRefIfAlive(&holder->u->urefcount))
                {
                    bk->cond.wait(lk);  // its last UMat is going right now
                    continue;
                }
                lk.unlock();
                viewOffset = (size_t)(vStart - holder->start);
                referenced = true;
                releaseBacking(bk);     // the view holds the backing already
                return holder->u;
            }
            break;
        }
        nu->size = (size_t)ext.size;
        nu->flags |= UMatData::USER_ALLOCATED;
        BufferState* v = addViewLocked(bk, imp, nu.get(), vStart, ext.size);
        v->readOnly = readOnly;
        // The content belongs to the owner and lives in the memory, which the device addresses.
        // Memory with a CPU view is also the host copy; COPY and UNCACHED memory get their host
        // copy on the first CPU access, memory without CPU access never.
        nu->markDeviceCopyObsolete(false);
        nu->markHostCopyObsolete(hostCopy || d.host_access == CV_ACCEL_HOST_ACCESS_NONE);
        lk.unlock();
        noteBufferCreated();
        return nu.release();
    }
    catch (...)
    {
        releaseBacking(bk);
        throw;
    }
}

}}} // namespace cv::accel::impl

namespace cv { namespace accel { namespace detail {

uint64 backingId(const UMat& m)
{
    return m.u && m.u->accelState ? m.u->accelState->backing->id : 0;
}

}}} // namespace cv::accel::detail
