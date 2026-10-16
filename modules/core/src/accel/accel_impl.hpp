// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_IMPL_HPP
#define OPENCV_CORE_SRC_ACCEL_IMPL_HPP

#include "opencv2/core/accel.hpp"
#include "opencv2/core/llapi/accel_hal.h"
#include "opencv2/core/utils/accel_dispatch.private.hpp"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cv { namespace accel { namespace impl {

class Backend;
struct BufferState;

//////////////////////////////// streams and fences ////////////////////////////////

/** A backend stream owned by core. uid identifies the stream for its whole life, so a
    destroyed stream whose handle gets reused is never mistaken for a live one. */
struct StreamHolder
{
    StreamHolder(Backend* b, int device, cvAccelStream handle);
    ~StreamHolder();
    Backend* backend;
    int device;
    cvAccelStream handle;
    uint64 uid;
};
typedef std::shared_ptr<StreamHolder> StreamPtr;

/** Counted reference to a backend fence, the stream it was recorded on and that stream's device.

An unknown fence stands for work whose end cannot be observed: a failure left no fence behind
and the device could not be synchronized, or a backend refused a reference. It completes once
a device_synchronize of its device that started after it succeeds, or once the backend reports
the device lost (backend rule 8: a lost device has stopped). Until then nothing the work may
touch is freed, and memory that belongs to someone else is not handed back. */
class FenceRef
{
public:
    FenceRef() : b_(0), f_(0), streamUid_(0), device_(-1), syncTicket_(0) {}
    //! adopts the reference returned by stream_record_fence
    FenceRef(Backend* b, cvAccelFence f, uint64 streamUid, int device)
        : b_(b), f_(f), streamUid_(streamUid), device_(device), syncTicket_(0) {}
    //! work on @p device that may still be running and has no fence (see above)
    static FenceRef unknown(Backend* b, int device) CV_NOEXCEPT;
    FenceRef(const FenceRef& o);
    FenceRef& operator=(const FenceRef& o);
    ~FenceRef() { reset(); }

    void reset();
    //! nothing to wait for
    bool empty() const { return f_ == 0 && syncTicket_ == 0; }
    bool isUnknown() const { return syncTicket_ != 0; }
    Backend* backend() const { return b_; }
    cvAccelFence get() const { return f_; }
    uint64 streamUid() const { return streamUid_; }
    int device() const { return device_; }

    //! polls; an empty fence is complete; throws if the device reports an error
    bool isComplete() const;
    /** CPU wait (counted in TransferStats::hostWaits); throws on device errors. With
        @p bounded false the OPENCV_ACCEL_WAIT_TIMEOUT_MS limit does not apply: for waits that
        guard memory going back to its owner (a user buffer, imported memory, a Mat's memory),
        which must never be handed back while the device may still use it. An unknown fence
        synchronizes the device; unbounded, it retries until the device proves it is idle or
        stopped. */
    void wait(bool bounded = true) const;

    enum Poll { DONE, PENDING, FAILED };
    /** Never throws: for release paths, which run from destructors. A failure is logged (and
        DEVICE_LOST marks the device stopped); the work behind the fence no longer runs, so the
        caller may free what it used. An unknown fence stays PENDING until it completes (see
        above): whatever waits for it is leaked rather than freed under the device. */
    Poll poll(bool logFailure = true) const CV_NOEXCEPT;
    //! never throws; false when the device failed (see wait() for @p bounded)
    bool waitNoThrow(bool bounded = true) const CV_NOEXCEPT;

private:
    void waitUnknown(bool bounded) const;
    Backend* b_;
    cvAccelFence f_;
    uint64 streamUid_;
    int device_;
    uint64 syncTicket_;     //!< unknown fence: the first device_synchronize ticket that covers the work
};

/** After a failed enqueue, or a failed fence record after one: the backend may have queued part
    of the work. Returns a fence after it, an empty fence once the device is idle or stopped,
    and an unknown fence (the device is then marked lost) when neither can be established.
    Never throws. */
FenceRef fenceAfterFailure(const StreamPtr& s) CV_NOEXCEPT;

//! the message of the exception being handled (call inside a catch block)
std::string currentExceptionMessage();

//! records a fence at the current end of @p s
FenceRef recordFence(const StreamPtr& s);

//! makes later work on @p s wait for @p f: on the device when both belong to one backend, on the CPU otherwise
void streamWait(const StreamPtr& s, const FenceRef& f);

//////////////////////////////////// backends ////////////////////////////////////

/** Per-device gate for memory the CPU may touch only while the device is idle
    (CV_ACCEL_HOST_ACCESS_DEVICE_IDLE). While the CPU holds such memory, no new device work
    is started on that device: it would make the CPU access fault.

    A handshake on two counters, so that nobody waits while holding a buffer lock: the CPU
    side increments cpuMappings, then waits until no submission is in progress (before it
    locks any buffer); a submission increments submitting, then backs off to the CPU path if
    cpuMappings is not zero. With sequentially consistent atomics at least one side sees the
    other, and the CPU side drains the device (device_synchronize) after its wait. Costs
    nothing until the first allocation of that class on the device. */
struct IdleGate
{
    IdleGate() : used(false), cpuMappings(0), submitting(0) {}
    std::atomic<bool> used;
    std::atomic<int> cpuMappings;
    std::atomic<int> submitting;
    std::mutex mutex;
    std::condition_variable cond;   //!< signalled when submitting drops to zero
    std::vector<std::thread::id> holders;   //!< threads with an open DeviceAccess here (under mutex)
};

/** Submission window of one operation on a device (see IdleGate). */
class SubmitWindow
{
public:
    explicit SubmitWindow(IdleGate& gate);
    ~SubmitWindow() { close(); }
    //! the CPU holds DEVICE_IDLE memory of this device: no device work may start now
    bool blocked() const { return blocked_; }
    void close();
private:
    IdleGate* gate_;
    bool open_;
    bool blocked_;
};

class Backend
{
public:
    Backend(int id, const OpenCV_Core_Accel_API* api, int priority, const std::string& origin);

    int id() const { return id_; }
    const std::string& name() const { return name_; }
    int priority() const { return priority_; }
    const std::string& origin() const { return origin_; }
    //! the backend's entries; those past the size of its table read as NULL
    const OpenCV_Core_Accel_API_v0_0_api_entries& fn() const { return fn_; }

    int deviceCount();
    cvAccelDeviceInfo deviceInfo(int device);

    //! throws cv::Exception with the backend message unless @p st is CV_ACCEL_OK; DEVICE_LOST marks @p device lost
    void check(cvAccelStatus st, const char* what, int device = -1) const;

    //! cached op table of a module, NULL when the backend has none
    const void* opTable(const char* module, uint32_t version);

    //! true when kernels on @p device can use memory described by @p desc in place
    bool canAccess(int device, const cvAccelMemoryDesc& desc);

    //! per-thread default stream of @p device
    StreamPtr defaultStream(int device);

    //! the generic allocator for @p device and memory kind (cvAccelMemoryKind), never released
    //! @p protectedMem: allocations for results computed from protected content
    MatAllocator* allocator(int device, int kind, bool protectedMem = false);

    /** device_synchronize, counted per device: an unknown fence completes once a
        synchronization that started after it succeeded. DEVICE_LOST marks the device stopped.
        Never throws. */
    cvAccelStatus synchronize(int device) CV_NOEXCEPT;
    //! the ticket the next synchronize() of @p device gets
    uint64 nextSyncTicket(int device) const CV_NOEXCEPT
    {
        return syncIssued_[slot(device)].load() + 1;
    }
    //! a synchronize() with a ticket of at least @p ticket succeeded
    bool syncedSince(int device, uint64 ticket) const CV_NOEXCEPT
    {
        return syncDone_[slot(device)].load() >= ticket;
    }

    //! tests only: forgets that devices were lost (accel::detail::reviveDevices)
    void revive() CV_NOEXCEPT
    {
        lost_.store(false);
        stopped_.store(false);
        for (int i = 0; i < MAX_TRACKED_DEVICES; i++)
        {
            lostDevices_[i].store(false);
            stoppedDevices_[i].store(false);
        }
    }
    /** A lost device gets no new work and none of its memory is freed through it (a call may
        fail or block). A device stays lost once lost; a failure without a known device loses
        the whole backend. Lock-free and never throws: called from release paths and check(). */
    bool isLost(int device) const CV_NOEXCEPT
    {
        return lost_.load() || (device >= 0 && device < MAX_TRACKED_DEVICES && lostDevices_[device].load());
    }
    //! core gave up on the device (a timed-out wait, a failure it could not recover from)
    void markLost(int device) CV_NOEXCEPT
    {
        if (device >= 0 && device < MAX_TRACKED_DEVICES)
            lostDevices_[device].store(true);
        else
            lost_.store(true);
    }
    //! the backend reported CV_ACCEL_DEVICE_LOST: the device is lost and has stopped (backend rule 8)
    void markStopped(int device) CV_NOEXCEPT
    {
        markLost(device);
        if (device >= 0 && device < MAX_TRACKED_DEVICES)
            stoppedDevices_[device].store(true);
        else
            stopped_.store(true);
    }
    //! nothing of the device runs or touches memory any more
    bool isStopped(int device) const CV_NOEXCEPT
    {
        return stopped_.load() || (device >= 0 && device < MAX_TRACKED_DEVICES && stoppedDevices_[device].load());
    }

    IdleGate& idleGate(int device);

    /** Whether pinned memory of @p device may hold the CPU's working copy of a buffer. Cleared
        the first time the backend hands out pinned memory that the CPU reaches uncached (older
        Jetson), write-combined or without coherence. */
    bool pinnedShadowAllowed(int device) const CV_NOEXCEPT
    {
        return device < 0 || device >= MAX_TRACKED_DEVICES || !noPinnedShadow_[device].load();
    }
    void disallowPinnedShadow(int device) CV_NOEXCEPT
    {
        if (device >= 0 && device < MAX_TRACKED_DEVICES)
            noPinnedShadow_[device].store(true);
    }

    /** Pinned host blocks for host copies, reused by exact size. Allocating pinned memory is
        slow and stalls the streams of the device, so it should not happen per buffer. */
    bool takePinned(int device, uint64_t size, cvAccelAllocation& out);
    //! keeps the block for reuse, or frees it when the cache is full (OPENCV_ACCEL_PINNED_CACHE_MB)
    void recyclePinned(cvAccelAllocation& a);

    /** Reserves the pages of a host range for an import. Pinning works on whole pages and
        overlapping imports fail or, worse, tear down each other's mappings (Level Zero), so
        a range touching pages that are already imported is refused and the caller copies. */
    bool reserveHostImport(int device, const void* ptr, size_t size);
    void releaseHostImport(int device, const void* ptr);

private:
    void enumerate();

    int id_;
    const OpenCV_Core_Accel_API* api_;
    OpenCV_Core_Accel_API_v0_0_api_entries fn_;     //!< zero-filled copy of the backend's table
    int priority_;
    std::string name_;
    std::string origin_;

    std::mutex mutex_;
    bool enumerated_;
    std::vector<cvAccelDeviceInfo> devices_;
    std::map<std::string, const void*> opTables_;
    std::map<std::pair<int, int>, MatAllocator*> allocators_;
    enum { MAX_TRACKED_DEVICES = 64 };  //!< devices past this are tracked together, as the backend
    static int slot(int device) { return device >= 0 && device < MAX_TRACKED_DEVICES ? device : MAX_TRACKED_DEVICES; }
    std::atomic<bool> lost_, stopped_;
    std::atomic<bool> lostDevices_[MAX_TRACKED_DEVICES];
    std::atomic<bool> stoppedDevices_[MAX_TRACKED_DEVICES];
    std::atomic<bool> noPinnedShadow_[MAX_TRACKED_DEVICES];
    std::atomic<uint64> syncIssued_[MAX_TRACKED_DEVICES + 1];  //!< synchronize() tickets handed out
    std::atomic<uint64> syncDone_[MAX_TRACKED_DEVICES + 1];    //!< highest ticket whose synchronize() succeeded
    std::vector<std::unique_ptr<IdleGate> > idleGates_;
    std::map<std::pair<int, uintptr_t>, uintptr_t> hostImports_;   //!< (device, first page) -> end
    std::multimap<std::pair<int, uint64_t>, cvAccelAllocation> pinnedCache_;
    uint64_t pinnedCacheBytes_;
};

class Registry
{
public:
    static Registry& getInstance();

    //! registers a backend table; returns its id (> 0), or 0 when the table is rejected
    int add(const OpenCV_Core_Accel_API* api, int priority, const std::string& origin);

    bool isActive() const { return active_.load(std::memory_order_acquire); }
    Backend* get(int id);                       //!< NULL for 0 and unknown ids
    Backend* find(const std::string& name);
    std::vector<Backend*> all();                //!< by priority, highest first

    //! loads the libraries listed in OPENCV_ACCEL_PLUGINS (';'-separated)
    void loadPlugins();

private:
    Registry();

    std::mutex mutex_;
    std::vector<Backend*> backends_;            //!< never shrinks; index = id - 1
    std::atomic<bool> active_;
};

//! loads built-in backends and plugins once; cheap afterwards
void ensureInitialized();

//////////////////////////////// execution scope ////////////////////////////////

struct ScopeEntry
{
    int backendId;
    int device;
    StreamPtr stream;   //!< empty: the device's per-thread default stream
};

//! innermost ExecutionScope of the calling thread, NULL outside of any scope
const ScopeEntry* currentScope();

//! allocator of the innermost scope's device, NULL when there is none (used by UMat::getStdAllocator)
MatAllocator* scopeAllocator();

//////////////////////////////// physical memory ////////////////////////////////

//! overflow-safe end of [begin, begin + size): false when it does not fit in 64 bits
inline bool rangeEnd(uint64_t begin, uint64_t size, uint64_t& end)
{
    if (size > UINT64_MAX - begin)
        return false;
    end = begin + size;
    return true;
}

//! [b0, b0 + s0) and [b1, b1 + s1) share a byte (ranges known to fit, see rangeEnd)
inline bool rangesOverlap(uint64_t b0, uint64_t s0, uint64_t b1, uint64_t s1)
{
    return s0 != 0 && s1 != 0 && b0 < b1 + s1 && b1 < b0 + s0;
}

//! [b1, b1 + s1) lies inside [b0, b0 + s0) (ranges known to fit, see rangeEnd)
inline bool rangeInside(uint64_t b0, uint64_t s0, uint64_t b1, uint64_t s1)
{
    return b1 >= b0 && b1 - b0 <= s0 && s1 <= s0 - (b1 - b0);
}

/** Where the identity of a piece of physical memory comes from. Core identifies host memory
    and file-backed memory (dma-bufs) itself, so that every backend that imports them meets the
    same state; other memory is identified by the backend that reaches it. */
enum BackingDomain
{
    BACKING_ANONYMOUS = 0,  //!< not known: the memory is reached through one allocation only
    BACKING_HOST = 1,       //!< host memory: virtual addresses of this process
    BACKING_FILE = 2,       //!< a dma-buf or another file-backed object: (st_dev, st_ino)
    BACKING_BACKEND = 3     //!< reported by one backend: (backend id, cvAccelAllocation::backing_id)
};

struct BackingKey
{
    BackingKey() : domain(BACKING_ANONYMOUS), a(0), b(0) {}
    BackingKey(int d, uint64_t a_, uint64_t b_) : domain(d), a(a_), b(b_) {}
    int domain;
    uint64_t a, b;
    bool operator<(const BackingKey& o) const
    {
        return domain != o.domain ? domain < o.domain : a != o.a ? a < o.a : b < o.b;
    }
};

/** One backend's import of a whole backing, or the allocation that a backing is. Byte 0 of
    alloc is byte 0 of the backing. A backing has one import per backend, device, layout and
    access (read-only or not); views of the backing use ranges of it. Imports live as long as
    their backing. */
struct Import
{
    Import();
    Backend* backend;
    int device;
    uint64_t modifier;              //!< the layout it was imported with
    bool readOnly;                  //!< imported for reading only (CV_ACCEL_MEMF_READ_ONLY asked for)
    cvAccelAllocation alloc;        //!< desc.backend_id and desc.device_index are set by core
    const void* hostReservation;    //!< host pages reserved with Backend::reserveHostImport, or NULL
};

//! bytes of a backing that failed device work may have written: readers get its error
struct PoisonRange
{
    uint64_t begin, end;
    std::string reason;
};

//! host memory of a released view, freed once the device work that may touch it completed
struct RetiredHostCopy
{
    RetiredHostCopy() : shadow(0), kind(0), pages(0), backend(0), device(-1) { memset(&alloc, 0, sizeof(alloc)); }
    uchar* shadow;
    int kind;                   //!< ShadowKind
    size_t pages;
    cvAccelAllocation alloc;    //!< pinned memory from the backend
    Backend* backend;
    int device;
    FenceRef writeFence;
    std::vector<FenceRef> readFences;
};

/** Everything core knows about one piece of physical memory, whoever allocated or imported it
    and however many UMatData objects view it: the order of device work on it, which of its
    bytes failed work may have written, who uses it right now (device operations, Mat views,
    DeviceAccess), its imports, and its lifetime. Two planes of one dma-buf, the same dma-buf
    imported by a GPU and an NPU backend, and a UMat's memory wrapped again through its device
    pointer are all one backing.

    Ordering is per backing, not per byte: device work on one plane waits for conflicting
    work on another plane of the same memory, as with the implicit fences of a dma-buf, which
    cover the whole buffer. */
struct PhysicalBacking
{
    PhysicalBacking();
    ~PhysicalBacking();

    // identity, fixed before the backing is shared
    uint64 id;              //!< unique in the process (diagnostics and tests)
    BackingKey key;
    uint64_t start;         //!< first byte in the key's address space (a host address, an offset in the file, ...)
    uint64_t size;
    uint32_t flags;         //!< attested by its imports: CV_ACCEL_MEMF_PROTECTED, CV_ACCEL_MEMF_EXTERNAL_SYNC
    bool borrowed;          //!< memory of someone else: its last release waits for the device without limit

    // under the mutex of the registry of backings (see backing.cpp)
    int refs;               //!< live views, and lookups in progress
    bool registered;        //!< findable through its key
    bool retiring;          //!< no reference left, being freed: lookups wait until it is gone
    bool deferred;          //!< no reference left, waiting for its fences

    // everything below is protected by mutex
    std::mutex mutex;
    std::condition_variable cond;
    bool busy;              //!< a host transition that waits outside of the mutex is running
    int pendingReaders;     //!< device operations between acquire and record that only read
    int pendingWriters;     //!< ... that write; a writer excludes everyone, a reader excludes writers
    struct Hold
    {
        std::thread::id thread;
        int mode;
    };
    std::vector<Hold> longHolds;        //!< open DeviceAccess objects, to report same-thread conflicts
    FenceRef writeFence;                //!< last device write
    std::vector<std::pair<uint64_t, uint64_t> > writeRanges;   //!< [begin, end) it covers: poisoned if it fails
    std::vector<FenceRef> readFences;   //!< device reads after the last write, one per stream
    std::vector<PoisonRange> poison;    //!< bytes that failed work may have written, until written whole
    uint64 generation;                  //!< incremented by every write
    std::vector<Import*> imports;       //!< owned
    std::vector<BufferState*> views;    //!< live views
    std::vector<RetiredHostCopy> retired;
    // CPU access. Memory that the CPU addresses directly is opened once for all of its views:
    // begin_cpu_access on every import that has a CPU view, nested for wider access, all ended
    // when the last view closes. Cache maintenance then never runs under another view's
    // unflushed writes (a dma-buf sync covers the whole buffer).
    int cpuViews;                       //!< views with an open Mat mapping
    std::vector<int> cpuLevels;         //!< access of each open begin_cpu_access level, outermost first
    uchar* cpuBase;                     //!< CPU address of byte 0 while levels are open
    IdleGate* idleGateHeld;             //!< DEVICE_IDLE memory: the gate the open mapping holds
    /** end_cpu_access calls that failed: the CPU access counts as open until a retry succeeds
        (backend rule 4), so no device work may use the memory meanwhile, and the device gate of
        DEVICE_IDLE memory stays held */
    std::vector<std::pair<Import*, int> > failedCpuEnds;
};

//! an image wrapped by wrapExternalImage: what operations see of its planes (cvAccelView::image)
struct ImageRecord
{
    cvAccelImageDesc desc;
};

//! host copy kinds (BufferState::shadowKind)
enum ShadowKind
{
    SHADOW_NONE = 0,
    SHADOW_FASTMALLOC = 1,
    SHADOW_BACKEND = 2,     //!< pinned host memory from the backend: one DMA per transfer
    SHADOW_USER = 3,        //!< the Mat that the UMat was created from (Mat::getUMat)
    SHADOW_PAGES = 4        //!< core's own pages, read-only except during downloads (read-only memory)
};

/** A view: the state of one UMatData, a range of a backing reached through one import.

Which copy is current is kept in UMatData::flags as for other allocators:
HOST_COPY_OBSOLETE (the CPU view is stale) and DEVICE_COPY_OBSOLETE (the device copy is
stale). Both set means the content is uninitialized and no transfer is ever needed. For
memory with a single physical copy (COHERENT, NONCOHERENT, ...), HOST_COPY_OBSOLETE means the
CPU must wait and invalidate before reading.

A host copy (COPY and UNCACHED memory) belongs to one view. Views of one import with host
copies never overlap: a wrap of bytes that a view with a host copy already covers becomes a
UMat over that view, so no byte has two host copies. Views of memory that the CPU addresses
directly may overlap freely. Everything here but the identity fields is protected by
backing->mutex. */
struct BufferState
{
    BufferState(PhysicalBacking* b, Import* imp, uint64_t start, uint64_t size);

    PhysicalBacking* backing;
    Import* imp;
    UMatData* u;                    //!< the UMatData of the view
    uint64_t start;                 //!< first byte in the backing (and in imp->alloc)
    uint64_t size;                  //!< bytes, UMatData::size
    bool readOnly;                  //!< writes refused: read-only memory, or the wrap asked for it
    bool fromMat;                   //!< created by Mat::getUMat: the content goes back to the Mat
    std::shared_ptr<const ImageRecord> image;   //!< the image the view is a plane of (wrapExternalImage), or none
    int plane;                      //!< index of that plane

    Backend* backend() const { return imp->backend; }
    int device() const { return imp->device; }
    const cvAccelAllocation& alloc() const { return imp->alloc; }

    //! host copy for HOST_ACCESS_COPY and HOST_ACCESS_UNCACHED memory; pinned when the backend can
    uchar* shadow;
    int shadowKind;                 //!< ShadowKind
    cvAccelAllocation shadowAlloc;  //!< valid for SHADOW_BACKEND
    size_t shadowPages;             //!< SHADOW_PAGES: size of the page mapping

    bool hostMapped;                //!< Mat views may exist (UMatData::refcount was > 0)
    int hostAccess;                 //!< union of AccessFlag over the current mapping
    /** The current mapping shows a copy of the content in pages core maps read-only, so that a
        write through it faults: a read-only view of memory without a read-only CPU mapping, and
        every view opened for reading under OPENCV_ACCEL_VALIDATE. */
    uchar* cpuCopy;
    size_t cpuCopyPages;
    bool cpuCopyWriteBack;          //!< the mapping was opened for writing since: the copy goes back at its end
    bool dying;                     //!< no UMat refers to the view any more: it is being released
    uint64 matGeneration;           //!< fromMat: the backing generation of the content copied from the Mat
};

//! UMat::getMat() for buffers with accelState
Mat getHostView(const UMat& m, AccessFlag access);
//! MatAllocator::unmap() for buffers with accelState (the last Mat view is gone)
void endHostView(UMatData* u);
//! UMat::handle() for buffers with accelState: waits for everything and returns the device handle
void* deviceHandleBlocking(UMatData* u, AccessFlag access);

enum DeviceMode
{
    DEV_READ = 1,
    DEV_WRITE = 2,      //!< partial write: the rest of the buffer must stay intact
    DEV_DISCARD = 4     //!< together with DEV_WRITE: the whole view is overwritten
};

struct DeviceUse
{
    UMatData* u;
    int mode;           //!< DEV_* flags
};

/** The buffers one piece of device work uses, acquired together.

acquire() takes their backings in one global order (by address), so concurrent acquirers never
wait for each other in a cycle; views of one backing are acquired together, with the union of
their modes, and a buffer added twice counts once. Each backing waits while a conflicting
operation sits between acquire and record, its views get their host copies uploaded when
those are newer, and it contributes the fences the work must wait for. If anything fails, or
the object goes away before record(), every backing acquired so far is released, so no path
leaves one acquired. A long hold (DeviceAccess) is remembered per thread, so that the same
thread touching the memory again gets an error instead of a deadlock. */
class DeviceAcquisition
{
public:
    explicit DeviceAcquisition(bool longHold = false) : longHold_(longHold), acquired_(0) {}
    ~DeviceAcquisition() { abandon(); }

    void add(UMatData* u, int mode);
    bool empty() const { return uses_.empty(); }

    /** Returns false and holds nothing when a live Mat view conflicts with a use. Appends the
        fences that work enqueued on @p stream must wait for to @p waits. */
    bool acquire(const StreamPtr& stream, std::vector<FenceRef>& waits);
    //! the work was enqueued: @p fence completes it; releases everything
    void record(const FenceRef& fence);
    /** The work failed after the backend may have queued part of it: every backing takes
        @p fence (empty: nothing runs any more) and the views written are poisoned; releases
        everything. */
    void fail(const FenceRef& fence, const std::string& reason);
    //! nothing was enqueued; releases everything (safe to call more than once)
    void abandon();

private:
    DeviceAcquisition(const DeviceAcquisition&);
    DeviceAcquisition& operator=(const DeviceAcquisition&);
    struct Group
    {
        PhysicalBacking* backing;
        size_t first, count;    //!< uses_[first, first + count) are views of it
        int mode;               //!< union of their DEV_READ / DEV_WRITE
    };
    void finish(const Group& g, const FenceRef* fence, const char* failure);
    //! the views of memory with implicit fences, one per import, with the access of all their uses
    std::vector<std::pair<const BufferState*, int> > externalUses() const;
    void endExternalUses(const FenceRef& fence) CV_NOEXCEPT;
    std::vector<DeviceUse> uses_;
    std::vector<Group> groups_;
    bool longHold_;
    size_t acquired_;           //!< groups acquired so far
    std::thread::id holder_;
};

//! what a wrap of external memory asks for (DeviceAllocator::wrap)
struct WrapRequest
{
    WrapRequest() : readOnly(false), protectedRequired(false), allowExistingView(false), plane(-1)
    {
        memset(&ext, 0, sizeof(ext));
        ext.struct_size = (uint32_t)sizeof(ext);
    }
    cvAccelExternalMemory ext;  //!< handle, offset and size of the bytes to view, layout
    bool readOnly;              //!< the caller restricts itself to reading
    bool protectedRequired;     //!< refused unless the backend attests protected content
    /** the result may be a UMat over a view that already covers the bytes with a host copy
        (otherwise such a wrap is refused, see BufferState) */
    bool allowExistingView;
    std::shared_ptr<const ImageRecord> image;   //!< the view is plane @p plane of this image
    int plane;
};

/** The allocator for one (backend, device, memory kind). It owns no state of its own: the
    coherence protocol lives in the functions above, so every backend gets the same rules. */
class DeviceAllocator CV_FINAL : public MatAllocator
{
public:
    DeviceAllocator(Backend* b, int device, int kind, bool protectedMem = false);

    UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step,
                       AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE;
    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE;
    void deallocate(UMatData* u) const CV_OVERRIDE;
    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE;
    void unmap(UMatData* u) const CV_OVERRIDE;
    void download(UMatData* u, void* dst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[], const size_t dststep[]) const CV_OVERRIDE;
    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[], const size_t srcstep[]) const CV_OVERRIDE;
    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool sync) const CV_OVERRIDE;
    bool getMemoryDesc(const UMatData* u, cvAccelMemoryDesc* desc) const CV_OVERRIDE;

    Backend* backend() const { return b_; }
    int device() const { return device_; }
    int kind() const { return kind_; }

    /** Wraps external memory (wrapExternalMemory). Returns the UMatData to view it through and
        the offset of the requested bytes in it. With @p referenced the UMatData is an existing
        view, and one more UMat reference is taken on it; otherwise it is new and unreferenced. */
    UMatData* wrap(const WrapRequest& req, size_t& viewOffset, bool& referenced) const;

private:
    int kindFor(UMatUsageFlags usageFlags) const;
    //! memory for @p total bytes from the backend (deferred releases collected once on OUT_OF_MEMORY)
    cvAccelAllocation allocateMemory(uint64_t total, int kind) const;

    Backend* b_;
    int device_;
    int kind_;
    bool protected_;    //!< allocates protected memory (CV_ACCEL_MEMF_PROTECTED)
};

//! takes one more reference unless the count already dropped to zero (the owner is releasing)
bool addRefIfAlive(int* counter);

/** A view of all of the memory core just allocated in @p a (allocate, Mat::getUMat), set on
    @p u. Takes ownership of @p a, also on failure. */
BufferState* attachOwnedMemory(UMatData* u, Backend* b, int device, const cvAccelAllocation& a);

/** Releases a view whose UMatData is gone: its host copy is freed once the device work that may
    touch it completed, and the backing loses a reference (see releaseBacking). Never throws. */
void releaseView(BufferState* v) CV_NOEXCEPT;

/** Drops a reference to a backing. The last one retires it: memory that belongs to someone
    else waits right here, without limit, until the device is done with it; core's own memory
    is freed once its fences complete (collectRetired). Until then the backing stays findable,
    so a wrap of the same memory revives it, fences included. Never throws. */
void releaseBacking(PhysicalBacking* b) CV_NOEXCEPT;

//! frees retired backings whose fences completed; with @p waitAll, waits for them first (bounded)
void collectRetired(bool waitAll) CV_NOEXCEPT;

//! release paths: the backing's fences all completed (a failed device counts as done)
bool fencesSettledLocked(PhysicalBacking* b) CV_NOEXCEPT;

//! frees the host copies of released views whose fences completed (with the backing's mutex held)
void pruneRetiredLocked(PhysicalBacking* b) CV_NOEXCEPT;

//! frees one host copy (fences complete, or the device stopped)
void freeHostCopy(RetiredHostCopy& c) CV_NOEXCEPT;

//! calls the failed end_cpu_access of @p b again; true when none is left (backing mutex held)
bool retryFailedCpuEndsLocked(PhysicalBacking* b) CV_NOEXCEPT;

//! poisoned bytes of @p b overlapping [begin, begin + size), NULL when there are none (backing mutex held)
const PoisonRange* findPoison(const PhysicalBacking* b, uint64_t begin, uint64_t size);
//! marks [begin, begin + size) of @p b as written by failed work (backing mutex held)
void addPoison(PhysicalBacking* b, uint64_t begin, uint64_t size, const std::string& reason);
//! [begin, begin + size) was written whole: earlier failures there no longer matter (backing mutex held)
void clearPoison(PhysicalBacking* b, uint64_t begin, uint64_t size);

//! true for allocators of accelerator backends: their data must never move elsewhere silently
bool isDeviceAllocator(const MatAllocator* a);

/** _OutputArray::createSameSize / fitSameSize: while it lives, an output UMat without an
    allocator of its own is created on the device that holds @p model, so a function without an
    accelerator hook keeps its data on the device even when it runs on the CPU. */
class SamePlacement
{
public:
    SamePlacement(const _OutputArray& out, const _InputArray& model);
    ~SamePlacement();
private:
    SamePlacement(const SamePlacement&);
    SamePlacement& operator=(const SamePlacement&);
    UMat* out_;
};

//! both buffers live on the same device of the same accelerator backend
bool sameAccelDevice(const UMatData* a, const UMatData* b);

//! live device buffers keep the hook in cv:: functions active (see detail::isActive)
void noteBufferCreated();
void noteBufferDestroyed();

//! describes @p u through its allocator; false when the allocator does not know
bool queryMemoryDesc(const UMatData* u, cvAccelMemoryDesc& desc);

//! host memory description used for Mat and host UMat arguments
cvAccelMemoryDesc hostMemoryDesc();

//! true when the bytes covered by @p v lie inside its allocation (overflow-safe)
bool viewFitsAllocation(const cvAccelView& v);

///////////////////////////////////// stats /////////////////////////////////////

struct Counters
{
    std::atomic<uint64> bytesH2D, bytesD2H, bytesD2D, transfers, implicitTransfers,
                        deviceOps, cpuFallbacks, hostWaits;
    Counters();
};
Counters& counters();

//! applies the TransferPolicy to an implicit transfer; throws for TRANSFER_FORBID
void onImplicitTransfer(const char* what, const char* opName);

///////////////////////////////////// built-in backends /////////////////////////////////////

const OpenCV_Core_Accel_API* emulatorBackendApi();
//! the emulator as a backend of API version 1 (no begin_device_access / end_device_access), for tests
const OpenCV_Core_Accel_API* emulatorBackendApiV1();
#ifdef HAVE_CUDA
const OpenCV_Core_Accel_API* cudaBackendApi();
#endif

}}} // namespace cv::accel::impl

#endif // OPENCV_CORE_SRC_ACCEL_IMPL_HPP
