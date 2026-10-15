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

/** Counted reference to a backend fence, the stream it was recorded on and that stream's device. */
class FenceRef
{
public:
    FenceRef() : b_(0), f_(0), streamUid_(0), device_(-1) {}
    //! adopts the reference returned by stream_record_fence
    FenceRef(Backend* b, cvAccelFence f, uint64 streamUid, int device)
        : b_(b), f_(f), streamUid_(streamUid), device_(device) {}
    FenceRef(const FenceRef& o);
    FenceRef& operator=(const FenceRef& o);
    ~FenceRef() { reset(); }

    void reset();
    bool empty() const { return f_ == 0; }
    Backend* backend() const { return b_; }
    cvAccelFence get() const { return f_; }
    uint64 streamUid() const { return streamUid_; }
    int device() const { return device_; }

    //! polls; an empty fence is complete; throws if the device reports an error
    bool isComplete() const;
    /** CPU wait (counted in TransferStats::hostWaits); throws on device errors. With
        @p bounded false the OPENCV_ACCEL_WAIT_TIMEOUT_MS limit does not apply: for waits that
        guard memory going back to its owner (a user buffer, imported memory, a Mat's memory),
        which must never be handed back while the device may still use it. */
    void wait(bool bounded = true) const;

    enum Poll { DONE, PENDING, FAILED };
    /** Never throws: for release paths, which run from destructors. A failure is logged (and
        DEVICE_LOST marks the device lost); the work behind the fence no longer runs, so the
        caller may free what it used. */
    Poll poll() const CV_NOEXCEPT;
    //! never throws; false when the device failed (see wait() for @p bounded)
    bool waitNoThrow(bool bounded = true) const CV_NOEXCEPT;

private:
    Backend* b_;
    cvAccelFence f_;
    uint64 streamUid_;
    int device_;
};

/** After a failed enqueue, or a failed fence record after one: the backend may have queued part
    of the work. Returns a fence after it, or an empty fence once the device is idle (or lost,
    when even that fails). Never throws. */
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

    //! takes ownership of a released buffer whose device work may still be running
    void deferRelease(BufferState* st);
    //! frees deferred buffers whose fences completed; with @p waitAll, waits for all of them
    void collectDeferred(bool waitAll);

    /** A device stays lost once lost; a failure without a known device loses the whole backend.
        Lock-free and never throws: called from release paths and from check(). */
    /** Live buffers by the identity of their physical memory (cvAccelAllocation::backing_id),
        so that aliases of one memory share one buffer and one synchronization state. */
    void registerBacking(BufferState* st, UMatData* u);
    //! runs while @p st's UMatData is still alive (deallocate), before anything is freed
    void unregisterBacking(const BufferState* st);
    /** The live buffer whose range holds [offset, offset+size) of @p id, or NULL; with
        @p reference, one more UMat reference is taken on it. Throws when the range overlaps a
        live buffer only partly, or when the buffer lacks @p requiredFlags (read-only, protected). */
    UMatData* findBacking(uint64_t id, uint64_t offset, uint64_t size, uint32_t requiredFlags, bool reference,
                          size_t& viewOffset);

    //! tests only: forgets that devices were lost (accel::detail::reviveDevices)
    void revive() CV_NOEXCEPT
    {
        lost_.store(false);
        for (int i = 0; i < MAX_TRACKED_DEVICES; i++)
            lostDevices_[i].store(false);
    }
    bool isLost(int device) const CV_NOEXCEPT
    {
        return lost_.load() || (device >= 0 && device < MAX_TRACKED_DEVICES && lostDevices_[device].load());
    }
    void markLost(int device) CV_NOEXCEPT
    {
        if (device >= 0 && device < MAX_TRACKED_DEVICES)
            lostDevices_[device].store(true);
        else
            lost_.store(true);
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
    std::vector<BufferState*> deferred_;
    enum { MAX_TRACKED_DEVICES = 64 };  //!< devices past this are tracked together, as the backend
    std::atomic<bool> lost_;
    std::atomic<bool> lostDevices_[MAX_TRACKED_DEVICES];
    std::atomic<bool> noPinnedShadow_[MAX_TRACKED_DEVICES];
    struct BackingEntry
    {
        BufferState* st;
        UMatData* u;
        uint64_t offset;
        uint64_t size;
    };
    std::mutex backingsMutex_;
    std::multimap<uint64_t, BackingEntry> backings_;
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

//////////////////////////////// buffer state and protocol ////////////////////////////////

/** State that core keeps for every UMatData created by a DeviceAllocator.

Which copy is current is kept in UMatData::flags as for other allocators:
HOST_COPY_OBSOLETE (the CPU view is stale) and DEVICE_COPY_OBSOLETE (the device copy is
stale). Both set means the content is uninitialized and no transfer is ever needed.
For memory with a single physical copy (COHERENT, NONCOHERENT, ...), HOST_COPY_OBSOLETE
means the CPU must wait and invalidate before reading. */
struct BufferState
{
    BufferState(Backend* b, int device);

    Backend* backend;
    int device;
    cvAccelAllocation alloc;        //!< the memory UMat::handle() refers to

    //! host copy for HOST_ACCESS_COPY and HOST_ACCESS_UNCACHED memory; pinned when the backend can
    uchar* shadow;
    cvAccelAllocation shadowAlloc;  //!< valid when the shadow came from the backend (pinned)
    bool shadowFromBackend;
    size_t shadowPages;             //!< > 0: the shadow is core's own page mapping of that size (read-only memory)

    std::mutex mutex;
    std::condition_variable cond;
    bool busy;                      //!< a host transition that waits outside of the mutex is running
    int pendingReaders;             //!< device operations between acquire and record that only read
    int pendingWriters;             //!< ... that write; a writer excludes everyone, a reader excludes writers
    struct Hold
    {
        std::thread::id thread;
        int mode;
    };
    std::vector<Hold> longHolds;    //!< open DeviceAccess objects, to report same-thread conflicts
    bool hostMapped;                //!< Mat views may exist (UMatData::refcount was > 0)
    int hostAccess;                 //!< union of AccessFlag over the current mapping
    FenceRef writeFence;            //!< last device write
    std::vector<FenceRef> readFences; //!< device reads after the last write, one per stream

    bool imported;                  //!< memory belongs to the user: releasing waits for the device
    bool fromMat;                   //!< created by Mat::getUMat: the content goes back to the Mat
    const void* hostImport;         //!< imported host range reserved with Backend::reserveHostImport
    bool idleGateHeld;              //!< the current mapping counts in IdleGate::cpuMappings
    bool poisoned;                  //!< the device work that wrote the content failed
    std::string poisonReason;
    uint64 generation;              //!< incremented by every write
    uint64 matGeneration;           //!< fromMat: the generation of the content copied from the Mat
    uint64 viewHash;                //!< OPENCV_ACCEL_VALIDATE: checksum of a read-only view when it opened
    bool viewHashed;
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
    DEV_DISCARD = 4     //!< together with DEV_WRITE: the whole allocation is overwritten
};

struct DeviceUse
{
    UMatData* u;
    int mode;           //!< DEV_* flags
};

/** The buffers one piece of device work uses, acquired together.

acquire() takes them in one global order (by state address), so concurrent acquirers never
wait for each other in a cycle, and a buffer added twice is acquired once with the union of
the modes. Each buffer waits while a conflicting operation sits between acquire and record,
gets its host copy uploaded when that is newer, and contributes the fences the work must
wait for. If anything fails, or the object goes away before record(), every buffer acquired
so far is released, so no path leaves a buffer acquired. A long hold (DeviceAccess) is
remembered per thread, so that the same thread touching the buffer again gets an error
instead of a deadlock. */
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
    //! the work was enqueued: @p fence completes it; releases every buffer
    void record(const FenceRef& fence);
    /** The work failed after the backend may have queued part of it: every buffer takes
        @p fence (empty: nothing runs any more) and the written ones are poisoned; releases
        every buffer. */
    void fail(const FenceRef& fence, const std::string& reason);
    //! nothing was enqueued; releases every buffer (safe to call more than once)
    void abandon();

private:
    DeviceAcquisition(const DeviceAcquisition&);
    DeviceAcquisition& operator=(const DeviceAcquisition&);
    std::vector<DeviceUse> uses_;
    bool longHold_;
    size_t acquired_;
    std::thread::id holder_;
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

    /** Wraps external memory (wrapExternalMemory). When the memory is a range of a buffer that
        core already holds, returns that buffer with one more UMat reference and its offset in
        @p viewOffset; without @p viewOffset such an alias is refused. */
    UMatData* wrap(const cvAccelExternalMemory& ext, size_t total, size_t* viewOffset = NULL) const;

private:
    UMatData* createState(UMatData* u, const cvAccelAllocation& a, bool imported) const;
    int kindFor(UMatUsageFlags usageFlags) const;

    Backend* b_;
    int device_;
    int kind_;
    bool protected_;    //!< allocates protected memory (CV_ACCEL_MEMF_PROTECTED)
};

//! frees the backend memory of a released buffer and deletes @p st (fences must be complete)
void destroyBufferState(BufferState* st);

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
#ifdef HAVE_CUDA
const OpenCV_Core_Accel_API* cudaBackendApi();
#endif

}}} // namespace cv::accel::impl

#endif // OPENCV_CORE_SRC_ACCEL_IMPL_HPP
