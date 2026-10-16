// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: backend registry, streams and fences, execution scopes and the public
// cv::accel API.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/core/utils/plugin_loader.private.hpp>

#include <chrono>
#include <limits>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace cv {

// implemented in umatrix.cpp
void setSize(UMat& m, int _dims, const int* _sz, const size_t* _steps, bool autoSteps);
void finalizeHdr(UMat& m);

namespace accel { namespace impl {

//////////////////////////////// activity, counters, policy ////////////////////////////////

// live device buffers plus live execution scopes, over all threads: while this is zero the
// hook in cv:: functions costs one atomic load
static std::atomic<long long>& activity()
{
    static std::atomic<long long>* a = new std::atomic<long long>(0);
    return *a;
}

static std::atomic<int>& scopeCount()
{
    static std::atomic<int>* c = new std::atomic<int>(0);
    return *c;
}

static std::atomic<int>& oclSuppressCount()
{
    static std::atomic<int>* c = new std::atomic<int>(0);
    return *c;
}

Counters::Counters()
    : bytesH2D(0), bytesD2H(0), bytesD2D(0), transfers(0), implicitTransfers(0),
      deviceOps(0), cpuFallbacks(0), hostWaits(0)
{}

Counters& counters()
{
    static Counters* c = new Counters();
    return *c;
}

static int initialTransferPolicy()
{
    const std::string s = utils::getConfigurationParameterString("OPENCV_ACCEL_TRANSFER_POLICY", "allow");
    if (s == "warn")
        return TRANSFER_WARN;
    if (s == "forbid")
        return TRANSFER_FORBID;
    return TRANSFER_ALLOW;
}

static std::atomic<int>& transferPolicy()
{
    static std::atomic<int>* p = new std::atomic<int>(initialTransferPolicy());
    return *p;
}

void onImplicitTransfer(const char* what, const char* opName)
{
    counters().implicitTransfers++;
    const int policy = transferPolicy().load();
    if (policy == TRANSFER_FORBID)
        CV_Error_(Error::StsError, ("accel: %s: %s; implicit transfers are forbidden "
                                    "(cv::accel::setTransferPolicy / OPENCV_ACCEL_TRANSFER_POLICY)",
                                    opName ? opName : "?", what));
    if (policy == TRANSFER_WARN)
        CV_LOG_WARNING(NULL, "accel: " << (opName ? opName : "?") << ": " << what);
}

//////////////////////////////////// streams and fences ////////////////////////////////////

static std::atomic<uint64>& streamUids()
{
    static std::atomic<uint64>* c = new std::atomic<uint64>(0);
    return *c;
}

StreamHolder::StreamHolder(Backend* b, int device_, cvAccelStream handle_)
    : backend(b), device(device_), handle(handle_), uid(++streamUids())
{}

StreamHolder::~StreamHolder()
{
    if (handle && !cv::__termination)
    {
        if (backend->fn().stream_destroy(handle) != CV_ACCEL_OK)
            CV_LOG_ERROR(NULL, "accel(" << backend->name() << "): stream_destroy failed: " << backend->fn().get_last_error());
    }
}

FenceRef FenceRef::unknown(Backend* b, int device) CV_NOEXCEPT
{
    FenceRef r;
    r.b_ = b;
    r.device_ = device;
    r.syncTicket_ = b->nextSyncTicket(device);  // a synchronization that starts from now on covers the work
    return r;
}

//! takes a reference to @p f for a copy; when the backend refuses one, the copy tracks the work as unknown
static void retainOrForget(Backend* b, cvAccelFence f, int device, cvAccelFence& outFence, uint64& outTicket) CV_NOEXCEPT
{
    outFence = 0;
    outTicket = 0;
    if (b->fn().fence_retain(f) == CV_ACCEL_OK)
    {
        outFence = f;
        return;
    }
    // holding the pointer without a reference would release it twice later
    outTicket = b->nextSyncTicket(device);
    try
    {
        CV_LOG_ERROR(NULL, "accel(" << b->name() << "): fence_retain failed: " << b->fn().get_last_error()
                     << "; the work is tracked by synchronizing device " << device);
    }
    catch (...) {}
}

FenceRef::FenceRef(const FenceRef& o)
    : b_(o.b_), f_(0), streamUid_(o.streamUid_), device_(o.device_), syncTicket_(o.syncTicket_)
{
    if (o.f_)
        retainOrForget(b_, o.f_, device_, f_, syncTicket_);
}

FenceRef& FenceRef::operator=(const FenceRef& o)
{
    if (this != &o)
    {
        cvAccelFence f = 0;
        uint64 ticket = o.syncTicket_;
        if (o.f_)
            retainOrForget(o.b_, o.f_, o.device_, f, ticket);
        reset();
        b_ = o.b_;
        f_ = f;
        streamUid_ = o.streamUid_;
        device_ = o.device_;
        syncTicket_ = ticket;
    }
    return *this;
}

void FenceRef::reset()
{
    if (f_ && !cv::__termination)
        b_->fn().fence_release(f_);
    b_ = 0;
    f_ = 0;
    streamUid_ = 0;
    device_ = -1;
    syncTicket_ = 0;
}

bool FenceRef::isComplete() const
{
    if (syncTicket_)
        return poll() == DONE;
    if (!f_)
        return true;
    cvAccelStatus st = b_->fn().fence_wait(f_, 0);
    if (st == CV_ACCEL_OK)
        return true;
    if (st == CV_ACCEL_TIMEOUT)
        return false;
    b_->check(st, "fence_wait", device_);
    return true;
}

//! OPENCV_ACCEL_WAIT_TIMEOUT_MS: a device that does not finish in time is considered lost (0: wait forever)
static std::atomic<uint64_t>& waitTimeoutNs()
{
    static std::atomic<uint64_t> ns((uint64_t)utils::getConfigurationParameterSizeT("OPENCV_ACCEL_WAIT_TIMEOUT_MS", 0) * 1000000u);
    return ns;
}

void FenceRef::waitUnknown(bool bounded) const
{
    // Nothing tells when the work ends but a synchronization of the whole device. Memory that
    // goes back to its owner is never handed back on a guess: retry until the device is idle or
    // reports that it stopped, however long that takes.
    unsigned sleepMs = 1;
    bool logged = false;
    for (;;)
    {
        if (poll() == DONE)
            return;
        const cvAccelStatus st = b_->synchronize(device_);
        if (st == CV_ACCEL_OK || st == CV_ACCEL_DEVICE_LOST)
            return;     // idle, or stopped (backend rule 8)
        if (bounded)
        {
            b_->markLost(device_);
            CV_Error_(Error::StsError, ("accel(%s): device %d failed and could not be synchronized (status %d): %s; "
                                        "the device is considered lost", b_->name().c_str(), device_, (int)st,
                                        b_->fn().get_last_error()));
        }
        if (!logged)
        {
            CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): device " << device_ << " failed and cannot be synchronized "
                         "(status " << (int)st << "): " << b_->fn().get_last_error() << "; memory that goes back "
                         "to its owner waits until the device is idle or stopped");
            logged = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
        sleepMs = std::min(sleepMs * 2, 1000u);
    }
}

void FenceRef::wait(bool bounded) const
{
    if (syncTicket_)
    {
        counters().hostWaits++;
        waitUnknown(bounded);
        return;
    }
    if (!f_)
        return;
    counters().hostWaits++;
    const uint64_t timeout = bounded ? waitTimeoutNs().load() : 0;
    const cvAccelStatus st = b_->fn().fence_wait(f_, timeout ? timeout : CV_ACCEL_INFINITE);
    if (st == CV_ACCEL_TIMEOUT)
    {
        b_->markLost(device_);
        CV_Error_(Error::StsError, ("accel(%s): device %d did not finish its work within %llu ms "
                                    "(OPENCV_ACCEL_WAIT_TIMEOUT_MS); the device is considered lost",
                                    b_->name().c_str(), device_, (unsigned long long)(timeout / 1000000u)));
    }
    b_->check(st, "fence_wait", device_);
}

FenceRef::Poll FenceRef::poll(bool logFailure) const CV_NOEXCEPT
{
    if (syncTicket_)
        return b_->isStopped(device_) || b_->syncedSince(device_, syncTicket_) ? DONE : PENDING;
    if (!f_)
        return DONE;
    const cvAccelStatus st = b_->fn().fence_wait(f_, 0);
    if (st == CV_ACCEL_OK)
        return DONE;
    if (st == CV_ACCEL_TIMEOUT)
        return PENDING;
    // the work behind the fence failed and is no longer running (see fence_wait)
    if (st == CV_ACCEL_DEVICE_LOST)
        b_->markStopped(device_);
    if (!logFailure)
        return FAILED;
    try
    {
        CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): device " << device_ << " failed (status " << (int)st
                     << "): " << b_->fn().get_last_error());
    }
    catch (...) {}
    return FAILED;
}

bool FenceRef::waitNoThrow(bool bounded) const CV_NOEXCEPT
{
    try
    {
        wait(bounded);
        return true;
    }
    catch (const std::exception& e)
    {
        try { CV_LOG_ERROR(NULL, "accel: " << e.what()); } catch (...) {}
    }
    catch (...) {}
    return false;
}

FenceRef fenceAfterFailure(const StreamPtr& s) CV_NOEXCEPT
{
    Backend* b = s->backend;
    cvAccelFence f = 0;
    if (b->fn().stream_record_fence(s->handle, &f) == CV_ACCEL_OK && f)
        return FenceRef(b, f, s->uid, s->device);
    const cvAccelStatus st = b->synchronize(s->device);
    if (st == CV_ACCEL_OK || st == CV_ACCEL_DEVICE_LOST)
        return FenceRef();      // idle, or stopped (backend rule 8): nothing of the work runs any more
    // Marking the device lost does not stop its DMA: the work counts as running until a later
    // synchronization succeeds, and nothing it may touch is freed or handed back before that.
    b->markLost(s->device);
    try
    {
        CV_LOG_ERROR(NULL, "accel(" << b->name() << "): device " << s->device << " failed and could not be "
                     "synchronized (status " << (int)st << "): " << b->fn().get_last_error()
                     << "; the device is considered lost and the memory of the failed work is kept");
    }
    catch (...) {}
    return FenceRef::unknown(b, s->device);
}

std::string currentExceptionMessage()
{
    try
    {
        throw;
    }
    catch (const cv::Exception& e)
    {
        return e.err;
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    catch (...)
    {
        return "unknown error";
    }
}

FenceRef recordFence(const StreamPtr& s)
{
    cvAccelFence f = 0;
    s->backend->check(s->backend->fn().stream_record_fence(s->handle, &f), "stream_record_fence", s->device);
    return FenceRef(s->backend, f, s->uid, s->device);
}

void streamWait(const StreamPtr& s, const FenceRef& f)
{
    if (f.isUnknown())
    {
        f.wait();   // no device-side handle: only a synchronization of its device orders it
        return;
    }
    if (f.empty() || f.streamUid() == s->uid)
        return;     // in-order stream: earlier work is already ordered
    if (f.backend() == s->backend)
    {
        s->backend->check(s->backend->fn().stream_wait_fence(s->handle, f.get()), "stream_wait_fence", s->device);
        return;
    }
    if (f.isComplete())
        return;
    // two backends: pass the fence as a sync_file when both can, otherwise wait on the CPU
    Backend* fb = f.backend();
    Backend* sb = s->backend;
    if (fb->fn().fence_export && sb->fn().stream_wait_external)
    {
        int64_t h = -1;
        if (fb->fn().fence_export(f.get(), CV_ACCEL_EXTFENCE_SYNC_FD, &h) == CV_ACCEL_OK &&
            sb->fn().stream_wait_external(s->handle, CV_ACCEL_EXTFENCE_SYNC_FD, h) == CV_ACCEL_OK)
            return;
    }
    f.wait();
}

//////////////////////////////////// Backend ////////////////////////////////////

Backend::Backend(int id, const OpenCV_Core_Accel_API* api, int priority, const std::string& origin)
    : id_(id), api_(api), priority_(priority),
      name_(api->v0.backend_name ? api->v0.backend_name : "unnamed"), origin_(origin),
      enumerated_(false), lost_(false), stopped_(false), pinnedCacheBytes_(0)
{
    // a backend built against an older header has a shorter table: what it lacks reads as NULL
    memset(&fn_, 0, sizeof(fn_));
    const size_t entries = api->api_header.valid_size - offsetof(OpenCV_Core_Accel_API_v0, v0);
    memcpy(&fn_, &api->v0, std::min(entries, sizeof(fn_)));
    for (int i = 0; i < MAX_TRACKED_DEVICES; i++)
    {
        lostDevices_[i].store(false);
        stoppedDevices_[i].store(false);
        noPinnedShadow_[i].store(false);
    }
    for (int i = 0; i <= MAX_TRACKED_DEVICES; i++)
    {
        syncIssued_[i].store(0);
        syncDone_[i].store(0);
    }
}

cvAccelStatus Backend::synchronize(int device) CV_NOEXCEPT
{
    const int k = slot(device);
    const uint64 ticket = ++syncIssued_[k];
    const cvAccelStatus st = fn().device_synchronize(device);
    if (st == CV_ACCEL_OK)
    {
        // everything queued before this call has finished: so has the work of every unknown
        // fence created before the ticket was taken
        uint64 done = syncDone_[k].load();
        while (done < ticket && !syncDone_[k].compare_exchange_weak(done, ticket))
            ;
    }
    else if (st == CV_ACCEL_DEVICE_LOST)
        markStopped(device);
    return st;
}

static uint64_t pinnedCacheLimit()
{
    static uint64_t limit = (uint64_t)utils::getConfigurationParameterSizeT("OPENCV_ACCEL_PINNED_CACHE_MB", 256) << 20;
    return limit;
}

bool Backend::takePinned(int device, uint64_t size, cvAccelAllocation& out)
{
    std::lock_guard<std::mutex> lk(mutex_);
    std::multimap<std::pair<int, uint64_t>, cvAccelAllocation>::iterator it = pinnedCache_.find(std::make_pair(device, size));
    if (it == pinnedCache_.end())
        return false;
    out = it->second;
    pinnedCacheBytes_ -= size;
    pinnedCache_.erase(it);
    return true;
}

void Backend::recyclePinned(cvAccelAllocation& a)
{
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (pinnedCacheBytes_ + a.size <= pinnedCacheLimit())
        {
            pinnedCache_.insert(std::make_pair(std::make_pair((int)a.desc.device_index, a.size), a));
            pinnedCacheBytes_ += a.size;
            return;
        }
    }
    if (fn().deallocate(&a) != CV_ACCEL_OK)
        CV_LOG_ERROR(NULL, "accel(" << name_ << "): failed to free pinned memory: " << fn().get_last_error());
}

void Backend::enumerate()
{
    if (enumerated_)
        return;
    enumerated_ = true;
    int n = 0;
    if (fn().get_device_count(&n) != CV_ACCEL_OK || n < 0)
    {
        CV_LOG_WARNING(NULL, "accel(" << name_ << "): get_device_count failed: " << fn().get_last_error());
        n = 0;
    }
    devices_.resize(n);
    idleGates_.clear();
    for (int i = 0; i < n; i++)
        idleGates_.push_back(std::unique_ptr<IdleGate>(new IdleGate()));
    for (int i = 0; i < n; i++)
    {
        cvAccelDeviceInfo& info = devices_[i];
        memset(&info, 0, sizeof(info));
        info.struct_size = (uint32_t)sizeof(info);
        if (fn().get_device_info(i, &info) != CV_ACCEL_OK)
        {
            CV_LOG_WARNING(NULL, "accel(" << name_ << "): get_device_info(" << i << ") failed: " << fn().get_last_error());
            memset(&info, 0, sizeof(info));
        }
        info.name[sizeof(info.name) - 1] = 0;
        if (info.caps & CV_ACCEL_CAP_DEVICE_IDLE_MEMORY)
            idleGates_[i]->used = true;     // before any submission, so none can slip past the gate
        CV_LOG_INFO(NULL, "accel(" << name_ << "): device " << i << ": " << info.name);
    }
}

int Backend::deviceCount()
{
    std::lock_guard<std::mutex> lk(mutex_);
    enumerate();
    return (int)devices_.size();
}

cvAccelDeviceInfo Backend::deviceInfo(int device)
{
    std::lock_guard<std::mutex> lk(mutex_);
    enumerate();
    CV_Assert(0 <= device && device < (int)devices_.size());
    return devices_[device];
}

void Backend::check(cvAccelStatus st, const char* what, int device) const
{
    if (st == CV_ACCEL_OK)
        return;
    if (st == CV_ACCEL_DEVICE_LOST)
        const_cast<Backend*>(this)->markStopped(device);     // backend rule 8
    const char* msg = fn().get_last_error ? fn().get_last_error() : "";
    const Error::Code code = st == CV_ACCEL_NOT_IMPLEMENTED ? Error::StsNotImplemented :
                             st == CV_ACCEL_OUT_OF_MEMORY ? Error::StsNoMem : Error::StsError;
    CV_Error_(code, ("accel(%s): %s failed (status %d): %s", name_.c_str(), what, (int)st, msg ? msg : ""));
}

const void* Backend::opTable(const char* module, uint32_t version)
{
    const std::string key = cv::format("%s/%u", module, version);
    std::lock_guard<std::mutex> lk(mutex_);
    std::map<std::string, const void*>::const_iterator it = opTables_.find(key);
    if (it != opTables_.end())
        return it->second;
    const void* table = 0;
    if (!fn().get_op_table || fn().get_op_table(module, version, &table) != CV_ACCEL_OK)
        table = 0;
    if (table && *(const size_t*)table < sizeof(size_t))
        table = 0;  // malformed
    opTables_[key] = table;
    return table;
}

bool Backend::canAccess(int device, const cvAccelMemoryDesc& desc)
{
    if (desc.backend_id == (uint32_t)id_ && (int)desc.device_index == device)
        return true;
    if (!fn().can_access)
        return false;
    // implicit fences are bracketed by the owning backend only (begin/end_device_access)
    if (desc.backend_id != (uint32_t)id_ && (desc.flags & CV_ACCEL_MEMF_EXTERNAL_SYNC))
        return false;
    const int own = desc.backend_id == (uint32_t)id_ ? (int)desc.device_index : -1;
    return fn().can_access(device, &desc, own) == CV_ACCEL_OK;
}

namespace {
struct ThreadStreams
{
    std::map<std::pair<int, int>, StreamPtr> streams;
};
TLSData<ThreadStreams>& threadStreams()
{
    CV_SINGLETON_LAZY_INIT_REF(TLSData<ThreadStreams>, new TLSData<ThreadStreams>())
}
} // namespace

StreamPtr Backend::defaultStream(int device)
{
    ThreadStreams& ts = threadStreams().getRef();
    const std::pair<int, int> key(id_, device);
    std::map<std::pair<int, int>, StreamPtr>::const_iterator it = ts.streams.find(key);
    if (it != ts.streams.end())
        return it->second;
    cvAccelStream h = 0;
    check(fn().stream_create(device, 0, &h), "stream_create", device);
    StreamPtr s = std::make_shared<StreamHolder>(this, device, h);
    ts.streams[key] = s;
    return s;
}

MatAllocator* Backend::allocator(int device, int kind, bool protectedMem)
{
    std::lock_guard<std::mutex> lk(mutex_);
    enumerate();
    CV_Assert(0 <= device && device < (int)devices_.size());
    const std::pair<int, int> key(device, kind | (protectedMem ? 1 << 16 : 0));
    std::map<std::pair<int, int>, MatAllocator*>::const_iterator it = allocators_.find(key);
    if (it != allocators_.end())
        return it->second;
    MatAllocator* a = new DeviceAllocator(this, device, kind, protectedMem);  // never deleted, like the OpenCL allocator
    allocators_[key] = a;
    return a;
}

SubmitWindow::SubmitWindow(IdleGate& gate) : gate_(&gate), open_(false), blocked_(false)
{
    if (!gate.used.load())
        return;
    gate.submitting++;
    open_ = true;
    if (gate.cpuMappings.load() > 0)
    {
        blocked_ = true;
        close();
    }
}

void SubmitWindow::close()
{
    if (!open_)
        return;
    open_ = false;
    if (--gate_->submitting == 0)
    {
        std::lock_guard<std::mutex> lk(gate_->mutex);
        gate_->cond.notify_all();
    }
}

IdleGate& Backend::idleGate(int device)
{
    std::lock_guard<std::mutex> lk(mutex_);
    enumerate();
    CV_Assert(0 <= device && device < (int)idleGates_.size());
    return *idleGates_[device];
}

static size_t hostPageSize()
{
#ifdef _WIN32
    return 4096;
#else
    static size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps ? ps : 4096;
#endif
}

bool Backend::reserveHostImport(int device, const void* ptr, size_t size)
{
    const uintptr_t page = hostPageSize();
    const uintptr_t begin = (uintptr_t)ptr & ~(page - 1);
    const uintptr_t end = ((uintptr_t)ptr + size + page - 1) & ~(page - 1);
    std::lock_guard<std::mutex> lk(mutex_);
    // the first range that starts at or after begin, and the one before it
    std::map<std::pair<int, uintptr_t>, uintptr_t>::iterator it = hostImports_.lower_bound(std::make_pair(device, begin));
    if (it != hostImports_.end() && it->first.first == device && it->first.second < end)
        return false;
    if (it != hostImports_.begin())
    {
        std::map<std::pair<int, uintptr_t>, uintptr_t>::iterator prev = it;
        --prev;
        if (prev->first.first == device && prev->second > begin)
            return false;
    }
    hostImports_[std::make_pair(device, begin)] = end;
    return true;
}

void Backend::releaseHostImport(int device, const void* ptr)
{
    const uintptr_t page = hostPageSize();
    const uintptr_t begin = (uintptr_t)ptr & ~(page - 1);
    std::lock_guard<std::mutex> lk(mutex_);
    hostImports_.erase(std::make_pair(device, begin));
}

//////////////////////////////////// Registry ////////////////////////////////////

static const int MAX_BACKENDS = 64;

Registry::Registry() : active_(false)
{
    backends_.reserve(MAX_BACKENDS);
}

Registry& Registry::getInstance()
{
    static Registry* r = new Registry();  // never destroyed: buffers may outlive static destruction
    return *r;
}

static bool hasRequiredEntries(const OpenCV_Core_Accel_API_v0_0_api_entries& e)
{
    return e.get_device_count && e.get_device_info && e.allocate && e.deallocate &&
           e.begin_cpu_access && e.end_cpu_access && e.copy &&
           e.stream_create && e.stream_destroy && e.stream_record_fence && e.stream_wait_fence &&
           e.fence_wait && e.fence_retain && e.fence_release && e.device_synchronize &&
           e.get_last_error;
}

int Registry::add(const OpenCV_Core_Accel_API* api, int priority, const std::string& origin)
{
    if (!api)
        return 0;
    const OpenCV_API_Header& h = api->api_header;
    // every table has the entries up to get_last_error; later ones were appended
    const size_t minSize = offsetof(OpenCV_Core_Accel_API_v0, v0) +
                           offsetof(OpenCV_Core_Accel_API_v0_0_api_entries, describe_allocation);
    if (h.valid_size < minSize)
    {
        CV_LOG_ERROR(NULL, "accel: backend from " << origin << " rejected: API table is too small");
        return 0;
    }
    if (h.min_api_version != CV_ACCEL_ABI_VERSION)
    {
        CV_LOG_ERROR(NULL, "accel: backend from " << origin << " rejected: ABI " << h.min_api_version
                     << ", OpenCV expects " << CV_ACCEL_ABI_VERSION);
        return 0;
    }
    if (h.opencv_version_major != CV_VERSION_MAJOR)
    {
        CV_LOG_ERROR(NULL, "accel: backend from " << origin << " rejected: built for OpenCV "
                     << h.opencv_version_major << ".x");
        return 0;
    }
    if (!hasRequiredEntries(api->v0) || !api->v0.backend_name)
    {
        CV_LOG_ERROR(NULL, "accel: backend from " << origin << " rejected: required entry points are missing");
        return 0;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    for (size_t i = 0; i < backends_.size(); i++)
    {
        if (backends_[i]->name() == api->v0.backend_name)
        {
            CV_LOG_WARNING(NULL, "accel: backend '" << api->v0.backend_name << "' from " << origin
                           << " ignored: already registered from " << backends_[i]->origin());
            return 0;
        }
    }
    if ((int)backends_.size() >= MAX_BACKENDS)
        return 0;
    const int id = (int)backends_.size() + 1;
    backends_.push_back(new Backend(id, api, priority, origin));  // capacity is reserved: get() never sees a reallocation
    active_.store(true, std::memory_order_release);
    CV_LOG_INFO(NULL, "accel: registered backend '" << api->v0.backend_name << "' (" << origin << ", API "
                << h.api_version << ", priority " << priority << ")");
    return id;
}

Backend* Registry::get(int id)
{
    if (id <= 0)
        return 0;
    std::lock_guard<std::mutex> lk(mutex_);
    return id <= (int)backends_.size() ? backends_[id - 1] : 0;
}

Backend* Registry::find(const std::string& name)
{
    std::lock_guard<std::mutex> lk(mutex_);
    for (size_t i = 0; i < backends_.size(); i++)
        if (backends_[i]->name() == name)
            return backends_[i];
    return 0;
}

std::vector<Backend*> Registry::all()
{
    std::vector<Backend*> r;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        r = backends_;
    }
    std::stable_sort(r.begin(), r.end(), [](const Backend* a, const Backend* b) { return a->priority() > b->priority(); });
    return r;
}

void Registry::loadPlugins()
{
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
    // explicit list for now; the search over plugin directories follows the parallel backends
    const std::string list = utils::getConfigurationParameterString("OPENCV_ACCEL_PLUGINS", "");
    size_t pos = 0;
    while (pos < list.size())
    {
        size_t end = list.find(';', pos);
        if (end == std::string::npos)
            end = list.size();
        const std::string path = list.substr(pos, end - pos);
        pos = end + 1;
        if (path.empty())
            continue;
        std::shared_ptr<plugin::impl::DynamicLib> lib =
                std::make_shared<plugin::impl::DynamicLib>(plugin::impl::toFileSystemPath(path));
        if (!lib->isLoaded())
        {
            CV_LOG_WARNING(NULL, "accel: can't load plugin " << path);
            continue;
        }
        FN_opencv_core_accel_plugin_init_t init = reinterpret_cast<FN_opencv_core_accel_plugin_init_t>(
                lib->getSymbol("opencv_core_accel_plugin_init_v0"));
        if (!init)
        {
            CV_LOG_WARNING(NULL, "accel: " << path << " is not an accelerator plugin");
            continue;
        }
        // the highest API version first, like the other plugin loaders: a plugin built against
        // an older header may only answer for the version it knows
        const OpenCV_Core_Accel_API* api = NULL;
        for (int v = CV_ACCEL_API_VERSION; v >= 0 && !api; v--)
            api = init(CV_ACCEL_ABI_VERSION, v, NULL);
        if (add(api, 50, "plugin " + path))
            lib->disableAutomaticLibraryUnloading();   // tables and code must outlive every buffer
    }
#endif
}

static void initBackends()
{
    Registry& r = Registry::getInstance();
    if (utils::getConfigurationParameterBool("OPENCV_ACCEL_EMULATOR", false))
        r.add(emulatorBackendApi(), -100, "built-in");
#ifdef HAVE_CUDA
    if (utils::getConfigurationParameterBool("OPENCV_ACCEL_CUDA", true))
        r.add(cudaBackendApi(), 100, "built-in");
#endif
    r.loadPlugins();
}

void ensureInitialized()
{
    static std::once_flag once;
    std::call_once(once, initBackends);
}

//////////////////////////////////// scopes ////////////////////////////////////

namespace {
struct ThreadScopes
{
    ThreadScopes() : oclSuppressed(0) {}
    std::vector<ScopeEntry> stack;
    int oclSuppressed;
};
TLSData<ThreadScopes>& threadScopes()
{
    CV_SINGLETON_LAZY_INIT_REF(TLSData<ThreadScopes>, new TLSData<ThreadScopes>())
}
} // namespace

const ScopeEntry* currentScope()
{
    if (scopeCount().load(std::memory_order_relaxed) == 0)
        return NULL;
    ThreadScopes& t = threadScopes().getRef();
    return t.stack.empty() ? NULL : &t.stack.back();
}

MatAllocator* scopeAllocator()
{
    const ScopeEntry* s = currentScope();
    if (!s || s->backendId == 0)
        return NULL;
    Backend* b = Registry::getInstance().get(s->backendId);
    if (!b)
        return NULL;
    return b->allocator(s->device, (int)b->deviceInfo(s->device).preferred_kind);
}

void noteBufferCreated() { activity()++; }
void noteBufferDestroyed() { activity()--; }

}} // namespace accel::impl

//////////////////////////////////// detail ////////////////////////////////////

namespace accel { namespace detail {

bool isActive()
{
    return impl::activity().load(std::memory_order_relaxed) > 0;
}

bool inExecutionScope()
{
    const impl::ScopeEntry* s = impl::currentScope();
    return s && s->backendId != 0;
}

bool isOpenCLSuppressed()
{
    if (impl::oclSuppressCount().load(std::memory_order_relaxed) == 0)
        return false;
    return impl::threadScopes().getRef().oclSuppressed > 0;
}

void OpenCLSuppressor::suppress()
{
    if (active_)
        return;
    active_ = true;
    impl::threadScopes().getRef().oclSuppressed++;
    impl::oclSuppressCount()++;
}

void OpenCLSuppressor::release()
{
    active_ = false;
    impl::threadScopes().getRef().oclSuppressed--;
    impl::oclSuppressCount()--;
}

int registerEmulatorBackend()
{
    impl::ensureInitialized();
    impl::Registry& r = impl::Registry::getInstance();
    if (impl::Backend* b = r.find("emulator"))
        return b->id();
    return r.add(impl::emulatorBackendApi(), -100, "built-in (test)");
}

int registerLegacyEmulatorBackend()
{
    registerEmulatorBackend();
    impl::Registry& r = impl::Registry::getInstance();
    if (impl::Backend* b = r.find("emulator-v1"))
        return b->id();
    return r.add(impl::emulatorBackendApiV1(), -101, "built-in (test, API version 1)");
}

}} // namespace accel::detail

//////////////////////////////////// public API ////////////////////////////////////

namespace accel {

using namespace impl;

static Backend* backendOf(const Device& d)
{
    Backend* b = Registry::getInstance().get(d.backendId());
    CV_Assert(b && "accel: unknown device");
    return b;
}

Device::Device() : backendId_(0), index_(0) {}

std::vector<Device> Device::enumerate()
{
    ensureInitialized();
    std::vector<Device> r;
    std::vector<Backend*> bs = Registry::getInstance().all();
    for (size_t i = 0; i < bs.size(); i++)
    {
        const int n = bs[i]->deviceCount();
        for (int d = 0; d < n; d++)
            r.push_back(Device(bs[i]->id(), d));
    }
    return r;
}

Device Device::find(const String& spec)
{
    ensureInitialized();
    std::string name = spec;
    int index = 0;
    const size_t colon = spec.find(':');
    if (colon != std::string::npos)
    {
        name = spec.substr(0, colon);
        index = atoi(spec.c_str() + colon + 1);
    }
    if (name.empty() || name == "host" || name == "cpu")
        return Device();
    Backend* b = Registry::getInstance().find(name);
    if (!b || index < 0 || index >= b->deviceCount())
        return Device();
    return Device(b->id(), index);
}

String Device::backend() const
{
    return empty() ? String("host") : String(backendOf(*this)->name());
}

String Device::name() const
{
    return empty() ? String("host") : String(backendOf(*this)->deviceInfo(index_).name);
}

DeviceType Device::type() const
{
    return empty() ? DEVICE_HOST : (DeviceType)backendOf(*this)->deviceInfo(index_).device_type;
}

MatAllocator* Device::allocator() const
{
    if (empty())
        return Mat::getDefaultAllocator();
    Backend* b = backendOf(*this);
    return b->allocator(index_, (int)b->deviceInfo(index_).preferred_kind);
}

MemoryInfo describe(const UMat& m)
{
    MemoryInfo info;
    cvAccelMemoryDesc d;
    if (!m.u || !queryMemoryDesc(m.u, d))
        return info;
    if (d.backend_id != 0)
        info.device = Device((int)d.backend_id, (int)d.device_index);
    info.kind = (MemoryKind)d.kind;
    info.hostAccess = (HostAccess)d.host_access;
    info.imported = (d.flags & CV_ACCEL_MEMF_IMPORTED) != 0;
    info.readOnly = (d.flags & CV_ACCEL_MEMF_READ_ONLY) != 0;
    info.protectedContent = (d.flags & CV_ACCEL_MEMF_PROTECTED) != 0;
    return info;
}

struct Stream::Impl
{
    StreamPtr s;
};

Stream::Stream() {}

Stream::Stream(const Device& device)
{
    CV_Assert(!device.empty() && "accel: the host has no streams");
    Backend* b = backendOf(device);
    cvAccelStream h = 0;
    b->check(b->fn().stream_create(device.index(), 0, &h), "stream_create", device.index());
    impl = makePtr<Impl>();
    impl->s = std::make_shared<StreamHolder>(b, device.index(), h);
}

Stream Stream::getDefault(const Device& device)
{
    Stream s;
    if (device.empty())
        return s;
    s.impl = makePtr<Impl>();
    s.impl->s = backendOf(device)->defaultStream(device.index());
    return s;
}

Device Stream::device() const
{
    return impl ? Device(impl->s->backend->id(), impl->s->device) : Device();
}

bool Stream::empty() const
{
    return !impl;
}

void Stream::waitForCompletion()
{
    if (impl)
        recordFence(impl->s).wait();
}

ExecutionScope::ExecutionScope(const Device& device)
{
    ScopeEntry e;
    e.backendId = device.backendId();
    e.device = device.index();
    threadScopes().getRef().stack.push_back(e);
    scopeCount()++;
    activity()++;
}

ExecutionScope::ExecutionScope(const Stream& stream)
{
    CV_Assert(!stream.empty());
    ScopeEntry e;
    e.backendId = stream.impl->s->backend->id();
    e.device = stream.impl->s->device;
    e.stream = stream.impl->s;
    threadScopes().getRef().stack.push_back(e);
    scopeCount()++;
    activity()++;
}

ExecutionScope::~ExecutionScope()
{
    std::vector<ScopeEntry>& stack = threadScopes().getRef().stack;
    CV_DbgAssert(!stack.empty());
    if (!stack.empty())
        stack.pop_back();
    scopeCount()--;
    activity()--;
}

void setTransferPolicy(TransferPolicy policy)
{
    transferPolicy().store((int)policy);
}

TransferPolicy getTransferPolicy()
{
    return (TransferPolicy)transferPolicy().load();
}

TransferStats getTransferStats()
{
    Counters& c = counters();
    TransferStats s;
    s.bytesHostToDevice = c.bytesH2D.load();
    s.bytesDeviceToHost = c.bytesD2H.load();
    s.bytesDeviceToDevice = c.bytesD2D.load();
    s.transfers = c.transfers.load();
    s.implicitTransfers = c.implicitTransfers.load();
    s.deviceOps = c.deviceOps.load();
    s.cpuFallbacks = c.cpuFallbacks.load();
    s.hostWaits = c.hostWaits.load();
    return s;
}

void resetTransferStats()
{
    Counters& c = counters();
    c.bytesH2D = 0; c.bytesD2H = 0; c.bytesD2D = 0; c.transfers = 0;
    c.implicitTransfers = 0; c.deviceOps = 0; c.cpuFallbacks = 0; c.hostWaits = 0;
}

//! a * b, false when it does not fit in 64 bits
static bool mulFits(uint64_t a, uint64_t b, uint64_t& r)
{
    if (b != 0 && a > UINT64_MAX / b)
        return false;
    r = a * b;
    return true;
}

/** Steps and extent (bytes from the first element to the end of the last) of an array of
    @p shape and @p type over external memory: the caller's steps, or packed ones. Every
    product and sum is checked, so that no shape can wrap around and look small. */
static uint64_t externalExtent(const MatShape& shape, int type, const size_t* steps, size_t* outSteps)
{
    const uint64_t esz = (uint64_t)CV_ELEM_SIZE(type);
    const int dims = shape.dims;
    if (steps && steps[dims - 1] != esz)
        CV_Error(Error::StsBadArg, "accel: the last step of a wrapped array must be its element size");
    uint64_t st[CV_ACCEL_MAX_DIMS];
    st[dims - 1] = esz;
    for (int i = dims - 1; i >= 0; i--)
    {
        if (shape.p[i] <= 0)
            CV_Error(Error::StsBadArg, "accel: a wrapped array has no empty dimension");
        if (i < dims - 1 && !(steps ? (st[i] = steps[i], true) : mulFits(st[i + 1], (uint64_t)shape.p[i + 1], st[i])))
            CV_Error(Error::StsBadArg, "accel: the wrapped array does not fit in the address space");
    }
    uint64_t extent = esz;
    for (int i = 0; i < dims; i++)
    {
        uint64_t span = 0;
        if (!mulFits((uint64_t)(shape.p[i] - 1), st[i], span) || !rangeEnd(extent, span, extent))
            CV_Error(Error::StsBadArg, "accel: the wrapped array does not fit in the address space");
    }
    if (extent > (uint64_t)std::numeric_limits<size_t>::max())
        CV_Error(Error::StsBadArg, "accel: the wrapped array does not fit in the address space");
    for (int i = 0; i < dims; i++)
        outSteps[i] = (size_t)st[i];
    return extent;
}

UMat wrapExternalMemory(const Device& device, const ExternalMemory& memory,
                        const MatShape& shape, int type, const size_t* steps)
{
    CV_Assert(!device.empty() && "accel: external memory needs a device that imports it");
    CV_Assert(shape.dims > 0 && shape.dims <= CV_ACCEL_MAX_DIMS);
    Backend* b = backendOf(device);
    DeviceAllocator* a = static_cast<DeviceAllocator*>(b->allocator(device.index(), (int)b->deviceInfo(device.index()).preferred_kind));

    type = CV_MAT_TYPE(type);
    size_t st[CV_ACCEL_MAX_DIMS];
    const uint64_t extent = externalExtent(shape, type, steps, st);
    uint64_t bytes = extent;
    if (memory.layoutModifier != 0)
    {
        // a tiled or compressed layout: its size cannot be computed from a shape and steps
        if (memory.size == 0)
            CV_Error(Error::StsBadArg, "accel: memory with a non-linear layout needs its size (ExternalMemory::size)");
        bytes = memory.size;
    }
    else if (memory.size != 0 && extent > memory.size)
        CV_Error(Error::StsBadArg, "accel: the array does not fit in the external memory");

    UMat m;
    m.flags = UMat::MAGIC_VAL | type;
    setSize(m, shape.dims, shape.p, st, false);

    WrapRequest req;
    req.ext.type = (uint32_t)memory.type;
    req.ext.handle = (int64_t)memory.handle;
    req.ext.offset = memory.offset;
    req.ext.size = bytes;
    req.ext.flags = (memory.readOnly ? CV_ACCEL_MEMF_READ_ONLY : 0) | (memory.protectedContent ? CV_ACCEL_MEMF_PROTECTED : 0);
    req.ext.layout_modifier = memory.layoutModifier;
    req.ext.fourcc = memory.fourcc;
    req.readOnly = memory.readOnly;
    req.protectedRequired = memory.protectedContent;
    req.allowExistingView = true;
    size_t viewOffset = 0;
    bool referenced = false;
    m.u = a->wrap(req, viewOffset, referenced);
    m.offset = viewOffset;
    finalizeHdr(m);
    if (!referenced)
        m.addref();     // a new view; an existing one was referenced by wrap
    return m;
}

namespace {

#define CV_ACCEL_FOURCC(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

//! what a known format requires of one plane: its type, and its size relative to plane 0
struct FormatPlane
{
    int type;
    int hsub, vsub;     //!< plane 0's width and height are divided by these (rounded up)
};

struct Format
{
    uint32_t fourcc;
    int planes;
    FormatPlane p[CV_ACCEL_MAX_PLANES];
};

//! DRM formats OpenCV checks images against (planes as matrices: plane 0 has the image's size in pixels)
const Format* findFormat(uint32_t fourcc)
{
    static const Format formats[] = {
        { CV_ACCEL_FOURCC('N', 'V', '1', '2'), 2, { { CV_8UC1, 1, 1 }, { CV_8UC2, 2, 2 } } },
        { CV_ACCEL_FOURCC('N', 'V', '2', '1'), 2, { { CV_8UC1, 1, 1 }, { CV_8UC2, 2, 2 } } },
        { CV_ACCEL_FOURCC('N', 'V', '1', '6'), 2, { { CV_8UC1, 1, 1 }, { CV_8UC2, 2, 1 } } },
        { CV_ACCEL_FOURCC('N', 'V', '6', '1'), 2, { { CV_8UC1, 1, 1 }, { CV_8UC2, 2, 1 } } },
        { CV_ACCEL_FOURCC('N', 'V', '2', '4'), 2, { { CV_8UC1, 1, 1 }, { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('N', 'V', '4', '2'), 2, { { CV_8UC1, 1, 1 }, { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('P', '0', '1', '0'), 2, { { CV_16UC1, 1, 1 }, { CV_16UC2, 2, 2 } } },
        { CV_ACCEL_FOURCC('P', '0', '1', '2'), 2, { { CV_16UC1, 1, 1 }, { CV_16UC2, 2, 2 } } },
        { CV_ACCEL_FOURCC('P', '0', '1', '6'), 2, { { CV_16UC1, 1, 1 }, { CV_16UC2, 2, 2 } } },
        { CV_ACCEL_FOURCC('P', '2', '1', '0'), 2, { { CV_16UC1, 1, 1 }, { CV_16UC2, 2, 1 } } },
        { CV_ACCEL_FOURCC('Y', 'U', '1', '2'), 3, { { CV_8UC1, 1, 1 }, { CV_8UC1, 2, 2 }, { CV_8UC1, 2, 2 } } },
        { CV_ACCEL_FOURCC('Y', 'V', '1', '2'), 3, { { CV_8UC1, 1, 1 }, { CV_8UC1, 2, 2 }, { CV_8UC1, 2, 2 } } },
        { CV_ACCEL_FOURCC('Y', 'U', '1', '6'), 3, { { CV_8UC1, 1, 1 }, { CV_8UC1, 2, 1 }, { CV_8UC1, 2, 1 } } },
        { CV_ACCEL_FOURCC('Y', 'V', '1', '6'), 3, { { CV_8UC1, 1, 1 }, { CV_8UC1, 2, 1 }, { CV_8UC1, 2, 1 } } },
        { CV_ACCEL_FOURCC('Y', 'U', '2', '4'), 3, { { CV_8UC1, 1, 1 }, { CV_8UC1, 1, 1 }, { CV_8UC1, 1, 1 } } },
        { CV_ACCEL_FOURCC('Y', 'U', 'Y', 'V'), 1, { { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('Y', 'V', 'Y', 'U'), 1, { { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('U', 'Y', 'V', 'Y'), 1, { { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('V', 'Y', 'U', 'Y'), 1, { { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('R', '8', ' ', ' '), 1, { { CV_8UC1, 1, 1 } } },
        { CV_ACCEL_FOURCC('R', '1', '6', ' '), 1, { { CV_16UC1, 1, 1 } } },
        { CV_ACCEL_FOURCC('G', 'R', '8', '8'), 1, { { CV_8UC2, 1, 1 } } },
        { CV_ACCEL_FOURCC('R', 'G', '1', '6'), 1, { { CV_8UC2, 1, 1 } } },      // RGB565, as cvtColor takes it
        { CV_ACCEL_FOURCC('R', 'G', '2', '4'), 1, { { CV_8UC3, 1, 1 } } },
        { CV_ACCEL_FOURCC('B', 'G', '2', '4'), 1, { { CV_8UC3, 1, 1 } } },
        { CV_ACCEL_FOURCC('X', 'R', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('X', 'B', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('A', 'R', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('A', 'B', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('R', 'X', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('B', 'X', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('R', 'A', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
        { CV_ACCEL_FOURCC('B', 'A', '2', '4'), 1, { { CV_8UC4, 1, 1 } } },
    };
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
        if (formats[i].fourcc == fourcc)
            return &formats[i];
    return NULL;
}

std::string fourccName(uint32_t f)
{
    char s[5] = { (char)(f & 0xff), (char)((f >> 8) & 0xff), (char)((f >> 16) & 0xff), (char)(f >> 24), 0 };
    return s;
}

std::atomic<uint64>& imageIds()
{
    static std::atomic<uint64>* c = new std::atomic<uint64>(0);
    return *c;
}

} // namespace

std::vector<UMat> wrapExternalImage(const Device& device, const ExternalImage& image)
{
    CV_Assert(!device.empty() && "accel: external memory needs a device that imports it");
    const size_t n = image.planes.size();
    if (n < 1 || n > CV_ACCEL_MAX_PLANES)
        CV_Error(Error::StsBadArg, "accel: an image has 1 to 4 planes");
    Backend* b = backendOf(device);
    DeviceAllocator* a = static_cast<DeviceAllocator*>(b->allocator(device.index(), (int)b->deviceInfo(device.index()).preferred_kind));

    // 1. the planes, each with checked arithmetic: where it lies, how large it is, which buffer
    std::shared_ptr<ImageRecord> rec = std::make_shared<ImageRecord>();
    cvAccelImageDesc& d = rec->desc;
    memset(&d, 0, sizeof(d));
    d.struct_size = (uint32_t)sizeof(d);
    d.fourcc = image.fourcc;
    d.id = ++imageIds();
    d.plane_count = (uint32_t)n;
    d.color_model = (uint32_t)image.color.model;
    d.color_range = (uint32_t)image.color.range;
    d.color_primaries = (uint32_t)image.color.primaries;
    d.transfer = (uint32_t)image.color.transfer;
    d.chroma_siting_x = (uint32_t)image.color.chromaSitingX;
    d.chroma_siting_y = (uint32_t)image.color.chromaSitingY;
    for (size_t i = 0; i < n; i++)
    {
        const ExternalPlane& p = image.planes[i];
        cvAccelImagePlane& q = d.planes[i];
        if (p.rows <= 0 || p.cols <= 0 || p.type < 0 || p.type != CV_MAT_TYPE(p.type))
            CV_Error_(Error::StsBadArg, ("accel: plane %d has no valid shape or type", (int)i));
        q.rows = p.rows;
        q.cols = p.cols;
        q.type = p.type;
        q.layout_modifier = p.layoutModifier == ExternalPlane::SAME_AS_IMAGE ? image.layoutModifier : p.layoutModifier;
        q.offset = p.offset;
        uint64_t rowBytes = 0;
        if (!mulFits((uint64_t)p.cols, (uint64_t)CV_ELEM_SIZE(p.type), rowBytes))
            CV_Error_(Error::StsBadArg, ("accel: plane %d is larger than the address space", (int)i));
        q.pitch = p.pitch ? (uint64_t)p.pitch : rowBytes;
        if (q.layout_modifier == 0)
        {
            uint64_t span = 0, linear = 0;
            if (q.pitch < rowBytes)
                CV_Error_(Error::StsBadArg, ("accel: the pitch of plane %d is shorter than its rows", (int)i));
            if (!mulFits(q.pitch, (uint64_t)(p.rows - 1), span) || !rangeEnd(span, rowBytes, linear))
                CV_Error_(Error::StsBadArg, ("accel: plane %d is larger than the address space", (int)i));
            if (p.size && p.size < linear)
                CV_Error_(Error::StsBadArg, ("accel: plane %d is smaller than its rows and pitch say", (int)i));
            q.size = p.size ? (uint64_t)p.size : linear;
        }
        else
        {
            // tiled or compressed: only the producer knows how many bytes the plane takes
            if (p.size == 0)
                CV_Error_(Error::StsBadArg, ("accel: plane %d has a non-linear layout: its size must be given", (int)i));
            q.size = p.size;
        }
        uint64_t end = 0;
        if (!rangeEnd(q.offset, q.size, end) || q.size > (uint64_t)std::numeric_limits<size_t>::max())
            CV_Error_(Error::StsBadArg, ("accel: plane %d lies outside of the address space", (int)i));
        // planes with one handle lie in one buffer, and never share a byte
        q.buffer = (uint32_t)i;
        for (size_t j = 0; j < i; j++)
        {
            if (image.planes[j].handle != p.handle)
                continue;
            q.buffer = d.planes[j].buffer;
            if (rangesOverlap(d.planes[j].offset, d.planes[j].size, q.offset, q.size))
                CV_Error_(Error::StsBadArg, ("accel: planes %d and %d overlap in their buffer", (int)j, (int)i));
        }
    }

    // 2. the format, when OpenCV knows it: plane count, types, subsampled sizes
    if (const Format* f = findFormat(image.fourcc))
    {
        if ((int)n != f->planes)
            CV_Error_(Error::StsBadArg, ("accel: %s has %d planes, not %d", fourccName(image.fourcc).c_str(), f->planes, (int)n));
        const int width = d.planes[0].cols, height = d.planes[0].rows;
        for (int i = 0; i < f->planes; i++)
        {
            const FormatPlane& fp = f->p[i];
            const int cols = (width + fp.hsub - 1) / fp.hsub, rows = (height + fp.vsub - 1) / fp.vsub;
            if (d.planes[i].type != fp.type || d.planes[i].cols != cols || d.planes[i].rows != rows)
                CV_Error_(Error::StsBadArg, ("accel: plane %d of a %dx%d %s image must be %dx%d of type %s",
                                             i, width, height, fourccName(image.fourcc).c_str(), cols, rows,
                                             typeToString(fp.type).c_str()));
        }
    }

    // 3. one view per plane, each carrying the image
    std::vector<UMat> planes;
    for (size_t i = 0; i < n; i++)
    {
        const ExternalPlane& p = image.planes[i];
        const cvAccelImagePlane& q = d.planes[i];
        WrapRequest req;
        req.ext.type = (uint32_t)image.type;
        req.ext.handle = (int64_t)p.handle;
        req.ext.offset = q.offset;
        req.ext.size = q.size;
        req.ext.flags = (image.readOnly ? CV_ACCEL_MEMF_READ_ONLY : 0) | (image.protectedContent ? CV_ACCEL_MEMF_PROTECTED : 0);
        req.ext.layout_modifier = q.layout_modifier;
        req.ext.fourcc = image.fourcc;
        req.readOnly = image.readOnly;
        req.protectedRequired = image.protectedContent;
        req.image = rec;
        req.plane = (int)i;
        size_t viewOffset = 0;
        bool referenced = false;
        UMat m;
        m.flags = UMat::MAGIC_VAL | p.type;
        const int sz[2] = { p.rows, p.cols };
        const size_t steps[2] = { (size_t)q.pitch, (size_t)CV_ELEM_SIZE(p.type) };
        setSize(m, 2, sz, steps, false);
        m.u = a->wrap(req, viewOffset, referenced);
        CV_Assert(!referenced && viewOffset == 0);  // a plane always gets a view of its own
        finalizeHdr(m);
        m.addref();
        planes.push_back(m);
    }
    return planes;
}

ImagePlaneInfo describePlane(const UMat& m)
{
    ImagePlaneInfo info;
    const BufferState* v = m.u ? m.u->accelState : NULL;
    if (!v || !v->image)
        return info;
    const cvAccelImageDesc& d = v->image->desc;
    info.imageId = d.id;
    info.plane = v->plane;
    info.planeCount = (int)d.plane_count;
    info.fourcc = d.fourcc;
    info.layoutModifier = d.planes[v->plane].layout_modifier;
    info.color.model = (ColorModel)d.color_model;
    info.color.range = (ColorRange)d.color_range;
    info.color.primaries = (ColorPrimaries)d.color_primaries;
    info.color.transfer = (TransferFunction)d.transfer;
    info.color.chromaSitingX = (ChromaSiting)d.chroma_siting_x;
    info.color.chromaSitingY = (ChromaSiting)d.chroma_siting_y;
    return info;
}

struct DeviceAccess::Impl
{
    Impl() : gate(0), acq(true) {}
    ~Impl()
    {
        if (gate)
        {
            std::lock_guard<std::mutex> lk(gate->mutex);
            std::vector<std::thread::id>::iterator it = std::find(gate->holders.begin(), gate->holders.end(), holder);
            if (it != gate->holders.end())
                gate->holders.erase(it);
        }
    }
    /** The caller may run device work at any time while the access exists: like an operation,
        it keeps the CPU out of device-idle memory of the device (see IdleGate). */
    void enterGate()
    {
        IdleGate& g = s->backend->idleGate(s->device);
        window.reset(new SubmitWindow(g));
        if (window->blocked())
            CV_Error(Error::StsError, "accel: DeviceAccess: the CPU holds device-idle memory of this device "
                                      "(a Mat view of it is open)");
        if (g.used.load())
        {
            std::lock_guard<std::mutex> lk(g.mutex);
            holder = std::this_thread::get_id();
            g.holders.push_back(holder);
            gate = &g;
        }
    }
    UMat m;
    StreamPtr s;
    std::unique_ptr<SubmitWindow> window;   //!< open for the whole access, closed last
    IdleGate* gate;                         //!< set while registered in gate->holders
    std::thread::id holder;
    DeviceAcquisition acq;  //!< a long hold: released when the DeviceAccess goes away
};

DeviceAccess::DeviceAccess(const UMat& m, const Stream& stream, AccessFlag access) : p(0)
{
    CV_Assert(m.u && m.u->accelState && "accel: DeviceAccess needs a UMat allocated on a device");
    CV_Assert(!stream.empty());
    // the caller's work runs on that stream with this memory's handle
    const BufferState* v = m.u->accelState;
    const StreamPtr& sp = stream.impl->s;
    if (sp->backend != v->backend())
        CV_Error_(Error::StsBadArg, ("accel: DeviceAccess: the stream belongs to backend '%s', the UMat to '%s'",
                                     sp->backend->name().c_str(), v->backend()->name().c_str()));
    cvAccelMemoryDesc desc;
    CV_Assert(queryMemoryDesc(m.u, desc));
    if (sp->device != v->device() && !sp->backend->canAccess(sp->device, desc))
        CV_Error_(Error::StsBadArg, ("accel: DeviceAccess: device %d cannot reach the UMat's memory on device %d",
                                     sp->device, v->device()));
    std::unique_ptr<Impl> impl_(new Impl());
    impl_->m = m;
    impl_->s = stream.impl->s;
    impl_->enterGate();
    int mode = ((access & ACCESS_READ) ? DEV_READ : 0) | ((access & ACCESS_WRITE) ? DEV_WRITE : 0);
    impl_->acq.add(m.u, mode ? mode : DEV_READ);
    std::vector<FenceRef> waits;
    if (!impl_->acq.acquire(impl_->s, waits))
        CV_Error(Error::StsError, "accel: DeviceAccess: a conflicting Mat view of the UMat is alive");
    for (size_t i = 0; i < waits.size(); i++)
        streamWait(impl_->s, waits[i]);     // on failure the acquisition is released with impl_
    p = impl_.release();
}

DeviceAccess::~DeviceAccess()
{
    try
    {
        FenceRef done;
        try
        {
            done = recordFence(p->s);
        }
        catch (const std::exception& e)
        {
            // the caller's work is not tracked: what it touched waits for the device, and what
            // it could write is invalid until overwritten
            CV_LOG_ERROR(NULL, "accel: DeviceAccess: can't record the stream position: " << e.what());
            p->acq.fail(fenceAfterFailure(p->s), e.what());
        }
        p->acq.record(done);    // nothing left to record after fail()
    }
    catch (...) {}
    delete p;
}

namespace detail {

void setWaitTimeout(uint64 ms)
{
    impl::waitTimeoutNs().store(ms * 1000000u);
}

void reviveDevices()
{
    impl::ensureInitialized();
    std::vector<impl::Backend*> all = impl::Registry::getInstance().all();
    for (size_t i = 0; i < all.size(); i++)
        all[i]->revive();
}

} // namespace detail

uint64 DeviceAccess::handle() const
{
    return p->m.u->accelState->imp->alloc.handle;
}

size_t DeviceAccess::offset() const
{
    // the UMat's data is a view of a range of the import that handle() refers to
    return (size_t)(p->m.u->accelState->start + p->m.offset);
}

} // namespace accel
} // namespace cv
