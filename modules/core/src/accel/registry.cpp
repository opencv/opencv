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

FenceRef::FenceRef(const FenceRef& o) : b_(o.b_), f_(o.f_), streamUid_(o.streamUid_), device_(o.device_)
{
    if (f_)
        b_->fn().fence_retain(f_);
}

FenceRef& FenceRef::operator=(const FenceRef& o)
{
    if (this != &o)
    {
        if (o.f_)
            o.b_->fn().fence_retain(o.f_);
        reset();
        b_ = o.b_;
        f_ = o.f_;
        streamUid_ = o.streamUid_;
        device_ = o.device_;
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
}

bool FenceRef::isComplete() const
{
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
static uint64_t waitTimeoutNs()
{
    static uint64_t ns = (uint64_t)utils::getConfigurationParameterSizeT("OPENCV_ACCEL_WAIT_TIMEOUT_MS", 0) * 1000000u;
    return ns;
}

void FenceRef::wait() const
{
    if (!f_)
        return;
    counters().hostWaits++;
    const uint64_t timeout = waitTimeoutNs();
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

FenceRef::Poll FenceRef::poll() const CV_NOEXCEPT
{
    if (!f_)
        return DONE;
    const cvAccelStatus st = b_->fn().fence_wait(f_, 0);
    if (st == CV_ACCEL_OK)
        return DONE;
    if (st == CV_ACCEL_TIMEOUT)
        return PENDING;
    // the work behind the fence failed and is no longer running (see fence_wait)
    if (st == CV_ACCEL_DEVICE_LOST)
        b_->markLost(device_);
    try
    {
        CV_LOG_ERROR(NULL, "accel(" << b_->name() << "): device " << device_ << " failed (status " << (int)st
                     << "): " << b_->fn().get_last_error());
    }
    catch (...) {}
    return FAILED;
}

bool FenceRef::waitNoThrow() const CV_NOEXCEPT
{
    try
    {
        wait();
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
    if (b->fn().device_synchronize(s->device) != CV_ACCEL_OK)
        b->markLost(s->device);     // nothing more can be known about what still runs there
    return FenceRef();
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
      enumerated_(false), lost_(false), pinnedCacheBytes_(0)
{
    for (int i = 0; i < MAX_TRACKED_DEVICES; i++)
        lostDevices_[i].store(false);
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
        const_cast<Backend*>(this)->markLost(device);
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

MatAllocator* Backend::allocator(int device, int kind)
{
    std::lock_guard<std::mutex> lk(mutex_);
    enumerate();
    CV_Assert(0 <= device && device < (int)devices_.size());
    const std::pair<int, int> key(device, kind);
    std::map<std::pair<int, int>, MatAllocator*>::const_iterator it = allocators_.find(key);
    if (it != allocators_.end())
        return it->second;
    MatAllocator* a = new DeviceAllocator(this, device, kind);  // never deleted, like the OpenCL allocator
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

void Backend::deferRelease(BufferState* st)
{
    std::lock_guard<std::mutex> lk(mutex_);
    deferred_.push_back(st);
}

void Backend::collectDeferred(bool waitAll)
{
    std::vector<BufferState*> pending;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (deferred_.empty())
            return;
        pending.swap(deferred_);
    }
    // No UMat refers to these buffers any more, so no other thread touches them. This runs
    // at the end of every release, from destructors: it must not throw, and a failed device
    // counts as done (nothing will run there any more; host-side state is freed regardless).
    std::vector<BufferState*> keep;
    for (size_t i = 0; i < pending.size(); i++)
    {
        BufferState* st = pending[i];
        if (waitAll)
        {
            st->writeFence.waitNoThrow();
            for (size_t j = 0; j < st->readFences.size(); j++)
                st->readFences[j].waitNoThrow();
        }
        bool done = st->writeFence.poll() != FenceRef::PENDING;
        for (size_t j = 0; done && j < st->readFences.size(); j++)
            done = st->readFences[j].poll() != FenceRef::PENDING;
        if (done)
        {
            st->writeFence.reset();
            st->readFences.clear();
            destroyBufferState(st);
        }
        else
            keep.push_back(st);
    }
    if (!keep.empty())
    {
        std::lock_guard<std::mutex> lk(mutex_);
        deferred_.insert(deferred_.end(), keep.begin(), keep.end());
    }
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
    if (h.valid_size < sizeof(OpenCV_Core_Accel_API_v0))
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
    CV_LOG_INFO(NULL, "accel: registered backend '" << api->v0.backend_name << "' (" << origin << ", priority " << priority << ")");
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
        const OpenCV_Core_Accel_API* api = init(CV_ACCEL_ABI_VERSION, CV_ACCEL_API_VERSION, NULL);
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

UMat wrapExternalMemory(const Device& device, const ExternalMemory& memory,
                        const MatShape& shape, int type, const size_t* steps)
{
    CV_Assert(!device.empty() && "accel: external memory needs a device that imports it");
    CV_Assert(shape.dims > 0 && shape.dims <= CV_ACCEL_MAX_DIMS);
    Backend* b = backendOf(device);
    DeviceAllocator* a = static_cast<DeviceAllocator*>(b->allocator(device.index(), (int)b->deviceInfo(device.index()).preferred_kind));

    UMat m;
    m.flags = UMat::MAGIC_VAL | CV_MAT_TYPE(type);
    setSize(m, shape.dims, shape.p, steps, steps == 0);
    // bytes covered by the view, with the steps the caller gave
    size_t total = CV_ELEM_SIZE(type);
    for (int i = 0; i < shape.dims; i++)
    {
        CV_Assert(shape.p[i] > 0);
        total += (size_t)(shape.p[i] - 1) * m.step.p[i];
    }
    CV_Assert(memory.size == 0 || total <= memory.size);

    cvAccelExternalMemory ext;
    memset(&ext, 0, sizeof(ext));
    ext.struct_size = (uint32_t)sizeof(ext);
    ext.type = (uint32_t)memory.type;
    ext.handle = (int64_t)memory.handle;
    ext.offset = memory.offset;
    ext.size = memory.size ? memory.size : total;
    ext.flags = memory.readOnly ? CV_ACCEL_MEMF_READ_ONLY : 0;
    m.u = a->wrap(ext, total);
    if (memory.readOnly)
        m.u->accelState->alloc.desc.flags |= CV_ACCEL_MEMF_READ_ONLY;
    m.offset = 0;
    finalizeHdr(m);
    m.addref();
    return m;
}

struct DeviceAccess::Impl
{
    Impl() : acq(true) {}
    UMat m;
    StreamPtr s;
    DeviceAcquisition acq;  //!< a long hold: released when the DeviceAccess goes away
};

DeviceAccess::DeviceAccess(const UMat& m, const Stream& stream, AccessFlag access) : p(0)
{
    CV_Assert(m.u && m.u->accelState && "accel: DeviceAccess needs a UMat allocated on a device");
    CV_Assert(!stream.empty());
    std::unique_ptr<Impl> impl_(new Impl());
    impl_->m = m;
    impl_->s = stream.impl->s;
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

uint64 DeviceAccess::handle() const
{
    return p->m.u->accelState->alloc.handle;
}

size_t DeviceAccess::offset() const
{
    return p->m.offset;
}

} // namespace accel
} // namespace cv
