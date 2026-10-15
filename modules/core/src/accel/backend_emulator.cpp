// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: a backend that runs on the CPU but behaves like four accelerators with
// different memory models, so that the coherence protocol can be tested anywhere.
//
// device 0, discrete:    device memory holds the bytes XOR-scrambled; reaching it without a
//                        transfer gives garbage. flip and transpose only move bytes, so they
//                        work on the scrambled form unchanged.
// device 1, coherent:    shared memory; can also use host memory in place (like UMA).
// device 2, noncoherent: shared memory with a separate CPU view standing in for CPU caches;
//                        only begin_cpu_access (invalidate) and end_cpu_access (clean) move
//                        data between the two, so a missing cache operation shows as wrong data.
// device 3, protected:   no CPU access at all.
// device 4, managed:     managed memory without concurrent access (Jetson before Thor,
//                        Windows): CPU access while any device work runs on the device is
//                        counted as a violation instead of faulting.
//
// Every stream is a worker thread; OPENCV_ACCEL_EMULATOR_DELAY_US delays each task.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/tls.hpp>

#include <chrono>
#include <deque>
#include <functional>
#include <set>
#include <thread>

namespace cv { namespace accel { namespace impl {

namespace {

enum { EMU_DISCRETE = 0, EMU_COHERENT = 1, EMU_NONCOHERENT = 2, EMU_PROTECTED = 3, EMU_MANAGED = 4,
       EMU_FAILING = 5, EMU_DEVICES = 6 };

const uchar SCRAMBLE = 0x5A;
const uchar POISON = 0xCD;

TLSData<std::string>& lastErrorTLS()
{
    CV_SINGLETON_LAZY_INIT_REF(TLSData<std::string>, new TLSData<std::string>())
}

cvAccelStatus fail(cvAccelStatus st, const std::string& msg)
{
    lastErrorTLS().getRef() = msg;
    return st;
}

std::atomic<long long> g_liveAllocations(0);
std::atomic<long long> g_liveByDevice[EMU_DEVICES];   // zero: static storage

void countAllocation(int device, int delta)
{
    g_liveAllocations += delta;
    g_liveByDevice[device] += delta;
}

// EMU_MANAGED: what would fault on real hardware
std::atomic<int> g_managedCpuAccess(0);
std::atomic<int> g_managedDeviceWork(0);
std::atomic<long long> g_violations(0);

// EMU_FAILING: the status its work reports (emulatorInjectFailure)
std::atomic<int> g_failingStatus(CV_ACCEL_OK);
std::atomic<bool> g_failSynchronously(false);

//! what a fence recorded now on @p device reports once the work before it is done
cvAccelStatus asyncFailure(int device)
{
    return device == EMU_FAILING && !g_failSynchronously.load() ? (cvAccelStatus)g_failingStatus.load() : CV_ACCEL_OK;
}

//! what a call on @p device returns after queuing its work (a backend failing halfway through)
cvAccelStatus syncFailure(int device)
{
    const cvAccelStatus st = device == EMU_FAILING && g_failSynchronously.load() ?
                             (cvAccelStatus)g_failingStatus.load() : CV_ACCEL_OK;
    return st == CV_ACCEL_OK ? st : fail(st, "emulated failure after queuing the work");
}

//! a lost device rejects every call and never makes anyone wait (backend rule 8)
bool deviceLost(int device)
{
    return device == EMU_FAILING && g_failingStatus.load() == CV_ACCEL_DEVICE_LOST;
}

cvAccelStatus lostError()
{
    return fail(CV_ACCEL_DEVICE_LOST, "emulator:5 is lost");
}

struct EmuAlloc
{
    int device;
    uint32_t kind;
    size_t size;
    uchar* mem;         //!< what the device reads and writes
    uchar* cpuView;     //!< what the CPU sees: a stand-in for CPU caches on the noncoherent device
    bool owned;         //!< false for imported user memory
    bool scrambled;     //!< bytes are stored XOR SCRAMBLE
    bool cpuOpen;       //!< between begin_cpu_access and end_cpu_access
};

bool isScrambledDesc(const cvAccelMemoryDesc& d, int device)
{
    if (d.kind == CV_ACCEL_MEM_HOST || d.kind == CV_ACCEL_MEM_HOST_PINNED || d.handle_kind == CV_ACCEL_HANDLE_HOST_POINTER)
        return false;
    return device == EMU_DISCRETE || device == EMU_PROTECTED || device == EMU_FAILING;
}

uchar* devicePtr(const cvAccelAllocation* a)
{
    return a->backend_data ? ((EmuAlloc*)a->backend_data)->mem : (uchar*)(uintptr_t)a->handle;
}

bool isScrambled(const cvAccelAllocation* a)
{
    return a && a->backend_data && ((EmuAlloc*)a->backend_data)->scrambled;
}

struct EmuFence
{
    explicit EmuFence(int device_)
        : done(false), refs(1), device(device_), status(asyncFailure(device_)) {}
    std::mutex m;
    std::condition_variable cv;
    bool done;
    std::atomic<int> refs;
    int device;
    cvAccelStatus status;   //!< what the work before the fence reports once it is done

    void signal()
    {
        std::lock_guard<std::mutex> lk(m);
        done = true;
        cv.notify_all();
    }
    bool wait(uint64_t timeout_ns)
    {
        std::unique_lock<std::mutex> lk(m);
        if (timeout_ns == CV_ACCEL_INFINITE)
        {
            cv.wait(lk, [this]{ return done; });
            return true;
        }
        return cv.wait_for(lk, std::chrono::nanoseconds(timeout_ns), [this]{ return done; });
    }
    void release()
    {
        if (--refs == 0)
            delete this;
    }
};

struct EmuStream;
//! the worker of a stream: runs its queue, and deletes the stream once it is retired and drained
void runStream(EmuStream* s);

struct EmuStream
{
    explicit EmuStream(int device_) : device(device_), stop(false), retired(false)
    {
        delayUs = utils::getConfigurationParameterSizeT("OPENCV_ACCEL_EMULATOR_DELAY_US", 0);
    }
    /** stream_destroy, called with streamsMutex held: returns at once, like cudaStreamDestroy.
        The worker finishes the queue and then deletes the stream; joining it here would
        deadlock, since core destroys a thread's streams during OpenCV's thread-exit cleanup,
        under a lock that the exiting worker needs for its own cleanup. Allocates nothing. */
    void retire()
    {
        retired = true;
        std::lock_guard<std::mutex> lk(m);  // notify under the lock: the worker cannot delete us before we let go
        stop = true;
        cv.notify_all();
    }
    //! @p deviceWork: a kernel or a copy (as opposed to fence bookkeeping)
    void push(std::function<void()> task, bool deviceWork = false)
    {
        {
            std::lock_guard<std::mutex> lk(m);
            q.push_back(Task(std::move(task), deviceWork));
        }
        cv.notify_all();
    }
    void loop()
    {
        for (;;)
        {
            Task task;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this]{ return stop || !q.empty(); });
                if (q.empty())
                    return;
                task = std::move(q.front());
                q.pop_front();
            }
            if (delayUs)
                std::this_thread::sleep_for(std::chrono::microseconds(delayUs));
            const bool managedWork = task.second && device == EMU_MANAGED;
            if (managedWork)
            {
                g_managedDeviceWork++;
                if (g_managedCpuAccess.load() > 0)
                    g_violations++;     // the CPU is inside managed memory while the device runs
            }
            try
            {
                task.first();
            }
            catch (const std::exception& e)
            {
                CV_LOG_ERROR(NULL, "accel(emulator): task failed: " << e.what());
            }
            if (managedWork)
                g_managedDeviceWork--;
        }
    }

    typedef std::pair<std::function<void()>, bool> Task;
    int device;
    size_t delayUs;
    std::mutex m;
    std::condition_variable cv;
    std::deque<Task> q;
    bool stop;
    bool retired;   //!< destroyed by its owner, still draining (under streamsMutex)
};

std::mutex& streamsMutex()
{
    static std::mutex* m = new std::mutex();
    return *m;
}

//! every stream whose worker runs, retired ones included until they drain (under streamsMutex)
std::set<EmuStream*>& liveStreams()
{
    static std::set<EmuStream*>* s = new std::set<EmuStream*>();
    return *s;
}

//! signalled when a retired stream has drained and left liveStreams
std::condition_variable& streamsDrained()
{
    static std::condition_variable* c = new std::condition_variable();
    return *c;
}

void runStream(EmuStream* s)
{
    s->loop();  // returns once the stream is retired and its queue is empty
    {
        std::lock_guard<std::mutex> lk(streamsMutex());
        liveStreams().erase(s);
        streamsDrained().notify_all();
    }
    delete s;
}

//! n-dimensional strided copy, sz[dims-1] in bytes; XOR when exactly one side is scrambled
void copyND(uchar* dst, const uint64_t* dststep, const uchar* src, const uint64_t* srcstep,
            int dims, const uint64_t* sz, bool xorBytes)
{
    if (dims <= 1)
    {
        if (!xorBytes)
            memcpy(dst, src, (size_t)sz[0]);
        else
            for (uint64_t i = 0; i < sz[0]; i++)
                dst[i] = src[i] ^ SCRAMBLE;
        return;
    }
    for (uint64_t i = 0; i < sz[0]; i++)
        copyND(dst + i * dststep[0], dststep + 1, src + i * srcstep[0], srcstep + 1, dims - 1, sz + 1, xorBytes);
}

////////////////////////////////// entry points //////////////////////////////////

cvAccelStatus CV_API_CALL emu_get_device_count(int* count) CV_ACCEL_NOEXCEPT
{
    if (!count)
        return CV_ACCEL_INVALID_ARGUMENT;
    *count = EMU_DEVICES;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL emu_get_device_info(int device, cvAccelDeviceInfo* info) CV_ACCEL_NOEXCEPT
{
    if (!info || device < 0 || device >= EMU_DEVICES)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad device");
    static const char* names[EMU_DEVICES] = {
        "emulated discrete GPU", "emulated coherent shared memory",
        "emulated non-coherent shared memory", "emulated protected memory",
        "emulated managed memory without concurrent access", "emulated discrete GPU for failure tests" };
    static const uint32_t types[EMU_DEVICES] = {
        CV_ACCEL_DEVICE_GPU_DISCRETE, CV_ACCEL_DEVICE_GPU_INTEGRATED, CV_ACCEL_DEVICE_NPU, CV_ACCEL_DEVICE_OTHER,
        CV_ACCEL_DEVICE_GPU_INTEGRATED, CV_ACCEL_DEVICE_GPU_DISCRETE };
    static const uint32_t preferred[EMU_DEVICES] = {
        CV_ACCEL_MEM_DEVICE, CV_ACCEL_MEM_SHARED, CV_ACCEL_MEM_SHARED, CV_ACCEL_MEM_DEVICE, CV_ACCEL_MEM_MANAGED,
        CV_ACCEL_MEM_DEVICE };
    cvAccelDeviceInfo d;
    memset(&d, 0, sizeof(d));
    d.device_type = types[device];
    strncpy(d.name, names[device], sizeof(d.name) - 1);
    d.caps = CV_ACCEL_CAP_ASYNC | CV_ACCEL_CAP_HOST_PINNED | CV_ACCEL_CAP_STREAMS_THREAD_SAFE;
    if (device == EMU_COHERENT || device == EMU_NONCOHERENT)
        d.caps |= CV_ACCEL_CAP_SHARED;
    if (device == EMU_MANAGED)
        d.caps |= CV_ACCEL_CAP_MANAGED | CV_ACCEL_CAP_DEVICE_IDLE_MEMORY;
    if (device == EMU_COHERENT)
    {
        d.caps |= CV_ACCEL_CAP_IMPORT_HOST_PTR;
        d.host_import_alignment = 64;
    }
    d.total_memory = (uint64_t)1 << 32;
    d.preferred_kind = preferred[device];
    // the caller's struct may be older (smaller) than ours: write what it has room for
    const uint32_t size = info->struct_size;
    memcpy(info, &d, std::min<size_t>(size, sizeof(d)));
    info->struct_size = size;
    return CV_ACCEL_OK;
}

/** What memory of @p kind on @p device is: one answer for allocate and describe_allocation.
    False when the device has none: protected memory exists on the discrete-like devices only,
    as on a GPU with a secure mode. */
bool describeKind(int device, uint32_t kind, uint32_t flags, cvAccelMemoryDesc& d)
{
    memset(&d, 0, sizeof(d));
    d.device_index = (uint32_t)device;
    d.base_alignment = CV_MALLOC_ALIGN;
    if (flags & CV_ACCEL_MEMF_PROTECTED)
    {
        if (kind == CV_ACCEL_MEM_HOST_PINNED || !(device == EMU_DISCRETE || device == EMU_FAILING || device == EMU_PROTECTED))
            return false;
        d.kind = CV_ACCEL_MEM_DEVICE;
        d.host_access = CV_ACCEL_HOST_ACCESS_NONE;
        d.handle_kind = device == EMU_PROTECTED ? CV_ACCEL_HANDLE_OPAQUE : CV_ACCEL_HANDLE_DEVICE_POINTER;
        d.flags = CV_ACCEL_MEMF_PROTECTED;
        return true;
    }
    if (kind == CV_ACCEL_MEM_HOST_PINNED)
    {
        d.kind = CV_ACCEL_MEM_HOST_PINNED;
        d.host_access = CV_ACCEL_HOST_ACCESS_COHERENT;
        d.handle_kind = CV_ACCEL_HANDLE_HOST_POINTER;
        return true;
    }
    switch (device)
    {
    case EMU_DISCRETE:
    case EMU_FAILING:
        d.kind = CV_ACCEL_MEM_DEVICE;
        d.host_access = CV_ACCEL_HOST_ACCESS_COPY;
        d.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        break;
    case EMU_COHERENT:
        d.kind = CV_ACCEL_MEM_SHARED;
        d.host_access = CV_ACCEL_HOST_ACCESS_COHERENT;
        d.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        break;
    case EMU_NONCOHERENT:
        d.kind = CV_ACCEL_MEM_SHARED;
        d.host_access = CV_ACCEL_HOST_ACCESS_NONCOHERENT;
        d.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        d.coherence_granule = 64;
        break;
    case EMU_MANAGED:
        d.kind = CV_ACCEL_MEM_MANAGED;
        d.host_access = CV_ACCEL_HOST_ACCESS_DEVICE_IDLE;
        d.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        break;
    default:
        d.kind = CV_ACCEL_MEM_DEVICE;
        d.host_access = CV_ACCEL_HOST_ACCESS_NONE;
        d.handle_kind = CV_ACCEL_HANDLE_OPAQUE;
        d.flags |= CV_ACCEL_MEMF_PROTECTED;
        break;
    }
    return true;
}

cvAccelStatus CV_API_CALL emu_describe_allocation(int device, uint64_t /*size*/, uint32_t kind, uint32_t flags,
                                                  cvAccelMemoryDesc* desc) CV_ACCEL_NOEXCEPT
{
    if (!desc || device < 0 || device >= EMU_DEVICES)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    if (!describeKind(device, kind, flags, *desc))
        return fail(CV_ACCEL_NOT_IMPLEMENTED, "no protected memory on this device");
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL emu_allocate(int device, uint64_t size, uint32_t kind, uint32_t flags,
                                       cvAccelAllocation* out) CV_ACCEL_NOEXCEPT
{
    if (!out || device < 0 || device >= EMU_DEVICES)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    if (deviceLost(device))
        return lostError();
    cvAccelMemoryDesc desc;
    if (!describeKind(device, kind, flags, desc))
        return fail(CV_ACCEL_NOT_IMPLEMENTED, "no protected memory on this device");
    try
    {
        memset(out, 0, sizeof(*out));
        EmuAlloc* a = new EmuAlloc();
        a->device = device;
        a->size = (size_t)size;
        a->owned = true;
        a->mem = (uchar*)fastMalloc(a->size ? a->size : 1);
        a->cpuView = a->mem;
        a->cpuOpen = false;
        cvAccelMemoryDesc& d = out->desc;
        d = desc;
        a->kind = d.kind;
        a->scrambled = isScrambledDesc(d, device);
        if (d.host_access == CV_ACCEL_HOST_ACCESS_NONCOHERENT)
        {
            a->cpuView = (uchar*)fastMalloc(a->size ? a->size : 1);
            memset(a->cpuView, POISON, a->size);   // stale "cache lines"
        }
        out->handle = (uint64_t)(uintptr_t)a->mem;
        out->size = size;
        out->backend_data = a;
        countAllocation(device, 1);
        return CV_ACCEL_OK;
    }
    catch (const std::bad_alloc&)
    {
        return fail(CV_ACCEL_OUT_OF_MEMORY, "out of memory");
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_deallocate(cvAccelAllocation* alloc) CV_ACCEL_NOEXCEPT
{
    if (!alloc || !alloc->backend_data)
        return CV_ACCEL_OK;
    EmuAlloc* a = (EmuAlloc*)alloc->backend_data;
    if (a->cpuView != a->mem)
        fastFree(a->cpuView);
    if (a->owned)
        fastFree(a->mem);
    countAllocation(a->device, -1);
    delete a;
    alloc->backend_data = 0;
    alloc->handle = 0;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL emu_import_memory(int device, const cvAccelExternalMemory* ext,
                                            cvAccelAllocation* out) CV_ACCEL_NOEXCEPT
{
    if (!ext || !out)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    if (device == EMU_DISCRETE && ext->type == CV_ACCEL_EXTMEM_DEVICE_POINTER)
    {
        // memory of another emulator allocation on this device, like a cudaMalloc pointer
        // from another library: stored scrambled, reached by the CPU only through copies
        try
        {
            memset(out, 0, sizeof(*out));
            EmuAlloc* a = new EmuAlloc();
            a->device = device;
            a->kind = CV_ACCEL_MEM_DEVICE;
            a->size = (size_t)ext->size;
            a->mem = a->cpuView = (uchar*)(intptr_t)ext->handle + ext->offset;
            a->owned = false;
            a->scrambled = true;
            a->cpuOpen = false;
            // a protected source (a secure decoder's frame, say) stays protected
            const bool prot = (ext->flags & CV_ACCEL_MEMF_PROTECTED) != 0;
            out->desc.device_index = (uint32_t)device;
            out->desc.kind = CV_ACCEL_MEM_DEVICE;
            out->desc.host_access = prot ? CV_ACCEL_HOST_ACCESS_NONE : CV_ACCEL_HOST_ACCESS_COPY;
            out->desc.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
            out->desc.flags = CV_ACCEL_MEMF_IMPORTED | (ext->flags & (CV_ACCEL_MEMF_READ_ONLY | CV_ACCEL_MEMF_PROTECTED));
            out->handle = (uint64_t)(uintptr_t)a->mem;
            out->size = ext->size;
            out->backend_data = a;
            countAllocation(device, 1);
            return CV_ACCEL_OK;
        }
        catch (const std::exception& e)
        {
            return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
        }
    }
    if (device != EMU_COHERENT || ext->type != CV_ACCEL_EXTMEM_HOST_POINTER)
        return fail(CV_ACCEL_NOT_IMPLEMENTED, "the coherent device imports host memory, the discrete one device pointers");
    try
    {
        memset(out, 0, sizeof(*out));
        EmuAlloc* a = new EmuAlloc();
        a->device = device;
        a->kind = CV_ACCEL_MEM_SHARED;
        a->size = (size_t)ext->size;
        a->mem = a->cpuView = (uchar*)(intptr_t)ext->handle + ext->offset;
        a->owned = false;
        a->scrambled = false;
        a->cpuOpen = false;
        out->desc.device_index = (uint32_t)device;
        out->desc.kind = CV_ACCEL_MEM_SHARED;
        out->desc.host_access = CV_ACCEL_HOST_ACCESS_COHERENT;
        out->desc.handle_kind = CV_ACCEL_HANDLE_HOST_POINTER;
        out->desc.flags = CV_ACCEL_MEMF_IMPORTED | (ext->flags & CV_ACCEL_MEMF_READ_ONLY);
        out->handle = (uint64_t)(uintptr_t)a->mem;
        out->size = ext->size;
        out->backend_data = a;
        countAllocation(device, 1);
        return CV_ACCEL_OK;
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_can_access(int device, const cvAccelMemoryDesc* desc, int own_device) CV_ACCEL_NOEXCEPT
{
    if (!desc)
        return CV_ACCEL_INVALID_ARGUMENT;
    if (own_device >= 0)
        return CV_ACCEL_NOT_IMPLEMENTED;    // no peer access between emulated devices
    // like a unified-memory GPU, the coherent device addresses ordinary host memory
    if (device == EMU_COHERENT && desc->handle_kind == CV_ACCEL_HANDLE_HOST_POINTER &&
        (desc->kind == CV_ACCEL_MEM_HOST || desc->kind == CV_ACCEL_MEM_HOST_PINNED))
        return CV_ACCEL_OK;
    return CV_ACCEL_NOT_IMPLEMENTED;
}

cvAccelStatus CV_API_CALL emu_begin_cpu_access(const cvAccelAllocation* alloc, uint64_t offset, uint64_t size,
                                               uint32_t /*access*/, void** host_ptr) CV_ACCEL_NOEXCEPT
{
    if (!alloc || !alloc->backend_data || !host_ptr || offset + size > alloc->size)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    EmuAlloc* a = (EmuAlloc*)alloc->backend_data;
    if (deviceLost(a->device))
        return lostError();
    if (a->device == EMU_MANAGED && a->kind == CV_ACCEL_MEM_MANAGED)
    {
        if (!a->cpuOpen)
        {
            a->cpuOpen = true;
            g_managedCpuAccess++;
        }
        if (g_managedDeviceWork.load() > 0)
            g_violations++;     // real hardware would fault here
        *host_ptr = a->mem;
        return CV_ACCEL_OK;
    }
    if (a->kind == CV_ACCEL_MEM_HOST_PINNED || !a->owned || a->device == EMU_COHERENT)
    {
        *host_ptr = a->mem;
        return CV_ACCEL_OK;
    }
    if (a->device == EMU_NONCOHERENT)
    {
        // invalidate: drop stale CPU lines, the CPU now sees what the device wrote
        memcpy(a->cpuView + offset, a->mem + offset, (size_t)size);
        *host_ptr = a->cpuView;
        return CV_ACCEL_OK;
    }
    return fail(CV_ACCEL_NOT_IMPLEMENTED, "this memory has no CPU mapping");
}

cvAccelStatus CV_API_CALL emu_end_cpu_access(const cvAccelAllocation* alloc, uint64_t offset, uint64_t size,
                                             uint32_t access) CV_ACCEL_NOEXCEPT
{
    if (!alloc || !alloc->backend_data || offset + size > alloc->size)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    EmuAlloc* a = (EmuAlloc*)alloc->backend_data;
    if (a->device == EMU_MANAGED && a->cpuOpen)
    {
        a->cpuOpen = false;
        g_managedCpuAccess--;
    }
    if (a->device == EMU_NONCOHERENT && a->owned && a->kind != CV_ACCEL_MEM_HOST_PINNED && (access & (uint32_t)ACCESS_WRITE))
    {
        // clean: write CPU lines back so that the device sees them
        memcpy(a->mem + offset, a->cpuView + offset, (size_t)size);
    }
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL emu_copy(int device, cvAccelStream stream, const cvAccelCopyRegion* r) CV_ACCEL_NOEXCEPT
{
    if (!stream || !r || r->dims <= 0 || r->dims > CV_ACCEL_MAX_DIMS)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    if (deviceLost(device))
        return lostError();
    if ((!r->src.alloc && !r->src.host) || (!r->dst.alloc && !r->dst.host))
        return fail(CV_ACCEL_INVALID_ARGUMENT, "copy endpoint without memory");
    try
    {
        uchar* src = (r->src.alloc ? devicePtr(r->src.alloc) : (uchar*)r->src.host) + r->src.offset;
        uchar* dst = (r->dst.alloc ? devicePtr(r->dst.alloc) : (uchar*)r->dst.host) + r->dst.offset;
        const bool xorBytes = isScrambled(r->src.alloc) != isScrambled(r->dst.alloc);
        const int dims = r->dims;
        std::vector<uint64_t> sz(r->size, r->size + dims);
        std::vector<uint64_t> sstep(r->src.step, r->src.step + dims), dstep(r->dst.step, r->dst.step + dims);
        ((EmuStream*)stream)->push([=]() {
            copyND(dst, dstep.data(), src, sstep.data(), dims, sz.data(), xorBytes);
        }, true);
        return syncFailure(device);
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_stream_create(int device, uint32_t /*flags*/, cvAccelStream* out) CV_ACCEL_NOEXCEPT
{
    if (!out || device < 0 || device >= EMU_DEVICES)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    if (deviceLost(device))
        return lostError();
    try
    {
        std::unique_ptr<EmuStream> s(new EmuStream(device));
        {
            std::lock_guard<std::mutex> lk(streamsMutex());
            liveStreams().insert(s.get());
        }
        try
        {
            std::thread(runStream, s.get()).detach();   // from here on the worker owns the stream
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lk(streamsMutex());
            liveStreams().erase(s.get());
            throw;
        }
        *out = (cvAccelStream)s.release();
        return CV_ACCEL_OK;
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_stream_destroy(cvAccelStream stream) CV_ACCEL_NOEXCEPT
{
    if (!stream)
        return CV_ACCEL_OK;
    try
    {
        std::lock_guard<std::mutex> lk(streamsMutex());
        ((EmuStream*)stream)->retire();     // the worker drains the queue, then deletes the stream
    }
    catch (...)
    {
        return CV_ACCEL_INTERNAL_ERROR;     // a mutex that cannot be locked: the stream stays alive
    }
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL emu_stream_record_fence(cvAccelStream stream, cvAccelFence* out) CV_ACCEL_NOEXCEPT
{
    if (!stream || !out)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    if (deviceLost(((EmuStream*)stream)->device))
        return lostError();
    try
    {
        EmuFence* f = new EmuFence(((EmuStream*)stream)->device);
        f->refs++;  // one reference for the task
        ((EmuStream*)stream)->push([f]() { f->signal(); f->release(); });
        *out = (cvAccelFence)f;
        return CV_ACCEL_OK;
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_stream_wait_fence(cvAccelStream stream, cvAccelFence fence) CV_ACCEL_NOEXCEPT
{
    if (!stream || !fence)
        return fail(CV_ACCEL_INVALID_ARGUMENT, "bad argument");
    try
    {
        EmuFence* f = (EmuFence*)fence;
        f->refs++;
        ((EmuStream*)stream)->push([f]() { f->wait(CV_ACCEL_INFINITE); f->release(); });
        return CV_ACCEL_OK;
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_fence_wait(cvAccelFence fence, uint64_t timeout_ns) CV_ACCEL_NOEXCEPT
{
    if (!fence)
        return CV_ACCEL_OK;
    EmuFence* f = (EmuFence*)fence;
    if (deviceLost(f->device))
        return lostError();
    if (!f->wait(timeout_ns))
        return CV_ACCEL_TIMEOUT;
    return f->status == CV_ACCEL_OK ? CV_ACCEL_OK : fail(f->status, "emulated device failure");
}

cvAccelStatus CV_API_CALL emu_fence_retain(cvAccelFence fence) CV_ACCEL_NOEXCEPT
{
    if (fence)
        ((EmuFence*)fence)->refs++;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL emu_fence_release(cvAccelFence fence) CV_ACCEL_NOEXCEPT
{
    if (fence)
        ((EmuFence*)fence)->release();
    return CV_ACCEL_OK;
}

//! fences that are released when the holder goes away, whatever happens in between
struct FenceList
{
    ~FenceList()
    {
        for (size_t i = 0; i < fences.size(); i++)
            fences[i]->release();
    }
    std::vector<EmuFence*> fences;
};

cvAccelStatus CV_API_CALL emu_device_synchronize(int device) CV_ACCEL_NOEXCEPT
{
    if (deviceLost(device))
        return lostError();
    try
    {
        FenceList list;
        std::unique_lock<std::mutex> lk(streamsMutex());
        list.fences.reserve(liveStreams().size());
        for (std::set<EmuStream*>::iterator it = liveStreams().begin(); it != liveStreams().end(); ++it)
        {
            if ((*it)->device != device || (*it)->retired)
                continue;
            cvAccelFence f = 0;
            if (emu_stream_record_fence((cvAccelStream)*it, &f) == CV_ACCEL_OK)
                list.fences.push_back((EmuFence*)f);    // capacity is reserved: never throws
        }
        // retired streams drain on their own; their workers leave liveStreams when done
        streamsDrained().wait(lk, [device]() {
            for (std::set<EmuStream*>::const_iterator it = liveStreams().begin(); it != liveStreams().end(); ++it)
                if ((*it)->device == device && (*it)->retired)
                    return false;
            return true;
        });
        lk.unlock();
        for (size_t i = 0; i < list.fences.size(); i++)
            list.fences[i]->wait(CV_ACCEL_INFINITE);
        return CV_ACCEL_OK;
    }
    catch (const std::bad_alloc&)
    {
        return CV_ACCEL_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return CV_ACCEL_INTERNAL_ERROR;
    }
}

////////////////////////////////// operations //////////////////////////////////

bool sameFormat(const cvAccelCall* call, const cvAccelView* a, const cvAccelView* b)
{
    return isScrambledDesc(a->alloc->desc, call->device) == isScrambledDesc(b->alloc->desc, call->device);
}

//! like a real kernel, the emulator must know how to address memory (a device pointer, ...)
bool addressable(const cvAccelView* v)
{
    return v->alloc && v->alloc->desc.handle_kind != CV_ACCEL_HANDLE_NONE;
}

Mat viewToMat(const cvAccelView* v)
{
    return Mat(v->size[0], v->size[1], v->type, devicePtr(v->alloc) + v->offset, (size_t)v->step[0]);
}

cvAccelStatus CV_API_CALL emu_flip(const cvAccelCall* call, const cvAccelView* src, const cvAccelView* dst,
                                   int flipMode) CV_ACCEL_NOEXCEPT
{
    if (deviceLost(call->device))
        return lostError();
    try
    {
        if (!addressable(src) || !addressable(dst) || src->dims != 2 || dst->dims != 2 || src->type != dst->type)
            return CV_ACCEL_NOT_IMPLEMENTED;
        if (src->size[0] != dst->size[0] || src->size[1] != dst->size[1])
            return fail(CV_ACCEL_INVALID_ARGUMENT, "flip: size mismatch");
        if (!sameFormat(call, src, dst))
            return CV_ACCEL_NOT_IMPLEMENTED;
        if (call->mode == CV_ACCEL_QUERY)
            return CV_ACCEL_OK;
        Mat s = viewToMat(src), d = viewToMat(dst);
        ((EmuStream*)call->stream)->push([s, d, flipMode]() mutable { cv::flip(s, d, flipMode); }, true);
        return syncFailure(call->device);
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

cvAccelStatus CV_API_CALL emu_transpose(const cvAccelCall* call, const cvAccelView* src, const cvAccelView* dst) CV_ACCEL_NOEXCEPT
{
    if (deviceLost(call->device))
        return lostError();
    try
    {
        if (!addressable(src) || !addressable(dst) || src->dims != 2 || dst->dims != 2 || src->type != dst->type)
            return CV_ACCEL_NOT_IMPLEMENTED;
        if (src->size[0] != dst->size[1] || src->size[1] != dst->size[0])
            return fail(CV_ACCEL_INVALID_ARGUMENT, "transpose: size mismatch");
        if (CV_ELEM_SIZE(src->type) > 8 || !sameFormat(call, src, dst))
            return CV_ACCEL_NOT_IMPLEMENTED;   // like a real backend, not every type is covered
        if (call->mode == CV_ACCEL_QUERY)
            return CV_ACCEL_OK;
        Mat s = viewToMat(src), d = viewToMat(dst);
        ((EmuStream*)call->stream)->push([s, d]() mutable { cv::transpose(s, d); }, true);
        return syncFailure(call->device);
    }
    catch (const std::exception& e)
    {
        return fail(CV_ACCEL_INTERNAL_ERROR, e.what());
    }
}

const cvAccelOpsCore_v0 g_coreOps = {
    sizeof(cvAccelOpsCore_v0),
    emu_flip,
    emu_transpose
};

cvAccelStatus CV_API_CALL emu_get_op_table(const char* module, uint32_t version, const void** table) CV_ACCEL_NOEXCEPT
{
    if (!module || !table)
        return CV_ACCEL_INVALID_ARGUMENT;
    if (strcmp(module, CV_ACCEL_OPS_CORE) == 0 && version == CV_ACCEL_OPS_CORE_VERSION)
    {
        *table = &g_coreOps;
        return CV_ACCEL_OK;
    }
    return CV_ACCEL_NOT_IMPLEMENTED;
}

const char* CV_API_CALL emu_get_last_error() CV_ACCEL_NOEXCEPT
{
    return lastErrorTLS().getRef().c_str();
}

const OpenCV_Core_Accel_API g_emulatorApi = {
    {
        sizeof(OpenCV_Core_Accel_API),
        CV_ACCEL_ABI_VERSION,
        CV_ACCEL_API_VERSION,
        CV_VERSION_MAJOR,
        CV_VERSION_MINOR,
        CV_VERSION_REVISION,
        CV_VERSION_STATUS,
        "OpenCV accelerator emulator"
    },
    {
        "emulator",
        emu_get_device_count,
        emu_get_device_info,
        emu_allocate,
        emu_deallocate,
        emu_import_memory,
        NULL,               // export_memory
        emu_can_access,
        emu_begin_cpu_access,
        emu_end_cpu_access,
        emu_copy,
        emu_stream_create,
        emu_stream_destroy,
        emu_stream_record_fence,
        emu_stream_wait_fence,
        emu_fence_wait,
        emu_fence_retain,
        emu_fence_release,
        NULL,               // fence_export
        NULL,               // stream_wait_external
        emu_device_synchronize,
        emu_get_op_table,
        emu_get_last_error,
        emu_describe_allocation
    }
};

} // namespace

const OpenCV_Core_Accel_API* emulatorBackendApi()
{
    return &g_emulatorApi;
}

}} // namespace accel::impl

namespace accel { namespace detail {

int64 emulatorLiveAllocations(int device)
{
    if (device < 0 || device >= impl::EMU_DEVICES)
        return (int64)impl::g_liveAllocations.load();
    return (int64)impl::g_liveByDevice[device].load();
}

int64 emulatorViolations()
{
    return (int64)impl::g_violations.load();
}

void emulatorInjectFailure(int status, bool synchronous)
{
    impl::g_failSynchronously.store(synchronous);
    impl::g_failingStatus.store(status);
}

}} // namespace accel::detail

} // namespace cv
