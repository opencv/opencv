// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: CUDA backend, memory and synchronization primitives.
//
// Kernels are not wired yet (get_op_table reports none), so every operation falls back to
// the CPU; the memory side is complete, which already gives the transfer behaviour of the
// protocol: a pinned host copy so each transfer is a single DMA, no write-back after
// read-only CPU access, uploads only when the device needs the data, events as fences.
//
// This replaces the state machine of cv::cuda::getCudaAllocator() (cuda_umat.cpp, #29451);
// once the DNN CUDA backend has moved to UMat (#29807), getCudaAllocator() can return
// Device::find("cuda").allocator().

#include "../precomp.hpp"
#include "accel_impl.hpp"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda.h>   // driver API types only: libcuda is loaded at run time (see Driver)

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/tls.hpp>

#include <chrono>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cv { namespace accel { namespace impl {

namespace {

TLSData<std::string>& lastErrorTLS()
{
    CV_SINGLETON_LAZY_INIT_REF(TLSData<std::string>, new TLSData<std::string>())
}

bool isStickyError(cudaError_t e)
{
    // these leave the context unusable until the process restarts
    switch (e)
    {
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchFailure:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorECCUncorrectable:
    case cudaErrorAssert:
        return true;
    default:
        return false;
    }
}

cvAccelStatus status(cudaError_t e, const char* what)
{
    if (e == cudaSuccess)
        return CV_ACCEL_OK;
    lastErrorTLS().getRef() = cv::format("%s: %s", what, cudaGetErrorString(e));
    cudaGetLastError();   // clear non-sticky errors
    if (e == cudaErrorMemoryAllocation)
        return CV_ACCEL_OUT_OF_MEMORY;
    if (isStickyError(e))
        return CV_ACCEL_DEVICE_LOST;
    if (e == cudaErrorNotReady)
        return CV_ACCEL_TIMEOUT;
    return CV_ACCEL_INTERNAL_ERROR;
}

//! the runtime's current device is per thread state shared with the user's code: restore it
/** The driver API's context stack. Core links the CUDA runtime only; libcuda comes with the
    driver, so it is loaded at run time and a missing one does not keep OpenCV from loading. */
struct Driver
{
    typedef CUresult (CUDAAPI *Init)(unsigned int);
    typedef CUresult (CUDAAPI *DeviceGet)(CUdevice*, int);
    typedef CUresult (CUDAAPI *PrimaryRetain)(CUcontext*, CUdevice);
    typedef CUresult (CUDAAPI *CtxGet)(CUcontext*);
    typedef CUresult (CUDAAPI *CtxPush)(CUcontext);
    typedef CUresult (CUDAAPI *CtxPop)(CUcontext*);

    Driver() : deviceGet(0), primaryRetain(0), ctxGet(0), ctxPush(0), ctxPop(0), ok(false)
    {
#if defined(_WIN32)
        HMODULE lib = LoadLibraryA("nvcuda.dll");
#define CV_CUDA_DRIVER_SYM(name) (lib ? (void*)GetProcAddress(lib, name) : NULL)
#else
        void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#define CV_CUDA_DRIVER_SYM(name) (lib ? dlsym(lib, name) : NULL)
#endif
        // the library stays loaded: contexts outlive every guard
        Init init = (Init)CV_CUDA_DRIVER_SYM("cuInit");
        deviceGet = (DeviceGet)CV_CUDA_DRIVER_SYM("cuDeviceGet");
        primaryRetain = (PrimaryRetain)CV_CUDA_DRIVER_SYM("cuDevicePrimaryCtxRetain");
        ctxGet = (CtxGet)CV_CUDA_DRIVER_SYM("cuCtxGetCurrent");
        ctxPush = (CtxPush)CV_CUDA_DRIVER_SYM("cuCtxPushCurrent_v2");
        ctxPop = (CtxPop)CV_CUDA_DRIVER_SYM("cuCtxPopCurrent_v2");
#undef CV_CUDA_DRIVER_SYM
        ok = init && deviceGet && primaryRetain && ctxGet && ctxPush && ctxPop && init(0) == CUDA_SUCCESS;
    }

    //! primary context of a runtime device (the ordinals match), retained once; NULL if unavailable
    CUcontext primary(int device)
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (device >= (int)primaries.size())
            primaries.resize(device + 1, (CUcontext)NULL);
        if (!primaries[device])
        {
            CUdevice dev = 0;
            CUcontext ctx = NULL;
            if (deviceGet(&dev, device) == CUDA_SUCCESS && primaryRetain(&ctx, dev) == CUDA_SUCCESS)
                primaries[device] = ctx;
        }
        return primaries[device];
    }

    DeviceGet deviceGet;
    PrimaryRetain primaryRetain;
    CtxGet ctxGet;
    CtxPush ctxPush;
    CtxPop ctxPop;
    bool ok;
    std::mutex mutex;
    std::vector<CUcontext> primaries;
};

Driver& driver()
{
    static Driver* d = new Driver();    // never destroyed: guards may run during static destruction
    return *d;
}

/** Makes the device current for the scope and then restores exactly what the thread had: the
    caller's own context (NVDEC, FFmpeg and other driver-API users push theirs), or none at all.
    Restoring with cudaSetDevice cannot do that: since CUDA 12 it creates the primary context of
    the device it restores, so a thread that never used GPU 0 would get one there (hundreds of
    MB of its memory), and a driver-API context would be replaced by a primary one. */
struct DeviceGuard
{
    explicit DeviceGuard(int device) : pushed(false), prev(-1)
    {
        Driver& d = driver();
        CUcontext ctx = d.ok ? d.primary(device) : NULL;
        CUcontext cur = NULL;
        if (ctx && d.ctxGet(&cur) == CUDA_SUCCESS)
        {
            if (cur != ctx)
                pushed = d.ctxPush(ctx) == CUDA_SUCCESS;
            return;
        }
        // no driver API: the runtime's current device, with the caveat above
        if (cudaGetDevice(&prev) != cudaSuccess || prev == device || cudaSetDevice(device) != cudaSuccess)
            prev = -1;
    }
    ~DeviceGuard()
    {
        if (pushed)
        {
            CUcontext popped = NULL;
            driver().ctxPop(&popped);
        }
        else if (prev >= 0)
            cudaSetDevice(prev);
    }
    bool pushed;
    int prev;
};

struct DeviceProps
{
    cudaDeviceProp prop;
    int concurrentManagedAccess;
    int pageableMemoryAccess;
    int hostRegisterSupported;
    uint32_t sharedHostAccess;   //!< how the CPU sees pinned memory mapped into this GPU
    uint64_t peerEnabled;        //!< bit j: kernels here reach memory of GPU j (peer access is on)
};

std::vector<DeviceProps>& devices()
{
    static std::vector<DeviceProps>* d = 0;
    static std::once_flag once;
    std::call_once(once, []() {
        d = new std::vector<DeviceProps>();
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess)
        {
            cudaGetLastError();
            n = 0;
        }
        d->resize(n);
        for (int i = 0; i < n; i++)
        {
            DeviceProps& p = (*d)[i];
            memset(&p, 0, sizeof(p));
            cudaGetDeviceProperties(&p.prop, i);
            cudaDeviceGetAttribute(&p.concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, i);
            cudaDeviceGetAttribute(&p.pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, i);
            cudaDeviceGetAttribute(&p.hostRegisterSupported, cudaDevAttrHostRegisterSupported, i);
            // Jetson: Xavier (7.2) and later snoop CPU caches for pinned memory; earlier Tegra
            // maps pinned memory uncached on the CPU. Discrete GPUs reach it over PCIe, coherently.
            const int cc = p.prop.major * 10 + p.prop.minor;
            p.sharedHostAccess = (p.prop.integrated && cc < 72) ? CV_ACCEL_HOST_ACCESS_UNCACHED
                                                                : CV_ACCEL_HOST_ACCESS_COHERENT;
        }
        // Peer access maps every allocation of one GPU into the other's address space and makes
        // every later allocation and free on both slower: only on request, once, here (so that
        // can_access stays free of side effects).
        if (n > 1 && utils::getConfigurationParameterBool("OPENCV_ACCEL_CUDA_PEER_ACCESS", false))
        {
            for (int i = 0; i < n && i < 64; i++)
            {
                for (int j = 0; j < n && j < 64; j++)
                {
                    int can = 0;
                    if (i == j || cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can)
                        continue;
                    DeviceGuard guard(i);
                    const cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                    if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled)
                        (*d)[i].peerEnabled |= (uint64_t)1 << j;
                    cudaGetLastError();
                }
            }
        }
    });
    return *d;
}

struct CudaAlloc
{
    int device;
    uint32_t kind;
    void* devPtr;       //!< device address (equal to hostPtr for mapped memory on UVA systems)
    void* hostPtr;      //!< CPU address, NULL for device-local memory
    bool owned;
    bool registered;    //!< imported with cudaHostRegister
};

struct CudaStream
{
    cudaStream_t s;
    int device;
};

struct CudaFence
{
    cudaEvent_t e;
    int device;
    std::atomic<int> refs;
};

cvAccelStatus CV_API_CALL cu_get_device_count(int* count) CV_ACCEL_NOEXCEPT
{
    *count = (int)devices().size();
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_get_device_info(int device, cvAccelDeviceInfo* info) CV_ACCEL_NOEXCEPT
{
    if (device < 0 || device >= (int)devices().size())
        return CV_ACCEL_INVALID_ARGUMENT;
    const DeviceProps& p = devices()[device];
    const uint32_t size = info->struct_size;
    memset(info, 0, sizeof(*info));
    info->struct_size = size;
    info->device_type = p.prop.integrated ? CV_ACCEL_DEVICE_GPU_INTEGRATED : CV_ACCEL_DEVICE_GPU_DISCRETE;
    strncpy(info->name, p.prop.name, sizeof(info->name) - 1);
    info->vendor_id = 0x10de;
    info->total_memory = p.prop.totalGlobalMem;
    info->caps = CV_ACCEL_CAP_ASYNC | CV_ACCEL_CAP_HOST_PINNED | CV_ACCEL_CAP_MANAGED |
                 CV_ACCEL_CAP_STREAMS_THREAD_SAFE;
    if (p.peerEnabled)
        info->caps |= CV_ACCEL_CAP_PEER_ACCESS;
    if (p.prop.integrated)
        info->caps |= CV_ACCEL_CAP_SHARED;
    if (!p.concurrentManagedAccess)
        info->caps |= CV_ACCEL_CAP_DEVICE_IDLE_MEMORY;  // managed memory faults on the CPU while a kernel runs
    if (p.hostRegisterSupported)
    {
        // any address: the driver pins whole pages, and core refuses two imports that share a
        // page (cudaHostRegister would fail with cudaErrorHostMemoryAlreadyRegistered)
        info->caps |= CV_ACCEL_CAP_IMPORT_HOST_PTR;
        info->host_import_alignment = 0;
    }
    info->preferred_kind = p.prop.integrated ? CV_ACCEL_MEM_SHARED : CV_ACCEL_MEM_DEVICE;
    return CV_ACCEL_OK;
}

//! what memory of @p kind on @p device is; one answer for allocate and describe_allocation
uint32_t describeKind(const DeviceProps& p, int device, uint32_t kind, cvAccelMemoryDesc& d)
{
    memset(&d, 0, sizeof(d));
    d.device_index = (uint32_t)device;
    d.base_alignment = 256;
    if (kind == CV_ACCEL_MEM_SHARED && !p.prop.integrated)
        kind = CV_ACCEL_MEM_MANAGED;    // no physically shared memory on a discrete GPU
    switch (kind)
    {
    case CV_ACCEL_MEM_HOST_PINNED:
    case CV_ACCEL_MEM_SHARED:
        d.kind = kind;
        d.host_access = p.sharedHostAccess;
        d.handle_kind = kind == CV_ACCEL_MEM_HOST_PINNED ? CV_ACCEL_HANDLE_HOST_POINTER : CV_ACCEL_HANDLE_DEVICE_POINTER;
        break;
    case CV_ACCEL_MEM_MANAGED:
        d.kind = CV_ACCEL_MEM_MANAGED;
        // without concurrent access the CPU faults on managed memory while any kernel runs
        d.host_access = p.concurrentManagedAccess ? CV_ACCEL_HOST_ACCESS_COHERENT : CV_ACCEL_HOST_ACCESS_DEVICE_IDLE;
        d.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        break;
    default:
        d.kind = CV_ACCEL_MEM_DEVICE;
        d.host_access = CV_ACCEL_HOST_ACCESS_COPY;
        d.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        break;
    }
    return d.kind;
}

cvAccelStatus CV_API_CALL cu_describe_allocation(int device, uint64_t /*size*/, uint32_t kind, uint32_t flags,
                                                 cvAccelMemoryDesc* desc) CV_ACCEL_NOEXCEPT
{
    if (device < 0 || device >= (int)devices().size() || !desc)
        return CV_ACCEL_INVALID_ARGUMENT;
    if (flags & CV_ACCEL_MEMF_PROTECTED)
        return CV_ACCEL_NOT_IMPLEMENTED;    // CUDA has no protected memory
    describeKind(devices()[device], device, kind, *desc);
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_allocate(int device, uint64_t size, uint32_t kind, uint32_t flags,
                                      cvAccelAllocation* out) CV_ACCEL_NOEXCEPT
{
    if (device < 0 || device >= (int)devices().size() || !out)
        return CV_ACCEL_INVALID_ARGUMENT;
    if (flags & CV_ACCEL_MEMF_PROTECTED)
        return CV_ACCEL_NOT_IMPLEMENTED;    // CUDA has no protected memory
    const DeviceProps& p = devices()[device];
    DeviceGuard guard(device);
    memset(out, 0, sizeof(*out));
    CudaAlloc* a = new (std::nothrow) CudaAlloc();
    if (!a)
        return CV_ACCEL_OUT_OF_MEMORY;
    memset(a, 0, sizeof(*a));
    a->device = device;
    a->owned = true;
    cvAccelMemoryDesc& d = out->desc;
    kind = describeKind(p, device, kind, d);
    cudaError_t e = cudaSuccess;
    switch (kind)
    {
    case CV_ACCEL_MEM_HOST_PINNED:
    case CV_ACCEL_MEM_SHARED:
        e = cudaHostAlloc(&a->hostPtr, size, cudaHostAllocPortable | cudaHostAllocMapped);
        if (e == cudaSuccess)
            e = cudaHostGetDevicePointer(&a->devPtr, a->hostPtr, 0);
        out->handle = (uint64_t)(uintptr_t)(kind == CV_ACCEL_MEM_HOST_PINNED ? a->hostPtr : a->devPtr);
        break;
    case CV_ACCEL_MEM_MANAGED:
        e = cudaMallocManaged(&a->devPtr, size, cudaMemAttachGlobal);
        a->hostPtr = a->devPtr;
        out->handle = (uint64_t)(uintptr_t)a->devPtr;
        break;
    default:
        e = cudaMalloc(&a->devPtr, size);
        out->handle = (uint64_t)(uintptr_t)a->devPtr;
        break;
    }
    a->kind = d.kind;
    if (e != cudaSuccess)
    {
        delete a;
        memset(out, 0, sizeof(*out));
        return status(e, "allocate");
    }
    out->size = size;
    out->backend_data = a;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_deallocate(cvAccelAllocation* alloc) CV_ACCEL_NOEXCEPT
{
    CudaAlloc* a = alloc ? (CudaAlloc*)alloc->backend_data : 0;
    if (!a)
        return CV_ACCEL_OK;
    DeviceGuard guard(a->device);
    cudaError_t e = cudaSuccess;
    if (a->registered)
        e = cudaHostUnregister(a->hostPtr);
    else if (a->owned)
    {
        if (a->kind == CV_ACCEL_MEM_HOST_PINNED || a->kind == CV_ACCEL_MEM_SHARED)
            e = cudaFreeHost(a->hostPtr);
        else
            e = cudaFree(a->devPtr);
    }
    delete a;
    alloc->backend_data = 0;
    alloc->handle = 0;
    return status(e, "deallocate");
}

cvAccelStatus CV_API_CALL cu_import_memory(int device, const cvAccelExternalMemory* ext,
                                           cvAccelAllocation* out) CV_ACCEL_NOEXCEPT
{
    if (device < 0 || device >= (int)devices().size() || !ext || !out)
        return CV_ACCEL_INVALID_ARGUMENT;
    DeviceGuard guard(device);
    memset(out, 0, sizeof(*out));
    void* ptr = (uchar*)(intptr_t)ext->handle + ext->offset;
    if (ext->type == CV_ACCEL_EXTMEM_HOST_POINTER)
    {
        if (!devices()[device].hostRegisterSupported)
            return CV_ACCEL_NOT_IMPLEMENTED;
        unsigned flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
        if (ext->flags & CV_ACCEL_MEMF_READ_ONLY)
            flags |= cudaHostRegisterReadOnly;
        cudaError_t e = cudaHostRegister(ptr, ext->size, flags);
        if (e == cudaErrorHostMemoryAlreadyRegistered)
        {
            cudaGetLastError();
            return CV_ACCEL_NOT_IMPLEMENTED;    // overlaps another registration: the caller copies
        }
        if (e != cudaSuccess)
            return status(e, "cudaHostRegister");
        CudaAlloc* a = new CudaAlloc();
        memset(a, 0, sizeof(*a));
        a->device = device;
        a->kind = CV_ACCEL_MEM_HOST_PINNED;
        a->hostPtr = ptr;
        a->registered = true;
        e = cudaHostGetDevicePointer(&a->devPtr, ptr, 0);
        if (e != cudaSuccess)
        {
            cudaHostUnregister(ptr);
            delete a;
            return status(e, "cudaHostGetDevicePointer");
        }
        out->desc.kind = CV_ACCEL_MEM_HOST_PINNED;
        out->desc.host_access = devices()[device].sharedHostAccess;
        out->desc.handle_kind = CV_ACCEL_HANDLE_HOST_POINTER;
        out->handle = (uint64_t)(uintptr_t)ptr;
        out->backend_data = a;
    }
    else if (ext->type == CV_ACCEL_EXTMEM_DEVICE_POINTER)
    {
        // non-owning wrap of memory from another library (cuDNN workspace, a framework tensor)
        cudaPointerAttributes attr;
        cudaError_t e = cudaPointerGetAttributes(&attr, ptr);
        if (e != cudaSuccess)
            return status(e, "cudaPointerGetAttributes");
        if (attr.type != cudaMemoryTypeDevice || attr.device != device)
            return CV_ACCEL_INVALID_ARGUMENT;
        CudaAlloc* a = new CudaAlloc();
        memset(a, 0, sizeof(*a));
        a->device = device;
        a->kind = CV_ACCEL_MEM_DEVICE;
        a->devPtr = ptr;
        a->owned = false;
        out->desc.kind = CV_ACCEL_MEM_DEVICE;
        out->desc.host_access = CV_ACCEL_HOST_ACCESS_COPY;
        out->desc.handle_kind = CV_ACCEL_HANDLE_DEVICE_POINTER;
        out->handle = (uint64_t)(uintptr_t)ptr;
        out->backend_data = a;
    }
    else
        return CV_ACCEL_NOT_IMPLEMENTED;    // TODO: cudaImportExternalMemory for dma-buf / opaque fds
    out->desc.device_index = (uint32_t)device;
    out->desc.flags = CV_ACCEL_MEMF_IMPORTED | (ext->flags & CV_ACCEL_MEMF_READ_ONLY);
    out->size = ext->size;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_can_access(int device, const cvAccelMemoryDesc* desc, int own_device) CV_ACCEL_NOEXCEPT
{
    if (device < 0 || device >= (int)devices().size() || !desc)
        return CV_ACCEL_INVALID_ARGUMENT;
    if (own_device >= 0)
    {
        // mapped pinned and managed memory is visible to every GPU
        if (desc->kind == CV_ACCEL_MEM_HOST_PINNED || desc->kind == CV_ACCEL_MEM_SHARED || desc->kind == CV_ACCEL_MEM_MANAGED)
            return CV_ACCEL_OK;
        // device memory of another GPU: only where peer access was enabled (see devices())
        const uint64_t peers = devices()[device].peerEnabled;
        return own_device < 64 && ((peers >> own_device) & 1) ? CV_ACCEL_OK : CV_ACCEL_NOT_IMPLEMENTED;
    }
    // plain host memory: only with HMM / ATS (Grace Hopper, recent drivers)
    if (desc->kind == CV_ACCEL_MEM_HOST && desc->handle_kind == CV_ACCEL_HANDLE_HOST_POINTER &&
        devices()[device].pageableMemoryAccess)
        return CV_ACCEL_OK;
    return CV_ACCEL_NOT_IMPLEMENTED;
}

cvAccelStatus CV_API_CALL cu_begin_cpu_access(const cvAccelAllocation* alloc, uint64_t offset, uint64_t size,
                                              uint32_t /*access*/, void** host_ptr) CV_ACCEL_NOEXCEPT
{
    CudaAlloc* a = alloc ? (CudaAlloc*)alloc->backend_data : 0;
    if (!a || !host_ptr || offset + size > alloc->size)
        return CV_ACCEL_INVALID_ARGUMENT;
    if (!a->hostPtr)
        return CV_ACCEL_NOT_IMPLEMENTED;    // device-local memory: core uses a host copy
    // pinned and managed memory are coherent (or uncached); ordering is done by core with fences
    *host_ptr = a->hostPtr;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_end_cpu_access(const cvAccelAllocation*, uint64_t, uint64_t, uint32_t) CV_ACCEL_NOEXCEPT
{
    return CV_ACCEL_OK;
}

void* endpointPtr(const cvAccelCopyEndpoint& ep)
{
    if (ep.alloc)
    {
        CudaAlloc* a = (CudaAlloc*)ep.alloc->backend_data;
        return (uchar*)(a->devPtr ? a->devPtr : a->hostPtr) + ep.offset;
    }
    return (uchar*)ep.host + ep.offset;
}

cvAccelStatus copyRecursive(uchar* dst, const uint64_t* dststep, const uchar* src, const uint64_t* srcstep,
                            int dims, const uint64_t* sz, cudaStream_t s)
{
    // with unified addressing cudaMemcpyDefault infers the direction from the pointers
    if (dims == 1)
        return status(cudaMemcpyAsync(dst, src, (size_t)sz[0], cudaMemcpyDefault, s), "cudaMemcpyAsync");
    if (dims == 2)
        return status(cudaMemcpy2DAsync(dst, (size_t)dststep[0], src, (size_t)srcstep[0],
                                        (size_t)sz[1], (size_t)sz[0], cudaMemcpyDefault, s), "cudaMemcpy2DAsync");
    for (uint64_t i = 0; i < sz[0]; i++)
    {
        cvAccelStatus st = copyRecursive(dst + i * dststep[0], dststep + 1, src + i * srcstep[0], srcstep + 1,
                                         dims - 1, sz + 1, s);
        if (st != CV_ACCEL_OK)
            return st;
    }
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_copy(int device, cvAccelStream stream, const cvAccelCopyRegion* r) CV_ACCEL_NOEXCEPT
{
    if (!stream || !r || r->dims <= 0 || r->dims > CV_ACCEL_MAX_DIMS)
        return CV_ACCEL_INVALID_ARGUMENT;
    DeviceGuard guard(device);
    // pageable host memory makes cudaMemcpyAsync stage through a driver buffer and block;
    // core keeps its host copies pinned (CV_ACCEL_COPY_HOST_PINNED) to get one async DMA
    return copyRecursive((uchar*)endpointPtr(r->dst), r->dst.step, (const uchar*)endpointPtr(r->src), r->src.step,
                         r->dims, r->size, ((CudaStream*)stream)->s);
}

cvAccelStatus CV_API_CALL cu_stream_create(int device, uint32_t flags, cvAccelStream* out) CV_ACCEL_NOEXCEPT
{
    DeviceGuard guard(device);
    CudaStream* s = new (std::nothrow) CudaStream();
    if (!s)
        return CV_ACCEL_OUT_OF_MEMORY;
    s->device = device;
    // a blocking stream is ordered with the legacy default stream, which most CUDA code (and
    // cv::cuda::Stream::Null()) uses to hand buffers over without events
    const unsigned cudaFlags = (flags & CV_ACCEL_STREAM_NON_BLOCKING) ? cudaStreamNonBlocking : cudaStreamDefault;
    cudaError_t e = cudaStreamCreateWithFlags(&s->s, cudaFlags);
    if (e != cudaSuccess)
    {
        delete s;
        return status(e, "cudaStreamCreateWithFlags");
    }
    *out = (cvAccelStream)s;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_stream_destroy(cvAccelStream stream) CV_ACCEL_NOEXCEPT
{
    CudaStream* s = (CudaStream*)stream;
    if (!s)
        return CV_ACCEL_OK;
    DeviceGuard guard(s->device);
    cudaError_t e = cudaStreamDestroy(s->s);   // returns at once; pending work still completes
    delete s;
    return status(e, "cudaStreamDestroy");
}

cvAccelStatus CV_API_CALL cu_stream_record_fence(cvAccelStream stream, cvAccelFence* out) CV_ACCEL_NOEXCEPT
{
    CudaStream* s = (CudaStream*)stream;
    DeviceGuard guard(s->device);
    CudaFence* f = new (std::nothrow) CudaFence();
    if (!f)
        return CV_ACCEL_OUT_OF_MEMORY;
    f->device = s->device;
    f->refs = 1;
    cudaError_t e = cudaEventCreateWithFlags(&f->e, cudaEventDisableTiming);
    if (e == cudaSuccess)
        e = cudaEventRecord(f->e, s->s);
    if (e != cudaSuccess)
    {
        delete f;
        return status(e, "cudaEventRecord");
    }
    *out = (cvAccelFence)f;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_stream_wait_fence(cvAccelStream stream, cvAccelFence fence) CV_ACCEL_NOEXCEPT
{
    CudaStream* s = (CudaStream*)stream;
    DeviceGuard guard(s->device);
    return status(cudaStreamWaitEvent(s->s, ((CudaFence*)fence)->e, 0), "cudaStreamWaitEvent");
}

cvAccelStatus CV_API_CALL cu_fence_wait(cvAccelFence fence, uint64_t timeout_ns) CV_ACCEL_NOEXCEPT
{
    CudaFence* f = (CudaFence*)fence;
    if (!f)
        return CV_ACCEL_OK;
    if (timeout_ns == CV_ACCEL_INFINITE)
        return status(cudaEventSynchronize(f->e), "cudaEventSynchronize");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
    for (;;)
    {
        cudaError_t e = cudaEventQuery(f->e);
        if (e != cudaErrorNotReady)
            return status(e, "cudaEventQuery");
        cudaGetLastError();
        if (std::chrono::steady_clock::now() >= deadline)
            return CV_ACCEL_TIMEOUT;
        std::this_thread::yield();
    }
}

cvAccelStatus CV_API_CALL cu_fence_retain(cvAccelFence fence) CV_ACCEL_NOEXCEPT
{
    if (fence)
        ((CudaFence*)fence)->refs++;
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_fence_release(cvAccelFence fence) CV_ACCEL_NOEXCEPT
{
    CudaFence* f = (CudaFence*)fence;
    if (f && --f->refs == 0)
    {
        DeviceGuard guard(f->device);
        cudaEventDestroy(f->e);
        delete f;
    }
    return CV_ACCEL_OK;
}

cvAccelStatus CV_API_CALL cu_device_synchronize(int device) CV_ACCEL_NOEXCEPT
{
    DeviceGuard guard(device);
    return status(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

cvAccelStatus CV_API_CALL cu_get_op_table(const char*, uint32_t, const void** table) CV_ACCEL_NOEXCEPT
{
    // TODO: kernels (own .cu files, or NPP) for the core and imgproc tables
    *table = 0;
    return CV_ACCEL_NOT_IMPLEMENTED;
}

const char* CV_API_CALL cu_get_last_error() CV_ACCEL_NOEXCEPT
{
    return lastErrorTLS().getRef().c_str();
}

const OpenCV_Core_Accel_API g_cudaApi = {
    {
        sizeof(OpenCV_Core_Accel_API),
        CV_ACCEL_ABI_VERSION,
        CV_ACCEL_API_VERSION,
        CV_VERSION_MAJOR,
        CV_VERSION_MINOR,
        CV_VERSION_REVISION,
        CV_VERSION_STATUS,
        "OpenCV CUDA accelerator backend"
    },
    {
        "cuda",
        cu_get_device_count,
        cu_get_device_info,
        cu_allocate,
        cu_deallocate,
        cu_import_memory,
        NULL,               // export_memory: TODO cudaIpc / fabric handles
        cu_can_access,
        cu_begin_cpu_access,
        cu_end_cpu_access,
        cu_copy,
        cu_stream_create,
        cu_stream_destroy,
        cu_stream_record_fence,
        cu_stream_wait_fence,
        cu_fence_wait,
        cu_fence_retain,
        cu_fence_release,
        NULL,               // fence_export: TODO cudaExternalSemaphore
        NULL,               // stream_wait_external
        cu_device_synchronize,
        cu_get_op_table,
        cu_get_last_error,
        cu_describe_allocation
    }
};

} // namespace

const OpenCV_Core_Accel_API* cudaBackendApi()
{
    return &g_cudaApi;
}

}}} // namespace cv::accel::impl

#endif // HAVE_CUDA
