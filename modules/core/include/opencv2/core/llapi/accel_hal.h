// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_LLAPI_ACCEL_HAL_H
#define OPENCV_CORE_LLAPI_ACCEL_HAL_H

/**
@addtogroup core_lowlevel_api
@{

Non-CPU HAL: the C interface between OpenCV and an accelerator backend (discrete or
integrated GPU, NPU, DSP, 2D engine, ...).

A backend provides three groups of primitives:
- memory: allocate, import, describe and copy buffers, and bracket CPU access
  (begin_cpu_access / end_cpu_access) so that OpenCV can keep the CPU view coherent;
- streams and fences: in-order execution queues and completion points;
- operations: per-module tables of typed functions (core, imgproc, ...).

OpenCV core owns the coherence protocol (which copy is current, which fences must be
waited for, when caches must be cleaned or invalidated). A backend never changes UMat
state itself; it only executes the primitives it is asked to execute.

Compatibility rules:
- structs with a struct_size field grow at their end only. The side that fills one writes
  at most struct_size bytes of it, and the side that reads one reads only the fields that
  lie within struct_size (the others count as zero);
- structs without one (cvAccelMemoryDesc, cvAccelAllocation, cvAccelView,
  cvAccelCopyEndpoint) keep their size, since they are embedded or passed in arrays: new
  fields take their reserved space, which the side that creates them sets to zero;
- a table entry outside of the size reported by the backend reads as NULL, and the plugin
  loader asks for the highest API version first;
- NULL entries and CV_ACCEL_NOT_IMPLEMENTED both mean "not supported, no side effects";
- cvAccelStatus is 32 bits wide whatever the compiler's enum size (see its MAX_ENUM).

Rules a backend must follow, because core relies on them for correctness:
1. Report the true host-access class of every allocation, decided at run time (the same
   SoC family can be coherent on one part and not on the next). begin_cpu_access must
   never copy behind core's back: memory that needs a copy (strided, tiled, device-only)
   is reported as CV_ACCEL_HOST_ACCESS_COPY or CV_ACCEL_HOST_ACCESS_NONE.
2. A completed fence makes every device write made before it visible to the CPU after
   begin_cpu_access (a system-scope release; e.g. hipEventReleaseToSystem).
3. Allocations with CV_ACCEL_HOST_ACCESS_NONCOHERENT start and end on coherence_granule
   boundaries, so that cache maintenance of one buffer never touches bytes of another.
4. begin_cpu_access / end_cpu_access may widen the range (dma-buf syncs whole buffers,
   Vulkan whole atoms) and must retry interrupted system calls (EINTR / EAGAIN).
5. An imported dma-buf may carry implicit fences from other drivers (camera, decoder,
   display). Work enqueued on it waits for them, and after writing, the backend attaches
   its own fence to the dma-buf (DMA_BUF_IOCTL_EXPORT_SYNC_FILE / IMPORT_SYNC_FILE).
6. A backend without CV_ACCEL_CAP_ASYNC runs work during the enqueue call and returns
   fences that are already complete.
7. Imported host memory is never freed or reused by the backend; overlapping imports are
   refused by core, which falls back to a copy.
8. Once a call has returned CV_ACCEL_DEVICE_LOST for a device, waits on its fences return
   promptly with an error instead of blocking, and fence_release / stream_destroy still
   succeed. Core then frees no more allocations of that device (they are leaked, as after a
   timed-out wait, see OPENCV_ACCEL_WAIT_TIMEOUT_MS) but keeps releasing its own host-side
   state, so that destroying a UMat never blocks or throws.
9. An operation or copy that returns CV_ACCEL_NOT_IMPLEMENTED or CV_ACCEL_OUT_OF_MEMORY has
   queued nothing (core falls back, or frees memory and retries). After any other error part
   of the work may be queued: core records a fence after it (or synchronizes the device) and
   treats everything the work writes as invalid until it is written whole.

@note Preview only (no backward compatibility until ABI version 1).
*/

#include <stddef.h>
#include <stdint.h>

#include "opencv2/core/llapi/llapi.h"

#ifdef __cplusplus
extern "C" {
#endif

//! exceptions must never cross this boundary
#ifdef __cplusplus
#  define CV_ACCEL_NOEXCEPT noexcept
#else
#  define CV_ACCEL_NOEXCEPT
#endif

#define CV_ACCEL_ABI_VERSION 0
//! 1: describe_allocation
#define CV_ACCEL_API_VERSION 1

//! same as cv::MatShape::MAX_DIMS
#define CV_ACCEL_MAX_DIMS 10

//! timeout value meaning "wait forever"
#define CV_ACCEL_INFINITE ((uint64_t)-1)

/** @brief Flags of stream_create. */
typedef enum cvAccelStreamFlags
{
    CV_ACCEL_STREAM_NON_BLOCKING = 1 << 0  //!< no implicit ordering with the device API's default stream
} cvAccelStreamFlags;

typedef enum cvAccelStatus
{
    CV_ACCEL_OK = 0,
    CV_ACCEL_NOT_IMPLEMENTED = 1,   //!< unsupported arguments or feature; nothing was changed
    CV_ACCEL_OUT_OF_MEMORY = 2,     //!< nothing was changed; the caller may free caches and retry
    CV_ACCEL_INVALID_ARGUMENT = 3,
    CV_ACCEL_TIMEOUT = 4,           //!< a wait expired, or a fence is still pending (timeout 0)
    CV_ACCEL_DEVICE_LOST = 5,       //!< the device can no longer be used
    CV_ACCEL_INTERNAL_ERROR = 6,
    CV_ACCEL_STATUS_MAX_ENUM = 0x7FFFFFFF   //!< keeps the type 32 bits wide (-fshort-enums, DSP toolchains)
} cvAccelStatus;

typedef enum cvAccelDeviceType
{
    CV_ACCEL_DEVICE_OTHER = 0,
    CV_ACCEL_DEVICE_GPU_DISCRETE = 1,
    CV_ACCEL_DEVICE_GPU_INTEGRATED = 2,
    CV_ACCEL_DEVICE_NPU = 3,
    CV_ACCEL_DEVICE_DSP = 4,
    CV_ACCEL_DEVICE_2D_ENGINE = 5,  //!< fixed-function blit/scale/convert unit (RGA, VIC, ...)
    CV_ACCEL_DEVICE_FPGA = 6
} cvAccelDeviceType;

/** @brief What kind of memory an allocation is. */
typedef enum cvAccelMemoryKind
{
    CV_ACCEL_MEM_HOST = 0,          //!< ordinary pageable host memory
    CV_ACCEL_MEM_HOST_PINNED = 1,   //!< page-locked host memory: DMA-capable, may be mapped into the device
    CV_ACCEL_MEM_DEVICE = 2,        //!< device-local memory (VRAM on a discrete GPU, carve-out on an SoC)
    CV_ACCEL_MEM_SHARED = 3,        //!< one physical allocation addressed by both CPU and device (UMA, SVM, dma-buf)
    CV_ACCEL_MEM_MANAGED = 4        //!< migrating unified memory (cudaMallocManaged, hipMallocManaged, USM shared)
} cvAccelMemoryKind;

/** @brief How the CPU may touch an allocation. This drives the coherence protocol in core.

| class        | CPU view                         | what core does around CPU access                 |
|--------------|----------------------------------|--------------------------------------------------|
| NONE         | none (protected, tiled, ...)     | refuses; CPU fallback is an error                |
| COPY         | host shadow                      | DMA into / out of the shadow, lazily             |
| COHERENT     | same bytes, HW-coherent caches   | waits for fences only                            |
| NONCOHERENT  | same bytes, CPU caches not snooped | waits for fences, then begin/end_cpu_access    |
| UNCACHED     | same bytes, uncached or WC       | treated as COPY for CPU kernels (reads are slow) |
| DEVICE_IDLE  | same bytes, only if device idle  | synchronizes the whole device first              |
*/
typedef enum cvAccelHostAccess
{
    CV_ACCEL_HOST_ACCESS_NONE = 0,
    CV_ACCEL_HOST_ACCESS_COPY = 1,
    CV_ACCEL_HOST_ACCESS_COHERENT = 2,
    CV_ACCEL_HOST_ACCESS_NONCOHERENT = 3,
    CV_ACCEL_HOST_ACCESS_UNCACHED = 4,
    CV_ACCEL_HOST_ACCESS_DEVICE_IDLE = 5
} cvAccelHostAccess;

/** @brief Meaning of cvAccelAllocation::handle. */
typedef enum cvAccelHandleKind
{
    CV_ACCEL_HANDLE_NONE = 0,
    CV_ACCEL_HANDLE_HOST_POINTER = 1,   //!< CPU address
    CV_ACCEL_HANDLE_DEVICE_POINTER = 2, //!< flat device address: handle + offset is a valid device address
    CV_ACCEL_HANDLE_CL_MEM = 3,         //!< OpenCL cl_mem; offsets are passed separately
    CV_ACCEL_HANDLE_VK_BUFFER = 4,      //!< Vulkan VkBuffer
    CV_ACCEL_HANDLE_DMABUF_FD = 5,      //!< Linux dma-buf file descriptor
    CV_ACCEL_HANDLE_OPAQUE = 6          //!< backend-private object
} cvAccelHandleKind;

/** @brief cvAccelMemoryDesc::flags */
enum
{
    CV_ACCEL_MEMF_IMPORTED = 1 << 0,    //!< memory is owned outside of OpenCV (user pointer, dma-buf, ...)
    CV_ACCEL_MEMF_READ_ONLY = 1 << 1,   //!< writes are not allowed (e.g. imported read-only memory)
    CV_ACCEL_MEMF_PROTECTED = 1 << 2    //!< protected content: never readable by the CPU
};

/** @brief Where an allocation lives and how it can be accessed.

Reported by the allocator that owns the memory (MatAllocator::getMemoryDesc), never
guessed from the allocator's identity. Two allocations with equal backend_id,
device_index and handle_kind can be used by the same kernels.
*/
typedef struct cvAccelMemoryDesc
{
    uint32_t backend_id;        //!< id given by the registry; 0 for host memory and unknown allocators
    uint32_t device_index;      //!< device of that backend
    uint32_t kind;              //!< cvAccelMemoryKind
    uint32_t host_access;       //!< cvAccelHostAccess
    uint32_t handle_kind;       //!< cvAccelHandleKind
    uint32_t flags;             //!< CV_ACCEL_MEMF_*
    uint64_t layout_modifier;   //!< 0 = linear; otherwise a DRM-style format modifier with no linear CPU view
    uint32_t base_alignment;    //!< alignment of the allocation start, bytes
    uint32_t coherence_granule; //!< NONCOHERENT: cache line or atom size cache operations work on; else 0
    uint32_t reserved[6];       //!< zero; room for new fields (the size stays)
} cvAccelMemoryDesc;

/** @brief One physical allocation, filled in by the backend. */
typedef struct cvAccelAllocation
{
    uint64_t handle;            //!< see cvAccelHandleKind; 64 bits so device addresses fit on 32-bit hosts
    uint64_t size;              //!< bytes requested (allocate) or imported, never rounded up: whole-buffer copies use it
    void* backend_data;         //!< private to the backend
    cvAccelMemoryDesc desc;
    uint64_t reserved[2];       //!< zero; room for new fields (the size stays)
} cvAccelAllocation;

/** @brief An n-dimensional strided view into an allocation (an argument of an operation).

In QUERY mode alloc->handle is 0 and backend_data is NULL; only the description is valid.
Core checks before every EXECUTE call that the bytes covered by the view lie inside the
allocation, so a backend never receives a view that would make the device access memory
outside of the buffer.
*/
typedef struct cvAccelView
{
    const cvAccelAllocation* alloc;
    uint64_t offset;                    //!< bytes from the allocation start to the first element
    int32_t type;                       //!< OpenCV type, e.g. CV_8UC3
    int32_t dims;
    int32_t size[CV_ACCEL_MAX_DIMS];
    uint64_t step[CV_ACCEL_MAX_DIMS];   //!< bytes; step[dims-1] is the element size
    uint64_t reserved[2];               //!< zero; room for new fields (views are passed in arrays)
} cvAccelView;

typedef struct cvAccelStream_t* cvAccelStream; //!< in-order queue, owned by the backend
typedef struct cvAccelFence_t* cvAccelFence;   //!< completion point on a stream, reference counted

/** @brief cvAccelCall::mode */
enum
{
    CV_ACCEL_QUERY = 0,     //!< decide whether the call is supported; no handles, no side effects
    CV_ACCEL_EXECUTE = 1    //!< enqueue the operation on cvAccelCall::stream
};

/** @brief Common first argument of every operation. */
typedef struct cvAccelCall
{
    uint32_t struct_size;
    uint32_t mode;          //!< CV_ACCEL_QUERY or CV_ACCEL_EXECUTE
    int32_t device;
    cvAccelStream stream;   //!< EXECUTE only
    const char* op_name;    //!< e.g. "core.flip", for logs
} cvAccelCall;

/** @brief cvAccelCopyEndpoint: either alloc or host is set. */
typedef struct cvAccelCopyEndpoint
{
    const cvAccelAllocation* alloc;
    void* host;
    uint64_t offset;
    uint64_t step[CV_ACCEL_MAX_DIMS];
    uint64_t reserved[2];       //!< zero; room for new fields (the size stays)
} cvAccelCopyEndpoint;

/** @brief cvAccelCopyRegion::flags */
enum
{
    CV_ACCEL_COPY_HOST_PINNED = 1 << 0  //!< the host endpoint is pinned memory from this backend
};

/** @brief n-dimensional strided copy; size[dims-1] is in bytes (MatAllocator convention). */
typedef struct cvAccelCopyRegion
{
    uint32_t struct_size;
    uint32_t flags;
    int32_t dims;
    uint64_t size[CV_ACCEL_MAX_DIMS];
    cvAccelCopyEndpoint src;
    cvAccelCopyEndpoint dst;
} cvAccelCopyRegion;

/** @brief External memory for import / export. */
typedef enum cvAccelExternalMemoryType
{
    CV_ACCEL_EXTMEM_HOST_POINTER = 1,   //!< plain CPU memory (register for DMA or zero-copy use)
    CV_ACCEL_EXTMEM_DMABUF_FD = 2,      //!< Linux dma-buf
    CV_ACCEL_EXTMEM_OPAQUE_FD = 3,      //!< Vulkan / CUDA / HIP opaque POSIX fd
    CV_ACCEL_EXTMEM_OPAQUE_WIN32 = 4,   //!< Windows NT handle
    CV_ACCEL_EXTMEM_DEVICE_POINTER = 5, //!< native device pointer of the same backend (non-owning)
    CV_ACCEL_EXTMEM_AHARDWAREBUFFER = 6 //!< Android AHardwareBuffer*
} cvAccelExternalMemoryType;

typedef struct cvAccelExternalMemory
{
    uint32_t struct_size;
    uint32_t type;          //!< cvAccelExternalMemoryType
    int64_t handle;         //!< pointer, fd or native handle
    uint64_t offset;
    uint64_t size;
    uint32_t flags;         //!< CV_ACCEL_MEMF_READ_ONLY, CV_ACCEL_MEMF_PROTECTED
} cvAccelExternalMemory;

/** @brief External fence types for cross-device synchronization without a CPU wait. */
typedef enum cvAccelExternalFenceType
{
    CV_ACCEL_EXTFENCE_SYNC_FD = 1,      //!< Linux sync_file
    CV_ACCEL_EXTFENCE_OPAQUE_FD = 2,
    CV_ACCEL_EXTFENCE_OPAQUE_WIN32 = 3
} cvAccelExternalFenceType;

/** @brief cvAccelDeviceInfo::caps */
enum
{
    CV_ACCEL_CAP_ASYNC = 1 << 0,            //!< operations and copies return before completion
    CV_ACCEL_CAP_HOST_PINNED = 1 << 1,      //!< can allocate CV_ACCEL_MEM_HOST_PINNED
    CV_ACCEL_CAP_SHARED = 1 << 2,           //!< can allocate CV_ACCEL_MEM_SHARED
    CV_ACCEL_CAP_MANAGED = 1 << 3,          //!< can allocate CV_ACCEL_MEM_MANAGED
    CV_ACCEL_CAP_IMPORT_HOST_PTR = 1 << 4,  //!< import of user host memory without a copy
    CV_ACCEL_CAP_IMPORT_DMABUF = 1 << 5,
    CV_ACCEL_CAP_EXPORT_DMABUF = 1 << 6,
    CV_ACCEL_CAP_EXTERNAL_FENCE = 1 << 7,   //!< fence_export / stream_wait_external
    CV_ACCEL_CAP_PEER_ACCESS = 1 << 8,      //!< kernels reach memory of another device of this backend (can_access tells which)
    CV_ACCEL_CAP_STREAMS_THREAD_SAFE = 1 << 9, //!< one stream may be used from several threads at once
    /** may hand out memory that the CPU may touch only while the device is idle
        (CV_ACCEL_HOST_ACCESS_DEVICE_IDLE); core then gates every submission from the start */
    CV_ACCEL_CAP_DEVICE_IDLE_MEMORY = 1 << 10
};

typedef struct cvAccelDeviceInfo
{
    uint32_t struct_size;
    uint32_t device_type;               //!< cvAccelDeviceType
    char name[128];
    uint32_t vendor_id;
    uint32_t caps;                      //!< CV_ACCEL_CAP_*
    uint64_t total_memory;
    uint32_t host_import_alignment;     //!< required alignment of imported host pointers (0: any)
    uint32_t preferred_kind;            //!< cvAccelMemoryKind used for UMat data by default
} cvAccelDeviceInfo;

/** @brief Backend entry points, API version 0. */
struct OpenCV_Core_Accel_API_v0_0_api_entries
{
    //! short lowercase name used in device specs and logs: "cuda", "hip", "rknpu", ...
    const char* backend_name;

    //! @name Devices
    //! @{
    cvAccelStatus (CV_API_CALL *get_device_count)(int* count) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *get_device_info)(int device, cvAccelDeviceInfo* info) CV_ACCEL_NOEXCEPT;
    //! @}

    //! @name Memory
    //! @{
    /** Allocates @p size bytes of the requested kind. The backend may return another kind
        (e.g. MANAGED when SHARED is requested on a discrete GPU) but desc must describe the
        result truthfully. backend_id in desc is filled in by core. @p flags may hold
        CV_ACCEL_MEMF_PROTECTED: protected memory, for results computed from protected
        content; a device without any returns CV_ACCEL_NOT_IMPLEMENTED. */
    cvAccelStatus (CV_API_CALL *allocate)(int device, uint64_t size, uint32_t kind, uint32_t flags,
                                          cvAccelAllocation* out) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *deallocate)(cvAccelAllocation* alloc) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *import_memory)(int device, const cvAccelExternalMemory* ext,
                                               cvAccelAllocation* out) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *export_memory)(const cvAccelAllocation* alloc, uint32_t type,
                                               cvAccelExternalMemory* out) CV_ACCEL_NOEXCEPT;
    /** OK when kernels of @p device can read and write memory described by @p desc in place.
        @p own_device is the device index when the memory was allocated or imported by this
        backend, and -1 when it belongs to the host or to another backend. Core never asks
        about the backend's own memory on the same device: that is always accessible. Core
        asks while planning, so this has no side effects: what it reports is set up beforehand
        (peer access, for instance, when the devices are enumerated). */
    cvAccelStatus (CV_API_CALL *can_access)(int device, const cvAccelMemoryDesc* desc, int own_device) CV_ACCEL_NOEXCEPT;
    /** Starts CPU access to [offset, offset+size) with access flags (cv::AccessFlag values).
        Called after all device work on the allocation finished. For NONCOHERENT memory this
        invalidates CPU caches when reading (dma-buf SYNC_START); for mapped APIs it maps.
        Returns the CPU address of the allocation start. */
    cvAccelStatus (CV_API_CALL *begin_cpu_access)(const cvAccelAllocation* alloc, uint64_t offset,
                                                  uint64_t size, uint32_t access, void** host_ptr) CV_ACCEL_NOEXCEPT;
    /** Ends CPU access. For NONCOHERENT memory written by the CPU this cleans (flushes) CPU
        caches before any device may touch the memory (dma-buf SYNC_END). */
    cvAccelStatus (CV_API_CALL *end_cpu_access)(const cvAccelAllocation* alloc, uint64_t offset,
                                                uint64_t size, uint32_t access) CV_ACCEL_NOEXCEPT;
    /** Enqueues a strided copy (host to device, device to host or device to device). Host
        memory must stay valid until the stream passes this point. */
    cvAccelStatus (CV_API_CALL *copy)(int device, cvAccelStream stream, const cvAccelCopyRegion* region) CV_ACCEL_NOEXCEPT;
    //! @}

    //! @name Streams and fences
    //! @{
    /** Creates an in-order queue. Unless @p flags hold CV_ACCEL_STREAM_NON_BLOCKING, its work
        is ordered with the device API's default stream (the CUDA/HIP legacy stream), so that
        a kernel the caller launches there after UMat::handle() runs before OpenCV's next use
        of the buffer, as with an OpenCL queue that both share. */
    cvAccelStatus (CV_API_CALL *stream_create)(int device, uint32_t flags, cvAccelStream* out) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *stream_destroy)(cvAccelStream stream) CV_ACCEL_NOEXCEPT;
    //! new fence (reference count 1) that completes when all work enqueued so far completes
    cvAccelStatus (CV_API_CALL *stream_record_fence)(cvAccelStream stream, cvAccelFence* out) CV_ACCEL_NOEXCEPT;
    //! device-side wait: later work on @p stream starts after @p fence completes
    cvAccelStatus (CV_API_CALL *stream_wait_fence)(cvAccelStream stream, cvAccelFence fence) CV_ACCEL_NOEXCEPT;
    /** CPU wait; timeout 0 polls (CV_ACCEL_TIMEOUT while pending). Any status other than OK
        and TIMEOUT means the work behind the fence failed and is no longer running: core then
        frees the memory that work used. */
    cvAccelStatus (CV_API_CALL *fence_wait)(cvAccelFence fence, uint64_t timeout_ns) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *fence_retain)(cvAccelFence fence) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *fence_release)(cvAccelFence fence) CV_ACCEL_NOEXCEPT;
    //! optional (CV_ACCEL_CAP_EXTERNAL_FENCE): a new handle (e.g. sync_file fd) owned by the caller
    cvAccelStatus (CV_API_CALL *fence_export)(cvAccelFence fence, uint32_t type, int64_t* handle) CV_ACCEL_NOEXCEPT;
    //! optional: later work on @p stream waits for an external fence; takes ownership of @p handle even on failure
    cvAccelStatus (CV_API_CALL *stream_wait_external)(cvAccelStream stream, uint32_t type, int64_t handle) CV_ACCEL_NOEXCEPT;
    /** Waits for all work on the device. Before CPU access to DEVICE_IDLE memory core calls
        this and holds back new work on the device until the CPU access ends (on Jetson
        before Thor, the CPU faults on any managed allocation while any kernel runs). */
    cvAccelStatus (CV_API_CALL *device_synchronize)(int device) CV_ACCEL_NOEXCEPT;
    //! @}

    //! @name Operations
    //! @{
    /** Returns the operation table for an OpenCV module ("core", "imgproc", ...). The table
        starts with a size_t holding its size in bytes. NOT_IMPLEMENTED when the backend has
        no operations for that module or version. */
    cvAccelStatus (CV_API_CALL *get_op_table)(const char* module, uint32_t version, const void** table) CV_ACCEL_NOEXCEPT;
    //! @}

    //! message for the last error on the calling thread (never NULL)
    const char* (CV_API_CALL *get_last_error)(void) CV_ACCEL_NOEXCEPT;

    // Entries below were appended later: tables of older backends end above, and core reads
    // what they lack as NULL.

    /** optional: the description allocate() would give @p size bytes of @p kind on @p device,
        without allocating. Core asks while it plans a call whose outputs or temporaries do not
        exist yet, so that the QUERY step sees the memory the operation will really get. */
    cvAccelStatus (CV_API_CALL *describe_allocation)(int device, uint64_t size, uint32_t kind, uint32_t flags,
                                                     cvAccelMemoryDesc* desc) CV_ACCEL_NOEXCEPT;
};

typedef struct OpenCV_Core_Accel_API_v0
{
    OpenCV_API_Header api_header;
    struct OpenCV_Core_Accel_API_v0_0_api_entries v0;
} OpenCV_Core_Accel_API_v0;

typedef OpenCV_Core_Accel_API_v0 OpenCV_Core_Accel_API;

//! plugin entry point, exported by opencv_core_accel_<name> libraries
typedef const OpenCV_Core_Accel_API* (CV_API_CALL *FN_opencv_core_accel_plugin_init_t)
        (int requested_abi_version, int requested_api_version, void* reserved /*NULL*/);

/////////////////////////////// Operation tables ///////////////////////////////

/** @brief True when @p table (of struct type @p T) is long enough to contain @p field and the entry is set. */
#define CV_ACCEL_OP_PRESENT(T, table, field) \
    ((table) != NULL && offsetof(T, field) + sizeof(((T*)0)->field) <= (table)->struct_size && (table)->field != NULL)

#define CV_ACCEL_OPS_CORE "core"
#define CV_ACCEL_OPS_CORE_VERSION 0

/** @brief core operations. Append-only. */
typedef struct cvAccelOpsCore_v0
{
    size_t struct_size;
    cvAccelStatus (CV_API_CALL *flip)(const cvAccelCall* call, const cvAccelView* src,
                                      const cvAccelView* dst, int flip_mode) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *transpose)(const cvAccelCall* call, const cvAccelView* src,
                                           const cvAccelView* dst) CV_ACCEL_NOEXCEPT;
} cvAccelOpsCore_v0;

#ifdef __cplusplus
} // extern "C"
#endif

//! @}

#endif // OPENCV_CORE_LLAPI_ACCEL_HAL_H
