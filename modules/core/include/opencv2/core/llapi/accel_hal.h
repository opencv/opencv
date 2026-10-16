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

Core keeps one state per piece of physical memory, whoever imported it: the same dma-buf
imported by a GPU backend and by an NPU backend, two planes of one frame, or a UMat's memory
wrapped again through its device pointer all share one ordering of device work, one record
of failed writes and one CPU mapping. Core identifies dma-bufs and host memory itself; for
other memory it uses the identity a backend reports (cvAccelAllocation::backing_id).

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
   Vulkan whole atoms) and must retry interrupted system calls (EINTR / EAGAIN). They nest:
   core calls end_cpu_access once for every begin_cpu_access that succeeded, in reverse order
   and with the same access, and a nested begin returns the address the outer one returned (a
   backend whose API does not nest, such as AHardwareBuffer_lock, counts the levels itself).
   After a failed end_cpu_access the CPU access counts as still open: core calls
   end_cpu_access again before any device work may use the memory.
5. Imported memory that carries implicit fences of other drivers (a dma-buf from a camera,
   decoder or display) is flagged CV_ACCEL_MEMF_EXTERNAL_SYNC, and every dma-buf counts as
   such whatever the backend reports. A backend that imports such memory implements
   begin_device_access and end_device_access; core refuses the import otherwise. Core brackets
   every device use of it, OpenCV's and the caller's (DeviceAccess), with them; when
   end_device_access fails, core waits on the CPU for the work before the call that queued it
   returns. begin_cpu_access waits for those fences as DMA_BUF_IOCTL_SYNC does.
6. A backend without CV_ACCEL_CAP_ASYNC runs work during the enqueue call and returns
   fences that are already complete.
7. Imported host memory is never freed or reused by the backend; overlapping imports are
   refused by core, which falls back to a copy.
8. CV_ACCEL_DEVICE_LOST means that the device has stopped: no work of it runs and none of
   its DMA touches memory any more (a backend resets or tears down the device context before
   it reports the status). From then on waits on its fences return promptly with an error,
   and fence_release / stream_destroy still succeed. Core frees no more device allocations of
   that device and keeps releasing its own host-side state, so that destroying a UMat never
   blocks or throws. When core cannot establish that work stopped (a fence cannot be
   recorded after a failure and device_synchronize fails with another status, or a wait timed
   out, see OPENCV_ACCEL_WAIT_TIMEOUT_MS), it leaks every byte that work may touch, and memory
   that belongs to someone else is not handed back until device_synchronize reports OK or
   CV_ACCEL_DEVICE_LOST.
9. An operation or copy that returns CV_ACCEL_NOT_IMPLEMENTED or CV_ACCEL_OUT_OF_MEMORY has
   queued nothing (core falls back, or frees memory and retries). After any other error part
   of the work may be queued: core records a fence after it (or synchronizes the device) and
   treats everything the work writes as invalid until it is written whole.
10. begin_cpu_access on memory flagged CV_ACCEL_MEMF_READ_ONLY returns a read-only mapping
   where the platform allows it (a dma-buf mmap'ed PROT_READ), and then flags the memory
   CV_ACCEL_MEMF_CPU_MAPPING_READ_ONLY. Core gives every other CPU view of read-only memory
   through a copy in pages it maps read-only itself (host copies of such memory are such
   pages), and refuses CPU access when it cannot protect them: a write through a Mat view of
   read-only memory always faults.

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
/** 1: describe_allocation; 2: begin_device_access / end_device_access; 3: cvAccelAllocation
    reports the size of the memory it lies in and has reserved space, views carry the image
    they are a plane of */
#define CV_ACCEL_API_VERSION 3

//! same as cv::MatShape::MAX_DIMS
#define CV_ACCEL_MAX_DIMS 10

//! planes of one image (cvAccelImageDesc)
#define CV_ACCEL_MAX_PLANES 4

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
    CV_ACCEL_MEMF_PROTECTED = 1 << 2,   //!< protected content: never readable by the CPU
    CV_ACCEL_MEMF_EXTERNAL_SYNC = 1 << 3, //!< carries implicit fences of other drivers (dma-buf): see begin_device_access
    /** begin_cpu_access returns a mapping that the CPU cannot write (writes fault), e.g. a
        dma-buf mapped PROT_READ (rule 10) */
    CV_ACCEL_MEMF_CPU_MAPPING_READ_ONLY = 1 << 4
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

/** @brief One allocation or import, filled in by the backend.

The backing fields tell core which physical memory the allocation lies in, so that every
allocation and import of one memory shares one state in core. A backend that does not know
leaves them zero, and the allocation is its own memory. When the size of the memory is known,
core imports all of it once per backend and device, and passes ranges of that import to
operations.
*/
typedef struct cvAccelAllocation
{
    uint64_t handle;            //!< see cvAccelHandleKind; 64 bits so device addresses fit on 32-bit hosts
    uint64_t size;              //!< bytes requested (allocate) or imported, never rounded up: whole-buffer copies use it
    void* backend_data;         //!< private to the backend
    cvAccelMemoryDesc desc;
    /** identity of the physical memory: the same for every allocation and import that reach it
        while it lives (its base address in a flat device address space, a buffer id, ...), 0 when
        not known. Ignored for dma-bufs and host memory, which core identifies itself. */
    uint64_t backing_id;
    uint64_t backing_offset;    //!< where byte 0 of this allocation lies in that memory
    uint64_t backing_size;      //!< size of that memory (cuMemGetAddressRange, ...), 0 when not known
    uint64_t reserved[5];       //!< zero; room for new fields (the size stays)
} cvAccelAllocation;

/** @brief Colour interpretation of an image (cvAccelImageDesc, cv::accel::ColorInfo). 0 is "unknown". */
typedef enum cvAccelColorModel
{
    CV_ACCEL_COLOR_MODEL_UNKNOWN = 0,
    CV_ACCEL_COLOR_MODEL_RGB = 1,       //!< no YCbCr matrix (RGB or grey data)
    CV_ACCEL_COLOR_MODEL_BT601 = 2,
    CV_ACCEL_COLOR_MODEL_BT709 = 3,
    CV_ACCEL_COLOR_MODEL_BT2020 = 4     //!< non-constant luminance
} cvAccelColorModel;

typedef enum cvAccelColorRange
{
    CV_ACCEL_COLOR_RANGE_UNKNOWN = 0,
    CV_ACCEL_COLOR_RANGE_LIMITED = 1,   //!< 16..235 for 8-bit luma ("studio", "TV")
    CV_ACCEL_COLOR_RANGE_FULL = 2       //!< 0..255 for 8-bit luma ("PC", JPEG)
} cvAccelColorRange;

typedef enum cvAccelColorPrimaries
{
    CV_ACCEL_PRIMARIES_UNKNOWN = 0,
    CV_ACCEL_PRIMARIES_BT709 = 1,       //!< also sRGB
    CV_ACCEL_PRIMARIES_BT601_625 = 2,   //!< BT.470 B/G (PAL)
    CV_ACCEL_PRIMARIES_BT601_525 = 3,   //!< SMPTE 170M (NTSC)
    CV_ACCEL_PRIMARIES_BT2020 = 4,
    CV_ACCEL_PRIMARIES_DCI_P3 = 5,
    CV_ACCEL_PRIMARIES_DISPLAY_P3 = 6
} cvAccelColorPrimaries;

typedef enum cvAccelTransfer
{
    CV_ACCEL_TRANSFER_UNKNOWN = 0,
    CV_ACCEL_TRANSFER_LINEAR = 1,
    CV_ACCEL_TRANSFER_SRGB = 2,
    CV_ACCEL_TRANSFER_BT709 = 3,        //!< also BT.601 and BT.2020 for SDR
    CV_ACCEL_TRANSFER_PQ = 4,           //!< SMPTE ST 2084
    CV_ACCEL_TRANSFER_HLG = 5           //!< ARIB STD-B67
} cvAccelTransfer;

typedef enum cvAccelChromaSiting
{
    CV_ACCEL_CHROMA_SITING_UNKNOWN = 0,
    CV_ACCEL_CHROMA_SITING_COSITED = 1, //!< chroma samples sit on the first luma sample of their block
    CV_ACCEL_CHROMA_SITING_MIDPOINT = 2 //!< halfway between the luma samples of their block
} cvAccelChromaSiting;

/** @brief One plane of an image. */
typedef struct cvAccelImagePlane
{
    uint64_t offset;            //!< bytes from the start of the plane's buffer to the plane
    uint64_t size;              //!< bytes the plane covers in its buffer (given by the producer for non-linear layouts)
    uint64_t pitch;             //!< bytes from one row to the next (its meaning is layout-specific for non-linear layouts)
    uint64_t layout_modifier;   //!< DRM format modifier of this plane: 0 = linear
    int32_t rows, cols, type;   //!< the plane as a matrix: NV12 has 8UC1 luma and 8UC2 chroma at half size
    uint32_t buffer;            //!< planes with the same index lie in the same buffer (one dma-buf, one allocation)
} cvAccelImagePlane;

/** @brief An image whose planes are arguments of operations (cvAccelView::image).

Every plane of one wrapped image refers to the same object, so that a backend can tell from
the views alone which arguments form one image, its format, how its planes lie and how its
colours are encoded. When core knows the fourcc it has checked the planes against it. */
typedef struct cvAccelImageDesc
{
    uint32_t struct_size;
    uint32_t fourcc;            //!< DRM fourcc (NV12, P010, ...), 0 if not given
    uint64_t id;                //!< unique in the process for every wrapped image
    uint32_t plane_count;
    uint32_t color_model;       //!< cvAccelColorModel: the YCbCr matrix
    uint32_t color_range;       //!< cvAccelColorRange
    uint32_t color_primaries;   //!< cvAccelColorPrimaries
    uint32_t transfer;          //!< cvAccelTransfer
    uint32_t chroma_siting_x;   //!< cvAccelChromaSiting
    uint32_t chroma_siting_y;   //!< cvAccelChromaSiting
    uint32_t reserved32;
    cvAccelImagePlane planes[CV_ACCEL_MAX_PLANES];
} cvAccelImageDesc;

/** @brief An n-dimensional strided view into an allocation (an argument of an operation).

In QUERY mode alloc->handle is 0 and backend_data is NULL; only the description is valid.
Core checks before every EXECUTE call that the bytes covered by the view lie inside the
allocation and inside the range of it that the argument may use, so a backend never receives
a view that would make the device access memory outside of the buffer. For a non-linear
layout (alloc->desc.layout_modifier != 0) the steps are informational: the view is a whole
plane, image->planes[plane].size bytes from offset.
*/
typedef struct cvAccelView
{
    const cvAccelAllocation* alloc;
    uint64_t offset;                    //!< bytes from the allocation start to the first element
    int32_t type;                       //!< OpenCV type, e.g. CV_8UC3
    int32_t dims;
    int32_t size[CV_ACCEL_MAX_DIMS];
    uint64_t step[CV_ACCEL_MAX_DIMS];   //!< bytes; step[dims-1] is the element size
    /** the image this view is a plane (or a part of a plane) of, NULL otherwise. Valid during
        the call; the same pointer for every plane of one image. offset and size still say
        which bytes the argument is. */
    const cvAccelImageDesc* image;
    uint32_t plane;                     //!< index of this view's plane in image->planes
    uint32_t reserved;                  //!< zero (views are passed in arrays: the size stays)
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
    /** DRM format modifier of the memory: 0 for linear, anything else for a tiled or compressed
        layout without a linear CPU view. The backend reports it back in desc.layout_modifier;
        core refuses an import whose backend reports another layout (one that predates this
        field reports linear). */
    uint64_t layout_modifier;
    uint32_t fourcc;        //!< DRM fourcc of the image the memory holds, 0 if none; operations see the image in cvAccelView
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
    /** Imports [ext->offset, ext->offset + ext->size) of external memory without copying it.
        The backend checks that the range lies inside the memory (for a device pointer, inside
        the allocation it points into) and fails with CV_ACCEL_INVALID_ARGUMENT otherwise. */
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
        Returns the CPU address of the allocation start. Calls nest (rule 4): core asks for
        ACCESS_READ, or for ACCESS_READ | ACCESS_WRITE, never for a write alone (a partial write
        of a cache line must not merge with stale bytes). */
    cvAccelStatus (CV_API_CALL *begin_cpu_access)(const cvAccelAllocation* alloc, uint64_t offset,
                                                  uint64_t size, uint32_t access, void** host_ptr) CV_ACCEL_NOEXCEPT;
    /** Ends one level of CPU access, with the access its begin_cpu_access had. For NONCOHERENT
        memory written by the CPU this cleans (flushes) CPU caches before any device may touch
        the memory (dma-buf SYNC_END). */
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
    /** Returns at once, like cudaStreamDestroy: work already queued still completes and its
        fences stay valid. It must not wait for other threads: core destroys a thread's streams
        during OpenCV's thread-exit cleanup, under a lock other exiting threads need. */
    cvAccelStatus (CV_API_CALL *stream_destroy)(cvAccelStream stream) CV_ACCEL_NOEXCEPT;
    //! new fence (reference count 1) that completes when all work enqueued so far completes
    cvAccelStatus (CV_API_CALL *stream_record_fence)(cvAccelStream stream, cvAccelFence* out) CV_ACCEL_NOEXCEPT;
    //! device-side wait: later work on @p stream starts after @p fence completes
    cvAccelStatus (CV_API_CALL *stream_wait_fence)(cvAccelStream stream, cvAccelFence fence) CV_ACCEL_NOEXCEPT;
    /** CPU wait; timeout 0 polls (CV_ACCEL_TIMEOUT while pending). Any status other than OK
        and TIMEOUT means the work behind the fence failed and is no longer running: core then
        frees the memory that work used. */
    cvAccelStatus (CV_API_CALL *fence_wait)(cvAccelFence fence, uint64_t timeout_ns) CV_ACCEL_NOEXCEPT;
    /** Takes one more reference. When it fails, the caller holds no new reference (core then
        tracks that work by synchronizing the device instead). */
    cvAccelStatus (CV_API_CALL *fence_retain)(cvAccelFence fence) CV_ACCEL_NOEXCEPT;
    cvAccelStatus (CV_API_CALL *fence_release)(cvAccelFence fence) CV_ACCEL_NOEXCEPT;
    //! optional (CV_ACCEL_CAP_EXTERNAL_FENCE): a new handle (e.g. sync_file fd) owned by the caller
    cvAccelStatus (CV_API_CALL *fence_export)(cvAccelFence fence, uint32_t type, int64_t* handle) CV_ACCEL_NOEXCEPT;
    //! optional: later work on @p stream waits for an external fence; takes ownership of @p handle even on failure
    cvAccelStatus (CV_API_CALL *stream_wait_external)(cvAccelStream stream, uint32_t type, int64_t handle) CV_ACCEL_NOEXCEPT;
    /** Waits for all work on the device. Returns OK once every piece of work queued on the
        device before the call has finished, successfully or not (failures are reported by the
        fences of that work), and CV_ACCEL_DEVICE_LOST when the device has stopped (rule 8).
        Any other status means that the backend could not establish either (a fence could not
        be recorded, say), and core treats the device's work as possibly still running. Before
        CPU access to DEVICE_IDLE memory core calls this and holds back new work on the device
        until the CPU access ends (on Jetson before Thor, the CPU faults on any managed
        allocation while any kernel runs). */
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

    /** for memory flagged CV_ACCEL_MEMF_EXTERNAL_SYNC, required with it (rule 5): later work on
        @p stream waits for the fences other drivers attached to the memory (their last write for
        a read, every use for a write), e.g. DMA_BUF_IOCTL_EXPORT_SYNC_FILE and a wait on the
        sync_file. Core calls it before any device work touches such memory. @p access:
        cv::AccessFlag values. */
    cvAccelStatus (CV_API_CALL *begin_device_access)(const cvAccelAllocation* alloc, cvAccelStream stream,
                                                     uint32_t access) CV_ACCEL_NOEXCEPT;
    /** required with begin_device_access: attaches @p fence, which completes the work begun
        with begin_device_access, to the memory, so that other drivers wait for it
        (DMA_BUF_IOCTL_IMPORT_SYNC_FILE: as the exclusive write fence after a write, as one more
        shared read fence otherwise; a writer of the memory waits for all of them). */
    cvAccelStatus (CV_API_CALL *end_device_access)(const cvAccelAllocation* alloc, cvAccelFence fence,
                                                   uint32_t access) CV_ACCEL_NOEXCEPT;
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
