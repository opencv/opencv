// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_ACCEL_HPP
#define OPENCV_CORE_ACCEL_HPP

#ifndef __cplusplus
#  error accel.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"

/**
@defgroup core_accel Non-CPU HAL: accelerator devices and memory
@ingroup core

UMat data can live on an accelerator: a discrete GPU, an integrated GPU, an NPU, a DSP or a
2D engine. A backend for that device (built into OpenCV or loaded as a plugin, see
opencv2/core/llapi/accel_hal.h) allocates the memory and runs cv:: functions on it. A cv::
call whose arguments are on a device runs there when the backend implements it, and runs
on the CPU otherwise; in both cases the result is placed on the same device.

Correctness does not depend on the user: OpenCV tracks which copy of a buffer is current,
waits for pending device work before the CPU touches the data, cleans and invalidates CPU
caches for memory that is shared with a device without hardware coherence, and keeps
buffers alive until the device is done with them.

@code
cv::accel::Device gpu = cv::accel::Device::find("cuda");
cv::UMat img;
img.allocator = gpu.allocator();   // place the data on the GPU
frame.copyTo(img);                 // one host-to-device DMA
cv::UMat flipped;
cv::flip(img, flipped, 1);         // runs on the GPU, result stays on the GPU
cv::Mat result = flipped.getMat(cv::ACCESS_READ); // waits, one DMA, no write-back
@endcode
*/

namespace cv { namespace accel {

//! @addtogroup core_accel
//! @{

enum DeviceType
{
    DEVICE_HOST = -1,           //!< no accelerator: plain host memory, CPU code
    DEVICE_OTHER = 0,
    DEVICE_GPU_DISCRETE = 1,
    DEVICE_GPU_INTEGRATED = 2,
    DEVICE_NPU = 3,
    DEVICE_DSP = 4,
    DEVICE_2D_ENGINE = 5,
    DEVICE_FPGA = 6
};

//! What kind of memory an allocation is (values match cvAccelMemoryKind)
enum MemoryKind
{
    MEMORY_HOST = 0,
    MEMORY_HOST_PINNED = 1,
    MEMORY_DEVICE = 2,
    MEMORY_SHARED = 3,
    MEMORY_MANAGED = 4
};

//! How the CPU can access an allocation (values match cvAccelHostAccess)
enum HostAccess
{
    HOST_ACCESS_NONE = 0,         //!< no CPU access at all (protected or tiled memory)
    HOST_ACCESS_COPY = 1,         //!< through a host copy and DMA transfers (discrete GPU memory)
    HOST_ACCESS_COHERENT = 2,     //!< same bytes, caches kept coherent by hardware
    HOST_ACCESS_NONCOHERENT = 3,  //!< same bytes, OpenCV performs cache maintenance
    HOST_ACCESS_UNCACHED = 4,     //!< same bytes, but uncached: CPU code works on a cached copy
    HOST_ACCESS_DEVICE_IDLE = 5   //!< same bytes, only while the device is idle
};

/** @brief An accelerator device, or the host when empty. Cheap to copy. */
class CV_EXPORTS Device
{
public:
    //! the host
    Device();

    //! all devices of all available backends
    static std::vector<Device> enumerate();

    /** @brief Finds a device by specification.
    @param spec "backend" or "backend:index", for example "cuda", "cuda:1" or "hip:0".
    Returns an empty Device (the host) when nothing matches. */
    static Device find(const String& spec);

    bool empty() const { return backendId_ == 0; }
    String backend() const;
    int index() const { return index_; }
    String name() const;
    DeviceType type() const;

    /** @brief Allocator that places UMat data on this device.
    UMatUsageFlags select the memory kind: USAGE_ALLOCATE_DEVICE_MEMORY (device-local),
    USAGE_ALLOCATE_HOST_MEMORY (pinned host memory mapped into the device) and
    USAGE_ALLOCATE_SHARED_MEMORY (shared or managed memory). USAGE_DEFAULT uses the
    device's preferred kind. Returns the host allocator for the host. */
    MatAllocator* allocator() const;

    bool operator==(const Device& other) const { return backendId_ == other.backendId_ && index_ == other.index_; }
    bool operator!=(const Device& other) const { return !(*this == other); }

    //! @cond IGNORED
    Device(int backendId, int index) : backendId_(backendId), index_(index) {}
    int backendId() const { return backendId_; }
    //! @endcond

private:
    int backendId_;
    int index_;
};

/** @brief Where UMat data lives and how the CPU can reach it. */
struct CV_EXPORTS MemoryInfo
{
    MemoryInfo() : kind(MEMORY_HOST), hostAccess(HOST_ACCESS_COHERENT), imported(false), readOnly(false),
                   protectedContent(false) {}
    Device device;          //!< empty for host memory
    MemoryKind kind;
    HostAccess hostAccess;
    bool imported;          //!< memory belongs to someone else (user pointer, dma-buf, ...)
    bool readOnly;
    //! protected content (secure video): never readable by the CPU, results computed from it stay protected
    bool protectedContent;
};

//! describes the memory behind @p m; host memory and legacy OpenCL buffers report an empty device
CV_EXPORTS MemoryInfo describe(const UMat& m);

/** @brief An in-order queue of work on a device.

cv:: functions enqueue work on the calling thread's current stream (see ExecutionScope)
and return without waiting. OpenCV waits only where the CPU needs the result, for example
in UMat::getMat(). */
class CV_EXPORTS Stream
{
public:
    Stream();                               //!< empty
    explicit Stream(const Device& device);  //!< creates a new stream on @p device

    //! the per-thread default stream of @p device
    static Stream getDefault(const Device& device);

    Device device() const;
    bool empty() const;

    //! blocks until all work enqueued so far has completed
    void waitForCompletion();

    //! @cond IGNORED
    struct Impl;
    Ptr<Impl> impl;
    //! @endcond
};

/** @brief Routes cv:: calls made on this thread to a device or a stream.

Inside the scope, new UMat data created by UMat::create() without an explicit allocator is
placed on the device, and operations that the device implements run there, Mat arguments
included: the device uses host memory in place where it can address it and copies it
otherwise. Scopes nest; the innermost one wins. */
class CV_EXPORTS ExecutionScope
{
public:
    explicit ExecutionScope(const Device& device);
    explicit ExecutionScope(const Stream& stream);
    ~ExecutionScope();
private:
    ExecutionScope(const ExecutionScope&);
    ExecutionScope& operator=(const ExecutionScope&);
};

/** @brief What to do when data has to cross between the host and a device, or between
two devices, without the user asking for it (an operation falls back to the CPU on
device data, or an input has to be uploaded for a device operation). */
enum TransferPolicy
{
    TRANSFER_ALLOW = 0,     //!< move the data (default; compatible with the T-API)
    TRANSFER_WARN = 1,      //!< move the data and log a warning naming the operation
    TRANSFER_FORBID = 2     //!< throw cv::Exception: useful to find hidden copies
};

CV_EXPORTS void setTransferPolicy(TransferPolicy policy);
CV_EXPORTS TransferPolicy getTransferPolicy();

/** @brief Process-wide counters, to verify where data moves. */
struct CV_EXPORTS TransferStats
{
    TransferStats() : bytesHostToDevice(0), bytesDeviceToHost(0), bytesDeviceToDevice(0),
        transfers(0), implicitTransfers(0), deviceOps(0), cpuFallbacks(0), hostWaits(0) {}
    uint64 bytesHostToDevice;
    uint64 bytesDeviceToHost;
    uint64 bytesDeviceToDevice;
    uint64 transfers;           //!< DMA transfers issued
    uint64 implicitTransfers;   //!< transfers the user did not ask for
    uint64 deviceOps;           //!< operations run on a device
    uint64 cpuFallbacks;        //!< operations on device data that ran on the CPU
    uint64 hostWaits;           //!< times the CPU waited for device work
};

CV_EXPORTS TransferStats getTransferStats();
CV_EXPORTS void resetTransferStats();

/** @brief Memory owned outside of OpenCV, to wrap into a UMat without copying. */
struct CV_EXPORTS ExternalMemory
{
    enum Type
    {
        HOST_POINTER = 1,       //!< user host memory: registered for DMA or used in place if the device can
        DMABUF_FD = 2,          //!< Linux dma-buf (cameras, decoders, other accelerators)
        OPAQUE_FD = 3,          //!< Vulkan / CUDA / HIP exported memory
        OPAQUE_WIN32 = 4,
        DEVICE_POINTER = 5,     //!< native pointer of the device's own API (non-owning)
        AHARDWAREBUFFER = 6
    };
    ExternalMemory() : type(HOST_POINTER), handle(0), offset(0), size(0), readOnly(false), protectedContent(false),
                       layoutModifier(0), fourcc(0) {}
    Type type;
    int64 handle;           //!< pointer, file descriptor or native handle
    size_t offset;
    size_t size;
    bool readOnly;
    /** The memory must hold protected content (a dma-buf from a secure heap, say): the wrap is
        refused unless the backend attests it. Protection itself always comes from the backend. */
    bool protectedContent;
    /** DRM format modifier: 0 for linear memory; a tiled or compressed layout (GPU tiling, DCC,
        AFBC, ...) has no CPU view, and only operations of a device that knows the layout use it. */
    uint64 layoutModifier;
    unsigned fourcc;        //!< DRM fourcc of the image the memory holds (NV12, P010, ...), 0 if none
};

//! one plane of an external image: its buffer and where the plane lies in it
struct CV_EXPORTS ExternalPlane
{
    ExternalPlane() : handle(0), offset(0), pitch(0), rows(0), cols(0), type(-1) {}
    int64 handle;           //!< the plane's buffer (dma-buf fd, device pointer, ...); planes may share one
    size_t offset;          //!< bytes from the start of the buffer to the plane
    size_t pitch;           //!< bytes from one row of the plane to the next
    int rows, cols, type;   //!< the plane as a matrix: NV12 is 8UC1 luma and 8UC2 chroma at half size
};

//! a multi-plane image in external memory (a decoder's NV12 or P010 frame, a camera buffer)
struct CV_EXPORTS ExternalImage
{
    ExternalImage() : type(ExternalMemory::DMABUF_FD), layoutModifier(0), fourcc(0), readOnly(false),
                      protectedContent(false) {}
    ExternalMemory::Type type;
    std::vector<ExternalPlane> planes;
    uint64 layoutModifier;  //!< see ExternalMemory::layoutModifier
    unsigned fourcc;        //!< DRM fourcc, informational
    bool readOnly;
    bool protectedContent;
};

/** @brief Wraps external memory into a UMat without copying.
The memory must stay valid while the UMat (and every UMat or Mat derived from it) is alive;
releasing the last reference waits for pending device work on it. Throws if the device
cannot import that kind of memory. */
CV_EXPORTS UMat wrapExternalMemory(const Device& device, const ExternalMemory& memory,
                                   const MatShape& shape, int type, const size_t* steps = 0);

/** @brief Wraps each plane of an external image into a UMat, in the order of the planes, with
the plane's pitch as its row step. Planes in one buffer and planes in separate buffers (one fd
each) both work; every plane keeps the synchronization of the buffer it lies in. */
CV_EXPORTS std::vector<UMat> wrapExternalImage(const Device& device, const ExternalImage& image);

/** @brief Explicit device access for the user's own kernels.

On construction, waits (on the device, where possible) for OpenCV work that conflicts with
@p access and makes the device copy current. While the object is alive, handle() + offset()
may be used by work enqueued on @p stream. On destruction, the stream's progress is recorded
so that later OpenCV calls wait for that work. */
class CV_EXPORTS DeviceAccess
{
public:
    DeviceAccess(const UMat& m, const Stream& stream, AccessFlag access);
    ~DeviceAccess();

    uint64 handle() const;      //!< native handle (device pointer for CUDA / HIP)
    size_t offset() const;      //!< byte offset of the first element of the UMat

    //! @cond IGNORED
    struct Impl;
    //! @endcond
private:
    DeviceAccess(const DeviceAccess&);
    DeviceAccess& operator=(const DeviceAccess&);
    Impl* p;
};

//! @}

}} // namespace cv::accel

#endif // OPENCV_CORE_ACCEL_HPP
