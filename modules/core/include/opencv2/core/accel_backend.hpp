// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_ACCEL_BACKEND_HPP
#define OPENCV_CORE_ACCEL_BACKEND_HPP

#include "opencv2/core/accel.hpp"
#include "opencv2/core/llapi/llapi.h"

/*
Interface for accelerator backends (CUDA, HIP, NPU runtimes, ...).

The interface is C++, not a C ABI: its objects carry UMat, String and Ptr across the plugin
boundary. A plugin is reached through a C entry point (CV_ACCEL_DEFINE_PLUGIN), as core's parallel
plugins are, and is loaded only when it was built against the same OpenCV version, with the same
CV_ACCEL_PLUGIN_ABI_VERSION and a compiler and C++ standard library compatible with OpenCV's.
Backends report errors by throwing cv::Exception.

A backend provides device memory, copies, streams and events. OpenCV keeps the UMat
bookkeeping: which copy is current, which device work a buffer waits for, when memory can be
freed. Operations come in one interface per module (CoreOps, ...); each one returns false,
before queueing anything, when the device cannot run that particular call.
*/

namespace cv { namespace accel {

//! @addtogroup core_accel
//! @{

/** @brief A point in a stream. It completes when the work queued before it has finished. */
class CV_EXPORTS Event
{
public:
    virtual ~Event();
    //! true when completed; throws the error of failed device work
    virtual bool query() = 0;
    //! blocks the CPU until completed; throws the error of failed device work
    virtual void wait() = 0;
};

/** @brief An in-order queue of device work. */
class CV_EXPORTS Stream
{
public:
    Stream() : owner_(0) {}
    virtual ~Stream();
    //! an event that completes when the work queued so far has finished
    virtual Ptr<Event> record() = 0;
    /** @brief Makes work queued on this stream later wait for @p event, without blocking the CPU.
    @return false when this stream cannot wait for that event (another backend's); OpenCV then
    waits on the CPU instead */
    virtual bool waitEvent(const Ptr<Event>& event) = 0;
    //! blocks until all queued work has finished; throws the error of failed device work
    virtual void synchronize() = 0;

    //! the device OpenCV created this stream for (createStream); empty for any other stream
    Device device() const { return Device(owner_); }

private:
    friend struct Device::Impl;     // sets owner_ on the streams it creates
    Device::Impl* owner_;
};

enum CopyKind
{
    COPY_HOST_TO_DEVICE = 0,
    COPY_DEVICE_TO_HOST = 1,
    COPY_DEVICE_TO_DEVICE = 2   //!< within one device
};

/** @brief Where bytes are, as the memory they belong to and an offset into it.

A backend's memory is whatever allocate() returned, and what that is only the backend knows:
CUDA and HIP hand back an address and add @p offset to it, while memory that is an object with
no address of its own (a Metal or Vulkan buffer) takes the offset as a separate argument.
Describing a location as a pair rather than as an address is what lets a backend hold
something other than a pointer in UMatData::handle.

On the host side of a copy @p handle is an ordinary pointer, so @p handle + @p offset is the
address in every backend. */
struct DeviceRef
{
    DeviceRef() : handle(0), offset(0) {}
    // explicit: a call site has to say which memory and which offset, so that adding the
    // offset to an address cannot slip back in where the handle is not one
    explicit DeviceRef(void* h, size_t o = 0) : handle(h), offset(o) {}
    explicit DeviceRef(const void* h, size_t o = 0) : handle(const_cast<void*>(h)), offset(o) {}

    void* handle;       //!< memory from allocate(), or host memory on the host side of a copy
    size_t offset;      //!< bytes from the start of @p handle
};

//! base of the per-module operation interfaces
class CV_EXPORTS Ops
{
public:
    virtual ~Ops();
};

/** @brief Operations of the core module.

Arguments are UMats in this device's memory (use deviceRef()), ordered on @p stream: the
work queued on it runs after everything that writes them. Outputs have their final size and
type and never overlap inputs. An operation returns false, without queueing anything, when the
device cannot run the call; OpenCV then runs it on the CPU. */
class CV_EXPORTS CoreOps : public Ops
{
public:
    static const char* module() { return "core"; }
    //! cv::flip: @p flipCode 0 flips around the x axis, > 0 around the y axis, < 0 both
    virtual bool flip(const UMat& src, UMat& dst, int flipCode, Stream& stream);
    //! cv::transpose of a 2-D UMat
    virtual bool transpose(const UMat& src, UMat& dst, Stream& stream);
};

/** @brief An accelerator runtime: its devices, their memory, streams and operations.

OpenCV calls a backend from any thread, so its methods must be thread-safe. */
class CV_EXPORTS Backend
{
public:
    virtual ~Backend();

    //! short lowercase name used in device names, "cuda"
    virtual String name() const = 0;
    virtual int deviceCount() const = 0;
    virtual String deviceDescription(int device) const = 0;

    /** @brief The MemoryKind values this device can provide, as a bitmask. At least one bit
    must be set. A device whose memory the CPU cannot address offers MEMORY_DEVICE alone. */
    virtual int memoryKinds(int device) const = 0;
    /** @brief The kind of a new buffer when nothing asks for another one.

    By default the most CPU-accessible kind the device offers, so a device with shared memory
    uses it and the CPU reaches its buffers without copies. Override to choose another one. */
    virtual MemoryKind defaultMemoryKind(int device) const;

    /** @brief @p size bytes of memory of @p kind, one of memoryKinds(), or NULL when the
    device is out of memory. */
    virtual void* allocate(int device, size_t size, MemoryKind kind) = 0;
    /** @brief Frees memory from allocate(), of the @p kind it was allocated with. OpenCV calls
    it only after all work using the memory has finished. */
    virtual void deallocate(int device, void* ptr, MemoryKind kind) = 0;

    //! the CPU address of memory of a shared @p kind; returns @p ptr by default
    virtual void* hostAddress(int device, void* ptr, MemoryKind kind);
    /** @brief Called before the CPU accesses memory of a shared @p kind, after the device work
    on it has finished: makes the device's writes visible to the CPU (cache invalidation on
    MEMORY_SHARED_NONCOHERENT). Does nothing by default, as coherent memory needs nothing. */
    virtual void beginCpuAccess(int device, void* ptr, size_t size, MemoryKind kind, bool write);
    //! called when the CPU is done: makes its writes visible to the device (cache clean). Does nothing by default.
    virtual void endCpuAccess(int device, void* ptr, size_t size, MemoryKind kind, bool written);

    /** @brief Host memory for the CPU's copy of device memory, preferably pinned so copies to
    and from it run asynchronously. fastMalloc by default. NULL when out of memory. */
    virtual void* allocateHost(int device, size_t size);
    virtual void deallocateHost(int device, void* ptr);

    /** @brief Queues the copy of @p rows rows of @p widthBytes bytes on @p stream.

    For COPY_HOST_TO_DEVICE with @p hostStable false, the host memory belongs to the user and
    may change as soon as this returns, so the backend must have read it by then. In every other
    case OpenCV keeps the host memory alive and unchanged until the stream reaches the copy, and
    waits for the stream before it reads memory written by a COPY_DEVICE_TO_HOST. */
    virtual void copy(int device, CopyKind kind, DeviceRef dst, size_t dstStep,
                      DeviceRef src, size_t srcStep, size_t widthBytes, size_t rows,
                      Stream& stream, bool hostStable) = 0;

    virtual Ptr<Stream> createStream(int device) = 0;

    /** @brief Imports a Linux dma-buf of @p size bytes as device memory, without copying.

    @p fd stays the caller's (dup() it to keep it). Sets @p kind to the kind of the imported
    memory, which the exporter decides, not defaultMemoryKind(). deallocate() releases the
    import. Returns NULL when the device cannot import it, which is the default. */
    virtual void* importDmaBuf(int device, int fd, size_t size, MemoryKind& kind);
    /** @brief Exports allocate()'d or imported memory of @p kind as a dma-buf: a new file
    descriptor that the caller closes, or -1 when that is not possible, which is the default. */
    virtual int exportDmaBuf(int device, void* ptr, size_t size, MemoryKind kind);

    //! operations for @p module ("core", ...), or NULL when the backend has none
    virtual Ops* ops(const String& module);
};

/** @brief Adds a backend: its devices appear in Device::enumerate().
Built-in backends (CUDA) register themselves. Backends stay registered for the life of the
process. */
CV_EXPORTS void registerBackend(const Ptr<Backend>& backend);

//! where @p m's first element is, for operations
static inline DeviceRef deviceRef(const UMat& m)
{
    return m.u ? DeviceRef(m.u->handle, m.offset) : DeviceRef();
}

/** @brief The device address of @p m's first element.

Only for a backend whose allocate() returns an address, which CUDA and HIP do. Anything that
has to work on every backend uses deviceRef() instead. */
static inline void* devicePtr(const UMat& m)
{
    return m.u ? (void*)((uchar*)m.u->handle + m.offset) : 0;
}

//! @}

}} // namespace cv::accel

/** @brief Version of the classes a plugin implements (Backend, Stream, Event, Ops). Bumped
whenever one of them changes its layout, so a plugin built against another layout is not loaded. */
#define CV_ACCEL_PLUGIN_ABI_VERSION 1
//! version of what the interface offers within one ABI version
#define CV_ACCEL_PLUGIN_API_VERSION 0

/** @brief What a backend plugin exports, in the form of core's other plugin interfaces (llapi.h). */
typedef struct OpenCV_Core_Accel_Plugin_API
{
    OpenCV_API_Header api_header;
    //! a new Backend, owned by the caller; CV_ERROR_FAIL when the plugin cannot create one
    CvResult (CV_API_CALL *createBackend)(CV_OUT cv::accel::Backend** backend) CV_NOEXCEPT;
} OpenCV_Core_Accel_Plugin_API;

typedef const OpenCV_Core_Accel_Plugin_API* (CV_API_CALL *FN_opencv_core_accel_plugin_init_t)
        (int requested_abi_version, int requested_api_version, void* reserved /*NULL*/);
#define CV_ACCEL_PLUGIN_INIT_NAME "opencv_core_accel_plugin_init_v1"

/** @brief Defines the entry point of a backend plugin; @p expr creates its Backend.
@code
CV_ACCEL_DEFINE_PLUGIN(new MyBackend())
@endcode */
#define CV_ACCEL_DEFINE_PLUGIN(expr) \
    static CvResult CV_API_CALL cv_accel_plugin_create_backend(CV_OUT cv::accel::Backend** backend) CV_NOEXCEPT \
    { \
        if (!backend) \
            return CV_ERROR_FAIL; \
        try { *backend = (expr); } catch (...) { *backend = 0; } \
        return *backend ? CV_ERROR_OK : CV_ERROR_FAIL; \
    } \
    static const OpenCV_Core_Accel_Plugin_API cv_accel_plugin_api = \
    { \
        { \
            sizeof(OpenCV_Core_Accel_Plugin_API), CV_ACCEL_PLUGIN_ABI_VERSION, CV_ACCEL_PLUGIN_API_VERSION, \
            CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS, \
            "OpenCV accel backend plugin" \
        }, \
        cv_accel_plugin_create_backend \
    }; \
    extern "C" CV_PLUGIN_EXPORTS const OpenCV_Core_Accel_Plugin_API* CV_API_CALL \
    opencv_core_accel_plugin_init_v1(int requested_abi_version, int requested_api_version, void* /*reserved*/) CV_NOEXCEPT \
    { \
        if (requested_abi_version == CV_ACCEL_PLUGIN_ABI_VERSION && requested_api_version <= CV_ACCEL_PLUGIN_API_VERSION) \
            return &cv_accel_plugin_api; \
        return 0; \
    }

#endif // OPENCV_CORE_ACCEL_BACKEND_HPP
