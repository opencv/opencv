// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_ACCEL_BACKEND_HPP
#define OPENCV_CORE_ACCEL_BACKEND_HPP

#include "opencv2/core/accel.hpp"

/*
Interface for accelerator backends (CUDA, HIP, NPU runtimes, ...).

This is a source-level C++ API: a backend is compiled against the headers of the OpenCV
version it runs with, inside OpenCV or as a plugin built with it. It may change between
OpenCV releases. Backends report errors by throwing cv::Exception.

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
    virtual ~Stream();
    //! an event that completes when the work queued so far has finished
    virtual Ptr<Event> record() = 0;
    /** @brief Makes work queued on this stream later wait for @p event, without blocking the CPU.
    @return false when this stream cannot wait for that event (another backend's); OpenCV then
    waits on the CPU instead */
    virtual bool waitEvent(const Ptr<Event>& event) = 0;
    //! blocks until all queued work has finished; throws the error of failed device work
    virtual void synchronize() = 0;
};

enum CopyKind
{
    COPY_HOST_TO_DEVICE = 0,
    COPY_DEVICE_TO_HOST = 1,
    COPY_DEVICE_TO_DEVICE = 2   //!< within one device
};

//! base of the per-module operation interfaces
class CV_EXPORTS Ops
{
public:
    virtual ~Ops();
};

/** @brief Operations of the core module.

Arguments are UMats in this device's memory (use devicePtr()), ordered on @p stream: the
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

    /** @brief true when the CPU can address this device's memory (hostAddress()), so CPU
    access needs no copy; false when the CPU works on a copy of it. */
    virtual bool sharesHostMemory(int device) const = 0;

    //! @p size bytes of device memory, or NULL when the device is out of memory
    virtual void* allocate(int device, size_t size) = 0;
    //! frees allocate()'d memory. OpenCV calls it only after all work using the memory has finished.
    virtual void deallocate(int device, void* ptr) = 0;

    //! the CPU address of shared memory (sharesHostMemory()); returns @p ptr by default
    virtual void* hostAddress(int device, void* ptr);
    /** @brief Called before the CPU accesses shared memory, after the device work on it has
    finished: makes the device's writes visible to the CPU (cache invalidation on non-coherent
    memory). Does nothing by default, as coherent memory needs nothing. */
    virtual void beginCpuAccess(int device, void* ptr, size_t size, bool write);
    //! called when the CPU is done: makes its writes visible to the device (cache clean). Does nothing by default.
    virtual void endCpuAccess(int device, void* ptr, size_t size, bool written);

    /** @brief Host memory for the CPU's copy of device memory, preferably pinned so copies to
    and from it run asynchronously. fastMalloc by default. NULL when out of memory. */
    virtual void* allocateHost(int device, size_t size);
    virtual void deallocateHost(int device, void* ptr);

    /** @brief Queues the copy of @p rows rows of @p widthBytes bytes on @p stream.

    For COPY_HOST_TO_DEVICE with @p hostStable false, the host memory belongs to the user and
    may change as soon as this returns, so the backend must have read it by then. In every other
    case OpenCV keeps the host memory alive and unchanged until the stream reaches the copy, and
    waits for the stream before it reads memory written by a COPY_DEVICE_TO_HOST. */
    virtual void copy(int device, CopyKind kind, void* dst, size_t dstStep,
                      const void* src, size_t srcStep, size_t widthBytes, size_t rows,
                      Stream& stream, bool hostStable) = 0;

    virtual Ptr<Stream> createStream(int device) = 0;

    /** @brief Imports a Linux dma-buf of @p size bytes as device memory, without copying.
    @p fd stays the caller's (dup() it to keep it). deallocate() releases the import. Returns
    NULL when the device cannot import it, which is the default. */
    virtual void* importDmaBuf(int device, int fd, size_t size);
    /** @brief Exports allocate()'d or imported memory as a dma-buf: a new file descriptor that
    the caller closes, or -1 when that is not possible, which is the default. */
    virtual int exportDmaBuf(int device, void* ptr, size_t size);

    //! operations for @p module ("core", ...), or NULL when the backend has none
    virtual Ops* ops(const String& module);
};

/** @brief Adds a backend: its devices appear in Device::enumerate().
Built-in backends (CUDA) register themselves. Backends stay registered for the life of the
process. */
CV_EXPORTS void registerBackend(const Ptr<Backend>& backend);

//! the device address of @p m's first element, for operations
static inline void* devicePtr(const UMat& m)
{
    return m.u ? (void*)((uchar*)m.u->handle + m.offset) : 0;
}

/** @brief Entry point of a backend plugin.

A plugin is a shared library built against the headers of the OpenCV version that loads it.
It exports this function, which returns a new Backend, or NULL when @p major and @p minor are
not the version it was built for. Define it with CV_ACCEL_DEFINE_PLUGIN. */
typedef Backend* (*FN_opencv_core_accel_plugin_init_v1)(int major, int minor, int patch);
#define CV_ACCEL_PLUGIN_INIT_NAME "opencv_core_accel_plugin_init_v1"

//! @}

}} // namespace cv::accel

#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define CV_ACCEL_PLUGIN_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define CV_ACCEL_PLUGIN_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define CV_ACCEL_PLUGIN_EXPORTS
#endif

/** @brief Defines the entry point of a backend plugin; @p expr creates its Backend.
@code
CV_ACCEL_DEFINE_PLUGIN(new MyBackend())
@endcode */
#define CV_ACCEL_DEFINE_PLUGIN(expr) \
    extern "C" CV_ACCEL_PLUGIN_EXPORTS cv::accel::Backend* opencv_core_accel_plugin_init_v1(int major, int minor, int patch) \
    { \
        (void)patch; \
        if (major != CV_VERSION_MAJOR || minor != CV_VERSION_MINOR) \
            return 0; \
        try { return (expr); } catch (...) { return 0; } \
    }

#endif // OPENCV_CORE_ACCEL_BACKEND_HPP
