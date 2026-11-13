// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_ACCEL_BACKEND_HPP
#define OPENCV_CORE_ACCEL_BACKEND_HPP

#include "opencv2/core/accel.hpp"

/*
Interface of the accelerator backends built into OpenCV (CUDA, ...).

A backend provides device memory, copies, streams and events, and one operations interface per
module (CoreOps, ...). OpenCV keeps the UMat bookkeeping. Backends report errors by throwing
cv::Exception.
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
    //! blocks until completed; throws the error of failed device work
    virtual void wait() = 0;
};

/** @brief An in-order queue of device work. OpenCV creates one per device and CPU thread
(Device::threadStream()). */
class CV_EXPORTS Stream
{
public:
    Stream();
    //! waits for the work that uses buffers kept by retain(), then releases them
    virtual ~Stream();

    //! an event that completes when the work queued so far has finished
    virtual Ptr<Event> record() = 0;
    //! blocks until all queued work has finished; throws the error of failed device work
    virtual void synchronize() = 0;

    /** @brief Keeps the buffers of @p arrays alive until the work queued so far has finished.

    An operation calls it after queueing its work, so that the caller may release the UMats at
    once. Buffers kept for finished work are released here and by releaseFinished(). */
    void retain(const std::vector<UMat>& arrays);
    //! releases the buffers kept for work that has finished
    void releaseFinished();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

private:
    struct Impl;
    Impl* p_;
};

enum CopyKind
{
    COPY_HOST_TO_DEVICE = 0,
    COPY_DEVICE_TO_HOST = 1,
    COPY_DEVICE_TO_DEVICE = 2   //!< within one device
};

/** @brief Where bytes are: memory from Backend::allocate() and an offset into it, or host memory
on the host side of a copy. A handle is not necessarily an address, so the offset is kept apart. */
struct DeviceRef
{
    DeviceRef() : handle(0), offset(0) {}
    explicit DeviceRef(void* h, size_t o = 0) : handle(h), offset(o) {}
    explicit DeviceRef(const void* h, size_t o = 0) : handle(const_cast<void*>(h)), offset(o) {}

    void* handle;
    size_t offset;
};

//! base of the per-module operation interfaces
class CV_EXPORTS Ops
{
public:
    virtual ~Ops();
};

/** @brief Operations of the core module.

Arguments are UMats on the device of @p stream, outputs already have their final size and type.
An operation takes the memory of its arguments with UMat::handle() (ACCESS_WRITE for outputs),
queues its work on @p stream and passes the arguments to Stream::retain(). It returns false,
before queueing anything, when it cannot run that call, for example while the CPU holds a Mat view
of an argument; the CPU then runs it. An output may be the input itself (an in-place call), and
each operation handles that as suits it. */
class CV_EXPORTS CoreOps : public Ops
{
public:
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

    //! @p size bytes of memory on @p device, or NULL when the device is out of memory
    virtual void* allocate(int device, size_t size) = 0;
    //! frees memory from allocate(); OpenCV calls it once the work using the memory has finished
    virtual void deallocate(int device, void* handle) = 0;
    /** @brief The CPU address of memory from allocate(), or NULL (the default) when the CPU
    reaches it only through copy(). Memory the CPU addresses, such as an integrated GPU's, is
    used in place by UMat::getMat(). */
    virtual void* hostAddress(int device, void* handle);

    /** @brief Queues the copy of @p rows rows of @p widthBytes bytes on @p stream, after the
    work queued there before. A copy between host and device memory has finished when this
    returns; a device-to-device copy may still be running. */
    virtual void copy(int device, CopyKind kind, DeviceRef dst, size_t dstStep,
                      DeviceRef src, size_t srcStep, size_t widthBytes, size_t rows,
                      Stream& stream) = 0;

    virtual Ptr<Stream> createStream(int device) = 0;

    //! the operations of @p module ("core", ...), or NULL (the default) when the backend has none
    virtual Ops* ops(const char* module);
};

/** @brief Adds a backend: its devices appear in Device::enumerate().
The built-in CUDA backend registers itself. Backends stay registered for the life of the process. */
CV_EXPORTS void registerBackend(const Ptr<Backend>& backend);

//! @}

}} // namespace cv::accel

#endif // OPENCV_CORE_ACCEL_BACKEND_HPP
