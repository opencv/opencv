// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_ACCEL_HPP
#define OPENCV_CORE_ACCEL_HPP

#include "opencv2/core.hpp"

/**
@defgroup core_accel Accelerators (non-CPU HAL)

UMat data can live on an accelerator: a discrete GPU, an integrated GPU or an NPU. cv::
functions called on such UMats run on the device that holds their first input, when its
backend implements them, and on the CPU otherwise. Their outputs are created on that device
either way, so a pipeline stays where its data is.

Two kinds of device memory exist, decided by the device:
- memory the CPU reaches only through a copy (discrete GPUs): UMat::getMat() downloads,
  releasing the Mat uploads what the CPU wrote;
- memory shared with the CPU (NPUs, integrated GPUs, Jetson): UMat::getMat() returns the same
  bytes, after the device has finished with them. Nothing is copied.

Device work is queued and runs asynchronously. The CPU waits where it needs the data.
*/

namespace cv { namespace accel {

class Stream;   // accel_backend.hpp

//! @addtogroup core_accel
//! @{

/** @brief A device of an accelerator backend, or the CPU (an empty Device). */
class CV_EXPORTS Device
{
public:
    //! the CPU
    Device() : impl_(0) {}

    //! all accelerator devices, in the order their backends were registered
    static std::vector<Device> enumerate();

    /** @brief Finds a device: "cuda" (its first device) or "cuda:1".
    Returns the CPU (an empty Device) when nothing matches. */
    static Device find(const String& spec);

    bool empty() const { return impl_ == 0; }
    String backend() const;     //!< "cuda"; empty for the CPU
    int index() const;          //!< index within the backend; -1 for the CPU
    String name() const;        //!< "cuda:0", or "cpu"
    String description() const; //!< what the backend reports, for example the GPU model

    /** @brief true when the CPU reaches this device's memory directly (NPUs, integrated GPUs),
    false when it works on a copy (discrete GPUs). */
    bool sharesHostMemory() const;

    /** @brief Allocator that places UMat data on this device.
    Use it as `UMat u; u.allocator = device.allocator();` before creating the data.
    The CPU's allocator for an empty Device. */
    MatAllocator* allocator() const;

    bool operator==(const Device& other) const { return impl_ == other.impl_; }
    bool operator!=(const Device& other) const { return impl_ != other.impl_; }

    //! @cond IGNORED
    struct Impl;
    explicit Device(Impl* impl) : impl_(impl) {}
    Impl* impl() const { return impl_; }
    //! @endcond

private:
    Impl* impl_;    // owned by the registry for the life of the process
};

//! the device that holds @p m: empty for host memory, OpenCL buffers and other UMats
CV_EXPORTS Device getDevice(const UMat& m);

/** @brief A new in-order queue of work on @p device (see accel_backend.hpp), for ExecutionScope.
Work on different streams may run concurrently. OpenCV orders the work of all streams that
touch the same UMat. */
CV_EXPORTS Ptr<Stream> createStream(const Device& device);

/** @brief Runs this thread's cv:: calls on @p device.

Inside the scope, cv:: calls with UMat arguments run on @p device whatever device their
inputs are on (inputs elsewhere are copied to it), and new UMat data created without an
allocator of its own is placed on it. A scope for the CPU (an empty Device) keeps calls and
new data on the CPU. Scopes nest; the innermost one wins. */
class CV_EXPORTS ExecutionScope
{
public:
    explicit ExecutionScope(const Device& device);
    //! as above, and the work of the calls is queued on @p stream, a stream of @p device
    ExecutionScope(const Device& device, const Ptr<Stream>& stream);
    ~ExecutionScope();
private:
    ExecutionScope(const ExecutionScope&);
    ExecutionScope& operator=(const ExecutionScope&);
    bool prevActive_;
    Device prev_;
    const Ptr<Stream>* prevStream_;
    Ptr<Stream> stream_;
};

//! blocks until all work queued on @p device's default stream has finished; throws the device's first error
CV_EXPORTS void synchronize(const Device& device);

/** @brief Devices to try first for calls on memory shared with the CPU.

Memory that NPUs, integrated GPUs and 2D engines share with the CPU can be reached by several
of them. A call on such memory is offered to these devices in order (each reaches the memory
through a dma-buf import, without copies), then to the device that holds it, then it runs on
the CPU. The data stays where it is. Has no effect on memory the CPU reaches only through a copy
(discrete GPUs). Empty by default. */
CV_EXPORTS void setPreferredDevices(const std::vector<Device>& devices);
CV_EXPORTS std::vector<Device> getPreferredDevices();

/** @brief Wraps a Linux dma-buf (a camera frame, a decoder output, another device's buffer)
into a UMat on @p device, without copying.

@p fd stays the caller's; it may be closed after the call. The producer's writes must be
finished. The UMat starts @p offset bytes into the buffer, with rows @p step bytes apart.
Outputs of cv:: calls that are wrapped memory get their results copied in; they never move.
Throws when @p device cannot import dma-bufs. */
CV_EXPORTS UMat wrapDmaBuf(const Device& device, int fd, Size size, int type,
                           size_t step = Mat::AUTO_STEP, size_t offset = 0);

/** @brief Exports the memory of @p m as a dma-buf, without copying: a new file descriptor that
the caller closes.

The descriptor covers the whole buffer of @p m; m.offset and m.step locate @p m in it. OpenCV's
work on @p m has finished when this returns. Do not use @p m while someone else writes the
buffer. Throws when the device cannot export its memory. */
CV_EXPORTS int exportDmaBuf(const UMat& m);

/** @brief Loads a backend plugin, a shared library built with this OpenCV version (see
CV_ACCEL_DEFINE_PLUGIN). Returns false when it cannot be loaded or was built for another version.

Plugins named like libopencv_core_accel_<name>.so are also loaded on first use, from the
directories in OPENCV_CORE_PLUGIN_PATH or, without it, from the directory of the OpenCV
libraries. OPENCV_ACCEL_PLUGINS lists files to load instead; OPENCV_ACCEL_PLUGINS_ENABLE=0
turns automatic loading off. */
CV_EXPORTS bool loadPlugin(const String& path);

/** @brief What to do when a cv:: call would copy data between the host and a device's
memory without being asked to (an input that has to be uploaded, a result copied into a Mat,
a call that falls back to the CPU on data in copy-only device memory). */
enum TransferPolicy
{
    TRANSFER_ALLOW = 0,     //!< copy (default, as the T-API does)
    TRANSFER_FORBID = 1     //!< throw cv::Exception naming the call, to find hidden copies
};

CV_EXPORTS void setTransferPolicy(TransferPolicy policy);
CV_EXPORTS TransferPolicy getTransferPolicy();

/** @brief Process-wide counters, to check where data moves and where calls run. */
struct CV_EXPORTS TransferStats
{
    TransferStats() : bytesHostToDevice(0), bytesDeviceToHost(0), bytesDeviceToDevice(0),
        transfers(0), deviceOps(0), cpuFallbacks(0), hostWaits(0) {}
    uint64 bytesHostToDevice;
    uint64 bytesDeviceToHost;
    uint64 bytesDeviceToDevice;
    uint64 transfers;       //!< copies between host and device memory
    uint64 deviceOps;       //!< cv:: calls that ran on a device
    uint64 cpuFallbacks;    //!< cv:: calls on device data that ran on the CPU
    uint64 hostWaits;       //!< times the CPU waited for unfinished device work
};

CV_EXPORTS TransferStats getTransferStats();
CV_EXPORTS void resetTransferStats();

//! @}

}} // namespace cv::accel

#endif // OPENCV_CORE_ACCEL_HPP
