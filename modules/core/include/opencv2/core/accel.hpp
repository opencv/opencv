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

/** @brief Runs this thread's cv:: calls on @p device.

Inside the scope, cv:: calls with UMat arguments run on @p device whatever device their
inputs are on (inputs elsewhere are copied to it), and new UMat data created without an
allocator of its own is placed on it. A scope for the CPU (an empty Device) keeps calls and
new data on the CPU. Scopes nest; the innermost one wins. */
class CV_EXPORTS ExecutionScope
{
public:
    explicit ExecutionScope(const Device& device);
    ~ExecutionScope();
private:
    ExecutionScope(const ExecutionScope&);
    ExecutionScope& operator=(const ExecutionScope&);
    bool prevActive_;
    Device prev_;
};

//! blocks until all work queued on @p device has finished; throws the device's first error
CV_EXPORTS void synchronize(const Device& device);

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
