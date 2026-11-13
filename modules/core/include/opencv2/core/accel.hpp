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

UMat data is placed on an accelerator explicitly, with UMat::createOn(), Mat::upload() or
UMat::upload(). A cv:: function whose input UMat is on a device runs there when the device's
backend implements it, and on the CPU otherwise. Its output UMat is created on that device either
way, so a pipeline stays where its data is.

Device work is queued on one stream per device and CPU thread, and the call returns before the work
has finished. The work of one thread runs in order, and the CPU waits for it where it needs the
data: UMat::getMat(), a copy to host memory, synchronize(). Work queued by different threads is not
ordered: the thread that produced the data calls synchronize() before another thread uses it.
*/

namespace cv { namespace accel {

class Ops;      // accel_backend.hpp
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

    /** @brief The calling thread's stream on this device, created on first use. Operations
    queue their work on it. */
    Stream& threadStream() const;

    /** @brief The operations of @p module ("core", ...) that this device's backend implements,
    as the module's interface @p OpsT (CoreOps for "core"), or NULL. */
    template<typename OpsT> OpsT* ops(const char* module) const
    {
        return dynamic_cast<OpsT*>(opsImpl(module));
    }
    //! the operations of @p module as their base class; ops() casts them
    Ops* opsImpl(const char* module) const;

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

//! the device that holds @p m; the CPU (an empty Device) for host memory, OpenCL buffers and other UMats
CV_EXPORTS Device getDevice(const UMat& m);

//! blocks until the work the calling thread queued on @p device has finished; throws the device's first error
CV_EXPORTS void synchronize(const Device& device);

//! @}

}} // namespace cv::accel

#endif // OPENCV_CORE_ACCEL_HPP
