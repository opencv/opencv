// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_COMPUTE_DEVICE_HPP
#define OPENCV_CORE_COMPUTE_DEVICE_HPP

#include "opencv2/core/mat.hpp"

namespace cv {

//! @addtogroup core_basic
//! @{

/** @brief A device that can hold array data: the CPU, an OpenCL device or a CUDA device.

Devices are named "<backend>:<index>", for example "opencl:0" or "cuda:0"; the CPU is "cpu".
A default-constructed ComputeDevice is the CPU. The device of a UMat is the one whose allocator
owns its buffer.

@sa Tensor, DeviceScope
*/
class CV_EXPORTS ComputeDevice
{
public:
    //! the CPU
    ComputeDevice() : impl_(0) {}

    //! the CPU
    static ComputeDevice cpu();

    /** @brief All devices this build can use on this machine, the CPU first. */
    static std::vector<ComputeDevice> enumerate();

    /** @brief Finds a device by name.
    @param spec "cpu", a backend name such as "opencl" or "cuda" for its first device, or
    "<backend>:<index>"
    @return the matching device, or the CPU when nothing matches */
    static ComputeDevice find(const String& spec);

    /** @brief The device whose memory holds the data of @p m.
    The CPU for host memory, for an empty UMat and for allocators no device owns. */
    static ComputeDevice of(const UMat& m);

    /** @brief The device of this thread's innermost DeviceScope, or the CPU outside a scope. */
    static ComputeDevice current();

    bool isCPU() const { return impl_ == 0; }
    String backend() const;         //!< "cpu", "opencl" or "cuda"
    int index() const;              //!< index within the backend
    String name() const;            //!< "cpu", "opencl:0", "cuda:0", ...
    String description() const;    //!< what the backend reports, for example the GPU model

    /** @brief The allocator that places UMat data on this device.
    Mat::getDefaultAllocator() for the CPU. */
    MatAllocator* allocator() const;

    bool operator==(const ComputeDevice& other) const { return impl_ == other.impl_; }
    bool operator!=(const ComputeDevice& other) const { return impl_ != other.impl_; }

    //! @cond IGNORED
    struct Impl;
    explicit ComputeDevice(Impl* impl) : impl_(impl) {}
    Impl* impl() const { return impl_; }
    //! @endcond

private:
    Impl* impl_;    // owned by the device registry for the life of the process
};

/** @brief Makes @p device the current device of this thread until the scope ends.

Inside the scope, a Tensor created without a device is created on @p device, and so is an empty
Tensor passed to a cv:: function as an output. A UMat created without an allocator of its own is
placed there too, unless @p device is the CPU, which leaves UMat placement to the T-API. Scopes
nest, and the innermost one wins.

@code
cv::DeviceScope scope("cuda:0");
cv::Tensor gray;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);    // gray is created on cuda:0
@endcode
*/
class CV_EXPORTS DeviceScope
{
public:
    explicit DeviceScope(const ComputeDevice& device);
    /** @overload
    The device is ComputeDevice::find(spec). */
    explicit DeviceScope(const String& spec);
    ~DeviceScope();

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    ComputeDevice prev_;
};

//! @} core_basic

} // namespace cv

#endif // OPENCV_CORE_COMPUTE_DEVICE_HPP
