// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Internals shared by the HIP backend and its per-module operations.

#ifndef OPENCV_CORE_SRC_ACCEL_HIP_HPP
#define OPENCV_CORE_SRC_ACCEL_HIP_HPP

#include "opencv2/core/accel_backend.hpp"

#include <hip/hip_runtime_api.h>

namespace cv { namespace accel { namespace hip_impl {

void check(hipError_t e, const char* what);

//! makes @p device current for the calling thread while it lives
class DeviceGuard
{
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard();
private:
    int prev_;
};

class HipEvent CV_FINAL : public Event
{
public:
    explicit HipEvent(hipEvent_t e) : e_(e) {}
    ~HipEvent();

    bool query() CV_OVERRIDE;
    void wait() CV_OVERRIDE;

    hipEvent_t e_;
};

class HipStream CV_FINAL : public Stream
{
public:
    explicit HipStream(int device);
    ~HipStream();

    Ptr<Event> record() CV_OVERRIDE;
    bool waitEvent(const Ptr<Event>& event) CV_OVERRIDE;
    void synchronize() CV_OVERRIDE;

    const int device_;
    hipStream_t s_;
};

HipStream& hipStreamOf(Stream& s);

//! the core module's operations, implemented in core/ops.cpp
CoreOps* coreOps();

}}} // namespace cv::accel::hip_impl

#endif // OPENCV_CORE_SRC_ACCEL_HIP_HPP
