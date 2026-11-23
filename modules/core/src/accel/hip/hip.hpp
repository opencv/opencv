// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Internals shared by the built-in HIP backend (accel/backend_hip.cpp) and its operations in
// hal/accel/hip. gpu/gpu_backend.impl.hpp keeps its own helpers to backend_hip.cpp.

#ifndef OPENCV_CORE_SRC_ACCEL_HIP_HPP
#define OPENCV_CORE_SRC_ACCEL_HIP_HPP

#include "opencv2/core/private/accel_backend.hpp"

#include <hip/hip_runtime_api.h>

namespace cv { namespace accel { namespace hip_impl {

//! throws cv::Exception for an error @p e of @p what
void hipCheck(hipError_t e, const char* what);

//! makes @p device current for the calling thread while it lives
class HipDeviceGuard
{
public:
    explicit HipDeviceGuard(int device);
    ~HipDeviceGuard();
private:
    int prev_;
    HipDeviceGuard(const HipDeviceGuard&);
    HipDeviceGuard& operator=(const HipDeviceGuard&);
};

//! the device and the HIP stream behind a Stream of this backend
struct HipStreamRef
{
    int device;
    hipStream_t stream;
};
HipStreamRef hipStreamOf(Stream& s);

//! setTo and convertTo, implemented in hal/accel/hip/core/ops.cpp
CoreOps* coreOps();

//! the imgproc module's operations, implemented in hal/accel/hip/imgproc/ops.cpp
ImgprocOps* imgprocOps();

}}} // namespace cv::accel::hip_impl

#endif // OPENCV_CORE_SRC_ACCEL_HIP_HPP
