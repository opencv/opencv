// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_GPU_ACCEL_GPU_HPP
#define OPENCV_HAL_GPU_ACCEL_GPU_HPP

// Non-CPU HAL: the GPU backends of cv::accel that hal/gpu builds; core's registry creates them.

#include "opencv2/core/private/accel_backend.hpp"

namespace cv { namespace accel {

namespace cuda_impl {
//! the CUDA backend (hal/gpu/src/backend_cuda.cpp)
Ptr<Backend> createCudaBackend();
}

}} // namespace cv::accel

#endif // OPENCV_HAL_GPU_ACCEL_GPU_HPP
