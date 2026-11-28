// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_NON_CPU_HAL_GPU_HAL_CORE_HPP
#define OPENCV_NON_CPU_HAL_GPU_HAL_CORE_HPP

// Non-CPU HAL, GPU: the core operations of the CUDA and HIP backends, one family per
// src/<family>.impl.hpp. Core's GPU backend creates them.

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! the backend's CoreOps (src/precomp.hpp)
Ptr<CoreOps> createCoreOps();

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // OPENCV_NON_CPU_HAL_GPU_HAL_CORE_HPP
