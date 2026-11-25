// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_NON_CPU_HAL_GPU_PRECOMP_HPP
#define OPENCV_NON_CPU_HAL_GPU_PRECOMP_HPP

// Non-CPU HAL, GPU: the kernels and operations of the CUDA and HIP backends, one folder per module laid
// out as the CPU HAL's (include/, and src/ with a file per family that declares its own kernels). Core
// compiles this file twice on its runtime (src/accel/gpu/gpu_runtime.hpp, included first): from
// src/cuda/accel_kernels.cu and src/hip/accel_kernels.hip with CV_ACCEL_GPU_KERNELS for the kernels,
// and from the end of src/accel/gpu/gpu_backend.impl.hpp for the operations, on its streams and errors.

#include <stddef.h>

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

const int BLOCK_X = 32, BLOCK_Y = 8;

// gridDim.y is at most 65535: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

static inline dim3 grid(int rows, int cols, int bx, int by)
{
    const unsigned gy = (unsigned)((rows + by - 1) / by);
    return dim3((unsigned)((cols + bx - 1) / bx), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

// LaunchKernel returns the status of this launch, where GetLastError() after <<<>>> would
// return the oldest error left on the thread by any earlier call

} // namespace

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#endif // CV_ACCEL_GPU_KERNELS

#include "core/src/precomp.hpp"
#include "imgproc/src/precomp.hpp"

#endif // OPENCV_NON_CPU_HAL_GPU_PRECOMP_HPP
