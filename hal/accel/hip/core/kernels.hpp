// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_ACCEL_HIP_CORE_KERNELS_HPP
#define OPENCV_HAL_ACCEL_HIP_CORE_KERNELS_HPP

#include <hip/hip_runtime_api.h>
#include <stddef.h>

// setTo and convertTo kernels of the HIP backend, queued on @p s; rows are @p sstep / @p dstep
// bytes apart. flip and transpose are the kernels the backend shares with CUDA (src/accel/gpu).
namespace cv { namespace accel { namespace hip_impl { namespace kernels {

//! @p val is a Scalar's 4 doubles; every element is written
hipError_t setTo(unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                 const double* val, hipStream_t s);

//! dst = src * @p alpha + @p beta, saturating to @p ddepth
hipError_t convertToScale(const unsigned char* src, size_t sstep, int sdepth,
                          unsigned char* dst, size_t dstep, int ddepth,
                          int rows, int cols, int cn, double alpha, double beta, hipStream_t s);

}}}} // namespace cv::accel::hip_impl::kernels

#endif // OPENCV_HAL_ACCEL_HIP_CORE_KERNELS_HPP
