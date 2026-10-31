// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_CUDA_ACCEL_KERNELS_HPP
#define OPENCV_CORE_SRC_CUDA_ACCEL_KERNELS_HPP

#include <cuda_runtime.h>
#include <stddef.h>

// Kernels of the CUDA backend of the non-CPU HAL (accel_kernels.cu), launched on @p s. Elements
// are @p esz bytes (supportedElementSize), rows are @p sstep / @p dstep bytes apart.
namespace cv { namespace accel { namespace impl { namespace cuda_kernels {

bool supportedElementSize(int esz);
//! OpenCV's flip codes: 0 around the x axis, > 0 around the y axis, < 0 both
cudaError_t flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                 int rows, int cols, int esz, int mode, cudaStream_t s);
//! @p rows x @p cols in, cols x rows out
cudaError_t transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, int esz, cudaStream_t s);

}}}} // namespace cv::accel::impl::cuda_kernels

#endif // OPENCV_CORE_SRC_CUDA_ACCEL_KERNELS_HPP
