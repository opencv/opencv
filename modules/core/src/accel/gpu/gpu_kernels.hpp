// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_SRC_ACCEL_GPU_KERNELS_HPP
#define OPENCV_CORE_SRC_ACCEL_GPU_KERNELS_HPP

#include "gpu_runtime.hpp"
#include <stddef.h>

// Kernels of the CUDA and HIP backends (gpu_kernels.impl.hpp), launched on @p s. Elements are
// @p esz bytes (supportedElementSize), rows are @p sstep / @p dstep / @p step bytes apart.
namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

bool supportedElementSize(int esz);
//! OpenCV's flip codes: 0 around the x axis, > 0 around the y axis, < 0 both
CV_GPU(Error_t) flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, int mode, CV_GPU(Stream_t) s);
//! flip of @p data in place: every element trades places with its mirror image
CV_GPU(Error_t) flipInplace(unsigned char* data, size_t step, int rows, int cols, int esz, int mode,
                            CV_GPU(Stream_t) s);
//! @p rows x @p cols in, cols x rows out
CV_GPU(Error_t) transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                          int rows, int cols, int esz, CV_GPU(Stream_t) s);
//! transpose of the square @p n x @p n matrix @p data in place
CV_GPU(Error_t) transposeInplace(unsigned char* data, size_t step, int n, int esz, CV_GPU(Stream_t) s);

// imgproc: images of @p depth (CV_8U, ...) with @p cn channels; borders are cv::BorderTypes

//! cv::InterpolationFlags, which core does not see
enum { INTER_NEAREST = 0, INTER_LINEAR = 1 };

//! the longest GaussianBlur kernel, in taps
const int GAUSSIAN_MAX_TAPS = 63;
/** @brief GaussianBlur, a row pass into @p buf, then a column pass. @p kx, @p ky and @p buf are
CV_16U for CV_8U images, CV_32U for CV_16U (fixed point, as the CPU's), CV_32F otherwise. */
CV_GPU(Error_t) gaussianBlur(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                             unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                             const void* kx, int kxn, const void* ky, int kyn, int border, CV_GPU(Stream_t) s);
/** @brief warpAffine with the inverse map @p M (2x3), INTER_NEAREST or INTER_LINEAR, @p bval the
BORDER_CONSTANT pixel already in the range of @p depth. */
CV_GPU(Error_t) warpAffine(const unsigned char* src, size_t sstep, int srows, int scols,
                           unsigned char* dst, size_t dstep, int drows, int dcols, int depth, int cn,
                           const float* M, int interpolation, int border, const float* bval,
                           CV_GPU(Stream_t) s);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#endif // OPENCV_CORE_SRC_ACCEL_GPU_KERNELS_HPP
