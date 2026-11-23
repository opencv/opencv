// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_RMS_NORM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_RMS_NORM_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

/* RMS normalisation: y = scale * x / sqrt(mean(x^2) + epsilon), applied over the last
 * `norm_size` elements of each of `loops` rows. `scale` holds `norm_size` elements.
 *
 * Mirrors fastNormImpl(input, scale, output, eps, axis, recenter=false) in
 * cpu_kernels/fast_norm.cpp -- the recenter=false path forces mean to 0, so the
 * variance term collapses to the mean square.
 *
 * Unlike the LayerNorm path this needs no workspace: one block owns one row and
 * reduces it in registers plus shared memory, instead of accumulating into a global
 * scratch buffer with atomicAdd.
 */
template <class T>
void rms_norm(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input, csl::View<T> scale,
    std::size_t loops, std::size_t norm_size, float epsilon);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_RMS_NORM_HPP */
