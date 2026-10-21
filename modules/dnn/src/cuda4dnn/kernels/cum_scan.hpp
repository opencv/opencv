// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CUM_SCAN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CUM_SCAN_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

/* ONNX CumSum / CumProd over one axis, with the input viewed as
 * [outer_size, target_size, inner_size].
 *
 * `is_prod` picks the operator and the exclusive-mode seed (0 for sum, 1 for product);
 * `exclusive` and `reverse` follow the ONNX attributes. Mirrors the scan in
 * cumsum_layer.cpp / cumprod_layer.cpp exactly, including their offset arithmetic.
 *
 * The scan along the axis is inherently sequential, so a thread owns one
 * (outer, inner) pair and walks the axis; parallelism comes from outer * inner, which
 * is where the width is in practice.
 */
template <class T>
void cum_scan(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    std::size_t outer_size, std::size_t target_size, std::size_t inner_size,
    bool is_prod, bool exclusive, bool reverse);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CUM_SCAN_HPP */
