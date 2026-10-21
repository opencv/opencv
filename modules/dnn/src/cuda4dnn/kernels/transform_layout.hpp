// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TRANSFORM_LAYOUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TRANSFORM_LAYOUT_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
void transform_layout_to_block(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input, int C, int C0, int C1, std::int64_t planesize);

template <class T>
void transform_layout_from_block(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input, int C, int C0, int C1, std::int64_t planesize);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_TRANSFORM_LAYOUT_HPP */
