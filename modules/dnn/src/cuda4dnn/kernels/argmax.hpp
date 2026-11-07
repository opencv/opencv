// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ARGMAX_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ARGMAX_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
void arg_min_max(const csl::Stream& stream,
    csl::Span<std::int64_t> output, csl::View<T> input,
    int outer_size, int axis_size, int inner_size, bool is_argmax, bool select_last_index);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ARGMAX_HPP */
