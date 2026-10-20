// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_AVERAGE_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_AVERAGE_POOLING_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
void average_pool(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& input_shape,
    const std::vector<std::int64_t>& output_shape,
    const std::vector<std::int64_t>& kernel_shape,
    const std::vector<std::int64_t>& strides,
    const std::vector<std::int64_t>& pads,
    const std::vector<std::int64_t>& dilations,
    bool count_include_pad);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_AVERAGE_POOLING_HPP */
