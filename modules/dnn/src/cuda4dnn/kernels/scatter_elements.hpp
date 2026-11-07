// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SCATTER_ELEMENTS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SCATTER_ELEMENTS_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T, class TIdx>
void scatter_elements(const csl::Stream& stream,
    csl::Span<T> output, csl::View<TIdx> indices, csl::View<T> updates,
    int rank, int axis,
    const std::vector<std::int64_t>& indices_dims,
    const std::vector<std::int64_t>& output_strides,
    std::int64_t axis_dim_size);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SCATTER_ELEMENTS_HPP */
