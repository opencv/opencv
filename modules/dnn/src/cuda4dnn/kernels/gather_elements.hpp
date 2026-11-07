// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GATHER_ELEMENTS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GATHER_ELEMENTS_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include "../../cuda/fast_divmod.hpp"

#include <cstddef>
#include <cstdint>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
void gather_elements(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const void* indices, std::size_t index_element_size,
    std::int64_t axis_size_data, std::int64_t inner_size,
    csl::device::fast_divmod axis_inner_size, csl::device::fast_divmod inner_size_divmod);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GATHER_ELEMENTS_HPP */
