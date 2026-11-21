// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GATHER_ND_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GATHER_ND_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T, class TIdx>
void gather_nd(const csl::Stream& stream,
    csl::Span<T> output, csl::View<TIdx> indices, csl::View<T> input,
    std::int64_t last_index_dimension,
    const std::vector<std::int64_t>& element_counts,
    const std::vector<std::int64_t>& input_dims,
    std::int64_t inner_size,
    std::int64_t batch_stride,
    std::int64_t num_slices_per_batch);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_GATHER_ND_HPP */
