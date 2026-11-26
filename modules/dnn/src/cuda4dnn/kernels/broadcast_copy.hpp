// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BROADCAST_COPY_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BROADCAST_COPY_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

/* output[coord] = input[coord % in_dims] per axis, which serves both Tile and Expand;
 * in_dims must be left-padded with 1s to the output rank (at most kMaxBroadcastRank) */
constexpr int kMaxBroadcastRank = 8;

template <class T>
void broadcast_copy(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& in_dims,
    const std::vector<std::int64_t>& out_dims);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BROADCAST_COPY_HPP */
