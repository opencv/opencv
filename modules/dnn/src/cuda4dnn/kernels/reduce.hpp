// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_REDUCE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_REDUCE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
void reduce_sum(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_mean(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_max(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_min(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_prod(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_l1(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_l2(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_sum_square(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_log_sum(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

template <class T>
void reduce_log_sum_exp(const csl::Stream& stream,
    csl::Span<T> output, csl::View<T> input,
    const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& axes);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_REDUCE_HPP */
