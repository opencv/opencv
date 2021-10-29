// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MVN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MVN_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
void reduce_mean(const csl::Stream& stream, csl::Span<float> means, csl::View<T> input, std::size_t inner_size);

template <class T>
void reduce_mean_sqr_sum(const csl::Stream& stream, csl::Span<float> means, csl::Span<float> sum_sqrs, csl::View<T> input, std::size_t inner_size);

void compute_normalization_scale(const csl::Stream& stream, csl::Span<float> scale, csl::View<float> means, csl::View<float> sum_sqrs, std::size_t inner_size, float eps);

template <class T>
void normalize_mean(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, csl::View<float> means, std::size_t inner_size);

template <class T>
void normalize_mean_variance(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, csl::View<float> means, csl::View<float> scale, std::size_t inner_size);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MVN_HPP */
