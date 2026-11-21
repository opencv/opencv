// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_NORMALIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_NORMALIZE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void normalize(
        const csl::Stream& stream,
        csl::Span<T> output, csl::View<T> input,
        std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, std::size_t norm, T epsilon,
        csl::Span<T> workspace);

    // ONNX LRN: y = x * (bias + alpha_scaled * sum of x^2 over channels [c - radius, c + radius])^-beta
    template <class T>
    void lrn_across_channels(
        const csl::Stream& stream,
        csl::Span<T> output, csl::View<T> input,
        std::size_t batch, std::size_t channels, std::size_t plane_size,
        int radius, float alpha_scaled, float bias, float beta);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_NORMALIZE_HPP */
