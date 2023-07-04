// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ELTWISE_ACTIVATION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ELTWISE_ACTIVATION_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    /* output = activation(x + y) */

    template <class T>
    void eltwise_sum_2_relu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y, T slope);

    template <class T>
    void eltwise_sum_2_clipped_relu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y, T floor, T ceiling);

    template <class T>
    void eltwise_sum_2_tanh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_sum_2_swish(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_sum_2_mish(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_sum_2_sigmoid(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y);

    template <class T>
    void eltwise_sum_2_power(const csl::Stream& stream, csl::Span<T> output, csl::View<T> x, csl::View<T> y, T exp, T scale, T shift);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ELTWISE_ACTIVATION_HPP */
