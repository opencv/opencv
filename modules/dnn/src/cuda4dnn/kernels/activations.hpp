// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATIONS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATIONS_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void relu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T slope);

    template <class T>
    void clipped_relu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T floor, T ceiling);

    template <class T>
    void axiswise_relu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, std::size_t inner_size, csl::View<T> slope);

    template <class T>
    void tanh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void swish(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void mish(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void sigmoid(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void elu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void abs(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void bnll(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void power(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T exp, T scale, T shift);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATIONS_HPP */
