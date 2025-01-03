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
    void elu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T alpha);

    template <class T>
    void abs(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void bnll(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void ceil(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void floor(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void log(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void rint(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void sqrt(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void not_k(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void acos(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void acosh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void asin(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void asinh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void atan(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void atanh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void cos(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void cosh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void erf(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void hardswish(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void sin(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void sinh(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void softplus(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void softsign(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void tan(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void celu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T alpha);

    template <class T>
    void hardsigmoid(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T alpha, T beta);

    template <class T>
    void selu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T alpha, T gamma);

    template <class T>
    void gelu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void thresholdedrelu(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T alpha);

    template <class T>
    void power(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T exp, T scale, T shift);

    template <class T>
    void exp(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T normScale, T normShift);

    template <class T>
    void sign(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);

    template <class T>
    void shrink(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T bias, T lambd);

    template <class T>
    void reciprocal(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input);
}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATIONS_HPP */
