// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BIAS_ACTIVATION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BIAS_ACTIVATION_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void biasN_relu_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, T slope);

    template <class T>
    void biasN_clipped_relu_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, T floor, T ceiling);

    template <class T>
    void biasN_power_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, T exp);

    template <class T>
    void biasN_tanh_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias);

    template <class T>
    void biasN_sigmoid_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias);

    template <class T>
    void biasN_swish_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias);

    template <class T>
    void biasN_mish_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BIAS_ACTIVATION_HPP */
