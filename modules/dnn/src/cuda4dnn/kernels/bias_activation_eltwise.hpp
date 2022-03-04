// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BIAS_ACTIVATION_ELTWISE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BIAS_ACTIVATION_ELTWISE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    /* inplace_output = activation(inplace_output + bias) + eltwise
     * broadcasting on `bias` is allowed but not on `eltwise`
     */

    template <class T>
    void biasN_relu_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise, T slope);

    template <class T>
    void biasN_clipped_relu_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise, T floor, T ceiling);

    template <class T>
    void biasN_tanh_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise);

    template <class T>
    void biasN_sigmoid_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise);

    template <class T>
    void biasN_swish_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise);

    template <class T>
    void biasN_mish_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise);

    template <class T>
    void biasN_power_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, std::size_t inner_size, csl::View<T> bias, csl::View<T> eltwise, T exp, T scale, T shift);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_BIAS_ACTIVATION_ELTWISE_HPP */
