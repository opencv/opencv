// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATION_ELTWISE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATION_ELTWISE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    /* inplace_output = activation(inplace_output) + eltwise */

    template <class T>
    void relu_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise, T slope);

    template <class T>
    void clipped_relu_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise, T floor, T ceiling);

    template <class T>
    void tanh_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise);

    template <class T>
    void swish_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise);

    template <class T>
    void mish_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise);

    template <class T>
    void sigmoid_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise);

    template <class T>
    void power_eltwise_sum_2_inplace(const csl::Stream& stream, csl::Span<T> inplace_output, csl::View<T> eltwise, T exp, T scale, T shift);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ACTIVATION_ELTWISE_HPP */
