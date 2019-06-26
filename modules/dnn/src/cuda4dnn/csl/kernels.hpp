// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_KERNELS_HPP
#define OPENCV_DNN_CUDA4DNN_KERNELS_HPP

#include "stream.hpp"
#include "memory.hpp"
#include "span.hpp"
#include "tensor.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace kernels {

    template <class T>
    void abs(const Stream& stream, span<T> dest, view<T> src);

    template <class T>
    void tanh(const Stream& stream, span<T> dest, view<T> src);

    template <class T>
    void sigmoid(const Stream& stream, span<T> dest, view<T> src);

    template <class T>
    void bnll(const Stream& stream, span<T> dest, view<T> src);

    template <class T>
    void elu(const Stream& stream, span<T> dest, view<T> src);

    template <class T>
    void relu(const Stream& stream, span<T> dest, view<T> src, T slope);

    template <class T>
    void clipped_relu(const Stream& stream, span<T> dest, view<T> src, T floor, T ceiling);

    template <class T>
    void axiswise_relu(const Stream& stream, span<T> dest, view<T> src, view<T> slope, std::size_t inner_size, std::size_t channel_size);

    template <class T>
    void power(const Stream& stream, span<T> dest, view<T> src, T exp, T scale, T shift);

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis);

    template <class T>
    void scale(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights);

    template <class T>
    void scale_with_bias(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights, TensorView<T> bias);

    template <class T>
    void eltwise_max_2(const Stream& stream, span<T> output, view<T> x, view<T> y);

    template <class T>
    void eltwise_sum_2(const Stream& stream, span<T> output, view<T> x, view<T> y);

    template <class T>
    void eltwise_sum_coeff_2(const Stream& stream, span<T> output, T coeff_x, view<T> x, T coeff_y, view<T> y);

    template <class T>
    void eltwise_prod_2(const Stream& stream, span<T> output, view<T> x, view<T> y);

}}}}} /* namespace cv::dnn::cuda4dnn::csl::kernels */

#endif /* OPENCV_DNN_CUDA4DNN_KERNELS_HPP */
