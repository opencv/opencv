// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SCALE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SCALE_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void biasN(const csl::Stream& stream,
        csl::TensorSpan<T> output,
        csl::TensorView<T> input, std::size_t inner_size,
        csl::TensorView<T> bias);

    template <class T>
    void scaleN(const csl::Stream& stream,
        csl::TensorSpan<T> output,
        csl::TensorView<T> input, std::size_t inner_size,
        csl::TensorView<T> weights);

    template <class T>
    void scale1_with_bias1(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input, T alpha, T beta);

    template <class T>
    void scaleN_with_biasN(
        const csl::Stream& stream,
        csl::TensorSpan<T> output,
        csl::TensorView<T> input, std::size_t inner_size,
        csl::TensorView<T> weights, csl::TensorView<T> bias);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_SCALE_HPP */
