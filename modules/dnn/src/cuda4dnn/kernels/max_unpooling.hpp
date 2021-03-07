// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MAX_UNPOOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MAX_UNPOOLING_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include <cstddef>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void max_pooling_with_indices(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, csl::TensorSpan<T> indices, csl::TensorView<T> input,
        const std::vector<std::size_t>& kernel_size, const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& padding_left);

    template <class T>
    void max_unpooling(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, csl::TensorView<T> input, csl::TensorView<T> indices,
        const std::vector<std::size_t>& window_size, const std::vector<std::size_t>& strides,
        const std::vector<std::size_t>& padding_left);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_MAX_UNPOOLING_HPP */
