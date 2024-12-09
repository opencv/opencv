// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CONCAT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CONCAT_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include <cstddef>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void concat(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, std::size_t output_axis_offset,
        csl::TensorView<T> input, std::size_t axis);

    template <class T>
    void concat_with_offsets(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input, std::vector<std::size_t> axis_offsets);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_CONCAT_HPP */
