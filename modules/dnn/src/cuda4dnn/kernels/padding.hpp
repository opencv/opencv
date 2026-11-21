// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_PADDING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_PADDING_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void copy_with_reflection101(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, csl::TensorView<T> input,
        std::vector<std::pair<std::size_t, std::size_t>> ranges);

    // border rules follow cv::borderInterpolate, as the CPU Pad2 does
    enum class PadMode { CONSTANT, REFLECT101, REPLICATE, WRAP };

    // pads_begin may be negative (crop); output shape already includes the pads
    template <class T>
    void pad_nd(
        const csl::Stream& stream,
        csl::TensorSpan<T> output, csl::TensorView<T> input,
        const std::vector<int>& pads_begin, PadMode mode, T value);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_PADDING_HPP */
