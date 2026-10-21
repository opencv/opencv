// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_QUANTIZE_DEQUANTIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_QUANTIZE_DEQUANTIZE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

// Per-tensor (sz_a == 1) and per-axis (sz_a == input.size(axis)) ONNX DequantizeLinear/
// QuantizeLinear, block_size == 0 only. zero_point may be an empty view (implies zero).
template <class T>
void dequantize_linear(const csl::Stream& stream, csl::Span<float> output, csl::View<T> input,
                        csl::View<float> scale, csl::View<T> zero_point,
                        std::size_t sz_a, std::size_t slice_size);

template <class T>
void quantize_linear(const csl::Stream& stream, csl::Span<T> output, csl::View<float> input,
                      csl::View<float> scale, csl::View<T> zero_point,
                      std::size_t sz_a, std::size_t slice_size);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_QUANTIZE_DEQUANTIZE_HPP */
