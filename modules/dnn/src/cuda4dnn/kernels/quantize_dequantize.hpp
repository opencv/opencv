// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_QUANTIZE_DEQUANTIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_QUANTIZE_DEQUANTIZE_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>
#include <cstdint>

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

// Number of (min, max) partial pairs dynamic_quantize_scale() keeps in scratch for n inputs.
std::size_t dynamic_quantize_partials(std::size_t n);

// ONNX DynamicQuantizeLinear y_scale and uint8 y_zero_point from the input range, which is forced
// to include zero; scratch holds 2 * dynamic_quantize_partials(input.size()) floats.
void dynamic_quantize_scale(const csl::Stream& stream, csl::Span<float> scale, csl::Span<std::uint8_t> zero_point,
                            csl::View<float> input, csl::Span<float> scratch);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_QUANTIZE_DEQUANTIZE_HPP */
