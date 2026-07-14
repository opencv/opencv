// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

#include <opencv2/core.hpp>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

namespace raw {
    template <class T, std::size_t N>
    __global__ void input_shortcut_vec(
        Span<T> output,
        View<T> input, index_type c_input, /* `c_input` = number of channels in `input` */
        View<T> from, index_type c_from, /* `c_from` = number of channels in `from` */
        size_type channel_stride /* common for both `input` and `from` */)
    {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto input_vPtr = vector_type::get_pointer(input.data());
        auto from_vPtr = vector_type::get_pointer(from.data());

        auto batch_stride_input = c_input * channel_stride;
        auto batch_stride_from = c_from * channel_stride;

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            const auto actual_idx = i * vector_type::size();
            const auto b = actual_idx / batch_stride_input; /* `input` and `output` have the same shape */
            const auto c = (actual_idx % batch_stride_input) / channel_stride;
            const auto c_offset = actual_idx % channel_stride;

            vector_type vec_input;
            v_load(vec_input, input_vPtr[i]);

            /* We can break down the shortcut operation into two steps:
             * - copy `input` to `output`
             * - add `from` to corresponding channels in `output`
             *
             * In this scheme, only some channels in the `output` differ from `input`. They differ in the channels
             * which have a corresponding channel in `from`.
             */
            if (c < c_from) {
                const auto from_actual_idx = b * batch_stride_from + c * channel_stride + c_offset;
                const auto from_vec_idx = from_actual_idx / vector_type::size();

                vector_type vec_from;
                v_load(vec_from, from_vPtr[from_vec_idx]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec_input.data[j] += vec_from.data[j];
            }

            v_store(output_vPtr[i], vec_input);
        }
    }
}

template <class T, std::size_t N>
void launch_vectorized_input_shortcut(const Stream& stream, Span<T> output, View<T> input, std::size_t c_input, View<T> from, std::size_t c_from, std::size_t channel_stride) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(input, N));
    CV_Assert(is_fully_aligned<T>(from, N));
    CV_Assert(channel_stride % N == 0);

    auto kernel = raw::input_shortcut_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, input, c_input, from, c_from, channel_stride);
}

template <class T>
void input_shortcut(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input, csl::TensorView<T> from) {
    CV_Assert(is_shape_same(output, input));
    CV_Assert(output.rank() == from.rank());
    for (int i = 0; i < output.rank(); i++) {
        if (i != 1) {
            CV_Assert(from.get_axis_size(i) == output.get_axis_size(i));
        }
    }

    auto channel_stride = output.size_range(2, output.rank()); /* same for `output`, `input` and `from` */
    auto c_input = input.get_axis_size(1);
    auto c_from = from.get_axis_size(1);

    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && is_fully_aligned<T>(from, 4) && channel_stride % 4 == 0) {
        launch_vectorized_input_shortcut<T, 4>(stream, output, input, c_input, from, c_from, channel_stride);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && is_fully_aligned<T>(from, 2) && channel_stride % 2 == 0) {
        launch_vectorized_input_shortcut<T, 2>(stream, output, input, c_input, from, c_from, channel_stride);
    } else {
        launch_vectorized_input_shortcut<T, 1>(stream, output, input, c_input, from, c_from, channel_stride);
    }
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void input_shortcut(const Stream&, TensorSpan<__half>, TensorView<__half>, TensorView<__half>);
#endif
template void input_shortcut(const Stream&, TensorSpan<float>, TensorView<float>, TensorView<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
