// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void bias1(span<T> output, view<T> input, T beta) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = input[i] + beta;
        }

        template <class T, std::size_t N>
        __global__ void biasN_vec(span<T> output, view<T> input, size_type inner_size, view<T> bias) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type vec;
                v_load<vector_type>(vec, input_vPtr[i]);
                for(int j = 0; j < vec.size(); j++)
                    vec.data[j] = vec.data[j] + bias[bias_idx];
                v_store<vector_type>(output_vPtr[i], vec);
            }
        }

        template <class T>
        __global__ void scale1(span<T> output, view<T> input, T alpha)
        {
            for (auto i : grid_stride_range(output.size()))
                output[i] = alpha * input[i];
        }

        template <class T, std::size_t N>
        __global__ void scaleN_vec(span<T> output, view<T> input, size_type inner_size, view<T> weights)
        {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                const index_type scale_idx = (i / inner_size) % static_cast<size_type>(weights.size());

                vector_type vec;
                v_load<vector_type>(vec, input_vPtr[i]);
                for (int j = 0; j < vec.size(); j++)
                    vec.data[j] = vec.data[j] * weights[scale_idx];
                v_store<vector_type>(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void scale1_with_bias1_vec(span<T> output, view<T> input, T alpha, T beta)
        {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load<vector_type>(vec, input_vPtr[i]);
                for (int j = 0; j < vec.size(); j++)
                    vec.data[j] = alpha * vec.data[j] + beta;
                v_store<vector_type>(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void scaleN_with_biasN_vec(span<T> output, view<T> input, size_type inner_size, view<T> weights, view<T> bias)
        {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                const index_type scale_idx = (i / inner_size) % static_cast<size_type>(weights.size());

                vector_type vec;
                v_load<vector_type>(vec, input_vPtr[i]);
                for (int j = 0; j < vec.size(); j++)
                    vec.data[j] = vec.data[j] * weights[scale_idx] + bias[scale_idx];
                v_store<vector_type>(output_vPtr[i], vec);
            }
        }
    }

    template <class T>
    void bias1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T beta) {
        CV_Assert(is_shape_same(input, output));

        auto kernel = raw::scale1<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, beta);
    }

    template void bias1<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, __half);
    template void bias1<float>(const Stream&, TensorSpan<float>, TensorView<float>, float);
    template void bias1<double>(const Stream&, TensorSpan<double>, TensorView<double>, double);

    template <class T, std::size_t N> static
    void launch_biasN_vec_kernel(const Stream& stream, span<T> output, view<T> input, std::size_t inner_size, view<T> bias) {
        auto kernel = raw::biasN_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, bias);
    }

    template <class T>
    void biasN(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> bias)
    {
        CV_Assert(is_shape_same(input, output));

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            launch_biasN_vec_kernel<T, 4>(stream, output, input, inner_size, bias);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            launch_biasN_vec_kernel<T, 2>(stream, output, input, inner_size, bias);
        } else {
            launch_biasN_vec_kernel<T, 1>(stream, output, input, inner_size, bias);
        }
    }

    template void biasN<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, std::size_t, TensorView<__half>);
    template void biasN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>);
    template void biasN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>);

    template <class T>
    void scale1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T alpha) {
        CV_Assert(is_shape_same(input, output));

        auto kernel = raw::scale1<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, alpha);
    }

    template void scale1<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, __half);
    template void scale1<float>(const Stream&, TensorSpan<float>, TensorView<float>, float);
    template void scale1<double>(const Stream&, TensorSpan<double>, TensorView<double>, double);

    template <class T, std::size_t N> static
    void launch_scaleN_vec_kernel(const Stream& stream, span<T> output, view<T> input, std::size_t inner_size, view<T> weights) {
        auto kernel = raw::scaleN_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, weights);
    }

    template <class T>
    void scaleN(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights)
    {
        CV_Assert(is_shape_same(input, output));

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            launch_scaleN_vec_kernel<T, 4>(stream, output, input, inner_size, weights);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            launch_scaleN_vec_kernel<T, 2>(stream, output, input, inner_size, weights);
        } else {
            launch_scaleN_vec_kernel<T, 1>(stream, output, input, inner_size, weights);
        }
    }

    template void scaleN<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, std::size_t, TensorView<__half>);
    template void scaleN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>);
    template void scaleN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>);

    template <class T, std::size_t N> static
    void launch_scale1_with_bias1_vec_kernel(const Stream& stream, span<T> output, view<T> input, T alpha, T beta) {
        auto kernel = raw::scale1_with_bias1_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, alpha, beta);
    }

    template <class T>
    void scale1_with_bias1(const Stream& stream, span<T> output, view<T> input, T alpha, T beta) {
        CV_Assert(output.size() == input.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_scale1_with_bias1_vec_kernel<T, 4>(stream, output, input, alpha, beta);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_scale1_with_bias1_vec_kernel<T, 2>(stream, output, input, alpha, beta);
        } else {
            launch_scale1_with_bias1_vec_kernel<T, 1>(stream, output, input, alpha, beta);
        }
    }

    template void scale1_with_bias1<__half>(const Stream&, span<__half>, view<__half>, __half, __half);
    template void scale1_with_bias1<float>(const Stream&, span<float>, view<float>, float, float);
    template void scale1_with_bias1<double>(const Stream&, span<double>, view<double>, double, double);

    template <class T, std::size_t N> static
    void launch_scaleN_with_biasN_vec_kernel(const Stream& stream, span<T> output, view<T> input, std::size_t inner_size, view<T> weights, view<T> bias) {
        auto kernel = raw::scaleN_with_biasN_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, weights, bias);
    }

    template <class T>
    void scaleN_with_biasN(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights, TensorView<T> bias)
    {
        CV_Assert(is_shape_same(input, output));
        CV_Assert(weights.size() == bias.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            launch_scaleN_with_biasN_vec_kernel<T, 4>(stream, output, input, inner_size, weights, bias);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            launch_scaleN_with_biasN_vec_kernel<T, 2>(stream, output, input, inner_size, weights, bias);
        } else {
            launch_scaleN_with_biasN_vec_kernel<T, 1>(stream, output, input, inner_size, weights, bias);
        }
    }

    template void scaleN_with_biasN<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, std::size_t, TensorView<__half>, TensorView<__half>);
    template void scaleN_with_biasN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>, TensorView<float>);
    template void scaleN_with_biasN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>, TensorView<double>);

}}}} /* cv::dnn::cuda4dnn::kernels */
