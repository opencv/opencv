// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "math.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T, std::size_t N>
        __global__ void biasN_relu_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type vec;
                v_load(vec, inplace_output_vPtr[i]);
                for(int j = 0; j < vec.size(); j++) {
                    using device::max;
                    vec.data[j] = max(vec.data[j] + bias[bias_idx], T(0));
                }
                v_store(inplace_output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_clipped_relu_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, T floor, T ceil) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type vec;
                v_load(vec, inplace_output_vPtr[i]);
                for(int j = 0; j < vec.size(); j++) {
                    using device::clamp;
                    vec.data[j] = clamp(vec.data[j] + bias[bias_idx], floor, ceil);
                }
                v_store(inplace_output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_tanh_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type vec;
                v_load(vec, inplace_output_vPtr[i]);
                for(int j = 0; j < vec.size(); j++) {
                    using device::tanh;
                    vec.data[j] = tanh(vec.data[j] + bias[bias_idx]);
                }
                v_store(inplace_output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_sigmoid_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type vec;
                v_load(vec, inplace_output_vPtr[i]);
                for(int j = 0; j < vec.size(); j++) {
                    using device::sigmoid;
                    vec.data[j] = sigmoid(vec.data[j] + bias[bias_idx]);
                }
                v_store(inplace_output_vPtr[i], vec);
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_biasN_relu_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_relu_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias);
    }

    template <class T>
    void biasN_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
        if (is_fully_aligned<T>(inplace_output, 4) && inner_size % 4 == 0) {
            launch_biasN_relu_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias);
        } else if (is_fully_aligned<T>(inplace_output, 2) && inner_size % 2 == 0) {
            launch_biasN_relu_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias);
        } else {
            launch_biasN_relu_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias);
        }
    }

    template void biasN_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
    template void biasN_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);

    template <class T, std::size_t N> static
    void launch_biasN_clipped_relu_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, T floor, T ceil){
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_clipped_relu_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, floor, ceil);
    }

    template <class T>
    void biasN_clipped_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, T floor, T ceil) {
        if (is_fully_aligned<T>(inplace_output, 4) && inner_size % 4 == 0) {
            launch_biasN_clipped_relu_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, floor, ceil);
        } else if (is_fully_aligned<T>(inplace_output, 2) && inner_size % 2 == 0) {
            launch_biasN_clipped_relu_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, floor, ceil);
        } else {
            launch_biasN_clipped_relu_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, floor, ceil);
        }
    }

    template void biasN_clipped_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, __half, __half);
    template void biasN_clipped_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, float, float);

    template <class T, std::size_t N> static
    void launch_biasN_tanh_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_tanh_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias);
    }

    template <class T>
    void biasN_tanh_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
        if (is_fully_aligned<T>(inplace_output, 4) && inner_size % 4 == 0) {
            launch_biasN_tanh_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias);
        } else if (is_fully_aligned<T>(inplace_output, 2) && inner_size % 2 == 0) {
            launch_biasN_tanh_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias);
        } else {
            launch_biasN_tanh_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias);
        }
    }

    template void biasN_tanh_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
    template void biasN_tanh_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);

    template <class T, std::size_t N> static
    void launch_biasN_sigmoid_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_sigmoid_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias);
    }

    template <class T>
    void biasN_sigmoid_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias) {
        if (is_fully_aligned<T>(inplace_output, 4) && inner_size % 4 == 0) {
            launch_biasN_sigmoid_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias);
        } else if (is_fully_aligned<T>(inplace_output, 2) && inner_size % 2 == 0) {
            launch_biasN_sigmoid_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias);
        } else {
            launch_biasN_sigmoid_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias);
        }
    }

    template void biasN_sigmoid_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>);
    template void biasN_sigmoid_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
