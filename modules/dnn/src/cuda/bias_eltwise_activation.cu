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
        __global__ void biasN_eltwise_identity_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for (int j = 0; j < output_vec.size(); j++)
                    output_vec.data[j] += bias[bias_idx] + eltwise_vec.data[j];
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_relu_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise, T slope) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    auto value = output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j];
                    output_vec.data[j] = value >= T(0) ? value : slope * value;
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_clipped_relu_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise, T floor, T ceil) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    using device::clamp;
                    output_vec.data[j] = clamp(output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j], floor, ceil);
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_power_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise, T power) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    using device::pow;
                    output_vec.data[j] = pow(output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j], power);
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_tanh_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    using device::tanh;
                    output_vec.data[j] = tanh(output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j]);
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_sigmoid_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    using device::sigmoid;
                    output_vec.data[j] = sigmoid(output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j]);
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_swish_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    using device::sigmoid;
                    auto value = output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j];
                    output_vec.data[j] = value * sigmoid(value);
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void biasN_eltwise_mish_inplace_vec(Span<T> inplace_output, size_type inner_size, View<T> bias, View<T> eltwise) {
            using vector_type = get_vector_type_t<T, N>;

            auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
            auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
                const index_type bias_idx = (i / inner_size) % static_cast<size_type>(bias.size());

                vector_type output_vec, eltwise_vec;
                v_load(output_vec, inplace_output_vPtr[i]);
                v_load(eltwise_vec, eltwise_vPtr[i]);
                for(int j = 0; j < output_vec.size(); j++) {
                    using device::tanh;
                    using device::log1pexp;
                    auto value = output_vec.data[j] + bias[bias_idx] + eltwise_vec.data[j];
                    output_vec.data[j] = value * tanh(log1pexp(value));
                }
                v_store(inplace_output_vPtr[i], output_vec);
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_identity_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_identity_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise);
    }

    template <class T>
    void biasN_eltwise_identity_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_identity_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_identity_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise);
        } else {
            launch_biasN_eltwise_identity_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise);
        }
    }

    template void biasN_eltwise_identity_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
    template void biasN_eltwise_identity_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_relu_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T slope) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_relu_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise, slope);
    }

    template <class T>
    void biasN_eltwise_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T slope) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_relu_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise, slope);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_relu_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise, slope);
        } else {
            launch_biasN_eltwise_relu_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise, slope);
        }
    }

    template void biasN_eltwise_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>, __half);
    template void biasN_eltwise_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>, float);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_clipped_relu_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T floor, T ceil){
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_clipped_relu_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise, floor, ceil);
    }

    template <class T>
    void biasN_eltwise_clipped_relu_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T floor, T ceil) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_clipped_relu_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise, floor, ceil);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_clipped_relu_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise, floor, ceil);
        } else {
            launch_biasN_eltwise_clipped_relu_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise, floor, ceil);
        }
    }

    template void biasN_eltwise_clipped_relu_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>, __half, __half);
    template void biasN_eltwise_clipped_relu_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>,  float, float);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_power_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T power){
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_power_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise, power);
    }

    template <class T>
    void biasN_eltwise_power_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise, T power) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_power_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise, power);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_power_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise, power);
        } else {
            launch_biasN_eltwise_power_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise, power);
        }
    }

    template void biasN_eltwise_power_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>, __half);
    template void biasN_eltwise_power_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>, float);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_tanh_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_tanh_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise);
    }

    template <class T>
    void biasN_eltwise_tanh_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_tanh_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_tanh_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise);
        } else {
            launch_biasN_eltwise_tanh_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise);
        }
    }

    template void biasN_eltwise_tanh_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
    template void biasN_eltwise_tanh_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_sigmoid_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_sigmoid_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise);
    }

    template <class T>
    void biasN_eltwise_sigmoid_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_sigmoid_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_sigmoid_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise);
        } else {
            launch_biasN_eltwise_sigmoid_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise);
        }
    }

    template void biasN_eltwise_sigmoid_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
    template void biasN_eltwise_sigmoid_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_swish_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_swish_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise);
    }

    template <class T>
    void biasN_eltwise_swish_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_swish_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_swish_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise);
        } else {
            launch_biasN_eltwise_swish_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise);
        }
    }

    template void biasN_eltwise_swish_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
    template void biasN_eltwise_swish_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);

    template <class T, std::size_t N> static
    void launch_biasN_eltwise_mish_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        CV_Assert(is_fully_aligned<T>(inplace_output, N));
        CV_Assert(is_fully_aligned<T>(eltwise, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::biasN_eltwise_mish_inplace_vec<T, N>;
        auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
        launch_kernel(kernel, policy, inplace_output, inner_size, bias, eltwise);
    }

    template <class T>
    void biasN_eltwise_mish_inplace(const Stream& stream, Span<T> inplace_output, std::size_t inner_size, View<T> bias, View<T> eltwise) {
        if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4) && inner_size % 4 == 0) {
            launch_biasN_eltwise_mish_inplace_vec_kernel<T, 4>(stream, inplace_output, inner_size, bias, eltwise);
        } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2) && inner_size % 2 == 0) {
            launch_biasN_eltwise_mish_inplace_vec_kernel<T, 2>(stream, inplace_output, inner_size, bias, eltwise);
        } else {
            launch_biasN_eltwise_mish_inplace_vec_kernel<T, 1>(stream, inplace_output, inner_size, bias, eltwise);
        }
    }

    template void biasN_eltwise_mish_inplace<__half>(const Stream&, Span<__half>, std::size_t, View<__half>, View<__half>);
    template void biasN_eltwise_mish_inplace<float>(const Stream&, Span<float>, std::size_t, View<float>, View<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
