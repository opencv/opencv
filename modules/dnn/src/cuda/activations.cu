// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn  { namespace kernels {

    namespace raw {
        template <class T, std::size_t N>
        __global__ void abs_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::abs;
                    vec.data[j] = abs(vec.data[j]);
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void tanh_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::tanh;
                    vec.data[j] = tanh(vec.data[j]);
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void swish_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::sigmoid;
                    vec.data[j] = vec.data[j] * sigmoid(vec.data[j]);
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void mish_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::tanh;
                    using device::log1pexp;
                    vec.data[j] = vec.data[j] * tanh(log1pexp(vec.data[j]));
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void sigmoid_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::sigmoid;
                    vec.data[j] = sigmoid(vec.data[j]);
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void bnll_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::log1pexp;
                    vec.data[j] = vec.data[j] > T(0) ? vec.data[j] + log1pexp(-vec.data[j]) : log1pexp(vec.data[j]);
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void elu_vec(Span<T> output, View<T> input) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++) {
                    using device::expm1;
                    vec.data[j] = vec.data[j] >= T(0) ? vec.data[j] : expm1(vec.data[j]);
                }
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void relu_vec(Span<T> output, View<T> input, T slope) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for(int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = vec.data[j] >= T(0) ? vec.data[j] : slope * vec.data[j];
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void clipped_relu_vec(Span<T> output, View<T> input, T floor, T ceiling) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                using device::clamp;

                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = clamp(vec.data[j], floor, ceiling);
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void axiswise_relu_vec(Span<T> output, View<T> input, size_type inner_size, View<T> slope) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            inner_size /= vector_type::size();
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                const index_type c = (i / inner_size) % static_cast<size_type>(slope.size());

                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = vec.data[j] > T(0) ? vec.data[j] : vec.data[j] * slope[c];
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T, std::size_t N>
        __global__ void power_vec(Span<T> output, View<T> input, T exp, T scale, T shift) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            auto input_vPtr = vector_type::get_pointer(input.data());

            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                using device::pow;

                vector_type vec;
                v_load(vec, input_vPtr[i]);
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = pow(shift + scale * vec.data[j], exp);
                v_store(output_vPtr[i], vec);
            }
        }
    }

    template <class T, std::size_t N>
    void launch_vectorized_abs(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::abs_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void abs(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_abs<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_abs<T, 2>(stream, output, input);
        } else {
            launch_vectorized_abs<T, 1>(stream, output, input);
        }
    }

    template void abs<__half>(const Stream& stream, Span<__half> output, View<__half> input);
    template void abs<float>(const Stream& stream, Span<float> output, View<float> input);

    template <class T, std::size_t N>
    void launch_vectorized_tanh(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::tanh_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void tanh(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_tanh<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_tanh<T, 2>(stream, output, input);
        } else {
            launch_vectorized_tanh<T, 1>(stream, output, input);
        }
    }

    template void tanh<__half>(const Stream&, Span<__half>, View<__half>);
    template void tanh<float>(const Stream&, Span<float>, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_swish(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::swish_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void swish(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_swish<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_swish<T, 2>(stream, output, input);
        } else {
            launch_vectorized_swish<T, 1>(stream, output, input);
        }
    }

    template void swish<__half>(const Stream&, Span<__half>, View<__half>);
    template void swish<float>(const Stream&, Span<float>, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_mish(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::mish_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void mish(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_mish<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_mish<T, 2>(stream, output, input);
        } else {
            launch_vectorized_mish<T, 1>(stream, output, input);
        }
    }

    template void mish<__half>(const Stream&, Span<__half>, View<__half>);
    template void mish<float>(const Stream&, Span<float>, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_sigmoid(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::sigmoid_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void sigmoid(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_sigmoid<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_sigmoid<T, 2>(stream, output, input);
        } else {
            launch_vectorized_sigmoid<T, 1>(stream, output, input);
        }
    }

    template void sigmoid<__half>(const Stream&, Span<__half>, View<__half>);
    template void sigmoid<float>(const Stream&, Span<float>, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_bnll(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::bnll_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void bnll(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_bnll<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_bnll<T, 2>(stream, output, input);
        } else {
            launch_vectorized_bnll<T, 1>(stream, output, input);
        }
    }

    template void bnll<__half>(const Stream&, Span<__half>, View<__half>);
    template void bnll<float>(const Stream&, Span<float>, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_elu(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::elu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template <class T>
    void elu(const Stream& stream, Span<T> output, View<T> input) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_elu<T, 4>(stream, output, input);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_elu<T, 2>(stream, output, input);
        } else {
            launch_vectorized_elu<T, 1>(stream, output, input);
        }
    }

    template void elu<__half>(const Stream&, Span<__half>, View<__half>);
    template void elu<float>(const Stream&, Span<float>, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_relu(const Stream& stream, Span<T> output, View<T> input, T slope) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::relu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, slope);
    }

    template <class T>
    void relu(const Stream& stream, Span<T> output, View<T> input, T slope) {
        CV_Assert(input.size() == output.size());

        if(is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_relu<T, 4>(stream, output, input, slope);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_relu<T, 2>(stream, output, input, slope);
        } else {
            launch_vectorized_relu<T, 1>(stream, output, input, slope);
        }
    }

    template void relu<__half>(const Stream&, Span<__half>, View<__half>, __half);
    template void relu<float>(const Stream&, Span<float>, View<float>, float);

    template <class T, std::size_t N>
    void launch_vectorized_clipped_relu(const Stream& stream, Span<T> output, View<T> input, T floor, T ceiling) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::clipped_relu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, floor, ceiling);
    }

    template <class T>
    void clipped_relu(const Stream& stream, Span<T> output, View<T> input, T floor, T ceiling) {
        CV_Assert(input.size() == output.size());
        CV_Assert(static_cast<double>(floor) <= static_cast<double>(ceiling));

        if(is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            launch_vectorized_clipped_relu<T, 4>(stream, output, input, floor, ceiling);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            launch_vectorized_clipped_relu<T, 2>(stream, output, input, floor, ceiling);
        } else {
            launch_vectorized_clipped_relu<T, 1>(stream, output, input, floor, ceiling);
        }
    }

    template void clipped_relu<__half>(const Stream&, Span<__half>, View<__half>, __half, __half);
    template void clipped_relu<float>(const Stream&, Span<float>, View<float>, float, float);

    template <class T, std::size_t N>
    void launch_vectorized_axiswise_relu(const Stream& stream, Span<T> output, View<T> input, std::size_t inner_size, View<T> slope) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));
        CV_Assert(inner_size % N == 0);

        auto kernel = raw::axiswise_relu_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, slope);
    }

    template <class T>
    void axiswise_relu(const Stream& stream, Span<T> output, View<T> input, std::size_t inner_size, View<T> slope) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            launch_vectorized_axiswise_relu<T, 4>(stream, output, input, inner_size, slope);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            launch_vectorized_axiswise_relu<T, 2>(stream, output, input, inner_size, slope);
        } else {
            launch_vectorized_axiswise_relu<T, 1>(stream, output, input, inner_size, slope);
        }
    }

    template void axiswise_relu<__half>(const Stream&, Span<__half>, View<__half>, std::size_t, View<__half>);
    template void axiswise_relu<float>(const Stream&, Span<float>, View<float>, std::size_t, View<float>);

    template <class T, std::size_t N>
    void launch_vectorized_power(const Stream& stream, Span<T> output, View<T> input, T exp, T scale, T shift) {
        CV_Assert(is_fully_aligned<T>(output, N));
        CV_Assert(is_fully_aligned<T>(input, N));

        auto kernel = raw::power_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, input, exp, scale, shift);
    }

    template <class T>
    void power(const Stream& stream, Span<T> output, View<T> input, T exp, T scale, T shift) {
        CV_Assert(input.size() == output.size());

        if (static_cast<float>(exp) == 1.0f) {
            scale1_with_bias1(stream, output, input, scale, shift);
            return;
        }

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && output.size()) {
            launch_vectorized_power<T, 4>(stream, output, input, exp, scale, shift);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && output.size()) {
            launch_vectorized_power<T, 2>(stream, output, input, exp, scale, shift);
        } else {
            launch_vectorized_power<T, 1>(stream, output, input, exp, scale, shift);
        }
    }

    template void power<__half>(const Stream&, Span<__half>, View<__half>, __half, __half, __half);
    template void power<float>(const Stream&, Span<float>, View<float>, float, float, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
