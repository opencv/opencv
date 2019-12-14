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
    __global__ void eltwise_sum_2_relu_vec(Span<T> output, View<T> x, View<T> y, T slope) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                T value = vec_x.data[j] + vec_y.data[j];
                vec_x.data[j] = value >= T(0) ? value : slope * value;
            }
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, std::size_t N>
    __global__ void eltwise_sum_2_clipped_relu_vec(Span<T> output, View<T> x, View<T> y, T floor, T ceil) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                using device::clamp;
                vec_x.data[j] = clamp(vec_x.data[j] + vec_y.data[j], floor, ceil);
            }
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, std::size_t N>
    __global__ void eltwise_sum_2_power_vec(Span<T> output, View<T> x, View<T> y, T power) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                using device::pow;
                vec_x.data[j] = pow(vec_x.data[j] + vec_y.data[j], power);
            }
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, std::size_t N>
    __global__ void eltwise_sum_2_tanh_vec(Span<T> output, View<T> x, View<T> y) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                using device::tanh;
                vec_x.data[j] = tanh(vec_x.data[j] + vec_y.data[j]);
            }
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, std::size_t N>
    __global__ void eltwise_sum_2_sigmoid_vec(Span<T> output, View<T> x, View<T> y) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                using device::sigmoid;
                vec_x.data[j] = sigmoid(vec_x.data[j] + vec_y.data[j]);
            }
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, std::size_t N>
    __global__ void eltwise_sum_2_swish_vec(Span<T> output, View<T> x, View<T> y) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                using device::sigmoid;
                T value = vec_x.data[j] + vec_y.data[j];
                vec_x.data[j] = value * sigmoid(value);
            }
            v_store(output_vPtr[i], vec_x);
        }
    }

    template <class T, std::size_t N>
    __global__ void eltwise_sum_2_mish_vec(Span<T> output, View<T> x, View<T> y) {
        using vector_type = get_vector_type_t<T, N>;

        auto output_vPtr = vector_type::get_pointer(output.data());
        auto x_vPtr = vector_type::get_pointer(x.data());
        auto y_vPtr = vector_type::get_pointer(y.data());

        for (auto i : grid_stride_range(output.size() / vector_type::size())) {
            vector_type vec_x, vec_y;
            v_load(vec_x, x_vPtr[i]);
            v_load(vec_y, y_vPtr[i]);
            for(int j = 0; j < vector_type::size(); j++) {
                using device::tanh;
                using device::log1pexp;
                T value = vec_x.data[j] + vec_y.data[j];
                vec_x.data[j] = value * tanh(log1pexp(value));
            }
            v_store(output_vPtr[i], vec_x);
        }
    }
}

template <class T, std::size_t N> static
void launch_eltwise_sum_2_relu_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y, T slope) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_relu_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y, slope);
}

template <class T>
void eltwise_sum_2_relu(const Stream& stream, Span<T> output, View<T> x, View<T> y, T slope) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_relu_vec_kernel<T, 4>(stream, output, x, y, slope);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_relu_vec_kernel<T, 2>(stream, output, x, y, slope);
    } else {
        launch_eltwise_sum_2_relu_vec_kernel<T, 1>(stream, output, x, y, slope);
    }
}

template void eltwise_sum_2_relu<__half>(const Stream&, Span<__half>, View<__half>, View<__half>, __half);
template void eltwise_sum_2_relu<float>(const Stream&, Span<float>, View<float>, View<float>, float);

template <class T, std::size_t N> static
void launch_eltwise_sum_2_clipped_relu_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y, T floor, T ceil){
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_clipped_relu_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y, floor, ceil);
}

template <class T>
void eltwise_sum_2_clipped_relu(const Stream& stream, Span<T> output, View<T> x, View<T> y, T floor, T ceil) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_clipped_relu_vec_kernel<T, 4>(stream, output, x, y, floor, ceil);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_clipped_relu_vec_kernel<T, 2>(stream, output, x, y, floor, ceil);
    } else {
        launch_eltwise_sum_2_clipped_relu_vec_kernel<T, 1>(stream, output, x, y, floor, ceil);
    }
}

template void eltwise_sum_2_clipped_relu<__half>(const Stream&, Span<__half>, View<__half>, View<__half>, __half, __half);
template void eltwise_sum_2_clipped_relu<float>(const Stream&, Span<float>, View<float>, View<float>, float, float);

template <class T, std::size_t N> static
void launch_eltwise_sum_2_power_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y, T power){
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_power_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y, power);
}

template <class T>
void eltwise_sum_2_power(const Stream& stream, Span<T> output, View<T> x, View<T> y, T power) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_power_vec_kernel<T, 4>(stream, output, x, y, power);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_power_vec_kernel<T, 2>(stream, output, x, y, power);
    } else {
        launch_eltwise_sum_2_power_vec_kernel<T, 1>(stream, output, x, y, power);
    }
}

template void eltwise_sum_2_power<__half>(const Stream&, Span<__half>, View<__half>, View<__half>, __half);
template void eltwise_sum_2_power<float>(const Stream&, Span<float>, View<float>, View<float>, float);

template <class T, std::size_t N> static
void launch_eltwise_sum_2_tanh_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_tanh_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y);
}

template <class T>
void eltwise_sum_2_tanh(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_tanh_vec_kernel<T, 4>(stream, output, x, y);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_tanh_vec_kernel<T, 2>(stream, output, x, y);
    } else {
        launch_eltwise_sum_2_tanh_vec_kernel<T, 1>(stream, output, x, y);
    }
}

template void eltwise_sum_2_tanh<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_tanh<float>(const Stream&, Span<float>, View<float>, View<float>);

template <class T, std::size_t N> static
void launch_eltwise_sum_2_sigmoid_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_sigmoid_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y);
}

template <class T>
void eltwise_sum_2_sigmoid(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_sigmoid_vec_kernel<T, 4>(stream, output, x, y);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_sigmoid_vec_kernel<T, 2>(stream, output, x, y);
    } else {
        launch_eltwise_sum_2_sigmoid_vec_kernel<T, 1>(stream, output, x, y);
    }
}

template void eltwise_sum_2_sigmoid<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_sigmoid<float>(const Stream&, Span<float>, View<float>, View<float>);

template <class T, std::size_t N> static
void launch_eltwise_sum_2_swish_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_swish_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y);
}

template <class T>
void eltwise_sum_2_swish(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_swish_vec_kernel<T, 4>(stream, output, x, y);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_swish_vec_kernel<T, 2>(stream, output, x, y);
    } else {
        launch_eltwise_sum_2_swish_vec_kernel<T, 1>(stream, output, x, y);
    }
}

template void eltwise_sum_2_swish<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_swish<float>(const Stream&, Span<float>, View<float>, View<float>);

template <class T, std::size_t N> static
void launch_eltwise_sum_2_mish_vec_kernel(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    CV_Assert(is_fully_aligned<T>(output, N));
    CV_Assert(is_fully_aligned<T>(x, N));
    CV_Assert(is_fully_aligned<T>(y, N));

    auto kernel = raw::eltwise_sum_2_mish_vec<T, N>;
    auto policy = make_policy(kernel, output.size() / N, 0, stream);
    launch_kernel(kernel, policy, output, x, y);
}

template <class T>
void eltwise_sum_2_mish(const Stream& stream, Span<T> output, View<T> x, View<T> y) {
    if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(x, 4) && is_fully_aligned<T>(y, 4)) {
        launch_eltwise_sum_2_mish_vec_kernel<T, 4>(stream, output, x, y);
    } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(x, 2) && is_fully_aligned<T>(y, 2)) {
        launch_eltwise_sum_2_mish_vec_kernel<T, 2>(stream, output, x, y);
    } else {
        launch_eltwise_sum_2_mish_vec_kernel<T, 1>(stream, output, x, y);
    }
}

template void eltwise_sum_2_mish<__half>(const Stream&, Span<__half>, View<__half>, View<__half>);
template void eltwise_sum_2_mish<float>(const Stream&, Span<float>, View<float>, View<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
