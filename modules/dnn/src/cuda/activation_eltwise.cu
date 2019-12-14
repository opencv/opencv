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
    __global__ void relu_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise, T slope) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                auto value = output_vec.data[j];
                output_vec.data[j] = value >= T(0) ? value : slope * value;
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void clipped_relu_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise, T floor, T ceil) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                using device::clamp;
                output_vec.data[j] = clamp(output_vec.data[j], floor, ceil);
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void power_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise, T power) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                using device::pow;
                output_vec.data[j] = pow(output_vec.data[j], power);
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void tanh_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                using device::tanh;
                output_vec.data[j] = tanh(output_vec.data[j]);
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void sigmoid_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                using device::sigmoid;
                output_vec.data[j] = sigmoid(output_vec.data[j]);
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void swish_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                using device::sigmoid;
                auto value = output_vec.data[j];
                output_vec.data[j] = value * sigmoid(value);
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }

    template <class T, std::size_t N>
    __global__ void mish_eltwise_inplace_vec(Span<T> inplace_output, View<T> eltwise) {
        using vector_type = get_vector_type_t<T, N>;

        auto inplace_output_vPtr = vector_type::get_pointer(inplace_output.data());
        auto eltwise_vPtr = vector_type::get_pointer(eltwise.data());

        for (auto i : grid_stride_range(inplace_output.size() / vector_type::size())) {
            vector_type output_vec, eltwise_vec;
            v_load(output_vec, inplace_output_vPtr[i]);
            v_load(eltwise_vec, eltwise_vPtr[i]);
            for(int j = 0; j < output_vec.size(); j++) {
                using device::tanh;
                using device::log1pexp;
                auto value = output_vec.data[j];
                output_vec.data[j] = value * tanh(log1pexp(value));
                output_vec.data[j] += eltwise_vec.data[j];
            }
            v_store(inplace_output_vPtr[i], output_vec);
        }
    }
}

template <class T, std::size_t N> static
void launch_relu_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T slope) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::relu_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise, slope);
}

template <class T>
void relu_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T slope) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_relu_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise, slope);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_relu_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise, slope);
    } else {
        launch_relu_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise, slope);
    }
}

template void relu_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>, __half);
template void relu_eltwise_inplace<float>(const Stream&, Span<float>, View<float>, float);

template <class T, std::size_t N> static
void launch_clipped_relu_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T floor, T ceil){
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::clipped_relu_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise, floor, ceil);
}

template <class T>
void clipped_relu_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T floor, T ceil) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_clipped_relu_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise, floor, ceil);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_clipped_relu_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise, floor, ceil);
    } else {
        launch_clipped_relu_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise, floor, ceil);
    }
}

template void clipped_relu_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>, __half, __half);
template void clipped_relu_eltwise_inplace<float>(const Stream&, Span<float>, View<float>,  float, float);

template <class T, std::size_t N> static
void launch_power_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T power) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::power_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise, power);
}

template <class T>
void power_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise, T power) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_power_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise, power);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_power_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise, power);
    } else {
        launch_power_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise, power);
    }
}

template void power_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>, __half);
template void power_eltwise_inplace<float>(const Stream&, Span<float>, View<float>, float);

template <class T, std::size_t N> static
void launch_tanh_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::tanh_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise);
}

template <class T>
void tanh_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_tanh_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_tanh_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise);
    } else {
        launch_tanh_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise);
    }
}

template void tanh_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void tanh_eltwise_inplace<float>(const Stream&, Span<float>, View<float>);

template <class T, std::size_t N> static
void launch_sigmoid_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::sigmoid_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise);
}

template <class T>
void sigmoid_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_sigmoid_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_sigmoid_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise);
    } else {
        launch_sigmoid_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise);
    }
}

template void sigmoid_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void sigmoid_eltwise_inplace<float>(const Stream&, Span<float>, View<float>);

template <class T, std::size_t N> static
void launch_swish_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::swish_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise);
}

template <class T>
void swish_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_swish_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_swish_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise);
    } else {
        launch_swish_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise);
    }
}

template void swish_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void swish_eltwise_inplace<float>(const Stream&, Span<float>, View<float>);

template <class T, std::size_t N> static
void launch_mish_eltwise_inplace_vec_kernel(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    CV_Assert(is_fully_aligned<T>(inplace_output, N));
    CV_Assert(is_fully_aligned<T>(eltwise, N));

    auto kernel = raw::mish_eltwise_inplace_vec<T, N>;
    auto policy = make_policy(kernel, inplace_output.size() / N, 0, stream);
    launch_kernel(kernel, policy, inplace_output, eltwise);
}

template <class T>
void mish_eltwise_inplace(const Stream& stream, Span<T> inplace_output, View<T> eltwise) {
    if (is_fully_aligned<T>(inplace_output, 4) && is_fully_aligned<T>(eltwise, 4)) {
        launch_mish_eltwise_inplace_vec_kernel<T, 4>(stream, inplace_output, eltwise);
    } else if (is_fully_aligned<T>(inplace_output, 2) && is_fully_aligned<T>(eltwise, 2)) {
        launch_mish_eltwise_inplace_vec_kernel<T, 2>(stream, inplace_output, eltwise);
    } else {
        launch_mish_eltwise_inplace_vec_kernel<T, 1>(stream, inplace_output, eltwise);
    }
}

template void mish_eltwise_inplace<__half>(const Stream&, Span<__half>, View<__half>);
template void mish_eltwise_inplace<float>(const Stream&, Span<float>, View<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
