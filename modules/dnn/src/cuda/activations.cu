// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {

        using index_type = gpu::index_type;
        using size_type = gpu::size_type;

        template <class T>
        __global__ void abs(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::abs;
                output[i] = abs(input[i]);
            }
        }

        template <class T>
        __global__ void tanh(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::tanh;
                output[i] = tanh(input[i]);
            }
        }

        template <class T>
        __global__ void sigmoid(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::sigmoid;
                output[i] = sigmoid(input[i]);
            }
        }

        template <class T>
        __global__ void bnll(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::log1pexp;
                output[i] = input[i] > T(0) ? input[i] + log1pexp(-input[i]) : log1pexp(input[i]);
            }
        }

        template <class T>
        __global__ void elu(span<T> output, view<T> input) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::exp;
                output[i] = input[i] >= T(0) ? input[i] : expm1(input[i]);
            }
        }

        template <class T>
        __global__ void relu_vec4(span<T> output, view<T> input, T slope) {
            using vector_type = typename get_vector_type<T, 4>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            for (auto i : grid_stride_range(output.size() / 4)) {
                vector_type vec = srcPtr[i];
                vec.w = vec.w >= T(0) ? vec.w : slope * vec.w;
                vec.x = vec.x >= T(0) ? vec.x : slope * vec.x;
                vec.y = vec.y >= T(0) ? vec.y : slope * vec.y;
                vec.z = vec.z >= T(0) ? vec.z : slope * vec.z;
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void relu_vec2(span<T> output, view<T> input, T slope) {
            using vector_type = typename get_vector_type<T, 2>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            for (auto i : grid_stride_range(output.size() / 2)) {
                vector_type vec = srcPtr[i];
                vec.x = vec.x >= T(0) ? vec.x : slope * vec.x;
                vec.y = vec.y >= T(0) ? vec.y : slope * vec.y;
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void relu(span<T> output, view<T> input, T slope) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = input[i] >= T(0) ? input[i] : slope * input[i];
        }

        template <class T>
        __global__ void clipped_relu_vec4(span<T> output, view<T> input, T floor, T ceiling) {
            using vector_type = typename get_vector_type<T, 4>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            for (auto i : grid_stride_range(output.size() / 4)) {
                using utils::clamp;

                vector_type vec = srcPtr[i];
                vec.w = clamp(vec.w, floor, ceiling);
                vec.x = clamp(vec.x, floor, ceiling);
                vec.y = clamp(vec.y, floor, ceiling);
                vec.z = clamp(vec.z, floor, ceiling);
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void clipped_relu_vec2(span<T> output, view<T> input, T floor, T ceiling) {
            using vector_type = typename get_vector_type<T, 2>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            for (auto i : grid_stride_range(output.size() / 2)) {
                using utils::clamp;

                vector_type vec = srcPtr[i];
                vec.x = clamp(vec.x, floor, ceiling);
                vec.y = clamp(vec.y, floor, ceiling);
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void clipped_relu(span<T> output, view<T> input, T floor, T ceiling) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::clamp;
                output[i] = clamp(input[i], floor, ceiling);
            }
        }

        template <class T>
        __global__ void axiswise_relu(span<T> output, view<T> input, size_type inner_size, view<T> slope) {
            for (auto i : grid_stride_range(output.size())) {
                const index_type c = (i % inner_size) / static_cast<size_type>(slope.size());
                output[i] = input[i] < T(0) ? input[i] * slope[c] : input[i];
            }
        }

        template <class T>
        __global__ void power_vec4(span<T> output, view<T> input, T exp, T scale, T shift) {
            using vector_type = typename get_vector_type<T, 4>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            for (auto i : grid_stride_range(output.size() / 4)) {
                using utils::pow;

                vector_type vec = srcPtr[i];
                vec.w = pow(shift + scale * vec.w, exp);
                vec.x = pow(shift + scale * vec.x, exp);
                vec.y = pow(shift + scale * vec.y, exp);
                vec.z = pow(shift + scale * vec.z, exp);
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void power_vec2(span<T> output, view<T> input, T exp, T scale, T shift) {
            using vector_type = typename get_vector_type<T, 2>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            for (auto i : grid_stride_range(output.size() / 2)) {
                using utils::pow;

                vector_type vec = srcPtr[i];
                vec.x = pow(shift + scale * vec.x, exp);
                vec.y = pow(shift + scale * vec.y, exp);
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void power(span<T> output, view<T> input, T exp, T scale, T shift) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::pow;
                output[i] = pow(shift + scale * input[i], exp);
            }
        }
    }

    template <class T>
    void abs(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::abs<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void abs<__half>(const Stream& stream, span<__half> output, view<__half> input);
    template void abs<float>(const Stream& stream, span<float> output, view<float> input);
    template void abs<double>(const Stream& stream, span<double> output, view<double> input);

    template <class T>
    void tanh(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::tanh<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void tanh<__half>(const Stream&, span<__half>, view<__half>);
    template void tanh<float>(const Stream&, span<float>, view<float>);
    template void tanh<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void sigmoid(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::sigmoid<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void sigmoid<__half>(const Stream&, span<__half>, view<__half>);
    template void sigmoid<float>(const Stream&, span<float>, view<float>);
    template void sigmoid<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void bnll(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::bnll<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void bnll<__half>(const Stream&, span<__half>, view<__half>);
    template void bnll<float>(const Stream&, span<float>, view<float>);
    template void bnll<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void elu(const Stream& stream, span<T> output, view<T> input) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::elu<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input);
    }

    template void elu<__half>(const Stream&, span<__half>, view<__half>);
    template void elu<float>(const Stream&, span<float>, view<float>);
    template void elu<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void relu(const Stream& stream, span<T> output, view<T> input, T slope) {
        CV_Assert(input.size() == output.size());
        if(is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            auto kernel = raw::relu_vec4<T>;
            auto policy = make_policy(kernel, output.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output, input, slope);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            auto kernel = raw::relu_vec2<T>;
            auto policy = make_policy(kernel, output.size() / 2, 0, stream);
            launch_kernel(kernel, policy, output, input, slope);
        } else {
            auto kernel = raw::relu<T>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, slope);
        }
    }

    template void relu<__half>(const Stream&, span<__half>, view<__half>, __half);
    template void relu<float>(const Stream&, span<float>, view<float>, float);
    template void relu<double>(const Stream&, span<double>, view<double>, double);

    template <class T>
    void clipped_relu(const Stream& stream, span<T> output, view<T> input, T floor, T ceiling) {
        CV_Assert(input.size() == output.size());
        CV_Assert(double(floor) <= double(ceiling));

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4)) {
            auto kernel = raw::clipped_relu_vec4<T>;
            auto policy = make_policy(kernel, output.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output, input, floor, ceiling);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2)) {
            auto kernel = raw::clipped_relu_vec2<T>;
            auto policy = make_policy(kernel, output.size() / 2, 0, stream);
            launch_kernel(kernel, policy, output, input, floor, ceiling);
        } else {
            auto kernel = raw::clipped_relu<T>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, floor, ceiling);
        }
    }

    template void clipped_relu<__half>(const Stream&, span<__half>, view<__half>, __half, __half);
    template void clipped_relu<float>(const Stream&, span<float>, view<float>, float, float);
    template void clipped_relu<double>(const Stream&, span<double>, view<double>, double, double);

    template <class T>
    void axiswise_relu(const Stream& stream, span<T> output, view<T> input, view<T> slope, std::size_t inner_size) {
        CV_Assert(input.size() == output.size());

        auto kernel = raw::axiswise_relu<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, slope);
    }

    template void axiswise_relu<__half>(const Stream&, span<__half>, view<__half>, view<__half>, std::size_t);
    template void axiswise_relu<float>(const Stream&, span<float>, view<float>, view<float>, std::size_t);
    template void axiswise_relu<double>(const Stream&, span<double>, view<double>, view<double>, std::size_t);

    template <class T>
    void power(const Stream& stream, span<T> output, view<T> input, T exp, T scale, T shift) {
        CV_Assert(input.size() == output.size());

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && output.size() > 1024 * 16 * 4) {
            auto kernel = raw::power_vec4<T>;
            auto policy = make_policy(kernel, output.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output, input, exp, scale, shift);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && output.size() > 1024 * 16 * 2) {
            auto kernel = raw::power_vec2<T>;
            auto policy = make_policy(kernel, output.size() / 2, 0, stream);
            launch_kernel(kernel, policy, output, input, exp, scale, shift);
        } else {
            auto kernel = raw::power<T>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, exp, scale, shift);
        }
    }

    template void power<__half>(const Stream&, span<__half>, view<__half>, __half, __half, __half);
    template void power<float>(const Stream&, span<float>, view<float>, float, float, float);
    template void power<double>(const Stream&, span<double>, view<double>, double, double, double);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
