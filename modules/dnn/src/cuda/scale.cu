// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        using index_type = gpu::index_type;
        using size_type = gpu::size_type;

        template <class T>
        __global__ void bias1(span<T> output, view<T> input, T beta) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = input[i] + beta;
        }

        template <class T>
        __global__ void biasN_vec4(span<T> output, view<T> input, size_type inner_size, view<T> bias) {
            using vector_type = typename get_vector_type<T, 4>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            inner_size /= 4;
            for (auto i : grid_stride_range(output.size() / 4)) {
                const index_type bias_idx = (i / inner_size) % bias.size();

                vector_type vec = srcPtr[i];
                vec.w = vec.w + bias[bias_idx];
                vec.x = vec.x + bias[bias_idx];
                vec.y = vec.y + bias[bias_idx];
                vec.z = vec.z + bias[bias_idx];
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void biasN_vec2(span<T> output, view<T> input, size_type inner_size, view<T> bias)
        {
            using vector_type = typename get_vector_type<T, 2>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            inner_size /= 2;
            for (auto i : grid_stride_range(output.size() / 2)) {
                const index_type bias_idx = (i / inner_size) % bias.size();

                vector_type vec = srcPtr[i];
                vec.x = vec.x + bias[bias_idx];
                vec.y = vec.y + bias[bias_idx];
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void biasN(span<T> output, view<T> input, size_type inner_size, view<T> bias)
        {
            for (auto i : grid_stride_range(output.size())) {
                const index_type bias_idx = (i / inner_size) % bias.size();
                output[i] = input[i] + bias[bias_idx];
            }
        }

        template <class T>
        __global__ void scale1(span<T> output, view<T> input, T alpha)
        {
            for (auto i : grid_stride_range(output.size()))
                output[i] = alpha * input[i];
        }

        template <class T>
        __global__ void scaleN_vec4(span<T> output, view<T> input, size_type inner_size, view<T> weights)
        {
            using vector_type = typename get_vector_type<T, 4>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            inner_size /= 4;
            for (auto i : grid_stride_range(output.size() / 4)) {
                const index_type scale_idx = (i / inner_size) % weights.size();

                vector_type vec = srcPtr[i];
                vec.w = vec.w * weights[scale_idx];
                vec.x = vec.x * weights[scale_idx];
                vec.y = vec.y * weights[scale_idx];
                vec.z = vec.z * weights[scale_idx];
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void scaleN_vec2(span<T> output, view<T> input, size_type inner_size, view<T> weights)
        {
            using vector_type = typename get_vector_type<T, 2>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            inner_size /= 2;
            for (auto i : grid_stride_range(output.size() / 2)) {
                const index_type scale_idx = (i / inner_size) % weights.size();

                vector_type vec = srcPtr[i];
                vec.x = vec.x * weights[scale_idx];
                vec.y = vec.y * weights[scale_idx];
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void scaleN(span<T> output, view<T> input, size_type inner_size, view<T> weights)
        {
            for (auto i : grid_stride_range(output.size())) {
                const index_type scale_idx = (i / inner_size) % weights.size();
                output[i] = input[i] * weights[scale_idx];
            }
        }

        template <class T>
        __global__ void scale1_with_bias1(span<T> output, view<T> input, T alpha, T beta)
        {
            for (auto i : grid_stride_range(output.size()))
                output[i] = alpha * input[i] + beta;
        }

        template <class T>
        __global__ void scaleN_with_biasN_vec4(span<T> output, view<T> input, size_type inner_size, view<T> weights, view<T> bias)
        {
            using vector_type = typename get_vector_type<T, 4>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            inner_size /= 4;
            for (auto i : grid_stride_range(output.size() / 4)) {
                const index_type scale_idx = (i / inner_size) % weights.size();

                vector_type vec = srcPtr[i];
                vec.w = vec.w * weights[scale_idx] + bias[scale_idx];
                vec.x = vec.x * weights[scale_idx] + bias[scale_idx];
                vec.y = vec.y * weights[scale_idx] + bias[scale_idx];
                vec.z = vec.z * weights[scale_idx] + bias[scale_idx];
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void scaleN_with_biasN_vec2(span<T> output, view<T> input, size_type inner_size, view<T> weights, view<T> bias)
        {
            using vector_type = typename get_vector_type<T, 2>::type;

            vector_type* dstPtr = reinterpret_cast<vector_type*>(output.data().get());
            const vector_type* srcPtr = reinterpret_cast<const vector_type*>(input.data().get());

            inner_size /= 2;
            for (auto i : grid_stride_range(output.size() / 2)) {
                const index_type scale_idx = (i / inner_size) % weights.size();

                vector_type vec = srcPtr[i];
                vec.x = vec.x * weights[scale_idx] + bias[scale_idx];
                vec.y = vec.y * weights[scale_idx] + bias[scale_idx];
                dstPtr[i] = vec;
            }
        }

        template <class T>
        __global__ void scaleN_with_biasN(span<T> output, view<T> input, size_type inner_size, view<T> weights, view<T> bias)
        {
            for (auto i : grid_stride_range(output.size())) {
                const index_type scale_idx = (i / inner_size) % weights.size();
                output[i] = input[i] * weights[scale_idx] + bias[scale_idx];
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

    template <class T>
    void biasN(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> bias)
    {
        CV_Assert(is_shape_same(input, output));

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            auto kernel = raw::biasN_vec4<T>;
            auto policy = make_policy(kernel, output.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, bias);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            auto kernel = raw::biasN_vec2<T>;
            auto policy = make_policy(kernel, output.size() / 2, 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, bias);
        } else {
            auto kernel = raw::biasN<T>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, bias);
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

    template <class T>
    void scaleN(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights)
    {
        CV_Assert(is_shape_same(input, output));

        if (is_fully_aligned<T>(output, 4) && is_fully_aligned<T>(input, 4) && inner_size % 4 == 0) {
            auto kernel = raw::scaleN_vec4<T>;
            auto policy = make_policy(kernel, output.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, weights);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            auto kernel = raw::scaleN_vec2<T>;
            auto policy = make_policy(kernel, output.size() / 2, 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, weights);
        } else {
            auto kernel = raw::scaleN<T>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, weights);
        }
    }

    template void scaleN<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, std::size_t, TensorView<__half>);
    template void scaleN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>);
    template void scaleN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>);

    template <class T>
    void scale1_with_bias1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T alpha, T beta) {
        CV_Assert(is_shape_same(input, output));

        auto kernel = raw::scale1_with_bias1<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, alpha, beta);
    }

    template void scale1_with_bias1<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, __half, __half);
    template void scale1_with_bias1<float>(const Stream&, TensorSpan<float>, TensorView<float>, float, float);
    template void scale1_with_bias1<double>(const Stream&, TensorSpan<double>, TensorView<double>, double, double);

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
            auto kernel = raw::scaleN_with_biasN_vec4<T>;
            auto policy = make_policy(kernel, output.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, weights, bias);
        } else if (is_fully_aligned<T>(output, 2) && is_fully_aligned<T>(input, 2) && inner_size % 2 == 0) {
            auto kernel = raw::scaleN_with_biasN_vec2<T>;
            auto policy = make_policy(kernel, output.size() / 2, 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, weights, bias);
        } else {
            auto kernel = raw::scaleN_with_biasN<T>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, inner_size, weights, bias);
        }
    }

    template void scaleN_with_biasN<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, std::size_t, TensorView<__half>, TensorView<__half>);
    template void scaleN_with_biasN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>, TensorView<float>);
    template void scaleN_with_biasN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>, TensorView<double>);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
