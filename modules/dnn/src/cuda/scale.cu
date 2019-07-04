// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/pointer.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void bias1(span<T> output, view<T> input, T beta)
        {
            for (auto i : grid_stride_range(output.size()))
                output[i] = input[i] + beta;
        }

        template <class T>
        __global__ void biasN(span<T> output, view<T> input, std::size_t inner_size, view<T> bias)
        {
            for (auto i : grid_stride_range(output.size())) {
                const auto bias_idx = (i / inner_size) % bias.size();
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
        __global__ void scaleN(span<T> output, view<T> input, std::size_t inner_size, view<T> weights)
        {
            for (auto i : grid_stride_range(output.size())) {
                const auto scale_idx = (i / inner_size) % weights.size();
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
        __global__ void scaleN_with_biasN(span<T> output, view<T> input, std::size_t inner_size, view<T> weights, view<T> bias)
        {
            for (auto i : grid_stride_range(output.size())) {
                const auto scale_idx = (i / inner_size) % weights.size();
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

        auto kernel = raw::biasN<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, bias);
    }

    template void biasN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>);
    template void biasN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>);

    template <class T>
    void scale1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T alpha) {
        CV_Assert(is_shape_same(input, output));

        auto kernel = raw::scale1<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, alpha);
    }

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

        auto kernel = raw::scaleN<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, weights);
    }

    template void scaleN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>);
    template void scaleN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>);

    template <class T>
    void scale1_with_bias1(const Stream& stream, TensorSpan<T> output, TensorView<T> input, T alpha, T beta) {
        CV_Assert(is_shape_same(input, output));

        auto kernel = raw::scale1_with_bias1<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, alpha, beta);
    }

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

        auto kernel = raw::scaleN_with_biasN<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, inner_size, weights, bias);
    }

    template void scaleN_with_biasN<float>(const Stream&, TensorSpan<float>, TensorView<float>, std::size_t, TensorView<float>, TensorView<float>);
    template void scaleN_with_biasN<double>(const Stream&, TensorSpan<double>, TensorView<double>, std::size_t, TensorView<double>, TensorView<double>);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
