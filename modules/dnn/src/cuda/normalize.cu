// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "array.hpp"
#include "math.hpp"
#include "reduce.hpp"
#include "atomics.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T> static
        __global__ void zero(span<T> output) {
            for (auto idx : grid_stride_range(output.size()))
                output[idx] = 0;
        }

        template <class T> static
        __global__ void reduce_sum_abs(span<T> output, view<T> input, std::size_t outer_stride, std::size_t mid_stride)
        {
            for (auto idx : grid_stride_range(input.size())) {
                const auto outer_idx = idx / outer_stride;
                const auto inner_idx = idx % mid_stride;

                auto sum_idx = outer_idx * mid_stride + inner_idx;
                atomicAdd(&output[sum_idx], utils::abs(input[idx]));
            }
        }

        template <class T> static
        __global__ void reciprocal(span<T> output, T epsilon) {
            for (auto idx : grid_stride_range(output.size()))
                output[idx] = 1 / (output[idx] + epsilon);
        }

        template <class T> static
        __global__ void reduce_sum_squared(span<T> output, view<T> input, std::size_t outer_stride, std::size_t mid_stride)
        {
           for (auto idx : grid_stride_range(input.size())) {
                const auto outer_idx = idx / outer_stride;
                const auto inner_idx = idx % mid_stride;

                auto sum_idx = outer_idx * mid_stride + inner_idx;
                atomicAdd(&output[sum_idx], input[idx] * input[idx]);
           }
        }

        template <class T> static
        __global__ void rsqrt(span<T> output, T epsilon) {
            for (auto idx : grid_stride_range(output.size()))
                output[idx] = utils::rsqrt(output[idx] + epsilon);
        }

        template <class T> static
        __global__ void apply_norm(span<T> output, view<T> input, std::size_t outer_stride, std::size_t mid_stride, view<T> sums)
        {
            for (auto idx : grid_stride_range(output.size())) {
                const auto outer_idx = idx / outer_stride;
                const auto inner_idx = idx % mid_stride;

                auto sum_idx = outer_idx * mid_stride + inner_idx;
                output[idx] = input[idx] * sums[sum_idx];
            }
        }
    }

    template <class T>
    void normalize(
        const Stream& stream,
        span<T> output,
        view<T> input, std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, T norm, T epsilon,
        span<T> workspace_)
    {
        CV_Assert(norm == 1 || norm == 2);
        CV_Assert(workspace_.size() >= outer_size * inner_size);

        auto sums = span<T>(workspace_.data(), outer_size * inner_size);

        auto zero_kernel = raw::zero<T>;
        auto policy = make_policy(zero_kernel, 0, stream);
        launch_kernel(zero_kernel, policy, sums);

        if (norm == 1) {
            auto reduce_kernel = raw::reduce_sum_abs<T>;
            policy = make_policy(reduce_kernel, 0, stream);
            launch_kernel(reduce_kernel, policy, sums, input, mid_size * inner_size, inner_size);

            auto reciprocal_kernel = raw::reciprocal<T>;
            policy = make_policy(reciprocal_kernel, 0, stream);
            launch_kernel(reciprocal_kernel, policy, sums, epsilon);
        } else {
            auto reduce_kernel = raw::reduce_sum_squared<T>;
            policy = make_policy(reduce_kernel, 0, stream);
            launch_kernel(reduce_kernel, policy, sums, input, mid_size * inner_size, inner_size);

            auto rsqrt_kernel = raw::rsqrt<T>;
            policy = make_policy(rsqrt_kernel, 0, stream);
            launch_kernel(rsqrt_kernel, policy, sums, epsilon);
        }

        auto scale_kernel = raw::apply_norm<T>;
        policy = make_policy(scale_kernel, 0, stream);
        launch_kernel(scale_kernel, policy, output, input, mid_size * inner_size, inner_size, sums);
    }

    template void normalize<float>(const Stream&, span<float>, view<float>, std::size_t, std::size_t, std::size_t, float, float, span<float>);
    template void normalize<double>(const Stream&, span<double>, view<double>, std::size_t, std::size_t, std::size_t, double, double, span<double>);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
