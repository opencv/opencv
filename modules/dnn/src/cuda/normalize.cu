// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "math.hpp"
#include "types.hpp"
#include "atomics.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/fill.hpp"
#include "../cuda4dnn/kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void reduce_sum_abs(Span<T> output, View<T> input, size_type outer_stride, size_type mid_stride) {
            for (auto idx : grid_stride_range(input.size())) {
                const index_type outer_idx = idx / outer_stride;
                const index_type inner_idx = idx % mid_stride;

                const index_type sum_idx = outer_idx * mid_stride + inner_idx;
                atomicAdd(&output[sum_idx], device::abs(input[idx]));
            }
        }

        template <class T>
        __global__ void reciprocal(Span<T> output, T epsilon) {
            for (auto idx : grid_stride_range(output.size()))
                output[idx] = T(1) / (output[idx] + epsilon);
        }

        template <class T>
        __global__ void reduce_sum_squared(Span<T> output, View<T> input, size_type outer_stride, size_type mid_stride) {
           for (auto idx : grid_stride_range(input.size())) {
                const index_type outer_idx = idx / outer_stride;
                const index_type inner_idx = idx % mid_stride;

                const index_type sum_idx = outer_idx * mid_stride + inner_idx;
                atomicAdd(&output[sum_idx], input[idx] * input[idx]);
           }
        }

        template <class T>
        __global__ void rsqrt(Span<T> output, T epsilon) {
            for (auto idx : grid_stride_range(output.size())) {
                using device::sqrt;
                output[idx] = T(1) / sqrt(output[idx] + epsilon);
            }
        }

        template <class T>
        __global__ void apply_norm(Span<T> output, View<T> input, size_type outer_stride, size_type mid_stride, View<T> sums) {
            for (auto idx : grid_stride_range(output.size())) {
                const index_type outer_idx = idx / outer_stride;
                const index_type inner_idx = idx % mid_stride;

                const index_type sum_idx = outer_idx * mid_stride + inner_idx;
                output[idx] = input[idx] * sums[sum_idx];
            }
        }
    }

    template <class T>
    void normalize(
        const Stream& stream,
        Span<T> output,
        View<T> input, std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, std::size_t norm, T epsilon,
        Span<T> workspace)
    {
        CV_Assert(output.size() == input.size());
        CV_Assert(output.size() == outer_size * mid_size * inner_size);
        CV_Assert(norm == 1 || norm == 2);
        CV_Assert(workspace.size() >= outer_size * inner_size);

        auto sums = Span<T>(workspace.data(), outer_size * inner_size);

        fill<T>(stream, sums, 0.0);

        if (norm == 1) {
            auto reduce_kernel = raw::reduce_sum_abs<T>;
            auto policy = make_policy(reduce_kernel, input.size(), 0, stream);
            launch_kernel(reduce_kernel, policy, sums, input, mid_size * inner_size, inner_size);

            auto reciprocal_kernel = raw::reciprocal<T>;
            policy = make_policy(reciprocal_kernel, sums.size(), 0, stream);
            launch_kernel(reciprocal_kernel, policy, sums, epsilon);
        } else {
            auto reduce_kernel = raw::reduce_sum_squared<T>;
            auto policy = make_policy(reduce_kernel, input.size(), 0, stream);
            launch_kernel(reduce_kernel, policy, sums, input, mid_size * inner_size, inner_size);

            auto rsqrt_kernel = raw::rsqrt<T>;
            policy = make_policy(rsqrt_kernel, sums.size(), 0, stream);
            launch_kernel(rsqrt_kernel, policy, sums, epsilon);
        }

        auto scale_kernel = raw::apply_norm<T>;
        auto policy = make_policy(scale_kernel, output.size(), 0, stream);
        launch_kernel(scale_kernel, policy, output, input, mid_size * inner_size, inner_size, sums);
    }

    template void normalize(const Stream&, Span<__half>, View<__half>, std::size_t, std::size_t, std::size_t, std::size_t, __half, Span<__half>);
    template void normalize(const Stream&, Span<float>, View<float>, std::size_t, std::size_t, std::size_t, std::size_t, float, Span<float>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
