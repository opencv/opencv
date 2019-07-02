// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "array.hpp"
#include "math.hpp"
#include "reduce.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void reduce_sum_powN(span<T> output,
            view<T> input, std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, T norm)
        {
            for (int i = 0; i < outer_size; i++) {
                for (int j = 0; j < mid_size; j++) {
                    const auto outer_offset = i * mid_size * inner_size;
                    const auto mid_offset = j * mid_size;
                    const auto total_offset = outer_offset + mid_offset;

                    T thread_sum = 0;
                    for (auto idx : grid_stride_range(inner_size)) {
                        const auto full_idx = total_offset + idx;
                        thread_sum += utils::pow<T>(utils::abs(input[full_idx]), norm);
                    }

                    auto warp_sum = utils::warpReduceSum(thread_sum);
                    if ((threadIdx.x & (warpSize - 1)) == 0)
                        atomicAdd(&output[total_offset], warp_sum);
                }
            }
        }

        template <class T>
        __global__ void scale_inverse_powN(span<T> output,
            view<T> input, std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, T episilon, T norm,
            view<T> sums)
        {
            for (int i = 0; i < outer_size; i++) {
                for (int j = 0; j < mid_size; j++) {
                    const auto outer_offset = i * mid_size * inner_size;
                    const auto mid_offset = j * mid_size;
                    const auto total_offset = outer_offset + mid_offset;

                    const auto scale = 1 / utils::pow(sums[total_offset] + episilon, 1 / norm);
                    for (auto idx : grid_stride_range(inner_size)) {
                        const auto full_idx = total_offset + idx;
                        output[full_idx] = input[full_idx] * scale;
                    }
                }
            }
        }

        template <class T>
        __global__ void reduce_sum_powN_inner1(span<T> output, view<T> input, std::size_t outer_size, std::size_t mid_size, T pnorm)
        {
            for (int i = 0; i < outer_size; i++) {
                const auto outer_offset = i * mid_size;

                T thread_sum = 0;
                for (auto idx : grid_stride_range(mid_size)) {
                    const auto full_idx = outer_offset + idx;
                    thread_sum += utils::pow<T>(input[full_idx], pnorm);
                }

                auto warp_sum = utils::warpReduceSum(thread_sum);
                if ((threadIdx.x & (warpSize - 1)) == 0)
                    atomicAdd(&output[i], warp_sum);
            }
        }

        template <class T>
        __global__ void scale_inverse_powN_inner1(span<T> output, view<T> input, std::size_t outer_size, std::size_t mid_size, T epsilon, T pnorm,
            view<T> sums)
        {
            for (int i = 0; i < outer_size; i++) {
                const auto outer_offset = i * mid_size;
                const auto scale = 1 / utils::pow<T>(sums[i] + epsilon, 1/pnorm);
                for (auto idx : grid_stride_range(mid_size)) {
                    const auto full_idx = outer_offset + idx;
                    output[full_idx] = input[full_idx] * scale;
                }
            }
        }

        template <class T>
        __global__ void reduce_sum_pow2_inner1(span<T> output, view<T> input, std::size_t outer_size, std::size_t mid_size)
        {
            for (int i = 0; i < outer_size; i++) {
                const auto outer_offset = i * mid_size;

                T thread_sum = 0;
                for (auto idx : grid_stride_range(mid_size)) {
                    const auto full_idx = outer_offset + idx;
                    thread_sum += input[full_idx] * input[full_idx];
                }

                auto warp_sum = utils::warpReduceSum(thread_sum);
                if ((threadIdx.x & (warpSize - 1)) == 0)
                    atomicAdd(&output[i], warp_sum);
            }
        }

        template <class T>
        __global__ void scale_inverse_pow2_inner1(span<T> output, view<T> input, std::size_t outer_size, std::size_t mid_size, T epsilon,
            view<T> sums)
        {
            for (int i = 0; i < outer_size; i++) {
                const auto outer_offset = i * mid_size;
                const auto scale = 1 / utils::sqrt(sums[i] + epsilon);
                for (auto idx : grid_stride_range(mid_size)) {
                    const auto full_idx = outer_offset + idx;
                    output[full_idx] = input[full_idx] * scale;
                }
            }
        }

        template <class T>
        __global__ void reduce_sum_pow1_inner1(span<T> output, view<T> input, std::size_t outer_size, std::size_t mid_size)
        {
            for (int i = 0; i < outer_size; i++) {
                const auto outer_offset = i * mid_size;

                T thread_sum = 0;
                for (auto idx : grid_stride_range(mid_size)) {
                    const auto full_idx = outer_offset + idx;
                    thread_sum += utils::abs(input[full_idx]);
                }

                auto warp_sum = utils::warpReduceSum(thread_sum);
                if ((threadIdx.x & (warpSize - 1)) == 0)
                    atomicAdd(&output[i], warp_sum);
            }
        }

        template <class T>
        __global__ void scale_inverse_pow1_inner1(span<T> output, view<T> input, std::size_t outer_size, std::size_t mid_size, T epsilon,
            view<T> sums)
        {
            for (int i = 0; i < outer_size; i++) {
                 const auto outer_offset = i * mid_size;
                 const auto scale = 1/(sums[i] + epsilon);
                 for (auto idx : grid_stride_range(mid_size)) {
                     const auto full_idx = outer_offset + idx;
                     output[full_idx] = input[full_idx] * scale;
                 }
            }
        }
    }

    template <class T>
    void normalize(
        const Stream& stream,
        span<T> output,
        view<T> input, std::size_t outer_size, std::size_t mid_size, std::size_t inner_size, T norm, T epsilon,
        span<T> workspace)
    {
        if (inner_size == 1) {
            CV_Assert(workspace.size() >= outer_size);
            if (norm == 1) {
                auto reduce_kernel = raw::reduce_sum_pow1_inner1<T>;
                auto policy = make_policy(reduce_kernel, 0, stream);
                launch_kernel(reduce_kernel, policy, workspace, input, outer_size, mid_size);

                auto scale_kernel = raw::scale_inverse_pow1_inner1<T>;
                policy = make_policy(scale_kernel, 0, stream);
                launch_kernel(scale_kernel, policy, output, input, outer_size, mid_size, epsilon, workspace);
            } else if (norm == 2) {
                auto reduce_kernel = raw::reduce_sum_pow2_inner1<T>;
                auto policy = make_policy(reduce_kernel, 0, stream);
                launch_kernel(reduce_kernel, policy, workspace, input, outer_size, mid_size);

                auto scale_kernel = raw::scale_inverse_pow2_inner1<T>;
                policy = make_policy(scale_kernel, 0, stream);
                launch_kernel(scale_kernel, policy, output, input, outer_size, mid_size, epsilon, workspace);
            } else {
                auto reduce_kernel = raw::reduce_sum_powN_inner1<T>;
                auto policy = make_policy(reduce_kernel, 0, stream);
                launch_kernel(reduce_kernel, policy, workspace, input, outer_size, mid_size, norm);

                auto scale_kernel = raw::scale_inverse_powN_inner1<T>;
                policy = make_policy(scale_kernel, 0, stream);
                launch_kernel(scale_kernel, policy, output, input, outer_size, mid_size, epsilon, norm, workspace);
            }
        } else {
            auto reduce_kernel = raw::reduce_sum_powN<T>;
            auto policy = make_policy(reduce_kernel, 0, stream);
            launch_kernel(reduce_kernel, policy, workspace, input, outer_size, mid_size, inner_size, norm);

            auto scale_kernel = raw::scale_inverse_powN<T>;
            policy = make_policy(scale_kernel, 0, stream);
            launch_kernel(scale_kernel, policy, output, input, outer_size, mid_size, inner_size, epsilon, norm, workspace);
        }
    }

    template void normalize<float>(const Stream&, span<float>, view<float>, std::size_t, std::size_t, std::size_t, float, float, span<float>);
    /* double variant not available due to efficient atomicAdd implementation */
    //template void normalize<double>(const Stream&, span<double>, view<double>, std::size_t, std::size_t, std::size_t, unsigned, span<double>);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
