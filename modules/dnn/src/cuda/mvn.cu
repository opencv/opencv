// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "atomics.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

namespace raw {
    template <class T>
    __global__ void reduce_mean(Span<float> means, View<T> input, size_type inner_size) {
        for (auto idx : grid_stride_range(input.size())) {
            const index_type outer_idx = idx / inner_size;
            atomicAdd(&means[outer_idx], static_cast<float>(input[idx]) / inner_size);
        }
    }

    template <class T>
    __global__ void reduce_mean_sqr_sum(Span<float> means, Span<float> sum_sqrs, View<T> input, size_type inner_size) {
        for (auto idx : grid_stride_range(input.size())) {
            const index_type outer_idx = idx / inner_size;
            auto x = static_cast<float>(input[idx]);
            atomicAdd(&means[outer_idx], x / inner_size);
            atomicAdd(&sum_sqrs[outer_idx], x * x);
        }
    }

    __global__ void compute_normalization_scale(Span<float> scale, View<float> means, View<float> sums_sqr, size_type inner_size, float eps) {
        for (auto idx : grid_stride_range(scale.size())) {
            auto mean = means[idx];
            auto var = sums_sqr[idx] / inner_size - mean * mean;
            using device::rsqrt;
            scale[idx] = rsqrt(eps + var);
        }
    }

    template <class T>
    __global__ void normalize_mean(Span<T> output, View<T> input, View<float> means, size_type inner_size) {
        for (auto idx : grid_stride_range(output.size())) {
            const index_type outer_idx = idx / inner_size;
            output[idx] = static_cast<float>(input[idx]) - means[outer_idx];
        }
    }

    template <class T>
    __global__ void normalize_mean_variance(Span<T> output, View<T> input, View<float> means, View<float> scale, size_type inner_size) {
        for (auto idx : grid_stride_range(output.size())) {
            const index_type outer_idx = idx / inner_size;
            output[idx] = (static_cast<float>(input[idx]) - means[outer_idx]) * scale[outer_idx];
        }
    }

    template <class T>
    __global__ void normalize_mean_variance_channelwise(Span<T> output, View<T> input, View<T> scale, View<T> bias, View<float> means, View<float> inv_stddev, size_type inner_size, size_type C) {
        for (auto idx : grid_stride_range(output.size())) {
            const index_type outer_idx = idx / inner_size;
            const index_type c = outer_idx % C;
            auto s = static_cast<float>(scale[c]) * inv_stddev[outer_idx];
            auto b = static_cast<float>(bias[c]);
            output[idx] = (static_cast<float>(input[idx]) - means[outer_idx]) * s + b;
        }
    }

    template <class T>
    __global__ void normalize_mean_variance_groupwise(Span<T> output, View<T> input, View<T> scale, View<T> bias, View<float> means, View<float> inv_stddev, size_type inner_size, size_type C, size_type num_groups, size_type group_size) {
        for (auto idx : grid_stride_range(output.size())) {
            const index_type outer_idx = idx / inner_size;
            const index_type c = outer_idx % C;
            const index_type group_idx = outer_idx / group_size;
            auto s = static_cast<float>(scale[c]) * inv_stddev[group_idx];
            auto b = static_cast<float>(bias[c]);
            output[idx] = (static_cast<float>(input[idx]) - means[group_idx]) * s + b;
        }
    }

    template <class T>
    __global__ void normalize_mean_variance_layernorm(Span<T> output, View<T> input, View<T> scale, View<float> means, View<float> inv_stddev, size_type inner_size) {
        for (auto idx : grid_stride_range(output.size())) {
            const index_type outer_idx = idx / inner_size;
            const index_type inner_idx = idx % inner_size;
            auto s = static_cast<float>(scale[inner_idx]) * inv_stddev[outer_idx];
            output[idx] = (static_cast<float>(input[idx]) - means[outer_idx]) * s;
        }
    }

    template <class T>
    __global__ void normalize_mean_variance_layernorm_with_bias(Span<T> output, View<T> input, View<T> scale, View<T> bias, View<float> means, View<float> inv_stddev, size_type inner_size) {
        for (auto idx : grid_stride_range(output.size())) {
            const index_type outer_idx = idx / inner_size;
            const index_type inner_idx = idx % inner_size;
            auto s = static_cast<float>(scale[inner_idx]) * inv_stddev[outer_idx];
            auto b = static_cast<float>(bias[inner_idx]);
            output[idx] = (static_cast<float>(input[idx]) - means[outer_idx]) * s + b;
        }
    }
}

template <class T>
void reduce_mean(const Stream& stream, Span<float> means, View<T> input, std::size_t inner_size)
{
    CV_Assert(input.size() / inner_size == means.size());

    auto kernel = raw::reduce_mean<T>;
    auto policy = make_policy(kernel, input.size(), 0, stream);
    launch_kernel(kernel, policy, means, input, inner_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void reduce_mean(const Stream&, Span<float>, View<__half>, std::size_t);
#endif
template void reduce_mean(const Stream&, Span<float>, View<float>, std::size_t);

template <class T>
void reduce_mean_sqr_sum(const Stream& stream, Span<float> means, Span<float> sum_sqrs, View<T> input, std::size_t inner_size)
{
    CV_Assert(input.size() / inner_size == means.size());
    CV_Assert(input.size() / inner_size == sum_sqrs.size());

    auto kernel = raw::reduce_mean_sqr_sum<T>;
    auto policy = make_policy(kernel, input.size(), 0, stream);
    launch_kernel(kernel, policy, means, sum_sqrs, input, inner_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void reduce_mean_sqr_sum(const Stream&, Span<float>, Span<float>, View<__half>, std::size_t);
#endif
template void reduce_mean_sqr_sum(const Stream&, Span<float>, Span<float>, View<float>, std::size_t);

void compute_normalization_scale(const Stream& stream, Span<float> scale, View<float> means, View<float> sum_sqrs, std::size_t inner_size, float eps)
{
    CV_Assert(scale.size() == means.size());
    CV_Assert(scale.size() == sum_sqrs.size());

    auto kernel = raw::compute_normalization_scale;
    auto policy = make_policy(kernel, scale.size(), 0, stream);
    launch_kernel(kernel, policy, scale, means, sum_sqrs, inner_size, eps);
}

template <class T>
void normalize_mean(const Stream& stream, Span<T> output, View<T> input, View<float> means, std::size_t inner_size)
{
    CV_Assert(output.size() == input.size());
    CV_Assert(input.size() / inner_size == means.size());

    auto kernel = raw::normalize_mean<T>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, input, means, inner_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void normalize_mean(const Stream&, Span<__half>, View<__half>, View<float>, std::size_t);
#endif
template void normalize_mean(const Stream&, Span<float>, View<float>, View<float>, std::size_t);

template <class T>
void normalize_mean_variance(const Stream& stream, Span<T> output, View<T> input, View<float> means, View<float> scale, std::size_t inner_size)
{
    CV_Assert(input.size() == output.size());
    CV_Assert(input.size() / inner_size == means.size());
    CV_Assert(input.size() / inner_size == scale.size());

    auto kernel = raw::normalize_mean_variance<T>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, input, means, scale, inner_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void normalize_mean_variance(const Stream&, Span<__half>, View<__half>, View<float>, View<float>, std::size_t);
#endif
template void normalize_mean_variance(const Stream&, Span<float>, View<float>, View<float>, View<float>, std::size_t);

template <class T>
void normalize_mean_variance_channelwise(const Stream& stream, Span<T> output, View<T> input, View<T> scale, View<T> bias, View<float> means, View<float> inv_stddev, std::size_t inner_size, std::size_t C)
{
    CV_Assert(input.size() == output.size());
    CV_Assert(input.size() / inner_size == means.size());
    CV_Assert(means.size() == inv_stddev.size());

    auto kernel = raw::normalize_mean_variance_channelwise<T>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, input, scale, bias, means, inv_stddev, inner_size, C);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void normalize_mean_variance_channelwise(const Stream&, Span<__half> /*output*/, View<__half> /*input*/, View<__half> /*scale*/, View<__half> /*bias*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t, std::size_t);
#endif
template void normalize_mean_variance_channelwise(const Stream&, Span<float> /*output*/, View<float> /*input*/, View<float> /*scale*/, View<float> /*bias*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t, std::size_t);

template <class T>
void normalize_mean_variance_groupwise(const Stream& stream, Span<T> output, View<T> input, View<T> scale, View<T> bias, View<float> means, View<float> inv_stddev, std::size_t inner_size, std::size_t C, std::size_t num_groups, std::size_t group_size)
{
    CV_Assert(input.size() == output.size());
    CV_Assert(input.size() / inner_size == means.size() * group_size);
    CV_Assert(means.size() == inv_stddev.size());

    auto kernel = raw::normalize_mean_variance_groupwise<T>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, input, scale, bias, means, inv_stddev, inner_size, C, num_groups, group_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void normalize_mean_variance_groupwise(const Stream&, Span<__half> /*output*/, View<__half> /*input*/, View<__half> /*scale*/, View<__half> /*bias*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t, std::size_t, std::size_t, std::size_t);
#endif
template void normalize_mean_variance_groupwise(const Stream&, Span<float> /*output*/, View<float> /*input*/, View<float> /*scale*/, View<float> /*bias*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t, std::size_t, std::size_t, std::size_t);


template <class T>
void normalize_mean_variance_layernorm(const Stream& stream, Span<T> output, View<T> input, View<T> scale, View<float> means, View<float> inv_stddev, std::size_t inner_size)
{
    CV_Assert(input.size() == output.size());
    CV_Assert(input.size() / inner_size == means.size());
    CV_Assert(means.size() == inv_stddev.size());

    auto kernel = raw::normalize_mean_variance_layernorm<T>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, input, scale, means, inv_stddev, inner_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void normalize_mean_variance_layernorm(const Stream&, Span<__half> /*output*/, View<__half> /*input*/, View<__half> /*scale*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t);
#endif
template void normalize_mean_variance_layernorm(const Stream&, Span<float> /*output*/, View<float> /*input*/, View<float> /*scale*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t);

template <class T>
void normalize_mean_variance_layernorm(const Stream& stream, Span<T> output, View<T> input, View<T> scale, View<T> bias, View<float> means, View<float> inv_stddev, std::size_t inner_size)
{
    CV_Assert(input.size() == output.size());
    CV_Assert(input.size() / inner_size == means.size());
    CV_Assert(means.size() == inv_stddev.size());

    auto kernel = raw::normalize_mean_variance_layernorm_with_bias<T>;
    auto policy = make_policy(kernel, output.size(), 0, stream);
    launch_kernel(kernel, policy, output, input, scale, bias, means, inv_stddev, inner_size);
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template void normalize_mean_variance_layernorm(const Stream&, Span<__half> /*output*/, View<__half> /*input*/, View<__half> /*scale*/, View<__half> /*bias*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t);
#endif
template void normalize_mean_variance_layernorm(const Stream&, Span<float> /*output*/, View<float> /*input*/, View<float> /*scale*/, View<float> /*bias*/, View<float> /*means*/, View<float> /*inv_stddev*/, std::size_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
