// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "math.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void abs(span<T> dest, view<T> src) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::abs;
                dest[i] = abs(src[i]);
            }
        }

        template <class T>
        __global__ void tanh(span<T> dest, view<T> src) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::tanh;
                dest[i] = tanh(src[i]);
            }
        }

        template <class T>
        __global__ void sigmoid(span<T> dest, view<T> src) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::sigmoid;
                dest[i] = sigmoid(src[i]);
            }
        }

        template <class T>
        __global__ void bnll(span<T> dest, view<T> src) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::log1pexp;
                dest[i] = src[i] > 0 ? src[i] + log1pexp(-src[i]) : log1pexp(src[i]);
            }
        }

        template <class T>
        __global__ void elu(span<T> dest, view<T> src) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::exp;
                dest[i] = src[i] >= 0 ? src[i] : (exp(src[i]) - 1);
            }
        }

        template <class T>
        __global__ void relu(span<T> dest, view<T> src, T slope) {
            for (auto i : grid_stride_range(dest.size()))
                dest[i] = src[i] >= 0.0 ? src[i] : slope * src[i];
        }

        template <class T>
        __global__ void clipped_relu(span<T> dest, view<T> src, T floor, T ceiling) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::max;
                using utils::min;
                dest[i] = min(max(src[i], floor), ceiling);
            }
        }

        template <class T>
        __global__ void axiswise_relu(span<T> dest, view<T> src, std::size_t inner_size, view<T> slope) {
            for (auto i : grid_stride_range(dest.size())) {
                const auto c = (i % inner_size) / slope.size();
                dest[i] = src[i] < 0 ? src[i] * slope[c] : src[i];
            }
        }

        template <class T>
        __global__ void power(span<T> dest, view<T> src, T exp, T scale, T shift) {
            for (auto i : grid_stride_range(dest.size())) {
                using utils::pow;
                dest[i] = pow(shift + scale * src[i], exp);
            }
        }
    }

    template <class T>
    void abs(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::abs<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src);
    }

    template void abs<float>(const Stream& stream, span<float> dest, view<float> src);
    template void abs<double>(const Stream& stream, span<double> dest, view<double> src);

    template <class T>
    void tanh(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::tanh<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src);
    }

    template void tanh<float>(const Stream&, span<float>, view<float>);
    template void tanh<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void sigmoid(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::sigmoid<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src);
    }

    template void sigmoid<float>(const Stream&, span<float>, view<float>);
    template void sigmoid<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void bnll(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::bnll<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src);
    }

    template void bnll<float>(const Stream&, span<float>, view<float>);
    template void bnll<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void elu(const Stream& stream, span<T> dest, view<T> src) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::elu<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src);
    }

    template void elu<float>(const Stream&, span<float>, view<float>);
    template void elu<double>(const Stream&, span<double>, view<double>);

    template <class T>
    void relu(const Stream& stream, span<T> dest, view<T> src, T slope) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::relu<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src, slope);
    }

    template void relu<float>(const Stream&, span<float>, view<float>, float);
    template void relu<double>(const Stream&, span<double>, view<double>, double);

    template <class T>
    void clipped_relu(const Stream& stream, span<T> dest, view<T> src, T floor, T ceiling) {
        CV_Assert(src.size() >= dest.size());
        CV_Assert(floor <= ceiling);

        auto kernel = raw::clipped_relu<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src, floor, ceiling);
    }

    template void clipped_relu<float>(const Stream&, span<float>, view<float>, float, float);
    template void clipped_relu<double>(const Stream&, span<double>, view<double>, double, double);

    template <class T>
    void axiswise_relu(const Stream& stream, span<T> dest, view<T> src, view<T> slope, std::size_t inner_size) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::axiswise_relu<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src, inner_size, slope);
    }

    template void axiswise_relu<float>(const Stream&, span<float>, view<float>, view<float>, std::size_t);
    template void axiswise_relu<double>(const Stream&, span<double>, view<double>, view<double>, std::size_t);

    template <class T>
    void power(const Stream& stream, span<T> dest, view<T> src, T exp, T scale, T shift) {
        CV_Assert(src.size() >= dest.size());

        auto kernel = raw::power<T>;
        auto policy = make_policy(kernel, dest.size(), 0, stream);
        launch_kernel(kernel, policy, dest, src, exp, scale, shift);
    }

    template void power<float>(const Stream&, span<float>, view<float>, float, float, float);
    template void power<double>(const Stream&, span<double>, view<double>, double, double, double);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
