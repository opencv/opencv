// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/pointer.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void scale(
            std::size_t n,
            DevicePtr<T> output,
            DevicePtr<const T> input, std::size_t inner_size,
            DevicePtr<const T> weights, std::size_t scale_size)
        {
            for (auto i : grid_stride_range(n)) {
                const auto scale_idx = (i / inner_size) % scale_size;
                output[i] = input[i] * weights[scale_idx];
            }
        }

        template <class T>
        __global__ void scale_with_bias(
            std::size_t n,
            DevicePtr<T> output,
            DevicePtr<const T> input, std::size_t inner_size,
            DevicePtr<const T> weights, DevicePtr<const T> bias, std::size_t scale_bias_size)
        {
            for (auto i : grid_stride_range(n)) {
                const auto scale_idx = (i / inner_size) % scale_bias_size;
                output[i] = input[i] * weights[scale_idx] + bias[scale_idx];
            }
        }
    }

    template <class T>
    void scale(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights)
    {
        CV_Assert(is_shape_same(input, output));

        auto policy = make_policy(raw::scale<T>, 0, stream);
        launch_kernel(raw::scale<T>, policy,
            output.size(),
            output.get(),
            input.get(), inner_size,
            weights.get(), weights.size());
    }

    template void scale<float>(
        const Stream& stream,
        TensorSpan<float> output,
        TensorView<float> input, std::size_t inner_size,
        TensorView<float> weights);

    template void scale<double>(
        const Stream& stream,
        TensorSpan<double> output,
        TensorView<double> input, std::size_t inner_size,
        TensorView<double> weights);

    template <class T>
    void scale_with_bias(
        const Stream& stream,
        TensorSpan<T> output,
        TensorView<T> input, std::size_t inner_size,
        TensorView<T> weights, TensorView<T> bias)
    {
        CV_Assert(is_shape_same(input, output));
        CV_Assert(weights.size() == bias.size());

        auto policy = make_policy(raw::scale_with_bias<T>, 0, stream);
        launch_kernel(raw::scale_with_bias<T>, policy,
            output.size(),
            output.get(),
            input.get(), inner_size,
            weights.get(), bias.get(), weights.size());
    }

    template void scale_with_bias<float>(
        const Stream& stream,
        TensorSpan<float> output,
        TensorView<float> input, std::size_t inner_size,
        TensorView<float> weights, TensorView<float> bias);

    template void scale_with_bias<double>(
        const Stream& stream,
        TensorSpan<double> output,
        TensorView<double> input, std::size_t inner_size,
        TensorView<double> weights, TensorView<double> bias);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
