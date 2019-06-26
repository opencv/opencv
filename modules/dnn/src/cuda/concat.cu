// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/pointer.hpp"
#include "../cuda4dnn/csl/stream.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        /* Reference: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cu */
        template <class T>
        __global__ void concat(
            DevicePtr<T> output, DevicePtr<const T> input,
            std::size_t concat_size, std::size_t input_concat_axis_size,
            std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis,
            std::size_t n)
        {
            for (auto idx : grid_stride_range(n)) {
                const auto total_concat_size = concat_size * input_concat_axis_size;
                const auto concat_num = idx / total_concat_size;
                const auto concat_index = idx % total_concat_size;
                const auto top_index = concat_index +
                    (concat_num * output_concat_axis_size + output_offset_concat_axis) * concat_size;

                output[top_index] = input[idx];
            }
        }
    }

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis)
    {
        auto policy = make_policy(raw::concat<T>, 0, stream);
        launch_kernel(raw::concat<T>, policy,
            output.get(), input.get(),
            concat_size, input_concat_axis_size,
            output_concat_axis_size, output_offset_concat_axis,
            input.size());
    }

    template void concat<float>(
        const Stream& stream,
        TensorSpan<float> output, TensorView<float> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis);

    template void concat<double>(
        const Stream& stream,
        TensorSpan<double> output, TensorView<double> input,
        std::size_t concat_size, std::size_t input_concat_axis_size,
        std::size_t output_concat_axis_size, std::size_t output_offset_concat_axis);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
