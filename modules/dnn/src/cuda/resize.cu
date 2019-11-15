// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <cuda_runtime.h>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void resize_nn(
            Span<T> output, size_type out_height, size_type out_width,
            View<T> input, size_type in_height, size_type in_width)
        {
            auto in_image_size = in_height * in_width;
            auto out_image_size = out_height * out_width;

            /* o2i = output to input */
            auto o2i_fx = static_cast<float>(in_width) / out_width;
            auto o2i_fy = static_cast<float>(in_height) / out_height;

            /* think of the output and input as a collection of 2d images with the last axis
             * representing the width and the last but one axis representing the height
             *
             * the remaining axis together form a collection of these images
             */
            for (auto idx : grid_stride_range(output.size())) {
                const index_type n = idx / out_image_size;
                const index_type x = (idx % out_image_size) % out_width;
                const index_type y = (idx % out_image_size) / out_width;

                auto in_x = static_cast<index_type>(x * o2i_fx);
                auto in_y = static_cast<index_type>(y * o2i_fy);

                index_type in_idx = n * in_image_size + in_y * in_width + in_x;
                output[idx] = input[in_idx];
            }
        }

        template <class T>
        __global__ void resize_bilinear(
            Span<T> output, size_type out_height, size_type out_width,
            View<T> input, size_type in_height, size_type in_width,
            float o2i_fy, float o2i_fx)
        {
            auto in_image_size = in_height * in_width;
            auto out_image_size = out_height * out_width;

            /* think of the output and input as a collection of 2d images with the last axis
             * representing the width and the last but one axis representing the height
             *
             * the remaining axis together form a collection of these images
             */
            for (auto idx : grid_stride_range(output.size())) {
                const index_type n = idx / out_image_size;
                const index_type x = (idx % out_image_size) % out_width;
                const index_type y = (idx % out_image_size) / out_width;

                auto in_x = x * o2i_fx;
                auto in_y = y * o2i_fy;

                auto in_x0 = static_cast<index_type>(in_x);
                auto in_y0 = static_cast<index_type>(in_y);

                using device::min;
                auto in_x1 = min<index_type>(in_x0 + 1, in_width - 1);
                auto in_y1 = min<index_type>(in_y0 + 1, in_height - 1);

                const index_type in_offset_r0 = n * in_image_size + in_y0 * in_width;
                const index_type in_offset_r1 = n * in_image_size + in_y1 * in_width;

                auto v_00 = input[in_offset_r0 + in_x0],
                     v_01 = input[in_offset_r0 + in_x1],
                     v_10 = input[in_offset_r1 + in_x0],
                     v_11 = input[in_offset_r1 + in_x1];

                output[idx] =
                    v_00 +
                    T(in_y - in_y0) * T(v_10 - v_00) +
                    T(in_x - in_x0) * T(v_01 - v_00) +
                    T(in_y - in_y0) * T(in_x - in_x0) * T(v_11 - v_01 - v_10 + v_00);
            }
        }
    }

    template <class T>
    void resize_nn(const Stream& stream, TensorSpan<T> output, TensorView<T> input) {
        auto in_height = input.get_axis_size(-2);
        auto in_width = input.get_axis_size(-1);

        auto out_height = output.get_axis_size(-2);
        auto out_width = output.get_axis_size(-1);

        auto kernel = raw::resize_nn<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, out_height, out_width, input, in_height, in_width);
    }

    template void resize_nn<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>);
    template void resize_nn<float>(const Stream&, TensorSpan<float>, TensorView<float>);

    template <class T>
    void resize_bilinear(const Stream& stream, TensorSpan<T> output, TensorView<T> input, float scale_y, float scale_x) {
        auto in_height = input.get_axis_size(-2);
        auto in_width = input.get_axis_size(-1);

        auto out_height = output.get_axis_size(-2);
        auto out_width = output.get_axis_size(-1);

        auto kernel = raw::resize_bilinear<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x);
    }

    template void resize_bilinear<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, float, float);
    template void resize_bilinear<float>(const Stream&, TensorSpan<float>, TensorView<float>, float, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
