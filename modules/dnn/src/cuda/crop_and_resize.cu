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

#include <opencv2/core.hpp>

#include <cuda_runtime.h>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T>
        __global__ void crop_and_resize(
            Span<T> output, size_type out_height, size_type out_width,
            View<T> input, size_type in_height, size_type in_width,
            View<T> boxes,
            size_type num_channels)
        {
            // input [1, num_channels, in_height, in_width]
            // output [boxes, num_channels, out_height, out_width]

            const auto in_image_size = in_height * in_width;
            const auto out_image_size = out_height * out_width;
            const auto out_box_size = num_channels * out_image_size;

            const auto o2i_fy = static_cast<T>(in_height - 1) / static_cast<T>(out_height - 1);
            const auto o2i_fx = static_cast<T>(in_width - 1) / static_cast<T>(out_width - 1);

            /* think of the output and input as a collection of grayscale 2d images with the last axis
             * representing the width and the last but one axis representing the height
             *
             * the remaining axis together form a collection of these images
             */
            for (auto idx : grid_stride_range(output.size())) {
                const index_type box_no = idx / out_box_size;
                const index_type c = (idx % out_box_size) / out_image_size;
                const index_type y = (idx % out_image_size) / out_width;
                const index_type x = (idx % out_image_size) % out_width;

                const index_type box_offset = box_no * 7;
                const auto left = boxes[box_offset + 3],
                           top = boxes[box_offset + 4],
                           right = boxes[box_offset + 5],
                           bottom = boxes[box_offset + 6];

                const auto box_width = right - left;
                const auto box_height = bottom - top;

                const auto height_scale = box_height * o2i_fy;
                const auto width_scale = box_width * o2i_fx;

                const auto in_y = top * static_cast<T>(in_height - 1) + static_cast<T>(y) * height_scale;
                const auto in_x = left * static_cast<T>(in_width - 1) + static_cast<T>(x) * width_scale;

                const auto in_y0 = static_cast<index_type>(in_y);
                const auto in_x0 = static_cast<index_type>(in_x);

                using device::min;
                const auto in_x1 = min<index_type>(in_x0 + 1, in_width - 1);
                const auto in_y1 = min<index_type>(in_y0 + 1, in_height - 1);

                const index_type in_offset_r0 = c * in_image_size + in_y0 * in_width;
                const index_type in_offset_r1 = c * in_image_size + in_y1 * in_width;

                const auto v_00 = input[in_offset_r0 + in_x0],
                           v_01 = input[in_offset_r0 + in_x1],
                           v_10 = input[in_offset_r1 + in_x0],
                           v_11 = input[in_offset_r1 + in_x1];

                output[idx] =
                    v_00 +
                    T(in_y - T(in_y0)) * T(v_10 - v_00) +
                    T(in_x - T(in_x0)) * T(v_01 - v_00) +
                    T(in_y - T(in_y0)) * T(in_x - T(in_x0)) * T(v_11 - v_01 - v_10 + v_00);
            }
        }
    }

    template <class T>
    void crop_and_resize(const Stream& stream, TensorSpan<T> output, TensorView<T> input, View<T> boxes) {
        CV_Assert(input.get_axis_size(0) == 1); /* batch not supported */

        auto in_height = input.get_axis_size(-2);
        auto in_width = input.get_axis_size(-1);

        auto out_height = output.get_axis_size(-2);
        auto out_width = output.get_axis_size(-1);

        CV_Assert(input.get_axis_size(1) == output.get_axis_size(1));
        auto num_channels = input.get_axis_size(1);

        auto kernel = raw::crop_and_resize<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
    }

    template void crop_and_resize<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, View<__half> boxes);
    template void crop_and_resize<float>(const Stream&, TensorSpan<float>, TensorView<float>, View<float> boxes);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
