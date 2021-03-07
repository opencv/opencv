// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "memory.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cuda_runtime.h>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T, std::size_t CHANNELS_PER_ITER>
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

            /* we have to compute the output value for every combination of (box, c, y, x) in the output
             *
             * the computation involving (y, x) are identical for all non-spatial dimensions
             * the computation and memory requests involving the box are identical for remaining three axes
             *
             * we process multiple channels every iteration to reuse the identical computation
             * and memory requests involved with the box and spatial dimensions
             */

            /*
             * if we are processing `CHANNELS_PER_ITER` channels per iteration, we will need
             * (num_channels / CHANNELS_PER_ITER) iterations per (box, x, y)
             */
            auto num_channel_iters_per_box_xy = num_channels / CHANNELS_PER_ITER;

            /* we need `num_channel_iters_per_box_xy` iterations per (box, x, y) and there are
             * `num_boxes` boxes and `out_image_size` combinations of (x, y)
             */
            auto num_boxes = boxes.size() / 7; /* 7 values per box */
            auto iters_per_box = num_channel_iters_per_box_xy * out_image_size;
            auto iters_required = num_boxes * iters_per_box;

            for (auto iter : grid_stride_range(iters_required)) {
                const index_type box_no = iter / iters_per_box;
                const index_type c_start = ((iter % iters_per_box) / out_image_size) * CHANNELS_PER_ITER;

                /* note here that consecutive `iter` values will often have consecutive `x` values
                 * => stores into output will be coalesced across threads
                 */
                const index_type y = (iter % out_image_size) / out_width;
                const index_type x = iter % out_width;

                const index_type box_offset = box_no * 7;
                const auto left = boxes[box_offset + 3],
                           top = boxes[box_offset + 4],
                           right = boxes[box_offset + 5],
                           bottom = boxes[box_offset + 6];

                const auto box_width = right - left;
                const auto box_height = bottom - top;

                const auto o2i_fy = static_cast<T>(in_height - 1) / static_cast<T>(out_height - 1);
                const auto o2i_fx = static_cast<T>(in_width - 1) / static_cast<T>(out_width - 1);

                const auto height_scale = box_height * o2i_fy;
                const auto width_scale = box_width * o2i_fx;

                const auto in_y = top * static_cast<T>(in_height - 1) + static_cast<T>(y) * height_scale;
                const auto in_x = left * static_cast<T>(in_width - 1) + static_cast<T>(x) * width_scale;

                const auto in_y0 = static_cast<index_type>(in_y);
                const auto in_x0 = static_cast<index_type>(in_x);

                using device::min;
                const auto in_x1 = min<index_type>(in_x0 + 1, in_width - 1);
                const auto in_y1 = min<index_type>(in_y0 + 1, in_height - 1);

                index_type in_offset_r0 = c_start * in_image_size + in_y0 * in_width;
                index_type in_offset_r1 = c_start * in_image_size + in_y1 * in_width;
                index_type out_idx = box_no * out_box_size + c_start * out_image_size + y * out_width + x;

                #pragma unroll 1 /* disable unrolling */
                for (int i = 0; i < CHANNELS_PER_ITER; i++) {
                    auto v_00 = load_ldg(input[in_offset_r0 + in_x0]),
                         v_01 = load_ldg(input[in_offset_r0 + in_x1]),
                         v_10 = load_ldg(input[in_offset_r1 + in_x0]),
                         v_11 = load_ldg(input[in_offset_r1 + in_x1]);

                    output[out_idx] =
                        v_00 +
                        T(in_y - T(in_y0)) * T(v_10 - v_00) +
                        T(in_x - T(in_x0)) * T(v_01 - v_00) +
                        T(in_y - T(in_y0)) * T(in_x - T(in_x0)) * T(v_11 - v_01 - v_10 + v_00);

                    in_offset_r0 += in_image_size;
                    in_offset_r1 += in_image_size;
                    out_idx += out_image_size;
                }
            }
        }
    }

    template <class T, std::size_t CHANNELS_PER_ITER> static
    void launch_multichannel_crop_and_resize(const Stream& stream,
            Span<T> output, size_type out_height, size_type out_width,
            View<T> input, size_type in_height, size_type in_width,
            View<T> boxes, size_type num_channels)
    {
        auto kernel = raw::crop_and_resize<T, CHANNELS_PER_ITER>;
        auto policy = make_policy(kernel, output.size() / CHANNELS_PER_ITER, 0, stream);
        launch_kernel(kernel, policy, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
    }

    template <class T>
    void crop_and_resize(const Stream& stream, TensorSpan<T> output, TensorView<T> input, View<T> boxes) {
        CV_Assert(input.get_axis_size(0) == 1); /* batch not supported */
        CV_Assert(input.get_axis_size(1) == output.get_axis_size(1));

        auto out_height = output.get_axis_size(-2);
        auto out_width = output.get_axis_size(-1);

        auto in_height = input.get_axis_size(-2);
        auto in_width = input.get_axis_size(-1);

        auto num_channels = input.get_axis_size(1);

        if (num_channels % 64 == 0) {
            launch_multichannel_crop_and_resize<T, 64>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        } else if (num_channels % 32 == 0) {
            launch_multichannel_crop_and_resize<T, 32>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        } else if (num_channels % 16 == 0) {
            launch_multichannel_crop_and_resize<T, 16>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        } else if (num_channels % 8 == 0) {
            launch_multichannel_crop_and_resize<T, 8>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        } else if (num_channels % 4 == 0) {
            launch_multichannel_crop_and_resize<T, 4>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        } else if (num_channels % 2 == 0) {
            launch_multichannel_crop_and_resize<T, 2>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        } else {
            launch_multichannel_crop_and_resize<T, 1>(stream, output, out_height, out_width, input, in_height, in_width, boxes, num_channels);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void crop_and_resize<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, View<__half> boxes);
#endif
    template void crop_and_resize<float>(const Stream&, TensorSpan<float>, TensorView<float>, View<float> boxes);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
