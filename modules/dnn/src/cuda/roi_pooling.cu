// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "limits.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T, std::size_t CHANNELS_PER_ITER>
        __global__ void roi_pooling(
            Span<T> output, size_type pooled_height, size_type pooled_width,
            View<T> input, size_type in_height, size_type in_width,
            View<T> rois, size_type num_channels, float spatial_scale)
        {
            // input: [1, num_channels, in_height, in_width]
            const auto in_image_size = in_height * in_width;

            // rois: [num_rois, 5]
            auto num_rois = rois.size() / 5;

            // output: [num_rois, num_channels, pooled_height, pooled_width]
            const auto out_spatial_size = pooled_height * pooled_width;
            const auto out_roi_size = num_channels * out_spatial_size;

            /* we have to compute the output value for every combination of (roi, c, y, x) in the output
             *
             * the computation involving (y, x) are identical for all non-spatial dimensions
             * the computation and memory requests involving the roi are identical for remaining three axes
             *
             * we process multiple channels every iteration to reuse the identical computation
             * and memory requests involved with the roi and spatial dimensions
             */
            /*
             * if we are processing `CHANNELS_PER_ITER` channels per iteration, we will need
             * (num_channels / CHANNELS_PER_ITER) iterations per (roi, x, y)
             */
            auto num_channel_iters_per_roi_xy = num_channels / CHANNELS_PER_ITER;

            /* we need `num_channel_iters_per_roi_xy` iterations per (roi, x, y) and there are
             * `num_rois` rois and `out_spatial_size` combinations of (x, y)
             */
            auto iters_per_roi = num_channel_iters_per_roi_xy * out_spatial_size;
            auto iters_required = num_rois * iters_per_roi;

            for (auto iter : grid_stride_range(iters_required))
            {
                const index_type roi_no = iter / iters_per_roi;
                const index_type c_start = ((iter % iters_per_roi) / out_spatial_size) * CHANNELS_PER_ITER;

                /* note here that consecutive `iter` values will often have consecutive `x` values
                 * => stores into output will be coalesced across threads
                 */
                const index_type y = (iter % out_spatial_size) / pooled_width;
                const index_type x = iter % pooled_width;

                const index_type roi_offset = roi_no * 5;

                using device::round;
                const index_type batch_id = rois[roi_offset + 0];
                const index_type x_start_roi = round(static_cast<float>(rois[roi_offset + 1]) * spatial_scale);
                const index_type y_start_roi = round(static_cast<float>(rois[roi_offset + 2]) * spatial_scale);
                const index_type x_end_roi = round(static_cast<float>(rois[roi_offset + 3]) * spatial_scale);
                const index_type y_end_roi = round(static_cast<float>(rois[roi_offset + 4]) * spatial_scale);

                using device::max;
                const auto roi_width = max<index_type>(x_end_roi - x_start_roi + 1, 1);
                const auto roi_height = max<index_type>(y_end_roi - y_start_roi + 1, 1);

                const auto roi_width_ratio = static_cast<float>(roi_width) / pooled_width;
                const auto roi_height_ratio = static_cast<float>(roi_height) / pooled_height;

                auto x_start = x_start_roi + static_cast<index_type>(x * roi_width_ratio);
                auto y_start = y_start_roi + static_cast<index_type>(y * roi_height_ratio);

                using device::ceil;
                auto x_end = x_start_roi + static_cast<index_type>(ceil((x + 1) * roi_width_ratio));
                auto y_end = y_start_roi + static_cast<index_type>(ceil((y + 1) * roi_height_ratio));

                using device::max;
                x_start = max<index_type>(x_start, 0);
                y_start = max<index_type>(y_start, 0);

                using device::min;
                x_end = min<index_type>(x_end, in_width);
                y_end = min<index_type>(y_end, in_height);

                index_type in_offset = (batch_id * num_channels + c_start) * in_height * in_width;
                index_type out_idx = roi_no * out_roi_size + c_start * out_spatial_size + y * pooled_width + x;

                for (int i = 0; i < CHANNELS_PER_ITER; i++)
                {
                    /* We have to set the output to zero if (x_start >= x_end) or (y_start >= y_end). If either
                     * condition is true, the loops below won't execute even a single iteration. Hence, by setting
                     * `max_val` to zero in this case, we can combine it with the `else` code.
                     */
                    T max_val = (x_start >= x_end || y_start >= y_end) ? T(0) : device::numeric_limits<T>::lowest();

                    for (auto iy = y_start; iy < y_end; iy++)
                    {
                        const auto in_idx = in_offset + iy * in_width;
                        for (auto ix = x_start; ix < x_end; ix++)
                        {
                            max_val = max(max_val, input[in_idx + ix]);
                        }
                    }

                    output[out_idx] = max_val;

                    in_offset += in_image_size;
                    out_idx += out_spatial_size;
                }
            }
        }
    }

    template <class T, std::size_t CHANNELS_PER_ITER> static
    void launch_multichannel_roi_pooling(const Stream& stream,
        Span<T> output, size_type pooled_height, size_type pooled_width,
        View<T> input, size_type in_height, size_type in_width,
        View<T> rois, size_type num_channels, float spatial_scale)
    {
        auto kernel = raw::roi_pooling<T, CHANNELS_PER_ITER>;
        auto policy = make_policy(kernel, output.size() / CHANNELS_PER_ITER, 0, stream);
        launch_kernel(kernel, policy, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
    }

    template <class T>
    void roi_pooling(const Stream& stream, TensorSpan<T> output, TensorView<T> input, View<T> rois, float spatial_scale)
    {
        CV_Assert(input.get_axis_size(1) == output.get_axis_size(1));

        size_type num_channels = output.get_axis_size(1);

        size_type pooled_height = output.get_axis_size(2);
        size_type pooled_width = output.get_axis_size(3);

        size_type in_height = input.get_axis_size(2);
        size_type in_width = input.get_axis_size(3);

        if (num_channels % 64 == 0) {
            launch_multichannel_roi_pooling<T, 64>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        } else if (num_channels % 32 == 0) {
            launch_multichannel_roi_pooling<T, 32>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        } else if (num_channels % 16 == 0) {
            launch_multichannel_roi_pooling<T, 16>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        } else if (num_channels % 8 == 0) {
            launch_multichannel_roi_pooling<T, 8>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        } else if (num_channels % 4 == 0) {
            launch_multichannel_roi_pooling<T, 4>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        } else if (num_channels % 2 == 0) {
            launch_multichannel_roi_pooling<T, 2>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        } else {
            launch_multichannel_roi_pooling<T, 1>(stream, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void roi_pooling(const Stream& stream, TensorSpan<__half> output, TensorView<__half> input, View<__half> rois, float spatial_scale);
#endif
    template void roi_pooling(const Stream& stream, TensorSpan<float> output, TensorView<float> input, View<float> rois, float spatial_scale);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
