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

        template <class T>
        __global__ void roi_pooling(
            Span<T> output, size_type pooled_height, size_type pooled_width,
            View<T> input, size_type in_height, size_type in_width,
            View<T> rois, size_type num_channels, T spatial_scale)
        {
            // input: [1, num_channels, in_height, in_width]
            // rois: [num_rois, 5]

            // output: [num_rois, num_channels, pooled_height, pooled_width]
            const auto out_spatial_size = pooled_height * pooled_width;
            const auto out_roi_size = num_channels * out_spatial_size;

            /* every element in the output is mapped to a window in the input and each thread processes several windows */
            for (auto idx : grid_stride_range(output.size()))
            {
                const auto n = idx / out_roi_size;
                const auto c = (idx % out_roi_size) / out_spatial_size;
                const auto y = (idx % out_spatial_size) / pooled_width;
                const auto x = idx % pooled_width;

                const index_type roi_offset = n * 5;

                using device::round;
                const index_type batch_id = rois[roi_offset + 0];
                const index_type x_start_roi = round(rois[roi_offset + 1] * spatial_scale);
                const index_type y_start_roi = round(rois[roi_offset + 2] * spatial_scale);
                const index_type x_end_roi = round(rois[roi_offset + 3] * spatial_scale);
                const index_type y_end_roi = round(rois[roi_offset + 4] * spatial_scale);

                using device::max;
                const auto roi_width = max<index_type>(x_end_roi - x_start_roi + 1, 1);
                const auto roi_height = max<index_type>(y_end_roi - y_start_roi + 1, 1);

                const auto roi_width_ratio = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
                const auto roi_height_ratio = static_cast<T>(roi_height) / static_cast<T>(pooled_height);

                auto x_start = x_start_roi + static_cast<index_type>(static_cast<T>(x) * roi_width_ratio);
                auto y_start = y_start_roi + static_cast<index_type>(static_cast<T>(y) * roi_height_ratio);

                using device::ceil;
                auto x_end = x_start_roi + static_cast<index_type>(ceil(static_cast<T>(x + 1) * roi_width_ratio));
                auto y_end = y_start_roi + static_cast<index_type>(ceil(static_cast<T>(y + 1) * roi_height_ratio));

                using device::max;
                x_start = max<index_type>(x_start, 0);
                y_start = max<index_type>(y_start, 0);

                using device::min;
                x_end = min<index_type>(x_end, in_width);
                y_end = min<index_type>(y_end, in_height);

                /* We have to set the output to zero if (x_start >= x_end) or (y_start >= y_end). If either
                 * condition is true, the loops below won't execute even a single iteration. Hence, by setting
                 * `max_val` to zero in this case, we can combine it with the `else` code.
                 */
                T max_val = (x_start >= x_end || y_start >= y_end) ? T(0) : device::numeric_limits<T>::lowest();

                const index_type in_offset = (batch_id * num_channels + c) * in_height * in_width;
                for (auto iy = y_start; iy < y_end; iy++)
                {
                    for (auto ix = x_start; ix < x_end; ix++)
                    {
                        const auto in_idx = in_offset + iy * in_width + ix;
                        max_val = max(max_val, input[in_idx]);
                    }
                }

                output[idx] = max_val;
            }
        }
    }

    template <class T>
    void roi_pooling(const Stream& stream, TensorSpan<T> output, TensorView<T> input, View<T> rois, T spatial_scale)
    {
        CV_Assert(input.get_axis_size(1) == output.get_axis_size(1));

        size_type num_channels = output.get_axis_size(1);

        size_type pooled_height = output.get_axis_size(2);
        size_type pooled_width = output.get_axis_size(3);

        size_type in_height = input.get_axis_size(2);
        size_type in_width = input.get_axis_size(3);

        auto kernel = raw::roi_pooling<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, pooled_height, pooled_width, input, in_height, in_width, rois, num_channels, spatial_scale);
    }

    template void roi_pooling(const Stream& stream, TensorSpan<__half> output, TensorView<__half> input, View<__half> rois, __half spatial_scale);
    template void roi_pooling(const Stream& stream, TensorSpan<float> output, TensorView<float> input, View<float> rois, float spatial_scale);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
