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

#include <cuda_runtime.h>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t CHANNELS_PER_ITER>
        __global__ void resize_nn(
            Span<T> output, size_type out_height, size_type out_width,
            View<T> input, size_type in_height, size_type in_width,
            float o2i_fy, float o2i_fx, bool round, bool half_pixel_centers)
        {
            auto in_image_size = in_height * in_width;
            auto out_image_size = out_height * out_width;

            /* think of the output and input as a collection of 2d images with the last axis
             * representing the width and the last but one axis representing the height
             *
             * the remaining axis together form a collection of these images/channels
             */
            auto num_effective_channels = output.size() / out_image_size;

            /* we process multiple channels every iteration to reuse the identical computation
             * involved with the spatial dimensions
             *
             * if we are processing `CHANNELS_PER_ITER` channels per iteration, we will need
             * (num_effective_channels / CHANNELS_PER_ITER) iterations per (x, y) location
             */
            auto num_channel_iters_per_xy = (num_effective_channels / CHANNELS_PER_ITER);

            /* we need `num_channel_iters_per_xy` iterations per (x, y) and there are `out_image_size`
             * combinations of (x, y); hence, we'll need `num_channel_iters_per_xy * out_image_size`
             * iterations in total to finish the resize operation
             */
            auto iters_required = num_channel_iters_per_xy * out_image_size;
            for (auto iter : grid_stride_range(iters_required)) {
                const index_type c_start = (iter / out_image_size) * CHANNELS_PER_ITER;

                /* note here that consecutive `iter` values will often have consecutive `x` values
                 * => stores into output will be coalesced across threads
                 */
                const index_type y = (iter % out_image_size) / out_width;
                const index_type x = iter % out_width;

                auto in_yf = half_pixel_centers ? (y + 0.5f) * o2i_fy : y * o2i_fy;
                auto in_xf = half_pixel_centers ? (x + 0.5f) * o2i_fx : x * o2i_fx;

                using device::lround;
                index_type in_y = round ? lround(in_yf) : static_cast<index_type>(in_yf);
                index_type in_x = round ? lround(in_xf) : static_cast<index_type>(in_xf);

                using device::min;
                in_y = min(in_y, in_height - 1);
                in_x = min(in_x, in_width - 1);

                index_type in_idx = c_start * in_image_size + in_y * in_width + in_x;
                index_type out_idx = c_start * out_image_size + y * out_width + x;

                for (int i = 0; i < CHANNELS_PER_ITER; i++) {
                    output[out_idx] = load_ldg(input[in_idx]);

                    in_idx += in_image_size;
                    out_idx += out_image_size;
                }
            }
        }

        template <class T, std::size_t CHANNELS_PER_ITER>
        __global__ void resize_bilinear(
            Span<T> output, size_type out_height, size_type out_width,
            View<T> input, size_type in_height, size_type in_width,
            float o2i_fy, float o2i_fx, bool half_pixel_centers)
        {
            auto in_image_size = in_height * in_width;
            auto out_image_size = out_height * out_width;

            /* think of the output and input as a collection of 2d images with the last axis
             * representing the width and the last but one axis representing the height
             *
             * the remaining axis together form a collection of these images/channels
             */
            auto num_effective_channels = output.size() / out_image_size;

            /* we process multiple channels every iteration to reuse the identical computation
             * involved with the spatial dimensions
             *
             * if we are processing `CHANNELS_PER_ITER` channels per iteration, we will need
             * (num_effective_channels / CHANNELS_PER_ITER) iterations per (x, y) location
             */
            auto num_channel_iters_per_xy = (num_effective_channels / CHANNELS_PER_ITER);

            /* we need `num_channel_iters_per_xy` iterations per (x, y) and there are `out_image_size`
             * combinations of (x, y); hence, we'll need `num_channel_iters_per_xy * out_image_size`
             * iterations in total to finish the resize operation
             */
            auto iters_required = num_channel_iters_per_xy * out_image_size;

            for (auto iter : grid_stride_range(iters_required)) {
                const index_type c_start = (iter / out_image_size) * CHANNELS_PER_ITER;
                const index_type c_end = c_start + CHANNELS_PER_ITER;

                /* note here that consecutive `iter` values will often have consecutive `x` values
                 * => stores into output will be coalesced across threads
                 */
                const index_type y = (iter % out_image_size) / out_width;
                const index_type x = iter % out_width;

                using device::max;
                auto in_x = half_pixel_centers ? max<float>((x + 0.5f) * o2i_fx - 0.5f, 0.0f) : x * o2i_fx;
                auto in_y = half_pixel_centers ? max<float>((y + 0.5f) * o2i_fy - 0.5f, 0.0f) : y * o2i_fy;

                auto in_x0 = static_cast<index_type>(in_x);
                auto in_y0 = static_cast<index_type>(in_y);

                using device::min;
                auto in_x1 = min<index_type>(in_x0 + 1, in_width - 1);
                auto in_y1 = min<index_type>(in_y0 + 1, in_height - 1);

                index_type in_offset_r0 = c_start * in_image_size + in_y0 * in_width;
                index_type in_offset_r1 = c_start * in_image_size + in_y1 * in_width;
                index_type out_idx = c_start * out_image_size + y * out_width + x;

                #pragma unroll 1 /* disable unrolling to reduce register pressure; not sure how but it works */
                for (auto c = c_start; c < c_end; c++) {
                    auto v_00 = load_ldg(input[in_offset_r0 + in_x0]),
                         v_01 = load_ldg(input[in_offset_r0 + in_x1]),
                         v_10 = load_ldg(input[in_offset_r1 + in_x0]),
                         v_11 = load_ldg(input[in_offset_r1 + in_x1]);

                    output[out_idx] =
                        v_00 +
                        T(in_y - in_y0) * T(v_10 - v_00) +
                        T(in_x - in_x0) * T(v_01 - v_00) +
                        T(in_y - in_y0) * T(in_x - in_x0) * T(v_11 - v_01 - v_10 + v_00);

                    in_offset_r0 += in_image_size;
                    in_offset_r1 += in_image_size;
                    out_idx += out_image_size;
                }
            }
        }
    }

    template <class T, std::size_t CHANNELS_PER_ITER> static
    void launch_multichannel_resize_nn(const Stream& stream,
        Span<T> output, size_type out_height, size_type out_width,
        View<T> input, size_type in_height, size_type in_width,
        float scale_y, float scale_x, bool round, bool half_pixel_centers)
    {
        auto kernel = raw::resize_nn<T, CHANNELS_PER_ITER>;
        auto policy = make_policy(kernel, output.size() / CHANNELS_PER_ITER, 0, stream);
        launch_kernel(kernel, policy, output, out_height, out_width, input, in_height, in_width,  scale_y, scale_x, round, half_pixel_centers);
    }

    template <class T>
    void resize_nn(const Stream& stream, TensorSpan<T> output, TensorView<T> input, float scale_y, float scale_x, bool round, bool half_pixel_centers) {
        auto out_height = output.get_axis_size(-2);
        auto out_width = output.get_axis_size(-1);

        auto in_height = input.get_axis_size(-2);
        auto in_width = input.get_axis_size(-1);

        auto num_effective_channels = input.size_range(0, 2);
        auto num_iters = num_effective_channels * out_height * out_width;

        if (num_effective_channels % 32 == 0 && num_iters > 655360) {
            launch_multichannel_resize_nn<T, 32>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, round, half_pixel_centers);
        } else if (num_effective_channels % 16 == 0 && num_iters > 327680) {
            launch_multichannel_resize_nn<T, 16>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, round, half_pixel_centers);
        } else if (num_effective_channels % 8 == 0 && num_iters > 163840) {
            launch_multichannel_resize_nn<T, 8>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, round, half_pixel_centers);
        } else if (num_effective_channels % 4 == 0 && num_iters > 81920) {
            launch_multichannel_resize_nn<T, 4>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, round, half_pixel_centers);
        } else if (num_effective_channels % 2 == 0) {
            launch_multichannel_resize_nn<T, 2>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, round, half_pixel_centers);
        } else {
            launch_multichannel_resize_nn<T, 1>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, round, half_pixel_centers);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void resize_nn<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, float, float, bool, bool);
#endif
    template void resize_nn<float>(const Stream&, TensorSpan<float>, TensorView<float>, float, float, bool,bool);

    template <class T, std::size_t CHANNELS_PER_ITER> static
    void launch_multichannel_resize_bilinear(const Stream& stream,
        Span<T> output, size_type out_height, size_type out_width,
        View<T> input, size_type in_height, size_type in_width,
        float scale_y, float scale_x, bool half_pixel_centers)
    {
        auto kernel = raw::resize_bilinear<T, CHANNELS_PER_ITER>;
        auto policy = make_policy(kernel, output.size() / CHANNELS_PER_ITER, 0, stream);
        launch_kernel(kernel, policy, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, half_pixel_centers);
    }

    template <class T>
    void resize_bilinear(const Stream& stream, TensorSpan<T> output, TensorView<T> input, float scale_y, float scale_x, bool half_pixel_centers) {
        auto out_height = output.get_axis_size(-2);
        auto out_width = output.get_axis_size(-1);

        auto in_height = input.get_axis_size(-2);
        auto in_width = input.get_axis_size(-1);

        auto num_effective_channels = input.size_range(0, 2);
        auto num_iters = num_effective_channels * out_height * out_width;

        if (num_effective_channels % 16 == 0 && num_iters > 163840) {
            launch_multichannel_resize_bilinear<T, 16>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, half_pixel_centers);
        } else if (num_effective_channels % 8 == 0 && num_iters > 81920) {
            launch_multichannel_resize_bilinear<T, 8>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, half_pixel_centers);
        } else if (num_effective_channels % 4 == 0 && num_iters > 40960) {
            launch_multichannel_resize_bilinear<T, 4>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, half_pixel_centers);
        } else if (num_effective_channels % 2 == 0) {
            launch_multichannel_resize_bilinear<T, 2>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, half_pixel_centers);
        } else {
            launch_multichannel_resize_bilinear<T, 1>(stream, output, out_height, out_width, input, in_height, in_width, scale_y, scale_x, half_pixel_centers);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void resize_bilinear<__half>(const Stream&, TensorSpan<__half>, TensorView<__half>, float, float, bool);
#endif
    template void resize_bilinear<float>(const Stream&, TensorSpan<float>, TensorView<float>, float, float, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
