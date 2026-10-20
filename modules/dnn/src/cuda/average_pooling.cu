// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "fast_divmod.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/average_pooling.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        struct average_pool_accumulator {
            using type = T;
        };
        template <>
        struct average_pool_accumulator<__half> {
            using type = float;
        };

        template <class T>
        __global__ void average_pool(
            std::int64_t channels, std::int64_t height, std::int64_t width, std::int64_t depth,
            std::int64_t pooled_height, std::int64_t pooled_width, std::int64_t pooled_depth,
            std::int64_t kernel_h, std::int64_t kernel_w, std::int64_t kernel_d,
            std::int64_t stride_h, std::int64_t stride_w, std::int64_t stride_d,
            std::int64_t pad_h_head, std::int64_t pad_w_head, std::int64_t pad_d_head,
            std::int64_t pad_h_tail, std::int64_t pad_w_tail, std::int64_t pad_d_tail,
            std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t dilation_d,
            fast_divmod fdm_c, fast_divmod fdm_h, fast_divmod fdm_w, fast_divmod fdm_d,
            bool count_include_pad,
            View<T> input, size_type output_size, Span<T> output)
        {
            using AccT = typename average_pool_accumulator<T>::type;
            for (auto id : grid_stride_range(output_size)) {
                int d_index, w_index, h_index, c_index, n_index, id_tmp;
                fdm_d.divmod(id, id_tmp, d_index);
                fdm_w.divmod(id_tmp, id_tmp, w_index);
                fdm_h.divmod(id_tmp, id_tmp, h_index);
                fdm_c.divmod(id_tmp, n_index, c_index);

                std::int64_t h_start = static_cast<std::int64_t>(h_index) * stride_h - pad_h_head;
                std::int64_t w_start = static_cast<std::int64_t>(w_index) * stride_w - pad_w_head;
                std::int64_t d_start = static_cast<std::int64_t>(d_index) * stride_d - pad_d_head;

                std::int64_t h_end = csl::device::min<std::int64_t>(h_start + kernel_h * dilation_h, height + pad_h_tail);
                std::int64_t w_end = csl::device::min<std::int64_t>(w_start + kernel_w * dilation_w, width + pad_w_tail);
                std::int64_t d_end = csl::device::min<std::int64_t>(d_start + kernel_d * dilation_d, depth + pad_d_tail);

                // clip the window to the input once; the start advances by whole dilation steps to keep the phase
                auto clip_start = [](std::int64_t s, std::int64_t dil) { return s < 0 ? s + ((-s + dil - 1) / dil) * dil : s; };
                auto taps = [](std::int64_t lo, std::int64_t hi, std::int64_t dil) { return hi > lo ? 1 + (hi - 1 - lo) / dil : std::int64_t(0); };
                const std::int64_t h0 = clip_start(h_start, dilation_h), h1 = csl::device::min<std::int64_t>(h_end, height);
                const std::int64_t w0 = clip_start(w_start, dilation_w), w1 = csl::device::min<std::int64_t>(w_end, width);
                const std::int64_t d0 = clip_start(d_start, dilation_d), d1 = csl::device::min<std::int64_t>(d_end, depth);
                const std::int64_t counted = taps(h0, h1, dilation_h) * taps(w0, w1, dilation_w) * taps(d0, d1, dilation_d);

                AccT acc = static_cast<AccT>(0);
                std::int64_t offset = ((static_cast<std::int64_t>(n_index) * channels + c_index) * height) * width * depth;
                const T* p_slice = input.data().get() + offset;
                for (std::int64_t h = h0; h < h1; h += dilation_h)
                    for (std::int64_t w = w0; w < w1; w += dilation_w)
                        for (std::int64_t d = d0; d < d1; d += dilation_d)
                            acc += static_cast<AccT>(p_slice[(h * width + w) * depth + d]);

                AccT result = static_cast<AccT>(0);
                if (counted > 0) {
                    if (count_include_pad) {
                        std::int64_t divisor = (1 + (h_end - h_start - 1) / dilation_h) *
                                               (1 + (w_end - w_start - 1) / dilation_w) *
                                               (1 + (d_end - d_start - 1) / dilation_d);
                        result = acc / static_cast<AccT>(divisor);
                    } else {
                        result = acc / static_cast<AccT>(counted);
                    }
                }
                output[id] = static_cast<T>(result);
            }
        }
    }

    template <class T>
    void average_pool(const Stream& stream,
        Span<T> output, View<T> input,
        const std::vector<std::int64_t>& input_shape,
        const std::vector<std::int64_t>& output_shape,
        const std::vector<std::int64_t>& kernel_shape,
        const std::vector<std::int64_t>& strides,
        const std::vector<std::int64_t>& pads,
        const std::vector<std::int64_t>& dilations,
        bool count_include_pad)
    {
        std::int64_t channels = input_shape[1];
        std::int64_t height = input_shape[2];
        std::int64_t width = kernel_shape.size() > 1 ? input_shape[3] : 1;
        std::int64_t depth = kernel_shape.size() > 2 ? input_shape[4] : 1;

        std::int64_t pooled_height = output_shape[2];
        std::int64_t pooled_width = kernel_shape.size() > 1 ? output_shape[3] : 1;
        std::int64_t pooled_depth = kernel_shape.size() > 2 ? output_shape[4] : 1;

        const std::int64_t rank = static_cast<std::int64_t>(kernel_shape.size());
        std::int64_t kernel_h = kernel_shape[0];
        std::int64_t kernel_w = rank > 1 ? kernel_shape[1] : 1;
        std::int64_t kernel_d = rank > 2 ? kernel_shape[2] : 1;
        std::int64_t stride_h = strides[0];
        std::int64_t stride_w = rank > 1 ? strides[1] : 1;
        std::int64_t stride_d = rank > 2 ? strides[2] : 1;

        std::int64_t pad_h_head = pads[0];
        std::int64_t pad_w_head = rank > 1 ? pads[1] : 0;
        std::int64_t pad_d_head = rank > 2 ? pads[2] : 0;
        std::int64_t pad_h_tail = pads[rank + 0];
        std::int64_t pad_w_tail = rank > 1 ? pads[rank + 1] : 0;
        std::int64_t pad_d_tail = rank > 2 ? pads[rank + 2] : 0;

        std::int64_t dilation_h = dilations[0];
        std::int64_t dilation_w = rank > 1 ? dilations[1] : 1;
        std::int64_t dilation_d = rank > 2 ? dilations[2] : 1;

        std::int64_t output_size = 1;
        for (auto d : output_shape) output_size *= d;
        if (output_size == 0) return;

        fast_divmod fdm_c(static_cast<int>(channels));
        fast_divmod fdm_h(static_cast<int>(pooled_height));
        fast_divmod fdm_w(static_cast<int>(pooled_width));
        fast_divmod fdm_d(static_cast<int>(pooled_depth));

        auto kernel = raw::average_pool<T>;
        auto policy = make_policy(kernel, static_cast<std::size_t>(output_size), 0, stream);
        launch_kernel(kernel, policy,
            channels, height, width, depth,
            pooled_height, pooled_width, pooled_depth,
            kernel_h, kernel_w, kernel_d,
            stride_h, stride_w, stride_d,
            pad_h_head, pad_w_head, pad_d_head,
            pad_h_tail, pad_w_tail, pad_d_tail,
            dilation_h, dilation_w, dilation_d,
            fdm_c, fdm_h, fdm_w, fdm_d,
            count_include_pad,
            input, static_cast<size_type>(output_size), output);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void average_pool(const Stream&, Span<__half>, View<__half>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, bool);
#endif
    template void average_pool(const Stream&, Span<float>, View<float>, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
