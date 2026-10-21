// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/argmax.hpp"

#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, bool IsArgMax, bool SelectLast>
        __global__ void arg_min_max(View<T> input, Span<std::int64_t> output, size_type num_outputs, int axis_size, int inner_size) {
            for (auto id : grid_stride_range(num_outputs)) {
                int outer_index = id / inner_size;
                int inner_index = id % inner_size;
                std::int64_t base = (static_cast<std::int64_t>(outer_index) * axis_size) * inner_size + inner_index;

                const T* p_input = input.data().get();
                T best_value = p_input[base];
                std::int64_t best_index = 0;
                for (int k = 1; k < axis_size; ++k) {
                    const T value = p_input[base + static_cast<std::int64_t>(k) * inner_size];
                    if constexpr (IsArgMax) {
                        if (SelectLast ? value >= best_value : value > best_value) {
                            best_value = value;
                            best_index = k;
                        }
                    } else {
                        if (SelectLast ? value <= best_value : value < best_value) {
                            best_value = value;
                            best_index = k;
                        }
                    }
                }

                output[id] = best_index;
            }
        }
    }

    template <class T>
    void arg_min_max(const Stream& stream,
        Span<std::int64_t> output, View<T> input,
        int outer_size, int axis_size, int inner_size, bool is_argmax, bool select_last_index)
    {
        if (axis_size <= 0 || outer_size <= 0 || inner_size <= 0)
            return;

        const size_type num_outputs = static_cast<size_type>(outer_size) * inner_size;
        if (num_outputs == 0)
            return;

        if (is_argmax) {
            if (select_last_index) {
                auto kernel = raw::arg_min_max<T, true, true>;
                auto policy = make_policy(kernel, num_outputs, 0, stream);
                launch_kernel(kernel, policy, input, output, num_outputs, axis_size, inner_size);
            } else {
                auto kernel = raw::arg_min_max<T, true, false>;
                auto policy = make_policy(kernel, num_outputs, 0, stream);
                launch_kernel(kernel, policy, input, output, num_outputs, axis_size, inner_size);
            }
        } else {
            if (select_last_index) {
                auto kernel = raw::arg_min_max<T, false, true>;
                auto policy = make_policy(kernel, num_outputs, 0, stream);
                launch_kernel(kernel, policy, input, output, num_outputs, axis_size, inner_size);
            } else {
                auto kernel = raw::arg_min_max<T, false, false>;
                auto policy = make_policy(kernel, num_outputs, 0, stream);
                launch_kernel(kernel, policy, input, output, num_outputs, axis_size, inner_size);
            }
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void arg_min_max(const Stream&, Span<std::int64_t>, View<__half>, int, int, int, bool, bool);
#endif
    template void arg_min_max(const Stream&, Span<std::int64_t>, View<float>, int, int, int, bool, bool);
    template void arg_min_max(const Stream&, Span<std::int64_t>, View<int8_t>, int, int, int, bool, bool);
    template void arg_min_max(const Stream&, Span<std::int64_t>, View<uint8_t>, int, int, int, bool, bool);
    template void arg_min_max(const Stream&, Span<std::int64_t>, View<int32_t>, int, int, int, bool, bool);
    template void arg_min_max(const Stream&, Span<std::int64_t>, View<int64_t>, int, int, int, bool, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
