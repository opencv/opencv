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
#include "fast_divmod.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include "../cuda4dnn/kernels/gather_elements.hpp"

#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        __host__ __device__ inline std::int64_t get_element_index_value(const void* index_data, std::size_t index_element_size, std::size_t offset) {
            switch (index_element_size) {
                case sizeof(std::int32_t):
                    return *(reinterpret_cast<const std::int32_t*>(index_data) + offset);
                case sizeof(std::int64_t):
                    return *(reinterpret_cast<const std::int64_t*>(index_data) + offset);
                default:
                    break;
            }
            return 0;
        }

        template <class T>
        __global__ void gather_elements(
            Span<T> output, View<T> input,
            const void* indices, std::size_t index_element_size,
            std::int64_t axis_size_data, std::int64_t inner_size,
            fast_divmod axis_inner_size, fast_divmod inner_size_divmod)
        {
            for (auto id : grid_stride_range(output.size())) {
                int outer_index, axis_inner_offset;
                axis_inner_size.divmod(id, outer_index, axis_inner_offset);
                int axis_index, inner_index;
                inner_size_divmod.divmod(axis_inner_offset, axis_index, inner_index);
                (void)axis_index;

                std::int64_t idx = get_element_index_value(indices, index_element_size, id);
                idx = idx < 0 ? idx + axis_size_data : idx;
                if (idx < 0 || idx >= axis_size_data) {
                    output[id] = static_cast<T>(0);
                    continue;
                }

                std::int64_t input_index = static_cast<std::int64_t>(outer_index) * axis_size_data * inner_size +
                                           idx * inner_size + inner_index;
                output[id] = input.data().get()[input_index];
            }
        }
    }

    template <class T>
    void gather_elements(const Stream& stream,
        Span<T> output, View<T> input,
        const void* indices, std::size_t index_element_size,
        std::int64_t axis_size_data, std::int64_t inner_size,
        fast_divmod axis_inner_size, fast_divmod inner_size_divmod)
    {
        auto kernel = raw::gather_elements<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, indices, index_element_size,
                      axis_size_data, inner_size, axis_inner_size, inner_size_divmod);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void gather_elements(const Stream&, Span<__half>, View<__half>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod);
#endif
    template void gather_elements(const Stream&, Span<float>, View<float>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod);
    template void gather_elements(const Stream&, Span<int8_t>, View<int8_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod);
    template void gather_elements(const Stream&, Span<uint8_t>, View<uint8_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod);
    template void gather_elements(const Stream&, Span<int32_t>, View<int32_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod);
    template void gather_elements(const Stream&, Span<int64_t>, View<int64_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
