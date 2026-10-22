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

#include "../cuda4dnn/kernels/gather.hpp"

#include <cstddef>
#include <cstdint>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, class TIndex>
        __global__ void gather(
            Span<T> output, View<T> input,
            const TIndex* indices,
            std::int64_t input_block_size, std::int64_t indices_max,
            fast_divmod output_block_size, fast_divmod block_size, int* out_of_range)
        {
            for (auto id : grid_stride_range(output.size())) {
                int input_block_index, block_offset;
                output_block_size.divmod(id, input_block_index, block_offset);
                int indices_index, offset;
                block_size.divmod(block_offset, indices_index, offset);
                std::int64_t idx = static_cast<std::int64_t>(indices[indices_index]);
                idx = idx < 0 ? idx + indices_max : idx;
                if (idx < 0 || idx >= indices_max) {
                    // zero rather than stale memory; the net raises the error after the forward
                    *out_of_range = 1;
                    output[id] = T(0);
                    continue;
                }

                std::int64_t input_index = static_cast<std::int64_t>(input_block_index) * input_block_size +
                                           idx * block_size.d_ + offset;
                output[id] = input.data().get()[input_index];
            }
        }
    }

    template <class T>
    void gather(const Stream& stream,
        Span<T> output, View<T> input,
        const void* indices, std::size_t index_element_size,
        std::int64_t input_block_size, std::int64_t indices_max,
        fast_divmod output_block_size, fast_divmod block_size, int* out_of_range)
    {
        if (index_element_size == sizeof(std::int32_t)) {
            auto kernel = raw::gather<T, std::int32_t>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, static_cast<const std::int32_t*>(indices),
                          input_block_size, indices_max, output_block_size, block_size, out_of_range);
        } else if (index_element_size == sizeof(std::int64_t)) {
            auto kernel = raw::gather<T, std::int64_t>;
            auto policy = make_policy(kernel, output.size(), 0, stream);
            launch_kernel(kernel, policy, output, input, static_cast<const std::int64_t*>(indices),
                          input_block_size, indices_max, output_block_size, block_size, out_of_range);
        } else {
            CV_Error(Error::StsNotImplemented, "Gather: indices must be int32 or int64");
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void gather(const Stream&, Span<__half>, View<__half>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);
#endif
    template void gather(const Stream&, Span<float>, View<float>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);
    template void gather(const Stream&, Span<int8_t>, View<int8_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);
    template void gather(const Stream&, Span<uint8_t>, View<uint8_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);
    template void gather(const Stream&, Span<int32_t>, View<int32_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);
    template void gather(const Stream&, Span<int64_t>, View<int64_t>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);
    template void gather(const Stream&, Span<bool>, View<bool>, const void*, std::size_t, std::int64_t, std::int64_t, fast_divmod, fast_divmod, int*);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
